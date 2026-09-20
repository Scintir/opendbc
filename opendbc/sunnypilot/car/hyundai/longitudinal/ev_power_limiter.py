"""
EV Power Limiter for Hyundai PHEV vehicles.

Limits positive longitudinal acceleration commands to keep estimated propulsion
power below a user-configurable threshold, preventing ICE engine start.

Power model uses full road-load estimation:
  P_total = (F_accel + F_rolling + F_aero + F_grade) * v_ego

The limiter computes a dynamic maximum acceleration ceiling from the power limit,
filters it for stability, and blends with a low-speed accel cap.

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import json
import math
import os
import time

import numpy as np
from pathlib import Path

# --- Vehicle constants for 2022 Hyundai Santa Fe PHEV ---
VEHICLE_MASS = 1807.0        # kg (curb weight from spec)
GRAVITY = 9.81               # m/s^2
CRR = 0.011                  # rolling resistance coefficient (SUV tires)
RHO = 1.2                    # air density kg/m^3
CDA = 0.90                   # drag coefficient * frontal area (m^2), SUV estimate
P_MARGIN_KW = 3.0            # conservative power margin (kW)
EFFICIENCY = 0.92            # drivetrain efficiency factor (battery -> wheels)

# --- Speed thresholds ---
V_MIN = 3.0                  # minimum speed for power calculation denominator (m/s)
V_BLEND_LOW = 3.0            # speed below which low-speed cap dominates (m/s)
V_BLEND_HIGH = 8.0           # speed above which power-based limit dominates (m/s)

# --- Low-speed acceleration cap (prevents engine start from high torque at low RPM) ---
# Tuned from drive logs: old values (0.5 at standstill) were far too conservative,
# causing sluggish take-off and large gap growth behind lead vehicles.
# Power at these accels is well within budget: 1.0 m/s^2 @ 3 m/s = ~6 kW (vs 29+ kW budget)
EV_LOWSPEED_ACCEL_BP = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]    # m/s
EV_LOWSPEED_ACCEL_V  = [1.3, 1.3, 1.4, 1.5, 1.6, 2.0]    # m/s^2

# --- Filter time constants (controller runs at 20Hz) ---
# Pitch: asymmetric - fast for increasing uphill (safety-conservative), slow for decreasing
PITCH_UP_ALPHA = 0.30         # fast tracking when grade load increases (uphill onset)
PITCH_DOWN_ALPHA = 0.08       # slow tracking when grade load decreases (noise suppression)
ACCEL_CEIL_DOWN_ALPHA = 0.40  # fast response when ceiling drops (per 20Hz cycle, ~0.12s)
ACCEL_CEIL_UP_ALPHA = 0.12    # slower response when ceiling rises (per 20Hz cycle, ~0.4s)

# --- Logging ---
LOG_DIR = "/data/logs/ev_power_limit"
LOG_LOW_RATE_HZ = 2           # log every ~0.5s when well below limit
LOG_POWER_THRESHOLD = 0.70    # start high-rate logging at 70% of power limit

# --- Minimum usable power after efficiency+margin (kW at wheel) ---
MIN_USABLE_POWER_KW = 5.0    # Below this, clamp to avoid zero-accel


def _usable_power_w(power_limit_kw: float) -> float:
  """Compute usable wheel power in watts, applying efficiency and margin with floor."""
  p_wheel_kw = power_limit_kw * EFFICIENCY - P_MARGIN_KW
  return max(p_wheel_kw, MIN_USABLE_POWER_KW) * 1000.0


class EVPowerLimiter:
  def __init__(self, mass: float = VEHICLE_MASS):
    self.mass = mass

    # State
    self.enabled = False
    self.power_limit_kw = 35.0
    self.logging_enabled = False

    # Filtered values
    self.pitch_filtered = 0.0
    self.accel_ceiling_filtered = 2.0  # start at max allowed
    self.power_limited = False
    self.saturation_amount = 0.0  # how much accel was clipped (for anti-windup)
    self._input_fault = False     # set when NaN/Inf detected

    # Internal debug state (for logging)
    self._a_max_power = 0.0
    self._a_max_lowspeed = 0.0
    self._a_max_raw = 0.0
    self._blend = 0.0
    self._f_road = 0.0
    self._pitch_raw = 0.0
    self._floor_active = False

    # Logging state
    self._log_file = None
    self._log_counter = 0
    self._log_cycle_counter = 0
    self._session_start = time.monotonic()
    self._log_dir_created = False
    self._pipeline_ctx: dict = {}  # populated by set_pipeline_context()

  def update_params(self, enabled: bool, power_limit_kw: int, logging_enabled: bool) -> None:
    self.enabled = enabled
    self.power_limit_kw = float(max(power_limit_kw, 20))  # enforce minimum
    self.logging_enabled = logging_enabled
    self._floor_active = (self.power_limit_kw * EFFICIENCY - P_MARGIN_KW) < MIN_USABLE_POWER_KW

    if not self.logging_enabled and self._log_file is not None:
      self._close_log()

  def _compute_road_load(self, v_ego: float, pitch: float) -> float:
    """Compute total road-load force in Newtons (excludes acceleration force)."""
    f_rolling = self.mass * GRAVITY * CRR
    f_aero = 0.5 * RHO * CDA * v_ego * v_ego
    f_grade = self.mass * GRAVITY * math.sin(pitch)
    return f_rolling + f_aero + f_grade

  def _compute_max_accel_from_power(self, v_ego: float, pitch: float) -> float:
    """Compute maximum positive acceleration allowed to stay within power limit."""
    p_usable_w = _usable_power_w(self.power_limit_kw)
    v_effective = max(v_ego, V_MIN)
    f_max = p_usable_w / v_effective
    self._f_road = self._compute_road_load(v_ego, pitch)
    f_accel = f_max - self._f_road
    return max(f_accel / self.mass, 0.0)

  def _compute_estimated_power(self, accel_cmd: float, v_ego: float, pitch: float) -> float:
    """Estimate total propulsion power in kW (at the wheel) for a given accel command."""
    f_accel = self.mass * accel_cmd
    f_road = self._compute_road_load(v_ego, pitch)
    f_total = f_accel + f_road
    p_watts = max(f_total * max(v_ego, 0.1), 0.0)
    return p_watts / 1000.0

  def set_pipeline_context(self, context: dict) -> None:
    """Accept pipeline stage data from the controller for comprehensive logging."""
    self._pipeline_ctx = context

  def update(self, accel_cmd: float, v_ego: float, pitch: float) -> tuple[float, float]:
    """Apply EV power limiting to the acceleration command.

    Args:
      accel_cmd: Desired acceleration from upstream controller (m/s^2)
      v_ego: Current vehicle speed (m/s)
      pitch: Vehicle pitch angle in radians (positive = uphill)

    Returns:
      Tuple of (limited_accel, saturation_amount)
    """
    self._pitch_raw = pitch

    # FIX #1: Fail open on bad inputs - pass through accel, don't zero it
    if not (math.isfinite(accel_cmd) and math.isfinite(v_ego) and math.isfinite(pitch)):
      self._input_fault = True
      self.power_limited = False
      self.saturation_amount = 0.0
      return accel_cmd, 0.0
    self._input_fault = False

    # FIX #5: Asymmetric pitch filter - fast for increasing uphill, slow for decreasing
    # Increasing pitch (more uphill) means more road load = need to limit more = safety-critical
    if pitch > self.pitch_filtered:
      pitch_alpha = PITCH_UP_ALPHA    # fast: track uphill onset quickly
    else:
      pitch_alpha = PITCH_DOWN_ALPHA  # slow: smooth out noise / downhill transitions
    self.pitch_filtered += pitch_alpha * (pitch - self.pitch_filtered)

    if not self.enabled or accel_cmd <= 0.0:
      self.power_limited = False
      self.saturation_amount = 0.0
      self._log_cycle_counter += 1
      if self.logging_enabled:
        self._maybe_log(accel_cmd, accel_cmd, v_ego, 0.0, 2.0, False)
      return accel_cmd, 0.0

    # --- Compute raw accel ceiling from power limit ---
    self._a_max_power = self._compute_max_accel_from_power(v_ego, self.pitch_filtered)

    # --- Low-speed accel cap ---
    self._a_max_lowspeed = float(np.interp(v_ego, EV_LOWSPEED_ACCEL_BP, EV_LOWSPEED_ACCEL_V))

    # --- Blend between low-speed cap and power-based limit ---
    self._blend = float(np.clip((v_ego - V_BLEND_LOW) / (V_BLEND_HIGH - V_BLEND_LOW), 0.0, 1.0))
    self._a_max_raw = self._blend * self._a_max_power + (1.0 - self._blend) * min(self._a_max_power, self._a_max_lowspeed)

    # --- Asymmetric filter on the ceiling ---
    if self._a_max_raw < self.accel_ceiling_filtered:
      alpha = ACCEL_CEIL_DOWN_ALPHA
    else:
      alpha = ACCEL_CEIL_UP_ALPHA
    self.accel_ceiling_filtered += alpha * (self._a_max_raw - self.accel_ceiling_filtered)

    # FIX #2: Hard cap at raw ceiling exactly - no +0.05 overshoot allowed
    effective_ceiling = min(self.accel_ceiling_filtered, self._a_max_raw)

    # --- Apply ceiling to positive accel only ---
    accel_limited = min(accel_cmd, effective_ceiling)

    # --- Compute estimated power for logging ---
    p_est = self._compute_estimated_power(accel_limited, v_ego, self.pitch_filtered)

    # --- Track saturation for anti-windup ---
    self.saturation_amount = max(accel_cmd - accel_limited, 0.0)
    self.power_limited = self.saturation_amount > 0.01

    # --- Logging ---
    self._log_cycle_counter += 1
    if self.logging_enabled:
      try:
        self._maybe_log(accel_cmd, accel_limited, v_ego, p_est, effective_ceiling, self.power_limited)
      except Exception:
        self._close_log()

    return accel_limited, self.saturation_amount

  # --- Logging implementation ---

  def _ensure_log_dir(self):
    if not self._log_dir_created:
      try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        self._log_dir_created = True
      except OSError:
        pass

  def _open_log(self):
    self._ensure_log_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"ev_power_{timestamp}.jsonl")
    try:
      self._log_file = open(log_path, "a")
    except OSError:
      # log dir unavailable (e.g. off-device); run without file logging
      self._log_file = None
    self._log_counter = 0

  def _close_log(self):
    if self._log_file is not None:
      try:
        self._log_file.close()
      except Exception:
        pass
      self._log_file = None

  def _maybe_log(self, accel_cmd: float, accel_limited: float, v_ego: float,
                  p_est_kw: float, accel_ceiling: float, limited: bool):
    """Adaptive-rate logging: high rate near limit, low rate otherwise."""
    if not self.logging_enabled:
      return

    p_usable_kw = _usable_power_w(self.power_limit_kw) / 1000.0
    power_ratio = p_est_kw / max(p_usable_kw, 1.0)
    if power_ratio >= LOG_POWER_THRESHOLD or limited:
      cycles_per_log = 1  # 20Hz
    else:
      cycles_per_log = max(1, int(20.0 / LOG_LOW_RATE_HZ))

    if self._log_cycle_counter % cycles_per_log != 0:
      return

    if self._log_file is None:
      self._open_log()
    if self._log_file is None:
      return

    self._log_counter += 1
    if self._log_counter > 12000:
      self._close_log()
      self._open_log()

    # FIX #3: Expanded logging with all internal state for full debuggability
    entry = {
      "t": round(time.monotonic() - self._session_start, 4),
      # Vehicle state
      "v": round(v_ego, 2),
      "pitch_raw": round(self._pitch_raw, 4),
      "pitch_filt": round(self.pitch_filtered, 4),
      # Power limiter core
      "a_cmd": round(accel_cmd, 4),
      "a_lim": round(accel_limited, 4),
      "a_ceil": round(accel_ceiling, 4),
      "a_ceil_filt": round(self.accel_ceiling_filtered, 4),
      # Road load decomposition
      "a_max_pwr": round(self._a_max_power, 4),
      "a_max_ls": round(self._a_max_lowspeed, 4),
      "a_max_raw": round(self._a_max_raw, 4),
      "blend": round(self._blend, 3),
      "f_road": round(self._f_road, 1),
      # Power
      "p_est": round(p_est_kw, 2),
      "p_usable": round(p_usable_kw, 1),
      "p_lim_kw": round(self.power_limit_kw, 1),
      # Status flags
      "sat": round(self.saturation_amount, 4),
      "active": limited,
      "fault": self._input_fault,
      "floor": self._floor_active,
    }

    # FIX #4: Pipeline context (from previous cycle - noted as such)
    ctx = self._pipeline_ctx
    if ctx:
      entry.update({
        "a_pid": round(ctx.get("a_pid", 0.0), 4),
        "a_hw_clip": round(ctx.get("a_hw_clip", 0.0), 4),
        "a_jerk": round(ctx.get("a_jerk_out", 0.0), 4),
        "j_up": round(ctx.get("jerk_upper", 0.0), 3),
        "j_dn": round(ctx.get("jerk_lower", 0.0), 3),
        "can_raw": round(ctx.get("can_raw", 0.0), 4),
        "can_val": round(ctx.get("can_val", 0.0), 4),
        "lcs": ctx.get("long_state", ""),
        "a_ego": round(ctx.get("a_ego", 0.0), 4),
      })

    self._log_file.write(json.dumps(entry) + "\n")
    self._log_file.flush()

  def cleanup(self):
    self._close_log()
