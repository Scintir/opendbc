"""
Stopped Vehicle Approach (SVA) - Aggressive braking strategy for approaching
stopped or nearly-stopped forward vehicles.

Intervenes AFTER the MPC planner to override acceleration targets when
physics demands more aggressive braking than the comfort-oriented MPC provides.

Two-stage approach:
  1. SOFT_APPROACH: moderate confidence, cautious braking (-1.5 to -2.3 m/s^2)
  2. HARD_APPROACH: high confidence, aggressive braking (-2.8 to -3.4 m/s^2)

Minimum stopping gap: 1.9m planner target -> ~1.5m (5ft) real-world.

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import json
import os
import time
from enum import IntEnum
from pathlib import Path

import numpy as np


# --- SVA States ---
class SVAState(IntEnum):
  INACTIVE = 0        # No stopped lead or feature disabled
  MONITORING = 1      # Stopped lead detected, building confidence
  SOFT_APPROACH = 2   # Cautious pre-brake, moderate confidence
  HARD_APPROACH = 3   # Committed aggressive braking, high confidence
  FINAL_STOP = 4      # Near-stop, holding position


# --- Detection thresholds ---
STOPPED_V_THRESHOLD = 1.0       # m/s - lead velocity below this = "stopped"
STOPPED_V_CONFIRM = 0.7         # m/s - tighter threshold for confirmed stopped
MOVING_V_THRESHOLD = 2.0        # m/s - lead velocity above this = "moving" (exit SVA)
MOVING_EXIT_TIME = 0.25         # seconds lead must be moving before SVA exits
LEAD_LOST_EXIT_TIME = 0.3       # seconds lead must be absent before SVA exits
MIN_EGO_SPEED = 3.0             # m/s - don't activate SVA below this speed
MAX_LEAD_DISTANCE = 120.0       # m - don't consider leads farther than this
MIN_LEAD_DISTANCE = 2.5         # m - below this we're already stopped or too close

# --- Confidence scoring ---
CONFIDENCE_SOFT = 0.45           # enter SOFT_APPROACH above this
CONFIDENCE_HARD = 0.75           # enter HARD_APPROACH above this
CONFIDENCE_EXIT_HARD = 0.55      # drop from HARD to SOFT below this

# --- Stopping distance ---
# dRel is measured from camera (mesh frame), NOT bumper-to-bumper.
# Real-world bumper gap ≈ dRel - DREL_TO_BUMPER_OFFSET
# Offset accounts for: ego front-bumper-to-camera (~1.8m) + lead rear-bumper-to-reflector (~1.0m)
DREL_TO_BUMPER_OFFSET = 2.8      # m - empirically calibrated from drive logs vs actual gap
MIN_STOP_GAP_BUMPER = 1.5        # m - desired bumper-to-bumper gap (5 ft)
MIN_STOP_GAP = MIN_STOP_GAP_BUMPER + DREL_TO_BUMPER_OFFSET  # 4.3m in dRel coordinates
SYSTEM_DELAY = 0.55              # s - total perception + planner + actuator delay
DELAY_SAFETY_FACTOR = 1.15       # multiply delay distance by this for margin

# --- Deceleration limits (negative = braking) ---
SOFT_DECEL_BP = [0., 10., 20., 30.]            # m/s ego speed breakpoints
SOFT_DECEL_V  = [-1.2, -1.5, -2.0, -2.3]      # m/s^2 max decel in soft mode

HARD_DECEL_BP = [0., 3., 8., 15., 22., 35.]    # m/s ego speed breakpoints
# hard mode uses the full -3.5 m/s^2 envelope at high speed (Hyundai SCC hardware limit)
HARD_DECEL_V  = [-1.2, -1.8, -2.8, -3.2, -3.5, -3.5]  # m/s^2 max decel in hard mode

# --- Progressive braking profile (margin-based) ---
# Instead of using minimum-required physics decel (which feels like coasting),
# use a progressive profile that increases decel as distance margin shrinks.
# This provides a consistent, confidence-building braking feel.
# margin = distance to stop point using comfort decel; negative = need harder braking
# np.interp requires increasing x, so BP is low→high (tight→comfortable)
MARGIN_DECEL_BP = [-5., 0., 3., 8., 15., 25., 40.]    # meters of margin (ascending)
MARGIN_DECEL_V  = [-3.5, -3.2, -2.8, -2.3, -1.8, -1.2, -0.8]  # m/s^2 (aggressive→gentle)

# Emergency fallback if physics requires more than hard profile
EMERGENCY_DECEL = -3.8           # m/s^2 - absolute max (brief)
ACCEL_MIN_HW = -3.5              # m/s^2 - hardware limit for Hyundai SCC

# --- Final stop approach ---
FINAL_STOP_SPEED = 3.0          # m/s - enter FINAL_STOP below this speed (~7mph)
FINAL_STOP_DIST = 8.0           # m - enter FINAL_STOP when lead within this (dRel coordinates)
FINAL_STOP_DECEL = -1.8         # m/s^2 - moderate final braking to stop

# --- Rate limiting (per planner cycle, ~50ms/20Hz) ---
SOFT_ACCEL_RATE = 0.15           # m/s^2 per cycle - 3.0 m/s^2/s ramp rate
HARD_ACCEL_RATE = 0.25           # m/s^2 per cycle - 5.0 m/s^2/s ramp rate
URGENT_TTC = 3.0                 # s - bypass rate limiting when TTC below this

# --- Brake-response awareness (PHEV authority adaptation) ---
# Drive logs show ~70% of commanded decel is achieved under sustained heavy
# braking. We track achieved decel via v_ego derivative and detect when the
# brake is underdelivering, so SOFT/HARD/FINAL_STOP transitions can compensate.
BRAKE_GAP_LPF_TAU = 0.4          # s - low-pass filter on measured a_ego
BRAKE_GAP_THRESHOLD = 0.6        # m/s^2 - gap that triggers compensation
BRAKE_GAP_PERSIST_S = 0.2        # s - sustained gap before flagging
BRAKE_GAP_MIN_VEGO = 3.0         # m/s - only evaluate above this speed
BRAKE_GAP_MIN_CMD = -1.0         # m/s^2 - only evaluate when meaningfully braking
HW_ACCEL_FLOOR = -3.5            # m/s^2 - matches CarControllerParams.ACCEL_MIN

# --- SOFT_APPROACH escalation thresholds (item 2) ---
# Speed-dependent SOFT->HARD escalation: at higher speeds require more severe
# physics-required decel before escalating, giving SOFT more runway. Currently
# the flat -2.2 m/s^2 threshold means high-speed approaches skip SOFT entirely.
# Threshold values are kept moderate so SOFT phase is meaningful (~0.5-0.8s on
# highway approaches) but doesn't burn safety margin — PHEV's 70% brake
# authority means we have less stopping distance than commanded suggests.
SOFT_HARD_THRESH_BP = [0., 5., 10., 15., 25.]      # m/s
SOFT_HARD_THRESH_V  = [-1.5, -1.7, -2.1, -2.4, -2.7]  # m/s^2

# Geometry guards: even at low confidence/threshold, force HARD when distance
# is short or TTC is critical. This is the primary close-range collision
# protection (open-loop, doesn't depend on the brake-gap estimator's lag).
# Distance values account for PHEV brake authority (~2.45 m/s^2 achievable),
# so they're larger than a vehicle with full -3.5 m/s^2 authority would need.
SOFT_MIN_DIST_BP = [0., 5., 10., 15., 25.]         # m/s
SOFT_MIN_DIST_V  = [10., 18., 32., 50., 90.]       # m - if lead_d below this, exit SOFT
SOFT_MAX_TTC = 2.4                                  # s - if TTC below this, exit SOFT

# --- Logging ---
LOG_DIR = "/data/logs/stopped_vehicle_approach"
LOG_LOW_RATE_HZ = 2              # log every ~0.5s when no approach event
LOG_HIGH_RATE_HZ = 20            # log every cycle during approach events
LOG_BUFFER_SIZE = 15000          # max entries per log file


class StoppedVehicleApproach:
  def __init__(self, dt: float = 0.05):
    """Initialize SVA module.

    Args:
      dt: planner cycle time in seconds (default 50ms = 20Hz)
    """
    self.dt = dt

    # Feature state
    self.enabled = False
    self.logging_enabled = False

    # SVA state machine
    self.state = SVAState.INACTIVE
    self.prev_state = SVAState.INACTIVE

    # Confidence tracking
    self.confidence = 0.0
    self.stopped_persistence = 0.0     # how long lead has been classified as stopped
    self.lead_present_time = 0.0       # how long lead has been continuously tracked
    self.lead_lost_time = 0.0          # how long since lead was last seen
    self.lead_moving_time = 0.0        # how long lead has been classified as moving

    # Physics state
    self.a_target = 0.0                # SVA acceleration target output
    self.a_required = 0.0              # physics-required deceleration
    self.a_sva_last = 0.0              # previous SVA output (for rate limiting)
    self.a_decel_floor = 0.0           # monotonic floor: decel never eases during approach
    self.d_margin = 0.0                # distance margin to stop point
    self.ttc = 999.0                   # time to collision

    # Lead state cache
    self.lead_d = 0.0
    self.lead_v = 0.0
    self.lead_y = 0.0
    self.lead_a = 0.0
    self.lead_prob = 0.0
    self.lead_status = False

    # Output flags
    self.active = False                # True when SVA is overriding MPC target
    self.force_should_stop = False     # True when SVA wants to force stopping state

    # Brake-response awareness (item 1)
    self.a_achieved_filt = 0.0         # low-pass filtered measured longitudinal accel
    self.brake_authority_gap = 0.0     # current confirmed gap (commanded vs achieved)
    self._v_ego_prev: float | None = None
    self._gap_persist = 0.0            # seconds the raw gap has been above threshold

    # Logging
    self._log_file = None
    self._log_counter = 0
    self._log_cycle_counter = 0
    self._session_start = time.monotonic()
    self._log_dir_created = False

  def update_params(self, enabled: bool, logging_enabled: bool) -> None:
    """Update feature parameters from UI toggles."""
    prev_enabled = self.enabled
    self.enabled = enabled

    # Debounce logging toggle: only close log after sustained disable
    # to avoid filesystem read glitches creating empty files every second
    if logging_enabled:
      self.logging_enabled = True
      self._logging_off_count = 0
    else:
      self._logging_off_count = getattr(self, '_logging_off_count', 0) + 1
      if self._logging_off_count >= 5:  # 5 consecutive reads = 5 seconds off
        if self.logging_enabled and self._log_file is not None:
          self._close_log()
        self.logging_enabled = False

    # Reset state when feature is toggled off
    if prev_enabled and not enabled:
      self._reset_state()

  def _reset_state(self) -> None:
    """Reset all tracking state to initial values."""
    self.state = SVAState.INACTIVE
    self.confidence = 0.0
    self.stopped_persistence = 0.0
    self.lead_present_time = 0.0
    self.lead_lost_time = 0.0
    self.lead_moving_time = 0.0
    self.a_target = 0.0
    self.a_required = 0.0
    self.a_sva_last = 0.0
    self.a_decel_floor = 0.0
    self.d_margin = 0.0
    self.ttc = 999.0
    self.active = False
    self.force_should_stop = False
    self.a_achieved_filt = 0.0
    self.brake_authority_gap = 0.0
    self._v_ego_prev = None
    self._gap_persist = 0.0

  # --- Confidence Scoring ---

  def _compute_confidence(self) -> float:
    """Compute stopped-lead confidence score (0.0 - 1.0).

    In-path alignment is a gating factor: off-path leads get heavily penalized
    to avoid false-positive aggressive braking on adjacent-lane or roadside objects.

    Inputs:
      - In-path lateral alignment (0.30 max, gates persistence)
      - Stopped persistence time (0.35 max, reduced when off-path)
      - Lead velocity stability (0.25 max)
      - Model probability (0.10 max)
    """
    # 1. In-path alignment: gating factor for confidence
    abs_y = abs(self.lead_y)
    if abs_y < 0.75:
      path_conf = 0.30
    elif abs_y < 1.25:
      path_conf = 0.15
    elif abs_y < 1.75:
      path_conf = 0.05
    else:
      path_conf = 0.0  # Off-path: severely limit total confidence

    # 2. Persistence: gated by path confidence to prevent false positives
    if path_conf == 0.0:
      # Off-path lead: minimal persistence credit regardless of duration
      persist_conf = min(0.10, self.stopped_persistence * 0.1)
    else:
      if self.stopped_persistence > 0.5:
        persist_conf = 0.35
      elif self.stopped_persistence > 0.3:
        persist_conf = 0.25
      elif self.stopped_persistence > 0.15:
        persist_conf = 0.15
      elif self.stopped_persistence > 0.05:
        persist_conf = 0.05
      else:
        persist_conf = 0.0

    # 3. Velocity stability: how stopped is the lead?
    abs_v = abs(self.lead_v)
    if abs_v < STOPPED_V_CONFIRM:
      vel_conf = 0.25
    elif abs_v < STOPPED_V_THRESHOLD:
      vel_conf = 0.15
    elif abs_v < 1.5:
      vel_conf = 0.05
    else:
      vel_conf = 0.0

    # 4. Model probability
    if self.lead_prob > 0.9:
      prob_conf = 0.10
    elif self.lead_prob > 0.5:
      prob_conf = 0.05
    else:
      prob_conf = 0.0

    return min(persist_conf + path_conf + vel_conf + prob_conf, 1.0)

  # --- Physics Calculations ---

  def _compute_required_decel(self, v_ego: float) -> float:
    """Compute physics-required deceleration to stop at MIN_STOP_GAP from lead.

    Uses kinematic equation: a = -v^2 / (2 * d_brake)
    with delay compensation: d_brake = d_available - v_ego * t_delay

    Returns:
      Required deceleration (negative, m/s^2). More negative = harder braking needed.
    """
    d_available = self.lead_d - MIN_STOP_GAP
    if d_available <= 0:
      return EMERGENCY_DECEL

    # Delay compensation: distance traveled during system delay
    d_delay = v_ego * SYSTEM_DELAY * DELAY_SAFETY_FACTOR

    # Available braking distance after delay
    d_brake = d_available - d_delay
    if d_brake <= 0.5:
      # Already need emergency braking
      return EMERGENCY_DECEL

    # Kinematic: a = -v^2 / (2*d)
    # Use closing speed (v_ego - v_lead) for more accurate calc
    v_closing = max(v_ego - max(self.lead_v, 0.0), 0.0)
    if v_closing < 0.1:
      return 0.0  # Not closing on lead

    a_required = -(v_closing ** 2) / (2.0 * d_brake)
    return max(a_required, EMERGENCY_DECEL)

  def _compute_ttc(self, v_ego: float) -> float:
    """Compute time-to-collision with lead vehicle."""
    v_closing = v_ego - max(self.lead_v, 0.0)
    if v_closing <= 0.1:
      return 999.0
    d_gap = self.lead_d - MIN_STOP_GAP
    if d_gap <= 0:
      return 0.0
    return d_gap / v_closing

  def _compute_distance_margin(self, v_ego: float) -> float:
    """Compute how much braking distance margin we have.

    Positive = comfortable margin. Negative = need aggressive braking now.
    """
    # Distance needed to stop at comfortable decel
    a_comfort = 2.5  # COMFORT_BRAKE from MPC
    d_needed_comfort = (v_ego ** 2) / (2.0 * a_comfort) + v_ego * SYSTEM_DELAY + MIN_STOP_GAP
    return self.lead_d - d_needed_comfort

  # --- State Machine ---

  def _update_state(self, v_ego: float) -> None:
    """Update SVA state machine based on current conditions."""
    self.prev_state = self.state

    # Global exit conditions
    if not self.enabled:
      self.state = SVAState.INACTIVE
      return

    if not self.lead_status:
      self.lead_lost_time += self.dt
      if self.lead_lost_time > LEAD_LOST_EXIT_TIME:
        self.state = SVAState.INACTIVE
      return
    else:
      self.lead_lost_time = 0.0

    # Track lead presence
    self.lead_present_time += self.dt

    # Track stopped persistence
    if abs(self.lead_v) < STOPPED_V_THRESHOLD:
      self.stopped_persistence += self.dt
      self.lead_moving_time = 0.0
    else:
      self.stopped_persistence = max(0.0, self.stopped_persistence - self.dt * 2.0)  # decay 2x faster
      if self.lead_v > MOVING_V_THRESHOLD:
        self.lead_moving_time += self.dt
        if self.lead_moving_time > MOVING_EXIT_TIME:
          self.state = SVAState.INACTIVE
          self._reset_tracking()
          return
      else:
        self.lead_moving_time = 0.0

    # Speed gate
    if v_ego < MIN_EGO_SPEED:
      if self.state in (SVAState.HARD_APPROACH, SVAState.SOFT_APPROACH):
        # Transition to final stop if close enough
        if self.lead_d < FINAL_STOP_DIST:
          self.state = SVAState.FINAL_STOP
        # Stay in current state otherwise
      elif self.state == SVAState.FINAL_STOP:
        # Exit FINAL_STOP when lead is departing (moving away from stopped)
        # This prevents holding force_should_stop after a stop-to-go event
        if self.lead_v > STOPPED_V_THRESHOLD:
          self.state = SVAState.INACTIVE
          self._reset_tracking()
      else:
        self.state = SVAState.INACTIVE
      return

    # Distance gate
    if self.lead_d > MAX_LEAD_DISTANCE or self.lead_d < MIN_LEAD_DISTANCE:
      if self.state not in (SVAState.HARD_APPROACH, SVAState.FINAL_STOP):
        self.state = SVAState.INACTIVE
      return

    # State transitions based on confidence and urgency
    if self.state == SVAState.INACTIVE:
      if self.stopped_persistence > 0.05 and abs(self.lead_v) < STOPPED_V_THRESHOLD:
        self.state = SVAState.MONITORING

    elif self.state == SVAState.MONITORING:
      if self.confidence >= CONFIDENCE_SOFT:
        self.state = SVAState.SOFT_APPROACH
      elif self.stopped_persistence < 0.01:
        self.state = SVAState.INACTIVE

    elif self.state == SVAState.SOFT_APPROACH:
      # SOFT->HARD escalation is gated on GEOMETRY or PHYSICS only, not on
      # confidence alone and not on brake-authority gap. The gap is a
      # constant property of the PHEV (~30% authority deficit during sustained
      # braking) so it cannot serve as a "something is wrong" signal — it
      # would fire on every event and defeat the comfort phase. Instead the
      # gap is used only to bias FINAL_STOP entry distance (HARD branch below).

      # 1) Open-loop geometry guards (item 2 + primary close-range safety):
      #    force HARD when distance is short or TTC is critical. This is the
      #    primary close-range collision protection. Does NOT depend on the
      #    brake-gap estimator's lag.
      ttc_now = self.lead_d / max(v_ego, 0.1) if v_ego > 0.1 else 999.0
      soft_min_dist = float(np.interp(v_ego, SOFT_MIN_DIST_BP, SOFT_MIN_DIST_V))
      if self.lead_d < soft_min_dist or ttc_now < SOFT_MAX_TTC:
        self.state = SVAState.HARD_APPROACH
        return

      # 2) Speed-dependent physics threshold (item 2): only escalate when
      #    physics-required decel exceeds the speed-scaled threshold. At
      #    higher speeds we require more severe a_required, giving SOFT a
      #    meaningful runway for the comfort phase.
      hard_thresh = float(np.interp(v_ego, SOFT_HARD_THRESH_BP, SOFT_HARD_THRESH_V))
      if self.a_required < hard_thresh:
        self.state = SVAState.HARD_APPROACH
        return

      # 3) Drop back to MONITORING if confidence collapses
      if self.confidence < CONFIDENCE_SOFT * 0.5:
        self.state = SVAState.MONITORING

    elif self.state == SVAState.HARD_APPROACH:
      # Item 3: extend FINAL_STOP entry distance when brake authority is
      # gapped, so we commit to the final stop earlier and don't run out
      # of stopping distance before reaching standstill.
      final_stop_dist_eff = FINAL_STOP_DIST + 2.0
      if self.brake_authority_gap > 0.0:
        final_stop_dist_eff += float(np.interp(v_ego, [0., 5., 10., 15.], [1.0, 2.0, 3.0, 4.0]))

      if v_ego < FINAL_STOP_SPEED and self.lead_d < final_stop_dist_eff:
        self.state = SVAState.FINAL_STOP
      elif self.confidence < CONFIDENCE_EXIT_HARD and self.a_required > -1.5:
        self.state = SVAState.SOFT_APPROACH

    elif self.state == SVAState.FINAL_STOP:
      # Only exit FINAL_STOP if lead moves away or ego accelerates significantly
      # (e.g., driver resumes or lead departs). Generous hysteresis prevents
      # bouncing back to HARD_APPROACH from minor creep.
      if v_ego > FINAL_STOP_SPEED + 2.0 or self.lead_d > FINAL_STOP_DIST + 5.0:
        self.state = SVAState.HARD_APPROACH

  def _reset_tracking(self) -> None:
    """Reset tracking counters without resetting the full state."""
    self.stopped_persistence = 0.0
    self.lead_present_time = 0.0
    self.lead_moving_time = 0.0
    self.confidence = 0.0

  # --- Acceleration Target Computation ---

  def _compute_sva_accel(self, v_ego: float) -> float:
    """Compute the SVA acceleration target based on current state.

    Uses a progressive margin-based profile for HARD_APPROACH that provides
    steadily increasing deceleration as distance margin shrinks. This avoids
    the "coast then brake hard" pattern where only minimum-required physics
    decel is used (which feels like coasting at mid-range distances).

    The key insight: commanding MORE decel than physics requires early on
    builds driver confidence by providing a clear "I'm stopping" signal
    throughout the entire approach.
    """

    if self.state == SVAState.INACTIVE or self.state == SVAState.MONITORING:
      return 0.0  # No override

    if self.state == SVAState.FINAL_STOP:
      if v_ego < 0.1:
        # Vehicle at standstill: neutral command, force_should_stop keeps
        # long control in STOPPING state which handles brake hold.
        return 0.0
      # Pure monotonic ease-off for the last few mph: brake force only ever
      # decreases in magnitude as v_ego → 0. The "ramp back in for confidence"
      # phase happens earlier in HARD_APPROACH, NOT here. The cabin-feel
      # window for "harsh stop" is exactly the 0.5-1.5 m/s band, so any brake
      # reversal in that range feels like the car is about to slam — opposite
      # of the confidence we want to communicate.
      # Non-uniform breakpoints: denser at low speed where perception is
      # most sensitive to small decel changes.
      return float(np.interp(v_ego,
        [0.1,   0.4,   0.8,   1.4,   2.2,   3.0],    # m/s breakpoints
        [-0.20, -0.42, -0.65, -1.00, -1.45, FINAL_STOP_DECEL]))  # m/s^2

    if self.state == SVAState.SOFT_APPROACH:
      # Speed-dependent soft deceleration limit
      a_soft_limit = float(np.interp(v_ego, SOFT_DECEL_BP, SOFT_DECEL_V))
      # Physics-required with mild aggression factor (10% harder than minimum)
      a_target = max(self.a_required * 1.1, a_soft_limit)
      return a_target

    if self.state == SVAState.HARD_APPROACH:
      # Speed-dependent hard deceleration limit (absolute cap)
      a_hard_limit = float(np.interp(v_ego, HARD_DECEL_BP, HARD_DECEL_V))

      # Speed-dependent minimum approach decel floor.
      # The MPC naturally eases braking at low speeds (-1.3 → -0.7 → -0.2)
      # which feels like "coasting then late braking." This floor stays just
      # above MPC's natural profile to maintain perceptible braking without
      # causing large overshoot. The floor is ~0.1-0.2 above what MPC commands
      # at each speed, enough to feel the difference but not enough to stop
      # dramatically early.
      min_approach_decel = float(np.interp(v_ego,
        [1., 3., 6., 10., 15., 25.],         # m/s
        [-1.5, -1.5, -1.5, -1.45, -1.4, -1.3]))  # m/s^2

      # Use physics-required OR speed-dependent floor, whichever is more aggressive.
      a_target = min(self.a_required, min_approach_decel)

      # Bound by speed-dependent hardware limit
      a_target = max(a_target, a_hard_limit)

      # Emergency overshoot if TTC is critical
      if self.a_required < a_hard_limit and self.ttc < 2.0:
        a_target = max(self.a_required, ACCEL_MIN_HW)

      return a_target

    return 0.0

  def _rate_limit_accel(self, a_target: float) -> float:
    """Apply rate limiting to the SVA acceleration target.

    When TTC is urgent (< URGENT_TTC) in HARD_APPROACH, bypass rate limiting
    entirely to allow immediate full-authority braking. Safety > comfort.
    """
    # Urgent bypass: no rate limiting when TTC is critical
    if self.state == SVAState.HARD_APPROACH and self.ttc < URGENT_TTC:
      return a_target

    if self.state == SVAState.HARD_APPROACH:
      max_change = HARD_ACCEL_RATE
    elif self.state == SVAState.SOFT_APPROACH:
      max_change = SOFT_ACCEL_RATE
    else:
      max_change = SOFT_ACCEL_RATE

    # Allow faster ramp toward more negative (more braking)
    if a_target < self.a_sva_last:
      # Braking ramp-in: use full rate
      a_limited = max(a_target, self.a_sva_last - max_change)
    elif self.state == SVAState.FINAL_STOP:
      # FINAL_STOP brake release: use full rate so the taper profile can
      # actually take effect before the vehicle reaches standstill.
      # The halved rate can't ease from -1.2 to -0.2 fast enough at <2.5 m/s.
      a_limited = min(a_target, self.a_sva_last + max_change)
    else:
      # Brake release: slower rate (half speed)
      a_limited = min(a_target, self.a_sva_last + max_change * 0.5)

    return a_limited

  # --- Main Update ---

  def update(self, lead, v_ego: float, mpc_a_target: float, mpc_should_stop: bool) -> tuple[float, bool]:
    """Main update called by longitudinal planner each cycle.

    Args:
      lead: radarState.leadOne message
      v_ego: current ego vehicle speed (m/s)
      mpc_a_target: acceleration target from MPC planner (m/s^2)
      mpc_should_stop: should_stop flag from MPC planner

    Returns:
      Tuple of (a_target, should_stop):
        - a_target: final acceleration target (may be more aggressive than MPC)
        - should_stop: final should_stop flag
    """
    self._log_cycle_counter += 1

    # --- Brake-response awareness (item 1) ---
    # Estimate achieved decel from v_ego derivative and compare against the
    # *clipped* commanded value (since hardware caps at HW_ACCEL_FLOOR, raw
    # commands deeper than that wouldn't reach the SCC anyway).
    if self._v_ego_prev is not None:
      a_meas = (v_ego - self._v_ego_prev) / self.dt
      alpha = self.dt / (BRAKE_GAP_LPF_TAU + self.dt)
      self.a_achieved_filt = (1.0 - alpha) * self.a_achieved_filt + alpha * a_meas
    self._v_ego_prev = v_ego

    a_cmd_eff = max(self.a_target, HW_ACCEL_FLOOR)
    raw_gap = 0.0
    if v_ego > BRAKE_GAP_MIN_VEGO and a_cmd_eff < BRAKE_GAP_MIN_CMD:
      raw_gap = max(0.0, abs(a_cmd_eff) - abs(self.a_achieved_filt))
    if raw_gap > BRAKE_GAP_THRESHOLD:
      self._gap_persist += self.dt
    else:
      self._gap_persist = 0.0
    self.brake_authority_gap = raw_gap if self._gap_persist > BRAKE_GAP_PERSIST_S else 0.0

    if not self.enabled:
      self.active = False
      self.force_should_stop = False
      if self.logging_enabled:
        self._maybe_log(v_ego, mpc_a_target, mpc_a_target, mpc_should_stop)
      return mpc_a_target, mpc_should_stop

    # Cache lead state
    if lead is not None and lead.status:
      self.lead_status = True
      self.lead_d = lead.dRel
      self.lead_v = lead.vLeadK  # Use Kalman-filtered velocity
      self.lead_y = lead.yRel
      self.lead_a = lead.aLeadK
      self.lead_prob = lead.modelProb
    else:
      self.lead_status = False

    # Compute physics
    if self.lead_status and v_ego > 0.5:
      self.a_required = self._compute_required_decel(v_ego)
      self.ttc = self._compute_ttc(v_ego)
      self.d_margin = self._compute_distance_margin(v_ego)
    else:
      self.a_required = 0.0
      self.ttc = 999.0
      self.d_margin = 999.0

    # Update confidence
    self.confidence = self._compute_confidence()

    # Update state machine
    self._update_state(v_ego)

    # Compute SVA acceleration target
    a_sva_raw = self._compute_sva_accel(v_ego)

    # Rate limit the SVA target
    if self.state in (SVAState.SOFT_APPROACH, SVAState.HARD_APPROACH, SVAState.FINAL_STOP):
      a_sva = self._rate_limit_accel(a_sva_raw)
      self.a_sva_last = a_sva
    else:
      a_sva = 0.0
      self.a_sva_last = 0.0

    # Determine final output
    self.active = self.state in (SVAState.SOFT_APPROACH, SVAState.HARD_APPROACH, SVAState.FINAL_STOP)

    if self.active:
      # Use the more aggressive of MPC and SVA targets
      a_target = min(mpc_a_target, a_sva)
      self.a_target = a_target

      # Force should_stop to keep long control in STOPPING state.
      # This prevents the MPC from releasing brakes and allowing creep.
      # Active whenever SVA is in FINAL_STOP, regardless of exact speed/distance.
      self.force_should_stop = (
        self.state == SVAState.FINAL_STOP and
        self.lead_status and
        self.lead_d < FINAL_STOP_DIST + 3.0  # generous margin in dRel coordinates
      )
      should_stop = mpc_should_stop or self.force_should_stop
    else:
      a_target = mpc_a_target
      self.a_target = a_target
      self.force_should_stop = False
      should_stop = mpc_should_stop

    # Logging
    if self.logging_enabled:
      self._maybe_log(v_ego, mpc_a_target, a_target, should_stop)

    return a_target, should_stop

  # --- Rate Limit Helper for Planner ---

  def get_accel_clip_rate(self) -> float:
    """Return the rate limit multiplier for the planner's accel_clip rate limiting.

    When SVA is actively braking, allow faster rate changes so the
    planner bounds don't lag behind the aggressive decel target.
    """
    if self.state == SVAState.HARD_APPROACH:
      return 3.0  # 3x faster = 0.15/cycle = 3.0 m/s^2/s
    elif self.state == SVAState.SOFT_APPROACH:
      return 2.0  # 2x faster = 0.10/cycle = 2.0 m/s^2/s
    return 1.0

  # --- Logging ---

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
    log_path = os.path.join(LOG_DIR, f"sva_{timestamp}.jsonl")
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

  def _maybe_log(self, v_ego: float, mpc_a: float, final_a: float, should_stop: bool):
    """Adaptive-rate logging: high rate during approach events, low rate otherwise."""
    if not self.logging_enabled:
      return

    # Determine log rate
    if self.active or self.state == SVAState.MONITORING:
      cycles_per_log = max(1, int(20.0 / LOG_HIGH_RATE_HZ))  # 20Hz during events
    else:
      cycles_per_log = max(1, int(20.0 / LOG_LOW_RATE_HZ))   # 2Hz idle

    if self._log_cycle_counter % cycles_per_log != 0:
      return

    # Open log file once per session (not per entry)
    if self._log_file is None:
      self._open_log()
    if self._log_file is None:
      return

    # Rotate after max entries, but reuse handle until then
    self._log_counter += 1
    if self._log_counter > LOG_BUFFER_SIZE:
      self._close_log()
      self._open_log()
      if self._log_file is None:
        return

    entry = {
      "t": round(time.monotonic() - self._session_start, 4),
      # SVA state
      "state": self.state.name,
      "active": self.active,
      "conf": round(self.confidence, 3),
      # Lead data
      "lead": self.lead_status,
      "d_rel": round(self.lead_d, 2),
      "gap_ft": round(max(self.lead_d - DREL_TO_BUMPER_OFFSET, 0) * 3.28084, 1),
      "v_lead": round(self.lead_v, 3),
      "y_rel": round(self.lead_y, 3),
      "a_lead": round(self.lead_a, 3),
      "prob": round(self.lead_prob, 3),
      # Physics
      "v_ego": round(v_ego, 2),
      "a_req": round(self.a_required, 4),
      "ttc": round(self.ttc, 2),
      "d_margin": round(self.d_margin, 2),
      # Persistence
      "stop_persist": round(self.stopped_persistence, 3),
      "lead_time": round(self.lead_present_time, 3),
      "lost_time": round(self.lead_lost_time, 3),
      "move_time": round(self.lead_moving_time, 3),
      # Acceleration
      "a_mpc": round(mpc_a, 4),
      "a_sva": round(self.a_target, 4),
      "a_out": round(final_a, 4),
      "a_sva_last": round(self.a_sva_last, 4),
      # Flags
      "should_stop": should_stop,
      "force_stop": self.force_should_stop,
    }

    try:
      self._log_file.write(json.dumps(entry) + "\n")
      self._log_file.flush()
    except Exception:
      self._close_log()

  def cleanup(self):
    self._close_log()
