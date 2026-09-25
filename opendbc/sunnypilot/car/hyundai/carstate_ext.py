# ruff: noqa: E701, ISC002
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import StrEnum

import sys

from opendbc.car import Bus, structs
from opendbc.can.parser import CANParser
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.ev_limiter import get_shared_state as _ev_limiter_shared_state
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


# Module-global: only warn once per process if the expected EV-limiter
# signals (TCS13 aBasis, CLU13 DTE) are absent on a HYBRID car.
_EV_SIGNALS_MISSING_WARNED = False

# Approximate curb mass of a 2022 Santa Fe PHEV (kg). Used in the EV power
# estimator (carstate_ext._update_ev_limiter_signals).
VEHICLE_MASS_KG = 1950.0
GRAVITY_MS2 = 9.81

# Grade derivation from CAN-only signals (no openpilot service dependency).
# Body-frame longitudinal accel (ESP12.LONG_ACCEL) ≈ inertial accel + g·sin(pitch),
# while wheel-speed-derived aEgo is purely inertial in the ground frame, so:
#   g·sin(pitch) ≈ LONG_ACCEL - aEgo
# Sign verified empirically in drive #4 (corr +0.65 across 17.6k samples;
# bin analysis: ratio ≈ 0.94 of geometric expectation across pitch range).
GRADE_FILTER_TAU_S = 1.0                  # ~1 s LP for grade — slow grade vs noisy axle accel
GRADE_FILTER_DT_S = 0.01                  # carstate_ext.update() runs at 100 Hz (card.py Ratekeeper)
GRADE_FILTER_ALPHA = GRADE_FILTER_DT_S / (GRADE_FILTER_TAU_S + GRADE_FILTER_DT_S)
GRADE_ACCEL_RAW_CLIP_MS2 = 1.5            # clip raw input before filtering
# iter10 (drive #8): lowered from 1.0 to 0.5 m/s² (≈5% grade ceiling). The
# 1.0 ceiling allowed grade contribution alone to reach 64 kW at 73 mph, which
# combined with road_load and abasis produced HUD readings >80 kW on a motor
# that physically caps at ~50-60 kW EV-only. 0.5 m/s² ≈ 5% grade is sufficient
# for any sustained Loveland-area highway grade; sustained 7% climbs (≈0.69
# m/s²) will under-read by ~20 kW but should still trigger via abasis +
# road_load reaching the power_too_high threshold.
GRADE_ACCEL_FILTERED_CLIP_MS2 = 0.5

# Steady-state road load (rolling resistance + aero drag). Conservative
# defaults for a midsize SUV; can be calibrated later from logs. Drive #6
# 7:40 ICE event exposed an estimator blind spot: at 73 mph holding speed
# against grade, aBasis was -0.4 (commanded decel) but motor was doing
# real work (drag + grade hold) ~50 kW. Iter6's marginal-only formula
# read 0 kW. Iter7 baseline + grade-positive-clamp catches it.
ROLLING_RESISTANCE_COEFF = 0.011          # Crr (dimensionless)
AERO_DRAG_COEFF = 0.75                    # CdA (m²); Santa Fe is boxier than typical sedan
# iter10 (drive #8): lowered default from 1.225 (sea level) to 1.10 kg/m³,
# tuned for ~1500 m elevation (Loveland CO). At 73 mph this cuts the aero
# component by ~10% (16.5 → 14.8 kW). iter10b will switch to elevation-aware
# air density via liveLocationKalman.positionGeodetic.value[2] altitude.
AIR_DENSITY_KG_M3 = 1.10

# HUD LP filter on the published estPowerW.
# iter7-iter16: asymmetric (150 ms rise / 2 s fall). Drive 2026-09-24 forensics:
# with a jittery accel input (|ΔaBasis| p90 0.28 m/s² frame-to-frame at 70 mph
# = ±17 kW) the fast-rise/slow-fall pair rectifies noise — HUD p50 sat ~8 kW
# above the instantaneous p50 on flat highway. iter17: symmetric 0.5 s so the
# HUD shows the mean, not the envelope. The control path keeps its own filter.
POWER_TAU_RISE_S = 0.5
POWER_TAU_FALL_S = 0.5
DT_CLAMP_MIN_S = 0.001                    # safety: never let dt blow up alpha
DT_CLAMP_MAX_S = 0.1                      # 100 ms (10x nominal)

# iter14 v2 — short-tau LP for control-side power estimator (decoupled from HUD).
# Drive 18 t=1331-1352: HUD-smoothed estPowerW (2 s fall) lagged demand drops by
# many seconds, masking re-entry conditions for the state arbiter. Triple-output
# pattern: estPowerInstantW (raw, no LP) for forensic; estPowerControlW (50 ms rise
# / 300 ms fall, uncapped) for state arbiter; estPowerW (kept as iter13) for HUD.
POWER_CONTROL_TAU_RISE_S = 0.05           # 50 ms — basically tracks instant
POWER_CONTROL_TAU_FALL_S = 0.30           # 300 ms — fast enough to reflect demand drops

# iter17 — estimator rebuilt on MEASURED accel + grade (drive 2026-09-24 forensics,
# docs/ev-limiter/drive-2026-09-24-forensics.md):
#   * TCS13.aBasis already contains the grade component (at steady speed it tracks
#     kalman grade 1:1), so `m·v·aBasis + m·v·grade` double-counted hills, and the
#     iter11 saturation detector then misfired on steep grades (aBasis ≫ aEgo) and
#     zeroed the term. aBasis is also jittery. It is no longer used for power; it is
#     still published as accelDemand for the SCC-decel detector in ev_limiter.
#   * Wheel power is now m·v·max(0, aEgo_f + grade) + road_load, with aEgo LP'd at
#     AEGO_FILTER_TAU_S and grade from the calibrated kalman pitch (or the
#     LONG_ACCEL fallback, where aEgo + (LONG_ACCEL - aEgo) = LONG_ACCEL).
#   * Battery power = wheel power / η(v) + aux. The old model was wheel power only,
#     which under-reads launches (F·v is small at low v while motor/inverter losses
#     are largest) and reads ~0 when creeping with HVAC on.
# The iter9 grade deadband and the iter15 grade cap are retired: they only existed
# to fight noise and double counting in the aBasis-based formula. The raw grade
# power is still published (evLimiterGradePowerRawW); the capped-frames counter
# now stays 0.
AEGO_FILTER_TAU_S = 0.2                   # ~0.2 s LP on aEgo (wheel-speed derivative is noisy)
AEGO_FILTER_ALPHA = GRADE_FILTER_DT_S / (AEGO_FILTER_TAU_S + GRADE_FILTER_DT_S)
EFFICIENCY_LOW_SPEED = 0.72               # motor+inverter+gear η at launch (high torque, low rpm)
EFFICIENCY_HIGHWAY = 0.90                 # η once cruising
EFFICIENCY_RAMP_END_MS = 45.0 / 2.2369    # linear ramp from 0 → 45 mph
AUX_POWER_DEFAULT_W = 2500.0              # HVAC + DC-DC + accessories baseline
AUX_POWER_MIN_W = 0.0
AUX_POWER_MAX_W = 10_000.0


def drivetrain_efficiency(v_ego_ms: float) -> float:
  """Battery→wheel efficiency vs speed. Linear EFFICIENCY_LOW_SPEED →
  EFFICIENCY_HIGHWAY over 0 → EFFICIENCY_RAMP_END_MS, flat after."""
  if v_ego_ms <= 0.0:
    return EFFICIENCY_LOW_SPEED
  frac = min(1.0, v_ego_ms / EFFICIENCY_RAMP_END_MS)
  return EFFICIENCY_LOW_SPEED + (EFFICIENCY_HIGHWAY - EFFICIENCY_LOW_SPEED) * frac


def road_load_power_w(v_ego_ms: float) -> float:
  """Steady-state road load: rolling resistance (linear in v) + aero drag
  (cubic in v). With current Crr=0.011 and CdA=0.75 constants:
    33 m/s (74 mph) ≈ 24 kW
    27 m/s (60 mph) ≈ 14 kW
    22 m/s (50 mph) ≈ 10 kW
  Returns watts; never negative.
  """
  if v_ego_ms <= 0.0:
    return 0.0
  p_roll = ROLLING_RESISTANCE_COEFF * VEHICLE_MASS_KG * GRAVITY_MS2 * v_ego_ms
  p_aero = 0.5 * AIR_DENSITY_KG_M3 * AERO_DRAG_COEFF * v_ego_ms ** 3
  return p_roll + p_aero


class CarStateExt:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.aBasis = 0.0
    self.grade_accel_filtered = 0.0  # m/s^2, signed; positive = uphill
    # ESP12 silent-zero detector — drive #5 had iter5's grade fix publishing
    # gradeAccel=0 for 173k samples because lazy CANParser registration
    # failed silently. Now that ESP12 is explicitly subscribed, log once
    # if it still stays at zero for the first 5 s of operation.
    self._esp12_seen_nonzero = False
    self._esp12_zero_warning_logged = False
    self._esp12_first_call_frame = -1
    self._esp12_call_count = 0
    # Asymmetric LP filter for published estPowerW (iter7).
    self._power_filtered_w = 0.0
    self._power_filter_initialized = False
    # iter14 v2 — separate short-tau LP for control-side state arbiter input.
    # NOT capped, NOT shared with HUD path. EVLimiter reads est_power_control_w.
    self._power_control_filtered_w = 0.0
    self._power_control_filter_initialized = False
    # iter10 (drive #8): post-filter zero detector. Drive #8 had 203k
    # frames of evLimiterGradeAccel=0.0 published while estPowerW varied
    # normally — meaning either grade_f truly stuck at 0 (ESP12 silently
    # missing despite registration) or sequential-write bug. Track filter
    # output independently to disambiguate.
    self._grade_filter_seen_nonzero = False
    self._grade_filter_zero_warning_logged = False
    self._grade_filter_call_count = 0
    # iter10 Layer 3a: external grade source (liveLocationKalman pitch).
    # Set by card.py before each CI.update() call. None = use legacy
    # LONG_ACCEL-aEgo derivation. Set to a finite m/s² value when
    # liveLocationKalman is calibrated and inputsOK.
    self.grade_accel_external_ms2 = None
    self.kalman_reject_reason = 0   # iter11 Fix C: bitmask, set by card.py
    self.grade_accel_source = 0     # iter11 Fix C: 0=NONE, 1=LEGACY, 2=KALMAN

    # iter11 Fix E: power estimator EV cap + saturation substitution
    # iter13 v4: card.py is authoritative for assume_ev_only via attribute write
    # before each update tick; initialize here to fail-closed False / not-read for
    # the brief window between construction and the first update().
    self._ev_motor_cap_w = self._read_motor_cap_param()
    self._assume_ev_only = self._read_assume_ev_only_param()
    self._assume_ev_only_param_read_ok = False
    self._abasis_filtered = 0.0
    self._aego_filtered = 0.0

    # iter17: aux baseline, set by card.py each tick (EvLimiterAuxPowerW); default
    # AUX_POWER_DEFAULT_W when the attribute was never written (tests, bring-up).
    self._aux_power_w = AUX_POWER_DEFAULT_W
    # Telemetry kept from iter15: raw (uncapped) grade power each frame; the
    # capped-frames counter is retired (no cap any more) and stays 0.
    self._grade_power_capped_frames = 0
    self._grade_power_raw_w_last = 0.0

    # iter16a (Phase E1) — real HEV power ground truth (passive decode). Absent =>
    # source 0 (INVALID) + NaN power (gpt-5.5 #7: never publish 0 W for an absent
    # measurement — it would pollute estimator validation/retune). Populated by the
    # passive decoder once the motor-power CAN message identity is confirmed.
    self._real_motor_power_w = float("nan")
    self._real_power_source = 0

  # iter11 Fix E: param loaders. Use raw .get() (returns None on absent)
  # so we can default to True/sensible-default when param missing.
  def _read_motor_cap_param(self) -> float:
    try:
      from openpilot.common.params import Params
      raw = Params().get("EvLimiterMotorCapKW")
      if raw is None: return 60_000.0  # default 60 kW
      kw = max(30, min(int(raw), 100))  # bounds 30-100
      return kw * 1000.0
    except Exception:
      return 60_000.0

  def _read_assume_ev_only_param(self) -> bool:
    """iter13 v4: param read moved to card.py (selfdrive/car/card.py) so the
    openpilot-side Params() reaches reliably. Drive 17 forensics: param=1 on
    device but this returned False because Params() unreachable from opendbc
    context. card.py now sets self.assume_ev_only as an attribute each tick;
    this method is retained as a fallback only when card.py has not yet
    written the attribute (e.g. during early bring-up or unit tests).
    Default: FAIL-CLOSED False (do not silently re-enable EV cap on plumbing
    error)."""
    return bool(getattr(self, 'assume_ev_only', False))

  def _ev_mode_param_read_ok(self) -> bool:
    """iter13 v4 telemetry: True iff card.py successfully read the param.
    Published as evModeParamReadOk @24."""
    return bool(getattr(self, 'assume_ev_only_param_read_ok', False))

  def update_speed_limit(self, cp, cp_cam) -> float:
    speed_limit = 0

    if self.CP.flags & HyundaiFlags.CANFD:
      if self.CP_SP.flags & HyundaiFlagsSP.SPEED_LIMIT_AVAILABLE:
        bus = cp if self.CP.flags & HyundaiFlags.CANFD_LKA_STEER_MSG else cp_cam
        speed_limit = bus.vl["FR_CMR_02_100ms"]["ISLW_SpdCluMainDis"]
    else:
      nav, cam = 0, 0
      if self.CP_SP.flags & HyundaiFlagsSP.SPEED_LIMIT_AVAILABLE:
        nav = cp.vl["Navi_HU"]["SpeedLim_Nav_Clu"]
      if self.CP_SP.flags & HyundaiFlagsSP.HAS_LKAS12:
        cam = cp_cam.vl["LKAS12"]["CF_Lkas_TsrSpeed_Display_Clu"]

      speed_limit = cam if cam not in (0, 255) else nav

    if speed_limit in (0, 255):
      speed_limit = 0

    return speed_limit

  def update(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser], speed_conv: float) -> None:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]

    # iter13 v4 — sync param attributes from card.py (set once per tick before
    # CI.update()). Default fail-closed False if card.py hasn't written them
    # (early bring-up, tests, or after a plumbing failure).
    self._assume_ev_only = bool(getattr(self, 'assume_ev_only', False))
    self._assume_ev_only_param_read_ok = bool(getattr(self, 'assume_ev_only_param_read_ok', False))
    # iter17: aux baseline from card.py (EvLimiterAuxPowerW), bounds-clamped.
    try:
      aux = float(getattr(self, 'aux_power_w', AUX_POWER_DEFAULT_W))
    except (TypeError, ValueError):
      aux = AUX_POWER_DEFAULT_W
    self._aux_power_w = min(AUX_POWER_MAX_W, max(AUX_POWER_MIN_W, aux))

    self.aBasis = cp.vl["TCS13"]["aBasis"]

    # iter16a (Phase E1): passive real-power decode. Identity unconfirmed (gpt-5.5:
    # passive-only, confirm before any control use) — stays NaN / source 0 until the
    # motor-power / battery-VxI signal is verified against on-device raw CAN. This is
    # the single wiring point: once confirmed, fill _real_motor_power_w + source here.
    self._decode_real_power(cp)

    if self.CP_SP.flags & HyundaiFlagsSP.NON_SCC:
      cruise_msg = "LABEL11" if self.CP.flags & HyundaiFlags.EV else \
                   "E_CRUISE_CONTROL" if self.CP.flags & HyundaiFlags.HYBRID else \
                   "EMS16"
      cruise_available_sig = "CC_React" if self.CP.flags & HyundaiFlags.EV else "CRUISE_LAMP_M"
      cruise_enabled_sig = "CC_ACT" if self.CP.flags & HyundaiFlags.EV else "CRUISE_LAMP_S"
      cruise_speed_msg = "E_EMS11" if self.CP.flags & HyundaiFlags.EV else \
                         "ELECT_GEAR" if self.CP.flags & HyundaiFlags.HYBRID else \
                         "LVR12"
      cruise_speed_sig = "Cruise_Limit_Target" if self.CP.flags & HyundaiFlags.EV else \
                         "SLC_SET_SPEED" if self.CP.flags & HyundaiFlags.HYBRID else \
                         "CF_Lvr_CruiseSet"
      ret.cruiseState.available = cp.vl[cruise_msg][cruise_available_sig] != 0
      ret.cruiseState.enabled = cp.vl[cruise_msg][cruise_enabled_sig] != 0
      ret.cruiseState.speed = cp.vl[cruise_speed_msg][cruise_speed_sig] * speed_conv
      ret.cruiseState.standstill = False
      ret.cruiseState.nonAdaptive = False

      if not self.CP_SP.flags & HyundaiFlagsSP.NON_SCC_NO_FCA:
        cp_cruise = cp if self.CP_SP.flags & HyundaiFlagsSP.NON_SCC_RADAR_FCA else cp_cam

        aeb_src = "FCA11"
        aeb_warning = cp_cruise.vl[aeb_src]["CF_VSM_Warn"] != 0
        aeb_braking = cp_cruise.vl[aeb_src]["CF_VSM_DecCmdAct"] != 0 or cp_cruise.vl[aeb_src]["FCA_CmdAct"] != 0
        ret.stockFcw = aeb_warning and not aeb_braking
        ret.stockAeb = aeb_warning and aeb_braking

    ret_sp.speedLimit = self.update_speed_limit(cp, cp_cam) * speed_conv

    self._update_ev_limiter_signals(ret, ret_sp, cp)

  def _update_ev_limiter_signals(self, ret: structs.CarState, ret_sp: structs.CarStateSP, cp: CANParser) -> None:
    """Populate EV-limiter signals.

    Trigger input is an estimated propulsion power computed from TCS13.aBasis
    (aggregated longitudinal-accel demand — includes driver + stock SCC +
    control overlay) plus a grade-correction term derived from
    ESP12.LONG_ACCEL minus aEgo, times vEgo times an approximate vehicle
    mass. DTE comes from the cluster (CLU13.CF_Clu_DTE) as a "battery has
    juice" proxy.

    Gated on the HYBRID flag since the limiter itself is HYBRID-only.
    """
    if not (self.CP.flags & HyundaiFlags.HYBRID):
      return

    try:
      abasis = float(cp.vl["TCS13"]["aBasis"])
      v_ego = float(ret.vEgo)

      # iter10 Layer 3a: prefer kalman pitch source when available.
      # `grade_accel_external_ms2` is set by card.py from
      # liveLocationKalman.calibratedOrientationNED.value[1] (pitch radians)
      # converted to grade accel via g*sin(pitch). The kalman fuses IMU+camera
      # odometry and explicitly estimates accel_bias and gyro_bias, so it is
      # vastly less noisy than the LONG_ACCEL-aEgo derivation. When set
      # (kalman calibrated and inputsOK), bypass the legacy filter; the
      # kalman is already a fused estimate, no LP filter needed.
      if self.grade_accel_external_ms2 is not None:
        # Trust the kalman; clip and use directly.
        grade_f = self.grade_accel_external_ms2
        if grade_f > GRADE_ACCEL_FILTERED_CLIP_MS2:
          grade_f = GRADE_ACCEL_FILTERED_CLIP_MS2
        elif grade_f < -GRADE_ACCEL_FILTERED_CLIP_MS2:
          grade_f = -GRADE_ACCEL_FILTERED_CLIP_MS2
        # Mirror into self.grade_accel_filtered so legacy consumers and the
        # post-filter zero detector see the kalman-derived value.
        self.grade_accel_filtered = grade_f
      else:
        # Legacy fallback: LONG_ACCEL - aEgo derivation.
        # Sign verified empirically in drive #4 (corr +0.65 across 17.6k samples;
        # bin analysis: ratio ≈ 0.94 of geometric expectation across pitch range).
        # ESP12 is missing on some Hyundai PT buses (lazy `cp.vl[...]` access
        # raises AssertionError when the DBC doesn't define the message). On
        # miss we hold the last filter value rather than resetting — for a
        # transient miss after a valid history that stays conservative; on a
        # car that never has ESP12 at all the filter starts and stays at 0.
        try:
          long_accel = float(cp.vl["ESP12"]["LONG_ACCEL"])
          # Silent-zero detector: log once if ESP12 stays at exactly 0 for the
          # first ~5 s of carstate calls. iter5 had this happen unnoticed for
          # an entire 40 min drive.
          self._esp12_call_count += 1
          if long_accel != 0.0:
            self._esp12_seen_nonzero = True
          elif (
            not self._esp12_zero_warning_logged
            and not self._esp12_seen_nonzero
            and self._esp12_call_count >= 500   # 5 s at 100 Hz
          ):
            print("[ev_limiter] WARNING: ESP12.LONG_ACCEL stuck at 0.0 for 5 s — "
                  "grade-aware power is degraded to flat-only", file=sys.stderr)
            self._esp12_zero_warning_logged = True

          a_ego = float(ret.aEgo)
          grade_accel_raw = long_accel - a_ego
          if grade_accel_raw > GRADE_ACCEL_RAW_CLIP_MS2:
            grade_accel_raw = GRADE_ACCEL_RAW_CLIP_MS2
          elif grade_accel_raw < -GRADE_ACCEL_RAW_CLIP_MS2:
            grade_accel_raw = -GRADE_ACCEL_RAW_CLIP_MS2
          # First-order LP at ~1 s τ to smooth axle/IMU noise; sign retained
          # so consumers can see downhill grades for HUD/debug.
          self.grade_accel_filtered += GRADE_FILTER_ALPHA * (grade_accel_raw - self.grade_accel_filtered)
        except (KeyError, AssertionError):
          pass

        grade_f = self.grade_accel_filtered
        if grade_f > GRADE_ACCEL_FILTERED_CLIP_MS2:
          grade_f = GRADE_ACCEL_FILTERED_CLIP_MS2
        elif grade_f < -GRADE_ACCEL_FILTERED_CLIP_MS2:
          grade_f = -GRADE_ACCEL_FILTERED_CLIP_MS2

      # iter10 (drive #8): post-filter zero detector. If grade_f never moves
      # off zero across many calls while estPowerW publishes correctly, this
      # indicates broken grade detection (legacy ESP12 path) or unset external
      # source (kalman path). Logged once per process for forensics.
      self._grade_filter_call_count += 1
      if grade_f != 0.0:
        self._grade_filter_seen_nonzero = True
      elif (
        not self._grade_filter_zero_warning_logged
        and not self._grade_filter_seen_nonzero
        and self._grade_filter_call_count >= 1000   # 10 s at 100 Hz
      ):
        src = "kalman" if self.grade_accel_external_ms2 is not None else "ESP12-aEgo"
        print(f"[ev_limiter] WARNING: grade_filter ({src}) stuck at 0.0 for 10 s — "
              "grade-aware power is degraded to flat-only", file=sys.stderr)
        self._grade_filter_zero_warning_logged = True
      # iter17 power path (see module header). aEgo is LP'd (wheel-speed
      # derivative is noisy); grade is the kalman/LONG_ACCEL-derived value
      # above. Both keep their sign: a downhill cancels an equal acceleration,
      # an uphill adds to it. Only the positive (motoring) sum draws power.
      a_ego_signal = float(ret.aEgo)
      self._abasis_filtered += 0.1 * (abasis - self._abasis_filtered)   # telemetry only
      self._aego_filtered += AEGO_FILTER_ALPHA * (a_ego_signal - self._aego_filtered)
      a_net = self._aego_filtered + grade_f

      # Telemetry: raw grade power (uncapped, uphill only) — same definition as
      # iter15 minus the deadband. Not an input to the estimate on its own.
      grade_power_raw_w = VEHICLE_MASS_KG * v_ego * max(0.0, grade_f)
      self._grade_power_raw_w_last = grade_power_raw_w

      p_wheel_w = max(0.0, VEHICLE_MASS_KG * v_ego * a_net + road_load_power_w(v_ego))
      raw_power_w = p_wheel_w / drivetrain_efficiency(v_ego) + self._aux_power_w

      # iter14 v2 — triple-output power estimator.
      # 1) estPowerInstantW: truly raw, no LP, no cap. Forensic / diagnosis only.
      power_instant_w = raw_power_w

      # 2) estPowerControlW: short-tau LP, uncapped. Used by EVLimiter state arbiter.
      #    Drive 18 t=1331-1352 root cause: HUD-smoothed estPowerW (2 s fall) lagged
      #    real demand drops by many seconds, so the existing line-1450 RECOVERY→
      #    SOFT_CAP transition saw stale "below threshold" readings and stayed in
      #    RECOVERY for 21.5 s while at cap. Decoupling this LP fixes that.
      if not self._power_control_filter_initialized:
        self._power_control_filtered_w = raw_power_w
        self._power_control_filter_initialized = True
      else:
        dt = GRADE_FILTER_DT_S
        if dt < DT_CLAMP_MIN_S:
          dt = DT_CLAMP_MIN_S
        elif dt > DT_CLAMP_MAX_S:
          dt = DT_CLAMP_MAX_S
        if raw_power_w > self._power_control_filtered_w:
          alpha_ctl = dt / (POWER_CONTROL_TAU_RISE_S + dt)
        else:
          alpha_ctl = dt / (POWER_CONTROL_TAU_FALL_S + dt)
        self._power_control_filtered_w += alpha_ctl * (raw_power_w - self._power_control_filtered_w)
      power_control_w = self._power_control_filtered_w

      # Asymmetric LP for HUD smoothness + control responsiveness.
      # iter9: publish filtered only (was max(raw, filtered)) — drive #7
      # showed the max() pipeline locked in positive grade-noise transients,
      # producing ~2x phantom power on highway. The fast-rise tau (150 ms)
      # is short enough that a real power spike still drives protective
      # action quickly; the slow-fall tau (2 s) keeps HUD readable.
      if not self._power_filter_initialized:
        self._power_filtered_w = raw_power_w
        self._power_filter_initialized = True
      else:
        # dt-aware alpha; clamp dt so a timing hiccup can't blow up the filter.
        dt = GRADE_FILTER_DT_S
        if dt < DT_CLAMP_MIN_S:
          dt = DT_CLAMP_MIN_S
        elif dt > DT_CLAMP_MAX_S:
          dt = DT_CLAMP_MAX_S
        if raw_power_w > self._power_filtered_w:
          alpha = dt / (POWER_TAU_RISE_S + dt)
        else:
          alpha = dt / (POWER_TAU_FALL_S + dt)
        self._power_filtered_w += alpha * (raw_power_w - self._power_filtered_w)
      power_w_published = self._power_filtered_w

      # iter11 Fix E: ALWAYS publish raw (pre-cap) for forensics.
      raw_power_pre_cap_w = power_w_published
      ret_sp.estPowerRawW = raw_power_pre_cap_w

      # iter11 Fix E: cap at EV motor max if EvLimiterAssumeEvOnly param set.
      power_w_capped = power_w_published
      power_was_capped = False
      if self._assume_ev_only and power_w_published > self._ev_motor_cap_w:
        power_w_capped = self._ev_motor_cap_w
        power_was_capped = True
      ret_sp.estPowerCapped = bool(power_was_capped)
      ret_sp.estPowerSaturated = False   # iter17: saturation substitution retired
      ret_sp.evModeAssumed = bool(self._assume_ev_only)
      # iter13 v4 — publish param-read-success flag so device telemetry
      # distinguishes "param explicitly false" from "param read failed".
      ret_sp.evModeParamReadOk = bool(self._assume_ev_only_param_read_ok)
      ret_sp.abasisFiltered = float(self._abasis_filtered)
      ret_sp.aEgoFiltered = float(self._aego_filtered)

      ret_sp.accelDemand = abasis
      ret_sp.estPowerW = power_w_capped
      # iter14 v2 — publish triple-output power signals (NEW @41/@42/@45).
      ret_sp.estPowerInstantW = float(power_instant_w)
      ret_sp.estPowerControlW = float(power_control_w)
      ret_sp.evLimiterEstPowerRawIsFiltered = True   # clarifies @9 misleading name
      # Publish the clipped grade value — useful for HUD/debug.
      # NOTE drive #6 forensic: this field has been observed to publish 0
      # in practice while estPowerW above publishes correctly. The sequential
      # writes look identical, root cause unknown. Not blocking iter7 since
      # consumers (HUD, limiter) read estPowerW; revisit when reproducible.
      ret_sp.evLimiterGradeAccel = float(grade_f)
      ret_sp.evLimiterGradeAccelSource = int(getattr(self, 'grade_accel_source', 0))
      ret_sp.evLimiterKalmanRejectReason = int(getattr(self, 'kalman_reject_reason', 0))
      self.accel_demand = abasis
      self.est_power_w = power_w_capped
      # iter14 v2 — expose control-side power signal for EVLimiter state arbiter.
      self.est_power_control_w = power_control_w
      self.est_power_instant_w = power_instant_w
    except KeyError as e:
      global _EV_SIGNALS_MISSING_WARNED
      if not _EV_SIGNALS_MISSING_WARNED:
        print(f"[ev_limiter] TCS13 aBasis missing from parser: {e}", file=sys.stderr)
        _EV_SIGNALS_MISSING_WARNED = True

    try:
      dte_raw = int(cp.vl["CLU13"]["CF_Clu_DTE"])
      ret_sp.dteRaw = dte_raw
      self.dte_raw = dte_raw
    except KeyError:
      pass

    pub = _ev_limiter_shared_state()
    ret_sp.evLimiterActive = bool(pub["active"])
    ret_sp.evLimiterSetSpeedOffset = float(pub["set_speed_offset"])
    ret_sp.evLimiterUserTargetSpeed = float(pub.get("user_target", 0.0))
    # iter13 v4 state enum default = 8 (DISABLED) — was 7 prior to STANDSTILL_PRELAUNCH_SET
    # being inserted at @2; see opendbc.sunnypilot.car.hyundai.ev_limiter STATE_DISABLED.
    ret_sp.evLimiterState = int(pub.get("state", 8))

    # iter13 v4 telemetry: EVLimiter decision-side and standstill counters.
    # Wire-side counters (evLimiterSetEmitted/Dropped/AllBtnEmitted/LastBlockReason
    # /SuspectedSccCancelEvents/FaultInhibit*) are published by CarController
    # from the ClusterButtonRateLimiter — leave unset here (they default to 0
    # in the dataclass).
    ret_sp.evLimiterSetRequested = int(pub.get("set_requested", 0))
    ret_sp.evLimiterSetClusterDecrementAcked = int(pub.get("cluster_decrement_acked", 0))
    ret_sp.evLimiterSetNoAckEvents = int(pub.get("set_no_ack_events", 0))
    ret_sp.evLimiterStandstillEntered = int(pub.get("standstill_entered", 0))
    ret_sp.evLimiterStandstillExitedByAchieved = int(pub.get("standstill_exited_by_achieved", 0))
    ret_sp.evLimiterStandstillExitedByNoAckBackoff = int(pub.get("standstill_exited_by_no_ack_backoff", 0))
    ret_sp.evLimiterStandstillSetRequested = int(pub.get("standstill_set_requested", 0))

    # iter13 v4 wire-side fields (sourced from CarController via shared state).
    ret_sp.evLimiterSetEmitted = int(pub.get("set_emitted", 0))
    ret_sp.evLimiterSetDropped = int(pub.get("set_dropped", 0))
    ret_sp.evLimiterAllBtnEmitted = int(pub.get("all_btn_emitted", 0))
    ret_sp.evLimiterStandstillSetEmitted = int(pub.get("standstill_set_emitted", 0))
    ret_sp.evLimiterStandstillSetDropped = int(pub.get("standstill_set_dropped", 0))
    ret_sp.evLimiterSuspectedSccCancelEvents = int(pub.get("suspected_scc_cancel_events", 0))
    ret_sp.evLimiterFaultInhibitActive = bool(pub.get("fault_inhibit_active", False))
    ret_sp.evLimiterCarControllerLimiterTickRate = 100  # 100Hz tick

    # iter14 v2 — RECOVERY power-gate diagnostic counters + transition instrumentation.
    # All fields published every frame for replay forensics (gpt-5.5 R2 answer).
    ret_sp.evLimiterPowerCappedSustainFrames = int(pub.get("power_capped_control_sustain", 0))
    ret_sp.evLimiterPowerNearBudgetSustainFrames = int(pub.get("power_near_budget_sustain", 0))
    ret_sp.evLimiterStatePriorTransition = int(pub.get("state_prior_transition", 8))
    ret_sp.evLimiterStateCandidateBeforeGuard = int(pub.get("state_candidate_before_guard", 8))
    ret_sp.evLimiterStateAfterPowerGuard = int(pub.get("state", 8))   # = published state
    ret_sp.evLimiterPowerGuardYieldReason = str(pub.get("power_guard_yield_reason", "none"))
    ret_sp.evLimiterPowerGuardLockoutActive = bool(pub.get("power_guard_lockout_active", False))
    ret_sp.evLimiterRecoveryYieldEvents = int(pub.get("recovery_yield_events", 0))
    ret_sp.evLimiterRecoveryLockoutsEntered = int(pub.get("recovery_lockouts_entered", 0))

    # iter15 v2 — hard-preempt fix (Section A) + post-RES quiet (Section D) +
    # narrow standstill reset (Section C) telemetry. All sourced from shared
    # state populated by EVLimiter._publish().
    ret_sp.evLimiterGuardForcedTransition = bool(pub.get("guard_forced_transition", False))
    ret_sp.evLimiterGuardForcedTransitionEvents = int(pub.get("guard_forced_transition_events", 0))
    ret_sp.evLimiterLongStandstillResets = int(pub.get("long_standstill_resets", 0))
    ret_sp.evLimiterPostResQuietActive = bool(pub.get("post_res_quiet_active", False))
    ret_sp.evLimiterSoftcapDecrementSuppressedFrames = int(pub.get("softcap_decrement_suppressed_frames", 0))
    ret_sp.evLimiterSoftcapDecrementSuppressedEvents = int(pub.get("softcap_decrement_suppressed_events", 0))
    # @61-@62 RESERVED for iter16 HEV CAN — emit 0 placeholder so capnp serializer
    # populates the field. Do NOT add control logic here in iter15 (R2-MF-4).
    ret_sp.evLimiterReservedIter16A = 0
    ret_sp.evLimiterReservedIter16B = 0
    ret_sp.evLimiterRecoveryYieldEpisodes = int(pub.get("recovery_yield_episodes", 0))
    ret_sp.evLimiterStandstillExitStateSnapshot = str(pub.get("standstill_exit_state_snapshot", ""))[:200]
    ret_sp.evLimiterStandstillExitTimeS = float(pub.get("standstill_exit_time_s", 0.0))
    ret_sp.evLimiterStandstillExitToFirstResLatencyFrames = int(pub.get("standstill_exit_to_first_res_latency_frames", 0))
    ret_sp.evLimiterLongStandstillPrelaunchBackoffCleared = int(pub.get("long_standstill_prelaunch_backoff_cleared", 0))
    ret_sp.evLimiterLongStandstillSoftcapReasonCleared = int(pub.get("long_standstill_softcap_reason_cleared", 0))
    ret_sp.evLimiterPostResHardOverrideEvents = int(pub.get("post_res_hard_override_events", 0))

    # iter16a (Phase A) — live request-indicator signals.
    ret_sp.evLimiterRequestDir = int(pub.get("request_dir", 0))
    ret_sp.evLimiterButtonDir = int(pub.get("button_dir", 0))
    ret_sp.evLimiterRequestHonored = int(pub.get("request_honored", 0))

    # iter16a (Phase E1) — real HEV power ground truth (passive). carstate_ext owns
    # these (decoded in the power path). Absent => source 0 + NaN (gpt-5.5 #7).
    ret_sp.evLimiterRealMotorPowerW = float(self._real_motor_power_w)
    ret_sp.evLimiterRealPowerSource = int(self._real_power_source)

    # iter16a (Phase C1) — below-vEgo power droop LOG-ONLY telemetry (default-off).
    ret_sp.evLimiterPowerDroopWouldEnter = bool(pub.get("power_droop_would_enter", False))
    ret_sp.evLimiterPowerDroopRequestMph = float(pub.get("power_droop_request_mph", 0.0))
    ret_sp.evLimiterPowerDroopActive = bool(pub.get("power_droop_active", False))

    # iter15 v2 (Section B) — grade clamp telemetry. carstate_ext owns these;
    # they are not in _SHARED_STATE because they're computed in the power path
    # (above) before EVLimiter runs.
    ret_sp.evLimiterGradePowerRawW = float(self._grade_power_raw_w_last)
    ret_sp.evLimiterGradePowerCappedFrames = int(self._grade_power_capped_frames)

    # Block reason: enum field via ordinal lookup. CarController publishes
    # the string; we translate to capnp enum ordinal here.
    try:
      from opendbc.sunnypilot.car.hyundai.car_controller_button_limiter import BLOCK_REASON_ORDINAL
      reason_str = pub.get("last_block_reason", "none")
      ret_sp.evLimiterLastBlockReason = int(BLOCK_REASON_ORDINAL.get(reason_str, 0))
      fault_reason_str = pub.get("fault_inhibit_reason", "none")
      ret_sp.evLimiterFaultInhibitReason = int(BLOCK_REASON_ORDINAL.get(fault_reason_str, 0))
    except Exception:
      pass

  def _decode_real_power(self, cp) -> None:
    """iter16a (Phase E1) — passive real HEV power decode. PASSIVE, no control use.

    The 2022 Santa Fe PHEV exposes HV-battery / motor power on a CAN message whose
    identity is not yet confirmed against the on-device raw CAN (iter14 flagged a
    0x220 @100 Hz candidate). Until confirmed we publish NaN + source 0 (NEVER 0 W —
    a 0 would pollute estimator validation). Each candidate is attempted defensively;
    any miss leaves the value invalid. Wire the confirmed signal here, then iter16b
    can validate estPowerControlW/estPowerW against it before any retune.
    """
    self._real_motor_power_w = float("nan")
    self._real_power_source = 0
    # Candidate A: HV battery V x I (if the bus DBC defines these on this fingerprint).
    try:
      v = float(cp.vl["BMS_INFO"]["HV_BATTERY_VOLTAGE"])
      i = float(cp.vl["BMS_INFO"]["HV_BATTERY_CURRENT"])
      if v > 0.0:
        self._real_motor_power_w = v * i          # discharge positive
        self._real_power_source = 2
        return
    except Exception:
      pass
    # Candidate B: direct motor power signal (0x220 candidate, name unconfirmed).
    try:
      self._real_motor_power_w = float(cp.vl["MOTOR_INFO"]["MOTOR_POWER_KW"]) * 1000.0
      self._real_power_source = 1
    except Exception:
      pass

  def update_canfd_ext(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser],
                       speed_factor: float) -> None:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]

    self.aBasis = cp.vl["TCS"]["aBasis"]

    ret_sp.speedLimit = self.update_speed_limit(cp, cp_cam) * speed_factor
