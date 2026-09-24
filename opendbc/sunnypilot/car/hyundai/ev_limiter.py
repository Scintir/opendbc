# ruff: noqa: E116, ISC002, F841
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

EV Power Limiter — classic-CAN Hyundai HYBRID, stock-long only.

Prevents the ICE from kicking on during ACC by biasing the stock SCC set
speed via CLU11 button injection.

Iter6 (2026-04-28) — sliding-cap rewrite. The state-machine of iters 4-5
(SOFT_CAP entry triggered by power+aBasis, latched pre_cap_set, recovery
gated by load hysteresis, post-cap settle timers) was the wrong abstraction:
- aBasis>0.7 fallback fired on every Hyundai launch (1.5-2 m/s² stock-SCC
  launch accel), forcing 6 s of SOFT_CAP + 13 s of slow recovery on every
  red-light takeoff (drive #5: three 20 s dwells)
- power-threshold-only SOFT_CAP fired AFTER cluster gap had already built
  up SCC's accel demand into the ICE region (drive #5: ICE engaged at
  +1.99 s with cluster_set 13 mph above vEgo for the prior 15 s)

Iter6 replaces all of that with a continuous live cap. Each frame:
  margin = dynamic_margin(vEgo)
    20 mph at vEgo=0 → 5 mph at vEgo>=30, linear in between
  target_set = min(user_target, vEgo + margin)
  push DOWN (SET) when observed > target_set + deadband, OR when est_power
    is above threshold and there's a gap to close (observed > vEgo)
  push UP (RES) when observed < target_set - deadband
The dynamic margin caps SCC's accel demand by construction (small gap at
high speed → small accel command → bounded motor power). No more "cap →
release → recover" cycling.

States (published as evLimiterState uint8 — same enum as iter5; 2 and 3
re-purposed):
  0 IDLE                — at target, no press needed
  1 STANDSTILL_HOLD     — vehicle at/near stop; emit absolutely nothing
  2 SOFT_CAP_ACTIVE     — actively pushing set DOWN (HUD: LIMITING)
  3 RECOVERY_ACTIVE     — actively pushing set UP   (HUD: RECOVERING)
  4 DRIVER_OVERRIDE_SET — driver pressed wheel SET recently; we don't RES
  5 DRIVER_OVERRIDE_RES — driver pressed wheel RES recently
  6 BUS_FAULT_HOLD      — reserved
  7 DISABLED            — not supported / not enabled / CC off

State derivation is latched on recent activity (LIMITING_LATCH_FRAMES /
RECOVERING_LATCH_FRAMES) so HUD doesn't flicker between IDLE and active
on non-press frames. SOFT_CAP_ACTIVE has higher priority than driver
overrides so HUD reflects active control intent (and per drive #4: limiter
SET fires DURING DRIVER_OVERRIDE_RES — the driver-respect window does not
extend to letting motor cross ICE boundary).

What we KEEP from iter4/iter5:
  - Engagement edge fix: DISABLED handled before button processing,
    `_observed_set_speed_at_disable` frozen for engage seed,
    `engage_via_res` classification with disabled-frame lookback (drive #4
    race: RES press can land 1-2 frames before cc_enabled rises)
  - just_engaged skips both accelCruise + decelCruise on the engage frame
    (engage press isn't directional intent)
  - DRIVER_OVERRIDE_RES does NOT suppress our SET (drive #4 lesson)
  - Standstill suppression (no synthetic buttons below ~2 mph)
  - Echo filter (80 ms, first-match) so our own TX doesn't get re-counted
  - Self-direction-block (1.5 s after our SET, block our RES) — anti-osc
  - Quantization observer: 300 ms post-driver-edge window snaps user_target
    to observed if SCC's 5-mph step landed after a held button
  - Global rate limit: 6 LOGICAL presses/sec (burst copies don't count)

What we REMOVED (vs iter5):
  - SOFT_CAP entry/exit state machine, persistence frames, min-dwell
  - aBasis>0.7 fallback trigger (root cause of drive #5 dwells)
  - `_pre_cap_set_speed` latched recovery target (live target instead)
  - Recovery load gate hysteresis (`_recovery_load_paused`)
  - RECOVERY_AFTER_CAP_EXIT_FRAMES post-cap settle timer
  - AUTO_RESUME_GUARD post-engage fast-SET window (now baseline behavior)
"""
from collections import deque

from opendbc.car import structs
from opendbc.car.hyundai.values import Buttons, HyundaiFlags


try:
  from openpilot.common.params import Params as _Params
  _PARAMS_AVAILABLE = True
except Exception:  # opendbc may run outside openpilot (tests, standalone)
  _Params = None
  _PARAMS_AVAILABLE = False


ButtonType = structs.CarState.ButtonEvent.Type

TX_BUTTON_TO_EVENT_TYPE = {
  Buttons.RES_ACCEL: ButtonType.accelCruise,
  Buttons.SET_DECEL: ButtonType.decelCruise,
  Buttons.CANCEL:    ButtonType.cancel,
}

# State enum (stays in sync with evLimiterState @7 in cereal/custom.capnp).
# iter13 v4: STANDSTILL_PRELAUNCH_SET inserted at @2; SOFT_CAP/RECOVERY/etc shifted up.
STATE_IDLE                       = 0
STATE_STANDSTILL_HOLD            = 1
STATE_STANDSTILL_PRELAUNCH_SET   = 2  # NEW (iter13 v4): vEgo<0.1, cluster too high, SET allowed
STATE_SOFT_CAP_ACTIVE            = 3
STATE_RECOVERY_ACTIVE            = 4
STATE_DRIVER_OVERRIDE_SET        = 5
STATE_DRIVER_OVERRIDE_RES        = 6
STATE_BUS_FAULT_HOLD             = 7
STATE_DISABLED                   = 8

# iter11 Fix B: set of states the arbiter treats as "active" (subject to
# min-dwell on exit). All other states are quiescent or driver-priority.
ACTIVE_STATES = (STATE_SOFT_CAP_ACTIVE, STATE_RECOVERY_ACTIVE)

# Frame rate — carcontroller runs at 100 Hz.
FRAMES_PER_SEC = 100

# Burst / cadence
BURST_COPIES = 2                         # copies per commanded frame
SET_COOLDOWN_FRAMES = 15                 # 150 ms between commanded SET frames
                                          # (fast pull-down keeps up with SCC's autonomous
                                          # resume ramp ~5-8 mph/s; was 300 ms baseline
                                          # in iter5 with a separate 150 ms guard window —
                                          # consolidating to always-fast since the sliding
                                          # cap pulls down often enough that we need it.)
RES_COOLDOWN_FRAMES = 80                 # 800 ms between commanded RES frames
                                          # (1.25 mph/s gentle pull-up toward user_target;
                                          # user explicitly OK with slow recovery rate.)

# Disabled-frame RES press lookback: a wheel RES button event can land 1-2
# frames before cc_enabled rises (CAN ordering / SCC state propagation), so
# if we look only at the engage frame's buttonEvents we'll miss it. 20 frames
# = 200 ms is more than enough latitude for that race without false positives
# from older disabled-state presses.
DISABLED_RES_ENGAGE_WINDOW_FRAMES = 20
GLOBAL_RATE_LIMIT_PRESSES_PER_SEC = 6    # LOGICAL presses per rolling second; burst copies
                                          # are reliability dupes for the cluster, not separate
                                          # commands, so they don't count.
ECHO_FILTER_FRAMES = 8                   # 80 ms, first matching event only

# Driver-priority window lengths
DRIVER_OVERRIDE_SET_FRAMES = 200         # 2 s after last driver SET
DRIVER_OVERRIDE_RES_FRAMES = 300         # 3 s after last driver RES (extended per drive #3 fix)

# Standstill entry/exit
STANDSTILL_V_EGO_MS = 2 * 0.44704        # 2 mph
STANDSTILL_EXIT_V_EGO_MS = 3 * 0.44704   # 3 mph
BRAKE_LOW_SPEED_V_EGO_MS = 5 * 0.44704   # 5 mph (brake gates standstill only under this)
STANDSTILL_CONFIRM_FRAMES = 20           # 200 ms persistence on entry
RECOVERY_AFTER_STANDSTILL_FRAMES = 30    # 300 ms clean after standstill before RES allowed

# Sliding cap (iter6 core mechanism). cluster_set is held within
# `vEgo + dynamic_margin(vEgo, est_power)`. Margin shape:
#   - Wider at low speed so a stored set of e.g. 60 mph at standstill
#     doesn't drive aggressive launch accel.
#   - Tight at highway speed so SCC's accel demand stays bounded.
#   - PLUS a low-load bonus (iter7): when est_power_w is well below the
#     user's threshold, allow extra margin so SCC has room to accelerate
#     toward user_target on flats. Drive #6 confirmed iter6's flat 5 mph
#     cap throttled natural recovery to a crawl — vehicle and cluster_set
#     stuck together at vEgo+5 because SCC's accel demand for a 5 mph gap
#     under low load is essentially zero.
LOW_SPEED_MARGIN_MPH = 20.0              # base margin at vEgo = 0
HIGH_SPEED_MARGIN_MPH = 5.0              # base margin at vEgo >= MARGIN_BLEND_END_MPH
MARGIN_BLEND_END_MPH = 30.0              # vEgo above this uses HIGH_SPEED_MARGIN_MPH; below
                                          # this, base margin interpolates linearly
LOW_LOAD_BONUS_MPH = 10.0                # extra margin when load is well below threshold
LOAD_BONUS_LOW_FRAC = 0.4                # below 40% of threshold = full bonus
LOAD_BONUS_HIGH_FRAC = 0.8               # above 80% of threshold = no bonus
                                          # (linear taper between)

# Standstill SET. Iter7 added SET-only-at-standstill (pull set toward 20 mph
# cap when stopped); iter8 lowered the pulse cap from 30 → 10 because the
# drive #6 simulation suggested Hyundai SCC may ignore subsequent SETs at
# vEgo=0 (real iter6 fired 1 SET → cluster dropped 1 mph → no further
# response observed). Decel-fast cadence (below) does the actual cluster
# pull-down work BEFORE standstill latches, so 10 pulses is plenty as the
# residual safety net.
STANDSTILL_SET_PULSE_CAP = 10            # max synthetic SETs per single standstill window

# Decel-fast SET regime (iter8). When vEgo is in the low-speed range AND
# decelerating, SCC pulls cluster set down via our SET cascade. The default
# 150 ms SET cadence (= 6.7 mph/s pull-down rate) loses the race against
# typical brake-decel of 8-15 mph/s, leaving cluster set high when
# raw_standstill latches. During decel-fast we use a tighter cadence and a
# higher rate-limit ceiling so cluster keeps up with vEgo. Outside this
# regime, default cadence + rate limit apply.
DECEL_FAST_VEGO_THRESHOLD_MS = 30.0 * 0.44704     # below 30 mph
DECEL_FAST_AEGO_MS2 = -0.5                          # noticeable decel (negative aEgo)
SET_COOLDOWN_DECEL_FAST_FRAMES = 6                  # 60 ms minimum spacing between SETs
DECEL_FAST_RATE_LIMIT_PRESSES_PER_SEC = 12          # sustained 12 SETs/sec (=12 mph/s pull-down
                                                     # rate, vs default 6 mph/s); minimum
                                                     # spacing of 60 ms is below this and the
                                                     # rate limit is the binding constraint.
                                                     # Active only during decel-fast — RES
                                                     # isn't fired during decel so no conflict.

# Down-trigger (SET) deadbands
SET_TRIGGER_DEADBAND_MS = 0.5 * 0.44704  # 0.5 mph above target_set before we push down
POWER_GAP_DEADBAND_MS = 0.5 * 0.44704    # observed must be > vEgo + 0.5 mph to attribute high
                                          # power to SCC's accel demand (vs grade/drag/HVAC)

# Up-trigger (RES) constants
RECOVERY_DEADBAND_MS = 1.0 * 0.44704     # 1 mph below target_set before we push up
RECOVERY_V_EGO_FLOOR_MS = 3 * 0.44704    # must be moving (matches former soft-cap floor)

# State-latch durations — HUD/log state is "limiting" or "recovering" if
# we emitted a press recently OR want to emit one this frame. Avoids 100 Hz
# flicker when the underlying button cadence is slower than per-frame.
LIMITING_LATCH_FRAMES = 50               # 500 ms after our last SET counts as LIMITING
RECOVERING_LATCH_FRAMES = 100            # 1 s after our last RES counts as RECOVERING

# iter10 (drive #8) Layer 2: state machine dwell + hysteresis. Drive #8
# 7:36-7:45 oscillation period had 96 state transitions in 9 minutes
# (10.7/min). Without dwell, the soft-cap ↔ recovery loop fires whenever
# estPowerW crosses threshold, which happens ~once/s on grade with the
# noisy LP-filtered observer.
#
# Semantics:
#   IDLE → active state: governed by ENTER_SUSTAIN_FRAMES only (no min-dwell).
#     EXCEPTION: power_too_high bypasses entry sustain → immediate SOFT_CAP
#     entry. Preserves iter9 fast-protection.
#   active → IDLE: requires (time-in-state ≥ MIN_ACTIVE_STATE_DWELL_FRAMES)
#     AND (clear sustained for EXIT_SUSTAIN_FRAMES).
#   active ↔ active (cap ↔ recovery): allowed only after MIN_ACTIVE_STATE_DWELL.
#   any → STANDSTILL_HOLD / DRIVER_OVERRIDE_* / DISABLED: immediate (driver
#     priority states bypass all timers).
#   Sustain counter reset rule: increments while predicate true, resets to 0
#     when predicate false.
#   Cross-predicate priority: if set_too_high AND under_target both true
#     simultaneously after governor clamp, SOFT_CAP wins (down has priority).
MIN_ACTIVE_STATE_DWELL_FRAMES = 200      # 2.0 s minimum in any active state before exit
SOFT_CAP_ENTER_SUSTAIN_FRAMES = 30       # 0.3 s of (set_too_high or power_too_high)
SOFT_CAP_EXIT_SUSTAIN_FRAMES = 200       # 2.0 s of clear before allowing exit
RECOVERY_ENTER_SUSTAIN_FRAMES = 20       # 0.2 s of under_target
RECOVERY_EXIT_SUSTAIN_FRAMES = 100       # 1.0 s of at-target before allowing exit

# iter16a (B2, gpt-5.5 review): suppress counterproductive RES while the set-vs-actual
# delta is already large. Drives a8..b3 showed RECOVERY pushing set UP for 1078 frames
# mid-windup (set 18-40 mph above actual) — direct lead-clear-surge fuel. Hysteretic to
# avoid RES chatter at the boundary. Blocks UP only; never blocks SET-down protection.
RECOVERY_MAX_DELTA_ENTER_MPH = 12.0      # block RES when (cluster_set - vEgo) exceeds this
RECOVERY_MAX_DELTA_EXIT_MPH = 8.0        # re-allow RES once delta falls back below this

# iter14 v2 (drive 18 t=1331-1352): RECOVERY-while-capped state-arbiter guard.
# Independent of cur_state, runs after _derive_state() and before _publish() so
# it gates actuation, not just telemetry (R1-MF1). Reads est_power_control_w
# (short-tau LP, uncapped) — NEVER the HUD-smoothed est_power_w (R1-MF3).
RECOVERY_POWER_NEAR_BUDGET_FRAC = 0.95         # tier-2 trigger (LOWER frac = more aggressive)
POWER_NEAR_BUDGET_DEBOUNCE_FRAMES = 3          # 30 ms — filter single-frame IMU/aBasis spikes
RECOVERY_AFTER_SOFTCAP_LOCKOUT_FRAMES = 200    # 2 s minimum no-RECOVERY after guard fires
RECOVERY_REENTRY_HEADROOM_FRAC = 0.85          # power must drop to 0.85*cap for re-entry sustain
RECOVERY_REENTRY_SUSTAIN_FRAMES = 100          # 1 s sustained at headroom
# RECOVERY re-entry requires BOTH lockout_time_done AND headroom_done (R2-MF-B).

# iter16a (C1) — below-vEgo power-droop. LOG-ONLY / default-OFF this iteration
# (gpt-5.5: new authority on an unvalidated estimate must not ship active until a
# real-power-instrumented drive validates it). When the est power stays over the
# enter threshold for the sustain window, we SIMULATE a droop target (vEgo - droop),
# ramping the droop up to a cap, and publish what we WOULD do. Control only changes
# if the EvLimiterPowerDroopEnable param is explicitly on (default off).
POWER_DROOP_ENTER_KW = 45.0              # enter sim when est power exceeds this (hysteresis)
POWER_DROOP_EXIT_KW = 38.0              # exit sim when est power falls below this
POWER_DROOP_SUSTAIN_FRAMES = 200        # 2 s sustained over-cap before droop engages
POWER_DROOP_MAX_MPH = 4.0              # cap on how far below vEgo the droop will request
POWER_DROOP_RAMP_MPH_PER_FRAME = 1.0 / 100.0   # +1 mph per second at 100 Hz
POWER_DROOP_ENABLE_PARAM = "EvLimiterPowerDroopEnable"  # default off (fail-safe)

# iter15 v2 (Section C) — narrow standstill reset + state-vector telemetry.
# Drive-A/B forensics: takeoff after long red lights felt slow. Hypothesis is
# stale PRELAUNCH no-ack backoff and stale `_softcap_entry_reason` from a
# prior softcap episode that the long standstill made irrelevant. R1-MF-C
# narrowed the reset dramatically: PRESERVE sustain counters + lockout frame
# counter; CLEAR only PRELAUNCH backoff and stale softcap reason (conditional).
LONG_STANDSTILL_RESET_FRAMES = 500             # 5 s at 100 Hz — threshold for "long" stop
STALE_SOFTCAP_REASON_POWER_FRAC = 0.85         # softcap reason cleared only if power<0.85*cap

# iter15 v2 (Section D) — post-RES quiet period for SOFT_CAP decrement.
# Drive A/B: after RES emit, accel command spikes briefly → est_power read
# inflates → softcap-driven SET fires → net cruise speed loss. Suppress for 2 s
# UNLESS power genuinely far over cap. Edge-detect events (R1-MF-D).
POST_RES_QUIET_PERIOD_FRAMES = 200             # 2 s at 100 Hz
POST_RES_HARD_OVERRIDE_FRAC = 1.05             # override quiet if est_power_control_w > 1.05*cap

# iter10 (drive #8) Layer 1: bounded reference governor.
# Replaces iter9's blanket `gas_pressed` block on want_res with mode-based
# bound computation. Prevents Event A (dwell at 21 mph) and Event B
# (cluster < vEgo oscillation) by construction.
#
# Modes (selected exclusively each tick):
GOVERNOR_MODE_NORMAL       = 0   # default: max-deficit + no-below-vEgo invariants
GOVERNOR_MODE_GAS_CATCHUP  = 1   # driver pressing gas — cluster tracks vEgo upward
GOVERNOR_MODE_DECEL        = 2   # SCC commanding decel (lead, brake-recent) — let cluster dip
GOVERNOR_MODE_BRAKE        = 3   # brake pressed — limiter should not act
GOVERNOR_MODE_STANDSTILL   = 4   # at/near standstill — STANDSTILL_HOLD owns it

MAX_DEFICIT_DEFAULT_MPH = 7.0    # default cap on (user_target - cluster_set), UI-tunable
GAS_HEADROOM_MPH = 2.0           # while gas pressed, cluster ≤ vEgo + 2
GAS_CATCHUP_MIN_DEFICIT_MPH = 3.0  # arming: cluster < user_target - 3 mph
GAS_HOLD_MIN_FRAMES = 50         # 0.5 s @ 100 Hz before catch-up arms
SCC_DECEL_DETECT_MS2 = -0.5      # accelDemand below this = SCC commanding decel
SCC_DECEL_PERSIST_FRAMES = 50    # 0.5 s sustain to qualify as decel intent
BRAKE_RECENT_FRAMES = 100        # 1 s after brake release still counts as decel intent
GAS_RES_INTERVAL_FRAMES = 150    # 1.5 s between RES presses while gas held

# UI-tunable param key for max deficit
MAX_DEFICIT_PARAM = "EvLimiterMaxDeficitMph"

# One-direction-at-a-time cooldowns to prevent visible oscillation
LIMITER_OPPOSITE_DIR_BLOCK_FRAMES = 150  # after our SET, block our RES for 1.5 s, and vice versa

# Driver-adjust observation window: after a driver button edge, watch a short
# follow-up window to catch the SCC's eventual +5 mph quantization step on
# held buttons. During this window we update HUD-display user_target and the
# recovery-target latch from observed; we do NOT use it for any cap/control.
DRIVER_ADJUST_WINDOW_FRAMES = 30         # 300 ms — long enough to see the cluster respond
TX_ECHO_ATTRIBUTION_FRAMES = 20          # 200 ms — observed changes within this window of
                                          # our last TX are attributed to us, not the driver

# user_target clamp range
MPH_TO_MS = 0.44704
USER_TARGET_MIN_MS = 0.0
USER_TARGET_MAX_MS = 95.0 * MPH_TO_MS

# iter10: governor constants (require MPH_TO_MS, defined here)
EGO_SLOP_MS = 0.5                # 1.1 mph slop on "cluster ≥ vEgo" invariant
STANDSTILL_V_THRESHOLD_MS = 2 * MPH_TO_MS    # below this = STANDSTILL mode


# --- iter13 v4 constants ---------------------------------------------------

# EVLIMITER_ALLOWED_BUTTONS — R4-MF4 allowlist for `_desired_button`. Only
# None / SET / RES may be requested by EVLimiter. CANCEL/GAP_DIST are NEVER
# valid limiter outputs (CarController enforces this too via
# CARCONTROLLER_VALID_BUTTONS, but EVLimiter must never produce them).
EVLIMITER_ALLOWED_BUTTONS = (None, Buttons.SET_DECEL, Buttons.RES_ACCEL)


class EVLimiterError(Exception):
  """Raised when EVLimiter would emit an unsafe/invalid button. Use raise (not
  assert) — production may run with `python -O` which strips assertions, and
  R4-MF4 mandates a runtime guard."""


# ACK matcher (R4-MF3 / R4-MF5). Post-emission, 1-to-1, 500ms inclusive.
ACK_WINDOW_FRAMES = 50  # 500ms at 100Hz

# Pre-launch SET (iter13 v4 Section E). Reset on each PRELAUNCH_SET entry.
LAUNCH_DEADBAND_MPH = 2.0
STANDSTILL_TARGET_FLOOR_MPH = 21.0
ALLOWABLE_GAP_MPH = 5.0
STANDSTILL_PULSE_CAP_INITIAL = 10
STANDSTILL_PULSE_CAP_AFTER_ACK = 30
# iter16a (B1): a long red light used to permanently exhaust the per-stop pulse
# cap (~10 pulses) and then sit in STANDSTILL_HOLD for the rest of the stop with
# set speed pinned ≫ launch_target (driver-observed red-light windup). Allow the
# per-stop budget to refresh each time the no-ack backoff expires, so the limiter
# keeps making BOUNDED low-duty-cycle set-down attempts (≈5 pulses / 3 s) across a
# long stop. Still bounded: launch floor, driver defers, and a per-stop retry cap.
MAX_PRELAUNCH_BACKOFF_RETRIES = 8        # ≈ up to ~50 bounded pulses across a multi-min stop
STANDSTILL_NO_ACK_BACKOFF_AFTER_EMITTED = 5
STANDSTILL_NO_ACK_BACKOFF_S = 3.0
STANDSTILL_NO_ACK_BACKOFF_FRAMES = int(STANDSTILL_NO_ACK_BACKOFF_S * FRAMES_PER_SEC)

# iter13 v4 ack-driven SET cadence (replaces iter12 continuous-frame escape).
SET_HARD_MIN_INTERVAL_FRAMES = 50          # 0.5s logical min between desired SETs
SET_NORMAL_COOLDOWN_FRAMES_V13 = 150        # 1.5s gentle steady-state
SET_RESPONSE_TIMEOUT_FRAMES_V13 = 200       # 2.0s overall ack deadline
SET_NO_ACK_BACKOFF_TIER_FRAMES = (150, 200, 300)  # 1.5/2.0/3.0s adaptive

# Module-level singleton so the CarState-side publisher (carstate_ext) can
# read state without passing references through CarController plumbing.
_SHARED_STATE: dict = {
  "active": False,
  "set_speed_offset": 0.0,   # m/s, max(0, user_target - observed)
  "user_target": 0.0,        # m/s
  "state": STATE_DISABLED,
}


def get_shared_state() -> dict:
  return _SHARED_STATE


class EVLimiter:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.supported = (
      not bool(CP.flags & HyundaiFlags.CANFD)
      and bool(CP.flags & HyundaiFlags.HYBRID)
      and not CP.openpilotLongitudinalControl
    )

    self._params = _Params() if _PARAMS_AVAILABLE else None

    # user_target lifecycle — seeded on engage rising edge, never auto-zeroed
    self.user_target_speed = 0.0
    self.was_cc_enabled = False

    # TX bookkeeping
    self.last_set_frame = -10000
    self.last_res_frame = -10000
    self.press_history = deque()  # of (frame, copies) — for global rate limit

    # Echo filter (single-slot, first-match consumption)
    self._pending_echo_button = Buttons.NONE
    self._pending_echo_frame = -10000

    # Driver override windows — track LAST driver press of each direction
    self._driver_set_last_frame = -10000
    self._driver_res_last_frame = -10000

    # STANDSTILL persistence
    self._standstill_trigger_frames = 0
    self._standstill_on = False
    self._left_standstill_at_frame = -10000
    # iter7: bounded SET-pulse counter at standstill (prevents runaway TX
    # if Hyundai SCC ignores SET commands at vEgo=0).
    self._standstill_set_pulses = 0
    self._was_in_standstill_last_frame = False

    # Driver-adjust observation window — see DRIVER_ADJUST_WINDOW_FRAMES doc
    self._driver_adjust_until = -10000

    # Pre-engage observed setpoint — frozen during DISABLED, used to seed
    # user_target on RES re-engage (drive #4: SCC autonomously snaps cluster
    # up on RES; reading post-engage observed gives a polluted seed).
    self._observed_set_speed_at_disable = 0.0
    # Last frame a physical RES press was seen while DISABLED. The engage
    # frame may not contain the press itself if cc_enabled rises 1-2 frames
    # later, so engage classification looks back this far.
    self._last_disabled_res_press_frame = -10000

    # Published burst count (carcontroller reads this each frame)
    self.current_burst_count = BURST_COPIES

    # Debug state (for logging)
    self.state = STATE_DISABLED

    # iter10 Layer 2: state machine dwell + hysteresis (drive #8 Event B fix).
    # Tracks frame of last state transition + sustain counters for predicate
    # debouncing. Resets to 0 when entering states / when predicate clears.
    self._state_entered_frame = 0
    self._softcap_enter_sustain = 0   # increments while (set_too_high or power_too_high)
    self._softcap_exit_sustain = 0    # increments while NOT (set_too_high or power_too_high)
    self._recovery_enter_sustain = 0  # increments while under_target
    self._recovery_exit_sustain = 0   # increments while NOT under_target

    # iter10 Layer 1: bounded reference governor state.
    # Track gas-hold duration (catch-up arming), brake recency, SCC sustained
    # decel (lead-following / commanded slowdown intent), and last cluster set
    # we committed to (gas-catchup lower-bound anchor).
    self._gas_hold_frames = 0
    self._last_brake_frame = -10000
    self._scc_decel_persistent_frames = 0
    self._last_committed_observed_set = 0.0
    # Diagnostic: count bound inversions detected in NORMAL/DECEL modes.
    # MODE_GAS_CATCHUP is non-inverting by construction.
    self._bound_inversion_count = 0

    # iter11 Fix B: arbiter forensic counters
    self._transitions_blocked_by_dwell = 0
    self._transitions_blocked_by_sustain = 0
    self._power_too_high_recent = False   # set true on power_too_high entry, cleared on exit
    self._transition_log = deque(maxlen=10)

    # iter14 v2 — RECOVERY power-gate state (drive 18 t=1331-1352 fix).
    self._power_near_budget_sustain = 0          # debounce counter for tier-2 (>= 0.95*cap)
    self._power_capped_control_sustain = 0       # diagnostic counter (sustained-capped trigger removed per R2-MF-A)
    self._softcap_from_recovery_lockout_until = -1000000  # frame after which lockout time has elapsed
    self._recovery_reentry_sustain = 0           # frames at <= 0.85*cap (headroom)
    self._recovery_lockout_engaged = False        # True after a yield until lockout cleared (cold-start: False)
    self._evLimiter_recovery_yield_events = 0    # cumulative yields by power guard
    self._evLimiter_recovery_lockouts_held = 0   # cumulative frames RECOVERY blocked by lockout
    self._softcap_entry_reason = "none"          # "recovery_power_immediate" | "recovery_power_near_budget_debounced" | "none"
    self._power_guard_yield_reason_last = "none" # for instrumentation publish: "none" | "immediate" | "debounced"
    self._power_guard_lockout_active_last = False
    self._state_prior_transition_last = STATE_DISABLED   # state at start of update() (instrumentation)
    self._state_candidate_before_guard_last = STATE_DISABLED  # what state arbiter wanted

    # iter11 Fix A: recovery escape (rate-limited recovery below max-deficit floor)
    self._recovery_escape_active = False
    self._recovery_escape_start_t = 0.0
    self._recovery_escape_start_cluster = 0.0
    self._engaged_at_frame = -10000   # for engagement-transient safety override

    # iter11 Fix D: ineffective-RES watchdog
    self._res_sequence_start_frame = -10000
    self._res_sequence_press_count = 0
    self._res_sequence_start_cluster_ms = 0.0
    self._res_escape_until_frame = -10000
    self._res_escape_start_cluster_ms = 0.0
    self._last_escape_attempt_frame = -10000
    self._ineffective_res_events = 0

    # iter12 Fix F (rev): ack-driven SET cadence (iter13 v4 REMOVES the
    # continuous-frame escape fields below; only _cluster_at_last_set is
    # still consulted by the iter12 SET path during transition. Field kept
    # in __init__ for backwards-compat with any test that still references
    # it. The `_set_escape_until_frame` continuous-frame escape was the
    # cause of drive 17 SCC auto-cancels; iter13 removes it.)
    self._cluster_at_last_set = 0.0          # cluster reading at last SET fire (iter13: still used)
    self._ineffective_set_press_count = 0    # legacy iter12 field; iter13 uses _set_no_ack_events

    # iter11 Fix F: highway SET cooldown counter
    self._power_high_pending_frames = 0

    # iter11 Fix B: standstill 10-frame hysteresis
    self._standstill_entry_frames = 0
    self._standstill_exit_frames = 0

    # iter16a (B2): hysteretic RES-block-while-delta-large latch.
    self._recovery_delta_block = False
    # iter16a (Phase A): live request-indicator signals (per-frame).
    self._request_dir = 0       # 0 NONE, 1 UP(want_res), 2 DOWN(want_set)
    self._button_dir = 0        # 0 NONE, 1 UP(RES), 2 DOWN(SET) — actual emitted
    self._request_honored = 0   # 0 unknown, 1 honored, 2 ignored
    self._last_emit_dir = 0     # dir of last emitted button (for honored matching)
    self._last_emit_frame_for_honored = -10000
    self._cluster_at_emit_ms = 0.0

    # --- iter13 v4 fields ----------------------------------------------------

    # Advisory desired button + reason — CarController reads these instead of
    # using the (button, active) return value once Phase 4 wires the new path.
    self._desired_button: int | None = None
    self._desired_block_reason_advisory = "none"

    # ACK matcher state (R4-MF3 / R4-MF5). 500ms one-to-one, post-emission only.
    # CarController calls note_set_emitted(frame_idx) AFTER rate-limiter
    # acceptance. try_ack_set(...) consumes oldest unmatched on cluster
    # decrement (excluding pre-emission and physical-button-induced).
    self._unmatched_emitted_set_frames: deque = deque()
    self._cluster_decrement_acked = 0           # CarStateSP @28
    self._set_no_ack_events = 0                 # CarStateSP @29 (≥3 emit-no-ack)
    self._consecutive_no_ack_emits = 0          # transient counter feeding _set_no_ack_events
    self._cluster_prev_for_ack_ms = 0.0         # last cluster reading observed by try_ack_set
    self._set_no_ack_backoff_until_frame = -10000  # advisory backoff window

    # Standstill PRELAUNCH_SET state tracking (Section E).
    self._prelaunch_set_pulses_emitted = 0
    self._prelaunch_set_pulses_emitted_since_ack = 0
    self._prelaunch_set_pulse_cap = STANDSTILL_PULSE_CAP_INITIAL
    self._prelaunch_first_ack_seen = False
    self._prelaunch_no_ack_backoff_until = -10000
    self._was_in_prelaunch = False
    # iter16a (B1): per-stop bounded backoff-retry tracking.
    self._prelaunch_backoff_retries = 0
    self._prelaunch_backoff_was_active = False
    # iter16a (C1): below-vEgo power-droop sim (log-only / default-off).
    self._power_droop_sustain = 0
    self._power_droop_would_enter = False
    self._power_droop_request_mph = 0.0
    self._power_droop_active = False

    # Counters published to CarStateSP. CarController owns emitted/dropped;
    # EVLimiter owns requested (decision-side).
    self._set_requested = 0                     # CarStateSP @25 (decision-side)
    self._standstill_set_requested = 0          # CarStateSP @33 (decision-side)
    self._standstill_entered = 0                # CarStateSP @30
    self._standstill_exited_by_achieved = 0     # CarStateSP @31
    self._standstill_exited_by_no_ack_backoff = 0  # CarStateSP @32

    # iter15 v2 (Section A) — hard-preempt guard transition flag + counters.
    # `_guard_forced_transition_last_frame` is the per-frame published bool.
    # `_evLimiter_guard_forced_transition_events` counts frames where guard
    # forced transition (cumulative). `_evLimiter_recovery_yield_episodes` is
    # edge-detected RECOVERY→SOFT_CAP episodes (R2-MF-1 strict).
    self._guard_forced_transition_last_frame = False
    self._evLimiter_guard_forced_transition_events = 0
    self._evLimiter_recovery_yield_episodes = 0

    # iter15 v2 (Section C) — narrow standstill reset state + state-vector
    # telemetry. `_time_in_standstill_frames` accumulates while in_standstill,
    # resets on exit. The exit snapshot fields are LATCHED on the transition
    # from standstill→non-standstill so the next drive can review post-mortem.
    self._time_in_standstill_frames = 0
    self._evLimiter_long_standstill_resets = 0
    self._evLimiter_long_standstill_prelaunch_backoff_cleared = 0
    self._evLimiter_long_standstill_softcap_reason_cleared = 0
    self._standstill_exit_state_snapshot_last = ""
    self._standstill_exit_time_s_last = 0.0
    self._standstill_exit_to_first_res_latency_frames = 0
    self._standstill_exit_post_exit_active = False
    self._standstill_exit_first_res_seen = False

    # iter15 v2 (Section D) — post-RES quiet period for SOFT_CAP decrement.
    # `_last_res_emit_frame` updated when RES_ACCEL emitted. Edge-detect via
    # `_softcap_decrement_suppressed_last_frame` per R1-MF-D.
    self._last_res_emit_frame = -1_000_000
    self._softcap_decrement_suppressed_last_frame = False
    self._evLimiter_softcap_decrement_suppressed_frames = 0
    self._evLimiter_softcap_decrement_suppressed_events = 0
    self._evLimiter_post_res_hard_override_events = 0
    self._post_res_quiet_active_last = False

  # ----- Params helpers ---------------------------------------------------

  def _read_bool(self, key: str, default: bool) -> bool:
    if self._params is None:
      return default
    try:
      return bool(self._params.get_bool(key))
    except Exception:
      return default

  def _read_int(self, key: str, default: int) -> int:
    if self._params is None:
      return default
    try:
      raw = self._params.get(key)
      if raw is None:
        return default
      return int(raw)
    except (ValueError, TypeError):
      return default

  # ----- Helpers ----------------------------------------------------------

  def _consume_global_rate_limit(self, frame: int, n_logical: int,
                                  limit: int = GLOBAL_RATE_LIMIT_PRESSES_PER_SEC) -> bool:
    """Return True and record the TX if adding `n_logical` logical button
    commands in the last 1 s stays at or under `limit`.

    Burst copies are reliability duplicates (same logical press repeated for
    the cluster to see), so callers pass 1 per logical command — not BURST_COPIES.

    iter8: callers can pass a higher `limit` during the decel-fast regime
    (DECEL_FAST_RATE_LIMIT_PRESSES_PER_SEC = 12) so cluster pull-down can
    keep up with hard braking. Default behavior unchanged.
    """
    cutoff = frame - FRAMES_PER_SEC
    while self.press_history and self.press_history[0][0] < cutoff:
      self.press_history.popleft()
    total = sum(c for _, c in self.press_history)
    if total + n_logical > limit:
      return False
    self.press_history.append((frame, n_logical))
    return True

  def _process_button_events(self, CS, observed_set_speed: float, frame: int,
                              just_engaged: bool) -> None:
    """Feed driver wheel input into user_target + driver-override windows.

    First matching event within ECHO_FILTER_FRAMES of our TX is swallowed
    as our own echo; subsequent events pass through. Physical driver edges
    adjust user_target by ±1 mph each. The SCC's 5-mph quantization step
    that follows a held button is caught later in _maybe_observe_quantization
    (display tracking only — never used as a control input).

    Per drive #4 fix: on the just_engaged frame, accelCruise + decelCruise
    events are the user enabling cruise (RES re-engage / SET-from-off), NOT
    directional intent. Skip them entirely so we don't open driver-override
    windows that survive into the engaged session.
    """
    echo_window_open = (
      self._pending_echo_button != Buttons.NONE
      and (frame - self._pending_echo_frame) < ECHO_FILTER_FRAMES
    )
    for event in CS.out.buttonEvents:
      if not event.pressed:
        continue
      # Try to consume the echo on the first matching event
      if echo_window_open:
        our_type = TX_BUTTON_TO_EVENT_TYPE.get(self._pending_echo_button)
        if our_type is not None and event.type == our_type:
          self._pending_echo_button = Buttons.NONE
          echo_window_open = False
          continue
      # Real driver press
      if just_engaged and event.type in (ButtonType.accelCruise, ButtonType.decelCruise):
        continue
      if event.type == ButtonType.decelCruise:
        self.user_target_speed -= MPH_TO_MS
        self._driver_set_last_frame = frame
        self._driver_adjust_until = frame + DRIVER_ADJUST_WINDOW_FRAMES
      elif event.type == ButtonType.accelCruise:
        self.user_target_speed += MPH_TO_MS
        self._driver_res_last_frame = frame
        self._driver_adjust_until = frame + DRIVER_ADJUST_WINDOW_FRAMES
      # cancel handled by cc_enabled going False on the next frame
    self.user_target_speed = max(USER_TARGET_MIN_MS, min(USER_TARGET_MAX_MS, self.user_target_speed))

  def _maybe_observe_quantization(self, observed_set_speed: float, frame: int) -> None:
    """During the 300 ms after a real driver edge, watch for the Hyundai SCC's
    delayed +5 mph quantization step (held button → cluster bumps observed
    by 5 mph a few frames after our buttonEvent edge). Snap user_target to
    observed so the HUD `User XX mph` line stays consistent with what the
    driver actually got. Display tracking only — never used as a control
    input.
    """
    if frame >= self._driver_adjust_until:
      return
    # Don't react to observed changes that are likely our own SET/RES landing.
    last_tx_frame = max(self.last_set_frame, self.last_res_frame)
    if (frame - last_tx_frame) < TX_ECHO_ATTRIBUTION_FRAMES:
      return
    # Pick whichever driver edge is MORE RECENT (so a SET right after a RES
    # routes through the SET branch correctly).
    res_age = frame - self._driver_res_last_frame
    set_age = frame - self._driver_set_last_frame
    res_recent = res_age < DRIVER_ADJUST_WINDOW_FRAMES
    set_recent = set_age < DRIVER_ADJUST_WINDOW_FRAMES
    most_recent_is_res = res_recent and (not set_recent or res_age <= set_age)
    most_recent_is_set = set_recent and (not res_recent or set_age < res_age)
    if most_recent_is_res and observed_set_speed > self.user_target_speed + 0.5 * MPH_TO_MS:
      self.user_target_speed = min(USER_TARGET_MAX_MS, observed_set_speed)
    elif most_recent_is_set and observed_set_speed < self.user_target_speed - 0.5 * MPH_TO_MS:
      self.user_target_speed = max(USER_TARGET_MIN_MS, observed_set_speed)

  def _in_driver_override_set(self, frame: int) -> bool:
    return (frame - self._driver_set_last_frame) < DRIVER_OVERRIDE_SET_FRAMES

  def _in_driver_override_res(self, frame: int) -> bool:
    return (frame - self._driver_res_last_frame) < DRIVER_OVERRIDE_RES_FRAMES

  def _update_standstill(self, v_ego, brake_pressed, standstill_flag) -> bool:
    trigger = (
      v_ego < STANDSTILL_V_EGO_MS
      or standstill_flag
      or (brake_pressed and v_ego < BRAKE_LOW_SPEED_V_EGO_MS)
    )
    exit_ok = (v_ego >= STANDSTILL_EXIT_V_EGO_MS and not brake_pressed and not standstill_flag)
    if trigger:
      self._standstill_trigger_frames += 1
      if self._standstill_trigger_frames >= STANDSTILL_CONFIRM_FRAMES:
        self._standstill_on = True
    else:
      self._standstill_trigger_frames = 0
    if self._standstill_on and exit_ok:
      self._standstill_on = False
    return self._standstill_on

  @staticmethod
  def _dynamic_margin_ms(v_ego_ms: float, est_power_w: float, power_threshold_w: float) -> float:
    """Sliding cap formula with iter7 power-aware bonus.

    Base margin (vEgo-only):
      vEgo = 0       → LOW_SPEED_MARGIN_MPH (20 mph default)
      vEgo = 30 mph  → HIGH_SPEED_MARGIN_MPH (5 mph default)
      vEgo > 30 mph  → flat at HIGH_SPEED_MARGIN_MPH

    Low-load bonus (iter7 dwell fix):
      load_frac <= LOAD_BONUS_LOW_FRAC (0.4)  → +LOW_LOAD_BONUS_MPH (10 mph)
      load_frac >= LOAD_BONUS_HIGH_FRAC (0.8) → +0 mph
      between                                 → linear taper

    Fail safe: if power_threshold_w is invalid (≤0), bonus is 0 — never
    treat invalid threshold as low load.
    """
    v_ego_mph = v_ego_ms / MPH_TO_MS
    if v_ego_mph >= MARGIN_BLEND_END_MPH:
      base_mph = HIGH_SPEED_MARGIN_MPH
    elif v_ego_mph <= 0.0:
      base_mph = LOW_SPEED_MARGIN_MPH
    else:
      frac = v_ego_mph / MARGIN_BLEND_END_MPH
      base_mph = LOW_SPEED_MARGIN_MPH + frac * (HIGH_SPEED_MARGIN_MPH - LOW_SPEED_MARGIN_MPH)

    if power_threshold_w <= 0.0:
      bonus_mph = 0.0
    else:
      load_frac = max(0.0, est_power_w) / power_threshold_w
      if load_frac <= LOAD_BONUS_LOW_FRAC:
        bonus_mph = LOW_LOAD_BONUS_MPH
      elif load_frac >= LOAD_BONUS_HIGH_FRAC:
        bonus_mph = 0.0
      else:
        taper = (LOAD_BONUS_HIGH_FRAC - load_frac) / (LOAD_BONUS_HIGH_FRAC - LOAD_BONUS_LOW_FRAC)
        bonus_mph = LOW_LOAD_BONUS_MPH * taper

    margin_mph = base_mph + bonus_mph
    # Defensive clamp — should never trigger given the math above, but cheap insurance.
    if margin_mph < base_mph:
      margin_mph = base_mph
    elif margin_mph > base_mph + LOW_LOAD_BONUS_MPH:
      margin_mph = base_mph + LOW_LOAD_BONUS_MPH
    return margin_mph * MPH_TO_MS

  def _record_tx(self, frame: int, button: int) -> None:
    if button == Buttons.SET_DECEL:
      self.last_set_frame = frame
    elif button == Buttons.RES_ACCEL:
      self.last_res_frame = frame
    self._pending_echo_button = button
    self._pending_echo_frame = frame

  # --- iter13 v4 ACK API (R4-MF3) ---------------------------------------

  def note_set_emitted(self, frame_idx: int) -> None:
    """Called by CarController AFTER rate-limit acceptance for a SET emission.

    R4-MF3 invariant: ONLY post-acceptance frames enter the ACK matcher.
    CarController must NOT call this for desired-but-dropped SETs.
    """
    self._unmatched_emitted_set_frames.append(frame_idx)
    # Bound the deque so old un-acked entries don't grow unbounded.
    threshold = frame_idx - ACK_WINDOW_FRAMES
    while self._unmatched_emitted_set_frames and self._unmatched_emitted_set_frames[0] < threshold:
      self._unmatched_emitted_set_frames.popleft()

  def try_ack_set(self, cluster_now_ms: float, frame_now: int,
                   physical_button_in_window: bool) -> bool:
    """Match a cluster decrement to oldest unmatched emitted SET within ack window.

    R4-MF5 / Section F invariants:
      1. cluster_now must be strictly less than the previous reading (decrement).
      2. Candidate SET frames must be strictly less than frame_now (post-emission).
      3. (frame_now - f) <= ACK_WINDOW_FRAMES (500ms inclusive).
      4. One-to-one: each ACK consumes the oldest unmatched candidate.
      5. If a physical (driver) cruise button was active in the window, do NOT
         credit the ACK (decrement may have been driver-induced).
    """
    decremented = cluster_now_ms < self._cluster_prev_for_ack_ms
    self._cluster_prev_for_ack_ms = cluster_now_ms
    if not decremented:
      return False
    candidates = [f for f in self._unmatched_emitted_set_frames
                   if f < frame_now and (frame_now - f) <= ACK_WINDOW_FRAMES]
    if not candidates or physical_button_in_window:
      return False
    oldest = min(candidates)
    self._unmatched_emitted_set_frames.remove(oldest)
    self._cluster_decrement_acked += 1
    self._consecutive_no_ack_emits = 0
    if not self._prelaunch_first_ack_seen and self._was_in_prelaunch:
      self._prelaunch_first_ack_seen = True
      self._prelaunch_set_pulse_cap = STANDSTILL_PULSE_CAP_AFTER_ACK
    return True

  # --- iter13 v4 advisory helpers ----------------------------------------

  def _compute_launch_target_mph(self, user_target_ms: float, v_ego_ms: float) -> float:
    """v4 Section C launch target formula: max(21, min(user_target, vEgo+gap))."""
    user_target_mph = user_target_ms / MPH_TO_MS
    v_ego_mph = v_ego_ms / MPH_TO_MS
    return max(STANDSTILL_TARGET_FLOOR_MPH,
               min(user_target_mph, v_ego_mph + ALLOWABLE_GAP_MPH))

  def _compute_advisory_block_reason(self, frame: int, governor_mode: int,
                                      power_too_high: bool, want_set: bool,
                                      want_res: bool) -> str:
    """Compute the EVLimiter advisory block reason (Section D priorities 13-19).

    Deterministic priority within EVLimiter (highest first):
      13 modeForbidden          — state-machine BRAKE/DECEL/STANDSTILL_HOLD/IDLE non-actionable
      14 evModeAssumedFalse     — EV-only assumption disabled
      15 paramReadFailed        — Params() read failure (advisory; param plumbing
                                   handles the actual gate)
      16 minIntervalNotMet      — 0.5s hard min between desired SETs
      17 cooldownActive         — ack-driven cooldown
      18 standstillNoAckBackoff — 3s backoff after 5 emit-no-ack
      19 standstillCapReached
    Returns 'none' if no advisory reason applies (or if a SET/RES is desired).
    """
    if want_set or want_res:
      return "none"
    if governor_mode in (GOVERNOR_MODE_BRAKE, GOVERNOR_MODE_DECEL,
                          GOVERNOR_MODE_STANDSTILL):
      return "modeForbidden"
    # iter13 v4: advisory min-interval / cooldown / backoff. We fold these
    # into a single check ordered by priority.
    elapsed_set = frame - self.last_set_frame
    if (self._was_in_prelaunch
        and self._prelaunch_set_pulses_emitted >= self._prelaunch_set_pulse_cap):
      return "standstillCapReached"
    if frame < self._prelaunch_no_ack_backoff_until:
      return "standstillNoAckBackoff"
    if frame < self._set_no_ack_backoff_until_frame:
      return "cooldownActive"
    if elapsed_set < SET_HARD_MIN_INTERVAL_FRAMES and self.last_set_frame > 0:
      return "minIntervalNotMet"
    return "none"

  def _set_desired_button(self, desired: int | None) -> None:
    """R4-MF4 runtime allowlist guard. Use raise (not assert) — production may
    run with `python -O` which strips assertions."""
    if desired not in EVLIMITER_ALLOWED_BUTTONS:
      raise EVLimiterError(
        f"EVLimiter requested invalid button {desired!r}; "
        f"allowed = {EVLIMITER_ALLOWED_BUTTONS}")
    self._desired_button = desired

  def _reset_tx_cadence(self) -> None:
    """Called on entry to STANDSTILL_HOLD / driver override — drops pending
    cadence so we don't snap a stale cooldown the instant we exit."""
    self.last_set_frame = -10000
    self.last_res_frame = -10000

  # ----- iter10 Layer 1: bounded reference governor -----------------------

  def _has_decel_intent(self, frame: int) -> bool:
    """Return True iff POSITIVE evidence of intentional decel.
    Conservative: uncertain → False (caller defaults to MODE_NORMAL, which
    enforces the no-cluster-below-vEgo invariant). v2 critique resolved.
    Lead-radar TTC clause is plumbed via radar_decel_intent set by card.py
    (defaults to None when unavailable on classic-CAN HYBRID).
    """
    if (frame - self._last_brake_frame) < BRAKE_RECENT_FRAMES:
      return True
    if self._scc_decel_persistent_frames >= SCC_DECEL_PERSIST_FRAMES:
      return True
    return False

  def _select_governor_mode(self, frame: int, v_ego: float, gas_pressed: bool,
                             brake_pressed: bool, in_standstill: bool,
                             observed_set_speed: float) -> int:
    """Select exclusive governor mode for this tick. Order matters:
    standstill > brake > gas-catchup (if armed) > decel-intent > normal."""
    if in_standstill or v_ego < STANDSTILL_V_THRESHOLD_MS:
      return GOVERNOR_MODE_STANDSTILL
    if brake_pressed:
      return GOVERNOR_MODE_BRAKE
    deficit_ms = self.user_target_speed - observed_set_speed
    catchup_armed = (
      gas_pressed
      and self._gas_hold_frames >= GAS_HOLD_MIN_FRAMES
      and deficit_ms > GAS_CATCHUP_MIN_DEFICIT_MPH * MPH_TO_MS
    )
    if catchup_armed:
      return GOVERNOR_MODE_GAS_CATCHUP
    if self._has_decel_intent(frame):
      return GOVERNOR_MODE_DECEL
    return GOVERNOR_MODE_NORMAL

  def _compute_governor_bounds(self, mode: int, v_ego: float, frame: int,
                                observed_set_speed: float,
                                dynamic_ceiling: float) -> tuple[float, float]:
    """iter11 Fix A: max-deficit invariant is HARD in NORMAL mode, subject
    only to enumerated safety overrides:
    - lead/brake/decel handled via MODE_DECEL/MODE_BRAKE selection upstream
    - engagement transient (just engaged or vEgo too low to act) suspends floor

    Drive #9-13 forensics: the v0/iter10 'low-speed dynamic ceiling' suspension
    turned a hard invariant into a conditional and produced 22-44 mph offsets.
    iter11: floor is hard. If sliding cap wants cluster lower, it loses to
    the floor (cluster simply won't be cap-down'd below floor)."""
    upper_bound = self.user_target_speed
    max_deficit_mph = self._read_int(MAX_DEFICIT_PARAM, int(MAX_DEFICIT_DEFAULT_MPH))
    default_lower = max(USER_TARGET_MIN_MS, self.user_target_speed - max_deficit_mph * MPH_TO_MS)

    if mode == GOVERNOR_MODE_NORMAL:
      # iter11 Fix A: engagement-transient safety override. Lets cluster sit
      # wherever it lands at engage edge (probably below floor) for first 1.0 s
      # before strict floor enforcement.
      ENGAGEMENT_TRANSIENT_FRAMES = 100   # 1.0 s @ 100 Hz
      is_engagement_transient = (
        not self.was_cc_enabled
        or (frame - self._engaged_at_frame) < ENGAGEMENT_TRANSIENT_FRAMES
      )

      # iter12: REMOVED max-deficit hard floor entirely. iter11 Fix A was the
      # wrong abstraction — it was solving a SYMPTOM (offset) when the cause
      # was Hyundai SCC ignoring rapid RES bursts (Bug D). The hard floor blocked
      # the sliding cap from operating at low/mid vEgo with high user_target,
      # leading to cluster pinned 15-20 mph above vEgo → ICE activations
      # (today's drive). The right fix for the original problem is Fix D's
      # ineffective-RES escape, which iter12 retains.
      #
      # iter12 NORMAL bounds: ONLY the no-below-vEgo invariant (iter10 Fix B),
      # which is correct and orthogonal to max-deficit.
      if is_engagement_transient:
        lower_bound = USER_TARGET_MIN_MS
      elif self.user_target_speed > v_ego + EGO_SLOP_MS:
        # I2: never cluster < vEgo - slop when user_target above ego (oscillation prevention)
        lower_bound = max(USER_TARGET_MIN_MS, v_ego - EGO_SLOP_MS)
      else:
        lower_bound = USER_TARGET_MIN_MS

    elif mode == GOVERNOR_MODE_GAS_CATCHUP:
      # v2 fix: non-inverting by construction.
      default_upper = min(self.user_target_speed, v_ego + GAS_HEADROOM_MPH * MPH_TO_MS)
      lower_bound = max(USER_TARGET_MIN_MS, observed_set_speed)
      upper_bound = max(default_upper, observed_set_speed)
      # If cluster < vEgo+headroom: upper=vEgo+2, lower=observed → catch-up via RES (Event A).
      # If cluster > vEgo+headroom: upper=lower=observed → cluster holds (no RES, no SET).

    elif mode == GOVERNOR_MODE_DECEL:
      # SCC commanding decel — let cluster dip below vEgo for legitimate slowdown.
      lower_bound = USER_TARGET_MIN_MS

    elif mode == GOVERNOR_MODE_BRAKE:
      lower_bound = USER_TARGET_MIN_MS

    else:  # GOVERNOR_MODE_STANDSTILL
      lower_bound = USER_TARGET_MIN_MS

    if lower_bound > upper_bound:
      # Defensive: should only happen in NORMAL with degenerate user_target/vEgo.
      # Saturate to permissive bounds and log.
      self._bound_inversion_count += 1
      lower_bound = USER_TARGET_MIN_MS

    return lower_bound, upper_bound

  # ----- Main update ------------------------------------------------------

  def update(self, CC, CS, frame: int) -> tuple[int, bool]:
    """Advance the limiter one tick. Return (button, active)."""
    # iter14 v2 instrumentation: capture state at start of update() for replay forensics.
    self._state_prior_transition_last = int(self.state)

    if not self.supported:
      return self._publish(Buttons.NONE, STATE_DISABLED, 0.0, frame=frame, reason="unsupported")

    if not self._read_bool("EVLimiterEnabled", False):
      self.user_target_speed = 0.0
      return self._publish(Buttons.NONE, STATE_DISABLED, 0.0, frame=frame, reason="disabled_param")

    # Tunables (read every frame so params changes take effect live)
    power_threshold_w = float(self._read_int("EVLimiterPowerThresholdKW", 40)) * 1000.0
    dte_floor = float(self._read_int("EVLimiterDTEFloor", 5))

    # Inputs
    cc_enabled = bool(CC.enabled)
    v_ego = float(CS.out.vEgo)
    a_ego = float(CS.out.aEgo)
    observed_set_speed = float(CS.out.cruiseState.speed)
    brake_pressed = bool(CS.out.brakePressed)
    gas_pressed = bool(CS.out.gasPressed)
    standstill_flag = bool(getattr(CS.out.cruiseState, "standstill", False))
    est_power_w = float(getattr(CS, "est_power_w", 0.0))
    # iter14 v2 (R1-MF3): control-side power signal for state arbiter — short-tau LP,
    # uncapped. Decoupled from HUD estPowerW (2 s fall) to avoid drive 18 t=1331-1352
    # bug where stale HUD-smoothed value masked re-entry conditions. Falls back to
    # est_power_w if carstate_ext hasn't populated it yet (transition / unit tests).
    est_power_control_w = float(getattr(CS, "est_power_control_w", est_power_w))
    abasis = float(getattr(CS, "accel_demand", 0.0))
    dte_raw = float(getattr(CS, "dte_raw", 0.0))

    # Handle DISABLED state EARLY — track engage-classification inputs but
    # do NOT process button events through the normal driver-adjust path.
    # Drive #4 race fix: a wheel RES press can land 1-2 frames before
    # cc_enabled rises (CAN ordering). If we processed it as if cruise were
    # active we'd open DRV_RES + driver-adjust windows that survive the
    # engage transition and corrupt the engage-edge seed.
    if not cc_enabled:
      # Track cluster's last-observed set so RES re-engage can seed
      # user_target from this (vs SCC-polluted post-engage observed).
      self._observed_set_speed_at_disable = observed_set_speed
      # Note physical RES presses for engage classification — but only
      # ones that are NOT our own TX echoes (we shouldn't TX while
      # disabled, but defensive against stale echoes).
      echo_window_open = (
        self._pending_echo_button != Buttons.NONE
        and (frame - self._pending_echo_frame) < ECHO_FILTER_FRAMES
      )
      our_type = TX_BUTTON_TO_EVENT_TYPE.get(self._pending_echo_button) if echo_window_open else None
      for e in CS.out.buttonEvents:
        if not e.pressed or e.type != ButtonType.accelCruise:
          continue
        if echo_window_open and our_type == ButtonType.accelCruise:
          # Consume our own echo, don't count as physical press.
          self._pending_echo_button = Buttons.NONE
          echo_window_open = False
          our_type = None
          continue
        self._last_disabled_res_press_frame = frame
        break
      # Drop stale driver-adjust / override state from a previous engaged
      # session so the engage frame starts clean.
      self._driver_adjust_until = -10000
      self._driver_res_last_frame = -10000
      self._driver_set_last_frame = -10000
      # Reset control internals.
      self._reset_tx_cadence()
      self._standstill_trigger_frames = 0
      self.was_cc_enabled = cc_enabled
      return self._publish(Buttons.NONE, STATE_DISABLED, observed_set_speed,
                           frame=frame, reason="cc_off")

    # cc_enabled is True from here on.
    # Engage rising edge -> seed user_target.
    just_engaged = not self.was_cc_enabled
    if just_engaged:
      self._engaged_at_frame = frame   # iter11 Fix A: engagement-transient timing
      # Engage via RES? Check current-frame buttonEvents AND recent
      # disabled-frame RES presses (the wheel press may have landed before
      # cc_enabled rose). Engage via SET / main-switch leaves observed
      # clean (no SCC autonomous resume snap), so we only redirect the
      # seed for engage_via_res cases.
      engage_via_res = False
      for e in CS.out.buttonEvents:
        if e.pressed and e.type == ButtonType.accelCruise:
          engage_via_res = True
          break
      if not engage_via_res:
        recent_disabled_res = (frame - self._last_disabled_res_press_frame) <= DISABLED_RES_ENGAGE_WINDOW_FRAMES
        if recent_disabled_res:
          engage_via_res = True
      if engage_via_res and self._observed_set_speed_at_disable > 0.5 * MPH_TO_MS:
        # Drive #4: SCC autonomously snaps cluster up on RES re-engage.
        # Seed from the frozen pre-disable value instead.
        self.user_target_speed = self._observed_set_speed_at_disable
      else:
        self.user_target_speed = observed_set_speed
      # Defensive: ensure no stale adjust window survives into engaged
      # state (DISABLED branch already clears these but make it explicit
      # for readers expecting engage-edge invariants).
      self._driver_adjust_until = -10000
    self.was_cc_enabled = cc_enabled

    # Driver wheel input — only processed when cc_enabled (disabled-state
    # button events are handled in the DISABLED branch above). On the
    # just_engaged frame, accelCruise events are skipped (engage press
    # isn't accel intent).
    self._process_button_events(CS, observed_set_speed, frame, just_engaged)

    # Observe the SCC's 5-mph quantization step that lands a few frames after
    # a held driver button — display/recovery-target tracking only, never
    # used as a control input.
    self._maybe_observe_quantization(observed_set_speed, frame)

    if dte_raw <= dte_floor:
      return self._publish(Buttons.NONE, STATE_DISABLED, observed_set_speed,
                           frame=frame, reason="dte_floor")

    # Standstill gate. raw_standstill fires immediately (no 20-frame confirm)
    # so engaging while already stopped — or a brief sub-2 mph dip — suppresses
    # RES in-frame. The confirm-based `_standstill_on` continues to track the
    # post-standstill-clear hold-off (RECOVERY_AFTER_STANDSTILL).
    #
    # iter7: SET is allowed at standstill (pulse-bounded) so cluster_set can
    # be pulled down toward LOW_SPEED_MARGIN_MPH (20 mph) at red lights,
    # rather than being frozen wherever the deceleration SET cascade landed
    # when raw_standstill kicked in (drive #6: cluster frozen at 31 with
    # user_target=62). RES remains blocked at standstill.
    raw_standstill = (
      v_ego < STANDSTILL_V_EGO_MS
      or standstill_flag
      or (brake_pressed and v_ego < BRAKE_LOW_SPEED_V_EGO_MS)
    )
    standstill = self._update_standstill(v_ego, brake_pressed, standstill_flag)
    in_standstill = raw_standstill or standstill

    # Reset SET-pulse counter on each entry to standstill.
    if in_standstill and not self._was_in_standstill_last_frame:
      self._standstill_set_pulses = 0
      # iter13 v4: also reset PRELAUNCH_SET-state tracking on standstill entry.
      self._prelaunch_set_pulses_emitted = 0
      self._prelaunch_set_pulses_emitted_since_ack = 0
      self._prelaunch_set_pulse_cap = STANDSTILL_PULSE_CAP_INITIAL
      self._prelaunch_first_ack_seen = False
      self._prelaunch_no_ack_backoff_until = -10000
      # iter16a (B1): reset per-stop backoff-retry tracking on each new stop.
      self._prelaunch_backoff_retries = 0
      self._prelaunch_backoff_was_active = False
      self._standstill_entered += 1
      # iter15 v2 (Section C): reset time-in-standstill counter on entry.
      self._time_in_standstill_frames = 0

    # iter15 v2 (Section C): standstill-EXIT detection. When `_was_in_standstill_last_frame`
    # was True and now `in_standstill` is False, this is the exit frame. Run narrow
    # reset (R1-MF-C) + latch state-vector snapshot for next-drive forensics.
    standstill_just_exited = (self._was_in_standstill_last_frame and not in_standstill)
    if standstill_just_exited:
      self._on_leaving_standstill(
        frame=frame,
        time_in_standstill_frames=self._time_in_standstill_frames,
        est_power_control_w=est_power_control_w,
        power_threshold_w=power_threshold_w,
      )

    # Increment time-in-standstill while still in standstill.
    if in_standstill:
      self._time_in_standstill_frames += 1

    self._was_in_standstill_last_frame = in_standstill

    # iter15 v2 (Section C) post-exit RES-latency tracker. After standstill exit
    # (set above), increment each frame until a RES_ACCEL is emitted; latch the
    # frame count on first RES. Resets on next exit.
    if self._standstill_exit_post_exit_active and not self._standstill_exit_first_res_seen:
      self._standstill_exit_to_first_res_latency_frames += 1

    # iter14 v2 carryover (drives 18+19 forensic finding): _was_in_prelaunch flag
    # was sticky across the standstill→moving transition, causing SOFT_CAP / IDLE /
    # RECOVERY SET emissions while moving to be wrongly attributed to the standstill
    # slice counter. Reset BEFORE standstill block so the in_standstill branch can
    # set it True again if conditions hold; outside standstill it stays False.
    if not in_standstill:
      self._was_in_prelaunch = False

    if in_standstill:
      self._left_standstill_at_frame = frame
      # iter13 v4: STANDSTILL_PRELAUNCH_SET vs STANDSTILL_HOLD.
      # PRELAUNCH_SET when cluster is above launch_target + deadband and
      # driver is not braking/gas-pressing; HOLD otherwise (no SET emitted).
      launch_target_mph = self._compute_launch_target_mph(self.user_target_speed, v_ego)
      cluster_mph = observed_set_speed / MPH_TO_MS
      cluster_above_launch = cluster_mph > launch_target_mph + LAUNCH_DEADBAND_MPH
      no_ack_backoff_active = frame < self._prelaunch_no_ack_backoff_until
      # iter16a (B1): when the no-ack backoff expires (falling edge), refresh the
      # per-stop pulse budget so a long red light keeps making BOUNDED set-down
      # attempts instead of permanently giving up at the initial cap. Capped at
      # MAX_PRELAUNCH_BACKOFF_RETRIES per stop. Only matters before first ack; once
      # the SCC acks, the larger AFTER_ACK cap already gives ample budget.
      if (self._prelaunch_backoff_was_active and not no_ack_backoff_active
          and not self._prelaunch_first_ack_seen
          and self._prelaunch_backoff_retries < MAX_PRELAUNCH_BACKOFF_RETRIES):
        self._prelaunch_set_pulses_emitted = 0
        self._prelaunch_set_pulses_emitted_since_ack = 0
        self._prelaunch_backoff_retries += 1
      self._prelaunch_backoff_was_active = no_ack_backoff_active
      cap_reached = (self._prelaunch_set_pulses_emitted
                     >= self._prelaunch_set_pulse_cap)

      can_prelaunch_set = (
        cluster_above_launch
        and not gas_pressed
        and not brake_pressed
        and not cap_reached
        and not no_ack_backoff_active
        and (frame - self.last_set_frame) >= SET_HARD_MIN_INTERVAL_FRAMES
        and self._consume_global_rate_limit(frame, 1)
      )
      self._was_in_prelaunch = cluster_above_launch and not gas_pressed and not brake_pressed

      if can_prelaunch_set:
        # iter13 v4: count requested + log decision-side. CarController is
        # authoritative on whether the SET actually goes out and increments
        # _set_emitted/_standstill_set_emitted via its own counter source.
        self._set_requested += 1
        self._standstill_set_requested += 1
        self._prelaunch_set_pulses_emitted += 1
        self._prelaunch_set_pulses_emitted_since_ack += 1
        self.last_set_frame = frame
        self._cluster_at_last_set = observed_set_speed
        self._standstill_set_pulses += 1   # legacy counter (kept for backwards-compat)
        # Adaptive backoff: 5 emit-no-ack pulses → 3.0s backoff
        if (not self._prelaunch_first_ack_seen
            and self._prelaunch_set_pulses_emitted_since_ack
                  >= STANDSTILL_NO_ACK_BACKOFF_AFTER_EMITTED):
          self._prelaunch_no_ack_backoff_until = frame + STANDSTILL_NO_ACK_BACKOFF_FRAMES
        self._record_tx(frame, Buttons.SET_DECEL)
        return self._publish(Buttons.SET_DECEL, STATE_STANDSTILL_PRELAUNCH_SET,
                             observed_set_speed, frame=frame,
                             reason="prelaunch_set",
                             burst_count=1)

      # cluster at/below launch target, OR backoff active, OR cap reached:
      # publish STANDSTILL_HOLD (no SET).
      if cluster_above_launch and (cap_reached or no_ack_backoff_active):
        # Track exit-by-no-ack for telemetry on the next cycle's standstill exit
        if no_ack_backoff_active and self._was_in_prelaunch:
          # We will eventually exit; defer counter inc to leave-standstill block.
          pass
      elif not cluster_above_launch:
        # Achieved the launch target — count once per standstill window.
        if self._was_in_prelaunch and not self._prelaunch_first_ack_seen:
          # Only count if we actually entered PRELAUNCH this standstill.
          pass
      return self._publish(Buttons.NONE, STATE_STANDSTILL_HOLD, observed_set_speed,
                           frame=frame, reason="standstill")

    # Driver override windows.
    in_override_set = self._in_driver_override_set(frame)
    in_override_res = self._in_driver_override_res(frame)

    # One-direction-at-a-time anti-oscillation: after our SET, block our RES
    # for 1.5 s. Same symmetrically for RES blocking SET.
    self_set_recent = (frame - self.last_set_frame) < LIMITER_OPPOSITE_DIR_BLOCK_FRAMES
    self_res_recent = (frame - self.last_res_frame) < LIMITER_OPPOSITE_DIR_BLOCK_FRAMES

    # iter8: decel-fast regime. When in low-speed range AND decelerating, use
    # tighter SET cadence + higher rate-limit so cluster pull-down keeps up
    # with brake decel (typical 8-15 mph/s). Default 6.7 mph/s pull-down lost
    # the race in drive #6 (cluster frozen at 31 when raw_standstill latched).
    # Note: aEgo is the kinematic ground-frame accel from wheel speed, so a
    # negative value means actual decel regardless of cause (driver brake,
    # SCC command, coasting downhill).
    decel_active = (v_ego < DECEL_FAST_VEGO_THRESHOLD_MS) and (a_ego < DECEL_FAST_AEGO_MS2)
    set_cooldown_frames = SET_COOLDOWN_DECEL_FAST_FRAMES if decel_active else SET_COOLDOWN_FRAMES
    rate_limit = DECEL_FAST_RATE_LIMIT_PRESSES_PER_SEC if decel_active else GLOBAL_RATE_LIMIT_PRESSES_PER_SEC

    # iter10 Layer 1: update governor state-tracking counters every tick.
    if gas_pressed:
      self._gas_hold_frames += 1
    else:
      self._gas_hold_frames = 0
    if brake_pressed:
      self._last_brake_frame = frame
    # SCC decel detected via accelDemand (TCS13.aBasis aggregate). Available
    # on CarState.accel_demand — extracted by carstate_ext upstream.
    abasis_signal = float(getattr(CS, "accel_demand", 0.0))
    if abasis_signal < SCC_DECEL_DETECT_MS2:
      self._scc_decel_persistent_frames += 1
    else:
      self._scc_decel_persistent_frames = 0

    # Sliding cap (iter6 core). Cluster set is held within
    #   target_set = min(user_target, vEgo + dynamic_margin(vEgo))
    # Push DOWN when observed > target_set + deadband (sliding cap violation)
    #   OR when est_power > threshold AND there's a gap to close (load gate).
    # Push UP when observed < target_set - deadband (gentle recovery).
    margin = self._dynamic_margin_ms(v_ego, est_power_w, power_threshold_w)
    dynamic_ceiling = v_ego + margin
    target_set = min(self.user_target_speed, dynamic_ceiling)

    # iter10 Layer 1: clamp target_set to bounded-governor [lower, upper].
    # Mode selection runs each tick; bounds enforce invariants that prevent
    # Event A (dwell) and Event B (cluster < vEgo) by construction.
    governor_mode = self._select_governor_mode(
      frame, v_ego, gas_pressed, brake_pressed, in_standstill, observed_set_speed
    )
    lower_bound, upper_bound = self._compute_governor_bounds(
      governor_mode, v_ego, frame, observed_set_speed, dynamic_ceiling
    )
    target_set = max(lower_bound, min(upper_bound, target_set))

    # iter16a (C1): below-vEgo power-droop SIMULATION (log-only / default-off).
    # When the control-side est power stays over the enter threshold for the sustain
    # window with no driver input, simulate a droop target (vEgo - ramped droop) that
    # would force a real slowdown to cut power on a grade — the case the headroom-gated
    # SET path cannot handle. We publish what we WOULD do; we only actually lower
    # target_set if EvLimiterPowerDroopEnable is on (default OFF — fail-safe). gpt-5.5
    # blocked shipping this active until a real-power-instrumented drive validates it.
    droop_no_driver = (not gas_pressed and not brake_pressed
                       and not self._in_driver_override_set(frame)
                       and not self._in_driver_override_res(frame))
    power_ctl_kw = est_power_control_w / 1000.0
    if power_ctl_kw >= POWER_DROOP_ENTER_KW and droop_no_driver and v_ego > RECOVERY_V_EGO_FLOOR_MS:
      self._power_droop_sustain += 1
    elif power_ctl_kw < POWER_DROOP_EXIT_KW or not droop_no_driver:
      self._power_droop_sustain = 0
    self._power_droop_would_enter = self._power_droop_sustain >= POWER_DROOP_SUSTAIN_FRAMES
    if self._power_droop_would_enter:
      self._power_droop_request_mph = min(POWER_DROOP_MAX_MPH,
                                          self._power_droop_request_mph + POWER_DROOP_RAMP_MPH_PER_FRAME)
    else:
      self._power_droop_request_mph = 0.0
    self._power_droop_active = self._power_droop_would_enter and self._read_bool(POWER_DROOP_ENABLE_PARAM, False)
    if self._power_droop_active:
      droop_target = v_ego - self._power_droop_request_mph * MPH_TO_MS
      target_set = max(USER_TARGET_MIN_MS, min(target_set, droop_target))

    # iter11 Fix A: max-deficit violation forensic counter
    if governor_mode == GOVERNOR_MODE_NORMAL and not gas_pressed and not brake_pressed:
      if observed_set_speed < lower_bound - 0.5:   # any sustained dip below floor
        if not hasattr(self, '_max_deficit_violation_frames'):
          self._max_deficit_violation_frames = 0
        self._max_deficit_violation_frames += 1

    set_too_high = observed_set_speed > target_set + SET_TRIGGER_DEADBAND_MS
    power_too_high = (
      est_power_w > power_threshold_w
      and observed_set_speed > v_ego + POWER_GAP_DEADBAND_MS
    )
    # iter11 Fix B: arbiter reads this to allow IDLE→SOFT_CAP without sustain
    # (preserves iter9 fast-protect on power_too_high)
    self._power_too_high_recent = power_too_high
    under_target = observed_set_speed < target_set - RECOVERY_DEADBAND_MS

    # iter11 Fix A: recovery escape — when cluster < lower_bound, RES is
    # allowed even with power_too_high, but rate-limited to ≤2 mph/s elapsed.
    cluster_below_floor = observed_set_speed < lower_bound - 0.1   # 0.1 m/s slop
    if cluster_below_floor and not self._recovery_escape_active:
      # Entering escape mode
      self._recovery_escape_active = True
      self._recovery_escape_start_t = frame * 0.01   # 100 Hz → seconds
      self._recovery_escape_start_cluster = observed_set_speed
    elif not cluster_below_floor and self._recovery_escape_active:
      self._recovery_escape_active = False

    # Down-trigger: fires whenever sliding cap is violated OR load is high.
    # Driver overrides do NOT suppress us:
    # - DRIVER_OVERRIDE_RES (drive #4 lesson): respect window doesn't extend
    #   to letting motor cross ICE boundary
    # - DRIVER_OVERRIDE_SET: driver SET is the SAME direction as our SET,
    #   so silencing our SET during a driver-SET window would just leave a
    #   gap if conditions still warrant pulling cluster set down
    # iter9: anti-oscillation block (`self_res_recent`) is bypassed when
    # `power_too_high` — drive #7 forensics showed 53% of high-power frames
    # were stuck in RECOVERY because the limiter had recently pressed RES,
    # while estPowerW was already in ICE territory. Power protection beats
    # cosmetic anti-oscillation.
    suppress_set = self_res_recent and not power_too_high
    want_set = (
      (set_too_high or power_too_high)
      and not suppress_set
      and not gas_pressed
      and not brake_pressed
    )

    # iter15 v2 (Section D R1-MF-D) — post-RES quiet period for SOFT_CAP decrement.
    # After a RES emission, accel command spikes briefly → est_power read inflates →
    # softcap-driven SET fires → net cruise speed loss. Suppress softcap-driven
    # SET for POST_RES_QUIET_PERIOD_FRAMES UNLESS power genuinely far over cap
    # (est_power_control_w > POST_RES_HARD_OVERRIDE_FRAC * cap, the 1.05 frac).
    #
    # ONLY gates the softcap-driven SET path. Driver SET button echo, IDLE→SOFT_CAP
    # state-entry, manual driver-override paths, etc. unaffected.
    # Counter is edge-detected per R1-MF-D (transition from non-suppressed-last-frame
    # to suppressed-this-frame). Frame counter is separate (forensics).
    in_post_res_quiet = (frame - self._last_res_emit_frame) < POST_RES_QUIET_PERIOD_FRAMES
    power_far_over_cap = est_power_control_w > power_threshold_w * POST_RES_HARD_OVERRIDE_FRAC
    softcap_pred_now = set_too_high or power_too_high   # mirrors line ~1428 softcap_pred
    softcap_driven_set = want_set and softcap_pred_now
    decrement_suppressed_this_frame = (
      softcap_driven_set and in_post_res_quiet and not power_far_over_cap
    )
    if decrement_suppressed_this_frame:
      want_set = False
      self._evLimiter_softcap_decrement_suppressed_frames += 1
    # Edge-detect (R1-MF-D): increment only on transition non-suppressed → suppressed.
    if decrement_suppressed_this_frame and not self._softcap_decrement_suppressed_last_frame:
      self._evLimiter_softcap_decrement_suppressed_events += 1
    self._softcap_decrement_suppressed_last_frame = decrement_suppressed_this_frame
    # Informational counter: power "far over cap" overrode the quiet (would
    # have suppressed otherwise, but power is genuinely high).
    if softcap_driven_set and in_post_res_quiet and power_far_over_cap:
      self._evLimiter_post_res_hard_override_events += 1
    self._post_res_quiet_active_last = in_post_res_quiet

    # Up-trigger: gentle recovery toward target_set when below. Mutually
    # exclusive with want_set — never both same frame. Also explicitly
    # cancelled by power_too_high (defense-in-depth: even if some other
    # gate cleared want_set, RES toward user_target is wrong when motor
    # is already at ICE-territory power).
    # iter10 Layer 1: removed `not gas_pressed` from this chain. Gas behavior
    # is now governed by mode selection + bound clamping (Event A fix). The
    # MODE_GAS_CATCHUP upper-bound (vEgo + 2 mph) prevents over-recovery while
    # gas is held — RES will track vEgo upward but won't overshoot.
    standstill_clear = (frame - self._left_standstill_at_frame) >= RECOVERY_AFTER_STANDSTILL_FRAMES

    # iter11 Fix A: recovery escape — when cluster is below max-deficit floor,
    # power_too_high no longer cancels want_res. Recovery toward floor is
    # escape priority. Rate-limited via permitted-cluster check below.
    want_res = (
      under_target
      and not want_set
      and (not power_too_high or self._recovery_escape_active)
      and not in_override_set
      and not self_set_recent
      and not brake_pressed
      and v_ego > RECOVERY_V_EGO_FLOOR_MS
      and standstill_clear
    )

    # iter16a (B2, gpt-5.5): hysteretic suppression of RES while set-vs-actual delta
    # is already large. Never raise the set speed further from actual — that is the
    # lead-clear-surge fuel observed on a8..b3. Blocks UP only (SET-down protection
    # is untouched). Driver RES is unaffected (this only gates the limiter's own RES).
    delta_mph = (observed_set_speed - v_ego) / MPH_TO_MS
    if self._recovery_delta_block:
      if delta_mph <= RECOVERY_MAX_DELTA_EXIT_MPH:
        self._recovery_delta_block = False
    elif delta_mph >= RECOVERY_MAX_DELTA_ENTER_MPH:
      self._recovery_delta_block = True
    if self._recovery_delta_block and not self._recovery_escape_active:
      want_res = False

    # iter11 Fix A: rate-limited recovery — max 2 mph/sec elapsed-time-based.
    # When in escape mode, throttle RES so cluster gain never exceeds 2 mph/sec
    # since escape entry. Prevents jumpy recovery if SCC is responsive.
    if want_res and self._recovery_escape_active:
      elapsed_s = max(0.001, frame * 0.01 - self._recovery_escape_start_t)
      max_permitted_cluster = self._recovery_escape_start_cluster + 2.0 * MPH_TO_MS * elapsed_s
      if observed_set_speed >= max_permitted_cluster:
        want_res = False   # rate-limit hit; wait

    # iter10 Layer 1: gas-time RES rate cap. While gas held, throttle RES
    # cadence to one press per 1.5 s (vs 6/s normal global limit) so cluster
    # ratchets slowly behind vEgo, not in lockstep. Prevents over-recovery /
    # cluster jumping ahead.
    if want_res and gas_pressed:
      if (frame - self.last_res_frame) < GAS_RES_INTERVAL_FRAMES:
        want_res = False

    # iter13 v4: REMOVED iter12 continuous-frame SET escape.
    # The continuous-frame escape mode fired SET every CAN frame (100Hz) for
    # up to 1.5s when ≥3 ineffective SETs accumulated. On drive 17 it fired
    # 33-36 SETs in 0.5s on three separate occasions, each triggering an SCC
    # auto-cancel. iter13 replaces it with strict ACK-paced single-press SET
    # at the EVLimiter layer, plus an authoritative wire-side limiter in
    # CarController (car_controller_button_limiter.py) that enforces sliding
    # 100ms/500ms/1s windows on emitted frames.
    #
    # iter13 ack-driven cadence (advisory; CarController is authoritative):
    #   - SET_HARD_MIN_INTERVAL_FRAMES = 50 (0.5s) between desired SETs
    #   - SET_NORMAL_COOLDOWN_FRAMES_V13 = 150 (1.5s) gentle steady-state
    #   - SET_RESPONSE_TIMEOUT_FRAMES_V13 = 200 (2.0s) ack deadline
    #   - On 3 emitted SETs without ACK: increment _set_no_ack_events and
    #     enter SET_NO_ACK_BACKOFF_TIER_FRAMES adaptive backoff (1.5/2.0/3.0s).
    in_set_escape = False  # iter13 v4: continuous-frame escape removed
    effective_burst = 1    # iter13 v4: BURST_COPIES_SET = 1 (R4-MF6); CarController wire layer is authoritative

    # iter11 Fix D: ineffective-RES watchdog escape. When in escape window,
    # send continuous-frame RES (every tick) instead of discrete bursts.
    in_res_escape = frame < self._res_escape_until_frame
    if in_res_escape:
      # Hard-cap: abort escape if cluster already moved up by 3 mph
      MAX_ESCAPE_DELTA_MS = 3.0 * MPH_TO_MS
      if observed_set_speed - self._res_escape_start_cluster_ms >= MAX_ESCAPE_DELTA_MS:
        self._res_escape_until_frame = -10000
        in_res_escape = False

    # iter12 Fix F (rev): mode-gated emission. Per gpt-5.5 v1 review (I4):
    # BRAKE/DECEL/STANDSTILL_HOLD modes suppress BOTH SET and RES.
    # GAS_CATCHUP suppresses SET (driver wants UP, never SLOW), allows RES
    # (rate-capped, capped at vEgo+headroom — see iter10 GAS_CATCHUP rationale).
    # iter16a (B3, gpt-5.5): enforce the rolling window through lead-follow decel.
    # Previously MODE_DECEL fully deferred SET, so when SCC braked for a lead the
    # set speed stayed pinned ≫ actual (windup) and surged on lead-clear. Allow
    # SET-down in DECEL when set_too_high — target_set is already >= vEgo+margin,
    # so this only tracks the set speed DOWN toward the window, never below vEgo.
    # Driver BRAKE still fully defers; RES stays blocked in DECEL.
    decel_enforce_set = (governor_mode == GOVERNOR_MODE_DECEL and set_too_high and not brake_pressed)
    mode_blocks_set = (governor_mode in (GOVERNOR_MODE_BRAKE, GOVERNOR_MODE_DECEL,
                                          GOVERNOR_MODE_STANDSTILL, GOVERNOR_MODE_GAS_CATCHUP)
                       and not decel_enforce_set)
    mode_blocks_res = (governor_mode in (GOVERNOR_MODE_BRAKE, GOVERNOR_MODE_DECEL,
                                          GOVERNOR_MODE_STANDSTILL))
    if mode_blocks_set:
      want_set = False
    if mode_blocks_res:
      want_res = False

    button = Buttons.NONE

    # iter13 v4: ACK-paced single-press SET (no continuous-frame escape).
    # The CarController button limiter is authoritative on the wire side; the
    # EVLimiter layer just enforces a logical hard min interval and adaptive
    # backoff on consecutive ineffective SETs.
    #
    # ACK detection has two paths:
    #   - Internal `_cluster_at_last_set` fallback (always-on): cluster_drop
    #     ≥ 1 mph since last SET → treat as ack for cooldown reset. This
    #     keeps the limiter's cadence sensible in standalone mode and
    #     preserves backwards-compat with iter9-iter12 tests.
    #   - External ACK matcher (`_unmatched_emitted_set_frames`) populated
    #     by CarController via `note_set_emitted` after rate-limit acceptance.
    #     Used for evLimiterSetClusterDecrementAcked telemetry counter.
    if want_set:
      elapsed = frame - self.last_set_frame
      # Try external ACK matcher (telemetry side).
      physical_btn_in_window = (
        (frame - self._driver_set_last_frame) <= ACK_WINDOW_FRAMES
        or (frame - self._driver_res_last_frame) <= ACK_WINDOW_FRAMES
      )
      self.try_ack_set(observed_set_speed, frame, physical_btn_in_window)

      # Internal cluster_at_last_set ack — drives cooldown decisions.
      acked_via_cluster_drop = False
      if self.last_set_frame > 0 and self._cluster_at_last_set > 0:
        cluster_drop = self._cluster_at_last_set - observed_set_speed
        if cluster_drop >= 1.0 * MPH_TO_MS:
          acked_via_cluster_drop = True
          self._consecutive_no_ack_emits = 0   # reset; cluster responded

      # Determine cooldown.
      if self.last_set_frame <= 0:
        min_cooldown = SET_NORMAL_COOLDOWN_FRAMES_V13     # first SET, no history
      elif acked_via_cluster_drop:
        min_cooldown = SET_HARD_MIN_INTERVAL_FRAMES        # 0.5s after ack — fast progressive
      elif self._consecutive_no_ack_emits == 0:
        min_cooldown = SET_NORMAL_COOLDOWN_FRAMES_V13     # 1.5s baseline
      else:
        # Adaptive backoff after consecutive ineffective SETs.
        idx = min(self._consecutive_no_ack_emits, len(SET_NO_ACK_BACKOFF_TIER_FRAMES)) - 1
        min_cooldown = SET_NO_ACK_BACKOFF_TIER_FRAMES[idx]

      # Power-aware: reduce cooldown to 0.5s on power_too_high (real ICE risk).
      if power_too_high:
        min_cooldown = min(min_cooldown, SET_HARD_MIN_INTERVAL_FRAMES)

      # Standstill-no-ack backoff is advisory; honored at the gate below.
      backoff_active = (frame < self._prelaunch_no_ack_backoff_until
                        or frame < self._set_no_ack_backoff_until_frame)

      if (elapsed >= min_cooldown and not backoff_active
          and self._consume_global_rate_limit(frame, 1, limit=rate_limit)):
        # Decision-side: increment _set_requested. CarController is the only
        # site that increments _set_emitted/_set_dropped via its own counter
        # (sourced from the rate limiter).
        self._set_requested += 1
        if self._was_in_prelaunch:
          self._standstill_set_requested += 1

        button = Buttons.SET_DECEL
        self.last_set_frame = frame
        self._cluster_at_last_set = observed_set_speed
        # Track consecutive no-ack count (only if we did NOT just ack).
        if not acked_via_cluster_drop:
          self._consecutive_no_ack_emits += 1
        if self._consecutive_no_ack_emits >= 3:
          self._set_no_ack_events += 1
          tier_idx = min(self._consecutive_no_ack_emits - 3,
                         len(SET_NO_ACK_BACKOFF_TIER_FRAMES) - 1)
          self._set_no_ack_backoff_until_frame = (
            frame + SET_NO_ACK_BACKOFF_TIER_FRAMES[tier_idx])
        self.current_burst_count = effective_burst

    # RES-escape window
    if button == Buttons.NONE and in_res_escape:
      # Continuous-frame RES during escape (no cooldown)
      button = Buttons.RES_ACCEL
      self.current_burst_count = 1
    elif button == Buttons.NONE and want_res and (frame - self.last_res_frame) >= RES_COOLDOWN_FRAMES:
      if self._consume_global_rate_limit(frame, 1):
        button = Buttons.RES_ACCEL
        # iter11 Fix D: track RES sequence for ineffective-button detection
        if self._res_sequence_press_count == 0:
          self._res_sequence_start_frame = frame
          self._res_sequence_start_cluster_ms = observed_set_speed
        self._res_sequence_press_count += 1

    # iter15 v2 (Section D): mark the post-RES quiet window AND latch first-RES
    # for the standstill-exit latency metric (Section C state-vector telemetry).
    # Do this once after both RES emit branches resolve.
    if button == Buttons.RES_ACCEL:
      self._last_res_emit_frame = frame
      if self._standstill_exit_post_exit_active and not self._standstill_exit_first_res_seen:
        self._standstill_exit_first_res_seen = True
        # Stop the post-exit-RES-latency counter from advancing past this frame
        # (the value at this point is the recorded latency). Field stays latched
        # until next standstill exit resets it.

    # iter11 Fix D: detect ineffective RES sequence and trigger escape
    INEFFECTIVE_RES_WINDOW_FRAMES = 1000   # 10 s
    INEFFECTIVE_RES_MIN_PRESSES = 8
    INEFFECTIVE_RES_MIN_GAIN_MS = 1.0 * MPH_TO_MS
    INEFFECTIVE_RES_HOLD_FRAMES = 50       # 0.5 s continuous press
    INEFFECTIVE_RES_BACKOFF_FRAMES = 500   # 5 s between attempts
    if (self._res_sequence_press_count >= INEFFECTIVE_RES_MIN_PRESSES
        and (frame - self._res_sequence_start_frame) <= INEFFECTIVE_RES_WINDOW_FRAMES
        and (observed_set_speed - self._res_sequence_start_cluster_ms) < INEFFECTIVE_RES_MIN_GAIN_MS
        and (frame - self._last_escape_attempt_frame) > INEFFECTIVE_RES_BACKOFF_FRAMES
        and not in_res_escape):
      # Trigger ineffective-RES escape
      self._res_escape_until_frame = frame + INEFFECTIVE_RES_HOLD_FRAMES
      self._res_escape_start_cluster_ms = observed_set_speed
      self._last_escape_attempt_frame = frame
      self._ineffective_res_events += 1
      # Reset sequence
      self._res_sequence_press_count = 0
      self._res_sequence_start_frame = -10000

    # Reset RES sequence tracking on state/mode changes (handled implicitly:
    # any frame without RES press resets press_count IF we've moved out of
    # the window). Reset on override or brake or target change:
    if (in_override_set or in_override_res or brake_pressed
        or gas_pressed or self._res_sequence_press_count == 0):
      pass   # sequence already reset or invalid; do nothing

    if button != Buttons.NONE:
      self._record_tx(frame, button)

    # iter16a (Phase A): live request-indicator signals.
    #   request_dir = controller INTENT this tick (fail-closed if both true).
    #   button_dir  = actual CAN button EMITTED this tick.
    #   request_honored = did the SCC set speed move in the EMITTED direction within
    #     ACK_WINDOW, NOT attributable to a recent driver/manual button.
    if want_set and want_res:
      self._request_dir = 0   # invariant violation — should be mutually exclusive
    elif want_set:
      self._request_dir = 2
    elif want_res:
      self._request_dir = 1
    else:
      self._request_dir = 0

    if button == Buttons.SET_DECEL:
      self._button_dir = 2
    elif button == Buttons.RES_ACCEL:
      self._button_dir = 1
    else:
      self._button_dir = 0

    # Honored/ignored for the indicator. "Honored" = the SCC set speed moved in the
    # last emitted direction (not attributable to a recent driver/manual button).
    # "Ignored" = the limiter emitted but the SCC did not follow. For SET-down we
    # reuse the validated no-ack tracker (`_consecutive_no_ack_emits`, reset on a
    # cluster decrement) rather than a fixed timeout — a fixed timeout races with
    # the SET cadence and never latches in the exact "repeated SET, no movement"
    # case the driver cares about. For RES-up we fall back to a movement timeout.
    physical_btn_recent = (
      (frame - self._driver_set_last_frame) <= ACK_WINDOW_FRAMES
      or (frame - self._driver_res_last_frame) <= ACK_WINDOW_FRAMES
    )
    QUANT_MS = 0.5 * MPH_TO_MS   # ignore sub-quantization jitter
    new_dir = self._button_dir != 0 and self._button_dir != self._last_emit_dir
    # A same-direction re-emit after the previous request already resolved/expired
    # (an idle gap) should start a FRESH verdict, not inherit the stale one
    # (gpt-5.5 review caution): otherwise a long-ago "honored" persists onto a new
    # press, and repeated RES could keep deferring the "ignored" latch.
    stale_reemit = (self._button_dir != 0 and not new_dir
                    and (frame - self._last_emit_frame_for_honored) > ACK_WINDOW_FRAMES)
    if new_dir or stale_reemit:
      self._last_emit_dir = self._button_dir
      self._last_emit_frame_for_honored = frame
      self._cluster_at_emit_ms = observed_set_speed
      self._request_honored = 0
    elif self._button_dir != 0:
      # Same-direction re-emit within the window — advance the timeout reference.
      self._last_emit_frame_for_honored = frame
    if self._last_emit_dir == 2:        # DOWN (SET)
      moved_down = observed_set_speed <= self._cluster_at_emit_ms - QUANT_MS
      if moved_down and not physical_btn_recent:
        self._request_honored = 1
      elif self._consecutive_no_ack_emits >= 2:
        self._request_honored = 2       # repeated SET, SCC not following
    elif self._last_emit_dir == 1:      # UP (RES)
      moved_up = observed_set_speed >= self._cluster_at_emit_ms + QUANT_MS
      if moved_up and not physical_btn_recent:
        self._request_honored = 1
      elif (frame - self._last_emit_frame_for_honored) > ACK_WINDOW_FRAMES:
        self._request_honored = 2

    # iter10 Layer 2: dwell + hysteresis state derivation.
    # Update sustain counters every frame.
    softcap_pred = set_too_high or power_too_high
    recovery_pred = under_target and not softcap_pred
    if softcap_pred:
      self._softcap_enter_sustain += 1
      self._softcap_exit_sustain = 0
    else:
      self._softcap_enter_sustain = 0
      self._softcap_exit_sustain += 1
    if recovery_pred:
      self._recovery_enter_sustain += 1
      self._recovery_exit_sustain = 0
    else:
      self._recovery_enter_sustain = 0
      self._recovery_exit_sustain += 1

    cur_state = self.state
    time_in_state = frame - self._state_entered_frame
    new_state = cur_state  # default: hold

    # Driver-priority states bypass dwell/sustain entirely (immediate).
    # IDLE → active uses sustain only (no min-dwell delay on entry).
    # active → IDLE/active needs (min-dwell) AND (exit sustain).
    # If we reach this block coming from DISABLED/STANDSTILL_HOLD/BUS_FAULT,
    # we just left those (engage / left-standstill / bus-recovery) — treat
    # like IDLE-entry for state derivation purposes.
    if cur_state in (STATE_DISABLED, STATE_STANDSTILL_HOLD, STATE_BUS_FAULT_HOLD):
      cur_state = STATE_IDLE

    if cur_state == STATE_IDLE:
      # Power-too-high bypasses entry sustain — preserves iter9 fast-protect.
      if want_set or (frame - self.last_set_frame) < LIMITING_LATCH_FRAMES:
        new_state = STATE_SOFT_CAP_ACTIVE
      elif power_too_high:
        new_state = STATE_SOFT_CAP_ACTIVE
      elif self._softcap_enter_sustain >= SOFT_CAP_ENTER_SUSTAIN_FRAMES:
        new_state = STATE_SOFT_CAP_ACTIVE
      elif want_res or (frame - self.last_res_frame) < RECOVERING_LATCH_FRAMES:
        new_state = STATE_RECOVERY_ACTIVE
      elif self._recovery_enter_sustain >= RECOVERY_ENTER_SUSTAIN_FRAMES:
        new_state = STATE_RECOVERY_ACTIVE
      elif in_override_set:
        new_state = STATE_DRIVER_OVERRIDE_SET
      elif in_override_res:
        new_state = STATE_DRIVER_OVERRIDE_RES

    elif cur_state == STATE_SOFT_CAP_ACTIVE:
      # Driver/override states: immediate (priority).
      if in_override_set:
        new_state = STATE_DRIVER_OVERRIDE_SET
      # Min-dwell + exit-sustain required to leave SOFT_CAP.
      elif (time_in_state >= MIN_ACTIVE_STATE_DWELL_FRAMES
            and self._softcap_exit_sustain >= SOFT_CAP_EXIT_SUSTAIN_FRAMES):
        # Exit allowed: prefer RECOVERY if predicate sustained, else IDLE.
        if self._recovery_enter_sustain >= RECOVERY_ENTER_SUSTAIN_FRAMES:
          new_state = STATE_RECOVERY_ACTIVE
        else:
          new_state = STATE_IDLE

    elif cur_state == STATE_RECOVERY_ACTIVE:
      # Driver SET (immediate) or sustained SOFT_CAP predicate trumps recovery.
      if in_override_set:
        new_state = STATE_DRIVER_OVERRIDE_SET
      elif power_too_high:
        # Immediate SOFT_CAP entry on power excess (preserves fast-protect).
        new_state = STATE_SOFT_CAP_ACTIVE
      elif (time_in_state >= MIN_ACTIVE_STATE_DWELL_FRAMES
            and self._recovery_exit_sustain >= RECOVERY_EXIT_SUSTAIN_FRAMES):
        if self._softcap_enter_sustain >= SOFT_CAP_ENTER_SUSTAIN_FRAMES:
          new_state = STATE_SOFT_CAP_ACTIVE
        else:
          new_state = STATE_IDLE

    elif cur_state == STATE_DRIVER_OVERRIDE_SET:
      # Override windows are external-driven; let existing latch decide.
      if not in_override_set:
        new_state = STATE_IDLE

    elif cur_state == STATE_DRIVER_OVERRIDE_RES:
      if not in_override_res:
        new_state = STATE_IDLE

    # else (STANDSTILL_HOLD, DISABLED, BUS_FAULT_HOLD): handled before this
    # block returns from earlier branches in update().

    # ───────────────────────────────────────────────────────────────────
    # iter14 v2 — RECOVERY power-gate guard (drive 18 t=1331-1352 root cause).
    # Runs BEFORE _publish() / _arbitrate_state_transition() so guard gates
    # actuation, not just telemetry (R1-MF1).
    #
    # iter15 v2 (Section A R1-MF-A) — hard-preempt fix for BUG #1 from drives
    # A+B forensics: the guard returned the right new_state, but `_publish()`
    # internally calls `_arbitrate_state_transition()` which enforces
    # `MIN_ACTIVE_STATE_DWELL_FRAMES=200` on active-state EXIT — silently
    # demoting the guard's decision to a no-op for up to 2 s. iter15 fix:
    # the guard now returns `(new_state, guard_forced_transition)` and the
    # caller passes `hard_preempt=guard_forced_transition` so ONLY the guard's
    # forced transition bypasses min-dwell; all other state transitions still
    # respect dwell+sustain.
    #
    # `prior_published_state` (= self.state at this point) is the state at the
    # moment of the guard call; the edge-detected RECOVERY→SOFT_CAP episode
    # counter (R2-MF-1 strict) uses this with new_state.
    prior_published_state = int(self.state)
    new_state, guard_forced_transition = self._apply_recovery_power_guard(
      new_state, est_power_control_w, power_threshold_w, frame
    )

    # iter15 v2 R2-MF-1 STRICT: episode counter increments ONLY when guard
    # converted a RECOVERY-active candidate to SOFT_CAP_ACTIVE. Not on
    # RECOVERY→IDLE (lockout-block fallback). Not on non-RECOVERY priors.
    if (prior_published_state == STATE_RECOVERY_ACTIVE
        and new_state == STATE_SOFT_CAP_ACTIVE
        and guard_forced_transition):
      self._evLimiter_recovery_yield_episodes += 1
    # Per-frame guard_forced_transition published bool + cumulative counter.
    self._guard_forced_transition_last_frame = bool(guard_forced_transition)
    if guard_forced_transition:
      self._evLimiter_guard_forced_transition_events += 1

    # iter11 Fix B: arbiter handles _state_entered_frame mutation; remove
    # the manual update here.
    # iter15 v2 (Section A): pass hard_preempt=guard_forced_transition so the
    # arbiter bypasses min-dwell ONLY when the guard actually changed state
    # away from RECOVERY. Other transitions remain min-dwell-respecting.
    # iter11 Fix F: pass effective_burst (1 on highway, BURST_COPIES otherwise)
    return self._publish(button, new_state, observed_set_speed,
                         frame=frame, reason="derived",
                         hard_preempt=guard_forced_transition,
                         burst_count=effective_burst if button == Buttons.SET_DECEL else None)

  # ----- iter15 v2 Section C — narrow standstill reset + state-vector telemetry

  def _on_leaving_standstill(self, frame: int, time_in_standstill_frames: int,
                              est_power_control_w: float,
                              power_threshold_w: float) -> None:
    """iter15 v2 (Section C R1-MF-C) — narrow standstill-exit reset.

    Apply only when time-in-standstill > LONG_STANDSTILL_RESET_FRAMES (5 s).
    The reset is INTENTIONALLY NARROW per gpt-5.5 R1-MF-C / R2-MF-3:

      WHAT WE CLEAR (conditional, harmful-if-stale):
        1. `_prelaunch_no_ack_backoff_until` — if still in the future (a stale
           backoff window from a prior PRELAUNCH episode).
        2. `_softcap_entry_reason` — if est_power_control_w is now comfortably
           below cap (the reason no longer reflects current conditions).

      WHAT WE PRESERVE (R2-MF-3 PRESERVATION TESTS in test_iter15_standstill_windup):
        - `_softcap_from_recovery_lockout_until` (frame counter; expires naturally)
        - `_softcap_enter_sustain`, `_softcap_exit_sustain`
        - `_recovery_enter_sustain`, `_recovery_exit_sustain`
        - `_recovery_lockout_engaged`, `_recovery_reentry_sustain`
        - `_power_near_budget_sustain`, `_power_capped_control_sustain`
        - `_power_too_high_recent`

    Always (regardless of long/short) latch the state-vector snapshot for the
    next 1-2 drives' forensics. Short stops still publish snapshot but skip
    the reset path.
    """
    # State-vector snapshot — bounded to <=200 chars per capnp Text type.
    # Compact "/"-delimited key=value summary; readable from CarStateSP replay.
    cur_state = int(self.state)
    nb_remaining = max(0, self._prelaunch_no_ack_backoff_until - frame)
    snapshot_parts = [
      f"st={cur_state}",
      f"sec={self._softcap_enter_sustain},{self._softcap_exit_sustain}",
      f"rec={self._recovery_enter_sustain},{self._recovery_exit_sustain}",
      f"nb={nb_remaining}",
      f"lk={int(self._recovery_lockout_engaged)}",
      f"reentry={self._recovery_reentry_sustain}",
      f"pwr={est_power_control_w / 1000.0:.1f}kw",
      f"cap={power_threshold_w / 1000.0:.0f}kw",
      f"reason={self._softcap_entry_reason[:18]}",
    ]
    snapshot = "/".join(snapshot_parts)[:200]
    self._standstill_exit_state_snapshot_last = snapshot
    self._standstill_exit_time_s_last = float(time_in_standstill_frames) / float(FRAMES_PER_SEC)

    # Reset the exit-to-RES latency tracker on every exit (regardless of long/short).
    self._standstill_exit_to_first_res_latency_frames = 0
    self._standstill_exit_post_exit_active = True
    self._standstill_exit_first_res_seen = False

    # SHORT-STOP path: telemetry only; no reset of internal state.
    if time_in_standstill_frames <= LONG_STANDSTILL_RESET_FRAMES:
      return

    # LONG-STOP path: NARROW reset only (R1-MF-C). Increment the resets event
    # counter only if at least one of the two conditional clears fires.
    any_reset_fired = False

    if frame < self._prelaunch_no_ack_backoff_until:
      # Clear stale PRELAUNCH no-ack backoff window. The standstill just ended;
      # if the backoff window extended past the standstill exit, it's stale.
      self._prelaunch_no_ack_backoff_until = frame
      self._evLimiter_long_standstill_prelaunch_backoff_cleared += 1
      any_reset_fired = True

    if (est_power_control_w < power_threshold_w * STALE_SOFTCAP_REASON_POWER_FRAC
        and self._softcap_entry_reason
        and self._softcap_entry_reason != "none"):
      # Power is now comfortably below cap; clear the stale entry reason so
      # future SOFT_CAP entries report current cause.
      self._softcap_entry_reason = ""
      self._evLimiter_long_standstill_softcap_reason_cleared += 1
      any_reset_fired = True

    if any_reset_fired:
      self._evLimiter_long_standstill_resets += 1

  # ----- iter11 Fix B: centralized state arbiter --------------------------

  def _apply_recovery_power_guard(self, new_state: int, est_power_control_w: float,
                                   power_threshold_w: float, frame: int) -> tuple[int, bool]:
    """iter14 v2 — RECOVERY power-gate guard. Runs BEFORE _publish() so it
    gates actuation, not just telemetry (R1-MF1).

    Reads est_power_control_w (short-tau LP, uncapped) — NEVER est_power_w
    (HUD-smoothed, capped) per R1-MF3. Avoids drive 18 stale-reading bug.

    Two-tier yield (R1-MF4):
      Tier 1: power_control >= cap → immediate
      Tier 2: power_control >= 0.95*cap for 3 frames (30 ms) → debounced
    (Sustained-capped at 0.5 s removed in R2-MF-A; counter kept diagnostic.)

    Lockout (R2-MF-B): RECOVERY blocked until BOTH 2 s minimum elapsed AND
    power below 0.85*cap for sustained 1 s. Single conjunction governs.

    iter15 v2 (Section A R1-MF-A) — RETURNS TUPLE `(new_state, guard_forced_transition)`.
    `guard_forced_transition` is True ONLY when the guard concluded the
    current published RECOVERY state must yield AND the resulting `new_state`
    is non-RECOVERY (i.e., the transition genuinely leaves RECOVERY). Two
    sources of "yield" can both set this flag:
      (a) Guard's own RECOVERY→SOFT_CAP forcing on `power_should_yield`.
      (b) Caller's earlier state-derivation already chose SOFT_CAP/IDLE while
          published state was RECOVERY AND `power_should_yield` is True — in
          which case the arbiter's min-dwell would otherwise silently block
          the transition (iter14 BUG #1).
    Caller passes the flag to `_publish(..., hard_preempt=guard_forced_transition)`
    to bypass min-dwell on these guard-relevant transitions only.
    """
    # iter15 v2 R1-MF-A — capture prior state for change-detection.
    # `prior_candidate_state` is the state the state-derivation block chose
    # (the function's `new_state` parameter).
    # `prior_published_state` is the actually-published state (self.state) at
    # this instant — needed to detect iter14 BUG #1 where state-derivation
    # quietly chose SOFT_CAP but min-dwell silently blocked the transition.
    prior_candidate_state = new_state
    prior_published_state = int(self.state)
    guard_forced_transition = False

    self._state_candidate_before_guard_last = int(new_state)

    power_for_arbiter = est_power_control_w
    power_immediate_yield = power_for_arbiter >= power_threshold_w

    if power_for_arbiter >= power_threshold_w * RECOVERY_POWER_NEAR_BUDGET_FRAC:
      self._power_near_budget_sustain += 1
    else:
      self._power_near_budget_sustain = 0
    power_debounced_yield = self._power_near_budget_sustain >= POWER_NEAR_BUDGET_DEBOUNCE_FRAMES

    # R2-MF-A: diagnostic-only counter (sustained-capped trigger REMOVED).
    if power_for_arbiter >= power_threshold_w:
      self._power_capped_control_sustain += 1
    else:
      self._power_capped_control_sustain = 0

    power_should_yield = power_immediate_yield or power_debounced_yield

    # R2-MF-B: single conjunction lockout — both must be satisfied to allow RECOVERY.
    lockout_time_done = frame >= self._softcap_from_recovery_lockout_until

    if power_for_arbiter <= power_threshold_w * RECOVERY_REENTRY_HEADROOM_FRAC:
      self._recovery_reentry_sustain += 1
    else:
      self._recovery_reentry_sustain = 0
    headroom_done = self._recovery_reentry_sustain >= RECOVERY_REENTRY_SUSTAIN_FRAMES

    recovery_unlocked = lockout_time_done and headroom_done
    # Lockout flag clears once unlocked (one-shot release). On cold start, flag is
    # False so the elif below cannot block IDLE→RECOVERY when no yield ever fired.
    if self._recovery_lockout_engaged and recovery_unlocked:
      self._recovery_lockout_engaged = False
    self._power_guard_lockout_active_last = self._recovery_lockout_engaged
    self._power_guard_yield_reason_last = "none"

    if new_state == STATE_RECOVERY_ACTIVE:
      if power_should_yield:
        new_state = STATE_SOFT_CAP_ACTIVE
        self._softcap_from_recovery_lockout_until = frame + RECOVERY_AFTER_SOFTCAP_LOCKOUT_FRAMES
        self._recovery_lockout_engaged = True
        self._softcap_entry_reason = (
          "recovery_power_immediate" if power_immediate_yield
          else "recovery_power_near_budget_debounced"
        )
        self._power_guard_yield_reason_last = (
          "immediate" if power_immediate_yield else "debounced"
        )
        self._evLimiter_recovery_yield_events += 1
        self._recovery_reentry_sustain = 0  # reset headroom clock
      elif self._recovery_lockout_engaged and not recovery_unlocked:
        # Lockout was engaged by a prior yield AND not yet released — block RECOVERY.
        # Force SOFT_CAP if headroom still missing, else IDLE (lockout time only).
        new_state = STATE_SOFT_CAP_ACTIVE if not headroom_done else STATE_IDLE
        if not headroom_done:
          self._evLimiter_recovery_lockouts_held += 1
      # else: lockout cleared (or never engaged) — original RECOVERY stands

    # iter15 v2 R1-MF-A: guard_forced_transition = True when ANY of:
    #   (a) guard's own RECOVERY→SOFT_CAP forcing changed `new_state` away from
    #       `prior_candidate_state` (above branches).
    #   (b) iter14 BUG #1 path — caller's state-derivation already chose
    #       non-RECOVERY while published state was RECOVERY AND power_should_yield
    #       is True. Without this, the arbiter's min-dwell silently blocks the
    #       transition (1.31 s observed on drive A).
    guard_changed_state = (new_state != prior_candidate_state)
    bug1_path = (
      prior_published_state == STATE_RECOVERY_ACTIVE
      and new_state != STATE_RECOVERY_ACTIVE
      and power_should_yield
    )
    guard_forced_transition = guard_changed_state or bug1_path
    return new_state, guard_forced_transition

  def _arbitrate_state_transition(self, frame: int, requested: int, reason: str,
                                   hard_preempt: bool = False) -> int:
    """SOLE site that mutates self.state. Returns actual state after arbitration.
    Hard preempts (driver override, brake, disable, bus fault, standstill-with-
    hysteresis) always succeed. Active-state exits require min-dwell. Active-
    state entries from IDLE require enter sustain (unless hard preempt or
    power_too_high — preserves iter9 fast-protect)."""
    cur = self.state
    if cur == requested:
      return cur

    blocked_by = None
    allow = False

    if hard_preempt:
      allow = True
    elif cur in ACTIVE_STATES:
      # Min-dwell on active-state EXIT
      time_in_state = frame - self._state_entered_frame
      if time_in_state >= MIN_ACTIVE_STATE_DWELL_FRAMES:
        allow = True
      else:
        blocked_by = f"min_dwell_{MIN_ACTIVE_STATE_DWELL_FRAMES - time_in_state}f"
        self._transitions_blocked_by_dwell += 1
    else:
      # IDLE → active needs sustain (unless power_too_high recent for SOFT_CAP)
      if cur == STATE_IDLE and requested == STATE_SOFT_CAP_ACTIVE:
        if (self._softcap_enter_sustain >= SOFT_CAP_ENTER_SUSTAIN_FRAMES
            or self._power_too_high_recent):
          allow = True
        else:
          blocked_by = "softcap_enter_sustain"
          self._transitions_blocked_by_sustain += 1
      elif cur == STATE_IDLE and requested == STATE_RECOVERY_ACTIVE:
        if self._recovery_enter_sustain >= RECOVERY_ENTER_SUSTAIN_FRAMES:
          allow = True
        else:
          blocked_by = "recovery_enter_sustain"
          self._transitions_blocked_by_sustain += 1
      else:
        allow = True   # any other transition (IDLE→DISABLED, IDLE→OVR, etc) allowed

    actual = requested if allow else cur
    self._record_transition(frame, cur, requested, actual, reason, blocked_by)
    if allow:
      self._state_entered_frame = frame
      self.state = actual   # SOLE state mutation (besides __init__)
    return actual

  def _record_transition(self, frame: int, prev: int, requested: int, actual: int,
                          reason: str, blocked_by: str | None) -> None:
    """Append to bounded ring of recent transitions for forensic publish."""
    if not hasattr(self, '_transition_log'):
      self._transition_log = deque(maxlen=10)
    if prev != actual:   # only log actual transitions, not blocked ones
      self._transition_log.append(f"{frame},{prev},{actual},{reason[:8]}")

  # ----- Publish ----------------------------------------------------------

  def _publish(self, button: int, state: int, observed_set_speed: float,
               frame: int = 0, reason: str = "publish",
               hard_preempt: bool = True,
               burst_count: int | None = None) -> tuple[int, bool]:
    """Publish-and-arbitrate. iter11: state changes route through the arbiter
    even from this entry point. `hard_preempt=True` by default because most
    publish callers are early-return paths (DISABLED, STANDSTILL, BUS_FAULT)
    which legitimately preempt. The state-derivation block in update() passes
    hard_preempt=False explicitly. `burst_count=None` keeps default BURST_COPIES;
    iter11 highway SET path passes burst_count=1 to suppress multi-frame burst.

    iter13 v4: also sets `_desired_button` (R4-MF4 allowlist-checked) for
    Phase-4 CarController integration. Until Phase 4 wires this in, the
    legacy (button, active) return value remains the active emission path.
    """
    actual = self._arbitrate_state_transition(frame, state, reason, hard_preempt)
    active = actual in (STATE_SOFT_CAP_ACTIVE, STATE_RECOVERY_ACTIVE,
                        STATE_STANDSTILL_PRELAUNCH_SET)
    self.current_burst_count = burst_count if burst_count is not None else BURST_COPIES

    # iter13 v4: advisory desired_button + advisory block reason.
    desired_for_advisory: int | None = None
    if button == Buttons.SET_DECEL or button == Buttons.RES_ACCEL:
      desired_for_advisory = button
    elif button == Buttons.NONE:
      desired_for_advisory = None
    else:
      # Defensive: only None/SET/RES are allowed at the EVLimiter layer.
      desired_for_advisory = None
    self._set_desired_button(desired_for_advisory)  # raises EVLimiterError if invalid

    _SHARED_STATE["active"] = bool(active)
    _SHARED_STATE["set_speed_offset"] = max(0.0, self.user_target_speed - observed_set_speed)
    _SHARED_STATE["user_target"] = float(self.user_target_speed)
    _SHARED_STATE["state"] = int(actual)
    # iter13 v4: publish decision-side counters via shared state for
    # carstate_ext to read (matches existing user_target/state pattern).
    _SHARED_STATE["set_requested"] = int(self._set_requested)
    _SHARED_STATE["cluster_decrement_acked"] = int(self._cluster_decrement_acked)
    _SHARED_STATE["set_no_ack_events"] = int(self._set_no_ack_events)
    _SHARED_STATE["standstill_entered"] = int(self._standstill_entered)
    _SHARED_STATE["standstill_exited_by_achieved"] = int(self._standstill_exited_by_achieved)
    _SHARED_STATE["standstill_exited_by_no_ack_backoff"] = int(self._standstill_exited_by_no_ack_backoff)
    _SHARED_STATE["standstill_set_requested"] = int(self._standstill_set_requested)
    # iter16a (Phase A) — live request-indicator signals.
    _SHARED_STATE["request_dir"] = int(self._request_dir)
    _SHARED_STATE["button_dir"] = int(self._button_dir)
    _SHARED_STATE["request_honored"] = int(self._request_honored)
    # iter16a (C1) — below-vEgo power-droop sim (log-only / default-off).
    _SHARED_STATE["power_droop_would_enter"] = bool(self._power_droop_would_enter)
    _SHARED_STATE["power_droop_request_mph"] = float(self._power_droop_request_mph)
    _SHARED_STATE["power_droop_active"] = bool(self._power_droop_active)
    # iter14 v2 — RECOVERY power-gate counters + transition instrumentation.
    _SHARED_STATE["power_near_budget_sustain"] = int(self._power_near_budget_sustain)
    _SHARED_STATE["power_capped_control_sustain"] = int(self._power_capped_control_sustain)
    _SHARED_STATE["power_guard_yield_reason"] = str(self._power_guard_yield_reason_last)
    _SHARED_STATE["power_guard_lockout_active"] = bool(self._power_guard_lockout_active_last)
    _SHARED_STATE["recovery_yield_events"] = int(self._evLimiter_recovery_yield_events)
    _SHARED_STATE["recovery_lockouts_entered"] = int(self._evLimiter_recovery_lockouts_held)
    _SHARED_STATE["state_prior_transition"] = int(self._state_prior_transition_last)
    _SHARED_STATE["state_candidate_before_guard"] = int(self._state_candidate_before_guard_last)
    # iter15 v2 — hard-preempt fix (Section A), narrow standstill reset +
    # state-vector telemetry (Section C), post-RES quiet period (Section D).
    _SHARED_STATE["guard_forced_transition"] = bool(self._guard_forced_transition_last_frame)
    _SHARED_STATE["guard_forced_transition_events"] = int(self._evLimiter_guard_forced_transition_events)
    _SHARED_STATE["recovery_yield_episodes"] = int(self._evLimiter_recovery_yield_episodes)
    _SHARED_STATE["long_standstill_resets"] = int(self._evLimiter_long_standstill_resets)
    _SHARED_STATE["long_standstill_prelaunch_backoff_cleared"] = int(self._evLimiter_long_standstill_prelaunch_backoff_cleared)
    _SHARED_STATE["long_standstill_softcap_reason_cleared"] = int(self._evLimiter_long_standstill_softcap_reason_cleared)
    _SHARED_STATE["standstill_exit_state_snapshot"] = str(self._standstill_exit_state_snapshot_last)
    _SHARED_STATE["standstill_exit_time_s"] = float(self._standstill_exit_time_s_last)
    _SHARED_STATE["standstill_exit_to_first_res_latency_frames"] = int(self._standstill_exit_to_first_res_latency_frames)
    _SHARED_STATE["post_res_quiet_active"] = bool(self._post_res_quiet_active_last)
    _SHARED_STATE["softcap_decrement_suppressed_frames"] = int(self._evLimiter_softcap_decrement_suppressed_frames)
    _SHARED_STATE["softcap_decrement_suppressed_events"] = int(self._evLimiter_softcap_decrement_suppressed_events)
    _SHARED_STATE["post_res_hard_override_events"] = int(self._evLimiter_post_res_hard_override_events)
    return button, active
