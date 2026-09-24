# ruff: noqa: E701
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter13 v4 — Authoritative CarController CLU11 button rate limiter (Hyundai).

This module owns the wire-side safety invariant for synthetic CLU11 button
injection on classic-CAN Hyundai HYBRID with stock SCC long. The EVLimiter
(opendbc.sunnypilot.car.hyundai.ev_limiter) decides what button to *desire*
and is purely advisory; this rate limiter is the only place that decides
what actually goes on the bus.

Why it lives in CarController, not EVLimiter:
- iter12 fired 33-36 SETs in 0.5s on drive 17 (route 00000077--336f3cb0c9),
  triggering 3 SCC auto-cancel events (cruise dropped at t=256.67, 380.75,
  420.98). The continuous-frame escape mode in EVLimiter was the cause.
- gpt-5.5 review (round 4) flagged that any limiter living in EVLimiter is
  advisory at best — RES, CANCEL, GAP, future button paths could bypass it.
- Wire-safety must be enforced at the CLU11 emission point so no path
  bypasses it.

Design — sliding INCLUSIVE windows (R4-MF1 fix):
  SET frames /  100ms  ≤ 1
  SET frames /  500ms  ≤ 3   (NEW R4-MF1 — blocks 4-in-330ms cancel pattern)
  SET frames / 1000ms  ≤ 4
  All-button frames /  100ms ≤ 1
  All-button frames / 1000ms ≤ 6
  No multiple injected buttons same frame.

Section B (priority order MUST match Section D — R4-MF2):
  Stage 1 (unconditional):
    1. invalidButtonRequest
    2. busFailsafe
    3. clusterInvalid
    4. gearNotDrive
    5. doorOpen
    6. seatbeltUnbuckled
    7. systemUnavailable
    8. brakePressed (current OR 300ms debounce)        ← D2: physical state BEFORE latch
    9. gasPressed (current OR 300ms debounce)
    10. driverButtonConflict (any physical cruise btn in prior 200ms)
    11. sccCancelInhibit                               ← latch, after physical state
    12. cruiseDisabled
    13-19. advisory (modeForbidden / evModeAssumedFalse / paramReadFailed /
                     minIntervalNotMet / cooldownActive /
                     standstillNoAckBackoff / standstillCapReached)

  Stage 2 (sliding-window — checked only after stage-1 passes):
    20. rateLimitSameFrame
    21. rateLimitAll100ms
    22. rateLimitAll1s
    23. rateLimit100ms
    24. rateLimit500ms                                 ← R4-MF1
    25. rateLimit1s
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

from opendbc.car.hyundai.values import Buttons


# --- Authoritative wire-side constants (100Hz tick rate) ----------------------

# CarController-side (sliding INCLUSIVE windows)
SET_FRAMES_PER_100MS_MAX = 1
SET_FRAMES_PER_500MS_MAX = 3                # NEW (R4-MF1)
SET_FRAMES_PER_1SEC_MAX = 4
ALL_INJECTED_BUTTON_FRAMES_PER_100MS_MAX = 1
ALL_INJECTED_BUTTON_FRAMES_PER_1SEC_MAX = 6
ALLOW_MULTIPLE_INJECTED_BUTTONS_SAME_FRAME = False

# Burst (R4-MF6)
BURST_COPIES_SET = 1
BURST_COPIES_RES = 1

# Frame thresholds (100Hz: 10 frames = 100ms, 50 frames = 500ms, 100 frames = 1s)
WINDOW_100MS_FRAMES = 10
WINDOW_500MS_FRAMES = 50
WINDOW_1SEC_FRAMES = 100

# Circuit breaker (Section G)
SCC_CANCEL_SUSPECT_THRESHOLD = 4            # ≥4 emitted SETs in prior 1s near a cancel
INHIBIT_COOLDOWN_FRAMES = 200               # 2s after re-engagement edge

# Allowlists (R4-MF4)
EVLIMITER_ALLOWED_BUTTONS = (None, Buttons.SET_DECEL, Buttons.RES_ACCEL)
CARCONTROLLER_VALID_BUTTONS = (Buttons.SET_DECEL, Buttons.RES_ACCEL,
                                Buttons.CANCEL, Buttons.GAP_DIST)


# Block-reason name strings (must match capnp EvLimiterBlockReason ordinals)
class BlockReason:
  none = "none"
  invalidButtonRequest = "invalidButtonRequest"
  busFailsafe = "busFailsafe"
  clusterInvalid = "clusterInvalid"
  gearNotDrive = "gearNotDrive"
  doorOpen = "doorOpen"
  seatbeltUnbuckled = "seatbeltUnbuckled"
  systemUnavailable = "systemUnavailable"
  brakePressed = "brakePressed"
  gasPressed = "gasPressed"
  driverButtonConflict = "driverButtonConflict"
  sccCancelInhibit = "sccCancelInhibit"
  cruiseDisabled = "cruiseDisabled"
  modeForbidden = "modeForbidden"
  evModeAssumedFalse = "evModeAssumedFalse"
  paramReadFailed = "paramReadFailed"
  minIntervalNotMet = "minIntervalNotMet"
  cooldownActive = "cooldownActive"
  standstillNoAckBackoff = "standstillNoAckBackoff"
  standstillCapReached = "standstillCapReached"
  rateLimitSameFrame = "rateLimitSameFrame"
  rateLimitAll100ms = "rateLimitAll100ms"
  rateLimitAll1s = "rateLimitAll1s"
  rateLimit100ms = "rateLimit100ms"
  rateLimit500ms = "rateLimit500ms"
  rateLimit1s = "rateLimit1s"
  other = "other"
  # iter14 v2 — power-gated RECOVERY (drive 18 t=1331-1352)
  recoveryYieldedToSoftCap = "recoveryYieldedToSoftCap"
  powerOverBudget = "powerOverBudget"
  recoveryReentryLocked = "recoveryReentryLocked"
  # iter15 v2 — post-RES quiet period + grade clamp informational
  softcapDecrementSuppressedPostRes = "softcapDecrementSuppressedPostRes"
  gradeContributionCapped = "gradeContributionCapped"


# Maps reason string → capnp ordinal (must match cereal/custom.capnp).
BLOCK_REASON_ORDINAL = {
  BlockReason.none: 0,
  BlockReason.invalidButtonRequest: 1,
  BlockReason.busFailsafe: 2,
  BlockReason.clusterInvalid: 3,
  BlockReason.gearNotDrive: 4,
  BlockReason.doorOpen: 5,
  BlockReason.seatbeltUnbuckled: 6,
  BlockReason.systemUnavailable: 7,
  BlockReason.brakePressed: 8,
  BlockReason.gasPressed: 9,
  BlockReason.driverButtonConflict: 10,
  BlockReason.sccCancelInhibit: 11,
  BlockReason.cruiseDisabled: 12,
  BlockReason.modeForbidden: 13,
  BlockReason.evModeAssumedFalse: 14,
  BlockReason.paramReadFailed: 15,
  BlockReason.minIntervalNotMet: 16,
  BlockReason.cooldownActive: 17,
  BlockReason.standstillNoAckBackoff: 18,
  BlockReason.standstillCapReached: 19,
  BlockReason.rateLimitSameFrame: 20,
  BlockReason.rateLimitAll100ms: 21,
  BlockReason.rateLimitAll1s: 22,
  BlockReason.rateLimit100ms: 23,
  BlockReason.rateLimit500ms: 24,
  BlockReason.rateLimit1s: 25,
  BlockReason.other: 26,
  # iter14 v2
  BlockReason.recoveryYieldedToSoftCap: 27,
  BlockReason.powerOverBudget: 28,
  BlockReason.recoveryReentryLocked: 29,
  # iter15 v2
  BlockReason.softcapDecrementSuppressedPostRes: 30,
  BlockReason.gradeContributionCapped: 31,
}

# Section D priority list. Lower index = higher priority.
BLOCK_REASON_PRIORITY = [
  BlockReason.invalidButtonRequest,    # 1 (capnp @1)
  BlockReason.busFailsafe,              # 2
  BlockReason.clusterInvalid,           # 3
  BlockReason.gearNotDrive,             # 4
  BlockReason.doorOpen,                 # 5
  BlockReason.seatbeltUnbuckled,        # 6
  BlockReason.systemUnavailable,        # 7
  BlockReason.brakePressed,             # 8
  BlockReason.gasPressed,               # 9
  BlockReason.driverButtonConflict,     # 10
  BlockReason.sccCancelInhibit,         # 11
  BlockReason.cruiseDisabled,           # 12
  BlockReason.modeForbidden,            # 13 (advisory)
  BlockReason.evModeAssumedFalse,       # 14
  BlockReason.paramReadFailed,          # 15
  BlockReason.minIntervalNotMet,        # 16
  BlockReason.cooldownActive,           # 17
  BlockReason.standstillNoAckBackoff,   # 18
  BlockReason.standstillCapReached,     # 19
  BlockReason.rateLimitSameFrame,       # 20
  BlockReason.rateLimitAll100ms,        # 21
  BlockReason.rateLimitAll1s,           # 22
  BlockReason.rateLimit100ms,           # 23
  BlockReason.rateLimit500ms,           # 24 (R4-MF1)
  BlockReason.rateLimit1s,              # 25
  BlockReason.other,                    # 26
  # iter14 v2 — power-gated RECOVERY (drive 18 t=1331-1352)
  BlockReason.recoveryYieldedToSoftCap,  # 27
  BlockReason.powerOverBudget,           # 28
  BlockReason.recoveryReentryLocked,     # 29
  # iter15 v2 — post-RES quiet period + grade clamp informational
  BlockReason.softcapDecrementSuppressedPostRes,  # 30
  BlockReason.gradeContributionCapped,            # 31
]


@dataclass(frozen=True)
class ButtonEmitContext:
  """Per-frame context passed to ClusterButtonRateLimiter.maybe_emit().
  CarController constructs this from CS / EVLimiter advisory each tick.

  Required for stage-1 unconditional priority checks (Section D 1-12) plus
  advisory dispatch (Section D 13-19).
  """
  cruise_enabled: bool
  brake_pressed: bool
  gas_pressed: bool
  recent_brake_300ms: bool                   # brake released ≤ 300ms ago
  recent_gas_300ms: bool                     # gas released ≤ 300ms ago
  recent_physical_cruise_btn_200ms: bool     # physical CLU11 (driver) press ≤ 200ms ago
  bus_failsafe: bool
  cluster_invalid: bool
  gear_drive: bool                            # True iff gearShifter == drive
  door_open: bool
  seatbelt_buckled: bool
  system_unavailable: bool                   # cruiseControlAvailable false / EPS-ESP fault
  advisory_block_reason: str = BlockReason.none  # EVLimiter advisory subset @13-@19

  def __post_init__(self):
    if self.advisory_block_reason not in BLOCK_REASON_ORDINAL:
      raise ValueError(f"Unknown advisory_block_reason: {self.advisory_block_reason!r}")


def burst_copies_for(button: int) -> int:
  """Burst copies per emitted logical button. SET/RES fixed at 1 (R4-MF6)."""
  if button == Buttons.SET_DECEL: return BURST_COPIES_SET
  if button == Buttons.RES_ACCEL: return BURST_COPIES_RES
  return 1


class ClusterButtonRateLimiter:
  """Authoritative wire-safety guard for CLU11 button injection.

  Single non-destructive 1s-history purge (R4-MF1). Inclusive windows.
  Implementation order matches Section D priority (R4-MF2).
  Source of truth for evLimiterSetEmitted / evLimiterSetDropped /
  evLimiterAllBtnEmitted / evLimiterLastBlockReason published telemetry.
  """

  def __init__(self):
    # Sliding-window history (frame indices); capped at 1s by purge in maybe_emit.
    self._set_frames: collections.deque = collections.deque()
    self._all_btn_frames: collections.deque = collections.deque()

    # Circuit-breaker latch (Section G). Set by detect_suspected_cancel().
    self._fault_inhibit_active = False
    self._fault_inhibit_entry_frame: int | None = None
    self._fault_inhibit_reason = BlockReason.none

    # Authoritative counters — CarController publishes these to CarStateSP.
    self.set_emitted = 0
    self.set_dropped = 0
    self.all_btn_emitted = 0
    self.suspected_scc_cancel_events = 0
    self.last_block_reason = BlockReason.none  # only updated on non-None requests

  # ----- Public API ----------------------------------------------------------

  def maybe_emit(self, requested_button: int | None,
                 ctx: ButtonEmitContext, frame_idx: int) -> int | None:
    """Authoritative gate. Returns the emitted button code (or None if blocked).

    R4 telemetry: if requested_button is None, last_block_reason is NOT touched
    so diagnostics reflect actual blocked desired-emissions only.
    """
    if requested_button is None:
      return None

    # Stage 1 — unconditional blocks (Section D priority 1-19)
    block = self._compute_unconditional_block(requested_button, ctx)
    if block is not None:
      if requested_button == Buttons.SET_DECEL:
        self.set_dropped += 1
      self.last_block_reason = block
      return None

    # Stage 2 — single non-destructive 1s purge (R4-MF1 fix)
    threshold_1s = frame_idx - WINDOW_1SEC_FRAMES
    threshold_500ms = frame_idx - WINDOW_500MS_FRAMES
    threshold_100ms = frame_idx - WINDOW_100MS_FRAMES
    while self._set_frames and self._set_frames[0] < threshold_1s:
      self._set_frames.popleft()
    while self._all_btn_frames and self._all_btn_frames[0] < threshold_1s:
      self._all_btn_frames.popleft()

    # Priority 20 — rateLimitSameFrame (R4-MF2: order MUST match Section D)
    if not ALLOW_MULTIPLE_INJECTED_BUTTONS_SAME_FRAME:
      if self._all_btn_frames and self._all_btn_frames[-1] == frame_idx:
        if requested_button == Buttons.SET_DECEL:
          self.set_dropped += 1
        self.last_block_reason = BlockReason.rateLimitSameFrame
        return None

    # Sliding INCLUSIVE windows: f >= threshold_100ms means "in last 100ms"
    all_count_100ms = sum(1 for f in self._all_btn_frames if f >= threshold_100ms)
    all_count_1s = len(self._all_btn_frames)

    # Priority 21 — rateLimitAll100ms
    if all_count_100ms >= ALL_INJECTED_BUTTON_FRAMES_PER_100MS_MAX:
      if requested_button == Buttons.SET_DECEL:
        self.set_dropped += 1
      self.last_block_reason = BlockReason.rateLimitAll100ms
      return None

    # Priority 22 — rateLimitAll1s
    if all_count_1s >= ALL_INJECTED_BUTTON_FRAMES_PER_1SEC_MAX:
      if requested_button == Buttons.SET_DECEL:
        self.set_dropped += 1
      self.last_block_reason = BlockReason.rateLimitAll1s
      return None

    # Priorities 23-25 — SET-specific windows
    if requested_button == Buttons.SET_DECEL:
      set_count_100ms = sum(1 for f in self._set_frames if f >= threshold_100ms)
      set_count_500ms = sum(1 for f in self._set_frames if f >= threshold_500ms)
      set_count_1s = len(self._set_frames)
      # Priority 23 — rateLimit100ms
      if set_count_100ms >= SET_FRAMES_PER_100MS_MAX:
        self.set_dropped += 1
        self.last_block_reason = BlockReason.rateLimit100ms
        return None
      # Priority 24 — rateLimit500ms (R4-MF1 — blocks N, N+11, N+22, N+33 pattern)
      if set_count_500ms >= SET_FRAMES_PER_500MS_MAX:
        self.set_dropped += 1
        self.last_block_reason = BlockReason.rateLimit500ms
        return None
      # Priority 25 — rateLimit1s
      if set_count_1s >= SET_FRAMES_PER_1SEC_MAX:
        self.set_dropped += 1
        self.last_block_reason = BlockReason.rateLimit1s
        return None

    # All gates passed — emit
    if requested_button == Buttons.SET_DECEL:
      self._set_frames.append(frame_idx)
      self.set_emitted += 1
    self._all_btn_frames.append(frame_idx)
    self.all_btn_emitted += 1
    self.last_block_reason = BlockReason.none
    return requested_button

  def set_emitted_in_window(self, frame_low: int, frame_high: int) -> int:
    """Inclusive count of emitted SET frames in [frame_low, frame_high]. Used
    by Section G circuit-breaker detection."""
    return sum(1 for f in self._set_frames if frame_low <= f <= frame_high)

  def detect_and_latch_suspected_cancel(self, prev_cruise_enabled: bool,
                                         cs, frame_now: int,
                                         brake_recent_300ms: bool,
                                         gas_recent_300ms: bool,
                                         physical_cruise_btn_200ms: bool) -> bool:
    """Section G — circuit breaker. Returns True iff this frame increments
    suspected_scc_cancel_events and latches the fault inhibit.

    cs is the openpilot CarState struct; we read cruiseState.enabled,
    gearShifter, doorOpen, seatbeltUnlatched, cruiseControlAvailable.
    Driver-action exclusions (R4-MF4 broadened): brake/gas/300ms-debounce,
    any physical cruise button in prior 200ms (not just CANCEL),
    gear-not-drive, door open, seatbelt unlatched, system unavailable.
    """
    if not (prev_cruise_enabled and not bool(cs.cruiseState.enabled)):
      return False
    # Driver-action exclusions
    if bool(cs.brakePressed) or brake_recent_300ms: return False
    if bool(cs.gasPressed) or gas_recent_300ms: return False
    if physical_cruise_btn_200ms: return False
    # System-state disengagements
    try:
      from opendbc.car.structs import CarState
      drive = (cs.gearShifter == CarState.GearShifter.drive)
    except Exception:
      drive = True
    if not drive: return False
    if getattr(cs, "doorOpen", False): return False
    if getattr(cs, "seatbeltUnlatched", False): return False
    if not getattr(cs, "cruiseControlAvailable", True): return False

    # Did WE cause it? Count emitted SET frames in prior 1s.
    set_emitted_prior_1s = self.set_emitted_in_window(
      frame_now - WINDOW_1SEC_FRAMES, frame_now)
    if set_emitted_prior_1s >= SCC_CANCEL_SUSPECT_THRESHOLD:
      self.suspected_scc_cancel_events += 1
      self._fault_inhibit_active = True
      self._fault_inhibit_entry_frame = frame_now
      self._fault_inhibit_reason = BlockReason.sccCancelInhibit
      return True
    return False

  def maybe_release_fault_inhibit(self, prev_cruise_enabled: bool,
                                   cur_cruise_enabled: bool,
                                   frame_now: int) -> None:
    """Section G — exit on cruise re-engagement edge AND cooldown expired."""
    if not self._fault_inhibit_active:
      return
    if (cur_cruise_enabled and not prev_cruise_enabled
        and self._fault_inhibit_entry_frame is not None
        and (frame_now - self._fault_inhibit_entry_frame) >= INHIBIT_COOLDOWN_FRAMES):
      self._fault_inhibit_active = False
      self._fault_inhibit_entry_frame = None
      self._fault_inhibit_reason = BlockReason.none

  @property
  def fault_inhibit_active(self) -> bool:
    return self._fault_inhibit_active

  @property
  def fault_inhibit_reason(self) -> str:
    return self._fault_inhibit_reason

  # ----- Internals -----------------------------------------------------------

  def _compute_unconditional_block(self, requested_button: int,
                                    ctx: ButtonEmitContext) -> str | None:
    """Section D priorities 1-19 (D2 reorder applied: physical state @8-@10
    BEFORE sccCancelInhibit @11)."""
    if requested_button not in CARCONTROLLER_VALID_BUTTONS:
      return BlockReason.invalidButtonRequest                  # 1
    if ctx.bus_failsafe: return BlockReason.busFailsafe        # 2
    if ctx.cluster_invalid: return BlockReason.clusterInvalid  # 3
    if not ctx.gear_drive: return BlockReason.gearNotDrive      # 4
    if ctx.door_open: return BlockReason.doorOpen               # 5
    if not ctx.seatbelt_buckled: return BlockReason.seatbeltUnbuckled  # 6
    if ctx.system_unavailable: return BlockReason.systemUnavailable    # 7
    # D2 reorder: immediate physical state BEFORE latch
    if ctx.brake_pressed or ctx.recent_brake_300ms:
      return BlockReason.brakePressed                           # 8
    if ctx.gas_pressed or ctx.recent_gas_300ms:
      return BlockReason.gasPressed                             # 9
    if ctx.recent_physical_cruise_btn_200ms:
      return BlockReason.driverButtonConflict                   # 10
    # Latched fault state, after physical/system blocks
    if self._fault_inhibit_active:
      return BlockReason.sccCancelInhibit                       # 11
    if not ctx.cruise_enabled: return BlockReason.cruiseDisabled  # 12
    # Advisory dispatch (priorities 13-19)
    if ctx.advisory_block_reason != BlockReason.none:
      return ctx.advisory_block_reason
    return None
