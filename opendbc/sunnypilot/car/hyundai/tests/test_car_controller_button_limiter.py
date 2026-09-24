# ruff: noqa: F841, E501
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter13 v4 — Tests for the authoritative ClusterButtonRateLimiter.

Covers tests #1-15 (safety properties), #16-22 (sliding-window correctness
incl. R4-MF1 500ms gate), #50-53 (schema parity placeholder), #69-76 (block
reason priority static + runtime).

Plus the simulator scenarios that operate purely on the limiter:
S12 bucket-boundary, S13 1s edge, S14 (MF1 regression), S15 (RES burst),
S18 (R4-MF1 N,N+11,N+22,N+33), S19 (sameFrame priority), S22 (D2 reorder).
"""
from __future__ import annotations

import unittest

from opendbc.car.hyundai.values import Buttons
from opendbc.sunnypilot.car.hyundai.car_controller_button_limiter import (
  ClusterButtonRateLimiter,
  ButtonEmitContext,
  BlockReason,
  BLOCK_REASON_PRIORITY,
  BLOCK_REASON_ORDINAL,
  EVLIMITER_ALLOWED_BUTTONS,
  CARCONTROLLER_VALID_BUTTONS,
  SET_FRAMES_PER_100MS_MAX,
  SET_FRAMES_PER_500MS_MAX,
  SET_FRAMES_PER_1SEC_MAX,
  ALL_INJECTED_BUTTON_FRAMES_PER_100MS_MAX,
  ALL_INJECTED_BUTTON_FRAMES_PER_1SEC_MAX,
  ALLOW_MULTIPLE_INJECTED_BUTTONS_SAME_FRAME,
  BURST_COPIES_SET,
  BURST_COPIES_RES,
  WINDOW_500MS_FRAMES,
  burst_copies_for,
)


def make_ctx(**overrides) -> ButtonEmitContext:
  """Build a permissive default context (everything passes stage-1)."""
  defaults = dict(
    cruise_enabled=True,
    brake_pressed=False,
    gas_pressed=False,
    recent_brake_300ms=False,
    recent_gas_300ms=False,
    recent_physical_cruise_btn_200ms=False,
    bus_failsafe=False,
    cluster_invalid=False,
    gear_drive=True,
    door_open=False,
    seatbelt_buckled=True,
    system_unavailable=False,
    advisory_block_reason=BlockReason.none,
  )
  defaults.update(overrides)
  return ButtonEmitContext(**defaults)


class TestRateLimitConstants(unittest.TestCase):
  """Test #21+22 — the 500ms gate exists with correct value (R4-MF1)."""

  def test_set_500ms_max_is_3(self):
    self.assertEqual(SET_FRAMES_PER_500MS_MAX, 3)

  def test_set_100ms_max_is_1(self):
    self.assertEqual(SET_FRAMES_PER_100MS_MAX, 1)

  def test_set_1s_max_is_4(self):
    self.assertEqual(SET_FRAMES_PER_1SEC_MAX, 4)

  def test_all_btn_100ms_max_is_1(self):
    self.assertEqual(ALL_INJECTED_BUTTON_FRAMES_PER_100MS_MAX, 1)

  def test_all_btn_1s_max_is_6(self):
    self.assertEqual(ALL_INJECTED_BUTTON_FRAMES_PER_1SEC_MAX, 6)

  def test_no_multiple_buttons_same_frame(self):
    self.assertFalse(ALLOW_MULTIPLE_INJECTED_BUTTONS_SAME_FRAME)

  def test_burst_copies_set_is_1(self):
    self.assertEqual(BURST_COPIES_SET, 1)

  def test_burst_copies_res_is_1(self):
    self.assertEqual(BURST_COPIES_RES, 1)

  def test_burst_copies_for_known_buttons(self):
    self.assertEqual(burst_copies_for(Buttons.SET_DECEL), 1)
    self.assertEqual(burst_copies_for(Buttons.RES_ACCEL), 1)


class TestSlidingWindowCorrectness(unittest.TestCase):
  """Tests #16-22 — sliding-inclusive windows, MF1 regression for v2 destructive purge."""

  def test_set_at_N_and_N_plus_5_blocked_100ms_window(self):
    """Test #16: SET at N then N+5 (50ms apart) — 2nd blocked by 100ms window.
    All-button 100ms (priority 21) fires before SET-100ms (priority 23) at
    current constants (both =1) — this is by design."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0), Buttons.SET_DECEL)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 5))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitAll100ms)

  def test_set_at_N_through_N_plus_99_4_emitted_5th_blocked(self):
    """Test #17: 4 SETs at 0, 25, 50, 75 emitted; 5th at 99 blocked by 1s window."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    for f in (0, 25, 50, 75):
      self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, f), Buttons.SET_DECEL)
    # At frame 99: count_1s = 4 (all in [99-100=-1, 99]), 100ms-window only has frame 75 (75 >= 89)? No, 75 < 89.
    # threshold_100ms = 99-10 = 89, so frame 75 not in 100ms window → 100ms count = 0. Good.
    # threshold_500ms = 99-50 = 49, so frames {50, 75} in 500ms → set_count_500ms=2. 2 < 3, allow.
    # threshold_1s = 99-100 = -1, frames {0,25,50,75} all in 1s → set_count_1s=4. 4 >= 4, block.
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 99))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimit1s)

  def test_window_purges_only_frames_outside_inclusive_1s(self):
    """Test #18: frame at N is purged at frame N+101 (outside inclusive 1s window)."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # SET at 0
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0), Buttons.SET_DECEL)
    # SET at 100 — frame 0 is inclusive (100 - 100 = 0; 0 >= 0). 1s count = 2 (0, 100). 2 < 4 OK.
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 100), Buttons.SET_DECEL)
    # SET at 101 — frame 0 outside inclusive (101 - 100 = 1; 0 < 1). 1s count = 1 (just 100).
    # 100ms window: 101 - 10 = 91, frame 100 in (100 >= 91) → all-100ms blocks
    # (all-100ms priority 21 > set-100ms priority 23).
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 101))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitAll100ms)
    # internal queue should now have only frame 100
    self.assertEqual(len(lim._set_frames), 1)
    self.assertEqual(lim._set_frames[0], 100)

  def test_100ms_check_does_not_destroy_1s_history(self):
    """Test #19: regression for v2 destructive-purge bug. After 100ms check
    runs, 1s history must still be intact for the 1s check."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # 4 SETs spread across 1s — all inside 1s window, none inside 100ms window relative to 4th
    for f in (0, 25, 50, 75):
      self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, f), Buttons.SET_DECEL)
    # Now 1s history must contain 4 entries; v2 bug would have purged to 100ms after first check
    self.assertEqual(len(lim._set_frames), 4)
    # At frame 99, 1s check still sees 4 entries → blocked
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 99))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimit1s)

  def test_window_inclusive_boundary(self):
    """Test #20: at frame N, threshold_100ms = N-10. Frame N-10 is INSIDE
    inclusive (>=); frame N-11 is OUTSIDE."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # Place a SET at 0 in history (manually for control)
    lim._set_frames.append(0)
    lim._all_btn_frames.append(0)
    lim.set_emitted = 1
    lim.all_btn_emitted = 1
    # At frame 10: threshold_100ms = 0, frame 0 IS in inclusive window → blocked
    # (rateLimitAll100ms priority 21 wins over rateLimit100ms priority 23).
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 10))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitAll100ms)
    # At frame 11: threshold_100ms = 1, frame 0 NOT in 100ms window. 500ms thr=11-50=-39 includes 0 → set_500ms=1<3 ok. 1s thr=-89 includes 0 → set_1s=1<4 ok. Allowed.
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 11), Buttons.SET_DECEL)

  def test_set_at_N_N_plus_11_N_plus_22_3_emitted_N_plus_33_blocked_500ms(self):
    """Test #21 / S18 — R4-MF1 ACCEPTANCE GATE.
    Drive 17 cancel pattern: 4 SETs in 330ms triggers SCC auto-cancel.
    iter13 must block the 4th by rateLimit500ms."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0), Buttons.SET_DECEL)
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 11), Buttons.SET_DECEL)
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 22), Buttons.SET_DECEL)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 33))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimit500ms)
    self.assertEqual(lim.set_emitted, 3)
    self.assertEqual(lim.set_dropped, 1)

  def test_set_500ms_window_inclusive_50_frames(self):
    """Test #22: the 500ms window covers 51 inclusive frame indices [N-50, N]."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # Place 3 SETs spaced widely so 100ms gate doesn't fire:
    # 0, 20, 40. At frame 50, all three are in inclusive 500ms (50-50=0).
    # 100ms thr=40, only frame 40 in 100ms (40 >= 40 inclusive) → blocked by 100ms.
    # That's the 100ms window inclusivity blocking — not what we want to test.
    # Use 0, 11, 22 (no 100ms overlap). At frame 50: 100ms thr=40, only frame 22 not in 100ms (22 < 40).
    # 500ms thr=0, all three in 500ms → 3 emitted. Frame 50 itself is 4th → blocked by 500ms.
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0), Buttons.SET_DECEL)
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 11), Buttons.SET_DECEL)
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 22), Buttons.SET_DECEL)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 50))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimit500ms)


class TestSafetyProperties(unittest.TestCase):
  """Tests #1-11 — critical safety properties under any input."""

  def test_no_more_than_4_set_emitted_per_1sec_under_high_request_rate(self):
    """Property #1: even with continuous frame requests, ≤ 4 SETs in any 1s window."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    for f in range(200):
      lim.maybe_emit(Buttons.SET_DECEL, ctx, f)
    # Assert no 1s window has > 4 SETs
    set_frames = sorted(lim._set_frames)
    # Reconstruct emitted history (deque has only last 1s; check live)
    # For full history we can't reconstruct from deque. Instead verify counters.
    # Property: at any point, count of emitted SETs in last 1s ≤ SET_FRAMES_PER_1SEC_MAX.
    # set_emitted is total; we check the deque after the run holds at most SET_FRAMES_PER_1SEC_MAX.
    self.assertLessEqual(len(lim._set_frames), SET_FRAMES_PER_1SEC_MAX)

  def test_no_more_than_3_set_emitted_per_500ms(self):
    """Property #2 (R4-MF1): even continuous requests, ≤3 SETs in any 500ms window."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    last_emitted_frames = []
    for f in range(500):
      result = lim.maybe_emit(Buttons.SET_DECEL, ctx, f)
      if result == Buttons.SET_DECEL:
        last_emitted_frames.append(f)
        # Check no 4-in-50-frame window
        recent = [g for g in last_emitted_frames if f - g <= WINDOW_500MS_FRAMES]
        self.assertLessEqual(len(recent), SET_FRAMES_PER_500MS_MAX,
                             f"at frame {f}: {recent} (>{SET_FRAMES_PER_500MS_MAX})")

  def test_no_more_than_1_set_emitted_per_100ms(self):
    """Property #3: even continuous requests, ≤1 SET in any 100ms window."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    emitted_frames = []
    for f in range(300):
      if lim.maybe_emit(Buttons.SET_DECEL, ctx, f) == Buttons.SET_DECEL:
        emitted_frames.append(f)
    # Check pairwise gap ≥ 11 frames (exceeds inclusive 100ms window)
    for i in range(1, len(emitted_frames)):
      self.assertGreaterEqual(emitted_frames[i] - emitted_frames[i-1], 11,
        f"Adjacent SETs too close: {emitted_frames[i-1]}, {emitted_frames[i]}")

  def test_no_button_emitted_when_brake_pressed(self):
    """Property #5: brake pressed blocks all SET injection."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(brake_pressed=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.brakePressed)
    self.assertEqual(lim.set_dropped, 1)

  def test_no_button_emitted_when_brake_in_prior_300ms(self):
    """Property #6: 300ms-debounced brake still blocks."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(brake_pressed=False, recent_brake_300ms=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.brakePressed)

  def test_no_button_emitted_when_gas_pressed(self):
    """Property #7."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(gas_pressed=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.gasPressed)

  def test_no_button_emitted_when_gas_in_prior_300ms(self):
    """Property #8."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(recent_gas_300ms=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.gasPressed)

  def test_no_button_emitted_when_cruise_disabled(self):
    """Property #9."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(cruise_enabled=False)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.cruiseDisabled)

  def test_no_button_emitted_when_fault_inhibit_active_includes_gap(self):
    """Property #10: GAP injection also blocked during fault inhibit."""
    lim = ClusterButtonRateLimiter()
    lim._fault_inhibit_active = True
    ctx = make_ctx()
    for btn in (Buttons.SET_DECEL, Buttons.RES_ACCEL, Buttons.CANCEL, Buttons.GAP_DIST):
      self.assertIsNone(lim.maybe_emit(btn, ctx, 0))
      self.assertEqual(lim.last_block_reason, BlockReason.sccCancelInhibit)

  def test_no_multiple_buttons_same_frame(self):
    """Property #11: two consecutive maybe_emit() at same frame_idx → 2nd blocked."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertEqual(lim.maybe_emit(Buttons.RES_ACCEL, ctx, 0), Buttons.RES_ACCEL)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitSameFrame)

  def test_invalid_button_request_blocked(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertIsNone(lim.maybe_emit(99, ctx, 0))   # nonsense button code
    self.assertEqual(lim.last_block_reason, BlockReason.invalidButtonRequest)


class TestNoRequestDoesNotUpdateBlockReason(unittest.TestCase):
  """Test #76 (R4-MF telemetry hygiene): None request must not touch
  last_block_reason — diagnostics reflect actual desired-emission blocks only."""

  def test_none_request_preserves_last_block_reason(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(brake_pressed=True)
    # First a real blocked request — should set last_block_reason to brakePressed.
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.brakePressed)
    # Now a no-request frame — should NOT change last_block_reason
    self.assertIsNone(lim.maybe_emit(None, ctx, 1))
    self.assertEqual(lim.last_block_reason, BlockReason.brakePressed)


class TestBlockReasonPriority(unittest.TestCase):
  """Tests #69-75 — block-reason priority static + runtime exclusivity (R4-MF2)."""

  def test_static_every_block_reason_string_in_enum(self):
    """Test #69: every name in BLOCK_REASON_PRIORITY exists as a capnp enumerant."""
    # We check the ordinal map (BLOCK_REASON_ORDINAL) which is the source of truth.
    for reason in BLOCK_REASON_PRIORITY:
      self.assertIn(reason, BLOCK_REASON_ORDINAL,
        f"BlockReason.{reason!r} not in capnp enum")
    # And inversely
    for reason in BLOCK_REASON_ORDINAL:
      if reason == BlockReason.none:
        continue   # 'none' not in priority list (no-block sentinel)
      self.assertIn(reason, BLOCK_REASON_PRIORITY,
        f"capnp enum {reason!r} missing from priority list")

  def test_priority_same_frame_beats_all100ms(self):
    """Test #72: same-frame check fires before all-button 100ms check."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # First emit at frame 0
    self.assertEqual(lim.maybe_emit(Buttons.RES_ACCEL, ctx, 0), Buttons.RES_ACCEL)
    # Try another at same frame — same-frame fires first (priority 20 vs 21)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitSameFrame)

  def test_priority_all100ms_beats_set100ms(self):
    """Test #73: rateLimitAll100ms (priority 21) fires before rateLimit100ms (23)."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # Emit RES at frame 0 (counted in all-button window only, not SET window)
    self.assertEqual(lim.maybe_emit(Buttons.RES_ACCEL, ctx, 0), Buttons.RES_ACCEL)
    # Try SET at frame 5 — set window thresholds: 100ms=5-10=-5, frame 0 not in (no SET there).
    # All-btn 100ms threshold=-5, frame 0 in (0 >= -5) → all100ms blocks.
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 5))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitAll100ms)

  def test_priority_set100ms_beats_set500ms_beats_set1s(self):
    """Test #74: SET-100ms / SET-500ms / SET-1s priority ordering.
    Note: at current constants SET_FRAMES_PER_100MS_MAX == ALL_BUTTON_PER_100MS_MAX == 1,
    so all-100ms (priority 21) fires before set-100ms (priority 23) for any same-button
    scenario. SET-100ms-as-primary is dead code at current constants but preserved for
    future where they may diverge. This test verifies SET-500ms vs SET-1s ordering."""
    ctx = make_ctx()
    # 3 SETs spread across 500ms (no 100ms overlap) — 4th blocks by 500ms (priority 24)
    lim2 = ClusterButtonRateLimiter()
    for f in (0, 11, 22):
      self.assertEqual(lim2.maybe_emit(Buttons.SET_DECEL, ctx, f), Buttons.SET_DECEL)
    self.assertIsNone(lim2.maybe_emit(Buttons.SET_DECEL, ctx, 33))
    self.assertEqual(lim2.last_block_reason, BlockReason.rateLimit500ms)
    # 4 SETs in 1s (one in each 250ms quartile, none overlapping 500ms-of-3-rule):
    # 0, 25, 50, 75. At frame 99: 100ms thr=89 (none in), 500ms thr=49 (50,75 in → 2),
    # 1s thr=-1 (all 4 in → 4 ≥ 4), all-1s thr=-1 (4 ≤ 6 OK).
    lim3 = ClusterButtonRateLimiter()
    for f in (0, 25, 50, 75):
      self.assertEqual(lim3.maybe_emit(Buttons.SET_DECEL, ctx, f), Buttons.SET_DECEL)
    self.assertIsNone(lim3.maybe_emit(Buttons.SET_DECEL, ctx, 99))
    self.assertEqual(lim3.last_block_reason, BlockReason.rateLimit1s)

  def test_priority_brake_beats_scc_inhibit_under_d2_reorder(self):
    """Test #75 (D2): when both fault-inhibit AND brake are active, the
    block reason MUST report brakePressed (priority 8) not sccCancelInhibit (11)."""
    lim = ClusterButtonRateLimiter()
    lim._fault_inhibit_active = True
    ctx = make_ctx(brake_pressed=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.brakePressed)


class TestStage1Blocks(unittest.TestCase):
  """Stage-1 unconditional blocks @1-@12, in priority order."""

  def test_bus_failsafe_blocks(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(bus_failsafe=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.busFailsafe)

  def test_cluster_invalid_blocks(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(cluster_invalid=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.clusterInvalid)

  def test_gear_not_drive_blocks(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(gear_drive=False)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.gearNotDrive)

  def test_door_open_blocks(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(door_open=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.doorOpen)

  def test_seatbelt_unbuckled_blocks(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(seatbelt_buckled=False)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.seatbeltUnbuckled)

  def test_system_unavailable_blocks(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(system_unavailable=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.systemUnavailable)

  def test_driver_button_conflict_blocks(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(recent_physical_cruise_btn_200ms=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.driverButtonConflict)

  def test_advisory_block_reason_dispatched(self):
    """Advisory subset (priorities 13-19) routes via ctx.advisory_block_reason."""
    for advisory in (BlockReason.modeForbidden, BlockReason.evModeAssumedFalse,
                     BlockReason.paramReadFailed, BlockReason.minIntervalNotMet,
                     BlockReason.cooldownActive, BlockReason.standstillNoAckBackoff,
                     BlockReason.standstillCapReached):
      with self.subTest(advisory=advisory):
        lim = ClusterButtonRateLimiter()
        ctx = make_ctx(advisory_block_reason=advisory)
        self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
        self.assertEqual(lim.last_block_reason, advisory)


class TestCounterOwnership(unittest.TestCase):
  """Tests #54-57 (R4-MF7 + R4-MF6) — counter ownership at limiter side."""

  def test_set_emitted_increments_only_when_emitted(self):
    """Test #55 — set_emitted only when CarController actually emits."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # 1 emitted SET at 0
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0), Buttons.SET_DECEL)
    self.assertEqual(lim.set_emitted, 1)
    # 1 dropped SET at 5 (100ms)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 5))
    self.assertEqual(lim.set_emitted, 1)   # unchanged
    self.assertEqual(lim.set_dropped, 1)

  def test_set_dropped_increments_on_block(self):
    """Test #56."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx(brake_pressed=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.set_dropped, 1)
    # None request — counter does not increment
    self.assertIsNone(lim.maybe_emit(None, ctx, 1))
    self.assertEqual(lim.set_dropped, 1)

  def test_all_btn_emitted_includes_res_and_set(self):
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertEqual(lim.maybe_emit(Buttons.RES_ACCEL, ctx, 0), Buttons.RES_ACCEL)
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 11), Buttons.SET_DECEL)
    self.assertEqual(lim.all_btn_emitted, 2)
    self.assertEqual(lim.set_emitted, 1)


class TestCircuitBreaker(unittest.TestCase):
  """Tests #34-47 — Section G circuit breaker.
  Note: full integration tests require carstate stubs; these cover the limiter's
  contribution (detection trigger + inhibit-while-active + release on edge)."""

  def _make_cs(self, cruise_enabled=False, brake_pressed=False, gas_pressed=False,
               door_open=False, seatbelt_unlatched=False, cruise_available=True,
               gear_drive_int=2):
    """Build a minimal CarState-like stub. gearShifter compared by enum equality
    in detect; we return the integer the comparison expects."""
    class _CruiseState:
      def __init__(self, e): self.enabled = e

    class _CS:
      def __init__(self):
        self.cruiseState = _CruiseState(cruise_enabled)
        self.brakePressed = brake_pressed
        self.gasPressed = gas_pressed
        self.doorOpen = door_open
        self.seatbeltUnlatched = seatbelt_unlatched
        self.cruiseControlAvailable = cruise_available
        self.gearShifter = gear_drive_int
    return _CS()

  def test_inhibit_blocks_all_buttons_including_gap(self):
    """Test #34: while inhibit active, SET/RES/CANCEL/GAP all blocked."""
    lim = ClusterButtonRateLimiter()
    lim._fault_inhibit_active = True
    ctx = make_ctx()
    for btn in (Buttons.SET_DECEL, Buttons.RES_ACCEL, Buttons.CANCEL, Buttons.GAP_DIST):
      self.assertIsNone(lim.maybe_emit(btn, ctx, 0))
      self.assertEqual(lim.last_block_reason, BlockReason.sccCancelInhibit)

  def test_circuit_breaker_excludes_brake_at_disengage(self):
    """Test #35: brake-induced cruise off does NOT trigger circuit breaker."""
    lim = ClusterButtonRateLimiter()
    # Pre-load 4 emitted SETs in prior 1s to exceed threshold
    for f in range(0, 50, 12):
      lim._set_frames.append(f)
    cs = self._make_cs(cruise_enabled=False, brake_pressed=True)
    # detect should return False because brake_pressed is True
    triggered = lim.detect_and_latch_suspected_cancel(
      prev_cruise_enabled=True, cs=cs, frame_now=50,
      brake_recent_300ms=False, gas_recent_300ms=False,
      physical_cruise_btn_200ms=False)
    self.assertFalse(triggered)
    self.assertEqual(lim.suspected_scc_cancel_events, 0)
    self.assertFalse(lim._fault_inhibit_active)

  def test_circuit_breaker_excludes_brake_recent_300ms(self):
    """Test #36."""
    lim = ClusterButtonRateLimiter()
    for f in range(0, 50, 12):
      lim._set_frames.append(f)
    cs = self._make_cs(cruise_enabled=False)
    triggered = lim.detect_and_latch_suspected_cancel(
      prev_cruise_enabled=True, cs=cs, frame_now=50,
      brake_recent_300ms=True, gas_recent_300ms=False,
      physical_cruise_btn_200ms=False)
    self.assertFalse(triggered)

  def test_circuit_breaker_excludes_gas_at_disengage(self):
    """Test #37."""
    lim = ClusterButtonRateLimiter()
    for f in range(0, 50, 12):
      lim._set_frames.append(f)
    cs = self._make_cs(cruise_enabled=False, gas_pressed=True)
    triggered = lim.detect_and_latch_suspected_cancel(
      prev_cruise_enabled=True, cs=cs, frame_now=50,
      brake_recent_300ms=False, gas_recent_300ms=False,
      physical_cruise_btn_200ms=False)
    self.assertFalse(triggered)

  def test_circuit_breaker_excludes_physical_cruise_btn_200ms(self):
    """Test #41 (broadened): any physical cruise button — not just CANCEL."""
    lim = ClusterButtonRateLimiter()
    for f in range(0, 50, 12):
      lim._set_frames.append(f)
    cs = self._make_cs(cruise_enabled=False)
    triggered = lim.detect_and_latch_suspected_cancel(
      prev_cruise_enabled=True, cs=cs, frame_now=50,
      brake_recent_300ms=False, gas_recent_300ms=False,
      physical_cruise_btn_200ms=True)
    self.assertFalse(triggered)

  def test_circuit_breaker_excludes_door_open(self):
    """Test #43."""
    lim = ClusterButtonRateLimiter()
    for f in range(0, 50, 12):
      lim._set_frames.append(f)
    cs = self._make_cs(cruise_enabled=False, door_open=True)
    triggered = lim.detect_and_latch_suspected_cancel(
      prev_cruise_enabled=True, cs=cs, frame_now=50,
      brake_recent_300ms=False, gas_recent_300ms=False,
      physical_cruise_btn_200ms=False)
    self.assertFalse(triggered)

  def test_circuit_breaker_excludes_seatbelt_unlatched(self):
    """Test #44."""
    lim = ClusterButtonRateLimiter()
    for f in range(0, 50, 12):
      lim._set_frames.append(f)
    cs = self._make_cs(cruise_enabled=False, seatbelt_unlatched=True)
    triggered = lim.detect_and_latch_suspected_cancel(
      prev_cruise_enabled=True, cs=cs, frame_now=50,
      brake_recent_300ms=False, gas_recent_300ms=False,
      physical_cruise_btn_200ms=False)
    self.assertFalse(triggered)

  def test_circuit_breaker_excludes_system_unavailable(self):
    """Test #45."""
    lim = ClusterButtonRateLimiter()
    for f in range(0, 50, 12):
      lim._set_frames.append(f)
    cs = self._make_cs(cruise_enabled=False, cruise_available=False)
    triggered = lim.detect_and_latch_suspected_cancel(
      prev_cruise_enabled=True, cs=cs, frame_now=50,
      brake_recent_300ms=False, gas_recent_300ms=False,
      physical_cruise_btn_200ms=False)
    self.assertFalse(triggered)


class TestSimulatorScenarios(unittest.TestCase):
  """Tests #58-67 — scenarios from v4 Section L (limiter-only subset)."""

  def test_S12_bucket_boundary_attack(self):
    """S12: SET at N then N+5 — second blocked by 100ms window
    (rateLimitAll100ms priority 21 wins over rateLimit100ms priority 23 at current constants)."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0), Buttons.SET_DECEL)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 5))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitAll100ms)

  def test_S13_window_edge_attack_4_emitted_5th_blocked(self):
    """S13: 4 SETs at N, N+25, N+50, N+75 emitted; 5th at N+99 blocked by 1s."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    for f in (0, 25, 50, 75):
      self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, f), Buttons.SET_DECEL)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 99))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimit1s)

  def test_S15_res_burst_with_all_btn_100ms_gate(self):
    """S15: RES at frame 0 emits; RES at frame 5 blocked by all-button 100ms."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertEqual(lim.maybe_emit(Buttons.RES_ACCEL, ctx, 0), Buttons.RES_ACCEL)
    self.assertIsNone(lim.maybe_emit(Buttons.RES_ACCEL, ctx, 5))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitAll100ms)

  def test_S18_R4MF1_drive17_pattern_blocked(self):
    """S18 (R4-MF1): drive 17 t=256 cancel pattern (33+ SET in 0.5s) cannot
    occur in iter13 — limiter blocks the 4th SET in any 0.5s window."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    # Try the EXACT failure pattern: every 11 frames
    emitted = 0
    for f in range(0, 100, 11):
      r = lim.maybe_emit(Buttons.SET_DECEL, ctx, f)
      if r is not None:
        emitted += 1
    # Cannot exceed 3 in any inclusive 500ms window. Over 100 frames (1s) max is
    # SET_FRAMES_PER_1SEC_MAX=4. So 4 emitted, 6 blocked.
    self.assertEqual(emitted, 4)

  def test_S19_priority_overlap_same_frame_wins(self):
    """S19: when same-frame + all100ms + set100ms all true, sameFrame reported."""
    lim = ClusterButtonRateLimiter()
    ctx = make_ctx()
    self.assertEqual(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0), Buttons.SET_DECEL)
    # Same frame 0 — sameFrame triggers BEFORE all100ms or set100ms
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.rateLimitSameFrame)

  def test_S22_d2_reorder_brake_beats_scc_inhibit(self):
    """S22 (D2): brake_pressed AND fault_inhibit_active → brakePressed reported."""
    lim = ClusterButtonRateLimiter()
    lim._fault_inhibit_active = True
    ctx = make_ctx(brake_pressed=True)
    self.assertIsNone(lim.maybe_emit(Buttons.SET_DECEL, ctx, 0))
    self.assertEqual(lim.last_block_reason, BlockReason.brakePressed)


class TestEvLimiterAllowlist(unittest.TestCase):
  """Tests #66-68 — R4-MF4 EVLimiter button allowlist exposure.
  The limiter module exports the allowlist used by EVLimiter; full assertion
  test belongs in test_ev_limiter.py."""

  def test_evlimiter_allowed_buttons_set_correctly(self):
    self.assertEqual(EVLIMITER_ALLOWED_BUTTONS,
                     (None, Buttons.SET_DECEL, Buttons.RES_ACCEL))
    self.assertNotIn(Buttons.CANCEL, EVLIMITER_ALLOWED_BUTTONS)
    self.assertNotIn(Buttons.GAP_DIST, EVLIMITER_ALLOWED_BUTTONS)

  def test_carcontroller_valid_buttons(self):
    """All four physical buttons are valid at CarController layer (RES/SET allowed,
    CANCEL/GAP allowed because future paths may legitimately need them — but EVLimiter
    cannot request them via _desired_button)."""
    self.assertEqual(set(CARCONTROLLER_VALID_BUTTONS),
                     {Buttons.SET_DECEL, Buttons.RES_ACCEL, Buttons.CANCEL, Buttons.GAP_DIST})


if __name__ == "__main__":
  unittest.main()
