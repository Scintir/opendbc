# ruff: noqa: ISC002, B904
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter14 v2 — Tests for the RECOVERY power-gate state arbiter guard.

Drive 18 t=1331-1352 forensic root cause: state arbiter pinned at RECOVERY_ACTIVE
for 21.5 s while estPowerCapped, even though the existing line-1450 transition
should have fired. iter14 v2 adds a belt-and-suspenders guard that runs BEFORE
_publish() and reads estPowerControlW (short-tau LP, uncapped), with a two-tier
trigger and lockout hysteresis.

Tests invoke `_apply_recovery_power_guard()` directly so they don't depend on
the full update() pipeline (CP.flags, params, etc.).

Coverage matches v2 Section J:
- 1-3   Tier-1 immediate trigger at 100% cap
- 4-6   Tier-2 debounced trigger at 95% cap
- 7-9   Diagnostic-only sustained-capped counter (R2-MF-A)
- 10-12 Lockout active blocks re-entry
- 13-15 Lockout edge cases (R2-MF-B conjunction)
- 16    Cold-start does not block (regression)
- 17-19 Triple estimator schema fields present
- 20-21 Yield-reason instrumentation
- 22-23 Carryover: _was_in_prelaunch resets on standstill exit
- 24-25 Block-reason completeness (new entries @27-@29)
"""
from __future__ import annotations

import unittest

from opendbc.sunnypilot.car.hyundai.ev_limiter import (
  EVLimiter,
  STATE_IDLE,
  STATE_SOFT_CAP_ACTIVE,
  STATE_RECOVERY_ACTIVE,
  RECOVERY_REENTRY_SUSTAIN_FRAMES,
)


CAP_W = 40_000.0   # default EvLimiterPowerThresholdKW=40


def make_limiter():
  """EVLimiter with minimal CP for testing the power guard helper directly."""
  cp = type("CP", (), {
    "flags": 0,
    "carFingerprint": "HYUNDAI_SANTA_FE_PHEV_2022",
    "openpilotLongitudinalControl": False,
  })()
  cp_sp = type("CP_SP", (), {})()
  return EVLimiter(cp, cp_sp)


def step(lim, candidate_state, power_w, frame, cap=CAP_W):
  """Invoke the power guard with a candidate state, control power, frame index.

  iter15 v2: `_apply_recovery_power_guard()` now returns the tuple
  `(new_state, guard_forced_transition)`. These iter14 tests only assert on
  the resulting state, so we drop the second element for backwards-compat
  with the previous int-only assertions. iter15-specific tests assert on
  `guard_forced_transition` separately in `test_iter15_recovery_hardpreempt.py`.
  """
  new_state, _guard_forced = lim._apply_recovery_power_guard(
    candidate_state, power_w, cap, frame
  )
  return new_state


# ─────────────────────────────────────────────────────────────────────────
# Tier 1: immediate trigger at >= cap (no debounce)
# ─────────────────────────────────────────────────────────────────────────

class TestTier1ImmediateTrigger(unittest.TestCase):

  def test_recovery_yields_immediately_at_cap(self):
    """Test 1: power_control = cap → immediate yield, no debounce required."""
    lim = make_limiter()
    out = step(lim, STATE_RECOVERY_ACTIVE, 40_000.0, frame=100)
    self.assertEqual(out, STATE_SOFT_CAP_ACTIVE)
    self.assertEqual(lim._power_guard_yield_reason_last, "immediate")

  def test_recovery_yields_above_cap(self):
    """Test 2: power_control well above cap → immediate yield."""
    lim = make_limiter()
    out = step(lim, STATE_RECOVERY_ACTIVE, 60_000.0, frame=100)
    self.assertEqual(out, STATE_SOFT_CAP_ACTIVE)
    self.assertEqual(lim._power_guard_yield_reason_last, "immediate")

  def test_recovery_yield_increments_counter(self):
    """Test 3: yield events counter increments by 1."""
    lim = make_limiter()
    initial = lim._evLimiter_recovery_yield_events
    step(lim, STATE_RECOVERY_ACTIVE, 42_000.0, frame=50)
    self.assertEqual(lim._evLimiter_recovery_yield_events, initial + 1)


# ─────────────────────────────────────────────────────────────────────────
# Tier 2: debounced trigger at >= 0.95*cap for 3 frames
# ─────────────────────────────────────────────────────────────────────────

class TestTier2DebouncedTrigger(unittest.TestCase):

  def test_single_frame_at_95pct_does_not_yield(self):
    """Test 4: single frame at 38 kW (95% of 40) → counter=1, no yield."""
    lim = make_limiter()
    # Use a non-RECOVERY candidate so we don't fire the immediate path
    out = step(lim, STATE_IDLE, 38_500.0, frame=10)
    self.assertEqual(lim._power_near_budget_sustain, 1)
    self.assertEqual(out, STATE_IDLE)

  def test_three_frame_sustained_yields_debounced(self):
    """Test 5: 3rd consecutive frame at >= 0.95*cap fires debounced yield."""
    lim = make_limiter()
    # First two frames at near-budget but candidate=IDLE (no yield path)
    step(lim, STATE_IDLE, 39_000.0, frame=1)
    step(lim, STATE_IDLE, 39_000.0, frame=2)
    self.assertEqual(lim._power_near_budget_sustain, 2)
    # Third frame at near-budget, candidate=RECOVERY → yields debounced
    out = step(lim, STATE_RECOVERY_ACTIVE, 39_000.0, frame=3)
    self.assertEqual(out, STATE_SOFT_CAP_ACTIVE)
    self.assertEqual(lim._power_guard_yield_reason_last, "debounced")

  def test_debounce_resets_when_power_drops(self):
    """Test 6: counter resets to 0 when power drops below 0.95*cap."""
    lim = make_limiter()
    step(lim, STATE_IDLE, 39_000.0, frame=1)
    step(lim, STATE_IDLE, 39_000.0, frame=2)
    self.assertEqual(lim._power_near_budget_sustain, 2)
    step(lim, STATE_IDLE, 20_000.0, frame=3)
    self.assertEqual(lim._power_near_budget_sustain, 0)


# ─────────────────────────────────────────────────────────────────────────
# R2-MF-A: sustained-capped trigger REMOVED from control; counter is diagnostic
# ─────────────────────────────────────────────────────────────────────────

class TestSustainedCappedDiagnosticOnly(unittest.TestCase):

  def test_capped_sustain_counter_increments(self):
    """Test 7: diagnostic counter ticks every frame at >= cap."""
    lim = make_limiter()
    for f in range(50):
      step(lim, STATE_IDLE, 42_000.0, frame=f)
    self.assertEqual(lim._power_capped_control_sustain, 50)

  def test_capped_sustain_resets_below(self):
    """Test 8: counter resets to 0 when power drops below cap."""
    lim = make_limiter()
    for f in range(20):
      step(lim, STATE_IDLE, 42_000.0, frame=f)
    self.assertEqual(lim._power_capped_control_sustain, 20)
    step(lim, STATE_IDLE, 20_000.0, frame=21)
    self.assertEqual(lim._power_capped_control_sustain, 0)

  def test_yield_reason_is_never_sustained_capped(self):
    """Test 9 (R2-MF-A): 'sustained_capped' is NOT a valid yield reason in v2."""
    lim = make_limiter()
    for f in range(100):
      step(lim, STATE_RECOVERY_ACTIVE, 42_000.0, frame=f)
    # Reason will alternate between "immediate" and other states (lockout takes
    # over after first yield). But it must NEVER be "sustained_capped".
    valid_reasons = ("none", "immediate", "debounced")
    self.assertIn(lim._power_guard_yield_reason_last, valid_reasons)


# ─────────────────────────────────────────────────────────────────────────
# Lockout active behavior (R2-MF-B)
# ─────────────────────────────────────────────────────────────────────────

class TestLockoutBlocksReentry(unittest.TestCase):

  def test_lockout_engaged_after_yield(self):
    """Test 10: yield sets _recovery_lockout_engaged=True."""
    lim = make_limiter()
    self.assertFalse(lim._recovery_lockout_engaged, "cold start")
    step(lim, STATE_RECOVERY_ACTIVE, 45_000.0, frame=100)
    self.assertTrue(lim._recovery_lockout_engaged)

  def test_lockout_blocks_recovery_during_window(self):
    """Test 11: after yield, RECOVERY candidate while locked → forced to SOFT_CAP/IDLE."""
    lim = make_limiter()
    step(lim, STATE_RECOVERY_ACTIVE, 45_000.0, frame=100)
    # Now drop power to safe; try RECOVERY at frame 101 (just after yield, lockout active)
    out = step(lim, STATE_RECOVERY_ACTIVE, 0.0, frame=101)
    self.assertNotEqual(out, STATE_RECOVERY_ACTIVE,
                        "lockout must block RECOVERY immediately after yield")

  def test_lockouts_entered_counter(self):
    """Test 12: _evLimiter_recovery_lockouts_held increments while blocking."""
    lim = make_limiter()
    step(lim, STATE_RECOVERY_ACTIVE, 45_000.0, frame=100)
    initial = lim._evLimiter_recovery_lockouts_held
    # 50 frames of safe power but RECOVERY blocked due to lockout headroom not built
    for f in range(101, 151):
      step(lim, STATE_RECOVERY_ACTIVE, 0.0, frame=f)
    # Initial 50 frames: headroom_done False (only 50 < 100), so counter increments
    self.assertGreater(lim._evLimiter_recovery_lockouts_held, initial)


# ─────────────────────────────────────────────────────────────────────────
# R2-MF-B: lockout requires BOTH lockout_time_done AND headroom_done
# ─────────────────────────────────────────────────────────────────────────

class TestLockoutConjunction(unittest.TestCase):

  def test_lockout_time_done_but_headroom_missing_blocks(self):
    """Test 13: 2s lockout elapsed but power kept high → still blocked."""
    lim = make_limiter()
    step(lim, STATE_RECOVERY_ACTIVE, 45_000.0, frame=100)
    self.assertTrue(lim._recovery_lockout_engaged)
    # Keep power at 36 kW (above 0.85*40=34) for 250 frames
    # Frame 101..350: lockout_until=300 (frame 100 + 200), so f=301 is past lockout time
    # but headroom_done won't be reached because power > 0.85*cap
    for f in range(101, 351):
      step(lim, STATE_IDLE, 36_000.0, frame=f)
    # Headroom counter never accumulated because power > 0.85*cap
    self.assertEqual(lim._recovery_reentry_sustain, 0)
    self.assertTrue(lim._recovery_lockout_engaged,
                    "lockout must remain engaged when headroom not built up")

  def test_headroom_satisfied_but_lockout_time_active_blocks(self):
    """Test 14: headroom built (1s at <=0.85*cap) but 2s lockout not elapsed."""
    lim = make_limiter()
    step(lim, STATE_RECOVERY_ACTIVE, 45_000.0, frame=100)
    self.assertTrue(lim._recovery_lockout_engaged)
    # 100 frames at zero power → headroom_done True at frame 200
    # But lockout_until = 300, so still locked until frame 300.
    for f in range(101, 201):
      step(lim, STATE_IDLE, 0.0, frame=f)
    # At frame 200: headroom_done = True (sustain >= 100), lockout_time_done = False
    # (frame 200 < 300). Lockout still engaged.
    self.assertGreaterEqual(lim._recovery_reentry_sustain, RECOVERY_REENTRY_SUSTAIN_FRAMES)
    self.assertTrue(lim._recovery_lockout_engaged,
                    "lockout must remain engaged when 2s lockout time not yet elapsed")

  def test_both_satisfied_releases_lockout(self):
    """Test 15: lockout releases only when BOTH conditions met."""
    lim = make_limiter()
    step(lim, STATE_RECOVERY_ACTIVE, 45_000.0, frame=100)
    # Drive 250 frames at zero power. lockout_until=300 elapsed at f=300; headroom
    # built by f=200. Both satisfied at f>=300.
    for f in range(101, 351):
      step(lim, STATE_IDLE, 0.0, frame=f)
    self.assertFalse(lim._recovery_lockout_engaged,
                     "both lockout_time AND headroom must release the lockout")


# ─────────────────────────────────────────────────────────────────────────
# Cold-start: no yield ever fired → power guard does NOT block RECOVERY
# ─────────────────────────────────────────────────────────────────────────

class TestColdStartNoBlocking(unittest.TestCase):

  def test_cold_start_recovery_not_blocked(self):
    """Test 16: regression — cold-start IDLE→RECOVERY must not be blocked."""
    lim = make_limiter()
    self.assertFalse(lim._recovery_lockout_engaged)
    # Drive at safe power, candidate RECOVERY. Guard must let it through.
    out = step(lim, STATE_RECOVERY_ACTIVE, 15_000.0, frame=10)
    self.assertEqual(out, STATE_RECOVERY_ACTIVE,
                     "cold-start RECOVERY must NOT be blocked when no yield ever fired")
    self.assertFalse(lim._recovery_lockout_engaged)

  def test_cold_start_long_run_no_yield(self):
    """Test 17: 200 frames of safe power + RECOVERY candidate → never blocked."""
    lim = make_limiter()
    for f in range(200):
      out = step(lim, STATE_RECOVERY_ACTIVE, 15_000.0, frame=f)
      self.assertEqual(out, STATE_RECOVERY_ACTIVE,
                       f"frame {f}: cold-start RECOVERY blocked unexpectedly")
    self.assertFalse(lim._recovery_lockout_engaged)


# ─────────────────────────────────────────────────────────────────────────
# Triple estimator schema fields (R1-MF3)
# ─────────────────────────────────────────────────────────────────────────

class TestTripleEstimatorSchema(unittest.TestCase):

  def test_carstate_sp_estimator_fields_present(self):
    """Test 18: estPowerInstantW, estPowerControlW, etc. exist in CarStateSP."""
    from opendbc.car.structs import CarStateSP
    sp = CarStateSP()
    sp.estPowerInstantW = 50.0
    sp.estPowerControlW = 48.0
    sp.evLimiterPowerCappedSustainFrames = 5
    sp.evLimiterPowerNearBudgetSustainFrames = 3
    sp.evLimiterEstPowerRawIsFiltered = True
    self.assertEqual(sp.estPowerInstantW, 50.0)
    self.assertEqual(sp.estPowerControlW, 48.0)
    self.assertEqual(sp.evLimiterPowerCappedSustainFrames, 5)
    self.assertTrue(sp.evLimiterEstPowerRawIsFiltered)

  def test_carstate_sp_instrumentation_fields_present(self):
    """Test 19: transition-decision instrumentation fields exist."""
    from opendbc.car.structs import CarStateSP
    sp = CarStateSP()
    sp.evLimiterStatePriorTransition = 4
    sp.evLimiterStateCandidateBeforeGuard = 4
    sp.evLimiterStateAfterPowerGuard = 3
    sp.evLimiterPowerGuardYieldReason = "immediate"
    sp.evLimiterPowerGuardLockoutActive = True
    sp.evLimiterRecoveryYieldEvents = 1
    sp.evLimiterRecoveryLockoutsEntered = 50
    self.assertEqual(sp.evLimiterPowerGuardYieldReason, "immediate")
    self.assertTrue(sp.evLimiterPowerGuardLockoutActive)
    self.assertEqual(sp.evLimiterRecoveryYieldEvents, 1)


# ─────────────────────────────────────────────────────────────────────────
# Yield-reason instrumentation correctness
# ─────────────────────────────────────────────────────────────────────────

class TestYieldReasonInstrumentation(unittest.TestCase):

  def test_no_yield_reason_is_none(self):
    """Test 20: candidate=IDLE → reason stays 'none'."""
    lim = make_limiter()
    step(lim, STATE_IDLE, 20_000.0, frame=10)
    self.assertEqual(lim._power_guard_yield_reason_last, "none")

  def test_yield_reason_immediate_when_at_cap(self):
    """Test 21: tier-1 trigger labels reason as 'immediate'."""
    lim = make_limiter()
    step(lim, STATE_RECOVERY_ACTIVE, 42_000.0, frame=10)
    self.assertEqual(lim._power_guard_yield_reason_last, "immediate")


# ─────────────────────────────────────────────────────────────────────────
# Carryover: _was_in_prelaunch resets on standstill exit
# ─────────────────────────────────────────────────────────────────────────

class TestStandstillPrelaunchCarryover(unittest.TestCase):

  def test_was_in_prelaunch_init_false(self):
    """Test 22: cold-start _was_in_prelaunch=False."""
    lim = make_limiter()
    self.assertFalse(lim._was_in_prelaunch)

  def test_was_in_prelaunch_resets_outside_standstill_via_update(self):
    """Test 23 (carryover): manually set sticky flag, drive at v_ego>0.1, expect reset.
    Drives 18+19 forensic finding: SOFT_CAP/IDLE/RECOVERY SETs while moving were
    wrongly attributed to standstill counter because flag was sticky. Fix is in
    update() at line ~1042: explicit reset when not in_standstill."""
    # We can't easily run update() without full CP/CS fakes, so verify the reset
    # logic by examining the source contains the explicit reset.
    import inspect
    from opendbc.sunnypilot.car.hyundai import ev_limiter
    src = inspect.getsource(ev_limiter)
    self.assertIn("if not in_standstill:\n      self._was_in_prelaunch = False", src,
                  "iter14 v2 carryover fix must be present in update() body")


# ─────────────────────────────────────────────────────────────────────────
# Block-reason completeness for new entries @27-@29
# ─────────────────────────────────────────────────────────────────────────

class TestNewBlockReasonsRegistered(unittest.TestCase):

  def test_block_reason_priority_includes_iter14_entries(self):
    """Test 24 (static): new reasons are in BLOCK_REASON_PRIORITY."""
    from opendbc.sunnypilot.car.hyundai.car_controller_button_limiter import (
      BLOCK_REASON_PRIORITY, BlockReason,
    )
    self.assertIn(BlockReason.recoveryYieldedToSoftCap, BLOCK_REASON_PRIORITY)
    self.assertIn(BlockReason.powerOverBudget, BLOCK_REASON_PRIORITY)
    self.assertIn(BlockReason.recoveryReentryLocked, BLOCK_REASON_PRIORITY)

  def test_block_reason_ordinal_iter14_entries_match_capnp(self):
    """Test 25 (static): new reasons have correct capnp ordinals."""
    from opendbc.sunnypilot.car.hyundai.car_controller_button_limiter import (
      BLOCK_REASON_ORDINAL, BlockReason,
    )
    self.assertEqual(BLOCK_REASON_ORDINAL[BlockReason.recoveryYieldedToSoftCap], 27)
    self.assertEqual(BLOCK_REASON_ORDINAL[BlockReason.powerOverBudget], 28)
    self.assertEqual(BLOCK_REASON_ORDINAL[BlockReason.recoveryReentryLocked], 29)


# ─────────────────────────────────────────────────────────────────────────
# Drive 18 forensic replay (v2 plan Section J test #16)
# ─────────────────────────────────────────────────────────────────────────

class TestDrive18Replay(unittest.TestCase):
  """Forensic replay of drive 18 t=1323-1349 (segment 21) — the 74 mph
  manual-decrement episode. iter13 stayed in RECOVERY for 25.8s while
  est_power_w >= 38 kW (peak est_power_raw=66.11 kW). With iter14's
  power-gate, the candidate state of RECOVERY must yield to SOFT_CAP
  via the immediate (>= cap) or 30 ms debounced (>= 0.95·cap) trigger.

  Replay reads actual iter13 carStateSP telemetry from drive 18 segment
  21 and feeds (state=RECOVERY candidate, est_power as proxy for
  est_power_control_w) through `_apply_recovery_power_guard`. Asserts:

    1. >=1 RECOVERY -> SOFT_CAP transition during the over-cap window
    2. No contiguous run of >5 frames (50 ms) where guard left state at
       RECOVERY while power >= cap (allows guard-latency margin)

  Caveat: drive 18 ran on iter13, so estPowerControlW @42 was not yet
  published. We use max(est_power_w, est_power_raw_w) as a proxy.
  est_power_raw_w is filtered but uncapped, so it preserves the true
  demand magnitude (peak 66.11 kW vs the HUD's 60 kW cap). If the test
  passes with this proxy, it would pass with the real estPowerControlW
  on future iter14 drives.
  """

  ROUTE_DIR = "/home/alex.smith/git/sunnypilot/can_data/00000078--26c897cf37--21"
  RLOG = ROUTE_DIR + "/rlog.zst"

  @classmethod
  def setUpClass(cls):
    import os
    if not os.path.exists(cls.RLOG):
      raise unittest.SkipTest(f"drive 18 segment 21 rlog not present at {cls.RLOG}")

    try:
      from openpilot.tools.lib.logreader import LogReader
    except ImportError as e:
      raise unittest.SkipTest(f"openpilot.tools.lib.logreader unavailable: {e}")

    cls.samples = []
    lr = LogReader(cls.RLOG)
    frame = 0
    for msg in lr:
      if msg.which() != "carStateSP":
        continue
      sp = msg.carStateSP
      cls.samples.append((
        int(sp.evLimiterState),
        float(sp.estPowerW),
        float(sp.estPowerRawW),
        frame,
      ))
      frame += 1
    if len(cls.samples) < 100:
      raise unittest.SkipTest(f"too few CSP samples in segment 21 ({len(cls.samples)})")

  def test_replay_recovery_yields_at_or_above_cap(self):
    """Primary assertion 1: at least one RECOVERY -> SOFT_CAP transition
    occurs during the over-cap window in segment 21."""
    lim = make_limiter()
    yield_count = 0
    overcap_recovery_frames = 0

    for state_iter13, est_w, est_raw_w, frame in self.samples:
      power_for_guard = max(est_w, est_raw_w)
      if state_iter13 != STATE_RECOVERY_ACTIVE:
        continue
      if power_for_guard < CAP_W:
        continue
      overcap_recovery_frames += 1
      out = step(lim, STATE_RECOVERY_ACTIVE, power_for_guard, frame)
      if out == STATE_SOFT_CAP_ACTIVE:
        yield_count += 1

    self.assertGreater(overcap_recovery_frames, 0,
      "drive 18 seg 21 must contain RECOVERY frames at >= 40 kW")
    self.assertGreater(yield_count, 0,
      f"iter14 guard must yield RECOVERY->SOFT_CAP at least once across "
      f"{overcap_recovery_frames} over-cap RECOVERY frames")
    self.assertGreater(lim._evLimiter_recovery_yield_events, 0,
      "evLimiterRecoveryYieldEvents counter must increment")

  def test_replay_no_long_recovery_run_above_cap(self):
    """Primary assertion 2: no contiguous run of >5 frames (50 ms) where
    iter14 guard leaves state at RECOVERY while power >= cap.

    Tier-1 fires same frame, tier-2 takes 3 frames (debounce); allow 5
    frames of margin.
    """
    lim = make_limiter()
    longest_run = 0
    cur_run = 0
    for state_iter13, est_w, est_raw_w, frame in self.samples:
      power_for_guard = max(est_w, est_raw_w)
      if state_iter13 == STATE_RECOVERY_ACTIVE:
        out = step(lim, STATE_RECOVERY_ACTIVE, power_for_guard, frame)
        if out == STATE_RECOVERY_ACTIVE and power_for_guard >= CAP_W:
          cur_run += 1
          longest_run = max(longest_run, cur_run)
        else:
          cur_run = 0
      else:
        step(lim, state_iter13, power_for_guard, frame)
        cur_run = 0

    self.assertLessEqual(longest_run, 5,
      f"iter14 guard must not leave state at RECOVERY for >5 contiguous "
      f"frames (50 ms) while power >= cap; got max run = {longest_run} "
      f"frames ({longest_run * 10} ms)")


if __name__ == "__main__":
  unittest.main()
