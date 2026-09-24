# ruff: noqa: ISC002
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter15 v2 (Section C) — long-standstill narrow reset + state-vector telemetry.

R2-MF-3 PRESERVATION tests. v1 had two CLEARING tests that contradicted v1's
narrow-reset design. v2 replaces them with PRESERVATION tests that verify
critical state is INTACT after a long standstill exit:
  - `_softcap_from_recovery_lockout_until` (frame counter; expires naturally)
  - `_softcap_enter_sustain`, `_softcap_exit_sustain`
  - `_recovery_enter_sustain`, `_recovery_exit_sustain`
  - `_recovery_lockout_engaged`, `_recovery_reentry_sustain`

The ONLY two things the narrow reset clears (conditionally):
  - `_prelaunch_no_ack_backoff_until` (if still in the future)
  - `_softcap_entry_reason` (if power now comfortably below cap)

Plan Section I items 10-15.
"""
from __future__ import annotations

import unittest

from opendbc.sunnypilot.car.hyundai.ev_limiter import (
  EVLimiter,
)
from opendbc.car.hyundai.values import HyundaiFlags


def _make_limiter():
  cp = type("CP", (), {
    "flags": HyundaiFlags.HYBRID,
    "carFingerprint": "HYUNDAI_SANTA_FE_PHEV_2022",
    "openpilotLongitudinalControl": False,
  })()
  cp_sp = type("CP_SP", (), {"flags": 0})()
  return EVLimiter(cp, cp_sp)


# ─────────────────────────────────────────────────────────────────────────
# Plan #10 — short standstill: NO reset of anything.
# ─────────────────────────────────────────────────────────────────────────

class TestShortStandstillNoReset(unittest.TestCase):

  def test_short_standstill_no_reset(self):
    """time_in_standstill <= LONG_STANDSTILL_RESET_FRAMES → no reset at all."""
    lim = _make_limiter()
    # Seed nominal mid-drive state.
    lim._softcap_enter_sustain = 25
    lim._softcap_exit_sustain = 10
    lim._recovery_enter_sustain = 15
    lim._recovery_exit_sustain = 5
    lim._recovery_lockout_engaged = True
    lim._softcap_from_recovery_lockout_until = 1000
    lim._softcap_entry_reason = "recovery_power_immediate"
    # Set a future backoff so we can verify it is NOT cleared.
    lim._prelaunch_no_ack_backoff_until = 5000
    pre_resets = lim._evLimiter_long_standstill_resets

    lim._on_leaving_standstill(
      frame=100, time_in_standstill_frames=200,   # 2 s < 5 s threshold
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    # Counter NOT incremented; nothing cleared.
    self.assertEqual(lim._evLimiter_long_standstill_resets, pre_resets)
    self.assertEqual(lim._prelaunch_no_ack_backoff_until, 5000)
    self.assertEqual(lim._softcap_entry_reason, "recovery_power_immediate")
    self.assertEqual(lim._softcap_enter_sustain, 25)
    self.assertEqual(lim._softcap_exit_sustain, 10)
    self.assertEqual(lim._recovery_enter_sustain, 15)
    self.assertEqual(lim._recovery_exit_sustain, 5)


# ─────────────────────────────────────────────────────────────────────────
# R2-MF-3 PRESERVATION TESTS — long standstill must NOT touch these fields.
# ─────────────────────────────────────────────────────────────────────────

class TestLongStandstillPreservation(unittest.TestCase):
  """R2-MF-3: critical fields must be PRESERVED across long-standstill exit.
  v0's blunt reset cleared sustain counters; resetting `_softcap_exit_sustain=0`
  could ITSELF add 2 s delay (since SOFT_CAP→IDLE requires SOFT_CAP_EXIT_SUSTAIN_FRAMES
  of sustained exit predicate). Could WORSEN the symptom we're trying to fix.
  """

  def test_long_standstill_preserves_softcap_lockout_until_frame(self):
    """Plan #11: `_softcap_from_recovery_lockout_until` MUST be untouched.
    It's a frame counter; expires naturally. Resetting it could reopen
    RECOVERY in an unsafe window."""
    lim = _make_limiter()
    lim._softcap_from_recovery_lockout_until = 1000
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,   # 8 s > 5 s threshold
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._softcap_from_recovery_lockout_until, 1000,
                     "Lockout frame counter must NOT be cleared (R2-MF-3 PRESERVATION).")

  def test_long_standstill_preserves_sustain_counters(self):
    """Plan #12: sustain counters MUST be untouched. Resetting them could
    delay post-standstill state machine response by SOFT_CAP_EXIT_SUSTAIN_FRAMES
    (=200, 2 s) or longer — exactly the slow-takeoff we're trying to fix."""
    lim = _make_limiter()
    lim._softcap_enter_sustain = 22
    lim._softcap_exit_sustain = 88
    lim._recovery_enter_sustain = 33
    lim._recovery_exit_sustain = 77
    lim._recovery_reentry_sustain = 11
    lim._power_near_budget_sustain = 5
    lim._power_capped_control_sustain = 6
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._softcap_enter_sustain, 22)
    self.assertEqual(lim._softcap_exit_sustain, 88)
    self.assertEqual(lim._recovery_enter_sustain, 33)
    self.assertEqual(lim._recovery_exit_sustain, 77)
    self.assertEqual(lim._recovery_reentry_sustain, 11)
    self.assertEqual(lim._power_near_budget_sustain, 5)
    self.assertEqual(lim._power_capped_control_sustain, 6)

  def test_long_standstill_preserves_recovery_lockout_engaged_flag(self):
    """R2-MF-3 supplement: `_recovery_lockout_engaged` is one-shot; cleared
    only on recovery_unlocked. The long-standstill handler must not touch it."""
    lim = _make_limiter()
    lim._recovery_lockout_engaged = True
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertTrue(lim._recovery_lockout_engaged)


# ─────────────────────────────────────────────────────────────────────────
# The ONLY two things the long-standstill handler clears (conditional).
# ─────────────────────────────────────────────────────────────────────────

class TestLongStandstillNarrowClears(unittest.TestCase):

  def test_long_standstill_clears_stale_prelaunch_no_ack_backoff(self):
    """Plan #13: PRELAUNCH no-ack backoff is cleared IF and only IF still in
    the future at the exit frame (otherwise it already expired naturally)."""
    lim = _make_limiter()
    lim._prelaunch_no_ack_backoff_until = 5000
    pre_cleared = lim._evLimiter_long_standstill_prelaunch_backoff_cleared
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    # Was in future (5000 > 300); should have been moved to frame.
    self.assertEqual(lim._prelaunch_no_ack_backoff_until, 300)
    self.assertEqual(lim._evLimiter_long_standstill_prelaunch_backoff_cleared,
                     pre_cleared + 1)

  def test_long_standstill_NOT_clears_already_expired_prelaunch_backoff(self):
    """Conditional clear: if backoff already past, do NOTHING and DON'T
    increment counter."""
    lim = _make_limiter()
    lim._prelaunch_no_ack_backoff_until = 100   # already past current frame
    pre_cleared = lim._evLimiter_long_standstill_prelaunch_backoff_cleared
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._prelaunch_no_ack_backoff_until, 100)
    self.assertEqual(lim._evLimiter_long_standstill_prelaunch_backoff_cleared,
                     pre_cleared)

  def test_long_standstill_clears_stale_softcap_reason_if_power_now_below_threshold(self):
    """Plan #14: stale softcap_entry_reason cleared IFF power_control now
    < 0.85*cap (STALE_SOFTCAP_REASON_POWER_FRAC)."""
    lim = _make_limiter()
    lim._softcap_entry_reason = "recovery_power_immediate"
    pre_cleared = lim._evLimiter_long_standstill_softcap_reason_cleared
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      # power = 20 kW = 0.5 * 40 kW cap; well below 0.85*cap = 34 kW
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._softcap_entry_reason, "")
    self.assertEqual(lim._evLimiter_long_standstill_softcap_reason_cleared,
                     pre_cleared + 1)

  def test_long_standstill_NOT_clears_softcap_reason_if_power_still_near_cap(self):
    """Conditional clear negative case: if power still high, reason is current
    (not stale) and must NOT be cleared."""
    lim = _make_limiter()
    lim._softcap_entry_reason = "recovery_power_immediate"
    pre_cleared = lim._evLimiter_long_standstill_softcap_reason_cleared
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      # power = 38 kW = 0.95*cap; above 0.85*cap → NOT stale
      est_power_control_w=38_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._softcap_entry_reason, "recovery_power_immediate")
    self.assertEqual(lim._evLimiter_long_standstill_softcap_reason_cleared,
                     pre_cleared)


# ─────────────────────────────────────────────────────────────────────────
# Plan #15: long_standstill_resets event counter increments only on reset.
# ─────────────────────────────────────────────────────────────────────────

class TestLongStandstillResetsCounter(unittest.TestCase):

  def test_long_standstill_reset_event_counter_increments_only_on_reset(self):
    """Plan #15: `_evLimiter_long_standstill_resets` increments only if at
    least one of the two conditional clears actually fired."""
    lim = _make_limiter()
    pre_resets = lim._evLimiter_long_standstill_resets
    # Setup: no stale backoff, no stale reason (power is high so reason is
    # current). Nothing should fire.
    lim._prelaunch_no_ack_backoff_until = -10000
    lim._softcap_entry_reason = "recovery_power_immediate"
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      est_power_control_w=38_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._evLimiter_long_standstill_resets, pre_resets,
                     "No conditional clear fired → counter must NOT increment.")
    # Now setup BOTH conditions to fire.
    lim._prelaunch_no_ack_backoff_until = 5000
    lim._softcap_entry_reason = "recovery_power_immediate"
    lim._on_leaving_standstill(
      frame=300, time_in_standstill_frames=800,
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._evLimiter_long_standstill_resets, pre_resets + 1,
                     "At least one conditional clear fired → counter must "
                     "increment by exactly 1 (not 2 — single per-exit event).")


# ─────────────────────────────────────────────────────────────────────────
# State-vector telemetry tests (Section C R1-MF-C).
# ─────────────────────────────────────────────────────────────────────────

class TestStandstillExitStateVector(unittest.TestCase):

  def test_snapshot_published_on_every_exit(self):
    """Snapshot is latched on every exit, including short-stop. Bounded ≤200 chars."""
    lim = _make_limiter()
    lim._softcap_enter_sustain = 7
    lim._softcap_exit_sustain = 12
    lim._on_leaving_standstill(
      frame=100, time_in_standstill_frames=50,   # short stop
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    snap = lim._standstill_exit_state_snapshot_last
    self.assertTrue(len(snap) > 0, "Snapshot must populate.")
    self.assertLessEqual(len(snap), 200, "Snapshot must be ≤200 chars.")
    self.assertIn("st=", snap)
    self.assertIn("sec=", snap)
    self.assertIn("rec=", snap)

  def test_exit_time_s_recorded(self):
    """`_standstill_exit_time_s_last` = time-in-standstill in seconds at exit."""
    lim = _make_limiter()
    lim._on_leaving_standstill(
      frame=200, time_in_standstill_frames=750,   # 7.5 s
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertAlmostEqual(lim._standstill_exit_time_s_last, 7.5, places=2)

  def test_exit_to_first_res_latency_reset_on_each_exit(self):
    """Latency counter resets to 0 on every standstill exit."""
    lim = _make_limiter()
    lim._standstill_exit_to_first_res_latency_frames = 999
    lim._on_leaving_standstill(
      frame=100, time_in_standstill_frames=50,
      est_power_control_w=20_000.0, power_threshold_w=40_000.0,
    )
    self.assertEqual(lim._standstill_exit_to_first_res_latency_frames, 0)
    self.assertTrue(lim._standstill_exit_post_exit_active)
    self.assertFalse(lim._standstill_exit_first_res_seen)


if __name__ == "__main__":
  unittest.main()
