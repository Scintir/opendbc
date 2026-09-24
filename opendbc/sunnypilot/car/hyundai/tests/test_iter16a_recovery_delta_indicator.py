# ruff: noqa: E702
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter16a tests:
  B2 — hysteretic suppression of RES while set-vs-actual delta is large (gpt-5.5 review).
  Phase A — live request-indicator signals (request_dir / button_dir / request_honored).
"""
from __future__ import annotations

import unittest

from opendbc.car.hyundai.values import Buttons
from opendbc.sunnypilot.car.hyundai.ev_limiter import (
  MPH_TO_MS,
  RECOVERY_MAX_DELTA_ENTER_MPH,
  RECOVERY_MAX_DELTA_EXIT_MPH,
)
# Reuse the iter15 test harness fixtures.
from opendbc.sunnypilot.car.hyundai.tests.test_iter15_recovery_hardpreempt import (
  _make_limiter, _step, _engage,
)


class TestIter16aRecoveryDeltaBlock(unittest.TestCase):
  def test_delta_block_latches_with_hysteresis(self):
    """The block latches on at ENTER and only clears below EXIT (no chatter)."""
    lim = _make_limiter()
    # Engage at low speed where the 20 mph margin allows a big delta.
    _engage(lim, end_frame=30, vEgo_mph=15.0, observed_mph=20.0)
    f = 30
    # Delta just under ENTER -> not blocked yet.
    _step(lim, f, cc_enabled=True, vEgo=15.0 * MPH_TO_MS,
          observed_mph=15.0 + RECOVERY_MAX_DELTA_ENTER_MPH - 1.0); f += 1
    self.assertFalse(lim._recovery_delta_block)
    # Delta over ENTER -> blocked.
    _step(lim, f, cc_enabled=True, vEgo=15.0 * MPH_TO_MS,
          observed_mph=15.0 + RECOVERY_MAX_DELTA_ENTER_MPH + 2.0); f += 1
    self.assertTrue(lim._recovery_delta_block)
    # Delta between EXIT and ENTER -> stays blocked (hysteresis).
    _step(lim, f, cc_enabled=True, vEgo=15.0 * MPH_TO_MS,
          observed_mph=15.0 + (RECOVERY_MAX_DELTA_ENTER_MPH + RECOVERY_MAX_DELTA_EXIT_MPH) / 2.0); f += 1
    self.assertTrue(lim._recovery_delta_block)
    # Delta below EXIT -> clears.
    _step(lim, f, cc_enabled=True, vEgo=15.0 * MPH_TO_MS,
          observed_mph=15.0 + RECOVERY_MAX_DELTA_EXIT_MPH - 1.0); f += 1
    self.assertFalse(lim._recovery_delta_block)

  def test_no_RES_emitted_while_delta_block_active(self):
    """Safety invariant: the limiter never emits RES (UP) while delta is large."""
    lim = _make_limiter()
    _engage(lim, end_frame=30, vEgo_mph=12.0, observed_mph=18.0)
    emitted_res = 0
    for f in range(30, 230):
      # Sustained large delta: cluster set held ~25 mph above a 12 mph actual.
      btn, _active = _step(lim, f, cc_enabled=True, vEgo=12.0 * MPH_TO_MS, observed_mph=37.0)
      if btn == Buttons.RES_ACCEL:
        emitted_res += 1
    self.assertTrue(lim._recovery_delta_block)
    self.assertEqual(emitted_res, 0, "limiter must not raise set speed while delta is large")


class TestIter16aRequestIndicator(unittest.TestCase):
  def test_button_dir_maps_emitted_button(self):
    lim = _make_limiter()
    # Drive a windup that forces SET-down emissions and confirm DOWN maps to 2.
    _engage(lim, end_frame=30, vEgo_mph=45.0, observed_mph=60.0)
    saw_down = False
    for f in range(30, 120):
      btn, _active = _step(lim, f, cc_enabled=True, vEgo=45.0 * MPH_TO_MS, observed_mph=60.0,
                  est_power_w=50_000.0)
      if btn == Buttons.SET_DECEL:
        self.assertEqual(lim._button_dir, 2)
        saw_down = True
      elif btn == Buttons.NONE:
        self.assertEqual(lim._button_dir, 0)
    self.assertTrue(saw_down, "expected at least one SET-down emission in this scenario")

  def test_request_dir_never_both(self):
    """Intent is mutually exclusive: never publishes both UP and DOWN (fail-closed)."""
    lim = _make_limiter()
    _engage(lim, end_frame=30, vEgo_mph=40.0, observed_mph=55.0)
    for f in range(30, 200):
      _step(lim, f, cc_enabled=True, vEgo=40.0 * MPH_TO_MS, observed_mph=55.0,
            est_power_w=48_000.0)
      self.assertIn(lim._request_dir, (0, 1, 2))

  def test_honored_when_set_speed_moves_down_after_emit(self):
    lim = _make_limiter()
    _engage(lim, end_frame=30, vEgo_mph=45.0, observed_mph=60.0)
    f = 30
    # Drive until a SET-down emits.
    obs = 60.0
    emitted_frame = None
    for _ in range(90):
      btn, _active = _step(lim, f, cc_enabled=True, vEgo=45.0 * MPH_TO_MS, observed_mph=obs,
                  est_power_w=50_000.0)
      if btn == Buttons.SET_DECEL:
        emitted_frame = f
        break
      f += 1
    self.assertIsNotNone(emitted_frame, "no SET-down emitted to test honored path")
    # Now the SCC honors it: cluster set drops by >0.5 mph.
    obs -= 2.0
    f += 1
    _step(lim, f, cc_enabled=True, vEgo=45.0 * MPH_TO_MS, observed_mph=obs, est_power_w=50_000.0)
    self.assertEqual(lim._request_honored, 1)

  def test_ignored_when_set_speed_unchanged_past_ack_window(self):
    lim = _make_limiter()
    _engage(lim, end_frame=30, vEgo_mph=45.0, observed_mph=60.0)
    f = 30
    emitted_frame = None
    for _ in range(90):
      btn, _active = _step(lim, f, cc_enabled=True, vEgo=45.0 * MPH_TO_MS, observed_mph=60.0,
                  est_power_w=50_000.0)
      if btn == Buttons.SET_DECEL:
        emitted_frame = f
        break
      f += 1
    self.assertIsNotNone(emitted_frame)
    # Hold the set speed pinned (SCC ignores) well past the ACK window.
    for _ in range(120):
      f += 1
      _step(lim, f, cc_enabled=True, vEgo=45.0 * MPH_TO_MS, observed_mph=60.0, est_power_w=50_000.0)
    self.assertEqual(lim._request_honored, 2, "pinned set speed past ACK window must read as ignored")


class TestIter16aDecelWindowEnforcement(unittest.TestCase):
  """B3: SET-down enforces the rolling window during lead-follow decel (was deferred)."""

  def _drive_decel(self, lim, vEgo_mph, observed_mph, frames, brake=False, start=30):
    """Drive with negative accel_demand (abasis) to trigger MODE_DECEL."""
    saw_set = False
    f = start
    for _ in range(frames):
      btn, _a = _step(lim, f, cc_enabled=True, vEgo=vEgo_mph * MPH_TO_MS,
                      observed_mph=observed_mph, abasis=-1.5, brake=brake)
      if btn == Buttons.SET_DECEL:
        saw_set = True
      f += 1
    return saw_set

  def test_set_emitted_during_decel_when_window_violated(self):
    lim = _make_limiter()
    _engage(lim, end_frame=30, vEgo_mph=35.0, observed_mph=50.0)
    # vEgo 20, set held 50 (delta 30 >> margin) while SCC decelerates for a lead.
    saw_set = self._drive_decel(lim, vEgo_mph=20.0, observed_mph=50.0, frames=300)
    self.assertTrue(saw_set, "B3: limiter must pull set DOWN during lead-follow decel windup")

  def test_driver_brake_still_defers_during_decel(self):
    lim = _make_limiter()
    _engage(lim, end_frame=30, vEgo_mph=35.0, observed_mph=50.0)
    saw_set = self._drive_decel(lim, vEgo_mph=20.0, observed_mph=50.0, frames=300, brake=True)
    self.assertFalse(saw_set, "driver brake must still suppress limiter SET")


class TestIter16aPowerDroopLogOnly(unittest.TestCase):
  """C1: below-vEgo power droop is LOG-ONLY / default-off this iteration."""

  def test_would_enter_but_not_active_by_default(self):
    lim = _make_limiter()   # _read_bool returns True only for EVLimiterEnabled
    _engage(lim, end_frame=30, vEgo_mph=60.0, observed_mph=62.0)
    # Sustained high control power, no driver input, moving.
    for f in range(30, 30 + 320):
      _step(lim, f, cc_enabled=True, vEgo=60.0 * MPH_TO_MS, observed_mph=62.0,
            est_power_w=55_000.0, est_power_control_w=55_000.0)
    self.assertTrue(lim._power_droop_would_enter, "sustained over-cap should arm the droop sim")
    self.assertFalse(lim._power_droop_active, "droop must NOT be active by default (log-only)")
    self.assertGreater(lim._power_droop_request_mph, 0.0)

  def test_active_only_when_param_enabled(self):
    lim = _make_limiter()
    # Enable the droop param explicitly.
    base = lim._read_bool
    lim._read_bool = lambda key, default: True if key in ("EVLimiterEnabled", "EvLimiterPowerDroopEnable") else base(key, default)
    _engage(lim, end_frame=30, vEgo_mph=60.0, observed_mph=62.0)
    for f in range(30, 30 + 320):
      _step(lim, f, cc_enabled=True, vEgo=60.0 * MPH_TO_MS, observed_mph=62.0,
            est_power_w=55_000.0, est_power_control_w=55_000.0)
    self.assertTrue(lim._power_droop_active, "droop should activate when the param is on")
    # Bounded below-vEgo request.
    from opendbc.sunnypilot.car.hyundai.ev_limiter import POWER_DROOP_MAX_MPH
    self.assertLessEqual(lim._power_droop_request_mph, POWER_DROOP_MAX_MPH + 1e-6)


if __name__ == "__main__":
  unittest.main()
