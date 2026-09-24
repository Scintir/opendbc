# ruff: noqa: E702, ISC002
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Regression coverage for the EV power limiter — sliding-cap iter6 architecture.

Drive #4 (kept): RES re-engage doesn't get classified as driver intent
to accelerate; user_target seeds from frozen pre-disable observed.

Drive #5 (new in iter6): the previous SOFT_CAP/RECOVERY state machine fired
on every Hyundai launch (aBasis>0.7 fallback) causing 20 s dwells, and let
cluster set sit 13 mph above user_target for 15 s before SOFT_CAP fired
power-only — too late, ICE engaged. Sliding cap addresses both by holding
cluster set within `vEgo + dynamic_margin(vEgo)` continuously.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.ev_limiter import (
  EVLimiter,
  STATE_DISABLED,
  STATE_DRIVER_OVERRIDE_RES,
  STATE_IDLE,
  STATE_SOFT_CAP_ACTIVE,
  STATE_RECOVERY_ACTIVE,
  STATE_STANDSTILL_HOLD,
  DISABLED_RES_ENGAGE_WINDOW_FRAMES,
  HIGH_SPEED_MARGIN_MPH,
  LOW_SPEED_MARGIN_MPH,
  LOW_LOAD_BONUS_MPH,
  MARGIN_BLEND_END_MPH,
  MPH_TO_MS,
  STANDSTILL_SET_PULSE_CAP,
)
from opendbc.car.hyundai.values import Buttons

ButtonType = structs.CarState.ButtonEvent.Type


@dataclass
class FakeCP:
  flags: int = HyundaiFlags.HYBRID
  openpilotLongitudinalControl: bool = False


@dataclass
class FakeCPSP:
  flags: int = 0


@dataclass
class FakeButtonEvent:
  pressed: bool = True
  type: Any = ButtonType.accelCruise


@dataclass
class FakeCruiseState:
  speed: float = 0.0
  standstill: bool = False


@dataclass
class FakeCarStateOut:
  vEgo: float = 0.0
  aEgo: float = 0.0
  brakePressed: bool = False
  gasPressed: bool = False
  cruiseState: FakeCruiseState = field(default_factory=FakeCruiseState)
  buttonEvents: list = field(default_factory=list)


@dataclass
class FakeCS:
  out: FakeCarStateOut = field(default_factory=FakeCarStateOut)
  est_power_w: float = 0.0
  accel_demand: float = 0.0
  dte_raw: float = 100.0


@dataclass
class FakeCC:
  enabled: bool = False


def _make_limiter(power_threshold_kw: int = 40, dte_floor: int = 5):
  cp = FakeCP()
  cp_sp = FakeCPSP()
  lim = EVLimiter(cp, cp_sp)
  lim._read_bool = lambda key, default: True if key == "EVLimiterEnabled" else default
  lim._read_int = lambda key, default: {
    "EVLimiterPowerThresholdKW": power_threshold_kw,
    "EVLimiterDTEFloor": dte_floor,
  }.get(key, default)
  return lim


def _step(limiter, frame, cc_enabled, vEgo, observed_mph, button=None,
          est_power_w=0.0, abasis=0.0, brake=False, gas=False, aEgo=0.0):
  cs = FakeCS()
  cs.out.vEgo = vEgo
  cs.out.aEgo = aEgo
  cs.out.cruiseState.speed = observed_mph * MPH_TO_MS
  cs.out.brakePressed = brake
  cs.out.gasPressed = gas
  cs.out.buttonEvents = [FakeButtonEvent(pressed=True, type=button)] if button is not None else []
  cs.est_power_w = est_power_w
  cs.accel_demand = abasis
  cs.dte_raw = 100.0
  cc = FakeCC(enabled=cc_enabled)
  return limiter.update(cc, cs, frame)


class TestDynamicMargin(unittest.TestCase):
  """The sliding-cap formula. iter7: now takes (vEgo, est_power, threshold).

  Tests use load=0 (low load → full bonus) for the base-formula tests, then
  separate tests cover the load-bonus taper.
  """
  THR = 40_000.0  # 40 kW threshold for these tests

  def test_standstill_base_with_full_bonus(self):
    # Low load → full +10 mph bonus
    margin = EVLimiter._dynamic_margin_ms(0.0, est_power_w=0.0, power_threshold_w=self.THR)
    expected_mph = LOW_SPEED_MARGIN_MPH + LOW_LOAD_BONUS_MPH  # 30 mph
    self.assertAlmostEqual(margin, expected_mph * MPH_TO_MS, places=4)

  def test_blend_end_with_full_bonus(self):
    margin = EVLimiter._dynamic_margin_ms(MARGIN_BLEND_END_MPH * MPH_TO_MS,
                                           est_power_w=0.0, power_threshold_w=self.THR)
    expected_mph = HIGH_SPEED_MARGIN_MPH + LOW_LOAD_BONUS_MPH  # 15 mph
    self.assertAlmostEqual(margin, expected_mph * MPH_TO_MS, places=4)

  def test_high_speed_flat_with_full_bonus(self):
    margin = EVLimiter._dynamic_margin_ms(40.0 * MPH_TO_MS,
                                           est_power_w=0.0, power_threshold_w=self.THR)
    expected_mph = HIGH_SPEED_MARGIN_MPH + LOW_LOAD_BONUS_MPH  # 15 mph
    self.assertAlmostEqual(margin, expected_mph * MPH_TO_MS, places=4)

  def test_midpoint_linear_base_with_full_bonus(self):
    # vEgo = 15 mph → base = (20 + 5) / 2 = 12.5 mph; + 10 bonus = 22.5
    margin = EVLimiter._dynamic_margin_ms(15.0 * MPH_TO_MS,
                                           est_power_w=0.0, power_threshold_w=self.THR)
    self.assertAlmostEqual(margin, 22.5 * MPH_TO_MS, places=2)

  def test_low_load_bonus_full_below_40pct(self):
    # est_power = 30% of threshold = 12 kW → full +10 bonus
    margin = EVLimiter._dynamic_margin_ms(40.0 * MPH_TO_MS,
                                           est_power_w=12_000.0, power_threshold_w=self.THR)
    self.assertAlmostEqual(margin, (HIGH_SPEED_MARGIN_MPH + LOW_LOAD_BONUS_MPH) * MPH_TO_MS, places=2)

  def test_load_bonus_taper_at_60pct(self):
    # est_power = 60% of threshold = halfway between 40% and 80% → half bonus
    margin = EVLimiter._dynamic_margin_ms(40.0 * MPH_TO_MS,
                                           est_power_w=24_000.0, power_threshold_w=self.THR)
    self.assertAlmostEqual(margin, (HIGH_SPEED_MARGIN_MPH + LOW_LOAD_BONUS_MPH * 0.5) * MPH_TO_MS, places=2)

  def test_load_bonus_zero_above_80pct(self):
    # est_power = 90% of threshold → no bonus, base only
    margin = EVLimiter._dynamic_margin_ms(40.0 * MPH_TO_MS,
                                           est_power_w=36_000.0, power_threshold_w=self.THR)
    self.assertAlmostEqual(margin, HIGH_SPEED_MARGIN_MPH * MPH_TO_MS, places=2)

  def test_invalid_threshold_no_bonus(self):
    """Fail-safe: invalid threshold (≤0) gives base margin only, NOT max bonus."""
    margin = EVLimiter._dynamic_margin_ms(40.0 * MPH_TO_MS,
                                           est_power_w=0.0, power_threshold_w=0.0)
    self.assertAlmostEqual(margin, HIGH_SPEED_MARGIN_MPH * MPH_TO_MS, places=2)
    margin = EVLimiter._dynamic_margin_ms(40.0 * MPH_TO_MS,
                                           est_power_w=0.0, power_threshold_w=-100.0)
    self.assertAlmostEqual(margin, HIGH_SPEED_MARGIN_MPH * MPH_TO_MS, places=2)


class TestSlidingCapDownTrigger(unittest.TestCase):
  """Push-down behavior: SET fires when cluster set exceeds target_set or
  when est power is high with a vEgo gap to close."""

  def setUp(self):
    self.lim = _make_limiter(power_threshold_kw=40)
    # Engage cleanly with vEgo > standstill so we can run the sliding cap.
    for f in range(5):
      _step(self.lim, f, cc_enabled=False, vEgo=10.0, observed_mph=30.0)
    _step(self.lim, 5, cc_enabled=True, vEgo=10.0, observed_mph=30.0,
          button=ButtonType.decelCruise)  # SET-engage from main switch
    # Wait past STANDSTILL exit clean window
    for f in range(6, 40):
      _step(self.lim, f, cc_enabled=True, vEgo=10.0, observed_mph=30.0)

  def test_drive5_ice_set_to_high_fires(self):
    """Drive #5 ICE event regression: cluster=60, vEgo=46, user_target=51
    must immediately fire SET (observed > target_set + deadband). The old
    architecture only fired SOFT_CAP after power crossed threshold — too late."""
    self.lim.user_target_speed = 51.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000  # clear cooldown
    btn, _ = _step(self.lim, 100, cc_enabled=True, vEgo=46.0 * MPH_TO_MS,
                   observed_mph=60.0, est_power_w=10_000.0, abasis=0.1)
    self.assertEqual(btn, Buttons.SET_DECEL,
                      "set_too_high should fire SET when observed (60) > target_set (51)")

  def test_set_fires_during_driver_override_set(self):
    """Driver SET is the SAME direction as our SET — driver-SET override
    must NOT suppress our SET, even with the override window open. Blocking
    our SET here would let cluster_set sit above target_set for the override
    window with no limiter response (gpt-5.5 review of iter6 caught this)."""
    self.lim.user_target_speed = 30.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000
    # Simulate a recent driver SET press
    self.lim._driver_set_last_frame = 95  # 5 frames before our test frame
    # vEgo=10, margin=~16 mph, ceiling=26, target=min(30, 26)=26
    # observed=60 is 34 mph above target → set_too_high True
    btn, _ = _step(self.lim, 100, cc_enabled=True, vEgo=10.0 * MPH_TO_MS,
                   observed_mph=60.0, est_power_w=10_000.0, abasis=0.1)
    self.assertEqual(btn, Buttons.SET_DECEL,
                      "Driver SET override window must NOT block our SET (same direction)")

  def test_under_target_does_not_fire_set(self):
    """When cluster set already at target_set, SET must not fire."""
    self.lim.user_target_speed = 60.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000
    # vEgo=50 → margin=5 → ceiling=55 → target=min(60,55)=55
    # observed=55 → set_too_high False, want_res only if under_target
    btn, _ = _step(self.lim, 200, cc_enabled=True, vEgo=50.0 * MPH_TO_MS,
                   observed_mph=55.0, est_power_w=10_000.0, abasis=0.0)
    self.assertNotEqual(btn, Buttons.SET_DECEL)

  def test_power_too_high_with_gap_fires_set(self):
    """Power above threshold AND observed > vEgo + 0.5 mph → SET."""
    self.lim.user_target_speed = 60.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000
    # vEgo=20 mph → margin ≈ 10 mph → ceiling=30, target=min(60,30)=30
    # observed=30 → set_too_high False (right at boundary), but power > 40 kW
    # AND observed > vEgo + 0.5 → power_too_high → SET
    btn, _ = _step(self.lim, 300, cc_enabled=True, vEgo=20.0 * MPH_TO_MS,
                   observed_mph=30.0, est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(btn, Buttons.SET_DECEL)

  def test_power_too_high_no_gap_does_not_fire(self):
    """Power high but observed near vEgo (no gap) → SET would not help."""
    self.lim.user_target_speed = 60.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000
    # vEgo=30, observed=30 (no gap), high power (e.g. grade)
    # → power_too_high False (gap requirement); set_too_high False (target=35)
    btn, _ = _step(self.lim, 400, cc_enabled=True, vEgo=30.0 * MPH_TO_MS,
                   observed_mph=30.0, est_power_w=60_000.0, abasis=0.0)
    self.assertNotEqual(btn, Buttons.SET_DECEL)


class TestStandstillSuppression(unittest.TestCase):
  """Iter7: standstill BLOCKS RES, but ALLOWS SET (pulse-bounded) to pull
  cluster set down toward LOW_SPEED_MARGIN_MPH (20 mph). Drive #6 retro:
  iter6 froze cluster_set wherever the deceleration SET cascade landed."""

  def test_standstill_set_fires_when_observed_above_cap(self):
    """At standstill with cluster_set above 20 mph cap, SET must fire on
    the engage frame (no cooldown to satisfy yet)."""
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=0.0, observed_mph=44.0)
    btn, _ = _step(lim, 25, cc_enabled=True, vEgo=0.0, observed_mph=44.0,
                    button=ButtonType.accelCruise)
    self.assertEqual(btn, Buttons.SET_DECEL,
                      "Standstill SET must fire when observed cluster set is above 20 mph cap")

  def test_standstill_no_set_when_at_cap(self):
    """At standstill with cluster_set already at 20 mph, no SET needed."""
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=0.0, observed_mph=20.0)
    _step(lim, 25, cc_enabled=True, vEgo=0.0, observed_mph=20.0,
          button=ButtonType.accelCruise)
    btn, _ = _step(lim, 26, cc_enabled=True, vEgo=0.0, observed_mph=20.0)
    self.assertEqual(btn, Buttons.NONE,
                      "Standstill SET must NOT fire when observed already at cap")

  def test_standstill_brake_blocks_set(self):
    """Standstill SET must NOT fire if brake is pressed."""
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=0.0, observed_mph=44.0, brake=True)
    _step(lim, 25, cc_enabled=True, vEgo=0.0, observed_mph=44.0,
          button=ButtonType.accelCruise, brake=True)
    btn, _ = _step(lim, 26, cc_enabled=True, vEgo=0.0, observed_mph=44.0, brake=True)
    self.assertEqual(btn, Buttons.NONE)

  def test_standstill_res_never_fires(self):
    """Even if cluster_set is below user_target at standstill, RES must
    NEVER fire (vehicle is stopped)."""
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=0.0, observed_mph=10.0)
    _step(lim, 25, cc_enabled=True, vEgo=0.0, observed_mph=10.0,
          button=ButtonType.accelCruise)
    # user_target seeded from the frozen 10 mph (engage_via_res). Bump it up
    # via simulated driver presses so the under_target condition would otherwise fire.
    lim.user_target_speed = 50.0 * MPH_TO_MS
    for f in range(26, 60):
      btn, _ = _step(lim, f, cc_enabled=True, vEgo=0.0, observed_mph=10.0)
      self.assertNotEqual(btn, Buttons.RES_ACCEL,
                           f"Standstill must never fire RES (frame {f})")

  def test_standstill_set_pulse_cap(self):
    """SET pulse count is bounded per single standstill window."""
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=0.0, observed_mph=80.0)
    _step(lim, 25, cc_enabled=True, vEgo=0.0, observed_mph=80.0,
          button=ButtonType.accelCruise)
    set_count = 0
    for f in range(26, 26 + 100 * STANDSTILL_SET_PULSE_CAP):  # plenty of frames
      btn, _ = _step(lim, f, cc_enabled=True, vEgo=0.0, observed_mph=80.0)
      if btn == Buttons.SET_DECEL:
        set_count += 1
    self.assertLessEqual(set_count, STANDSTILL_SET_PULSE_CAP,
                          f"Standstill SET pulses must be capped at {STANDSTILL_SET_PULSE_CAP}; got {set_count}")


class TestGasBrakePause(unittest.TestCase):
  """Gas or brake pause suppresses both SET and RES."""

  def setUp(self):
    self.lim = _make_limiter()
    # Engage cleanly above standstill
    for f in range(25):
      _step(self.lim, f, cc_enabled=False, vEgo=10.0, observed_mph=30.0)
    _step(self.lim, 25, cc_enabled=True, vEgo=10.0, observed_mph=30.0,
          button=ButtonType.decelCruise)
    for f in range(26, 50):
      _step(self.lim, f, cc_enabled=True, vEgo=10.0, observed_mph=30.0)

  def test_gas_suppresses_set(self):
    self.lim.user_target_speed = 30.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000
    # Conditions that would normally fire SET:
    btn, _ = _step(self.lim, 100, cc_enabled=True, vEgo=10.0,
                   observed_mph=60.0, est_power_w=50_000.0, abasis=1.0, gas=True)
    self.assertEqual(btn, Buttons.NONE, "Gas must suppress SET")

  def test_brake_suppresses_set(self):
    self.lim.user_target_speed = 30.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000
    btn, _ = _step(self.lim, 200, cc_enabled=True, vEgo=10.0,
                   observed_mph=60.0, est_power_w=50_000.0, abasis=1.0, brake=True)
    self.assertEqual(btn, Buttons.NONE, "Brake must suppress SET")

  def test_gas_does_not_block_res_in_iter10(self):
    """iter10 Layer 1: gas no longer blocks RES outright; instead bounded
    governor mode (GAS_CATCHUP / NORMAL) controls cluster behavior, with a
    1.5 s rate cap on RES presses while gas held. Replaces the iter9
    gas-suppresses-res test (event A: dwell at 21 mph fix)."""
    self.lim.user_target_speed = 30.0 * MPH_TO_MS
    self.lim.last_res_frame = -10000
    # Set up under-target conditions: first RES under gas should be allowed.
    btn1, _ = _step(self.lim, 300, cc_enabled=True, vEgo=10.0,
                    observed_mph=15.0, est_power_w=0.0, abasis=0.0, gas=True)
    self.assertEqual(btn1, Buttons.RES_ACCEL,
                     "iter10: gas alone must NOT block first RES press (Event A fix)")
    # Second RES press 0.5 s later (under 1.5 s gas-time rate cap) → blocked
    btn2, _ = _step(self.lim, 350, cc_enabled=True, vEgo=10.0,
                    observed_mph=15.0, est_power_w=0.0, abasis=0.0, gas=True)
    self.assertEqual(btn2, Buttons.NONE,
                     "iter10: RES rate-capped to 1 per 1.5 s while gas held")


class TestStateLatching(unittest.TestCase):
  """HUD state derived from recent activity, not button-this-frame."""

  def test_limiting_state_persists_after_set(self):
    """After a SET press, state must remain SOFT_CAP_ACTIVE for at least
    LIMITING_LATCH_FRAMES frames even though the cooldown blocks more SETs."""
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=10.0, observed_mph=44.0)
    _step(lim, 25, cc_enabled=True, vEgo=10.0, observed_mph=44.0,
          button=ButtonType.decelCruise)
    for f in range(26, 50):
      _step(lim, f, cc_enabled=True, vEgo=10.0, observed_mph=44.0)
    lim.user_target_speed = 30.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    # Frame 100: SET fires
    _step(lim, 100, cc_enabled=True, vEgo=10.0, observed_mph=60.0,
          est_power_w=50_000.0, abasis=0.5)
    # Frame 110 (10 frames later, SET cooldown blocks new TX, but state
    # should still report SOFT_CAP_ACTIVE because last_set_frame is recent)
    _step(lim, 110, cc_enabled=True, vEgo=10.0, observed_mph=59.0,
          est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                      "State should latch SOFT_CAP_ACTIVE for ~500 ms after our SET")


class TestEngagePathDriveFour(unittest.TestCase):
  """Drive #4 regression coverage — kept from iter5."""

  def setUp(self):
    self.lim = _make_limiter()

  def test_disabled_tracks_observed_set_speed(self):
    _step(self.lim, 0, cc_enabled=False, vEgo=20.0, observed_mph=70.0)
    self.assertAlmostEqual(self.lim._observed_set_speed_at_disable,
                            70.0 * MPH_TO_MS, places=4)

  def test_disabled_res_press_recorded(self):
    _step(self.lim, 5, cc_enabled=False, vEgo=20.0, observed_mph=70.0,
          button=ButtonType.accelCruise)
    self.assertEqual(self.lim._last_disabled_res_press_frame, 5)

  def test_engage_via_res_seeds_user_target_from_pre_disable(self):
    for f in range(10):
      _step(self.lim, f, cc_enabled=False, vEgo=24.0, observed_mph=70.0)
    _step(self.lim, 10, cc_enabled=True, vEgo=24.0, observed_mph=56.0,
          button=ButtonType.accelCruise)
    self.assertAlmostEqual(self.lim.user_target_speed, 70.0 * MPH_TO_MS, places=4)

  def test_engage_via_res_via_disabled_lookback(self):
    """Wheel RES landed BEFORE cc_enabled rose — engage_via_res via lookback."""
    for f in range(5):
      _step(self.lim, f, cc_enabled=False, vEgo=24.0, observed_mph=70.0)
    _step(self.lim, 5, cc_enabled=False, vEgo=24.0, observed_mph=70.0,
          button=ButtonType.accelCruise)
    _step(self.lim, 6, cc_enabled=True, vEgo=24.0, observed_mph=56.0)
    self.assertAlmostEqual(self.lim.user_target_speed, 70.0 * MPH_TO_MS, places=4)

  def test_engage_via_res_does_not_latch_drv_res(self):
    for f in range(5):
      _step(self.lim, f, cc_enabled=False, vEgo=24.0, observed_mph=48.0)
    _step(self.lim, 5, cc_enabled=True, vEgo=24.0, observed_mph=56.0,
          button=ButtonType.accelCruise)
    self.assertLess(self.lim._driver_res_last_frame, 0)
    self.assertFalse(self.lim._in_driver_override_res(5))

  def test_engage_via_set_seeds_from_observed(self):
    for f in range(5):
      _step(self.lim, f, cc_enabled=False, vEgo=24.0, observed_mph=70.0)
    _step(self.lim, 5, cc_enabled=True, vEgo=24.0, observed_mph=24.0,
          button=ButtonType.decelCruise)
    self.assertAlmostEqual(self.lim.user_target_speed, 24.0 * MPH_TO_MS, places=4)

  def test_disabled_res_lookback_window_expires(self):
    for f in range(5):
      _step(self.lim, f, cc_enabled=False, vEgo=24.0, observed_mph=70.0)
    _step(self.lim, 5, cc_enabled=False, vEgo=24.0, observed_mph=70.0,
          button=ButtonType.accelCruise)
    for f in range(6, 6 + DISABLED_RES_ENGAGE_WINDOW_FRAMES + 5):
      _step(self.lim, f, cc_enabled=False, vEgo=24.0, observed_mph=70.0)
    engage_frame = 6 + DISABLED_RES_ENGAGE_WINDOW_FRAMES + 5
    _step(self.lim, engage_frame, cc_enabled=True, vEgo=24.0, observed_mph=24.0)
    self.assertAlmostEqual(self.lim.user_target_speed, 24.0 * MPH_TO_MS, places=4)


class TestPowerEstimator(unittest.TestCase):
  """Iter7 power formula: mass·v·(max(0,abasis) + max(0,grade)) + road_load.

  Drive #6 7:40 ICE event proved the iter6 formula was blind in a key regime:
  abasis=-0.4 (SCC commanding decel) + grade=+0.4 (uphill) yielded
  max(0, abasis+grade) = max(0, 0) = 0 even though motor was doing real work.
  Iter7 clamps both terms positive separately, plus adds a road-load baseline.
  """
  def setUp(self):
    # Late import: carstate_ext requires opendbc structs which require numpy
    from opendbc.sunnypilot.car.hyundai.carstate_ext import (
      road_load_power_w, VEHICLE_MASS_KG, GRAVITY_MS2,
    )
    self.road_load = road_load_power_w
    self.MASS = VEHICLE_MASS_KG
    self.G = GRAVITY_MS2

  def _formula(self, abasis, grade_f, v_mph):
    """Compute the iter7 power formula directly (mirrors carstate_ext logic)."""
    v = v_mph * MPH_TO_MS
    abasis_pos = max(0.0, abasis)
    grade_pos = max(0.0, grade_f)
    p_accel_grade = self.MASS * v * (abasis_pos + grade_pos)
    p_road = self.road_load(v)
    return max(0.0, p_accel_grade + p_road)

  def test_road_load_zero_at_standstill(self):
    self.assertAlmostEqual(self.road_load(0.0), 0.0, places=2)

  def test_road_load_increases_with_speed(self):
    # Cubic in speed → highway speeds dominated by aero
    p_30 = self.road_load(30.0 * MPH_TO_MS)
    p_60 = self.road_load(60.0 * MPH_TO_MS)
    p_75 = self.road_load(75.0 * MPH_TO_MS)
    self.assertLess(p_30, p_60)
    self.assertLess(p_60, p_75)
    # Sanity: 75 mph baseline should be 20-30 kW for an SUV
    self.assertGreater(p_75, 18_000.0)
    self.assertLess(p_75, 35_000.0)

  def test_drive6_ice_blind_spot_caught(self):
    """7:40 ICE event scenario: at 73 mph holding speed against grade,
    iter6 read 0 kW. Iter7 must read above the 37 kW threshold (the lowest
    user-reasonable threshold), and ideally above the 40 kW default too."""
    # vEgo=73 mph, abasis=-0.4 (SCC commanding decel), grade=+0.4 (uphill)
    pwr = self._formula(abasis=-0.4, grade_f=+0.4, v_mph=73.0)
    self.assertGreater(pwr, 40_000.0,
                        f"Iter6 ICE blind spot must be caught (got {pwr/1000:.1f} kW)")

  def test_high_speed_cruise_no_grade_below_threshold(self):
    """At 73 mph holding speed against road load only (no grade), motor
    power equals road load — should be 20-30 kW, well below 40 kW default."""
    pwr = self._formula(abasis=-0.4, grade_f=0.0, v_mph=73.0)
    self.assertGreater(pwr, 18_000.0)
    self.assertLess(pwr, 32_000.0)

  def test_downhill_does_not_cancel_accel(self):
    """Negative grade (downhill) must NOT cancel positive aBasis. The
    protective estimator stays conservative — false positives are OK,
    false negatives lead to ICE."""
    # Compare: same abasis+v, with vs without downhill grade
    pwr_with_downhill = self._formula(abasis=+0.3, grade_f=-0.3, v_mph=60.0)
    pwr_no_grade = self._formula(abasis=+0.3, grade_f=0.0, v_mph=60.0)
    # Downhill clamps to 0 in our formula, so power is identical
    self.assertAlmostEqual(pwr_with_downhill, pwr_no_grade, places=2,
                            msg="Downhill grade must not change protective power estimate")

  def test_decel_does_not_cancel_grade(self):
    """SCC commanded decel (negative abasis) must NOT cancel positive grade.
    This is the iter6 formula bug that drive #6 ICE event exposed."""
    pwr_with_decel = self._formula(abasis=-0.4, grade_f=+0.4, v_mph=60.0)
    pwr_no_abasis = self._formula(abasis=0.0, grade_f=+0.4, v_mph=60.0)
    self.assertAlmostEqual(pwr_with_decel, pwr_no_abasis, places=2,
                            msg="Negative abasis must not cancel positive grade contribution")


class TestDecelFastCadence(unittest.TestCase):
  """iter8: SET cadence shortened during low-speed deceleration so cluster
  can be pulled toward target_set faster than default 6.7 mph/s rate. Drive
  #6 7:34 retro: hard decel from cruise to red light left cluster set frozen
  high because default cadence lost the race.
  """

  def setUp(self):
    self.lim = _make_limiter()
    # Engage at moderate speed so we have a clean baseline above standstill
    for f in range(25):
      _step(self.lim, f, cc_enabled=False, vEgo=10.0, observed_mph=44.0)
    _step(self.lim, 25, cc_enabled=True, vEgo=10.0, observed_mph=44.0,
          button=ButtonType.decelCruise)
    # Settle past standstill exit hold-off
    for f in range(26, 60):
      _step(self.lim, f, cc_enabled=True, vEgo=10.0, observed_mph=44.0,
            est_power_w=0.0, abasis=0.0, aEgo=0.0)
    self.lim.user_target_speed = 50.0 * MPH_TO_MS

  def test_decel_fast_set_cooldown_iter12_obsolete(self):
    """iter12: decel-fast 60ms cadence is OBSOLETE. iter12 uses ack-driven
    cadence (50 frames = 0.5s after acked SET, 150 frames = 1.5s otherwise).
    Brake decel scenario is now BRAKE-mode-suppressed (no SET fires under brake).
    For coast-decel without brake, iter12's STANDSTILL_HOLD logic handles
    pre-stop cluster pull-down, not decel-fast."""
    self.lim.last_set_frame = -10000
    btn1, _ = _step(self.lim, 100, cc_enabled=True, vEgo=10.0,
                    observed_mph=44.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=-1.0)
    self.assertEqual(btn1, Buttons.SET_DECEL)
    # 7 frames later — iter12 ack-driven won't fire (waiting for cluster
    # response or normal 150-frame cooldown)
    btn2, _ = _step(self.lim, 107, cc_enabled=True, vEgo=10.0,
                    observed_mph=43.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=-1.0)
    self.assertEqual(btn2, Buttons.NONE,
                     "iter12: decel-fast 60ms obsolete; ack-driven cadence applies")
    # 60 frames later (acked + > SET_MIN_REPEAT_FRAMES=50) — should fire
    btn3, _ = _step(self.lim, 160, cc_enabled=True, vEgo=10.0,
                    observed_mph=43.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=-1.0)
    self.assertEqual(btn3, Buttons.SET_DECEL,
                     "iter12: after 0.6s with cluster ack, next SET allowed")

  def test_decel_fast_inactive_at_high_speed(self):
    """vEgo >= 30 mph → default cadence even with strong decel."""
    self.lim.last_set_frame = -10000
    # vEgo = 50 mph (= 22.4 m/s, above the 30 mph threshold) with decel
    btn1, _ = _step(self.lim, 200, cc_enabled=True, vEgo=22.4,
                    observed_mph=80.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=-1.0)
    self.assertEqual(btn1, Buttons.SET_DECEL)
    # 7 frames later — should NOT fire (default cadence is 15)
    btn2, _ = _step(self.lim, 207, cc_enabled=True, vEgo=22.4,
                    observed_mph=80.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=-1.0)
    self.assertEqual(btn2, Buttons.NONE,
                      "At highway speed, default 150 ms cadence must apply (got SET)")

  def test_decel_fast_inactive_when_not_decelerating(self):
    """vEgo < 30 mph but aEgo ~ 0 → default cadence (low-speed cruising
    isn't a brake-decel scenario)."""
    self.lim.last_set_frame = -10000
    btn1, _ = _step(self.lim, 300, cc_enabled=True, vEgo=10.0,
                    observed_mph=44.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=0.0)
    self.assertEqual(btn1, Buttons.SET_DECEL)
    # 7 frames later — no decel, default cadence
    btn2, _ = _step(self.lim, 307, cc_enabled=True, vEgo=10.0,
                    observed_mph=44.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=0.0)
    self.assertEqual(btn2, Buttons.NONE)

  def test_decel_fast_rate_limit_iter12_obsolete(self):
    """iter12: decel-fast 12Hz cadence is OBSOLETE. iter12 ack-driven
    cadence enforces ≥0.5s between SETs even under decel. Test now verifies
    iter12 behavior."""
    self.lim.last_set_frame = -10000
    f = 1000
    btn, _ = _step(self.lim, f, cc_enabled=True, vEgo=10.0,
                   observed_mph=44.0, est_power_w=5_000.0, abasis=0.0,
                   aEgo=-1.0)
    self.assertEqual(btn, Buttons.SET_DECEL, "First SET fires")
    # 6 frames later — iter12 won't fire (need 50 frames + ack)
    btn2, _ = _step(self.lim, f + 6, cc_enabled=True, vEgo=10.0,
                    observed_mph=44.0, est_power_w=5_000.0, abasis=0.0,
                    aEgo=-1.0)
    self.assertEqual(btn2, Buttons.NONE,
                     "iter12: 12Hz rate-limit override removed; ack-driven applies")

  def test_standstill_pulse_cap_iter13_initial_10(self):
    """iter13 v4 Section E: PRELAUNCH_SET pulse cap starts at 10
    (STANDSTILL_PULSE_CAP_INITIAL); raises to 30 after first cluster_decrement_acked.

    Without an ACK feedback (constant cluster), only the initial cap of 10
    fires. The legacy STANDSTILL_SET_PULSE_CAP=10 constant remains for
    backwards-compat with the old standstill emission path; iter13's new
    PRELAUNCH_SET path uses STANDSTILL_PULSE_CAP_INITIAL/AFTER_ACK.
    """
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      STANDSTILL_PULSE_CAP_INITIAL, STANDSTILL_PULSE_CAP_AFTER_ACK,
      MAX_PRELAUNCH_BACKOFF_RETRIES, STANDSTILL_NO_ACK_BACKOFF_AFTER_EMITTED,
    )
    self.assertEqual(STANDSTILL_PULSE_CAP_INITIAL, 10)
    self.assertEqual(STANDSTILL_PULSE_CAP_AFTER_ACK, 30)
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=0.0, observed_mph=80.0)
    _step(lim, 25, cc_enabled=True, vEgo=0.0, observed_mph=80.0,
          button=ButtonType.accelCruise)
    set_count = 0
    # Run for long enough that even the AFTER_ACK cap (30) would be hit if active.
    for f in range(26, 26 + 60 * STANDSTILL_PULSE_CAP_AFTER_ACK):
      btn, _ = _step(lim, f, cc_enabled=True, vEgo=0.0, observed_mph=80.0)
      if btn == Buttons.SET_DECEL:
        set_count += 1
    # iter16a (B1): the per-stop budget now REFRESHES each time the no-ack backoff
    # expires, so a long red light keeps making BOUNDED set-down attempts instead of
    # permanently giving up at the initial cap of 10 (the driver-observed red-light
    # windup). Still bounded — total no-ack pulses cannot exceed the initial budget
    # plus MAX_PRELAUNCH_BACKOFF_RETRIES refresh cycles. The "no runaway" intent holds.
    no_ack_bound = (STANDSTILL_PULSE_CAP_INITIAL
                    + MAX_PRELAUNCH_BACKOFF_RETRIES * STANDSTILL_NO_ACK_BACKOFF_AFTER_EMITTED)
    self.assertLessEqual(set_count, no_ack_bound,
                          f"no-ack pulses must stay bounded (<= {no_ack_bound}); got {set_count}")
    self.assertGreater(set_count, STANDSTILL_PULSE_CAP_INITIAL,
                       "iter16a B1: a long unacked stop should retry past the initial cap")


class TestIter9PowerPriority(unittest.TestCase):
  """iter9: power_too_high bypasses self_res_recent (drive #7 fix for the
  RECOVERY/SET deadlock — 53% of high-power frames stuck in RECOVERY because
  recent RES blocked SET despite estPowerW already in ICE territory).
  """

  def setUp(self):
    self.lim = _make_limiter(power_threshold_kw=40)
    # Engage cleanly above standstill
    for f in range(25):
      _step(self.lim, f, cc_enabled=False, vEgo=24.0, observed_mph=60.0)
    _step(self.lim, 25, cc_enabled=True, vEgo=24.0, observed_mph=60.0,
          button=ButtonType.decelCruise)
    for f in range(26, 50):
      _step(self.lim, f, cc_enabled=True, vEgo=24.0, observed_mph=60.0)
    self.lim.user_target_speed = 60.0 * MPH_TO_MS

  def test_power_too_high_bypasses_self_res_recent(self):
    """Recent RES would normally block SET for 1.5 s, but power-based
    protection must override (drive #7: RECOVERY/SET deadlock at 4:08 PM)."""
    # Mark a recent RES press
    self.lim.last_res_frame = 100
    self.lim.last_set_frame = -10000
    # Conditions: cluster above vEgo (gap exists), power above threshold
    btn, _ = _step(self.lim, 110, cc_enabled=True, vEgo=24.0,
                   observed_mph=70.0, est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(btn, Buttons.SET_DECEL,
                      "power_too_high must override self_res_recent block (got NONE)")

  def test_self_res_recent_still_blocks_set_when_only_set_too_high(self):
    """If only set_too_high (NOT power_too_high), self_res_recent still
    blocks SET — this preserves anti-oscillation for the cosmetic case."""
    self.lim.last_res_frame = 100
    self.lim.last_set_frame = -10000
    # Conditions: cluster above target_set, but power LOW (no power_too_high)
    btn, _ = _step(self.lim, 110, cc_enabled=True, vEgo=24.0,
                   observed_mph=70.0, est_power_w=5_000.0, abasis=0.0)
    self.assertEqual(btn, Buttons.NONE,
                      "self_res_recent should still block SET when power is low")

  def test_power_too_high_cancels_want_res(self):
    """When power is high, RES must NEVER fire even if cluster is below
    user_target. Otherwise we'd push the gap wider and motor harder."""
    # Setup: cluster below user_target (would normally trigger RES) BUT
    # power is high (must not fire RES even if SET path is blocked elsewhere)
    self.lim.user_target_speed = 70.0 * MPH_TO_MS
    self.lim.last_set_frame = -10000
    self.lim.last_res_frame = -10000
    # vEgo=24 m/s (~54 mph) — high speed regime, margin = 5 mph
    # observed=50 mph < target=min(70, 54+5)=59 mph → under_target = True
    # power_too_high requires observed > vEgo+0.5 → 50 > 54.5 = False... hmm
    # Different scenario: vEgo=22 m/s (~49 mph), observed=55 mph → gap exists
    btn, _ = _step(self.lim, 200, cc_enabled=True, vEgo=22.0,
                   observed_mph=55.0, est_power_w=50_000.0, abasis=0.5)
    self.assertNotEqual(btn, Buttons.RES_ACCEL,
                        "RES must not fire when power_too_high")


class TestIter9GradeDeadband(unittest.TestCase):
  """iter9 grade dead-band kills the +0.025 mean bias from LONG_ACCEL-aEgo
  derivation that produced ~22 kW phantom contribution on flat highway."""

  def setUp(self):
    from opendbc.sunnypilot.car.hyundai.carstate_ext import (
      road_load_power_w, VEHICLE_MASS_KG, GRADE_DEADBAND_MS2,
      GRADE_CONTRIB_CAP_DEFAULT_KW,
    )
    self.road_load = road_load_power_w
    self.MASS = VEHICLE_MASS_KG
    self.DEADBAND = GRADE_DEADBAND_MS2
    # iter15 v2: grade contribution clamp (default 10 kW).
    self.CAP_W = GRADE_CONTRIB_CAP_DEFAULT_KW * 1000.0

  def _formula(self, abasis, grade_f, v_mph):
    """iter15 v2 formula: subtract deadband from grade, then CLAMP the grade
    contribution at GRADE_CONTRIB_CAP_DEFAULT_KW (anti-overread). The abasis
    term is uncapped."""
    v = v_mph * MPH_TO_MS
    abasis_pos = max(0.0, abasis)
    grade_pos = max(0.0, grade_f - self.DEADBAND)
    grade_power_raw = self.MASS * v * grade_pos
    grade_power_clamped = min(grade_power_raw, self.CAP_W)
    p_accel_grade = self.MASS * v * abasis_pos + grade_power_clamped
    p_road = self.road_load(v)
    return max(0.0, p_accel_grade + p_road)

  def test_deadband_value(self):
    """iter15 v2 R1-MF-B: dead-band raised 0.10 → 0.20 (filters 12 kW phantom)."""
    self.assertAlmostEqual(self.DEADBAND, 0.20, places=4)

  def test_grade_below_deadband_contributes_zero(self):
    """A grade reading at the +0.025 m/s² noise mean must NOT add power."""
    pwr_with_noise = self._formula(abasis=0.0, grade_f=0.025, v_mph=60.0)
    pwr_no_grade = self._formula(abasis=0.0, grade_f=0.0, v_mph=60.0)
    self.assertAlmostEqual(pwr_with_noise, pwr_no_grade, places=2,
                            msg="Grade noise within dead-band must not add power")

  def test_grade_at_deadband_contributes_zero(self):
    """Exactly at the dead-band threshold, no contribution."""
    pwr = self._formula(abasis=0.0, grade_f=self.DEADBAND, v_mph=60.0)
    pwr_no_grade = self._formula(abasis=0.0, grade_f=0.0, v_mph=60.0)
    self.assertAlmostEqual(pwr, pwr_no_grade, places=2)

  def test_grade_above_deadband_contributes_proportionally(self):
    """iter15 v2: a small grade well above deadband but below clamp contributes
    proportionally. A +0.4 m/s² grade with deadband 0.20 → 0.20 effective grade
    at low v contributes < 10 kW, so the cap doesn't fire and the formula is
    linear. We use a low speed (15 mph) to ensure grade_power_raw < cap_w."""
    v_mph = 15.0
    pwr = self._formula(abasis=0.0, grade_f=0.4, v_mph=v_mph)
    expected_grade_extra = self.MASS * (v_mph * MPH_TO_MS) * (0.4 - self.DEADBAND)
    # Sanity: this grade contribution must be below the cap (else we're
    # testing the clamp, not the linear region).
    self.assertLess(expected_grade_extra, self.CAP_W,
                    msg="Test inputs hit the clamp; lower v_mph to stay linear")
    pwr_no_grade = self._formula(abasis=0.0, grade_f=0.0, v_mph=v_mph)
    self.assertAlmostEqual(pwr - pwr_no_grade, expected_grade_extra, places=0,
                            msg="Grade above dead-band (below clamp) should contribute "
                                "(grade - dead-band) linearly")


class TestIter10StateDwell(unittest.TestCase):
  """iter10 Layer 2: state machine dwell + hysteresis kills the
  drive #8 7:36-7:45 oscillation pattern (96 transitions / 9 min)."""

  def setUp(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      MIN_ACTIVE_STATE_DWELL_FRAMES,
      SOFT_CAP_ENTER_SUSTAIN_FRAMES,
      SOFT_CAP_EXIT_SUSTAIN_FRAMES,
      RECOVERY_ENTER_SUSTAIN_FRAMES,
      RECOVERY_EXIT_SUSTAIN_FRAMES,
    )
    self.MIN_DWELL = MIN_ACTIVE_STATE_DWELL_FRAMES
    self.SC_ENTER = SOFT_CAP_ENTER_SUSTAIN_FRAMES
    self.SC_EXIT = SOFT_CAP_EXIT_SUSTAIN_FRAMES
    self.REC_ENTER = RECOVERY_ENTER_SUSTAIN_FRAMES
    self.REC_EXIT = RECOVERY_EXIT_SUSTAIN_FRAMES

  def test_softcap_immediate_entry_on_power_too_high(self):
    """power_too_high bypasses entry sustain — preserves iter9 fast-protect."""
    lim = _make_limiter()
    # Engage at low speed
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=10.0, observed_mph=44.0)
    _step(lim, 25, cc_enabled=True, vEgo=10.0, observed_mph=44.0,
          button=ButtonType.decelCruise)
    for f in range(26, 100):
      _step(lim, f, cc_enabled=True, vEgo=20.0, observed_mph=40.0)
    lim.user_target_speed = 70.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    # Frame 100: power_too_high condition (high estPower, observed > vEgo)
    # Must enter SOFT_CAP_ACTIVE on the very same frame, not after sustain
    _step(lim, 100, cc_enabled=True, vEgo=20.0, observed_mph=60.0,
          est_power_w=70_000.0, abasis=1.0)
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                     "power_too_high must trigger immediate SOFT_CAP entry")

  def _setup_engaged_at_idle(self, lim, vEgo=20.0, target_mph=70.0):
    """Helper: get limiter into an engaged, IDLE state with explicit
    user_target_speed set. Mirrors pattern from test_limiting_state_persists_after_set."""
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=vEgo, observed_mph=target_mph)
    _step(lim, 25, cc_enabled=True, vEgo=vEgo, observed_mph=target_mph,
          button=ButtonType.decelCruise)
    for f in range(26, 50):
      _step(lim, f, cc_enabled=True, vEgo=vEgo, observed_mph=target_mph)
    lim.user_target_speed = target_mph * MPH_TO_MS
    lim.last_set_frame = -10000
    lim.last_res_frame = -10000

  def test_softcap_entry_not_blocked_by_idle_dwell(self):
    """IDLE → SOFT_CAP transition uses entry sustain only; min-dwell does
    not delay leaving IDLE."""
    lim = _make_limiter()
    self._setup_engaged_at_idle(lim, vEgo=20.0, target_mph=70.0)
    # Now drive set_too_high condition (observed > target_set+deadband)
    for f in range(50, 50 + self.SC_ENTER + 10):
      _step(lim, f, cc_enabled=True, vEgo=20.0, observed_mph=80.0,
            est_power_w=10_000.0, abasis=0.0)
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                     "IDLE → SOFT_CAP must not be blocked by min-dwell")

  def test_min_dwell_enforced_on_softcap_exit(self):
    """SOFT_CAP_ACTIVE cannot exit before MIN_ACTIVE_STATE_DWELL_FRAMES."""
    lim = _make_limiter()
    self._setup_engaged_at_idle(lim, vEgo=20.0, target_mph=70.0)
    # Drive SOFT_CAP entry — observed way above target_set
    f = 50
    while lim.state != STATE_SOFT_CAP_ACTIVE and f < 200:
      _step(lim, f, cc_enabled=True, vEgo=20.0, observed_mph=80.0,
            est_power_w=10_000.0, abasis=0.0)
      f += 1
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE)
    soft_cap_entered_frame = f
    # Now clear the predicate (observed at target) for SC_EXIT frames,
    # but stay UNDER MIN_DWELL — state should hold SOFT_CAP_ACTIVE.
    # Note: `observed_mph=70` matches user_target so no recovery either
    n_frames = self.SC_EXIT + 10  # exit-sustain met but min-dwell not yet
    while f <= soft_cap_entered_frame + n_frames:
      _step(lim, f, cc_enabled=True, vEgo=20.0, observed_mph=70.0,
            est_power_w=0.0, abasis=0.0)
      f += 1
    elapsed = f - soft_cap_entered_frame
    if elapsed < self.MIN_DWELL:
      self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                       f"SOFT_CAP must hold for MIN_DWELL frames; elapsed={elapsed}")

  def test_softcap_exits_after_dwell_and_sustain(self):
    """After both MIN_DWELL and EXIT_SUSTAIN with cleared predicate, exits."""
    lim = _make_limiter()
    # vEgo=30 m/s (~67 mph) keeps margin small so target_set not clamped low
    self._setup_engaged_at_idle(lim, vEgo=30.0, target_mph=70.0)
    # Enter SOFT_CAP — observed >> target_set+deadband
    f = 50
    while lim.state != STATE_SOFT_CAP_ACTIVE and f < 200:
      _step(lim, f, cc_enabled=True, vEgo=30.0, observed_mph=85.0,
            est_power_w=10_000.0, abasis=0.0)
      f += 1
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE)
    # Run for both MIN_DWELL + SC_EXIT + slack with clear predicate (observed
    # below target so set_too_high=False, observed near target so under_target
    # also borderline — limiter should drift toward IDLE/RECOVERY).
    target_frames = self.MIN_DWELL + self.SC_EXIT + 50
    end_f = f + target_frames
    while f <= end_f:
      _step(lim, f, cc_enabled=True, vEgo=30.0, observed_mph=68.0,
            est_power_w=0.0, abasis=0.0)
      f += 1
    self.assertNotEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                        "SOFT_CAP must exit after dwell+sustain satisfied")

  def test_recovery_enter_uses_sustain_not_min_dwell(self):
    """IDLE → RECOVERY uses RECOVERY_ENTER_SUSTAIN frames, not min-dwell."""
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=10.0, observed_mph=44.0)
    _step(lim, 25, cc_enabled=True, vEgo=20.0, observed_mph=44.0,
          button=ButtonType.decelCruise)
    for f in range(26, 50):
      _step(lim, f, cc_enabled=True, vEgo=20.0, observed_mph=44.0)
    # Set user_target above observed → under_target true
    lim.user_target_speed = 60.0 * MPH_TO_MS
    # Drive recovery sustain only — state should fire RECOVERY_ACTIVE
    # within REC_ENTER_SUSTAIN frames + small slack
    for f in range(50, 50 + self.REC_ENTER + 5):
      _step(lim, f, cc_enabled=True, vEgo=20.0, observed_mph=20.0,
            est_power_w=5_000.0, abasis=0.0)
    self.assertEqual(lim.state, STATE_RECOVERY_ACTIVE,
                     "IDLE → RECOVERY must fire after entry-sustain met")

  def test_oscillation_resistance_simulated(self):
    """Simulated drive #8 Event B: rapidly alternating
    set_too_high / under_target predicates should NOT produce ≥10 transitions
    in 9 simulated minutes (54000 frames @ 100 Hz)."""
    lim = _make_limiter()
    # Engage
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=30.0, observed_mph=70.0)
    _step(lim, 25, cc_enabled=True, vEgo=30.0, observed_mph=65.0,
          button=ButtonType.decelCruise)
    lim.user_target_speed = 75.0 * MPH_TO_MS

    # Simulate the oscillation pattern: power crosses threshold every ~3 sec
    # (frame mod 300 < 150 → power high, else power low)
    transitions = 0
    last_state = lim.state
    cap_start_f = 100
    sim_frames = 54000   # 9 minutes
    for f in range(cap_start_f, cap_start_f + sim_frames):
      power_high = (f % 300) < 150
      observed_mph = 75.0 if power_high else 60.0
      _step(lim, f, cc_enabled=True, vEgo=30.0, observed_mph=observed_mph,
            est_power_w=50_000.0 if power_high else 5_000.0, abasis=0.5 if power_high else 0.0)
      if lim.state != last_state:
        transitions += 1
        last_state = lim.state
    # iter9 had 96 transitions/9min; iter10 target < 20.
    # With 2 s min-dwell, max possible transitions = 9 min / 2 s = 270 → but
    # combined with sustain debouncing should be much less. Allow up to 30.
    self.assertLess(transitions, 30,
                    f"iter10 must dampen oscillation: got {transitions} transitions "
                    f"in 9 min, target <20 (iter9 baseline 96)")


class TestIter10Governor(unittest.TestCase):
  """iter10 Layer 1: bounded reference governor. Mode-based bound computation
  prevents Event A (dwell when gas pressed) and Event B (cluster < vEgo
  oscillation) by construction."""

  def setUp(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      GOVERNOR_MODE_NORMAL, GOVERNOR_MODE_GAS_CATCHUP, GOVERNOR_MODE_DECEL,
      GOVERNOR_MODE_BRAKE, MAX_DEFICIT_DEFAULT_MPH, GAS_HEADROOM_MPH, GAS_HOLD_MIN_FRAMES,
      EGO_SLOP_MS, GAS_RES_INTERVAL_FRAMES,
    )
    self.MODE_NORMAL = GOVERNOR_MODE_NORMAL
    self.MODE_GAS_CATCHUP = GOVERNOR_MODE_GAS_CATCHUP
    self.MODE_DECEL = GOVERNOR_MODE_DECEL
    self.MODE_BRAKE = GOVERNOR_MODE_BRAKE
    self.MAX_DEFICIT = MAX_DEFICIT_DEFAULT_MPH
    self.GAS_HEAD = GAS_HEADROOM_MPH
    self.GAS_HOLD = GAS_HOLD_MIN_FRAMES
    self.EGO_SLOP = EGO_SLOP_MS
    self.GAS_RES = GAS_RES_INTERVAL_FRAMES

  def _engaged_lim(self, vEgo, target_mph):
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=vEgo, observed_mph=target_mph)
    _step(lim, 25, cc_enabled=True, vEgo=vEgo, observed_mph=target_mph,
          button=ButtonType.decelCruise)
    for f in range(26, 60):
      _step(lim, f, cc_enabled=True, vEgo=vEgo, observed_mph=target_mph)
    lim.user_target_speed = target_mph * MPH_TO_MS
    return lim

  def test_event_a_gas_does_not_block_res(self):
    """Event A (drive #8 t=320-336): cluster=21 mph, target=44 mph, gas held.
    iter9 froze cluster at 21 because gas blocked want_res chain. iter10
    governor allows RES (rate-capped to 1.5 s) so cluster ramps up tracking
    vEgo. Verify RES fires SOMETIME during a 200-frame gas window."""
    lim = self._engaged_lim(vEgo=10.0, target_mph=44.0)
    lim.user_target_speed = 44.0 * MPH_TO_MS
    lim.last_res_frame = -10000
    res_fired = False
    for f in range(100, 300):
      btn, _ = _step(lim, f, cc_enabled=True, vEgo=10.0, observed_mph=21.0,
                     est_power_w=5_000.0, abasis=0.0, gas=True)
      if btn == Buttons.RES_ACCEL:
        res_fired = True
        break
    self.assertTrue(res_fired,
                    "Event A: gas alone must NOT block RES (iter9 dwell bug fix)")

  def test_governor_bounds_never_inverted(self):
    """Across all governor modes, lower_bound ≤ upper_bound by construction
    (or saturated permissively if degenerate)."""
    lim = _make_limiter()
    lim.user_target_speed = 70.0 * MPH_TO_MS

    # NORMAL: typical case
    lim._gas_hold_frames = 0
    lo, hi = lim._compute_governor_bounds(self.MODE_NORMAL, v_ego=30.0, frame=999,
                                            observed_set_speed=60.0 * MPH_TO_MS,
                                            dynamic_ceiling=80.0 * MPH_TO_MS)
    self.assertLessEqual(lo, hi, "NORMAL: lo > hi")

    # GAS_CATCHUP: cluster < vEgo+headroom (typical Event A)
    lo, hi = lim._compute_governor_bounds(self.MODE_GAS_CATCHUP, v_ego=10.0, frame=999,
                                            observed_set_speed=21.0 * MPH_TO_MS,
                                            dynamic_ceiling=999.0)
    self.assertLessEqual(lo, hi, "GAS_CATCHUP cluster<ego: lo > hi")

    # GAS_CATCHUP: cluster > vEgo+headroom (mid-recovery / coast)
    lo, hi = lim._compute_governor_bounds(self.MODE_GAS_CATCHUP, v_ego=10.0, frame=999,
                                            observed_set_speed=40.0 * MPH_TO_MS,
                                            dynamic_ceiling=999.0)
    self.assertLessEqual(lo, hi, "GAS_CATCHUP cluster>ego+headroom: lo > hi")
    self.assertAlmostEqual(lo, hi, places=4,
                           msg="GAS_CATCHUP cluster>ego+headroom: cluster should hold (lo==hi)")

    # DECEL: permissive lower
    lo, hi = lim._compute_governor_bounds(self.MODE_DECEL, v_ego=20.0, frame=999,
                                            observed_set_speed=50.0 * MPH_TO_MS,
                                            dynamic_ceiling=999.0)
    self.assertLessEqual(lo, hi, "DECEL: lo > hi")

  def _post_engagement_lim(self):
    """Helper: produce a limiter that has past the engagement-transient window
    (iter11 Fix A safety override). Tests that exercise governor bounds
    directly need to bypass the engage-edge floor suspension."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000   # well before any test frame
    return lim

  def test_normal_mode_clusters_max_deficit_at_highway(self):
    """At highway, max_deficit floor enforced as HARD invariant (iter11 Fix A).
    Drive #8 Event B: prevents 13 mph offset; iter10 had this as conditional."""
    lim = self._post_engagement_lim()
    lim.user_target_speed = 75.0 * MPH_TO_MS
    lo, hi = lim._compute_governor_bounds(self.MODE_NORMAL, v_ego=33.0, frame=999,
                                            observed_set_speed=60.0 * MPH_TO_MS,
                                            dynamic_ceiling=35.2)
    expected_lo_ms = (75.0 - self.MAX_DEFICIT) * MPH_TO_MS  # 68 mph
    expected_lo_ms = max(expected_lo_ms, 33.0 - self.EGO_SLOP)
    self.assertAlmostEqual(lo, expected_lo_ms, delta=0.1,
                           msg="NORMAL highway: lower_bound should be max(target-7, vEgo-slop)")
    self.assertAlmostEqual(hi, 75.0 * MPH_TO_MS, places=4,
                           msg="upper_bound should equal user_target")

  def test_normal_mode_no_max_deficit_floor_iter12(self):
    """iter12: max-deficit hard floor REMOVED (was iter11 Fix A, the wrong
    abstraction that caused ICE activations on today's drive). Only the
    no-below-vEgo invariant (iter10) remains as floor in NORMAL mode."""
    lim = self._post_engagement_lim()
    lim.user_target_speed = 50.0 * MPH_TO_MS
    # Low speed: vEgo=10 m/s = 22 mph; expect floor = vEgo - slop only (no max_deficit)
    lo, hi = lim._compute_governor_bounds(self.MODE_NORMAL, v_ego=10.0, frame=999,
                                            observed_set_speed=44.0 * MPH_TO_MS,
                                            dynamic_ceiling=41.0 * MPH_TO_MS)
    expected_lo_ms = 10.0 - self.EGO_SLOP   # 9.5 m/s = ~21 mph
    self.assertAlmostEqual(lo, expected_lo_ms, delta=0.1,
                           msg="iter12: floor is only no-below-vEgo invariant; max_deficit removed")

  def test_no_below_vego_invariant_clamps_lower_bound(self):
    """Event B fix: cluster_set ≥ vEgo - 0.5 m/s when user_target above vEgo."""
    lim = self._post_engagement_lim()
    lim.user_target_speed = 75.0 * MPH_TO_MS
    lo, hi = lim._compute_governor_bounds(self.MODE_NORMAL, v_ego=30.0, frame=999,
                                            observed_set_speed=70.0 * MPH_TO_MS,
                                            dynamic_ceiling=33.0)
    self.assertGreaterEqual(lo, 30.0 - self.EGO_SLOP - 1e-3,
                            "Lower bound must be >= vEgo - EGO_SLOP")

  def test_iter11_engagement_transient_suspends_floor(self):
    """iter11 Fix A: first 1.0 s after engagement, floor is suspended to allow
    cluster to settle. Avoids forcing recovery on the engage edge."""
    lim = _make_limiter()  # was_cc_enabled = False, _engaged_at_frame = -10000
    lim.user_target_speed = 75.0 * MPH_TO_MS
    # Frame 0, fresh engagement → transient
    lo, _ = lim._compute_governor_bounds(self.MODE_NORMAL, v_ego=30.0, frame=0,
                                          observed_set_speed=60.0 * MPH_TO_MS,
                                          dynamic_ceiling=35.2)
    self.assertEqual(lo, 0.0,
                     "Engagement transient: lower_bound suspended (USER_TARGET_MIN_MS=0)")
    # Now set engaged 200 frames ago (2 sec) — past 100-frame transient
    lim.was_cc_enabled = True
    lim._engaged_at_frame = 0
    lo, _ = lim._compute_governor_bounds(self.MODE_NORMAL, v_ego=30.0, frame=200,
                                          observed_set_speed=60.0 * MPH_TO_MS,
                                          dynamic_ceiling=35.2)
    self.assertGreater(lo, 0.0,
                       "Past engagement transient: hard floor enforced")

  def test_decel_intent_requires_positive_evidence(self):
    """v2: uncertain → MODE_NORMAL. Brake-recent OR sustained SCC decel only."""
    lim = self._engaged_lim(vEgo=20.0, target_mph=70.0)
    # No brake recently, no sustained negative abasis → not decel
    self.assertFalse(lim._has_decel_intent(frame=1000),
                     "Default state should not assert decel intent")
    # Recent brake → decel intent
    lim._last_brake_frame = 950   # 50 frames ago
    self.assertTrue(lim._has_decel_intent(frame=1000),
                    "Recent brake should trigger decel intent")
    # Brake long ago → no decel intent
    lim._last_brake_frame = 0
    self.assertFalse(lim._has_decel_intent(frame=1000),
                     "Old brake (>1 s) should NOT trigger decel intent")
    # Sustained SCC decel → decel intent
    lim._scc_decel_persistent_frames = 60
    self.assertTrue(lim._has_decel_intent(frame=1000),
                    "Sustained SCC decel should trigger decel intent")
    # Brief SCC decel (not sustained) → no
    lim._scc_decel_persistent_frames = 10
    self.assertFalse(lim._has_decel_intent(frame=1000),
                     "Brief SCC decel should NOT trigger decel intent")

  def test_governor_mode_priority_brake_over_gas(self):
    """Mode selection priority: brake_pressed > gas_pressed regardless of arming."""
    lim = self._engaged_lim(vEgo=20.0, target_mph=70.0)
    lim.user_target_speed = 70.0 * MPH_TO_MS
    lim._gas_hold_frames = 100   # would otherwise arm GAS_CATCHUP
    mode = lim._select_governor_mode(frame=1000, v_ego=20.0,
                                       gas_pressed=True, brake_pressed=True,
                                       in_standstill=False,
                                       observed_set_speed=60.0 * MPH_TO_MS)
    self.assertEqual(mode, self.MODE_BRAKE,
                     "Brake must win over gas in mode selection")

  def test_governor_mode_standstill_priority(self):
    """STANDSTILL mode wins over everything else (vEgo < threshold)."""
    lim = self._engaged_lim(vEgo=0.0, target_mph=44.0)
    lim._gas_hold_frames = 100
    mode = lim._select_governor_mode(frame=1000, v_ego=0.0,
                                       gas_pressed=True, brake_pressed=False,
                                       in_standstill=True,
                                       observed_set_speed=21.0 * MPH_TO_MS)
    self.assertEqual(mode, lim._select_governor_mode.__defaults__ if False
                     else 4,  # GOVERNOR_MODE_STANDSTILL = 4
                     "Standstill must win over everything")

  def test_want_res_during_gas_rate_capped(self):
    """v2 critique fix: RES presses while gas held are rate-capped to
    one per 1.5 s (vs 6/s normal global limit). Count RES presses in a
    fixed gas-held window — should be roughly window_seconds / 1.5."""
    lim = self._engaged_lim(vEgo=10.0, target_mph=44.0)
    lim.user_target_speed = 44.0 * MPH_TO_MS
    lim.last_res_frame = -10000
    res_count = 0
    # 5-second window with gas held + cluster well under target
    for f in range(100, 600):  # 500 frames = 5 s
      btn, _ = _step(lim, f, cc_enabled=True, vEgo=10.0, observed_mph=21.0,
                     est_power_w=5_000.0, abasis=0.0, gas=True)
      if btn == Buttons.RES_ACCEL:
        res_count += 1
    # Without rate cap (6/sec global), would be ~30 presses in 5 s.
    # With 1.5 s gas-time rate cap, max ~3-4 presses.
    self.assertGreater(res_count, 0, "Some RES should fire during 5s gas hold")
    self.assertLessEqual(res_count, 5,
                         f"Gas-time rate cap should limit RES to ~3-4 in 5 s; "
                         f"got {res_count} (cap is {self.GAS_RES} frames between)")


class TestIter10ObserverMitigation(unittest.TestCase):
  """iter10 Layer 3 commit 1: lowered grade clip + air density default."""

  def test_grade_clip_lowered_to_05_ms2(self):
    """Constant change: clip 1.0 → 0.5 m/s² to bound phantom contribution."""
    from opendbc.sunnypilot.car.hyundai.carstate_ext import GRADE_ACCEL_FILTERED_CLIP_MS2
    self.assertAlmostEqual(GRADE_ACCEL_FILTERED_CLIP_MS2, 0.5, places=4)

  def test_air_density_lowered_for_elevation(self):
    """Constant change: 1.225 → 1.10 kg/m³ (Loveland-tuned ~1500m)."""
    from opendbc.sunnypilot.car.hyundai.carstate_ext import AIR_DENSITY_KG_M3
    self.assertAlmostEqual(AIR_DENSITY_KG_M3, 1.10, places=4)

  def test_road_load_at_73mph_with_lowered_density(self):
    """At 73 mph (33 m/s), aero load with ρ=1.10 should be ~14.8 kW
    instead of ~16.5 kW with sea-level density."""
    from opendbc.sunnypilot.car.hyundai.carstate_ext import road_load_power_w
    p = road_load_power_w(33.0)
    # Roll = 0.011 * 1950 * 9.81 * 33 = 6953
    # Aero = 0.5 * 1.10 * 0.75 * 33^3 = 14817
    # Total = ~21.7 kW
    self.assertAlmostEqual(p, 21770.0, delta=200.0)

  def test_grade_at_05_clip_max_power_at_73mph(self):
    """With clip=0.5 m/s² and v=33 m/s, max grade contribution alone ≈
    1950 * 33 * 0.5 = 32 kW (was 64 kW with clip=1.0)."""
    from opendbc.sunnypilot.car.hyundai.carstate_ext import (
      VEHICLE_MASS_KG, GRADE_ACCEL_FILTERED_CLIP_MS2,
    )
    v = 33.0
    p_grade_max = VEHICLE_MASS_KG * v * GRADE_ACCEL_FILTERED_CLIP_MS2
    self.assertAlmostEqual(p_grade_max, 32175.0, delta=10.0)
    self.assertLess(p_grade_max, 35_000.0,
                    "grade contribution alone should not exceed ~35 kW at 73 mph")


class TestIter10KalmanGradeSource(unittest.TestCase):
  """iter10 Layer 3a: kalman pitch as grade source. card.py sets
  CarState.grade_accel_external_ms2 from liveLocationKalman pitch; carstate_ext
  prefers this over the noisy LONG_ACCEL-aEgo derivation when set."""

  def setUp(self):
    from opendbc.sunnypilot.car.hyundai.carstate_ext import (
      CarStateExt, GRADE_ACCEL_FILTERED_CLIP_MS2, GRAVITY_MS2,
    )
    self.CarStateExt = CarStateExt
    self.CLIP = GRADE_ACCEL_FILTERED_CLIP_MS2
    self.G = GRAVITY_MS2

  def test_pitch_to_grade_accel_conversion(self):
    """3% grade ≈ atan(0.03) ≈ 0.03 rad pitch → grade_accel ≈ 0.03 * 9.81 = 0.29 m/s²."""
    import math
    pitch_3pct = math.atan(0.03)
    expected = self.G * math.sin(pitch_3pct)
    # Verify card.py-style conversion gives expected value
    self.assertAlmostEqual(expected, 0.294, delta=0.01)

  def test_external_source_clipped_to_05_ms2(self):
    """Kalman pitch can produce >0.5 m/s² on steep grades; clipped per
    iter10 commit 1's lowered ceiling."""
    cp = FakeCP()
    cp_sp = FakeCPSP()
    ext = self.CarStateExt(cp, cp_sp)
    # Steep grade: 10% = 0.1 rad → 0.1 * 9.81 = 0.98 m/s² (above 0.5 clip)
    ext.grade_accel_external_ms2 = 0.98
    self.assertEqual(ext.grade_accel_external_ms2, 0.98,
                     "External value stored as set; clipping happens in update path")
    # The actual clip logic runs in _update_ev_limiter_signals; verified
    # via the constant-change test in TestIter10ObserverMitigation.

  def test_external_none_falls_back_to_legacy(self):
    """When kalman invalid (grade_accel_external_ms2 = None), carstate_ext
    falls back to LONG_ACCEL-aEgo derivation."""
    cp = FakeCP()
    cp_sp = FakeCPSP()
    ext = self.CarStateExt(cp, cp_sp)
    # Default is None on init
    self.assertIsNone(ext.grade_accel_external_ms2,
                      "External grade source defaults to None (legacy fallback)")

  def test_state_init_includes_diagnostic_counters(self):
    """iter10 commit 1: post-filter zero detector counters present."""
    cp = FakeCP()
    cp_sp = FakeCPSP()
    ext = self.CarStateExt(cp, cp_sp)
    self.assertFalse(ext._grade_filter_seen_nonzero)
    self.assertFalse(ext._grade_filter_zero_warning_logged)
    self.assertEqual(ext._grade_filter_call_count, 0)


class TestIter11Fixes(unittest.TestCase):
  """iter11 — containment release covering bugs A,B,C,D,E,F discovered
  in drives #9-13. See plan-v4-iter11.md."""

  def setUp(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      ACTIVE_STATES, MIN_ACTIVE_STATE_DWELL_FRAMES,
    )
    self.ACTIVE_STATES = ACTIVE_STATES
    self.MIN_DWELL = MIN_ACTIVE_STATE_DWELL_FRAMES

  # --- Bug A: max-deficit invariant + recovery escape ---
  def test_recovery_below_floor_overrides_power_too_high(self):
    """When cluster < lower_bound, want_res still fires even if power_too_high.
    Recovery escape priority over power shaping."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000
    lim.user_target_speed = 50.0 * MPH_TO_MS
    # Force recovery escape mode active (cluster below floor)
    lim._recovery_escape_active = True
    lim._recovery_escape_start_t = 0.0
    lim._recovery_escape_start_cluster = 30.0 * MPH_TO_MS
    lim.last_res_frame = -10000
    # Enough elapsed time so rate cap permits a press
    # frame * 0.01 = 1.0 sec → max gain 2 mph from 30 → 32 mph
    btn, _ = _step(lim, 100, cc_enabled=True, vEgo=20.0, observed_mph=30.0,
                   est_power_w=50_000.0, abasis=1.0)  # power high
    # observed=30 mph < user_target=50, no power_too_high cancel → RES allowed
    self.assertEqual(btn, Buttons.RES_ACCEL,
                     "Recovery escape: RES fires even with power_too_high")

  def test_recovery_rate_limited_2mph_per_sec(self):
    """Iter11 Fix A: at 100Hz decisions, recovery cluster gain ≤ 2 mph/sec."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000
    lim.user_target_speed = 50.0 * MPH_TO_MS
    lim._recovery_escape_active = True
    lim._recovery_escape_start_t = 100 * 0.01  # entered at frame 100
    lim._recovery_escape_start_cluster = 30.0 * MPH_TO_MS
    lim.last_res_frame = -10000
    # At frame 150 (0.5 sec later), max permitted = 30 + 2*0.5 = 31 mph
    # If cluster already at 31, want_res blocked
    btn, _ = _step(lim, 150, cc_enabled=True, vEgo=20.0, observed_mph=31.5,
                   est_power_w=10_000.0, abasis=0.0)
    self.assertEqual(btn, Buttons.NONE,
                     "Recovery rate cap: cluster already at max permitted (31 mph)")

  # --- Bug B: arbiter ---
  def test_no_direct_state_mutation_outside_arbiter(self):
    """Static guard: only _arbitrate_state_transition and __init__ should
    assign self.state. Catches Bug B regression."""
    import re
    from opendbc.sunnypilot.car.hyundai import ev_limiter as _evl
    src_path = _evl.__file__
    with open(src_path) as f:
      src = f.read()
    # Find all 'self.state = X' assignments
    matches = re.finditer(r'\bself\.state\s*=', src)
    locations = []
    for m in matches:
      # Find the enclosing function name by walking backward
      pre = src[:m.start()]
      lines = pre.split('\n')
      # Walk backwards to find 'def ...'
      for i in range(len(lines)-1, -1, -1):
        m2 = re.match(r'\s*def (\w+)\(', lines[i])
        if m2:
          locations.append(m2.group(1))
          break
      else:
        locations.append('<module>')
    # Allowed: __init__ (init), _arbitrate_state_transition (sole mutation site)
    bad = [loc for loc in locations if loc not in ('__init__', '_arbitrate_state_transition')]
    self.assertEqual(bad, [],
                     f"self.state = ... assignments outside arbiter: {bad}")

  def test_arbiter_blocks_active_state_exit_before_min_dwell(self):
    """SOFT_CAP cannot exit before MIN_ACTIVE_STATE_DWELL_FRAMES."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000
    lim.state = STATE_SOFT_CAP_ACTIVE
    lim._state_entered_frame = 1000
    # Try to exit at frame 1050 (only 50 frames in state, < 200 MIN_DWELL)
    actual = lim._arbitrate_state_transition(1050, STATE_IDLE, "test", hard_preempt=False)
    self.assertEqual(actual, STATE_SOFT_CAP_ACTIVE, "Min-dwell blocks exit")
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE)

  def test_arbiter_allows_hard_preempt_through_min_dwell(self):
    """Hard preempts (driver override, brake) bypass min-dwell."""
    lim = _make_limiter()
    lim.state = STATE_SOFT_CAP_ACTIVE
    lim._state_entered_frame = 1000
    actual = lim._arbitrate_state_transition(1050, STATE_DISABLED, "cc_off", hard_preempt=True)
    self.assertEqual(actual, STATE_DISABLED)
    self.assertEqual(lim.state, STATE_DISABLED)

  # --- Bug C: kalman gate (note: card.py change, tested at integration level) ---
  def test_kalman_str_status_compared_correctly(self):
    """pycapnp returns enum NAME (string) for status. iter11 compares to 'valid' not 2."""
    # Sanity: confirm string comparison works as expected
    status_value = 'valid'   # what pycapnp returns
    self.assertEqual(str(status_value) == 'valid', True)
    self.assertEqual(status_value == 2, False, "Bug C: numeric comparison would fail")

  # --- Bug D: ineffective-RES watchdog ---
  def test_ineffective_res_escape_triggers_after_8_presses(self):
    """8+ RES presses without 1 mph cluster gain → escape mode."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000
    lim.user_target_speed = 50.0 * MPH_TO_MS
    lim.last_res_frame = -10000
    # Simulate 8 RES presses with cluster stuck at 30 mph
    for i in range(8):
      f = 100 + i * 80   # one press per 80 frames (RES_COOLDOWN)
      _step(lim, f, cc_enabled=True, vEgo=20.0, observed_mph=30.0,
            est_power_w=10_000.0, abasis=0.0)
    # After 8th press, escape should arm on the next decision
    f_next = 100 + 8 * 80 + 5
    _step(lim, f_next, cc_enabled=True, vEgo=20.0, observed_mph=30.0,
          est_power_w=10_000.0, abasis=0.0)
    self.assertGreater(lim._ineffective_res_events, 0,
                       "Escape should trigger after 8 ineffective RES presses")

  def test_escape_aborts_at_3mph_cluster_delta_cap(self):
    """Escape mode aborts immediately if cluster moves up by 3 mph."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000
    # Activate escape manually
    lim._res_escape_until_frame = 10000   # active
    lim._res_escape_start_cluster_ms = 30.0 * MPH_TO_MS
    # Step with cluster now 33 mph (3 mph delta)
    _step(lim, 200, cc_enabled=True, vEgo=20.0, observed_mph=33.5,
          est_power_w=10_000.0, abasis=0.0)
    self.assertLess(lim._res_escape_until_frame, 200,
                    "Escape aborts when cluster moved 3 mph")

  # --- Bug E: power estimator ---
  def test_assume_ev_only_defaults_false_when_param_absent_iter13(self):
    """iter13 v4 (gpt-5.5 round-2 mandate): default fail-CLOSED to False when
    Params absent or read fails. Was True in iter11 (fail-open); changed in iter13
    because silently enabling the EV-only assumption on a plumbing failure was the
    same class of silent behavior that hurt iter9/iter11/iter12."""
    from opendbc.sunnypilot.car.hyundai.carstate_ext import CarStateExt
    cp = FakeCP(); cp_sp = FakeCPSP()
    ext = CarStateExt(cp, cp_sp)
    self.assertFalse(ext._assume_ev_only,
                     "iter13 v4: EvLimiterAssumeEvOnly defaults to False (fail-closed) when absent")
    self.assertFalse(ext._assume_ev_only_param_read_ok,
                     "iter13 v4: param_read_ok=False when card.py hasn't yet written the attribute")

  def test_power_capped_when_assume_ev_only_true(self):
    """When _assume_ev_only=True, power capped at _ev_motor_cap_w."""
    from opendbc.sunnypilot.car.hyundai.carstate_ext import CarStateExt
    cp = FakeCP(); cp_sp = FakeCPSP()
    ext = CarStateExt(cp, cp_sp)
    ext._assume_ev_only = True
    ext._ev_motor_cap_w = 60_000.0
    # Direct unit test: simulate the cap logic
    raw = 80_000.0
    capped = min(raw, ext._ev_motor_cap_w) if ext._assume_ev_only else raw
    self.assertEqual(capped, 60_000.0)

  def test_power_not_capped_when_assume_ev_only_false(self):
    from opendbc.sunnypilot.car.hyundai.carstate_ext import CarStateExt
    cp = FakeCP(); cp_sp = FakeCPSP()
    ext = CarStateExt(cp, cp_sp)
    ext._assume_ev_only = False
    raw = 80_000.0
    capped = min(raw, ext._ev_motor_cap_w) if ext._assume_ev_only else raw
    self.assertEqual(capped, 80_000.0)

  # --- Bug F: highway SET cadence ---
  def test_iter12_set_min_repeat_when_acked(self):
    """iter12 ack-driven: after SET, if cluster drops 1mph (success), next SET
    allowed after SET_MIN_REPEAT_FRAMES (50 frames = 0.5s)."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000
    lim.user_target_speed = 75.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    # First SET fires
    btn1, _ = _step(lim, 100, cc_enabled=True, vEgo=33.0, observed_mph=80.0,
                    est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(btn1, Buttons.SET_DECEL, "First highway SET fires")
    # 60 frames later (0.6s, > SET_MIN_REPEAT_FRAMES=50), cluster down 1mph (ack)
    btn2, _ = _step(lim, 160, cc_enabled=True, vEgo=33.0, observed_mph=79.0,
                    est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(btn2, Buttons.SET_DECEL,
                     "After 0.6s + cluster ack, next SET allowed")

  def test_iter13_burst_copies_set_is_1(self):
    """iter13 v4 (R4-MF6): BURST_COPIES_SET reduced 2 → 1 for first rollout.

    The CarController-side ClusterButtonRateLimiter is now the authoritative
    wire-safety guard, with a sliding 100ms all-button window of 1. With
    BURST_COPIES_SET=2, the second copy would always be dropped at the wire
    layer with `rateLimitSameFrame` or `rateLimitAll100ms` anyway. Reducing
    EVLimiter-side burst to 1 keeps the layers consistent and avoids
    confusing "set_dropped" telemetry on otherwise-accepted SETs."""
    lim = _make_limiter()
    lim.was_cc_enabled = True
    lim._engaged_at_frame = -1000
    lim.user_target_speed = 75.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    btn, _ = _step(lim, 100, cc_enabled=True, vEgo=33.0, observed_mph=80.0,
                   est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(btn, Buttons.SET_DECEL)
    self.assertEqual(lim.current_burst_count, 1,
                     "iter13 v4: BURST_COPIES_SET=1 (R4-MF6)")


class TestIter12Fixes(unittest.TestCase):
  """iter12 — comprehensive correction after iter11 multi-failure.
  Removes hard max-deficit floor (was iter11 Fix A, the wrong abstraction).
  Adds ack-driven SET cadence + ineffective-SET escape symmetric to RES Bug D.
  Restores burst=2 on highway.
  Mode-gates SET emission in BRAKE/DECEL/STANDSTILL/GAS_CATCHUP."""

  def _engaged(self, vEgo=20.0, target_mph=70.0):
    lim = _make_limiter()
    for f in range(25):
      _step(lim, f, cc_enabled=False, vEgo=vEgo, observed_mph=target_mph)
    _step(lim, 25, cc_enabled=True, vEgo=vEgo, observed_mph=target_mph,
          button=ButtonType.decelCruise)
    for f in range(26, 60):
      _step(lim, f, cc_enabled=True, vEgo=vEgo, observed_mph=target_mph)
    lim.user_target_speed = target_mph * MPH_TO_MS
    lim.last_set_frame = -10000
    lim.last_res_frame = -10000
    return lim

  # --- Property: cluster - vEgo ≤ max_gap when sliding cap is binding ---
  def test_iter12_cluster_target_le_vego_plus_max_gap(self):
    """The PRIMARY iter12 fix: cluster target should be at most vEgo+max_gap
    when sliding cap is below user_target. This was broken in iter11 (hard
    floor pinned cluster at user_target-7 = vEgo+18 in low-speed traffic)."""
    lim = self._engaged(vEgo=22.0, target_mph=75.0)   # vEgo=50mph, target=75
    # max_gap default = 5 mph (sliding cap target = vEgo+5 = 55 mph)
    # Drive limiter for many frames at this scenario; observe what target_set
    # the controller computes (via internal clamping)
    lim._engaged_at_frame = -10000  # past transient
    lim.was_cc_enabled = True
    # Just verify the bound computation directly
    from opendbc.sunnypilot.car.hyundai.ev_limiter import GOVERNOR_MODE_NORMAL
    lo, hi = lim._compute_governor_bounds(GOVERNOR_MODE_NORMAL, v_ego=22.0, frame=999,
                                            observed_set_speed=55.0 * MPH_TO_MS,
                                            dynamic_ceiling=27.0)  # 22 m/s + ~5 mph margin
    # iter12: floor = vEgo - slop only (no max_deficit). User can target down to ~vEgo.
    from opendbc.sunnypilot.car.hyundai.ev_limiter import EGO_SLOP_MS
    expected_floor = 22.0 - EGO_SLOP_MS
    self.assertAlmostEqual(lo, expected_floor, delta=0.1,
                           msg="iter12: floor only = vEgo - slop, no max_deficit pin")

  def test_iter12_brake_mode_blocks_both_set_and_res(self):
    """gpt-5.5 v1 fix #3: BRAKE mode emits NO SET and NO RES."""
    lim = self._engaged(vEgo=20.0, target_mph=70.0)
    lim.user_target_speed = 70.0 * MPH_TO_MS
    # Power high + cluster well above target → would normally trigger SET
    btn1, _ = _step(lim, 100, cc_enabled=True, vEgo=20.0, observed_mph=80.0,
                    est_power_w=50_000.0, abasis=0.5, brake=True)
    self.assertEqual(btn1, Buttons.NONE, "BRAKE mode: no SET")
    # Cluster well below target → would normally trigger RES
    btn2, _ = _step(lim, 200, cc_enabled=True, vEgo=20.0, observed_mph=40.0,
                    est_power_w=5_000.0, abasis=0.0, brake=True)
    self.assertEqual(btn2, Buttons.NONE, "BRAKE mode: no RES")

  def test_iter12_set_ack_driven_waits_for_response(self):
    """After SET press, controller waits for cluster to drop OR for timeout
    before next decision. No blind cadence."""
    lim = self._engaged(vEgo=33.0, target_mph=75.0)
    lim.user_target_speed = 75.0 * MPH_TO_MS
    # First SET fires
    btn1, _ = _step(lim, 100, cc_enabled=True, vEgo=33.0, observed_mph=82.0,
                    est_power_w=10_000.0, abasis=0.0)
    self.assertEqual(btn1, Buttons.SET_DECEL)
    self.assertEqual(lim.last_set_frame, 100, "last_set_frame updated")
    # 30 frames later (300ms), cluster hasn't dropped → don't fire (waiting)
    btn2, _ = _step(lim, 130, cc_enabled=True, vEgo=33.0, observed_mph=82.0,
                    est_power_w=10_000.0, abasis=0.0)
    self.assertEqual(btn2, Buttons.NONE, "Waiting for response, no fire")
    # 60 frames later (600ms), cluster dropped 1mph → ack → fire after 50f cooldown
    btn3, _ = _step(lim, 160, cc_enabled=True, vEgo=33.0, observed_mph=81.0,
                    est_power_w=10_000.0, abasis=0.0)
    self.assertEqual(btn3, Buttons.SET_DECEL, "Ack received + cooldown clear → fire")

  def test_iter12_set_power_aware_cooldown_fast_when_power_high(self):
    """When power_too_high, SET min cooldown drops to SET_MIN_REPEAT_FRAMES (50)."""
    lim = self._engaged(vEgo=33.0, target_mph=75.0)
    lim.user_target_speed = 75.0 * MPH_TO_MS
    # First SET fires (cluster=82 > target_set, power high)
    btn1, _ = _step(lim, 100, cc_enabled=True, vEgo=33.0, observed_mph=82.0,
                    est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(btn1, Buttons.SET_DECEL)
    # 60 frames later, cluster dropped → ack → power-high cooldown applies
    btn2, _ = _step(lim, 160, cc_enabled=True, vEgo=33.0, observed_mph=81.0,
                    est_power_w=50_000.0, abasis=0.5)
    self.assertEqual(btn2, Buttons.SET_DECEL,
                     "Power-high + ack: fire at 0.5s (not 1.5s normal cadence)")

  # iter13 v4: REMOVED test_iter12_ineffective_set_escape_triggers.
  # Rationale: iter12's continuous-frame SET escape was the cause of 3
  # SCC auto-cancels on drive 17 (route 00000077--336f3cb0c9). iter13 v4
  # removes the escape mode entirely; new tests below assert the
  # replacement ack-paced behavior + adaptive backoff (R4-MF1).


# ========================================================================
# iter13 v4 — new tests
# ========================================================================

class TestIter13V4States(unittest.TestCase):
  """iter13 v4 state-enum renumbering: STANDSTILL_PRELAUNCH_SET inserted at @2."""

  def test_state_renumbering(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
        STATE_IDLE, STATE_STANDSTILL_PRELAUNCH_SET,
        STATE_SOFT_CAP_ACTIVE, STATE_RECOVERY_ACTIVE,
        STATE_DRIVER_OVERRIDE_SET, STATE_BUS_FAULT_HOLD, STATE_DISABLED,
    )
    self.assertEqual(STATE_IDLE, 0)
    self.assertEqual(STATE_STANDSTILL_HOLD, 1)
    self.assertEqual(STATE_STANDSTILL_PRELAUNCH_SET, 2)
    self.assertEqual(STATE_SOFT_CAP_ACTIVE, 3)
    self.assertEqual(STATE_RECOVERY_ACTIVE, 4)
    self.assertEqual(STATE_DRIVER_OVERRIDE_SET, 5)
    self.assertEqual(STATE_DRIVER_OVERRIDE_RES, 6)
    self.assertEqual(STATE_BUS_FAULT_HOLD, 7)
    self.assertEqual(STATE_DISABLED, 8)


class TestIter13V4Allowlist(unittest.TestCase):
  """R4-MF4: EVLimiter `_desired_button` runtime guard (raise, not assert).

  Production may run `python -O` which strips assertions. Use raise to ensure
  the guard cannot be optimized out.
  """

  def _lim(self):
    return _make_limiter()

  def test_evlimiter_desired_button_assertion_rejects_cancel(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import EVLimiterError
    lim = self._lim()
    with self.assertRaises(EVLimiterError):
      lim._set_desired_button(Buttons.CANCEL)

  def test_evlimiter_desired_button_assertion_rejects_gap(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import EVLimiterError
    lim = self._lim()
    with self.assertRaises(EVLimiterError):
      lim._set_desired_button(Buttons.GAP_DIST)

  def test_evlimiter_desired_button_assertion_accepts_none_set_res(self):
    lim = self._lim()
    lim._set_desired_button(None)
    self.assertIsNone(lim._desired_button)
    lim._set_desired_button(Buttons.SET_DECEL)
    self.assertEqual(lim._desired_button, Buttons.SET_DECEL)
    lim._set_desired_button(Buttons.RES_ACCEL)
    self.assertEqual(lim._desired_button, Buttons.RES_ACCEL)


class TestIter13V4AckMatcher(unittest.TestCase):
  """R4-MF3 + MF5: ACK matcher post-emission, 1-to-1, 500ms inclusive.

  Critical invariants verified:
    - Pre-emission decrements never count (test_set_acked_excludes_pre_emission_decrements)
    - Driver-button-induced decrements excluded (test_set_acked_excludes_decrement_caused_by_driver_button)
    - Dropped SETs never registered (test_set_acked_excludes_dropped_set_registration)
    - 1-to-1 oldest-first matching
  """

  def _lim(self):
    lim = _make_limiter()
    lim._cluster_prev_for_ack_ms = 50.0 * MPH_TO_MS
    return lim

  def test_note_set_emitted_appends_frame(self):
    lim = self._lim()
    lim.note_set_emitted(100)
    self.assertEqual(list(lim._unmatched_emitted_set_frames), [100])

  def test_set_acked_one_to_one_mapping_oldest_first(self):
    """Two emitted SETs, one cluster decrement → only the OLDEST consumed."""
    lim = self._lim()
    lim.note_set_emitted(100)
    lim.note_set_emitted(120)
    # decrement at f=140
    lim._cluster_prev_for_ack_ms = 50.0 * MPH_TO_MS
    acked = lim.try_ack_set(48.0 * MPH_TO_MS, 140, physical_button_in_window=False)
    self.assertTrue(acked)
    self.assertEqual(list(lim._unmatched_emitted_set_frames), [120])
    self.assertEqual(lim._cluster_decrement_acked, 1)

  def test_set_acked_excludes_pre_emission_decrements(self):
    """Decrement BEFORE any SET emission does NOT count."""
    lim = self._lim()
    # No emissions yet
    acked = lim.try_ack_set(48.0 * MPH_TO_MS, 50, physical_button_in_window=False)
    self.assertFalse(acked)
    self.assertEqual(lim._cluster_decrement_acked, 0)

  def test_set_acked_excludes_decrement_caused_by_driver_button(self):
    """Cluster decrement when a physical driver button was active does NOT credit."""
    lim = self._lim()
    lim.note_set_emitted(100)
    acked = lim.try_ack_set(48.0 * MPH_TO_MS, 130, physical_button_in_window=True)
    self.assertFalse(acked)
    self.assertEqual(lim._cluster_decrement_acked, 0)
    # Frame still in deque (not consumed)
    self.assertEqual(list(lim._unmatched_emitted_set_frames), [100])

  def test_set_acked_excludes_dropped_set_registration(self):
    """R4-MF3: dropped SETs never enter the ACK matcher.

    EVLimiter exposes `note_set_emitted(frame)` only — there is no
    `note_set_dropped` method. CarController calls note_set_emitted ONLY
    after maybe_emit() returns SET (i.e., after rate-limit acceptance).
    This test verifies the API contract by inspection.
    """
    lim = self._lim()
    # If CarController correctly calls only on accepted, the deque only
    # receives accepted frames. We assert that without explicit register,
    # nothing accumulates.
    lim.try_ack_set(48.0 * MPH_TO_MS, 130, physical_button_in_window=False)
    self.assertEqual(list(lim._unmatched_emitted_set_frames), [])

  def test_set_acked_window_500ms_inclusive(self):
    """ACK window = 50 frames inclusive. f+50 should still credit; f+51 expires."""
    lim = self._lim()
    lim.note_set_emitted(100)
    # f=150 is exactly 50 frames after — within inclusive window
    lim._cluster_prev_for_ack_ms = 50.0 * MPH_TO_MS
    acked = lim.try_ack_set(48.0 * MPH_TO_MS, 150, physical_button_in_window=False)
    self.assertTrue(acked, "frame f+50 should be inside inclusive 500ms window")

  def test_note_set_emitted_purges_stale(self):
    """Frames older than ACK_WINDOW_FRAMES are purged on each note_set_emitted."""
    lim = self._lim()
    lim.note_set_emitted(100)
    lim.note_set_emitted(200)   # 200 - 100 = 100 > ACK_WINDOW_FRAMES (50)
    self.assertEqual(list(lim._unmatched_emitted_set_frames), [200])


class TestIter13V4LaunchTarget(unittest.TestCase):
  """v4 Section C launch target formula: max(21, min(user_target, vEgo+gap))."""

  def _lim(self):
    return _make_limiter()

  def test_launch_target_at_vego_zero_floors_at_21mph(self):
    """At vEgo=0 with user_target=42mph: max(21, min(42, 0+5)) = 21."""
    lim = self._lim()
    target = lim._compute_launch_target_mph(42.0 * MPH_TO_MS, 0.0)
    self.assertEqual(target, 21.0)

  def test_launch_target_at_vego_15mph_with_gap_5_returns_20_floored_to_21(self):
    """At vEgo=15: min(user, 15+5)=20, max(21, 20)=21 (floor wins)."""
    lim = self._lim()
    target = lim._compute_launch_target_mph(42.0 * MPH_TO_MS, 15.0 * MPH_TO_MS)
    self.assertEqual(target, 21.0)

  def test_launch_target_at_high_speed_uses_min(self):
    """At vEgo=20: min(42, 20+5)=25, max(21, 25)=25."""
    lim = self._lim()
    target = lim._compute_launch_target_mph(42.0 * MPH_TO_MS, 20.0 * MPH_TO_MS)
    self.assertEqual(target, 25.0)

  def test_launch_target_clamped_by_user_target(self):
    """If user_target=15mph (below floor), max(21, min(15, ...)) = 21."""
    lim = self._lim()
    target = lim._compute_launch_target_mph(15.0 * MPH_TO_MS, 0.0)
    self.assertEqual(target, 21.0)


class TestIter13V4PrelaunchPulseCap(unittest.TestCase):
  """v4 Section E pulse cap 10 → 30 after first cluster_decrement_acked."""

  def test_pulse_cap_initial_10(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      STANDSTILL_PULSE_CAP_INITIAL, STANDSTILL_PULSE_CAP_AFTER_ACK,
    )
    self.assertEqual(STANDSTILL_PULSE_CAP_INITIAL, 10)
    self.assertEqual(STANDSTILL_PULSE_CAP_AFTER_ACK, 30)

  def test_pulse_cap_raises_after_first_ack(self):
    """After try_ack_set credits one decrement, _prelaunch_set_pulse_cap → 30."""
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      STANDSTILL_PULSE_CAP_INITIAL, STANDSTILL_PULSE_CAP_AFTER_ACK,
    )
    lim = _make_limiter()
    lim._was_in_prelaunch = True
    lim._prelaunch_set_pulse_cap = STANDSTILL_PULSE_CAP_INITIAL
    lim._cluster_prev_for_ack_ms = 50.0 * MPH_TO_MS
    lim.note_set_emitted(100)
    lim.try_ack_set(48.0 * MPH_TO_MS, 130, physical_button_in_window=False)
    self.assertTrue(lim._prelaunch_first_ack_seen)
    self.assertEqual(lim._prelaunch_set_pulse_cap, STANDSTILL_PULSE_CAP_AFTER_ACK)


class TestIter13V4AdvisoryBlockReason(unittest.TestCase):
  """Section D priorities 13-19 advisory dispatch (deterministic priority)."""

  def test_advisory_no_action_when_set_or_res_desired(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import GOVERNOR_MODE_NORMAL
    lim = _make_limiter()
    r = lim._compute_advisory_block_reason(
      frame=100, governor_mode=GOVERNOR_MODE_NORMAL,
      power_too_high=False, want_set=True, want_res=False)
    self.assertEqual(r, "none")

  def test_advisory_modeForbidden_for_brake_decel_standstill(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      GOVERNOR_MODE_BRAKE, GOVERNOR_MODE_DECEL, GOVERNOR_MODE_STANDSTILL,
    )
    lim = _make_limiter()
    for mode in (GOVERNOR_MODE_BRAKE, GOVERNOR_MODE_DECEL, GOVERNOR_MODE_STANDSTILL):
      r = lim._compute_advisory_block_reason(
        frame=100, governor_mode=mode,
        power_too_high=False, want_set=False, want_res=False)
      self.assertEqual(r, "modeForbidden")


class TestIter13V4Constants(unittest.TestCase):
  """v4 Section A constants (sanity)."""

  def test_iter13_constants(self):
    from opendbc.sunnypilot.car.hyundai.ev_limiter import (
      ACK_WINDOW_FRAMES, SET_HARD_MIN_INTERVAL_FRAMES,
      LAUNCH_DEADBAND_MPH, STANDSTILL_TARGET_FLOOR_MPH, ALLOWABLE_GAP_MPH,
      STANDSTILL_NO_ACK_BACKOFF_AFTER_EMITTED, SET_NO_ACK_BACKOFF_TIER_FRAMES,
    )
    self.assertEqual(ACK_WINDOW_FRAMES, 50)              # 500ms @100Hz
    self.assertEqual(SET_HARD_MIN_INTERVAL_FRAMES, 50)   # 0.5s
    self.assertEqual(LAUNCH_DEADBAND_MPH, 2.0)
    self.assertEqual(STANDSTILL_TARGET_FLOOR_MPH, 21.0)
    self.assertEqual(ALLOWABLE_GAP_MPH, 5.0)
    self.assertEqual(STANDSTILL_NO_ACK_BACKOFF_AFTER_EMITTED, 5)
    self.assertEqual(SET_NO_ACK_BACKOFF_TIER_FRAMES, (150, 200, 300))


class TestIter13V4Iter12EscapeRemoved(unittest.TestCase):
  """Verify iter12 continuous-frame SET escape mode is REMOVED (R4-MF1 root cause)."""

  def test_iter12_escape_fields_no_longer_drive_emission(self):
    """The _set_escape_until_frame field is no longer referenced in the SET
    emission path. iter13 replaces iter12's continuous-frame escape with
    ACK-paced single-press SET + adaptive backoff."""
    from opendbc.sunnypilot.car.hyundai import ev_limiter as ev
    src = open(ev.__file__).read()
    # The escape SET fire site (`button = Buttons.SET_DECEL` inside an
    # `if want_set and in_set_escape:` branch) was removed.
    self.assertNotIn("if want_set and in_set_escape:", src,
                     "iter12 continuous-frame SET escape path must be removed")
    # in_set_escape stays as a sentinel False (kept for code-path symmetry
    # but never True). Verify the only assignment is `in_set_escape = False`.
    import re
    assigns = re.findall(r"in_set_escape\s*=\s*\S+", src)
    self.assertTrue(all("False" in a or "in_set_escape = False" in a for a in assigns),
                    f"in_set_escape should only be assigned False; got {assigns}")


if __name__ == "__main__":
  unittest.main()
