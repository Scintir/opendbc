"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter17 — power estimator rebuilt on measured accel + grade, with a drivetrain
efficiency map and an accessory baseline (drive 2026-09-24 forensics,
docs/ev-limiter/drive-2026-09-24-forensics.md).

Replaces test_iter15_grade_clamp.py: the grade deadband and the grade cap only
existed to fight noise and double counting in the aBasis-based formula, both of
which are gone.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.carstate_ext import (
  AEGO_FILTER_TAU_S,
  AUX_POWER_DEFAULT_W,
  AUX_POWER_MAX_W,
  EFFICIENCY_HIGHWAY,
  EFFICIENCY_LOW_SPEED,
  EFFICIENCY_RAMP_END_MS,
  POWER_TAU_FALL_S,
  POWER_TAU_RISE_S,
  VEHICLE_MASS_KG,
  CarStateExt,
  drivetrain_efficiency,
  road_load_power_w,
)
from opendbc.sunnypilot.car.hyundai.ev_limiter import MPH_TO_MS


class _VL(dict):
  """cp.vl lookalike: missing message -> KeyError like the real parser."""


def _make_ext(abasis: float = 0.0, long_accel: float | None = None) -> tuple[CarStateExt, SimpleNamespace]:
  CP = SimpleNamespace(flags=HyundaiFlags.HYBRID)
  CP_SP = SimpleNamespace(flags=0)
  ext = CarStateExt(CP, CP_SP)
  vl = _VL({"TCS13": {"aBasis": abasis}, "CLU13": {"CF_Clu_DTE": 100}})
  if long_accel is not None:
    vl["ESP12"] = {"LONG_ACCEL": long_accel}
  cp = SimpleNamespace(vl=vl)
  return ext, cp


def _run(ext: CarStateExt, cp, v_mph: float, a_ego: float, grade: float | None, frames: int = 300,
         abasis: float | None = None) -> structs.CarStateSP:
  """Run the power path `frames` times at fixed inputs so the LP filters settle."""
  ret = structs.CarState()
  ret.vEgo = v_mph * MPH_TO_MS
  ret.aEgo = a_ego
  if abasis is not None:
    cp.vl["TCS13"]["aBasis"] = abasis
  ext.grade_accel_external_ms2 = grade
  ret_sp = structs.CarStateSP()
  for _ in range(frames):
    ret_sp = structs.CarStateSP()
    ext._update_ev_limiter_signals(ret, ret_sp, cp)
  return ret_sp


def _expected(v_mph: float, a_net: float, aux_w: float = AUX_POWER_DEFAULT_W) -> float:
  v = v_mph * MPH_TO_MS
  wheel = max(0.0, VEHICLE_MASS_KG * v * a_net + road_load_power_w(v))
  return wheel / drivetrain_efficiency(v) + aux_w


class TestEfficiencyMap(unittest.TestCase):

  def test_endpoints_and_monotonic(self):
    self.assertAlmostEqual(drivetrain_efficiency(0.0), EFFICIENCY_LOW_SPEED)
    self.assertAlmostEqual(drivetrain_efficiency(EFFICIENCY_RAMP_END_MS), EFFICIENCY_HIGHWAY)
    self.assertAlmostEqual(drivetrain_efficiency(2 * EFFICIENCY_RAMP_END_MS), EFFICIENCY_HIGHWAY)
    prev = 0.0
    for mph in range(0, 80, 5):
      e = drivetrain_efficiency(mph * MPH_TO_MS)
      self.assertGreaterEqual(e, prev)
      self.assertGreater(e, 0.5)
      self.assertLessEqual(e, 1.0)
      prev = e


class TestPowerPath(unittest.TestCase):

  def test_steady_flat_highway_is_road_load_over_eta_plus_aux(self):
    ext, cp = _make_ext()
    sp = _run(ext, cp, v_mph=70.0, a_ego=0.0, grade=0.0)
    self.assertAlmostEqual(sp.estPowerInstantW, _expected(70.0, 0.0), delta=1.0)
    # 20-30 kW window: well below the 40 kW default threshold on flat road.
    self.assertGreater(sp.estPowerInstantW, 20_000.0)
    self.assertLess(sp.estPowerInstantW, 32_000.0)

  def test_abasis_does_not_enter_power(self):
    """The old formula multiplied aBasis by m·v (61 kW per m/s² at 70 mph)."""
    ext_a, cp_a = _make_ext(abasis=0.0)
    ext_b, cp_b = _make_ext(abasis=1.0)
    p_a = _run(ext_a, cp_a, 70.0, 0.0, 0.0).estPowerInstantW
    p_b = _run(ext_b, cp_b, 70.0, 0.0, 0.0).estPowerInstantW
    self.assertAlmostEqual(p_a, p_b, delta=1.0)
    # …but it is still published for the SCC-decel detector.
    self.assertAlmostEqual(_run(ext_b, cp_b, 70.0, 0.0, 0.0).accelDemand, 1.0)

  def test_uphill_adds_full_grade_power_no_cap_no_deadband(self):
    """iter15 capped grade at 10 kW and dropped the first 0.20 m/s². A real 3%
    grade at 70 mph (0.29 m/s²) is ~18 kW at the wheels and must all count."""
    ext, cp = _make_ext()
    flat = _run(ext, cp, 70.0, 0.0, 0.0).estPowerInstantW
    up = _run(ext, cp, 70.0, 0.0, 0.29).estPowerInstantW
    v = 70.0 * MPH_TO_MS
    expected_delta = VEHICLE_MASS_KG * v * 0.29 / drivetrain_efficiency(v)
    self.assertAlmostEqual(up - flat, expected_delta, delta=50.0)
    self.assertGreater(up - flat, 15_000.0)

  def test_downhill_cancels_equal_accel_but_never_negative(self):
    ext, cp = _make_ext()
    # accel 0.3 on a -0.3 downhill → net 0 → road load only
    p = _run(ext, cp, 60.0, 0.3, -0.3).estPowerInstantW
    self.assertAlmostEqual(p, _expected(60.0, 0.0), delta=50.0)
    # steep downhill, coasting → wheel power clamps at 0, aux remains
    p = _run(ext, cp, 60.0, 0.0, -0.5).estPowerInstantW
    self.assertAlmostEqual(p, AUX_POWER_DEFAULT_W, delta=1.0)

  def test_launch_reads_battery_not_wheel_power(self):
    """9 mph at 2.7 m/s²: wheel power ≈ 21 kW; battery ≈ 21/0.75 + 2.5 ≈ 31 kW."""
    ext, cp = _make_ext()
    p = _run(ext, cp, 9.0, 2.7, 0.0).estPowerInstantW
    v = 9.0 * MPH_TO_MS
    wheel = VEHICLE_MASS_KG * v * 2.7 + road_load_power_w(v)
    self.assertGreater(p, wheel + AUX_POWER_DEFAULT_W)
    self.assertAlmostEqual(p, _expected(9.0, 2.7), delta=50.0)
    self.assertGreater(p, 28_000.0)

  def test_creeping_reads_aux_baseline(self):
    ext, cp = _make_ext()
    p = _run(ext, cp, 2.0, 0.0, 0.0).estPowerInstantW
    self.assertGreater(p, AUX_POWER_DEFAULT_W)
    self.assertLess(p, AUX_POWER_DEFAULT_W + 2_000.0)

  def test_aux_attribute_from_card_is_used_and_clamped(self):
    ext, cp = _make_ext()
    # The attribute→_aux_power_w sync (with bounds) lives in update(); emulate it.
    ext.aux_power_w = 4000.0
    ext._aux_power_w = min(AUX_POWER_MAX_W, max(0.0, float(ext.aux_power_w)))
    p = _run(ext, cp, 60.0, 0.0, -0.5).estPowerInstantW
    self.assertAlmostEqual(p, 4000.0, delta=1.0)
    ext.aux_power_w = 1e9
    ext._aux_power_w = min(AUX_POWER_MAX_W, max(0.0, float(ext.aux_power_w)))
    p = _run(ext, cp, 60.0, 0.0, -0.5).estPowerInstantW
    self.assertAlmostEqual(p, AUX_POWER_MAX_W, delta=1.0)

  def test_legacy_long_accel_fallback_gives_same_net_accel(self):
    """Without kalman, grade = LONG_ACCEL - aEgo (filtered), so
    aEgo + grade → LONG_ACCEL. Steady uphill hold: LONG_ACCEL=0.3, aEgo=0."""
    ext, cp = _make_ext(long_accel=0.3)
    p = _run(ext, cp, 70.0, 0.0, None, frames=2000).estPowerInstantW
    self.assertAlmostEqual(p, _expected(70.0, 0.3), delta=300.0)

  def test_saturation_flag_retired_and_grade_cap_counter_zero(self):
    ext, cp = _make_ext(abasis=2.0)
    sp = _run(ext, cp, 70.0, 0.0, 0.5)
    self.assertFalse(sp.estPowerSaturated)
    self.assertEqual(sp.evLimiterGradePowerCappedFrames, 0)
    self.assertAlmostEqual(sp.evLimiterGradePowerRawW, VEHICLE_MASS_KG * 70.0 * MPH_TO_MS * 0.5, delta=1.0)


class TestFilters(unittest.TestCase):

  def test_hud_filter_is_symmetric(self):
    self.assertEqual(POWER_TAU_RISE_S, POWER_TAU_FALL_S)

  def test_aego_filter_rejects_single_frame_spike(self):
    """A one-frame 1 m/s² aEgo spike at 70 mph is 68 kW at the wheels raw; the
    0.2 s LP must keep the instantaneous estimate within a few kW of steady."""
    ext, cp = _make_ext()
    steady = _run(ext, cp, 70.0, 0.0, 0.0).estPowerInstantW
    spike = _run(ext, cp, 70.0, 1.0, 0.0, frames=1).estPowerInstantW
    alpha = 0.01 / (AEGO_FILTER_TAU_S + 0.01)
    self.assertLess(spike - steady, alpha * VEHICLE_MASS_KG * 70.0 * MPH_TO_MS * 1.0 / EFFICIENCY_HIGHWAY + 10.0)
    self.assertLess(spike - steady, 5_000.0)

  def test_hud_settles_to_instant_in_both_directions(self):
    ext, cp = _make_ext()
    hi = _run(ext, cp, 70.0, 0.5, 0.0, frames=500)
    self.assertAlmostEqual(hi.estPowerW, min(hi.estPowerInstantW, ext._ev_motor_cap_w) if ext._assume_ev_only else hi.estPowerInstantW,
                           delta=0.02 * hi.estPowerInstantW)
    lo = _run(ext, cp, 70.0, 0.0, 0.0, frames=500)
    self.assertAlmostEqual(lo.estPowerW, lo.estPowerInstantW, delta=0.02 * lo.estPowerInstantW)


if __name__ == "__main__":
  unittest.main()
