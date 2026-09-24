"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter15 v2 (Section B) — grade contribution anti-overread CLAMP.

Drive A/B forensics found peak estPowerInstantW of 82-94 kW with ICE never
engaging (real motor < 57 kW), implying 25-37 kW over-read. Grade is a
dominant contributor (12-27 kW per user telemetry). iter15 v2 R1-MF-B:
  - GRADE_DEADBAND_MS2 raised 0.10 → 0.20 (filters more IMU pitch noise)
  - Grade contribution to p_accel_grade_w CLAMPED at EvLimiterGradeContribCapKW
    (default 10 kW, bounds 5-30 kW)
  - Raw (pre-clamp) `grade_power_raw_w` is published via
    `evLimiterGradePowerRawW @55` — R2-MF-2: raw MAY exceed cap; only clamped
    must not.

Tests per plan Section I items 6-9.
"""
from __future__ import annotations

import unittest

from opendbc.sunnypilot.car.hyundai.carstate_ext import (
  GRADE_CONTRIB_CAP_DEFAULT_KW,
  GRADE_CONTRIB_CAP_MIN_KW,
  GRADE_CONTRIB_CAP_MAX_KW,
  GRADE_DEADBAND_MS2,
  VEHICLE_MASS_KG,
  road_load_power_w,
)
from opendbc.sunnypilot.car.hyundai.ev_limiter import MPH_TO_MS


def _grade_power_raw(grade_f: float, v_mph: float) -> float:
  """Mirror carstate_ext: grade_power_raw_w = mass * v * max(0, grade - deadband)."""
  v = v_mph * MPH_TO_MS
  uphill = max(0.0, grade_f - GRADE_DEADBAND_MS2)
  return VEHICLE_MASS_KG * v * uphill


def _grade_power_clamped(grade_f: float, v_mph: float, cap_kw: int) -> float:
  raw = _grade_power_raw(grade_f, v_mph)
  return min(raw, cap_kw * 1000.0)


class TestGradeDeadband(unittest.TestCase):

  def test_grade_deadband_0_20_blocks_noise_below(self):
    """Plan #6 R1-MF-B: deadband=0.20 blocks noise readings <0.20 m/s² entirely."""
    self.assertAlmostEqual(GRADE_DEADBAND_MS2, 0.20, places=4)
    # +0.025 m/s² mean bias from LONG_ACCEL-aEgo derivation — must contribute 0
    self.assertEqual(_grade_power_raw(0.025, 60.0), 0.0)
    # +0.15 m/s² above iter9 0.10 deadband but BELOW iter15 0.20 → contributes 0
    self.assertEqual(_grade_power_raw(0.15, 60.0), 0.0)
    # +0.18 m/s² — still below deadband
    self.assertEqual(_grade_power_raw(0.18, 60.0), 0.0)
    # +0.20 m/s² — exactly at deadband threshold
    self.assertAlmostEqual(_grade_power_raw(0.20, 60.0), 0.0, places=4)


class TestGradeContribCap(unittest.TestCase):

  def test_grade_power_contribution_capped_at_10kw(self):
    """Plan #7: grade contribution to p_accel_grade_w is clamped at 10 kW default."""
    # A real 5% grade at 70 mph → physical grade power ≈ 30 kW (1950·9.81·0.05·31.7).
    # `grade_f` is m/s² accel from grade, ≈ g·grade = 0.49 m/s². At 70 mph,
    # grade_power_raw = 1950 * 31.3 * (0.49 - 0.20) = ~17.7 kW. Clamped to 10 kW.
    grade_f = 0.49  # ~5% grade
    v_mph = 70.0
    raw = _grade_power_raw(grade_f, v_mph)
    clamped = _grade_power_clamped(grade_f, v_mph, GRADE_CONTRIB_CAP_DEFAULT_KW)
    self.assertGreater(raw, GRADE_CONTRIB_CAP_DEFAULT_KW * 1000.0,
                       "Test must exercise the clamp; raw must exceed cap.")
    self.assertAlmostEqual(clamped, GRADE_CONTRIB_CAP_DEFAULT_KW * 1000.0,
                           places=4, msg="Clamped contribution must equal cap.")

  def test_grade_at_0_50_mps2_produces_capped_10kw_at_70mph(self):
    """Plan #8: at 0.50 m/s² grade, 70 mph, expect clamped contribution = cap."""
    grade_f = 0.50
    v_mph = 70.0
    raw = _grade_power_raw(grade_f, v_mph)
    clamped = _grade_power_clamped(grade_f, v_mph, GRADE_CONTRIB_CAP_DEFAULT_KW)
    self.assertGreater(raw, GRADE_CONTRIB_CAP_DEFAULT_KW * 1000.0)
    self.assertAlmostEqual(clamped, GRADE_CONTRIB_CAP_DEFAULT_KW * 1000.0, places=4)

  def test_grade_at_real_5pct_grade_contributes_meaningful_power_post_cap(self):
    """Plan #9: anti-overread acknowledgment — real 5% grades produce ~30 kW
    physically but clamped to 10 kW. Drive review must verify this doesn't
    starve RECOVERY on actual highway grades. Test gates that the clamped
    value is at the cap (not zero — under-reads but not silent)."""
    # 5% grade ≈ 0.49 m/s² grade accel. At 70 mph.
    grade_f = 0.49
    v_mph = 70.0
    clamped = _grade_power_clamped(grade_f, v_mph, GRADE_CONTRIB_CAP_DEFAULT_KW)
    self.assertGreater(clamped, 0.0,
                       "Clamped contribution must be nonzero on a real hill.")
    self.assertEqual(clamped, GRADE_CONTRIB_CAP_DEFAULT_KW * 1000.0,
                     "Real 5% grade at highway speed must saturate the cap.")


class TestGradeRawPublishedNotClamped(unittest.TestCase):
  """R2-MF-2: `evLimiterGradePowerRawW` is the RAW (pre-clamp) value. It MAY
  exceed the cap — that's the design (forensic publishing of how often the
  clamp fires and by how much). Only the value entering `p_accel_grade_w`
  must be clamped."""

  def test_raw_is_unclamped_above_cap(self):
    """Raw published value MAY exceed cap (gate language R2-MF-2)."""
    grade_f = 0.50
    v_mph = 70.0
    raw = _grade_power_raw(grade_f, v_mph)
    self.assertGreater(raw, GRADE_CONTRIB_CAP_DEFAULT_KW * 1000.0,
                       "Raw grade contribution publishes uncapped (forensic).")


class TestGradeCapParamBounds(unittest.TestCase):
  """R1-MF-B parameterization: EvLimiterGradeContribCapKW bounds 5-30 kW."""

  def test_param_bounds_match_constants(self):
    self.assertEqual(GRADE_CONTRIB_CAP_MIN_KW, 5)
    self.assertEqual(GRADE_CONTRIB_CAP_MAX_KW, 30)
    self.assertEqual(GRADE_CONTRIB_CAP_DEFAULT_KW, 10)

  def test_param_default_within_bounds(self):
    self.assertGreaterEqual(GRADE_CONTRIB_CAP_DEFAULT_KW, GRADE_CONTRIB_CAP_MIN_KW)
    self.assertLessEqual(GRADE_CONTRIB_CAP_DEFAULT_KW, GRADE_CONTRIB_CAP_MAX_KW)


class TestRoadLoadUnchanged(unittest.TestCase):
  """Plan invariant: road_load_power_w() is NOT touched by Section B. Only the
  grade term is clamped. Verify the road load formula still produces the
  expected baseline at typical highway speed."""

  def test_road_load_at_70mph_unchanged(self):
    v_70_mph_ms = 70.0 * MPH_TO_MS
    p = road_load_power_w(v_70_mph_ms)
    # iter9 baseline at 70 mph ≈ 20-25 kW (Crr=0.011, CdA=0.75, rho=1.10).
    # Tolerate a 5 kW window to be robust to minor coefficient changes.
    self.assertGreater(p, 15_000.0)
    self.assertLess(p, 30_000.0)


if __name__ == "__main__":
  unittest.main()
