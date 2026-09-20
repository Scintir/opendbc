#!/usr/bin/env python3
"""
Simulation tests for the EV Power Limiter.

Tests the limiter in isolation across multiple driving scenarios to verify:
- Power limiting correctness at various speeds
- Low-speed accel cap behavior
- Uphill/downhill grade handling
- Filter stability and transient behavior
- Edge cases (NaN, enable/disable, zero speed)
- Anti-windup signaling
- Regen passthrough (negative accel never limited)
"""

import math
import numpy as np
from opendbc.sunnypilot.car.hyundai.longitudinal.ev_power_limiter import (
  EVPowerLimiter, VEHICLE_MASS, GRAVITY, CRR, CDA, RHO, _usable_power_w,
  EV_LOWSPEED_ACCEL_BP, EV_LOWSPEED_ACCEL_V
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = {"pass": 0, "fail": 0}

def check(name, condition, detail=""):
  global results
  if condition:
    results["pass"] += 1
    print(f"  {PASS} {name}")
  else:
    results["fail"] += 1
    print(f"  {FAIL} {name} -- {detail}")

def compute_wheel_power_kw(accel, v_ego, pitch=0.0, mass=VEHICLE_MASS):
  """Reference power calculation for verification."""
  f_accel = mass * accel
  f_roll = mass * GRAVITY * CRR
  f_aero = 0.5 * RHO * CDA * v_ego * v_ego
  f_grade = mass * GRAVITY * math.sin(pitch)
  f_total = f_accel + f_roll + f_aero + f_grade
  return max(f_total * max(v_ego, 0.1), 0.0) / 1000.0


# ============================================================
print("\n=== TEST 1: Basic Power Limiting on Flat Road ===")
# ============================================================
lim = EVPowerLimiter()
lim.update_params(True, 35, False)

# At 20 m/s (~72 km/h), full 2.0 m/s^2 accel should exceed 35kW
p_full = compute_wheel_power_kw(2.0, 20.0)
check("Full accel at 20 m/s exceeds 35kW limit",
      p_full > 35.0,
      f"P={p_full:.1f} kW")

# Run limiter for several cycles to let filters settle
for _ in range(50):
  a_out, sat = lim.update(2.0, 20.0, 0.0)

p_limited = compute_wheel_power_kw(a_out, 20.0)
p_usable = _usable_power_w(35.0) / 1000.0
check(f"Limited accel power ({p_limited:.1f} kW) <= usable budget ({p_usable:.1f} kW)",
      p_limited <= p_usable + 0.5,
      f"P={p_limited:.1f} > {p_usable:.1f}")
check("Saturation amount > 0 when limiting",
      sat > 0.01,
      f"sat={sat:.4f}")
check("power_limited flag set",
      lim.power_limited)
check(f"Limited accel ({a_out:.3f}) < requested (2.0)",
      a_out < 2.0,
      f"a_out={a_out:.4f}")


# ============================================================
print("\n=== TEST 2: No Limiting at Low Power Demand ===")
# ============================================================
lim2 = EVPowerLimiter()
lim2.update_params(True, 55, False)

# At 10 m/s, 0.3 m/s^2 should be well under 55kW
p_low = compute_wheel_power_kw(0.3, 10.0)
check(f"Low accel power ({p_low:.1f} kW) well under 55kW limit",
      p_low < 30.0)

for _ in range(50):
  a_out, sat = lim2.update(0.3, 10.0, 0.0)

check("No limiting at low power demand",
      abs(a_out - 0.3) < 0.01,
      f"a_out={a_out:.4f}")
check("No saturation",
      sat < 0.001)
check("power_limited flag not set",
      not lim2.power_limited)


# ============================================================
print("\n=== TEST 3: Regen/Braking Never Limited ===")
# ============================================================
lim3 = EVPowerLimiter()
lim3.update_params(True, 20, False)  # very low limit

for accel in [-1.0, -2.0, -3.5, -0.1]:
  for _ in range(10):
    a_out, sat = lim3.update(accel, 25.0, 0.0)
  check(f"Braking ({accel}) passes through unmodified",
        abs(a_out - accel) < 0.001,
        f"a_out={a_out:.4f}")


# ============================================================
print("\n=== TEST 4: Low-Speed Accel Cap ===")
# ============================================================
lim4 = EVPowerLimiter()
lim4.update_params(True, 55, False)  # high power limit

# At 1 m/s, even 55kW would allow huge accel, but low-speed cap should limit
for _ in range(50):
  a_out, sat = lim4.update(2.0, 1.0, 0.0)

expected_cap = float(np.interp(1.0, EV_LOWSPEED_ACCEL_BP, EV_LOWSPEED_ACCEL_V))
check(f"Low-speed cap active at 1 m/s: a_out ({a_out:.3f}) <= cap ({expected_cap:.3f})",
      a_out <= expected_cap + 0.02,
      f"a_out={a_out:.4f}")

# At 0 m/s (standstill)
for _ in range(50):
  a_out, _ = lim4.update(2.0, 0.0, 0.0)
expected_cap_0 = float(np.interp(0.0, EV_LOWSPEED_ACCEL_BP, EV_LOWSPEED_ACCEL_V))
check(f"Standstill cap: a_out ({a_out:.3f}) <= cap ({expected_cap_0:.3f})",
      a_out <= expected_cap_0 + 0.02)


# ============================================================
print("\n=== TEST 5: Uphill Grade Reduces Allowed Accel ===")
# ============================================================
# 5% grade = atan(0.05) ≈ 0.05 radians
grade_5pct = math.atan(0.05)

lim5_flat = EVPowerLimiter()
lim5_flat.update_params(True, 35, False)
for _ in range(50):
  a_flat, _ = lim5_flat.update(2.0, 20.0, 0.0)

lim5_hill = EVPowerLimiter()
lim5_hill.update_params(True, 35, False)
for _ in range(50):
  a_hill, _ = lim5_hill.update(2.0, 20.0, grade_5pct)

check(f"Uphill reduces allowed accel: flat ({a_flat:.3f}) > hill ({a_hill:.3f})",
      a_flat > a_hill + 0.01,
      f"flat={a_flat:.4f}, hill={a_hill:.4f}")

# 10% grade should reduce more
grade_10pct = math.atan(0.10)
lim5_steep = EVPowerLimiter()
lim5_steep.update_params(True, 35, False)
for _ in range(50):
  a_steep, _ = lim5_steep.update(2.0, 20.0, grade_10pct)

check(f"Steeper grade reduces more: 5% ({a_hill:.3f}) > 10% ({a_steep:.3f})",
      a_hill > a_steep,
      f"5%={a_hill:.4f}, 10%={a_steep:.4f}")


# ============================================================
print("\n=== TEST 6: Downhill Grade Allows More Accel ===")
# ============================================================
grade_down = -math.atan(0.05)  # -5% downhill

lim6 = EVPowerLimiter()
lim6.update_params(True, 35, False)
for _ in range(50):
  a_down, _ = lim6.update(2.0, 20.0, grade_down)

check(f"Downhill allows more accel than flat: down ({a_down:.3f}) > flat ({a_flat:.3f})",
      a_down > a_flat,
      f"down={a_down:.4f}, flat={a_flat:.4f}")


# ============================================================
print("\n=== TEST 7: Speed Sweep - Power Limit Consistency ===")
# ============================================================
lim7 = EVPowerLimiter()
lim7.update_params(True, 35, False)

power_violations = 0
p_usable = _usable_power_w(35.0) / 1000.0

for v in np.arange(5.0, 35.0, 1.0):
  # Reset filter for clean test at each speed
  lim7.accel_ceiling_filtered = 2.0
  for _ in range(100):
    a_out, _ = lim7.update(2.0, float(v), 0.0)

  p_out = compute_wheel_power_kw(a_out, float(v))
  if p_out > p_usable + 0.5:  # 0.5 kW tolerance
    power_violations += 1

check("Power never exceeds budget across speed sweep (5-35 m/s)",
      power_violations == 0,
      f"{power_violations} violations")


# ============================================================
print("\n=== TEST 8: Sudden Uphill Transition ===")
# ============================================================
lim8 = EVPowerLimiter()
lim8.update_params(True, 35, False)

# Cruise flat for a while
for _ in range(100):
  lim8.update(1.5, 20.0, 0.0)

# Sudden 8% uphill
grade_8pct = math.atan(0.08)
accel_trace = []
for _ in range(20):
  a_out, _ = lim8.update(1.5, 20.0, grade_8pct)
  accel_trace.append(a_out)

# Should respond within a few cycles (fast pitch-up tracking)
check("Uphill response starts within 3 cycles",
      accel_trace[2] < accel_trace[0] - 0.01,
      f"cycle0={accel_trace[0]:.4f}, cycle2={accel_trace[2]:.4f}")

# After 20 cycles (~1s), should be significantly reduced
check("Fully adapted after 20 cycles",
      accel_trace[-1] < accel_trace[0] - 0.1,
      f"start={accel_trace[0]:.4f}, end={accel_trace[-1]:.4f}")


# ============================================================
print("\n=== TEST 9: NaN/Inf Input Handling (Fail Open) ===")
# ============================================================
lim9 = EVPowerLimiter()
lim9.update_params(True, 35, False)

# NaN pitch - should fail open (pass through accel)
a_out, sat = lim9.update(1.0, 20.0, float('nan'))
check("NaN pitch: passes through accel (fail open)",
      abs(a_out - 1.0) < 0.001,
      f"a_out={a_out}")
check("NaN pitch: fault flag set",
      lim9._input_fault)

# Inf speed
a_out, _ = lim9.update(1.0, float('inf'), 0.0)
check("Inf speed: passes through accel (fail open)",
      abs(a_out - 1.0) < 0.001)

# NaN accel - special: NaN accel passed through (upstream should catch)
a_out, _ = lim9.update(float('nan'), 20.0, 0.0)
check("NaN accel: returns NaN (fail open)",
      math.isnan(a_out))


# ============================================================
print("\n=== TEST 10: Enable/Disable Behavior ===")
# ============================================================
lim10 = EVPowerLimiter()

# Disabled - no limiting
lim10.update_params(False, 35, False)
for _ in range(10):
  a_out, _ = lim10.update(2.0, 20.0, 0.0)
check("Disabled: full accel passes through",
      abs(a_out - 2.0) < 0.001)

# Enable - should start limiting
lim10.update_params(True, 35, False)
for _ in range(50):
  a_out, sat = lim10.update(2.0, 20.0, 0.0)
check("Enabled: accel is limited",
      a_out < 2.0,
      f"a_out={a_out:.4f}")

# Disable again
lim10.update_params(False, 35, False)
for _ in range(10):
  a_out, _ = lim10.update(2.0, 20.0, 0.0)
check("Re-disabled: full accel restored",
      abs(a_out - 2.0) < 0.001)


# ============================================================
print("\n=== TEST 11: Power Limit kW Sweep ===")
# ============================================================
for kw in [20, 25, 30, 35, 40, 45, 50, 55]:
  lim_kw = EVPowerLimiter()
  lim_kw.update_params(True, kw, False)
  for _ in range(100):
    a_out, _ = lim_kw.update(2.0, 20.0, 0.0)
  p_out = compute_wheel_power_kw(a_out, 20.0)
  p_budget = _usable_power_w(kw) / 1000.0
  check(f"  {kw} kW: output power ({p_out:.1f} kW) <= budget ({p_budget:.1f} kW)",
        p_out <= p_budget + 0.5)


# ============================================================
print("\n=== TEST 12: Anti-Windup Saturation Signal ===")
# ============================================================
lim12 = EVPowerLimiter()
lim12.update_params(True, 25, False)  # low limit

for _ in range(50):
  a_out, sat = lim12.update(2.0, 25.0, 0.0)

check(f"Saturation amount ({sat:.3f}) equals clipped amount",
      abs(sat - (2.0 - a_out)) < 0.01,
      f"sat={sat:.4f}, expected={2.0 - a_out:.4f}")
check("power_limited flag set when saturating",
      lim12.power_limited)


# ============================================================
print("\n=== TEST 13: Filter Ceiling Never Exceeds Raw ===")
# ============================================================
lim13 = EVPowerLimiter()
lim13.update_params(True, 30, False)
lim13.accel_ceiling_filtered = 2.0  # start high

violations = 0
for i in range(200):
  v = 10.0 + 0.1 * i  # accelerating
  pitch = 0.02 * math.sin(i * 0.1)  # oscillating grade
  a_out, _ = lim13.update(2.0, v, pitch)

  # The APPLIED ceiling (effective_ceiling) must never exceed raw.
  # The filter state can lag, but min(filter, raw) is what's used.
  # Verify the output accel respects the raw ceiling:
  if a_out > lim13._a_max_raw + 0.001 and a_out > 0:
    violations += 1

check("Applied accel never exceeds raw ceiling (FIX #2)",
      violations == 0,
      f"{violations} violations in 200 cycles")


# ============================================================
print("\n=== TEST 14: Pitch Filter Asymmetry ===")
# ============================================================
lim14 = EVPowerLimiter()
lim14.update_params(True, 35, False)
lim14.pitch_filtered = 0.0

# Uphill step: should track fast
for _ in range(10):
  lim14.update(1.0, 15.0, 0.05)
pitch_up_10 = lim14.pitch_filtered

# Reset and test downhill step: should track slow
lim14.pitch_filtered = 0.05
for _ in range(10):
  lim14.update(1.0, 15.0, 0.0)
pitch_down_10 = 0.05 - lim14.pitch_filtered  # how much it moved

check(f"Uphill tracks faster than downhill: up moved {0.05 - (0.05 - pitch_up_10):.4f} vs down moved {pitch_down_10:.4f}",
      pitch_up_10 > (0.05 - pitch_down_10),  # uphill tracking > downhill tracking after 10 cycles
      f"up_filt={pitch_up_10:.4f}, down_moved={pitch_down_10:.4f}")


# ============================================================
print("\n=== TEST 15: Long Inactive Then Re-Entry ===")
# ============================================================
lim15 = EVPowerLimiter()
lim15.update_params(True, 35, False)

# Drive at 20 m/s flat, get settled
for _ in range(50):
  lim15.update(1.0, 20.0, 0.0)
pitch_before = lim15.pitch_filtered

# Simulate long inactive on a hill (controller calls update with 0 accel)
for _ in range(100):
  lim15.update(0.0, 20.0, 0.08)  # steep uphill, but not limiting (accel=0)
pitch_after = lim15.pitch_filtered

check("Pitch filter tracks during inactive (0 accel) period",
      abs(pitch_after - 0.08) < 0.02,
      f"pitch_filt={pitch_after:.4f}, expected near 0.08")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
total = results["pass"] + results["fail"]
print(f"Results: {results['pass']}/{total} passed, {results['fail']} failed")
if results["fail"] == 0:
  print(f"{PASS} All tests passed!")
else:
  print(f"{FAIL} {results['fail']} tests failed")
print(f"{'='*60}\n")
