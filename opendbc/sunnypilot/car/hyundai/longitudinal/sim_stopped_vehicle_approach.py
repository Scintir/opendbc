#!/usr/bin/env python3
"""
Stopped Vehicle Approach (SVA) Simulation

Simulates approaching a stopped vehicle from various initial conditions
and reports braking performance: stopping distance, min gap, max decel,
and whether the vehicle stops safely.

Usage:
  python sim_stopped_vehicle_approach.py
"""

import sys
from dataclasses import dataclass

from opendbc.sunnypilot.car.hyundai.longitudinal.stopped_vehicle_approach import (
  StoppedVehicleApproach, MIN_STOP_GAP
)

DT = 0.05  # 20Hz planner cycle


@dataclass
class FakeLead:
  status: bool = True
  dRel: float = 80.0
  vLeadK: float = 0.0
  vLead: float = 0.0
  yRel: float = 0.0
  aLeadK: float = 0.0
  aLeadTau: float = 1.5
  modelProb: float = 0.95


@dataclass
class SimResult:
  scenario: str
  initial_speed_mph: float
  initial_distance_m: float
  final_gap_m: float
  final_gap_ft: float
  max_decel: float
  stop_time_s: float
  safe_stop: bool
  min_gap_m: float
  states_visited: list


def run_scenario(name: str, d_init: float, v_init: float, lead_v: float = 0.0,
                 lead_y: float = 0.3, model_prob: float = 0.95,
                 mpc_base_accel: float = -0.8) -> SimResult:
  """Run a single approach scenario.

  Args:
    name: Scenario description
    d_init: Initial distance to lead (m)
    v_init: Initial ego speed (m/s)
    lead_v: Lead vehicle speed (m/s)
    lead_y: Lead lateral offset (m)
    model_prob: Model probability (0-1)
    mpc_base_accel: Base MPC acceleration (simulates typical MPC output)
  """
  sva = StoppedVehicleApproach(dt=DT)
  sva.update_params(True, False)

  v_ego = v_init
  d_rel = d_init
  max_decel = 0.0
  min_gap = d_init
  states_visited = set()
  t = 0.0

  for step in range(2000):  # Max 100 seconds
    t = step * DT

    # Create lead with current distance
    lead = FakeLead(
      dRel=d_rel,
      vLeadK=lead_v,
      vLead=lead_v,
      yRel=lead_y,
      modelProb=model_prob
    )

    # Simulate MPC output: gradually increases braking as distance closes
    # Real MPC uses comfort-brake of 2.5 m/s^2 and ramps in based on safe distance
    d_safe = (v_ego ** 2) / (2 * 2.5) + 1.45 * v_ego + 6.0  # get_safe_obstacle_distance
    d_deficit = d_safe - d_rel
    if d_deficit > 0:
      # MPC commands braking proportional to deficit
      mpc_accel = max(-2.5, -d_deficit * 0.15)
    else:
      mpc_accel = mpc_base_accel

    # SVA update
    a_out, should_stop = sva.update(lead, v_ego, mpc_accel, False)

    # Track stats
    max_decel = min(max_decel, a_out)
    min_gap = min(min_gap, d_rel)
    states_visited.add(sva.state.name)

    # Apply acceleration (semi-implicit Euler for better accuracy)
    v_prev = v_ego
    v_ego = max(0.0, v_ego + a_out * DT)
    v_avg = (v_prev + v_ego) / 2.0
    d_rel = d_rel - (v_avg - lead_v) * DT

    # Stop condition: ego velocity near zero
    if v_ego < 0.05 and d_rel > 0:
      break

    # Collision check
    if d_rel < 0.3:
      break

  final_gap_ft = d_rel * 3.28084
  safe = d_rel >= MIN_STOP_GAP * 0.8  # Within 80% of target gap

  return SimResult(
    scenario=name,
    initial_speed_mph=v_init * 2.237,
    initial_distance_m=d_init,
    final_gap_m=round(d_rel, 2),
    final_gap_ft=round(final_gap_ft, 1),
    max_decel=round(max_decel, 2),
    stop_time_s=round(t, 1),
    safe_stop=safe,
    min_gap_m=round(min_gap, 2),
    states_visited=sorted(states_visited),
  )


def main():
  scenarios = [
    # Normal highway approach scenarios
    ("Highway 45mph, 80m", 80.0, 20.0, 0.0),
    ("Highway 55mph, 100m", 100.0, 24.5, 0.0),
    ("Highway 65mph, 120m", 120.0, 29.0, 0.0),

    # Close approach scenarios (the problematic ones)
    ("Close 45mph, 50m", 50.0, 20.0, 0.0),
    ("Close 55mph, 60m", 60.0, 24.5, 0.0),
    ("Close 35mph, 35m", 35.0, 15.6, 0.0),

    # Very close / late detection
    ("Late detect 45mph, 40m", 40.0, 20.0, 0.0),
    ("Late detect 35mph, 25m", 25.0, 15.6, 0.0),
    ("Late detect 25mph, 20m", 20.0, 11.2, 0.0),

    # Low speed approach
    ("City 25mph, 30m", 30.0, 11.2, 0.0),
    ("City 15mph, 15m", 15.0, 6.7, 0.0),

    # Nearly-stopped lead (rolling slowly)
    ("Slow lead 45mph, 80m, lead 2mph", 80.0, 20.0, 0.9),

    # Off-path lead (false positive test)
    ("Off-path lead 45mph, 60m", 60.0, 20.0, 0.0),

    # Low confidence lead
    ("Low prob lead 45mph, 60m", 60.0, 20.0, 0.0),
  ]

  results = []
  for s in scenarios:
    name = s[0]
    kwargs = {"d_init": s[1], "v_init": s[2], "lead_v": s[3]}

    # Special cases
    if "Off-path" in name:
      kwargs["lead_y"] = 3.0
    if "Low prob" in name:
      kwargs["model_prob"] = 0.3

    result = run_scenario(name, **kwargs)
    results.append(result)

  # Print results
  print("\n" + "=" * 100)
  print("STOPPED VEHICLE APPROACH SIMULATION RESULTS")
  print("=" * 100)
  print(f"{'Scenario':<40} {'Speed':>6} {'Dist':>5} {'Gap(m)':>7} {'Gap(ft)':>8} {'MaxG':>6} {'Time':>5} {'Safe':>5}")
  print("-" * 100)

  pass_count = 0
  fail_count = 0
  for r in results:
    status = "PASS" if r.safe_stop else "FAIL"
    if r.safe_stop:
      pass_count += 1
    else:
      fail_count += 1

    head = f"{r.scenario:<40} {r.initial_speed_mph:>5.0f}mph {r.initial_distance_m:>4.0f}m"
    gap = f"{r.final_gap_m:>6.2f}m {r.final_gap_ft:>7.1f}ft"
    tail = f"{r.max_decel:>5.2f}g {r.stop_time_s:>4.1f}s {status:>5}"
    print(f"{head} {gap} {tail}")

  print("-" * 100)
  print(f"Results: {pass_count} PASS, {fail_count} FAIL out of {len(results)} scenarios")
  print(f"Target minimum gap: {MIN_STOP_GAP:.1f}m ({MIN_STOP_GAP * 3.28084:.1f}ft)")
  print()

  # Detailed results for failed scenarios
  if fail_count > 0:
    print("\nFAILED SCENARIO DETAILS:")
    for r in results:
      if not r.safe_stop:
        print(f"  {r.scenario}:")
        print(f"    Final gap: {r.final_gap_m}m ({r.final_gap_ft}ft)")
        print(f"    Min gap: {r.min_gap_m}m")
        print(f"    Max decel: {r.max_decel} m/s^2 ({r.max_decel / -9.81:.2f}g)")
        print(f"    States: {r.states_visited}")

  # Print state transition details
  print("\nSTATE TRANSITIONS:")
  for r in results:
    print(f"  {r.scenario:<40} States: {', '.join(r.states_visited)}")

  return 0 if fail_count == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
