# ruff: noqa: ISC002
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter15 v2 (Section A) — hard-preempt fix for RECOVERY power guard.

Drive A/B forensic root cause (BUG #1): `_apply_recovery_power_guard()` returns
`STATE_SOFT_CAP_ACTIVE`, but `_publish()` internally calls
`_arbitrate_state_transition()` which enforces `MIN_ACTIVE_STATE_DWELL_FRAMES=200`
on active-state EXIT — silently demoting the guard's decision to a no-op for
up to 2 s. iter14 unit tests of `_apply_recovery_power_guard()` in isolation
passed because they tested the function's return value, not the published
state. iter15 fix: guard returns tuple `(new_state, guard_forced_transition)`
and caller passes `hard_preempt=guard_forced_transition` so ONLY the guard's
forced transition bypasses min-dwell.

Plan Section I items 1-5:
  1. test_guard_fires_immediate_state_transitions_within_one_frame
     (CRITICAL integration test — full update()→_publish() path)
  2. test_guard_fires_debounced_state_transitions_within_one_frame_after_debounce
  3. test_lockout_block_also_uses_hard_preempt
  4. test_non_guard_state_transitions_still_respect_min_dwell
  5. test_drive_85_trace_replay_no_recovery_capped_over_50ms
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.ev_limiter import (
  EVLimiter,
  STATE_SOFT_CAP_ACTIVE,
  STATE_RECOVERY_ACTIVE,
  MIN_ACTIVE_STATE_DWELL_FRAMES,
  MPH_TO_MS,
)

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
  est_power_control_w: float = 0.0
  est_power_instant_w: float = 0.0
  accel_demand: float = 0.0
  dte_raw: float = 100.0


@dataclass
class FakeCC:
  enabled: bool = False


def _make_limiter(power_threshold_kw: int = 40, dte_floor: int = 5):
  lim = EVLimiter(FakeCP(), FakeCPSP())
  lim._read_bool = lambda key, default: True if key == "EVLimiterEnabled" else default
  lim._read_int = lambda key, default: {
    "EVLimiterPowerThresholdKW": power_threshold_kw,
    "EVLimiterDTEFloor": dte_floor,
  }.get(key, default)
  return lim


def _engage(lim, end_frame=40, vEgo_mph=40.0, observed_mph=50.0):
  """Engage cruise with cluster above target so we land in RECOVERY/SOFT_CAP."""
  for f in range(5):
    _step(lim, f, cc_enabled=False, vEgo=vEgo_mph * MPH_TO_MS, observed_mph=observed_mph)
  _step(lim, 5, cc_enabled=True, vEgo=vEgo_mph * MPH_TO_MS, observed_mph=observed_mph,
        button=ButtonType.decelCruise)
  for f in range(6, end_frame):
    _step(lim, f, cc_enabled=True, vEgo=vEgo_mph * MPH_TO_MS, observed_mph=observed_mph)


def _step(lim, frame, cc_enabled, vEgo, observed_mph, button=None,
          est_power_w=0.0, est_power_control_w=None, abasis=0.0,
          brake=False, gas=False, aEgo=0.0):
  cs = FakeCS()
  cs.out.vEgo = vEgo
  cs.out.aEgo = aEgo
  cs.out.cruiseState.speed = observed_mph * MPH_TO_MS
  cs.out.brakePressed = brake
  cs.out.gasPressed = gas
  cs.out.buttonEvents = [FakeButtonEvent(pressed=True, type=button)] if button is not None else []
  cs.est_power_w = est_power_w
  cs.est_power_control_w = (est_power_control_w
                            if est_power_control_w is not None else est_power_w)
  cs.est_power_instant_w = cs.est_power_control_w
  cs.accel_demand = abasis
  cs.dte_raw = 100.0
  return lim.update(FakeCC(enabled=cc_enabled), cs, frame)


def _force_recovery(lim, frame: int) -> None:
  """Place the limiter into STATE_RECOVERY_ACTIVE just before `frame`.

  We bypass the arbiter (forcing self.state directly) and set
  `_state_entered_frame` so the time-in-state is mid-window — this is
  exactly the scenario where iter14's bug fires (state has accumulated
  some dwell but not enough for min-dwell to allow natural exit).
  """
  lim.state = STATE_RECOVERY_ACTIVE
  # Halfway into the dwell window — too early to exit naturally.
  lim._state_entered_frame = frame - (MIN_ACTIVE_STATE_DWELL_FRAMES // 2)
  # Clear lockout so the guard's RECOVERY→SOFT_CAP yield path is the active branch.
  lim._softcap_from_recovery_lockout_until = -10000
  lim._recovery_lockout_engaged = False


class TestGuardHardPreemptIntegration(unittest.TestCase):
  """CRITICAL: these tests use the FULL EVLimiter.update()→_publish()→arbiter
  path so they verify that the guard's RECOVERY→SOFT_CAP decision actually
  propagates to `self.state`. iter14 isolated tests of
  `_apply_recovery_power_guard()` missed BUG #1 because they did not exercise
  the arbiter's min-dwell rule.
  """

  def test_guard_fires_immediate_state_transitions_within_one_frame(self):
    """Plan #1: power_control_w = cap → state actually transitions on the next
    tick via hard_preempt. iter14 BUG #1 caused this to silently fail.

    Two yield-attribution paths exist:
      (a) State-derivation block catches power_too_high first and sets
          new_state=SOFT_CAP; iter15 guard observes BUG #1 path
          (prior_published=RECOVERY, new_state!=RECOVERY, power_should_yield)
          and sets guard_forced_transition=True.
      (b) Power_too_high condition not triggered upstream (e.g., vEgo gap not
          satisfied) but power_control_w >= cap, so the guard fires its own
          RECOVERY→SOFT_CAP and sets reason="immediate".

    Either path must yield `lim.state == STATE_SOFT_CAP_ACTIVE` and
    `guard_forced_transition=True`. We only assert that here.
    """
    lim = _make_limiter()
    _engage(lim, end_frame=40)
    _force_recovery(lim, frame=200)
    # est_power_control_w = cap (40 kW) → tier-1 immediate yield path
    btn, _ = _step(lim, 200, cc_enabled=True, vEgo=30.0 * MPH_TO_MS,
                   observed_mph=40.0, est_power_w=42_000.0,
                   est_power_control_w=42_000.0)
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                     "Guard's RECOVERY→SOFT_CAP decision must reach self.state "
                     "via the arbiter (hard_preempt=True). iter14 BUG #1 had "
                     "this stuck at RECOVERY for up to 2 s.")
    self.assertTrue(lim._guard_forced_transition_last_frame)

  def test_guard_fires_debounced_state_transitions_within_one_frame_after_debounce(self):
    """Plan #2: power_control_w at 0.95*cap for 3 frames → debounced yield →
    state transitions immediately after debounce.

    Use power below the state-derivation `power_too_high` threshold (38.5 kW
    vs 40 kW cap) so the state derivation block does NOT preempt; the guard
    is the sole source of the SOFT_CAP transition and `yield_reason` is
    set to "debounced".
    """
    lim = _make_limiter()
    _engage(lim, end_frame=40)
    _force_recovery(lim, frame=200)
    # est_power_w (state derivation) = 38.5 kW < 40 kW cap → power_too_high False.
    # est_power_control_w (guard) = 38.5 kW >= 0.95 * 40 = 38 kW → debounce counter ticks.
    _step(lim, 200, cc_enabled=True, vEgo=30.0 * MPH_TO_MS,
          observed_mph=40.0, est_power_w=38_500.0, est_power_control_w=38_500.0)
    _force_recovery(lim, frame=201)
    _step(lim, 201, cc_enabled=True, vEgo=30.0 * MPH_TO_MS,
          observed_mph=40.0, est_power_w=38_500.0, est_power_control_w=38_500.0)
    _force_recovery(lim, frame=202)
    # Third frame triggers the debounced yield.
    btn, _ = _step(lim, 202, cc_enabled=True, vEgo=30.0 * MPH_TO_MS,
                   observed_mph=40.0, est_power_w=38_500.0,
                   est_power_control_w=38_500.0)
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                     "Debounced yield (3 frames at 0.95*cap) must transition "
                     "state immediately on the 3rd frame via hard_preempt.")
    self.assertEqual(lim._power_guard_yield_reason_last, "debounced")
    self.assertTrue(lim._guard_forced_transition_last_frame)

  def test_lockout_block_also_uses_hard_preempt(self):
    """Plan #3: when guard fires RECOVERY→SOFT_CAP_ACTIVE under lockout
    (headroom_done=False), the transition is hard-preempted too — not held
    back by min-dwell. The lockout block also fires guard_forced_transition."""
    lim = _make_limiter()
    _engage(lim, end_frame=40)
    _force_recovery(lim, frame=200)
    # Engage lockout via prior yield by simulating one yield event.
    lim._recovery_lockout_engaged = True
    lim._softcap_from_recovery_lockout_until = 250   # not yet done
    lim._recovery_reentry_sustain = 0                  # headroom NOT done

    # Power_control below cap so guard does not re-fire `power_should_yield`,
    # but lockout still blocks RECOVERY → forces SOFT_CAP.
    btn, _ = _step(lim, 200, cc_enabled=True, vEgo=30.0 * MPH_TO_MS,
                   observed_mph=40.0, est_power_w=30_000.0,
                   est_power_control_w=30_000.0)
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                     "Lockout-block of RECOVERY must hard-preempt too.")
    self.assertTrue(lim._guard_forced_transition_last_frame,
                    "guard_forced_transition must be True on lockout-block path.")

  def test_non_guard_state_transitions_still_respect_min_dwell(self):
    """Plan #4: ONLY guard yields get hard_preempt=True. Other state
    transitions (e.g., normal IDLE→active path, or normal active→IDLE) must
    still respect min-dwell/sustain — narrow scope verification."""
    lim = _make_limiter()
    _engage(lim, end_frame=40)
    # Drive into SOFT_CAP normally.
    lim.user_target_speed = 51.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    _step(lim, 100, cc_enabled=True, vEgo=46.0 * MPH_TO_MS, observed_mph=60.0,
          est_power_w=10_000.0, est_power_control_w=10_000.0, abasis=0.1)
    # state should be SOFT_CAP_ACTIVE — natural transition (not guard).
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE)
    self.assertFalse(lim._guard_forced_transition_last_frame,
                     "Natural IDLE→SOFT_CAP transition must NOT set "
                     "guard_forced_transition.")
    # Now stop pushing — let SOFT_CAP try to exit. Should respect min-dwell.
    for f in range(101, 150):  # only 50 frames, less than 200 min-dwell
      _step(lim, f, cc_enabled=True, vEgo=46.0 * MPH_TO_MS, observed_mph=50.0,
            est_power_w=5_000.0, est_power_control_w=5_000.0, abasis=0.0)
    self.assertEqual(lim.state, STATE_SOFT_CAP_ACTIVE,
                     "Natural SOFT_CAP exit must wait for min-dwell — iter15 "
                     "scope is narrowly the guard yield path only.")

  def test_drive_85_trace_replay_no_recovery_capped_over_50ms(self):
    """Plan #5: replay representative drive-85 frames (high power_control_w
    sustained for ~1.3 s while at RECOVERY in iter14) and verify iter15 NEVER
    runs RECOVERY-while-capped for >50 ms (= 5 frames at 100 Hz).

    Frame pattern derived from forensic trace (Drive A first 30 frames at
    state=4 AND est_power_control_w>=40):
      154271.948 → power_control = 40.8 kW (just over cap)
      next ~130 frames: power_control 40-44 kW sustained
    Without the iter15 hard_preempt fix, state stayed at RECOVERY (4) for 1.31 s
    even though the guard returned SOFT_CAP_ACTIVE. With the fix, state must
    transition to SOFT_CAP_ACTIVE within 1-2 frames.
    """
    lim = _make_limiter()
    _engage(lim, end_frame=40)
    _force_recovery(lim, frame=200)
    # Simulate 200 frames (2 s) of sustained >= cap power while in RECOVERY.
    recovery_capped_frames = 0
    for f in range(200, 400):
      power_w = 41_000.0 + (f % 5) * 200.0   # 41-41.8 kW sustained
      _step(lim, f, cc_enabled=True, vEgo=30.0 * MPH_TO_MS, observed_mph=40.0,
            est_power_w=power_w, est_power_control_w=power_w)
      if lim.state == STATE_RECOVERY_ACTIVE and power_w >= 40_000.0:
        recovery_capped_frames += 1
    # iter14 BUG #1: this was 130+ frames. iter15 hard_preempt must hold it
    # to at most 5 frames (=50 ms) — typically 1 frame.
    self.assertLessEqual(recovery_capped_frames, 5,
                         f"RECOVERY-while-capped sustained for "
                         f"{recovery_capped_frames} frames (>50 ms). "
                         f"iter15 hard-preempt fix should hold this <=5.")


class TestGuardForcedTransitionFlag(unittest.TestCase):
  """Additional supporting tests for the `guard_forced_transition` bool flag
  semantics (R1-MF-A explicit tuple return; no double-counting)."""

  def test_flag_false_when_state_was_already_softcap(self):
    """R1-MF-A: guard returns guard_forced_transition=False when new_state was
    already STATE_SOFT_CAP_ACTIVE coming in (no state change)."""
    lim = _make_limiter()
    new_state, gft = lim._apply_recovery_power_guard(
      STATE_SOFT_CAP_ACTIVE, 50_000.0, 40_000.0, 100
    )
    # State unchanged → guard_forced_transition is False.
    self.assertEqual(new_state, STATE_SOFT_CAP_ACTIVE)
    self.assertFalse(gft, "Guard must not flag forced-transition when state "
                          "was already SOFT_CAP_ACTIVE (R1-MF-A).")

  def test_flag_true_only_on_actual_state_change(self):
    """R1-MF-A: flag True only when guard changes state away from candidate."""
    lim = _make_limiter()
    new_state, gft = lim._apply_recovery_power_guard(
      STATE_RECOVERY_ACTIVE, 50_000.0, 40_000.0, 100
    )
    self.assertEqual(new_state, STATE_SOFT_CAP_ACTIVE)
    self.assertTrue(gft, "Guard must flag forced-transition when state "
                         "actually changed from RECOVERY to SOFT_CAP.")

  def test_recovery_yield_episodes_strict_edge(self):
    """R2-MF-1: episode counter increments only when
    prior_published_state == STATE_RECOVERY_ACTIVE
    AND new_state == STATE_SOFT_CAP_ACTIVE AND guard_forced_transition.
    Verified via full update() path."""
    lim = _make_limiter()
    _engage(lim, end_frame=40)
    _force_recovery(lim, frame=200)
    pre = lim._evLimiter_recovery_yield_episodes
    _step(lim, 200, cc_enabled=True, vEgo=30.0 * MPH_TO_MS, observed_mph=40.0,
          est_power_w=42_000.0, est_power_control_w=42_000.0)
    self.assertEqual(lim._evLimiter_recovery_yield_episodes, pre + 1,
                     "RECOVERY→SOFT_CAP yield must increment episode counter.")

  def test_recovery_yield_episodes_not_incremented_when_prior_not_recovery(self):
    """R2-MF-1 negative case: if prior_published_state != RECOVERY (e.g.
    already SOFT_CAP), no episode increment."""
    lim = _make_limiter()
    _engage(lim, end_frame=40)
    # Force into SOFT_CAP so prior_published_state == SOFT_CAP at next tick.
    lim.state = STATE_SOFT_CAP_ACTIVE
    lim._state_entered_frame = 100
    pre = lim._evLimiter_recovery_yield_episodes
    _step(lim, 200, cc_enabled=True, vEgo=30.0 * MPH_TO_MS, observed_mph=40.0,
          est_power_w=42_000.0, est_power_control_w=42_000.0)
    self.assertEqual(lim._evLimiter_recovery_yield_episodes, pre,
                     "Episode counter must NOT increment when prior state "
                     "was not RECOVERY.")


if __name__ == "__main__":
  unittest.main()
