# ruff: noqa: ISC002
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter15 v2 (Section D) — post-RES quiet period for SOFT_CAP decrement.

Drive A/B observed +1/-2 high-speed oscillation: after the limiter emits RES,
SCC's accel command spikes briefly → est_power_control_w inflates → softcap-driven
SET fires → net cruise speed loss. iter15 v2 fix: after a RES emission, suppress
SOFT_CAP-driven SET decrement for `POST_RES_QUIET_PERIOD_FRAMES` (2 s) UNLESS
power genuinely exceeds cap by `POST_RES_HARD_OVERRIDE_FRAC` (1.05x).

Edge-detect events per R1-MF-D — counter increments on the TRANSITION from
non-suppressed-last-frame to suppressed-this-frame, NOT every frame within
the suppression window. Frame counter retained as separate field for forensics.

Plan Section I items 14-17 (overlapping numbering with standstill #14 — these
are the post-RES tests).
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from opendbc.car import structs
from opendbc.car.hyundai.values import Buttons, HyundaiFlags
from opendbc.sunnypilot.car.hyundai.ev_limiter import (
  EVLimiter,
  POST_RES_QUIET_PERIOD_FRAMES,
  POST_RES_HARD_OVERRIDE_FRAC,
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


def _engage_at_speed(lim, end_frame=40, vEgo_mph=60.0, observed_mph=58.0):
  """Engage cruise at highway speed."""
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


class TestPostResQuietConstants(unittest.TestCase):

  def test_quiet_period_2s(self):
    self.assertEqual(POST_RES_QUIET_PERIOD_FRAMES, 200)

  def test_hard_override_frac_1_05(self):
    self.assertAlmostEqual(POST_RES_HARD_OVERRIDE_FRAC, 1.05, places=4)


# ─────────────────────────────────────────────────────────────────────────
# Plan #14 (post-RES) — softcap decrement suppressed within 2 s of RES emit.
# Plan #15 — softcap decrement fires when power_far_over_cap even in quiet.
# Plan #16 — post-RES quiet releases after 2 s.
# Plan #17 — post-RES quiet does not affect other softcap paths.
# ─────────────────────────────────────────────────────────────────────────

class TestPostResQuietSuppression(unittest.TestCase):

  def test_softcap_decrement_suppressed_within_2s_of_res_emit(self):
    """Plan #14 (post-RES): just after RES emit + power slightly over cap,
    softcap-driven SET decrement is suppressed."""
    lim = _make_limiter()
    _engage_at_speed(lim)
    # Simulate RES emit at frame 100.
    lim._last_res_emit_frame = 100
    # 0.5 s later (within 2 s quiet) softcap predicate fires but power not
    # "far over" cap (40 kW < 1.05*40=42 kW).
    lim.user_target_speed = 60.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    pre_suppressed = lim._evLimiter_softcap_decrement_suppressed_frames
    pre_events = lim._evLimiter_softcap_decrement_suppressed_events
    btn, _ = _step(lim, 150, cc_enabled=True, vEgo=58.0 * MPH_TO_MS,
                   observed_mph=70.0,   # observed > target → set_too_high True
                   est_power_w=40_000.0, est_power_control_w=40_000.0)
    self.assertNotEqual(btn, Buttons.SET_DECEL,
                        "softcap-driven SET must be suppressed within RES quiet.")
    self.assertGreater(lim._evLimiter_softcap_decrement_suppressed_frames,
                       pre_suppressed,
                       "frame counter must tick when suppression active.")
    # Edge-detect: first frame of suppression should bump event counter.
    self.assertEqual(lim._evLimiter_softcap_decrement_suppressed_events,
                     pre_events + 1)
    self.assertTrue(lim._post_res_quiet_active_last)

  def test_softcap_decrement_fires_when_power_far_over_cap_even_in_quiet(self):
    """Plan #15: power > 1.05*cap overrides the quiet period — safety beats
    UX. SET still fires."""
    lim = _make_limiter()
    _engage_at_speed(lim)
    lim._last_res_emit_frame = 100   # RES emitted 0.5 s ago
    lim.user_target_speed = 60.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    pre_override = lim._evLimiter_post_res_hard_override_events
    btn, _ = _step(lim, 150, cc_enabled=True, vEgo=58.0 * MPH_TO_MS,
                   observed_mph=70.0,
                   # power = 1.10*cap = 44 kW > 1.05*cap → overrides quiet
                   est_power_w=44_000.0, est_power_control_w=44_000.0)
    self.assertEqual(btn, Buttons.SET_DECEL,
                     "power > 1.05*cap must override the quiet period.")
    self.assertGreater(lim._evLimiter_post_res_hard_override_events, pre_override,
                       "Hard-override counter must tick on this safety path.")

  def test_post_res_quiet_releases_after_2s(self):
    """Plan #16: after POST_RES_QUIET_PERIOD_FRAMES elapses, softcap SET fires
    normally again."""
    lim = _make_limiter()
    _engage_at_speed(lim)
    lim._last_res_emit_frame = 100
    lim.user_target_speed = 60.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    # Frame 100 + 201 = past quiet period
    test_frame = 100 + POST_RES_QUIET_PERIOD_FRAMES + 1
    btn, _ = _step(lim, test_frame, cc_enabled=True, vEgo=58.0 * MPH_TO_MS,
                   observed_mph=70.0,
                   est_power_w=40_000.0, est_power_control_w=40_000.0)
    self.assertEqual(btn, Buttons.SET_DECEL,
                     "After quiet period elapses, SET fires normally.")
    self.assertFalse(lim._post_res_quiet_active_last)

  def test_post_res_quiet_does_not_affect_other_softcap_paths(self):
    """Plan #17: post-RES suppression ONLY gates softcap-driven SET. It does
    NOT tick the suppression frame counter when softcap predicate is False
    (no `want_set` to gate). Capture pre-state RIGHT BEFORE the test step so
    any engage-time activity doesn't leak into the assertion."""
    lim = _make_limiter()
    # Engage at a higher observed (above vEgo) so no engage-time suppression.
    for f in range(5):
      _step(lim, f, cc_enabled=False, vEgo=60.0 * MPH_TO_MS, observed_mph=60.0)
    _step(lim, 5, cc_enabled=True, vEgo=60.0 * MPH_TO_MS, observed_mph=60.0,
          button=ButtonType.decelCruise)
    for f in range(6, 40):
      _step(lim, f, cc_enabled=True, vEgo=60.0 * MPH_TO_MS, observed_mph=60.0)
    # Capture state at frame 99 (just before quiet starts) so suppression
    # delta is observable per-frame.
    lim._last_res_emit_frame = 100
    pre = lim._evLimiter_softcap_decrement_suppressed_frames
    # Now: in-quiet window AND no softcap predicate.
    _step(lim, 150, cc_enabled=True, vEgo=60.0 * MPH_TO_MS,
          observed_mph=60.0,   # near target, no set_too_high; power=10kw < cap
          est_power_w=10_000.0, est_power_control_w=10_000.0)
    self.assertEqual(lim._evLimiter_softcap_decrement_suppressed_frames, pre,
                     "Suppression must not tick when softcap predicate is False.")
    self.assertTrue(lim._post_res_quiet_active_last,
                    "Quiet window still active, just no softcap path to gate.")


# ─────────────────────────────────────────────────────────────────────────
# R1-MF-D edge-detect: events vs frames.
# ─────────────────────────────────────────────────────────────────────────

class TestPostResQuietEdgeDetect(unittest.TestCase):

  def test_events_increment_only_once_per_suppression_window(self):
    """R1-MF-D: event counter must NOT increment every frame inside the window;
    only on the transition from non-suppressed to suppressed."""
    lim = _make_limiter()
    _engage_at_speed(lim)
    lim._last_res_emit_frame = 100
    lim.user_target_speed = 60.0 * MPH_TO_MS
    lim.last_set_frame = -10000
    pre_events = lim._evLimiter_softcap_decrement_suppressed_events
    pre_frames = lim._evLimiter_softcap_decrement_suppressed_frames
    # Drive 20 frames of sustained suppression.
    for f in range(150, 170):
      _step(lim, f, cc_enabled=True, vEgo=58.0 * MPH_TO_MS, observed_mph=70.0,
            est_power_w=40_000.0, est_power_control_w=40_000.0)
    # Events: incremented exactly ONCE (on the first suppressed frame).
    self.assertEqual(lim._evLimiter_softcap_decrement_suppressed_events,
                     pre_events + 1,
                     "Event counter must be edge-detected (R1-MF-D), incrementing "
                     "only once per suppression window.")
    # Frames: incremented every frame.
    self.assertGreaterEqual(lim._evLimiter_softcap_decrement_suppressed_frames,
                            pre_frames + 20)


if __name__ == "__main__":
  unittest.main()
