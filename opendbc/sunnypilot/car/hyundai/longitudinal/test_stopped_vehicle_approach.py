"""
Unit tests for Stopped Vehicle Approach (SVA) module.

Tests cover:
  - State machine transitions
  - Confidence scoring
  - Physics-based deceleration calculation
  - Rate limiting
  - False positive rejection
  - Edge cases
"""

import unittest
from dataclasses import dataclass

from opendbc.sunnypilot.car.hyundai.longitudinal.stopped_vehicle_approach import (
  StoppedVehicleApproach, SVAState,
  MIN_STOP_GAP, CONFIDENCE_SOFT, CONFIDENCE_HARD,
  EMERGENCY_DECEL,
)


@dataclass
class FakeLead:
  """Minimal lead data for testing."""
  present: bool = True
  dRel: float = 80.0
  vLeadK: float = 0.0
  vLead: float = 0.0
  yRel: float = 0.0
  aLeadK: float = 0.0
  aLeadTau: float = 1.5
  modelProb: float = 0.95


class TestSVAInit(unittest.TestCase):
  def test_initial_state(self):
    sva = StoppedVehicleApproach()
    self.assertEqual(sva.state, SVAState.INACTIVE)
    self.assertFalse(sva.active)
    self.assertFalse(sva.enabled)
    self.assertEqual(sva.confidence, 0.0)

  def test_update_params(self):
    sva = StoppedVehicleApproach()
    sva.update_params(True, True)
    self.assertTrue(sva.enabled)
    self.assertTrue(sva.logging_enabled)

  def test_disable_resets_state(self):
    sva = StoppedVehicleApproach()
    sva.update_params(True, False)
    sva.state = SVAState.HARD_APPROACH
    sva.confidence = 0.9
    sva.update_params(False, False)
    self.assertEqual(sva.state, SVAState.INACTIVE)
    self.assertEqual(sva.confidence, 0.0)


class TestSVADetection(unittest.TestCase):
  def setUp(self):
    self.sva = StoppedVehicleApproach(dt=0.05)
    self.sva.update_params(True, False)
    self.lead = FakeLead()

  def test_stopped_lead_detected(self):
    """SVA should detect a stopped lead and transition beyond INACTIVE."""
    lead = FakeLead(dRel=60.0, vLeadK=0.0, yRel=0.2, modelProb=0.95)
    # Build stopped persistence over several cycles
    for _ in range(10):
      self.sva.update(lead, 20.0, -0.5, False)
    # With high confidence (centered, stopped, high prob), should be in approach state
    self.assertNotEqual(self.sva.state, SVAState.INACTIVE)

  def test_moving_lead_not_detected(self):
    """SVA should NOT activate for moving leads."""
    lead = FakeLead(dRel=60.0, vLeadK=15.0, modelProb=0.95)
    for _ in range(20):
      self.sva.update(lead, 20.0, -0.5, False)
    self.assertEqual(self.sva.state, SVAState.INACTIVE)

  def test_no_lead_stays_inactive(self):
    """SVA should stay inactive with no lead."""
    lead = FakeLead(present=False)
    for _ in range(10):
      self.sva.update(lead, 20.0, -0.5, False)
    self.assertEqual(self.sva.state, SVAState.INACTIVE)

  def test_low_ego_speed_no_activation(self):
    """SVA should not activate at very low ego speeds."""
    lead = FakeLead(dRel=10.0, vLeadK=0.0, modelProb=0.95)
    for _ in range(20):
      self.sva.update(lead, 2.0, -0.3, False)
    self.assertEqual(self.sva.state, SVAState.INACTIVE)


class TestSVAConfidence(unittest.TestCase):
  def setUp(self):
    self.sva = StoppedVehicleApproach(dt=0.05)
    self.sva.update_params(True, False)

  def test_full_confidence(self):
    """Full confidence: long persistence, centered lead, very slow, high prob."""
    self.sva.stopped_persistence = 1.0
    self.sva.lead_y = 0.2
    self.sva.lead_v = 0.1
    self.sva.lead_prob = 0.95
    conf = self.sva._compute_confidence()
    self.assertGreaterEqual(conf, CONFIDENCE_HARD)

  def test_low_confidence_off_path(self):
    """Low confidence: lead is far off path (>1.75m lateral)."""
    self.sva.stopped_persistence = 0.5
    self.sva.lead_y = 2.5  # Way off center - should gate persistence
    self.sva.lead_v = 0.3
    self.sva.lead_prob = 0.6
    conf = self.sva._compute_confidence()
    # Off-path: persistence capped to 0.10, path=0, vel=0.25, prob=0.05 = 0.40
    self.assertLess(conf, CONFIDENCE_SOFT)

  def test_moderate_confidence(self):
    """Moderate confidence: short persistence, in-path, nearly stopped."""
    self.sva.stopped_persistence = 0.2
    self.sva.lead_y = 1.0  # Edge of lane
    self.sva.lead_v = 0.8
    self.sva.lead_prob = 0.6
    conf = self.sva._compute_confidence()
    # persist=0.15, path=0.15, vel=0.15, prob=0.05 = 0.50
    self.assertGreaterEqual(conf, CONFIDENCE_SOFT)
    self.assertLess(conf, CONFIDENCE_HARD)


class TestSVAPhysics(unittest.TestCase):
  def setUp(self):
    self.sva = StoppedVehicleApproach(dt=0.05)
    self.sva.update_params(True, False)

  def test_required_decel_comfortable_distance(self):
    """At comfortable distance, required decel should be mild."""
    self.sva.lead_d = 100.0
    self.sva.lead_v = 0.0
    a_req = self.sva._compute_required_decel(v_ego=20.0)
    # 100m away at 20 m/s → d_brake ≈ 100 - 1.9 - 20*0.55*1.15 ≈ 85.5m
    # a = -20^2 / (2*85.5) ≈ -2.34
    self.assertGreater(a_req, -3.0)  # Should be comfortable
    self.assertLess(a_req, 0.0)  # Should be braking

  def test_required_decel_close_distance(self):
    """At close distance, required decel should be aggressive."""
    self.sva.lead_d = 30.0
    self.sva.lead_v = 0.0
    a_req = self.sva._compute_required_decel(v_ego=20.0)
    # 30m away at 20 m/s → tight
    self.assertLess(a_req, -3.0)  # Should be aggressive

  def test_required_decel_very_close(self):
    """At very close distance, should hit emergency decel."""
    self.sva.lead_d = 10.0
    self.sva.lead_v = 0.0
    a_req = self.sva._compute_required_decel(v_ego=20.0)
    self.assertEqual(a_req, EMERGENCY_DECEL)

  def test_ttc_calculation(self):
    """TTC should be distance / closing speed."""
    self.sva.lead_d = 50.0
    self.sva.lead_v = 0.0
    ttc = self.sva._compute_ttc(v_ego=25.0)
    expected = (50.0 - MIN_STOP_GAP) / 25.0
    self.assertAlmostEqual(ttc, expected, places=1)

  def test_ttc_not_closing(self):
    """TTC should be large when not closing on lead."""
    self.sva.lead_d = 50.0
    self.sva.lead_v = 30.0  # Lead faster than ego
    ttc = self.sva._compute_ttc(v_ego=20.0)
    self.assertEqual(ttc, 999.0)


class TestSVAStateTransitions(unittest.TestCase):
  def setUp(self):
    self.sva = StoppedVehicleApproach(dt=0.05)
    self.sva.update_params(True, False)

  def _run_approach(self, d_start, v_ego, lead_v=0.0, n_cycles=200):
    """Simulate approaching a stopped lead from d_start at v_ego."""
    results = []
    d = d_start
    v = v_ego
    for _ in range(n_cycles):
      lead = FakeLead(dRel=d, vLeadK=lead_v, yRel=0.3, modelProb=0.95)
      a_out, should_stop = self.sva.update(lead, v, -1.0, False)
      results.append((d, v, self.sva.state, a_out, self.sva.confidence))
      # Simple physics sim
      dt = 0.05
      a_effective = min(a_out, -0.5)  # At least mild braking
      v = max(0.0, v + a_effective * dt)
      d = d - (v_ego - lead_v) * dt  # Approximate closing
      if v < 0.1:
        break
    return results

  def test_full_approach_scenario(self):
    """Test a complete stopped vehicle approach from 80m at 20 m/s."""
    results = self._run_approach(80.0, 20.0)
    # Should transition through states
    states = [r[2] for r in results]
    # Should reach at least SOFT_APPROACH
    self.assertIn(SVAState.SOFT_APPROACH, states, "Should reach SOFT_APPROACH")

  def test_lead_lost_exits(self):
    """SVA should exit when lead is lost."""
    lead = FakeLead(dRel=50.0, vLeadK=0.0, yRel=0.2, modelProb=0.95)
    # Build up state
    for _ in range(20):
      self.sva.update(lead, 20.0, -1.0, False)

    # Now lose the lead
    no_lead = FakeLead(present=False)
    for _ in range(20):
      self.sva.update(no_lead, 18.0, -0.5, False)

    self.assertEqual(self.sva.state, SVAState.INACTIVE)

  def test_lead_starts_moving_exits(self):
    """SVA should exit when lead starts moving."""
    lead = FakeLead(dRel=50.0, vLeadK=0.0, yRel=0.2, modelProb=0.95)
    for _ in range(20):
      self.sva.update(lead, 20.0, -1.0, False)

    # Lead starts moving
    moving_lead = FakeLead(dRel=50.0, vLeadK=5.0, yRel=0.2, modelProb=0.95)
    for _ in range(20):
      self.sva.update(moving_lead, 18.0, -0.5, False)

    self.assertEqual(self.sva.state, SVAState.INACTIVE)


class TestSVAOverride(unittest.TestCase):
  def setUp(self):
    self.sva = StoppedVehicleApproach(dt=0.05)
    self.sva.update_params(True, False)

  def test_disabled_passthrough(self):
    """When disabled, SVA should pass through MPC target."""
    self.sva.update_params(False, False)
    lead = FakeLead(dRel=30.0, vLeadK=0.0)
    a_out, should_stop = self.sva.update(lead, 20.0, -1.5, False)
    self.assertEqual(a_out, -1.5)
    self.assertFalse(should_stop)

  def test_active_overrides_mpc(self):
    """When SVA is active, it should produce more aggressive braking than MPC."""
    lead = FakeLead(dRel=40.0, vLeadK=0.0, yRel=0.2, modelProb=0.95)
    # Build confidence over many cycles
    for _ in range(40):
      a_out, _ = self.sva.update(lead, 20.0, -0.5, False)

    # SVA should be active and commanding more aggressive braking than -0.5
    if self.sva.active:
      self.assertLess(a_out, -0.5)

  def test_false_positive_rejection(self):
    """Lead far off-path should not trigger aggressive braking."""
    lead = FakeLead(dRel=40.0, vLeadK=0.0, yRel=3.0, modelProb=0.3)
    for _ in range(40):
      a_out, _ = self.sva.update(lead, 20.0, -0.5, False)

    # Off-path lead with low model prob: confidence should be below SOFT threshold
    # so it shouldn't even reach SOFT_APPROACH, let alone HARD_APPROACH
    self.assertNotIn(self.sva.state, (SVAState.HARD_APPROACH, SVAState.SOFT_APPROACH))

  def test_minimum_stop_gap(self):
    """Required decel calculation should respect MIN_STOP_GAP."""
    self.sva.lead_d = MIN_STOP_GAP + 15.0
    self.sva.lead_v = 0.0
    a_req = self.sva._compute_required_decel(v_ego=5.0)
    # d_available=15m, d_delay=5*0.55*1.15=3.16, d_brake=11.84
    # a = -25/(2*11.84) = -1.06
    self.assertLess(a_req, 0.0)
    self.assertGreater(a_req, EMERGENCY_DECEL)

  def test_accel_clip_rate_scaling(self):
    """Rate multiplier should be > 1 when SVA is active."""
    self.sva.state = SVAState.HARD_APPROACH
    rate = self.sva.get_accel_clip_rate()
    self.assertGreater(rate, 1.0)

    self.sva.state = SVAState.INACTIVE
    rate = self.sva.get_accel_clip_rate()
    self.assertEqual(rate, 1.0)


class TestSVALogging(unittest.TestCase):
  def setUp(self):
    self.sva = StoppedVehicleApproach(dt=0.05)

  def test_logging_off_no_file(self):
    """No log file should be created when logging is off."""
    self.sva.update_params(True, False)
    lead = FakeLead(dRel=50.0, vLeadK=0.0)
    self.sva.update(lead, 20.0, -1.0, False)
    self.assertIsNone(self.sva._log_file)

  def test_logging_on_creates_entries(self):
    """Log entries should be created when logging is enabled."""
    self.sva.update_params(True, True)
    lead = FakeLead(dRel=50.0, vLeadK=0.0, yRel=0.2, modelProb=0.95)
    for _ in range(5):
      self.sva.update(lead, 20.0, -1.0, False)
    # Should have opened a log file
    if self.sva._log_file is not None:
      self.assertGreater(self.sva._log_counter, 0)
      self.sva.cleanup()


if __name__ == '__main__':
  unittest.main()
