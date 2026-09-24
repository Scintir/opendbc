from dataclasses import dataclass as _dataclass, field, is_dataclass
from enum import Enum, StrEnum as _StrEnum, auto
from typing import dataclass_transform, get_origin

import os
import capnp
from opendbc.car.common.basedir import BASEDIR

capnp.remove_import_hook()
car = capnp.load(os.path.join(BASEDIR, "car.capnp"), imports=[BASEDIR])

CarState = car.CarState
RadarData = car.RadarData
CarControl = car.CarControl
CarParams = car.CarParams

CarStateT = capnp.lib.capnp._StructModule
RadarDataT = capnp.lib.capnp._StructModule
CarControlT = capnp.lib.capnp._StructModule
CarParamsT = capnp.lib.capnp._StructModule

# sunnypilot structs

AUTO_OBJ = object()


def auto_field():
  return AUTO_OBJ


@dataclass_transform()
def auto_dataclass(cls=None, /, **kwargs):
  cls_annotations = cls.__dict__.get('__annotations__', {})
  for name, typ in cls_annotations.items():
    current_value = getattr(cls, name)
    if current_value is AUTO_OBJ:
      origin_typ = get_origin(typ) or typ
      if isinstance(origin_typ, str):
        raise TypeError(f"Forward references are not supported for auto_field: '{origin_typ}'. Use a default_factory with lambda instead.")
      elif origin_typ in (int, float, str, bytes, list, tuple, bool) or is_dataclass(origin_typ):
        setattr(cls, name, field(default_factory=origin_typ))
      elif issubclass(origin_typ, Enum):  # first enum is the default
        setattr(cls, name, field(default=next(iter(origin_typ))))
      else:
        raise TypeError(f"Unsupported type for auto_field: {origin_typ}")

  # TODO: use slots, this prevents accidentally setting attributes that don't exist
  return _dataclass(cls, **kwargs)


class StrEnum(_StrEnum):
  @staticmethod
  def _generate_next_value_(name, *args):
    # auto() defaults to name.lower()
    return name


@auto_dataclass
class CarParamsSP:
  flags: int = auto_field()        # flags for car specific quirks
  safetyParam: int = auto_field()  # flags for custom safety flags
  pcmCruiseSpeed: bool = auto_field()
  intelligentCruiseButtonManagementAvailable: bool = auto_field()
  enableGasInterceptor: bool = auto_field()

  neuralNetworkLateralControl: 'CarParamsSP.NeuralNetworkLateralControl' = field(default_factory=lambda: CarParamsSP.NeuralNetworkLateralControl())

  @auto_dataclass
  class NeuralNetworkLateralControl:
    model: 'CarParamsSP.NeuralNetworkLateralControl.Model' = field(default_factory=lambda: CarParamsSP.NeuralNetworkLateralControl.Model())
    fuzzyFingerprint: bool = auto_field()

    @auto_dataclass
    class Model:
      path: str = auto_field()
      name: str = auto_field()


@auto_dataclass
class ModularAssistiveDrivingSystem:
  state: 'ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState' = field(
    default_factory=lambda: ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState.disabled
  )
  enabled: bool = auto_field()
  active: bool = auto_field()
  available: bool = auto_field()

  class ModularAssistiveDrivingSystemState(StrEnum):
    disabled = auto()
    paused = auto()
    enabled = auto()
    softDisabling = auto()
    overriding = auto()


@auto_dataclass
class IntelligentCruiseButtonManagement:
  state: 'IntelligentCruiseButtonManagement.IntelligentCruiseButtonManagementState' = field(
    default_factory=lambda: IntelligentCruiseButtonManagement.IntelligentCruiseButtonManagementState.inactive
  )
  sendButton: 'IntelligentCruiseButtonManagement.SendButtonState' = field(
    default_factory=lambda: IntelligentCruiseButtonManagement.SendButtonState.none
  )
  vTarget: float = auto_field()

  class IntelligentCruiseButtonManagementState(StrEnum):
    inactive = auto()
    preActive = auto()
    increasing = auto()
    decreasing = auto()
    holding = auto()

  class SendButtonState(StrEnum):
    none = auto()
    increase = auto()
    decrease = auto()


@auto_dataclass
class LeadData:
  dRel: float = auto_field()
  yRel: float = auto_field()
  vRel: float = auto_field()
  aRel: float = auto_field()
  vLead: float = auto_field()
  dPath: float = auto_field()
  vLat: float = auto_field()
  vLeadK: float = auto_field()
  aLeadK: float = auto_field()
  fcw: bool = auto_field()
  status: bool = auto_field()
  aLeadTau: float = auto_field()
  modelProb: float = auto_field()
  radar: bool = auto_field()
  radarTrackId: int = auto_field()

  aLeadDEPRECATED: float = auto_field()


@auto_dataclass
class CarControlSP:
  mads: 'ModularAssistiveDrivingSystem' = field(default_factory=lambda: ModularAssistiveDrivingSystem())
  params: list['CarControlSP.Param'] = auto_field()
  leadOne: 'LeadData' = field(default_factory=lambda: LeadData())
  leadTwo: 'LeadData' = field(default_factory=lambda: LeadData())
  intelligentCruiseButtonManagement: 'IntelligentCruiseButtonManagement' = field(default_factory=lambda: IntelligentCruiseButtonManagement())

  @auto_dataclass
  class Param:
    key: str = auto_field()
    value: bytes = auto_field()
    type: 'CarControlSP.ParamType' = field(
      default_factory=lambda: CarControlSP.ParamType.string
    )

  class ParamType(StrEnum):
    string = auto()
    bool = auto()
    int = auto()
    float = auto()
    time = auto()
    json = auto()
    bytes = auto()


@auto_dataclass
class CarStateSP:
  speedLimit: float = auto_field()

  # EV power limiter signals — kept in lockstep with cereal/custom.capnp
  # CarStateSP ordinals. carstate_ext writes them; UI / analyzer read them
  # via the cereal-published message. If this dataclass doesn't declare them,
  # writes from carstate silently fail to serialize.
  evLimiterActive: bool = auto_field()
  evLimiterSetSpeedOffset: float = auto_field()
  accelDemand: float = auto_field()
  dteRaw: int = auto_field()
  estPowerW: float = auto_field()
  evLimiterUserTargetSpeed: float = auto_field()
  evLimiterState: int = auto_field()
  evLimiterGradeAccel: float = auto_field()  # iter9 — was missing → root cause of drive 6-13 "publish 0" bug

  # iter11 telemetry — must lockstep with cereal/custom.capnp CarStateSP @9-@22
  estPowerRawW: float = auto_field()
  estPowerCapped: bool = auto_field()
  estPowerSaturated: bool = auto_field()
  evModeAssumed: bool = auto_field()
  evLimiterGradeAccelSource: int = auto_field()
  evLimiterKalmanRejectReason: int = auto_field()
  evLimiterIneffectiveResEvents: int = auto_field()
  evLimiterMaxDeficitViolationFrames: int = auto_field()
  evLimiterTransitionsBlockedByDwell: int = auto_field()
  evLimiterTransitionsBlockedBySustain: int = auto_field()
  evLimiterPowerHighPendingFrames: int = auto_field()
  abasisFiltered: float = auto_field()
  aEgoFiltered: float = auto_field()
  evLimiterRecentTransitions: str = auto_field()

  # iter13 v4 telemetry — must lockstep with cereal/custom.capnp CarStateSP @23-@40.
  # Capnp enums (EvLimiterBlockReason) serialize as integer ordinals on the dataclass
  # side; CarController sets these via cereal-published message.
  evLimiterLastBlockReason: int = auto_field()
  evModeParamReadOk: bool = auto_field()
  evLimiterSetRequested: int = auto_field()
  evLimiterSetEmitted: int = auto_field()
  evLimiterSetDropped: int = auto_field()
  evLimiterSetClusterDecrementAcked: int = auto_field()
  evLimiterSetNoAckEvents: int = auto_field()
  evLimiterStandstillEntered: int = auto_field()
  evLimiterStandstillExitedByAchieved: int = auto_field()
  evLimiterStandstillExitedByNoAckBackoff: int = auto_field()
  evLimiterStandstillSetRequested: int = auto_field()
  evLimiterStandstillSetEmitted: int = auto_field()
  evLimiterStandstillSetDropped: int = auto_field()
  evLimiterSuspectedSccCancelEvents: int = auto_field()
  evLimiterFaultInhibitActive: bool = auto_field()
  evLimiterFaultInhibitReason: int = auto_field()
  evLimiterAllBtnEmitted: int = auto_field()
  evLimiterCarControllerLimiterTickRate: int = auto_field()

  # iter14 v2 telemetry — must lockstep with cereal/custom.capnp CarStateSP @41-@56.
  # Triple-output power estimator (R1-MF3): control path uses estPowerControlW, NOT
  # the HUD-smoothed estPowerW. estPowerInstantW is forensic-only.
  estPowerInstantW: float = auto_field()
  estPowerControlW: float = auto_field()
  evLimiterPowerCappedSustainFrames: int = auto_field()
  evLimiterPowerNearBudgetSustainFrames: int = auto_field()
  evLimiterEstPowerRawIsFiltered: bool = auto_field()

  # Transition-decision instrumentation (R1-MF1): every-frame candidate vs final state
  # so we can diagnose why iter13's existing line-1450 transition didn't fire on episode #1.
  evLimiterStatePriorTransition: int = auto_field()
  evLimiterStateCandidateBeforeGuard: int = auto_field()
  evLimiterStateAfterPowerGuard: int = auto_field()
  evLimiterPowerGuardYieldReason: str = auto_field()
  evLimiterPowerGuardLockoutActive: bool = auto_field()
  evLimiterRecoveryYieldEvents: int = auto_field()
  evLimiterRecoveryLockoutsEntered: int = auto_field()

  # iter15 v2 telemetry — must lockstep with cereal/custom.capnp CarStateSP @53-@69.
  # Hard-preempt fix (Section A)
  evLimiterGuardForcedTransition: bool = auto_field()
  evLimiterGuardForcedTransitionEvents: int = auto_field()

  # Grade clamp (Section B); raw may exceed cap (R2-MF-2)
  evLimiterGradePowerRawW: float = auto_field()
  evLimiterGradePowerCappedFrames: int = auto_field()

  # Long-standstill narrow reset (Section C)
  evLimiterLongStandstillResets: int = auto_field()

  # Post-RES quiet (Section D)
  evLimiterPostResQuietActive: bool = auto_field()
  evLimiterSoftcapDecrementSuppressedFrames: int = auto_field()
  evLimiterSoftcapDecrementSuppressedEvents: int = auto_field()

  # @61-@62 RESERVED for iter16 HEV CAN passive — DO NOT IMPLEMENT in iter15
  evLimiterReservedIter16A: int = auto_field()
  evLimiterReservedIter16B: int = auto_field()

  # Edge-detected recovery yield episodes (Section A)
  evLimiterRecoveryYieldEpisodes: int = auto_field()

  # Long-standstill state vector telemetry (Section C R1-MF-C)
  evLimiterStandstillExitStateSnapshot: str = auto_field()
  evLimiterStandstillExitTimeS: float = auto_field()
  evLimiterStandstillExitToFirstResLatencyFrames: int = auto_field()
  evLimiterLongStandstillPrelaunchBackoffCleared: int = auto_field()
  evLimiterLongStandstillSoftcapReasonCleared: int = auto_field()

  # Post-RES informational override (Section D)
  evLimiterPostResHardOverrideEvents: int = auto_field()

  # iter16a — live request indicator (intent vs emitted vs honored)
  evLimiterRequestDir: int = auto_field()        # 0 NONE, 1 UP(want_res), 2 DOWN(want_set)
  evLimiterButtonDir: int = auto_field()         # 0 NONE, 1 UP(RES), 2 DOWN(SET) — actual emitted
  evLimiterRequestHonored: int = auto_field()    # 0 unknown, 1 honored, 2 ignored

  # iter16a — real HEV power ground truth (passive)
  evLimiterRealMotorPowerW: float = auto_field()   # W; NaN if unavailable
  evLimiterRealPowerSource: int = auto_field()     # 0 invalid, 1 motor CAN, 2 battery VxI

  # iter16a — C1 below-vEgo power droop LOG-ONLY (default-off)
  evLimiterPowerDroopWouldEnter: bool = auto_field()
  evLimiterPowerDroopRequestMph: float = auto_field()
  evLimiterPowerDroopActive: bool = auto_field()
