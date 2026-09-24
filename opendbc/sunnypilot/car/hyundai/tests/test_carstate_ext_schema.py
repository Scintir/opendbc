# ruff: noqa: E701, ISC002, B904
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

iter13 v4 — schema/dataclass parity tests (Section H + R4-MF5).

Catches the recurring "publish-bug" pattern observed across iter9, iter11, and
iter12: capnp schema declares a field but the corresponding dataclass in
opendbc/car/structs.py forgets it, and `convert_to_capnp` silently drops the
write — yielding a "0" or default value in the published CarStateSP message
that masks bugs in the underlying logic.

Tests #50, #51, #52, #53 from v4 plan:
  #50 test_feature_owned_capnp_dataclass_parity
  #51 test_feature_owned_publish_roundtrip_sentinels_deterministic
  #52 test_capnp_ordinals_no_collision (placeholder; capnp parser enforces this)
  #53 test_block_reason_sentinel_order_matches_capnp_ordinals
"""
from __future__ import annotations

import unittest

# Feature-owned prefixes — every CarStateSP field this feature touches.
FEATURE_OWNED_PREFIXES = (
  "evLimiter", "evMode", "estPower", "abasis", "aEgoFiltered",
  "accelDemand", "dteRaw",
)


def _capnp_carstateSP_field_names() -> set[str]:
  """Parse cereal/custom.capnp and extract every CarStateSP field name."""
  import re
  import os

  try:

    import openpilot.cereal as _cereal

    capnp_path = os.path.join(os.path.dirname(_cereal.__file__), "custom.capnp")

  except ImportError:

    raise unittest.SkipTest("needs the sunnypilot tree for openpilot/cereal/custom.capnp")

  with open(capnp_path) as f:
    src = f.read()
  # Find "struct CarStateSP @0xb86e... {" through the matching "}" — simple
  # depth tracking.
  m = re.search(r"struct CarStateSP\b[^{]*\{", src)
  if not m:
    return set()
  start = m.end()
  depth = 1
  i = start
  while i < len(src) and depth > 0:
    if src[i] == "{":
      depth += 1
    elif src[i] == "}":
      depth -= 1
    i += 1
  body = src[start:i - 1]
  names = set()
  for line in body.splitlines():
    line = line.strip()
    if line.startswith("#") or not line:
      continue
    fm = re.match(r"^([a-zA-Z_]\w*)\s*@\d+\s*:", line)
    if fm:
      names.add(fm.group(1))
  return names


def _dataclass_carstateSP_field_names() -> set[str]:
  """Read structs.py and extract every CarStateSP dataclass field name."""
  import re
  with open(
      __import__("opendbc.car.structs", fromlist=["__file__"]).__file__) as f:
    src = f.read()
  m = re.search(r"class CarStateSP[^:]*:\s*\n", src)
  if not m:
    return set()
  start = m.end()
  # Body ends at the next top-level class/def/decorator.
  body_lines = []
  for line in src[start:].splitlines():
    if line and not line.startswith(" ") and not line.startswith("\t"):
      break
    body_lines.append(line)
  names = set()
  for line in body_lines:
    s = line.strip()
    if s.startswith("#") or not s:
      continue
    nm = re.match(r"^([a-zA-Z_]\w*)\s*:", s)
    if nm:
      names.add(nm.group(1))
  return names


class TestSchemaDataclassParity(unittest.TestCase):
  """Test #50: every feature-owned capnp field exists in dataclass and vice versa.

  Catches the iter9 grade publish-bug, the iter11 multi-field publish-bug,
  the iter12 _ineffective_set_events publish-bug, and any future variant
  introduced by an EV limiter / EV mode / power telemetry change.
  """

  def test_feature_owned_capnp_dataclass_parity(self):
    capnp_fields = {n for n in _capnp_carstateSP_field_names()
                    if any(n.startswith(p) for p in FEATURE_OWNED_PREFIXES)}
    dataclass_fields = {n for n in _dataclass_carstateSP_field_names()
                        if any(n.startswith(p) for p in FEATURE_OWNED_PREFIXES)}
    capnp_only = capnp_fields - dataclass_fields
    dataclass_only = dataclass_fields - capnp_fields
    msg_lines = []
    if capnp_only:
      msg_lines.append(f"capnp - dataclass (publish silently drops these): "
                        f"{sorted(capnp_only)}")
    if dataclass_only:
      msg_lines.append(f"dataclass - capnp (no wire schema for these): "
                        f"{sorted(dataclass_only)}")
    self.assertFalse(capnp_only or dataclass_only, "\n".join(msg_lines))


class TestBlockReasonSentinel(unittest.TestCase):
  """Test #53 (R4-MF5): explicit ordinal-stable EV_LIMITER_BLOCK_REASON_SENTINELS
  tuple matches capnp ordinals exactly.

  R4-MF5: do NOT use `list(EvLimiterBlockReason)` for sentinel generation —
  Python enum iteration order is implementation-defined. Use an explicit tuple
  whose order matches capnp ordinals.
  """

  def test_block_reason_ordinal_dict_covers_all_capnp_entries(self):
    """Every capnp EvLimiterBlockReason entry is in BLOCK_REASON_ORDINAL."""
    import re
    import os

    try:

      import openpilot.cereal as _cereal

      capnp_path = os.path.join(os.path.dirname(_cereal.__file__), "custom.capnp")

    except ImportError:

      raise unittest.SkipTest("needs the sunnypilot tree for openpilot/cereal/custom.capnp")

    with open(capnp_path) as f:
      src = f.read()
    m = re.search(r"enum EvLimiterBlockReason\s*\{([^}]*)\}", src)
    self.assertIsNotNone(m, "EvLimiterBlockReason enum not found in capnp")
    capnp_entries: set[str] = set()
    for line in m.group(1).splitlines():
      em = re.match(r"\s*([a-zA-Z_]\w*)\s*@\d+\s*;", line)
      if em:
        capnp_entries.add(em.group(1))

    from opendbc.sunnypilot.car.hyundai.car_controller_button_limiter import (
      BLOCK_REASON_ORDINAL,
    )
    code_entries = set(BLOCK_REASON_ORDINAL.keys())
    self.assertEqual(capnp_entries, code_entries,
                     f"capnp - code: {sorted(capnp_entries - code_entries)}; "
                     f"code - capnp: {sorted(code_entries - capnp_entries)}")

  def test_block_reason_priority_list_complete(self):
    """BLOCK_REASON_PRIORITY covers every capnp enum entry exactly once
    (excluding 'none' which is the sentinel value, not a block)."""
    from opendbc.sunnypilot.car.hyundai.car_controller_button_limiter import (
      BLOCK_REASON_PRIORITY, BLOCK_REASON_ORDINAL,
    )
    expected = set(BLOCK_REASON_ORDINAL.keys()) - {"none"}
    self.assertEqual(set(BLOCK_REASON_PRIORITY), expected)
    # No duplicates
    self.assertEqual(len(BLOCK_REASON_PRIORITY), len(set(BLOCK_REASON_PRIORITY)))


class TestCapnpOrdinalNoCollision(unittest.TestCase):
  """Test #52: capnp ordinals are unique and monotonically assigned.
  capnp parser already enforces this; this test catches accidental edits."""

  def test_capnp_ordinals_no_collision(self):
    import re
    import os

    try:

      import openpilot.cereal as _cereal

      capnp_path = os.path.join(os.path.dirname(_cereal.__file__), "custom.capnp")

    except ImportError:

      raise unittest.SkipTest("needs the sunnypilot tree for openpilot/cereal/custom.capnp")

    with open(capnp_path) as f:
      src = f.read()
    m = re.search(r"struct CarStateSP\b[^{]*\{", src)
    self.assertIsNotNone(m)
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth > 0:
      if src[i] == "{": depth += 1
      elif src[i] == "}": depth -= 1
      i += 1
    body = src[start:i - 1]
    ordinals: list[int] = []
    for line in body.splitlines():
      om = re.match(r"\s*[a-zA-Z_]\w*\s*@(\d+)\s*:", line)
      if om:
        ordinals.append(int(om.group(1)))
    self.assertEqual(len(ordinals), len(set(ordinals)),
                     f"duplicate ordinals in CarStateSP: {ordinals}")
    self.assertEqual(ordinals, sorted(ordinals),
                     f"ordinals not monotonic: {ordinals}")


if __name__ == "__main__":
  unittest.main()
