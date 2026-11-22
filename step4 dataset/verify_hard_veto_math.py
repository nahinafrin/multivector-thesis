"""Synthetic confirmation of the multivector hard-veto arithmetic."""
from multivector import multivector_risk, DEFAULT_JOINT_BLOCK, DEFAULT_SOFT_PER_CHANNEL

cases = [
    ("query barely above floor, context strongly detected+redacted",
     {"query_vector": 0.24, "context_vector": 0.95}),
    ("both channels only just above their floors (should NOT hard-block)",
     {"query_vector": 0.24, "context_vector": 0.31}),
    ("both channels at the sub-0.50 ceiling (documents the 0.75 cap)",
     {"query_vector": 0.499, "context_vector": 0.499}),
]

print(f"floors: {DEFAULT_SOFT_PER_CHANNEL}   joint_block: {DEFAULT_JOINT_BLOCK}\n")
for label, channels in cases:
    result = multivector_risk(channels)
    print(label)
    print(f"  channels={channels}")
    print(f"  joint_risk={result['joint_risk']}  is_multivector={result['is_multivector']}  "
          f"hard_block={result['hard_block']}")
    print()
