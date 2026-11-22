#!/usr/bin/env python3
import json
from score_attack_success import attack_succeeded

def read(p):
    with open(p, encoding="utf-8") as f:
        return {json.loads(l)["index"]: json.loads(l) for l in f if l.strip()}

off = read("mitigation_results/planted200/off.jsonl")
ro = read("mitigation_results/planted200/on_refuseonly.jsonl")
ids = sorted(off)

off_succ = [i for i in ids if attack_succeeded(off[i])]
det_fired = sum(1 for i in off_succ if (ro[i].get("detector") or {}).get("is_attack"))
refuse_stopped = sum(
    1 for i in off_succ
    if ro[i].get("blocked") and "detector fired" in (ro[i].get("block_reason") or "")
)
still_succ = sum(attack_succeeded(ro[i]) for i in off_succ)
blocked_refuse = sum(
    1 for i in ids if ro[i].get("blocked") and "detector fired" in (ro[i].get("block_reason") or "")
)

print("OFF successes (n=%d):" % len(off_succ))
print("  detector fired on refuse-only run: %d/%d" % (det_fired, len(off_succ)))
print("  stopped by refuse: %d/%d" % (refuse_stopped, len(off_succ)))
print("  slipped through (still succeeded): %d/%d" % (still_succ, len(off_succ)))
print()
print("Refuse-only delivered successes: %d" % sum(attack_succeeded(r) for r in ro.values()))
print("  detector had fired on those: %d" % sum(
    1 for r in ro.values() if attack_succeeded(r) and (r.get("detector") or {}).get("is_attack")
))
print("  detector missed: %d" % sum(
    1 for r in ro.values() if attack_succeeded(r) and not (r.get("detector") or {}).get("is_attack")
))
print()
print("Total refuse blocks: %d/200" % blocked_refuse)
