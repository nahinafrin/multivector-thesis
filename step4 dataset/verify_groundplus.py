#!/usr/bin/env python3
"""Pressure-test groundplus attack/benign block dispositions."""
import json, sys
from score_attack_success import attack_succeeded, _row_verdict

def main():
    attack_path = sys.argv[1] if len(sys.argv) > 1 else "mitigation_results/planted30v2/on_groundplus.jsonl"
    rows = [json.loads(l) for l in open(attack_path, encoding="utf-8") if l.strip()]
    refuse = ground = other_blk = delivered = 0
    for r in rows:
        if not r.get("blocked"):
            delivered += 1
            continue
        br = r.get("block_reason") or ""
        if "detector fired" in br:
            refuse += 1
        elif "grounded" in br:
            ground += 1
        else:
            other_blk += 1
    del_rows = [r for r in rows if not r.get("blocked")]
    gen_succ = sum(attack_succeeded(r) for r in del_rows)
    print(f"attacks: refuse={refuse} grounding={ground} other_blocked={other_blk} delivered={delivered}")
    print(f"delivered generation-ASR successes: {gen_succ}/{delivered}")
    print(f"total generation-ASR successes: {sum(attack_succeeded(r) for r in rows)}/{len(rows)}")
    print(f"total operational-ASR successes: {gen_succ}/{len(rows)}")
    for r in del_rows:
        v = _row_verdict(r)
        g = r.get("grounding") or {}
        ans = (r.get("final_response") or "")[:80]
        print(f"  idx {r['index']}: marker={v['marker_emitted']} ground_pass={g.get('passed')} | {ans}")

if __name__ == "__main__":
    main()
