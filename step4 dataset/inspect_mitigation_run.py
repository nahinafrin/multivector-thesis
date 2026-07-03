#!/usr/bin/env python3
"""Quick eyeball of mitigation A/B JSONL arms (smoke-test sanity check)."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Eyeball mitigation OFF vs ON outputs")
    ap.add_argument("--run-dir", default=None, help="experiment dir with off.jsonl + on_*.jsonl")
    ap.add_argument("--off", default=None)
    ap.add_argument("--on", default=None)
    args = ap.parse_args()

    if args.run_dir:
        base = Path(args.run_dir)
        off_path = base / "off.jsonl"
        on_paths = sorted(base.glob("on_*.jsonl"))
        if not on_paths:
            on_paths = [base / "on_full.jsonl"]
    else:
        off_path = Path(args.off or "mitigation_off.jsonl")
        on_paths = [Path(args.on or "mitigation_on.jsonl")]

    for arm_path in [off_path] + on_paths:
        if not arm_path.is_file():
            print(f"\n=== MISSING: {arm_path} ===")
            continue
        rows = _load(arm_path)
        arm = arm_path.stem
        print(f"\n=== {arm} ({len(rows)} rows) — {arm_path} ===")
        for r in rows:
            ans = (r.get("generation") or {}).get("answer") or r.get("final_response") or ""
            ans = str(ans)[:70]
            det = r.get("detector") or {}
            print(f"  idx={r.get('index')} blocked={r.get('blocked')} "
                  f"det_fired={det.get('is_attack')} mit={r.get('mitigation_applied')} "
                  f"ans={ans!r}")


if __name__ == "__main__":
    main()
