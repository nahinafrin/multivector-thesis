"""
_run_slice_resumable.py — drive run_full_pipeline.py --slice to completion across
native crashes.

The full adversarial slice runs the 3-model ensemble + faiss + spacy on CPU; on
this Windows box that stack intermittently segfaults (exit 0xC0000005). This
wrapper:

  * runs `run_full_pipeline.py --slice ... --resume` in a subprocess,
  * restarts it after a crash (it skips already-completed ids),
  * sets the usual OpenMP-conflict mitigations,
  * if a restart makes NO progress (a row crashes deterministically), writes a
    minimal stub result for that one id so the run can move past it, and
  * stops when every id in the slice has a row (real or stub).

Usage:
    python _run_slice_resumable.py --slice adversarial_slice.jsonl \
        --out slice_results.jsonl --max-restarts 80
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _all_ids(slice_path: str) -> list[int]:
    ids = []
    for line in open(slice_path, encoding="utf-8"):
        line = line.strip()
        if line:
            ids.append(json.loads(line).get("id", 0))
    return ids


def _done_ids(out_path: str) -> set:
    done = set()
    if not Path(out_path).exists():
        return done
    for line in open(out_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line).get("index"))
        except json.JSONDecodeError:
            continue
    return done


def _slice_by_id(slice_path: str) -> dict:
    rows = {}
    for line in open(slice_path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r.get("id", 0)] = r
    return rows


def _write_stub(out_path: str, row: dict) -> None:
    """Append a minimal crashed-row stub so --resume treats this id as done."""
    stub = {
        "index": row.get("id", 0),
        "question": row.get("question", ""),
        "ground_truth": row.get("ground_truth"),
        "kind": row.get("kind"),
        "expectation": row.get("expectation"),
        "gate": {}, "retrieval": {}, "generation": {}, "grounding": {},
        "multivector": {}, "controller": {},
        "blocked": False, "block_stage": None, "block_reason": None,
        "crashed": True,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(stub, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-restarts", type=int, default=80)
    args = ap.parse_args()

    all_ids = _all_ids(args.slice)
    total = len(all_ids)
    slice_rows = _slice_by_id(args.slice)

    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"     # tolerate duplicate OpenMP runtimes
    env["OMP_NUM_THREADS"] = "1"             # reduce native threading contention

    py = sys.executable
    for attempt in range(1, args.max_restarts + 1):
        done = _done_ids(args.out)
        remaining = [i for i in all_ids if i not in done]
        print(f"[driver] attempt {attempt}: {len(done)}/{total} done, "
              f"{len(remaining)} remaining", flush=True)
        if not remaining:
            break

        before = len(done)
        cmd = [py, "run_full_pipeline.py", "--slice", args.slice,
               "--out", args.out, "--resume"]
        proc = subprocess.run(cmd, env=env)
        after = len(_done_ids(args.out))
        print(f"[driver] subprocess exit={proc.returncode}; "
              f"progress {before} -> {after}", flush=True)

        if after == before:
            # No progress -> the next remaining id crashes deterministically.
            stuck = remaining[0]
            print(f"[driver] no progress; stubbing crashed id={stuck} "
                  f"(kind={slice_rows.get(stuck, {}).get('kind')})", flush=True)
            _write_stub(args.out, slice_rows.get(stuck, {"id": stuck}))

    done = _done_ids(args.out)
    real = total - sum(1 for line in open(args.out, encoding="utf-8")
                       if line.strip() and json.loads(line).get("crashed"))
    print(f"[driver] FINISHED: {len(done)}/{total} ids have a row "
          f"({real} real, {len(done) - real} stubbed)", flush=True)


if __name__ == "__main__":
    main()
