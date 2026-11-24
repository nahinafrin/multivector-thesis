#!/usr/bin/env python3
"""
make_full_selection.py -- row sets for the full ADO Protocol B run (written once, before inference).

Deterministic; nothing is sampled:
  remaining_selection.json  all Protocol B rows NOT in pilot_selection.json (150 attacks + 150 benign)
  full_selection.json       all 200 attacks + 200 benign (pilot rows + remaining rows)
The pilot rows are reused as they are (same code and configuration). The functional-test rows are
in the remaining set and are rerun; the functional outputs are not used in any result.

    python make_full_selection.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTX = HERE.parent.parent / "step4 dataset" / "mitigation_results" / "baselines_protocolB"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    for f in ("remaining_selection.json", "full_selection.json"):
        if (HERE / f).exists():
            raise SystemExit(f"{f} already exists; it must not be regenerated")
    pilot = json.loads((HERE / "pilot_selection.json").read_text(encoding="utf-8"))
    allids = {s: sorted(json.loads(l)["index"] for l in open(CTX / f"contexts_{s}.jsonl", encoding="utf-8"))
              for s in ("attack", "benign")}
    meta = {"inputs": {"pilot_selection.json": sha(HERE / "pilot_selection.json"),
                       "contexts_attack.jsonl": sha(CTX / "contexts_attack.jsonl"),
                       "contexts_benign.jsonl": sha(CTX / "contexts_benign.jsonl")}}
    rem = {s: sorted(set(allids[s]) - set(pilot[s])) for s in allids}
    (HERE / "remaining_selection.json").write_text(json.dumps(
        {**meta, "method": "all Protocol B rows not in pilot_selection.json", **rem}, indent=1), encoding="utf-8")
    (HERE / "full_selection.json").write_text(json.dumps(
        {**meta, "method": "all Protocol B rows (pilot + remaining)", **allids}, indent=1), encoding="utf-8")
    print({s: (len(allids[s]), len(rem[s])) for s in allids})


if __name__ == "__main__":
    main()
