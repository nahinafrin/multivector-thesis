#!/usr/bin/env bash
# run_full.sh -- the remaining 150 + 150 Protocol B rows for ADO (same code/config as the pilot),
# then merge with the pilot rows into full_{attack,benign}_results.jsonl and check alignment.
# Resumable: rerunning skips rows already written to remaining_*_results.jsonl.
set -u
cd "$(dirname "$0")"
P="../../step4 dataset/.venv311gpu/Scripts/python.exe"
Q="../../step4 dataset/.venv311/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
"$Q" check_alignment.py --selection remaining_selection.json || exit 1
for s in attack benign; do
  "$P" run_ado_protocolB.py --selection remaining_selection.json --split $s --out remaining_${s}_results.jsonl \
       --av hf4bit >> remaining_${s}_stdout.txt 2>&1
  echo "$s rc=$? rows=$(wc -l < remaining_${s}_results.jsonl)"
done
"$Q" check_alignment.py --selection remaining_selection.json --outputs remaining || exit 1
"$Q" - <<'EOF'
import json
for s in ("attack", "benign"):
    rows = {}
    for f in (f"pilot_{s}_results.jsonl", f"remaining_{s}_results.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                assert r["index"] not in rows, (s, r["index"])
                rows[r["index"]] = l.rstrip("\n")
    open(f"full_{s}_results.jsonl", "w", encoding="utf-8", newline="\n").write(
        "\n".join(rows[i] for i in sorted(rows)) + "\n")
    print(s, "merged", len(rows))
EOF
"$Q" check_alignment.py --selection full_selection.json --outputs full
