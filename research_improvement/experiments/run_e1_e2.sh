#!/usr/bin/env bash
# run_e1_e2.sh -- preregistered E2 then E1 (see PREREGISTRATION_E1_E2.md). Defense code unchanged; outputs only
# under research_improvement/experiments/. Records SHA-256 of inputs before any inference.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
S4="$EXP/../../step4 dataset"
PY="$S4/.venv311/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
cd "$S4"
"$PY" - "$EXP" <<'EOF'
import hashlib, json, sys, datetime, subprocess
from pathlib import Path
exp = Path(sys.argv[1]); s4 = exp.parents[1] / "step4 dataset"
fs = [exp / "PREREGISTRATION_E1_E2.md", exp / "e1_attacks_NS.jsonl", exp / "e1_attacks_HO.jsonl",
      exp / "e1_build_variants.py", exp / "e2_closed_book.py", exp / "run_e1_e2.sh",
      s4 / "run_mitigation_ab.py", s4 / "mitigation_pipeline.py", s4 / "step_09_generator_llm.py",
      s4 / "planted_attacks_200.jsonl"]
m = {"written_before_inference": datetime.datetime.now().isoformat(timespec="seconds"),
     "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=exp).stdout.strip(),
     "sha256": {str(f.relative_to(exp.parents[1])): hashlib.sha256(f.read_bytes()).hexdigest() for f in fs}}
p = exp / "launch_manifest.json"
if not p.exists():
    p.write_text(json.dumps(m, indent=1), encoding="utf-8")
print("manifest ok")
EOF
date +"E2 start %H:%M"
"$PY" "$EXP/e2_closed_book.py" >> "$EXP/e2_stdout.txt" 2>&1; echo "E2 rc=$?"
for V in NS HO; do
  RD="$EXP/E1_$V"; SL="$EXP/e1_attacks_$V.jsonl"
  date +"E1 $V start %H:%M"
  [ -f "$RD/off.jsonl" ] && [ "$(wc -l < "$RD/off.jsonl")" -ge 200 ] || \
    "$PY" run_mitigation_ab.py --slice "$SL" --detector existing --run-dir "$RD" --off-only >> "$EXP/e1_${V}_stdout.txt" 2>&1
  [ -f "$RD/on_full.jsonl" ] && [ "$(wc -l < "$RD/on_full.jsonl")" -ge 200 ] || \
    "$PY" run_mitigation_ab.py --slice "$SL" --detector existing --run-dir "$RD" --skip-off --label full >> "$EXP/e1_${V}_stdout.txt" 2>&1
  [ -f "$RD/on_combo_sanitize_grounding_refuse.jsonl" ] && [ "$(wc -l < "$RD/on_combo_sanitize_grounding_refuse.jsonl")" -ge 200 ] || \
    "$PY" run_mitigation_ab.py --slice "$SL" --detector existing --run-dir "$RD" --skip-off \
      --label combo_sanitize_grounding_refuse --layers sanitize,grounding,refuse >> "$EXP/e1_${V}_stdout.txt" 2>&1
  for f in off on_full on_combo_sanitize_grounding_refuse; do echo "E1 $V $f rows=$(wc -l < "$RD/$f.jsonl" 2>/dev/null)"; done
done
date +"all done %H:%M"
