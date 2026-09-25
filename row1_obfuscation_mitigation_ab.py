#!/usr/bin/env python3
"""
row1_obfuscation_mitigation_ab.py  --  Row 1 external validation:
obfuscation + prompt injection, mitigation OFF vs ON

===========================================================================
WHY THIS SCRIPT EXISTS
===========================================================================
compat_check.py / obfuscate_slice.py / normalize_slice.py already exist and
are still useful for ad-hoc inspection, but running them as three separate
steps and then eyeballing three separate `step_03c_fusion_gate.py --eval`
console dumps (redirected to gate_eval_*.txt with PowerShell's default
encoding) has two problems this script fixes:

  1. `_collect()` in step_03c_fusion_gate.py scores row["prompt"] EXACTLY as
     written -- it never runs Step 2 (normalization) first. So a prior
     "gate_eval_normalized.txt vs gate_eval_obfuscated.txt" comparison was
     NOT an apples-to-apples mitigation-off-vs-on comparison on the same
     rows; each file was whatever was in its own already-built slice, scored
     with the exact same code path either way. This script builds ONE
     obfuscated file and scores it TWICE in-process: once with `normalize=
     False` (mitigation OFF -- what the harness always did before) and once
     with `normalize=True` (mitigation ON -- the new --normalize flag added
     to step_03c_fusion_gate.py, which runs the real Preprocessor, including
     the new base64-span decode, before scoring). Same rows, same detector,
     only the normalization step differs.

  2. PowerShell's `> file.txt` defaults to UTF-16LE, so redirected console
     output read back as UTF-8 comes out space-padded per character (that is
     what gate_eval_normalized.txt / gate_eval_obfuscated.txt look like).
     This script writes its report directly as UTF-8 JSON via `json.dump`,
     never through a console redirect, so that problem cannot recur. If you
     still want a human-readable console capture, redirect with
     `| Out-File -Encoding utf8` or `| Set-Content -Encoding utf8`.

===========================================================================
WHAT IT MEASURES
===========================================================================
Using real attacks from deepset/prompt-injections (already downloaded to
row1_obfuscation_prompt_injection/deepset_prompt_injections_test.jsonl):

  raw_baseline          : original, unobfuscated attacks + benign, scored
                          with normalize=False. Reference point only.
  obfuscated_off        : the SAME attacks after obfuscation (zero-width /
                          fullwidth / base64-wrap, round-robin per row),
                          scored with normalize=False. This is the
                          "mitigation absent" arm -- what an undefended gate
                          sees.
  obfuscated_on         : the identical obfuscated file, scored with
                          normalize=True. This is the "mitigation present"
                          arm -- what the gate sees once Step 2 (incl. the
                          new base64 decode) runs first.

Per-technique recall is reported separately (via step_03c_fusion_gate's own
`attack_type` breakdown) so a technique that a wrapper phrase alone makes
"accidentally easy" (e.g. base64-wrap's own English wrapper sentence reads as
suspicious on its face) does not hide inside an aggregate number.

===========================================================================
USAGE (run from the project root, venv311 active, Ollama serving)
===========================================================================
    python row1_obfuscation_mitigation_ab.py --n-attack 30 --n-benign 30

Needs: dataset/step_03c_fusion_gate.py, dataset/preprocess_dataset.py,
obfuscate_slice.py (all already in this repo), Ollama running
llama-guard3:1b, and the deberta-v3 injection model (downloaded on first
use). This cannot run from a shell without those installed -- see the
runbook for the exact PowerShell invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

# Reuse the three obfuscation techniques already written and validated in
# obfuscate_slice.py, rather than re-implementing them here.
from obfuscate_slice import TECHNIQUES  # noqa: E402

DEFAULT_SOURCE = HERE / "row1_obfuscation_prompt_injection" / "deepset_prompt_injections_test.jsonl"


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _text_and_label(row: dict) -> tuple[str, int] | None:
    text = row.get("text") or row.get("prompt")
    label = row.get("label")
    if text is None or label is None:
        return None
    return text, int(label)


def build_slices(source: Path, n_attack: int, n_benign: int) -> tuple[list[dict], list[dict]]:
    """Returns (raw_rows, obfuscated_rows). Both share the identical benign
    rows (unobfuscated -- obfuscating benign text is not part of this threat
    model) so the specificity/false-positive comparison is also matched."""
    rows = _load_rows(source)
    attacks, benigns = [], []
    for r in rows:
        parsed = _text_and_label(r)
        if parsed is None:
            continue
        text, label = parsed
        if label == 1 and len(attacks) < n_attack:
            attacks.append(text)
        elif label == 0 and len(benigns) < n_benign:
            benigns.append(text)
        if len(attacks) >= n_attack and len(benigns) >= n_benign:
            break

    if len(attacks) < n_attack:
        print(f"[warn] only found {len(attacks)}/{n_attack} attack rows in {source.name}")
    if len(benigns) < n_benign:
        print(f"[warn] only found {len(benigns)}/{n_benign} benign rows in {source.name}")

    raw_rows, obf_rows = [], []
    for i, text in enumerate(attacks):
        tech_name, tech_fn = TECHNIQUES[i % len(TECHNIQUES)]
        raw_rows.append({
            "prompt": text, "safety": "unsafe",
            "attack_type": "external_row1_raw",
            "_source_row_index": i, "_technique": "none",
        })
        obf_rows.append({
            "prompt": tech_fn(text), "safety": "unsafe",
            "attack_type": f"external_row1_obfuscated:{tech_name}",
            "_source_row_index": i, "_technique": tech_name,
            "_original_text": text,
        })
    for i, text in enumerate(benigns):
        benign_row = {
            "prompt": text, "safety": "safe",
            "attack_type": "external_row1_benign",
            "_source_row_index": i, "_technique": "none",
        }
        raw_rows.append(dict(benign_row))
        obf_rows.append(dict(benign_row))
    return raw_rows, obf_rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE),
                     help="deepset prompt-injections JSONL (text/label fields)")
    ap.add_argument("--n-attack", type=int, default=30)
    ap.add_argument("--n-benign", type=int, default=30)
    ap.add_argument("--out-dir", default="row1_mitigation_results")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--review-as-red", action="store_true")
    args = ap.parse_args()

    # Imported here (not at module top) so --help works without the ML deps.
    from step_03c_fusion_gate import evaluate, FusionConfig  # noqa: E402

    out_dir = Path(args.out_dir)
    raw_rows, obf_rows = build_slices(Path(args.source), args.n_attack, args.n_benign)
    raw_path = out_dir / "row1_raw.jsonl"
    obf_path = out_dir / "row1_obfuscated.jsonl"
    _write_jsonl(raw_rows, raw_path)
    _write_jsonl(obf_rows, obf_path)
    print(f"[build] wrote {len(raw_rows)} raw rows -> {raw_path}")
    print(f"[build] wrote {len(obf_rows)} obfuscated rows -> {obf_path}")
    by_tech = {}
    for r in obf_rows:
        by_tech[r["_technique"]] = by_tech.get(r["_technique"], 0) + 1
    print(f"[build] technique breakdown (incl. benign 'none'): {by_tech}")

    cfg = FusionConfig()
    print("\n[scoring] raw_baseline (unobfuscated attacks, normalize=False) ...")
    raw_baseline = evaluate(str(raw_path), cfg=cfg, base_url=args.base_url,
                             treat_review_as_red=args.review_as_red, normalize=False)

    print("\n[scoring] obfuscated_off (mitigation OFF: normalize=False) ...")
    obfuscated_off = evaluate(str(obf_path), cfg=cfg, base_url=args.base_url,
                               treat_review_as_red=args.review_as_red, normalize=False)

    print("\n[scoring] obfuscated_on (mitigation ON: normalize=True, incl. base64 decode) ...")
    obfuscated_on = evaluate(str(obf_path), cfg=cfg, base_url=args.base_url,
                              treat_review_as_red=args.review_as_red, normalize=True)

    report = {
        "n_attack": len(raw_rows) and sum(1 for r in raw_rows if r["safety"] == "unsafe"),
        "n_benign": sum(1 for r in raw_rows if r["safety"] == "safe"),
        "raw_baseline": raw_baseline,
        "obfuscated_off_mitigation": obfuscated_off,
        "obfuscated_on_mitigation": obfuscated_on,
    }
    report_path = out_dir / "row1_mitigation_ab_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {report_path}")

    def _fusion(rep):
        return rep["fusion"]

    print("\n=== ROW 1 OBFUSCATION MITIGATION -- SUMMARY (C3RF fused decision) ===")
    for label, rep in (("raw baseline (no obfuscation)", raw_baseline),
                        ("obfuscated, mitigation OFF (raw text scored)", obfuscated_off),
                        ("obfuscated, mitigation ON (Step-2 normalize first)", obfuscated_on)):
        f = _fusion(rep)
        print(f"  {label:52} P={f['precision']:.3f} R={f['recall']:.3f} "
              f"F1={f['f1']:.3f} FP={f['fp']} FN={f['fn']}")
    print("\nPer-technique recall, mitigation OFF vs ON (from per_attack_fusion):")
    off_tech = obfuscated_off["per_attack_fusion"]
    on_tech = obfuscated_on["per_attack_fusion"]
    for tech in sorted(set(off_tech) | set(on_tech)):
        if tech in ("external_row1_benign",):
            continue
        o = off_tech.get(tech, {})
        n_ = on_tech.get(tech, {})
        print(f"  {tech:45} OFF R={o.get('recall', 0):.3f} (n={o.get('n', 0):<3})  "
              f"ON R={n_.get('recall', 0):.3f} (n={n_.get('n', 0):<3})")
    print("\nBenign false-positive check (external_row1_benign), OFF vs ON:")
    ob = off_tech.get("external_row1_benign", {})
    onb = on_tech.get("external_row1_benign", {})
    print(f"  OFF: FP={ob.get('fp', 'n/a')}/{ob.get('n', 'n/a')}   "
          f"ON: FP={onb.get('fp', 'n/a')}/{onb.get('n', 'n/a')}")


if __name__ == "__main__":
    main()
