# MITIGATION.md — measure ASR OFF vs ON (existing detector + mitigation)

Goal: use an existing detector as a fixed input, put the contribution on
**mitigation**, and produce **multi-vector attack success rate (ASR) with mitigation
OFF vs ON** — plus benign cost and layer ablation.

All commands run from `step4 dataset/`. **Results go in separate directories**
(`--run-dir`) so runs never overwrite each other.

> **PowerShell:** use **one command per line**. Do not use bash `\` line continuation.
> Multi-line in PowerShell requires a trailing backtick `` ` `` — single-line is safer.

---

## Step 0 — modules (in repo)

```
detector_interface.py       mitigation_pipeline.py
run_mitigation_ab.py        score_mitigation_ab.py
check_off_asr.py            inspect_mitigation_run.py
run_benign_cost.py          score_benign_cost.py
preflight_check.py
```

---

## Step 1 — OFF-arm gate (run BEFORE any full A/B)

An A/B is meaningless if attacks don't succeed with mitigation **OFF**. Run the gate first:

```powershell
python preflight_check.py --slice planted_inband.jsonl
```

```powershell
python check_off_asr.py --slice planted_inband.jsonl --limit 30 --show-misses 10 --run-dir mitigation_results/planted30
```

Or score an existing OFF file:

```powershell
python check_off_asr.py --off mitigation_results/planted30/off.jsonl --show-misses 10
```

**Verdicts**

| Verdict | OFF ASR | Action |
|---|---|---|
| **HEALTHY** | >= 40% | Proceed to Step 2 (full A/B) |
| **MARGINAL** | 15–39% | A/B possible; consider stronger payloads or scale n |
| **TOO-LOW** | < 15% | **Stop.** Fix slice payloads; do not run full A/B yet |

Read **"Why rows did NOT fire"**. If `answered_truthfully` dominates, injections are
too weak — strengthen planted payloads, then re-run the gate.

**Honest fallback:** if attacks still won't fire after strengthening, that *is* the
result: undefended multi-vector attacks don't instantiate on this pipeline. Document
the gate output as evidence.

Equivalent (OFF arm only, no scoring):

```powershell
python run_mitigation_ab.py --slice planted_inband.jsonl --detector existing --off-only --run-dir mitigation_results/planted30 --limit 30
```

---

## Step 2 — smoke test wiring (5 rows, not for numbers)

```powershell
python run_mitigation_ab.py --slice planted_inband.jsonl --detector existing --limit 5 --run-dir mitigation_results/smoke5 --label full
```

```powershell
python inspect_mitigation_run.py --run-dir mitigation_results/smoke5
```

OFF: mostly not blocked, answers present. ON: mitigation acts. If identical, fix wiring.

---

## Step 3 — full A/B (only after gate says HEALTHY or MARGINAL)

```powershell
python run_mitigation_ab.py --slice planted_inband.jsonl --detector existing --run-dir mitigation_results/planted30 --label full
```

```powershell
python score_mitigation_ab.py --off mitigation_results/planted30/off.jsonl --on mitigation_results/planted30/on_full.jsonl --out mitigation_results/planted30/report_full.json
```

```powershell
python inspect_mitigation_run.py --run-dir mitigation_results/planted30
```

n=30 gives wide CIs — first real signal, not thesis-final.

---

## Step 4 — benign cost + trade-off

```powershell
python run_benign_cost.py --qa data/question-answer/test.jsonl --n 200 --out-off mitigation_results/planted30/benign_off.jsonl --out-on mitigation_results/planted30/benign_on.jsonl
```

```powershell
python score_benign_cost.py --off mitigation_results/planted30/benign_off.jsonl --on mitigation_results/planted30/benign_on.jsonl --asr-report mitigation_results/planted30/report_full.json --out mitigation_results/planted30/benign_cost_report.json
```

---

## Step 5 — read attribution; ablate if refuse is too blunt

```powershell
python -c "import json; r=json.load(open('mitigation_results/planted30/report_full.json')); print('ASR fixed by layer:', r.get('fixed_by_layer')); b=json.load(open('mitigation_results/planted30/benign_cost_report.json')); print('Benign broken by cause:', b.get('broken_by_cause'))"
```

If **refuse-on-detection** dominates reduction *and* benign cost:

```powershell
python run_mitigation_ab.py --slice planted_inband.jsonl --skip-off --no-refuse --run-dir mitigation_results/planted30 --label norefuse
```

```powershell
python score_mitigation_ab.py --off mitigation_results/planted30/off.jsonl --on mitigation_results/planted30/on_norefuse.jsonl --out mitigation_results/planted30/report_norefuse.json
```

Verification-only experiment:

```powershell
python run_mitigation_ab.py --slice planted_inband.jsonl --skip-off --only-grounding --run-dir mitigation_results/planted30 --label groundonly
```

```powershell
python score_mitigation_ab.py --off mitigation_results/planted30/off.jsonl --on mitigation_results/planted30/on_groundonly.jsonl --out mitigation_results/planted30/report_groundonly.json
```

Scale whichever configuration wins.

---

## Step 6 — scale slice + final thesis run

```powershell
python build_multivector_inband.py --qa-jsonl data/question-answer/test.jsonl --planted --n 200 --out planted_inband_200.jsonl
```

```powershell
python check_off_asr.py --slice planted_inband_200.jsonl --run-dir mitigation_results/planted200
```

```powershell
python run_mitigation_ab.py --slice planted_inband_200.jsonl --detector existing --run-dir mitigation_results/planted200 --label full
```

```powershell
python score_mitigation_ab.py --off mitigation_results/planted200/off.jsonl --on mitigation_results/planted200/on_full.jsonl --out mitigation_results/planted200/report_full.json
```

Resolve **bge-m3** / SSL before treating numbers as final (not TF-IDF fallback).

---

## Output layout

```
mitigation_results/
  planted30/
    off.jsonl                 off_asr_gate.json
    on_full.jsonl             report_full.json
    on_norefuse.jsonl         report_norefuse.json
    benign_off.jsonl          benign_cost_report.json
```

---

## Decision tree (short)

1. Fix PowerShell syntax (single-line commands).
2. `check_off_asr.py` on n=30 OFF only.
3. **TOO-LOW** → strengthen payloads → re-gate (or document robustness finding).
4. **HEALTHY/MARGINAL** → full A/B → benign cost → attribution → ablate → scale.

## What NOT to claim

- Full A/B when OFF-arm ASR is ~0% (meaningless 0→0 table).
- n=30 ASR as final without scaling.
- ASR win without benign cost.
