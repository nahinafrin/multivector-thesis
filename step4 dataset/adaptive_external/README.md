# adaptive_external/ — recovered adaptive-tier pipeline (external PoisonedRAG and BIPIA results)

This folder holds the code that produced `../ext_n200_results.jsonl`. That file is the source of the
external results in both papers. It is kept separate from the complete 13-stage pipeline (Protocol A) and
from the seven-layer harness (Protocol B).

## Contents

| Path | What it is |
|---|---|
| `original/` | The four recovered scripts, byte-identical to their last committed version (`fe71c56^`). |
| `runtime_snapshot_1d99b23/` | The 25 local modules they import, as committed with the results (`1d99b23`). |
| `environment_at_run/` | Ollama version, model digests and pip lock committed with the run (`92ed249`). |
| `historical_docs/INTEGRATION.md` | Hook documentation for the adaptive orchestrator. |
| `PROVENANCE.json` | Commit, git blob and SHA-256 for every recovered file. |
| `run_adaptive_external.py` | **New** launcher. It runs the original CLI against the snapshot and contains no pipeline logic. |
| `verification_report.json` | Output of `../verify_adaptive_external.py`. |

## Usage

```powershell
cd "step4 dataset"
.\.venv311\Scripts\python.exe adaptive_external\run_adaptive_external.py --check-imports
.\.venv311\Scripts\python.exe verify_adaptive_external.py --recompute-graded
```

The first command checks that every module loads from the recovered code. The second verifies
provenance and recomputes the reported numbers from the saved outputs.

For provenance, the stage-execution matrix, fresh-inference commands and limitations, see
[`docs/ADAPTIVE_EXTERNAL_REPRODUCIBILITY.md`](../../docs/ADAPTIVE_EXTERNAL_REPRODUCIBILITY.md).
