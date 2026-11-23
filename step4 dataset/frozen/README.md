# frozen/ — historical snapshots

These files are earlier snapshots kept for the record. **None of them is the source of a number in
the current paper**; the root README's reproduction table names the sources.

* `slice_scoring_historical_210row.json` (formerly `slice_scoring.json`): `score_slice.py` output for
  an intermediate 210-row run of the rebuilt slice. Its `input` field names `grounded_controller.jsonl`,
  but that file has since been replaced (now 222 rows), so re-scoring it gives different counts. The
  paper's Protocol A numbers come from `../protocolA_220/` (`../score_protocolA_220.py`).
* `slice_scoring_audit.json`, `slice_scoring_prod.json`, `slice_causal_results.json`,
  `results_subsection_slice.md`, `methodology_subsection_unified_state.md`: earlier scoring runs
  and write-ups of the slice.
* `gate_scores.jsonl`, `generated.jsonl`, `grounded.jsonl`, `results.jsonl`, `retrieval_audit.jsonl`,
  `thesis_results_locked.json`, `ablation_results.json`: the earlier 918-row benign run on
  rag-mini-wikipedia and its locked summary.
* `backups_files3/`: copies of two files before an earlier edit.
