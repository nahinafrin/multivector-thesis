#!/usr/bin/env python3
"""
run_llmguard_protocolB.py -- LLM-Guard prompt-injection firewall as an existing-defense baseline
on Protocol B (200 planted attacks, 200 benign questions).

Defense: the LLM-Guard PromptInjection input scanner (protectai/deberta-v3-base-prompt-injection-v2,
MatchType.FULL, threshold 0.5 -- the same configuration as run_external_guard_baseline.py) is run
on the user query and on every passage the undefended pipeline placed in the prompt
(dump_undefended_contexts.py). If any is flagged, the query is refused; otherwise the undefended
pipeline's stored answer is delivered unchanged. Because the firewall only blocks or passes, this
reproduces an LLM-Guard-guarded deployment of the undefended pipeline without regenerating answers.

    python run_llmguard_protocolB.py
Output: mitigation_results/baselines_protocolB/llmguard_{attack,benign}.jsonl
        (index, blocked, flagged_query, flagged_passages, max_risk)
"""
from __future__ import annotations

import json
from pathlib import Path

from run_external_guard_baseline import _scan

HERE = Path(__file__).resolve().parent
D = HERE / "mitigation_results" / "baselines_protocolB"
THRESHOLD = 0.5


def main() -> None:
    from llm_guard.input_scanners import PromptInjection
    from llm_guard.input_scanners.prompt_injection import MatchType
    scanner = PromptInjection(threshold=THRESHOLD, match_type=MatchType.FULL)
    for split in ("attack", "benign"):
        rows = [json.loads(l) for l in open(D / f"contexts_{split}.jsonl", encoding="utf-8") if l.strip()]
        with (D / f"llmguard_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for i, r in enumerate(rows, 1):
                q = _scan(scanner, r["question"])
                ps = [_scan(scanner, c) for c in r["ranked_context"]]
                out = {"index": r["index"], "flagged_query": q["flagged"],
                       "flagged_passages": sum(p["flagged"] for p in ps),
                       "max_risk": max([q["risk"]] + [p["risk"] for p in ps])}
                out["blocked"] = out["flagged_query"] or out["flagged_passages"] > 0
                fh.write(json.dumps(out) + "\n")
                if i % 25 == 0:
                    print(f"[{split}] {i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
