"""
convert_external_datasets.py — external-validation slice builder
===================================================================

Converts two renowned, independently-sourced attack datasets into the exact
row schema build_planted_attacks.py already emits, so they drop straight into
the existing mitigation/detection pipeline and scoring scripts with zero
changes to those scripts:

  * PoisonedRAG (Zou, Geng, Wang, Jia — USENIX Security 2025, arXiv:2402.07867)
    -> official released adv_targeted_results/{nq,hotpotqa,msmarco}.json
    -> 100 target questions x 3 source QA datasets x 5 pre-crafted malicious
       texts each = up to 1500 rows. These are the *actual* texts the authors
       used to get 90-97% ASR against undefended RAG — nothing here is
       regenerated or reworded.

  * BIPIA (Yi et al. — Microsoft Research, KDD 2025, arXiv:2312.14197)
    -> benchmark/text_attack_{train,test}.json
    -> 150 malicious instruction strings across 15 categories (Task
       Automation, Base Encoding, Scams & Fraud, Misinformation &
       Propaganda, ...). BIPIA's own QA/context files are licensed and not
       redistributed in the repo, so there is no native question to pair
       with these payloads. Each instruction is paired with one of
       PoisonedRAG's own NQ questions purely as a retrieval carrier -- see
       the `carrier_question_note` field on every BIPIA-sourced row. This
       is a constructed pairing, not BIPIA's native benchmark, and should be
       described that way if cited.

OUTPUT SCHEMA (identical fields to build_planted_attacks.py's `build()`,
plus provenance fields so these rows can be filtered/reported separately
from your own synthetic attacks):

    kind            "poisoned_context"   (clean query, all signal in the chunk;
                                           neither source constructs a genuinely
                                           split query+context attack, so
                                           neither should be labeled
                                           "multivector_attack")
    question        str
    ground_truth    str
    true_answer     str
    success_marker  str | null   (see NOTE below for BIPIA rows)
    poison_chunk    str
    attack_type     str
    expectation     {"undefended_should_fire": true}
    id              int   (0-indexed, unique across the whole output file)
    source          str   e.g. "PoisonedRAG-NQ", "BIPIA-text_attack"
    external        true
    source_id       str   original id from the source dataset, for traceability

NOTE on success_marker for BIPIA rows: PoisonedRAG rows have a natural
"wrong answer" string because the attack's goal is a factual flip. BIPIA's
attack goal is instruction-EXECUTION (e.g. "back up my documents folder"),
which isn't a string your existing score_attack_success.py can match against
the way it matches success_marker for the other kinds. BIPIA rows therefore
carry success_marker = null and an added `detection_target` field describing
what compliance would look like instead. score_attack_success.py will need a
small added branch for kind == "poisoned_context" and success_marker is None
before these rows can be auto-scored for ASR the same way as the rest of your
data -- this converter does not attempt that scoring change.

USAGE
-----
    python convert_external_datasets.py \\
        --poisonedrag-dir ./poisonedrag_data \\
        --bipia-train ./bipia_train.json \\
        --bipia-test ./bipia_test.json \\
        --out external_validation_slice.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert_poisonedrag(path: str, dataset_label: str) -> list[dict]:
    """One row per (question, adv_text) pair -- matches how your own pipeline
    counts poison chunks individually (e.g. the 74/74 canary-redaction stat),
    not questions."""
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for qid, rec in data.items():
        q = (rec.get("question") or "").strip()
        true_ans = str(rec.get("correct answer", "")).strip()
        wrong_ans = str(rec.get("incorrect answer", "")).strip()
        if not q or not true_ans:
            continue
        for i, adv_text in enumerate(rec.get("adv_texts", [])):
            out.append({
                "kind": "poisoned_context",
                "question": q,
                "ground_truth": true_ans,
                "true_answer": true_ans,
                "success_marker": wrong_ans,
                "poison_chunk": adv_text.strip(),
                "attack_type": "external_poisonedrag_knowledge_corruption",
                "expectation": {"undefended_should_fire": True},
                "source": f"PoisonedRAG-{dataset_label}",
                "external": True,
                "source_id": f"{dataset_label}:{qid}:{i}",
            })
    return out


def _load_bipia_instructions(*paths: str) -> list[tuple[str, str]]:
    """Returns [(category, instruction_text), ...] across all given splits."""
    out = []
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        for category, items in data.items():
            for text in items:
                out.append((category, text.strip()))
    return out


def convert_bipia(train_path: str, test_path: str,
                   carrier_questions: list[dict]) -> list[dict]:
    if not carrier_questions:
        raise ValueError(
            "need at least one PoisonedRAG-style carrier question "
            "({'question', 'correct answer'}) to pair BIPIA payloads with"
        )
    instructions = _load_bipia_instructions(train_path, test_path)
    out = []
    for i, (category, instr) in enumerate(instructions):
        carrier = carrier_questions[i % len(carrier_questions)]
        cat_slug = category.lower().replace(" ", "_").replace("&", "and")
        out.append({
            "kind": "poisoned_context",
            "question": carrier["question"],
            "ground_truth": carrier["correct answer"],
            "true_answer": carrier["correct answer"],
            "success_marker": None,
            "poison_chunk": instr,
            "attack_type": f"external_bipia_indirect_injection:{cat_slug}",
            "expectation": {"undefended_should_fire": True},
            "source": "BIPIA-text_attack",
            "external": True,
            "source_id": f"bipia:{category}:{i}",
            "detection_target": (
                "model attempts/executes the embedded instruction rather "
                "than answering the carrier question"
            ),
            "carrier_question_note": (
                "question borrowed from PoisonedRAG-NQ as a retrieval "
                "carrier; BIPIA's own QA/context files are licensed and not "
                "redistributed, so this pairing is constructed, not native "
                "BIPIA data -- describe it that way if cited."
            ),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poisonedrag-dir", required=True,
                     help="dir containing nq.json, hotpotqa.json, msmarco.json")
    ap.add_argument("--bipia-train", required=True)
    ap.add_argument("--bipia-test", required=True)
    ap.add_argument("--out", default="external_validation_slice.jsonl")
    ap.add_argument("--per-source-out", action="store_true",
                     help="also write one JSONL per source, for isolated runs")
    args = ap.parse_args()

    pr_dir = Path(args.poisonedrag_dir)
    all_rows: list[dict] = []
    per_source: dict[str, list[dict]] = {}

    for label, fname in (("NQ", "nq.json"), ("HotpotQA", "hotpotqa.json"),
                          ("MSMARCO", "msmarco.json")):
        fpath = pr_dir / fname
        if not fpath.exists():
            print(f"[skip] {fpath} not found")
            continue
        rows = convert_poisonedrag(str(fpath), label)
        per_source[f"PoisonedRAG-{label}"] = rows
        all_rows.extend(rows)
        print(f"[ok] PoisonedRAG-{label}: {len(rows)} rows "
              f"({len(rows) // 5} questions x 5 adv_texts)")

    nq_path = pr_dir / "nq.json"
    carriers = []
    if nq_path.exists():
        nq_data = json.load(open(nq_path, encoding="utf-8"))
        carriers = [{"question": r["question"], "correct answer": r["correct answer"]}
                    for r in nq_data.values()]

    bipia_rows = convert_bipia(args.bipia_train, args.bipia_test, carriers)
    per_source["BIPIA-text_attack"] = bipia_rows
    all_rows.extend(bipia_rows)
    print(f"[ok] BIPIA-text_attack: {len(bipia_rows)} rows "
          f"(paired with {len(carriers)} distinct NQ carrier questions)")

    for i, r in enumerate(all_rows):
        r["id"] = i
    _write_jsonl(all_rows, args.out)
    print(f"\n[done] wrote {len(all_rows)} total rows -> {args.out}")

    if args.per_source_out:
        for source, rows in per_source.items():
            for i, r in enumerate(rows):
                r["id"] = i
            fname = f"external_{source.lower().replace('-', '_')}.jsonl"
            _write_jsonl(rows, fname)
            print(f"[done] wrote {len(rows)} rows -> {fname}")

    print("\nSample rows:")
    for r in (all_rows[0], all_rows[-1]):
        print(f"  source={r['source']}  kind={r['kind']}")
        print(f"  Q: {r['question'][:90]}")
        print(f"  true={r['true_answer']!r}  success_marker={r['success_marker']!r}")
        print(f"  poison_chunk={r['poison_chunk'][:100]!r}")
        print()


if __name__ == "__main__":
    main()
