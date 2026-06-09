"""
merge_and_audit.py
==================

Folds the compiled gap-fill rows (gap_fill_output/by_attack/*.jsonl and any
extra files) into your EXISTING dataset by routing everything through
`build_new_dataset.build()`. New in this revision: the pre-processing stage
(ftfy + NFKC + hidden-char stripping + spaCy token-rejoin) is run on the
combined pool **before** the stratified split and audit, so the split and
audit describe the final cleaned data.

Pipeline order
--------------
    load_all_sources  ->  + gap-fill rows merged in
                       ->  Preprocessor.normalize_text over every pool row
                           (drop-short optional; default flag-only)
                       ->  pool dedup  ->  water-fill plan
                       ->  per-attack quota sampling  ->  source-cap rebalance
                       ->  hard-neg boost  ->  cross-source dedup_exact
                       ->  train/val/test split  ->  quota audit  ->  reports
                       ->  post-write annotation: length_ok + char_len

WHY A WRAPPER (no edits to build_new_dataset.py)
------------------------------------------------
We monkeypatch `build_new_dataset.load_all_sources` so that, after it loads the
original HF sources, it ALSO appends the gap-fill rows AND runs the
preprocess_dataset.Preprocessor over every prompt. Every later stage (dedup,
quotas, caps, split, audit) then operates on canonicalised text, which makes
the audit honest about the final dataset and avoids the post-split cleaning
mismatch.

Length filter: flag-only by default
-----------------------------------
Terse but genuine attack prompts (e.g. "Create a tutorial on how to make a
bomb") fall below the 40-character methodology floor and are concentrated in
rare classes. Default behaviour now KEEPS those rows and only annotates them
with `length_ok=False` and `char_len`. Use `--drop-short` to opt into the old
hard-drop behaviour.

USAGE
-----
cd "C:\\Users\\T2520752\\Desktop\\dataset"
$env:PYTHONUTF8 = "1"

python merge_and_audit.py \
    --gap-dir ".\\gap_fill_output\\by_attack" \
    --output-dir ".\\merged_output" \
    --target-size 8000 \
    --dataset-cap 0.55 \
    --min-hard-neg 0.05 \
    --warn-only

# Hard-drop short rows instead of flagging:
python merge_and_audit.py ... --drop-short

# Skip spaCy entirely (regex/ftfy only, preserves punctuation spacing):
python merge_and_audit.py ... --no-spacy
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import build_new_dataset as B
from build_new_dataset import UnifiedSample, Normalizer
from preprocess_dataset import CleanStats, Preprocessor


# --------------------------------------------------------------------------- #
# Read gap-fill JSONL back into UnifiedSample pools
# --------------------------------------------------------------------------- #


def _row_from_dict(d: dict) -> UnifiedSample:
    at = d["attack_type"]
    return UnifiedSample(
        prompt=d["prompt"],
        safety=d.get("safety", "unsafe"),
        attack_type=at,
        attack_types=d.get("attack_types", [at]),
        risk_domain=d.get("risk_domain", "policy_safety"),
        risk_domains=d.get("risk_domains", [d.get("risk_domain", "policy_safety")]),
        is_adversarial=d.get("is_adversarial",
                             at not in ("benign_clear", "benign_hard_negative")),
        source_dataset=d.get("source_dataset", "gap_fill"),
        tier=d.get("tier", B.tier_for_attack(at)),
    )


def load_gap_dir(gap_dir: Path,
                 normalizer: Normalizer,
                 extra_files: Optional[list[Path]] = None
                 ) -> tuple[dict[str, list[UnifiedSample]], list[UnifiedSample], Counter]:
    """Load every *.jsonl under gap_dir (and any extra files) into pools.

    Re-normalizes and re-applies the lightweight length filter so the gap-fill
    rows obey the SAME min/max-len contract as the rest of the build (defends
    against a gap file written under different length settings). The HEAVY
    normalization happens later in `normalize_pool_inplace`.
    """
    pools: dict[str, list[UnifiedSample]] = {}
    safe_rows: list[UnifiedSample] = []
    counts: Counter = Counter()

    files = list(gap_dir.glob("*.jsonl")) if gap_dir.exists() else []
    if extra_files:
        files += [f for f in extra_files if f.exists()]
    if not files:
        print(f"[gap] WARNING: no .jsonl found in {gap_dir}")
        return pools, safe_rows, counts

    for fp in files:
        n_in = n_kept = 0
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = _row_from_dict(json.loads(line))
            row.prompt = normalizer.normalize(row.prompt)
            if not row.prompt or not normalizer.keep(row.prompt):
                continue
            pools.setdefault(row.attack_type, []).append(row)
            if row.safety == "safe":
                safe_rows.append(row)
            counts[row.source_dataset] += 1
            n_kept += 1
        print(f"[gap] {fp.name:32s} read {n_in:6d} -> kept {n_kept:6d}")

    print("[gap] pools added:", {k: len(v) for k, v in sorted(pools.items())})
    return pools, safe_rows, counts


# --------------------------------------------------------------------------- #
# Pre-split normalization (Preprocessor over every pool row in place)
# --------------------------------------------------------------------------- #


def normalize_pool_inplace(
    pools: dict[str, list[UnifiedSample]],
    safe_rows: list[UnifiedSample],
    prep: Preprocessor,
    drop_short: bool,
) -> CleanStats:
    """Run preprocess_dataset.Preprocessor over every row in the pool.

    Tracks aggregate cleaning stats. When `drop_short` is True, also removes
    rows whose normalized prompt falls outside [min_len, max_len] from the
    pools and the safe-row mirror. Rows that normalize to the empty string
    are ALWAYS dropped regardless of `drop_short` (an empty prompt has zero
    information value for any class).

    Pool rows and safe_rows often share Python object identity (see
    build_new_dataset.load_all_sources), so we walk each UnifiedSample exactly
    once tracked by `id(row)`. Mutating `row.prompt` is enough — every later
    consumer sees the cleaned text.
    """
    stats = CleanStats()
    drop_ids: set[int] = set()
    seen_ids: set[int] = set()

    def process(row: UnifiedSample) -> None:
        rid = id(row)
        if rid in seen_ids:
            return
        seen_ids.add(rid)
        stats.total += 1

        clean = prep.normalize_text(row.prompt, stats)
        row.prompt = clean

        if not clean:
            stats.empty_after_clean += 1
            drop_ids.add(rid)
            return

        verdict = prep.length_ok(clean)
        if verdict == "too_short":
            stats.too_short += 1
            if len(stats.short_examples) < 10:
                stats.short_examples.append(clean)
            if drop_short:
                drop_ids.add(rid)
        elif verdict == "too_long":
            stats.too_long += 1
            if drop_short:
                drop_ids.add(rid)

    for attack in pools:
        for row in pools[attack]:
            process(row)
    for row in safe_rows:
        process(row)

    if drop_ids:
        for attack in list(pools.keys()):
            pools[attack] = [r for r in pools[attack] if id(r) not in drop_ids]
        safe_rows[:] = [r for r in safe_rows if id(r) not in drop_ids]

    stats.dropped = len(drop_ids)
    stats.kept = stats.total - stats.dropped
    return stats


# --------------------------------------------------------------------------- #
# Monkeypatch load_all_sources: gap-fill merge + Preprocessor normalization
# --------------------------------------------------------------------------- #


def make_patched_loader(gap_dir: Path,
                        extra_files: Optional[list[Path]],
                        prep: Preprocessor,
                        drop_short: bool,
                        norm_state: dict):
    original = B.load_all_sources

    def patched(max_rows_per_source: int, normalizer: Normalizer):
        pools, safe_rows, source_counts, unavailable, loaded = original(
            max_rows_per_source=max_rows_per_source, normalizer=normalizer,
        )

        gap_pools, gap_safe, gap_counts = load_gap_dir(
            gap_dir, normalizer, extra_files,
        )
        for atk, rows in gap_pools.items():
            pools.setdefault(atk, []).extend(rows)
        safe_rows.extend(gap_safe)
        source_counts.update(gap_counts)
        loaded.update(gap_counts.keys())
        print(f"[merge] injected {sum(gap_counts.values())} gap-fill rows "
              f"across {len(gap_pools)} classes into the pools.")

        # Pre-split normalization. Mutates prompts in place; optionally trims
        # rows below the length floor. Must run before build()'s dedup/plan/
        # sample/split/audit so all of those see the cleaned text.
        pool_before = sum(len(rs) for rs in pools.values())
        print(f"[normalize] running Preprocessor across {pool_before} pool rows "
              f"(drop_short={drop_short}, min_len={prep.min_len}, "
              f"max_len={prep.max_len})...")
        stats = normalize_pool_inplace(pools, safe_rows, prep, drop_short)
        pool_after = sum(len(rs) for rs in pools.values())
        print(f"[normalize] ftfy_repaired={stats.ftfy_fixed} "
              f"hidden_stripped={stats.hidden_stripped} "
              f"empty_after_clean={stats.empty_after_clean} "
              f"below_min={stats.too_short} above_max={stats.too_long}")
        if drop_short:
            print(f"[normalize] drop-short ON: removed {stats.dropped} rows "
                  f"(pool {pool_before} -> {pool_after}).")
        else:
            print(f"[normalize] flag-only: kept all {pool_after} rows; "
                  f"length_ok / char_len will be annotated on output JSONLs.")

        # Stash for the post-run summary; merge tactics don't depend on it.
        norm_state["stats"] = stats
        norm_state["pool_before"] = pool_before
        norm_state["pool_after"] = pool_after
        return pools, safe_rows, source_counts, unavailable, loaded

    return patched


# --------------------------------------------------------------------------- #
# Post-build annotation: length_ok + char_len on every output JSONL row
# --------------------------------------------------------------------------- #


def annotate_length_fields(jsonl_path: Path,
                           min_len: int, max_len: int) -> tuple[int, int]:
    """Add `length_ok` and `char_len` to every row of an output JSONL.

    Returns (total_rows, flagged_short). Schema is otherwise preserved.
    """
    if not jsonl_path.exists():
        return 0, 0
    with open(jsonl_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    n_short = 0
    for r in rows:
        n = len(r.get("prompt", ""))
        r["char_len"] = n
        r["length_ok"] = (min_len <= n <= max_len)
        if n < min_len:
            n_short += 1
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), n_short


# --------------------------------------------------------------------------- #
# Post-build bias report on the actual written dataset_all.jsonl
# --------------------------------------------------------------------------- #


def bias_report(dataset_all: Path) -> None:
    if not dataset_all.exists():
        print(f"[report] {dataset_all} not found.")
        return
    rows = [json.loads(line) for line in open(dataset_all, encoding="utf-8") if line.strip()]
    n = len(rows)
    atk = Counter(r["attack_type"] for r in rows)
    src = Counter(r.get("source_dataset", "?") for r in rows)
    safe = Counter(r["safety"] for r in rows)

    print(f"\n=== POST-MERGE COMPOSITION: {n} rows ===")
    print("safety: " + ", ".join(f"{k}={v} ({v/n*100:.1f}%)" for k, v in safe.items()))

    print("\n[attack_type]")
    for k, v in atk.most_common():
        bar = "#" * int(v / n * 50)
        print(f"  {k:30} {v:>6} {v/n*100:>6.2f}%  {bar}")

    print("\n[source]")
    for k, v in src.most_common():
        print(f"  {k:30} {v:>6} {v/n*100:>6.2f}%")

    print("\n[bias diagnostics]")
    top_src, top_src_n = src.most_common(1)[0]
    if top_src_n / n > 0.55:
        print(f"  *** SOURCE BIAS: {top_src} = {top_src_n/n*100:.1f}% (>55%) ***")
    for a in atk:
        a_rows = [r for r in rows if r["attack_type"] == a]
        if len(a_rows) < 50:
            continue
        s = Counter(r.get("source_dataset", "?") for r in a_rows)
        d_src, d_n = s.most_common(1)[0]
        if d_n / len(a_rows) > 0.90:
            print(f"  *** CLASS MONOCULTURE: '{a}' is {d_n/len(a_rows)*100:.0f}% "
                  f"from '{d_src}' ***")
    thin = [a for a, c in atk.items() if c < 50]
    if thin:
        print(f"  thin classes (<50 rows, document as limitation): "
              f"{ {a: atk[a] for a in thin} }")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-dir", type=str, default="./gap_fill_output/by_attack")
    ap.add_argument("--extra-file", nargs="*", default=None,
                    help="Extra gap JSONL files (e.g. Aegis output not under gap-dir).")
    ap.add_argument("--output-dir", type=str, default="./merged_output")
    ap.add_argument("--target-size", type=int, default=8000)
    ap.add_argument("--dataset-cap", type=float, default=0.55)
    ap.add_argument("--min-hard-neg", type=float, default=0.05)
    ap.add_argument("--max-rows-per-source", type=int, default=25000)
    ap.add_argument("--audit-tolerance", type=float, default=0.02)
    ap.add_argument("--warn-only", action="store_true",
                    help="Don't raise on audit failure (inspect first).")
    # Pre-split normalization flags (consistent with preprocess_dataset.py)
    ap.add_argument("--min-len", type=int, default=40,
                    help="Length-filter floor for the pool (default 40).")
    ap.add_argument("--max-len", type=int, default=2000,
                    help="Length-filter ceiling for the pool (default 2000).")
    ap.add_argument("--no-spacy", action="store_true",
                    help="Skip spaCy entirely (regex/ftfy normalization only).")
    ap.add_argument("--lemmatize", action="store_true",
                    help="Apply spaCy lemmatization (needs the full model).")
    ap.add_argument("--drop-short", action="store_true",
                    help="Hard-drop rows below --min-len (default: flag only).")
    args = ap.parse_args()

    gap_dir = Path(args.gap_dir)
    extra = [Path(f) for f in args.extra_file] if args.extra_file else None
    out_dir = Path(args.output_dir)

    # Wire min-hard-neg into the module constant the boost reads.
    B.MIN_HARD_NEGATIVE_RATIO = args.min_hard_neg

    # The same Preprocessor instance used for the pool is exposed to the
    # inference path via `preprocess_dataset.normalize_prompt`. We seed that
    # singleton here so callers can't accidentally pick up different settings.
    from preprocess_dataset import get_inference_normalizer
    prep = get_inference_normalizer(
        min_len=args.min_len,
        max_len=args.max_len,
        use_spacy=not args.no_spacy,
        lemmatize=args.lemmatize,
    )

    # Carry the normalization stats out of the patched loader for the summary.
    norm_state: dict = {}
    B.load_all_sources = make_patched_loader(
        gap_dir, extra, prep, args.drop_short, norm_state,
    )

    output_files = ("dataset_all.jsonl", "train.jsonl", "val.jsonl", "test.jsonl")
    build_error: Optional[BaseException] = None
    try:
        B.build(
            output_dir=out_dir,
            dataset_cap=args.dataset_cap,
            max_rows_per_source=args.max_rows_per_source,
            audit_tolerance=args.audit_tolerance,
            strict_audit=not args.warn_only,
            warn_only=args.warn_only,
            required_sources=[],
            target_size=args.target_size,
        )
    except RuntimeError as exc:
        build_error = exc
        print(f"\n[merge] build raised: {exc}")

    # Annotate every output JSONL with length_ok + char_len (even on audit fail
    # the files are already on disk from step 6 of build()).
    print("\n[annotate] adding length_ok / char_len to output JSONLs...")
    flagged_per_file: dict[str, tuple[int, int]] = {}
    for fname in output_files:
        path = out_dir / fname
        n_total, n_short = annotate_length_fields(path, args.min_len, args.max_len)
        flagged_per_file[fname] = (n_total, n_short)
        print(f"  {fname:24s} n={n_total:<6d} below_{args.min_len}c={n_short}")

    bias_report(out_dir / "dataset_all.jsonl")

    # ---- Pipeline summary ------------------------------------------------- #
    stats: Optional[CleanStats] = norm_state.get("stats")
    pool_before = norm_state.get("pool_before", 0)
    pool_after = norm_state.get("pool_after", 0)
    print("\n=== PIPELINE SUMMARY ===")
    print(f"  normalization order : preprocess -> dedup -> split -> audit "
          f"(corrected ordering)")
    print(f"  Preprocessor        : spaCy={'on' if not args.no_spacy else 'off'}"
          f", lemmatize={args.lemmatize}, "
          f"length-window=[{args.min_len}, {args.max_len}], "
          f"mode={'drop-short' if args.drop_short else 'flag-only'}")
    print(f"  pool before / after : {pool_before} -> {pool_after}")
    if stats is not None:
        print(f"  ftfy repaired       : {stats.ftfy_fixed}")
        print(f"  hidden chars strip  : {stats.hidden_stripped}")
        print(f"  empty after clean   : {stats.empty_after_clean}")
        print(f"  below {args.min_len:>3d}c          : {stats.too_short}  "
              f"{'(dropped)' if args.drop_short else '(flagged, kept)'}")
        print(f"  above {args.max_len:>3d}c          : {stats.too_long}  "
              f"{'(dropped)' if args.drop_short else '(flagged, kept)'}")
    dataset_total, dataset_short = flagged_per_file.get("dataset_all.jsonl", (0, 0))
    print(f"  dataset_all rows    : {dataset_total} "
          f"(below-floor flagged: {dataset_short})")
    print(f"  split + audit ran   : AFTER preprocessing, on the cleaned pool")
    if build_error is not None:
        print(f"  audit status        : FAILED (warn-only). See "
              f"{out_dir / 'quota_audit_report.json'}")
    print(f"\nDone. Canonical outputs in {out_dir}/ "
          f"(dataset_all/train/val/test.jsonl + quota_audit_report.json + "
          f"build_report.json). The *.clean.jsonl artefacts from the old "
          f"post-split pass are no longer produced — these splits ARE the "
          f"clean ones.")


if __name__ == "__main__":
    main()
