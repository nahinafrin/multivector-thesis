#!/usr/bin/env python3
"""
run_mitigation_ab.py  —  measure multi-vector ASR with mitigation OFF vs ON
===========================================================================

Runs the SAME multi-vector attack slice through the pipeline twice:
    off : detector runs, but NOTHING acts on it (attack allowed to land)
    on  : mitigation layers active (ablatable via CLI flags)

Generation is identical across arms, so ASR differences are attributable to
mitigation. Writes separate JSONL files per arm (and per run label) for scoring.

USAGE (from `step4 dataset`):
    python run_mitigation_ab.py --slice planted_inband.jsonl --run-dir mitigation_results/planted30
    python score_mitigation_ab.py --off mitigation_results/planted30/off.jsonl \
        --on mitigation_results/planted30/on_full.jsonl \
        --out mitigation_results/planted30/report_full.json
"""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path

from pipeline_common import read_jsonl
from mitigation_pipeline import process_with_mitigation, MitigationConfig
from detector_interface import get_detector

try:
    from run_full_pipeline import ensure_index
except Exception:
    def ensure_index(*a, **k):  # type: ignore
        pass


def _row_fields(row: dict) -> dict:
    """Normalize slice row fields for scoring and replay."""
    exp = row.get("expectation") or {}
    return {
        "question": row.get("question") or row.get("prompt") or "",
        "poison_chunk": row.get("poison_chunk"),
        "query_suffix": row.get("query_suffix"),
        "success_marker": row.get("success_marker") or exp.get("success_marker"),
        "true_answer": (row.get("true_answer") or row.get("ground_truth")
                        or exp.get("true_answer")),
    }


def _cfg_dict(cfg: MitigationConfig) -> dict:
    return {"enabled": cfg.enabled, "sanitize": cfg.sanitize,
            "tighten_retrieval": cfg.tighten_retrieval,
            "guarded_prompt": cfg.guarded_prompt,
            "grounding_gate": cfg.grounding_gate, "dlp": cfg.dlp,
            "refuse_on_detection": cfg.refuse_on_detection}


def build_on_config(args) -> MitigationConfig:
    if args.only_grounding:
        return MitigationConfig(enabled=True, sanitize=False, tighten_retrieval=False,
                                guarded_prompt=False, grounding_gate=True, dlp=False,
                                refuse_on_detection=False)
    return MitigationConfig(
        enabled=True,
        sanitize=not args.no_sanitize,
        tighten_retrieval=not args.no_tighten,
        guarded_prompt=not args.no_prompt,
        grounding_gate=not args.no_grounding,
        dlp=not args.no_dlp,
        refuse_on_detection=not args.no_refuse,
    )


def auto_label(args) -> str:
    if args.label:
        return args.label
    if args.only_grounding:
        return "groundonly"
    disabled = []
    if args.no_refuse:
        disabled.append("norefuse")
    if args.no_sanitize:
        disabled.append("nosanitize")
    if args.no_tighten:
        disabled.append("notighten")
    if args.no_prompt:
        disabled.append("noprompt")
    if args.no_grounding:
        disabled.append("nogrounding")
    if args.no_dlp:
        disabled.append("nodlp")
    if disabled == ["norefuse"]:
        return "norefuse"
    if not disabled:
        return "full"
    return "_".join(disabled)


def resolve_paths(args, label: str) -> tuple[Path, Path, Path | None]:
    """Return (off_path, on_path, manifest_path). Uses --run-dir when set."""
    if args.run_dir:
        base = Path(args.run_dir)
        base.mkdir(parents=True, exist_ok=True)
        off = base / "off.jsonl"
        on = base / f"on_{label}.jsonl"
        manifest = base / f"manifest_{label}.json"
        return off, on, manifest
    off = Path(args.out_off)
    on = Path(args.out_on)
    return off, on, None


def run_arm(slice_path, arm, out_path, detector, *, limit, base_url, on_cfg=None):
    cfg = (on_cfg or MitigationConfig()) if arm == "on" else MitigationConfig.off()
    ensure_index()
    n = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in read_jsonl(slice_path):
            if row.get("kind") not in (None, "multivector_attack"):
                continue
            if limit and n >= limit:
                break
            fields = _row_fields(row)
            q = fields["question"]
            if not q.strip():
                continue
            t0 = time.perf_counter()
            st = process_with_mitigation(
                q, mitigation=cfg, detector=detector,
                poison_chunk=fields["poison_chunk"],
                query_suffix=fields["query_suffix"],
                base_url=base_url)
            out = {
                "index": row.get("id", row.get("index", n)),
                "kind": row.get("kind", "multivector_attack"),
                "arm": arm,
                "question": q,
                "success_marker": fields["success_marker"],
                "true_answer": fields["true_answer"],
                "blocked": st.blocked,
                "block_reason": st.meta.get("block_reason"),
                "detector": st.meta.get("detector"),
                "mitigation_applied": st.meta.get("mitigation_applied", []),
                "mitigation_config": _cfg_dict(cfg) if arm == "on" else None,
                "final_response": st.output,
                "generation": {"answer": st.meta.get("answer", st.output)},
                "grounding": st.meta.get("grounding", {}),
                "latency_s": round(time.perf_counter() - t0, 4),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"[{arm}] wrote {n} multivector rows -> {out_path}")


def write_manifest(path: Path, *, slice_path, label, limit, on_cfg, off_path, on_path):
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slice": str(slice_path),
        "label": label,
        "limit": limit,
        "off_jsonl": str(off_path),
        "on_jsonl": str(on_path),
        "on_mitigation_config": _cfg_dict(on_cfg),
        "score_command": (
            f"python score_mitigation_ab.py --off {off_path} --on {on_path} "
            f"--out {on_path.parent / f'report_{label}.json'}"
        ),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Mitigation OFF vs ON on multi-vector attacks (ON arm ablatable)")
    ap.add_argument("--slice", required=True)
    ap.add_argument("--detector", default="existing")
    ap.add_argument("--run-dir", default=None,
                    help="directory for this experiment; writes off.jsonl + on_<label>.jsonl")
    ap.add_argument("--label", default=None,
                    help="run label (default: full, norefuse, groundonly, ...)")
    ap.add_argument("--out-off", default="mitigation_off.jsonl",
                    help="OFF arm path when --run-dir is not set")
    ap.add_argument("--out-on", default="mitigation_on.jsonl",
                    help="ON arm path when --run-dir is not set")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--skip-off", action="store_true",
                    help="reuse existing OFF arm; only (re)run the ON arm")
    ap.add_argument("--off-only", action="store_true",
                    help="run OFF arm only (for headroom check; skips ON)")
    ap.add_argument("--no-refuse", action="store_true",
                    help="ON arm without hard refuse-on-detection")
    ap.add_argument("--no-sanitize", action="store_true")
    ap.add_argument("--no-tighten", action="store_true")
    ap.add_argument("--no-prompt", action="store_true", help="no guarded prompt")
    ap.add_argument("--no-grounding", action="store_true")
    ap.add_argument("--no-dlp", action="store_true")
    ap.add_argument("--only-grounding", action="store_true",
                    help="ON arm = grounding gate only")
    args = ap.parse_args()

    label = auto_label(args)
    on_cfg = build_on_config(args)
    active = [k for k, v in _cfg_dict(on_cfg).items() if v and k != "enabled"]
    print(f"[config] label={label!r}  ON-arm active layers: {active or '(none)'}")

    off_path, on_path, manifest_path = resolve_paths(args, label)
    detector = get_detector(args.detector)

    if not args.skip_off:
        run_arm(args.slice, "off", off_path, detector,
                limit=args.limit, base_url=args.base_url)
    else:
        print(f"[off] skipped (reusing {off_path})")
        if not off_path.is_file():
            raise SystemExit(f"--skip-off but OFF arm missing: {off_path}")

    if args.off_only:
        print("[on] skipped (--off-only)")
        if manifest_path:
            write_manifest(manifest_path, slice_path=args.slice, label="off_only",
                           limit=args.limit, on_cfg=MitigationConfig.off(),
                           off_path=off_path, on_path=on_path)
        return

    run_arm(args.slice, "on", on_path, detector,
            limit=args.limit, base_url=args.base_url, on_cfg=on_cfg)

    if manifest_path:
        write_manifest(manifest_path, slice_path=args.slice, label=label,
                       limit=args.limit, on_cfg=on_cfg,
                       off_path=off_path, on_path=on_path)


if __name__ == "__main__":
    main()
