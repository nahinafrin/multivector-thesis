"""
step_03c_fusion_gate.py  —  STEP 3 augmentation: BALANCED RISK FUSION (C3RF)
==============================================================================

WHY THIS EXISTS
---------------
Step-3 (injection detector alone)     : Precision 0.794, Recall 0.384, FP  21
Step-3b (injection OR Llama-Guard)    : Precision 0.509, Recall 0.910, FP 185

The OR-gate trades 526 % more false positives for the recall lift, because:
  * the injection detector reasons about *form / adversarial intent*
  * Llama-Guard reasons about *topical content vs. its safety taxonomy*
  * these are near-orthogonal failure surfaces, so logical OR unions BOTH
    false-positive sets.

This module replaces the OR with C3RF — Calibrated, Category- and Context-aware
Risk Fusion — that:
  1. keeps the SOFT injection risk score s_I from Step 3,
  2. parses Llama-Guard's verdict into a soft score s_L plus an MLCommons
     category c (S1..S14),
  3. weighs s_L by category severity w_c (CBRN/self-harm dominate; mild S6
     "specialized advice" gets a low weight so benign medical/legal Qs are
     not over-blocked),
  4. blends both with an agreement bonus and an uncertainty discount,
  5. uses two thresholds  (REVIEW, BLOCK)  so deployers can dial the
     precision/recall trade-off from a SINGLE knob,
  6. exposes hard-overrides for fast-path ALLOW and high-confidence BLOCK,
  7. (v2) adds MIRROR-IMAGE escalations for Llama-Guard side:
       * lg_hard_block   : high s_L AND high-severity cat -> BLOCK
       * lg_soft_escalate: any unsafe verdict on a mid+ severity cat -> REVIEW
     These recover the recall lost on cyberattack (S2), misinformation (S13),
     violent crimes (S1) and social engineering, without re-introducing the
     OR-gate's over-blocking on benign S6 medical / S8 IP prompts (gated by w_c).

Risk score:
    R = a*s_I + b*w_c*s_L + g*s_I*s_L + d*rho - e*u*(1-w_c)

Decision:
    R >= tau_block   -> BLOCK (red)
    R >= tau_review  -> REVIEW (amber; routed to LLM-judge / human queue)
    else             -> ALLOW (green)

Use as a drop-in for Step 3:
    python step_03c_fusion_gate.py --prompt "..."
    python step_03c_fusion_gate.py --eval ./merged_output/dataset_all.clean.jsonl --limit 500
    python step_03c_fusion_gate.py --eval ./merged_output/dataset_all.clean.jsonl --sweep

The --sweep mode runs a 2-D grid over (tau_review, tau_block) and reports the
operating point that maximises F1 while keeping FPs <= --fp-budget.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

from pipeline_common import PipelineState, read_jsonl
import step_03_injection_detection as s3   # reuse scanner cache + EvalCounts


# --------------------------------------------------------------------------- #
# Configuration: weights, category severity, thresholds                       #
# --------------------------------------------------------------------------- #
@dataclass
class FusionConfig:
    """Tunable knobs for C3RF. Defaults chosen for balanced operation."""

    # Risk-score weights. Re-balanced so strong solo-injection signals reach
    # the REVIEW band on their own, while solo Llama-Guard flags on low-severity
    # categories stay below ALLOW. Tunable via --sweep.
    alpha: float = 0.50        # injection contribution (was 0.40 -- raised so
                               #   s_I=0.7 alone -> R=0.35 -> REVIEW)
    beta: float = 0.40         # category-weighted safety contribution
    gamma: float = 0.15        # agreement bonus (both detectors agree)
    delta: float = 0.10        # contextual risk contribution
    eta: float = 0.08          # uncertainty discount on low-severity flags

    # Two-threshold decision
    tau_review: float = 0.35
    tau_block: float = 0.60

    # Hard overrides
    inj_hard_block: float = 0.90            # s_I above this -> BLOCK regardless
    inj_soft_escalate: float = 0.70         # s_I above this -> at least REVIEW
    severe_lg_block: float = 0.85           # s_L above this AND severe cat -> BLOCK

    # Llama-Guard mirror-image overrides (added to recover recall on mid-severity
    # categories like S2/S13 that were silently allowed by the additive score).
    # Both are gated by category weight so benign S6/S8 medical/IP prompts are
    # NOT affected.
    lg_hard_block_s: float = 0.85           # high s_L
    lg_hard_block_w: float = 0.80           # AND high-severity cat -> BLOCK
    lg_soft_escalate_s: float = 0.55        # any non-trivial unsafe verdict
    lg_soft_escalate_w: float = 0.65        # AND mid+ severity cat -> REVIEW

    fastpath_inj: float = 0.10              # below these -> ALLOW fast-path
    fastpath_lg: float = 0.30
    fastpath_rho: float = 0.20

    # Soft Llama-Guard score mapping (unsafe verdict -> s_L by category severity).
    # Raised in v2: Llama-Guard's *unsafe* verdict is a strong signal in itself;
    # the category code tells us WHICH kind of unsafe, but the verdict alone
    # carries high recall on its own categories, so we no longer dampen it
    # below the threshold-crossing zone.
    sl_unsafe_high: float = 0.95            # severe cat (CBRN, self-harm, CSE)
    sl_unsafe_med: float = 0.88             # mid-severe (violence, hate, weapons)
    sl_unsafe_low: float = 0.72             # low-severe / advice / IP / misinfo
    sl_unsafe_uncat: float = 0.80           # unsafe but no category parsed
    sl_safe: float = 0.05                   # tiny prior so the term is never 0


# Llama-Guard 3 category severities. CBRN, self-harm, CSE -> 1.0; mild advice low.
CATEGORY_SEVERITY: dict[str, float] = {
    "S1":  0.90,   # Violent crimes
    "S2":  0.70,   # Non-violent crimes
    "S3":  0.95,   # Sex-related crimes
    "S4":  1.00,   # Child sexual exploitation
    "S5":  0.50,   # Defamation
    "S6":  0.45,   # Specialized advice (medical/legal) — often benign overlap
    "S7":  0.70,   # Privacy
    "S8":  0.45,   # Intellectual property
    "S9":  1.00,   # Indiscriminate weapons (CBRN)
    "S10": 0.80,   # Hate
    "S11": 1.00,   # Suicide & self-harm
    "S12": 0.65,   # Sexual content
    "S13": 0.80,   # Elections / misinformation
    "S14": 0.85,   # Code interpreter abuse (Llama-Guard 3.1)
    "UNCATEGORIZED": 0.65,
    "NONE": 0.05,
}

SEVERE_CATEGORIES = {"S3", "S4", "S9", "S11"}        # never under-block these


# --------------------------------------------------------------------------- #
# Soft-score extraction                                                       #
# --------------------------------------------------------------------------- #
def _injection_score(prompt: str, threshold: float = 0.5,
                     use_toxicity: bool = True) -> tuple[float, dict]:
    """Return the SOFT injection risk score without forcing a block.

    Reuses Step-3's cached scanners but skips the .block() side-effect so the
    fusion layer sees the underlying probability.
    """
    scanners = s3._get_scanners(threshold=threshold, use_toxicity=use_toxicity)
    worst = 0.0
    detail: dict[str, dict] = {}
    for scanner in scanners:
        name = type(scanner).__name__
        _sanitized, is_valid, risk = scanner.scan(prompt)
        detail[name] = {"valid": bool(is_valid), "score": float(risk)}
        worst = max(worst, float(risk))
    return worst, detail


def _llamaguard_soft(prompt: str, cfg: FusionConfig,
                     model: str = "llama-guard3:1b",
                     base_url: str = "http://localhost:11434"
                     ) -> tuple[float, str, str]:
    """Run Llama-Guard and return (s_L_soft, category, raw_verdict).

    The Ollama llama-guard3 model emits two lines: "safe" or "unsafe", then a
    comma-separated list of violated S-codes when unsafe. We map that onto a
    soft score using FusionConfig's sl_* knobs so the fusion layer gets a
    real-valued signal rather than a hard 0/1.
    """
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)
        verdict = llm.invoke([{"role": "user", "content": prompt}]).content.strip()
    except Exception as e:
        return cfg.sl_safe, "NONE", f"unavailable ({e})"

    lines = [ln.strip() for ln in verdict.splitlines() if ln.strip()]
    if not lines:
        return cfg.sl_safe, "NONE", verdict

    first = lines[0].lower()
    if not first.startswith("unsafe"):
        return cfg.sl_safe, "NONE", verdict

    # parse categories (line 2, sometimes comma-separated)
    cats: list[str] = []
    if len(lines) >= 2:
        for tok in lines[1].replace(",", " ").split():
            tok = tok.strip().upper()
            if tok.startswith("S") and tok[1:].isdigit():
                cats.append(tok)
    if not cats:
        return cfg.sl_unsafe_uncat, "UNCATEGORIZED", verdict

    # pick the most severe category, drive s_L from its severity tier
    cats_sorted = sorted(cats, key=lambda c: -CATEGORY_SEVERITY.get(c, 0.0))
    top = cats_sorted[0]
    sev = CATEGORY_SEVERITY.get(top, 0.65)
    if sev >= 0.95:
        s_l = cfg.sl_unsafe_high
    elif sev >= 0.75:
        s_l = cfg.sl_unsafe_med
    else:
        s_l = cfg.sl_unsafe_low
    return s_l, top, verdict


# --------------------------------------------------------------------------- #
# Core fusion math                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class FusionResult:
    R: float
    decision: str                # "BLOCK" | "REVIEW" | "ALLOW"
    reason: str
    s_I: float
    s_L: float
    w_c: float
    category: str
    rho: float
    u: float
    override: Optional[str] = None
    audit: dict = field(default_factory=dict)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def fuse(s_I: float, s_L: float, category: str = "NONE", rho: float = 0.0,
         cfg: Optional[FusionConfig] = None) -> FusionResult:
    """Apply C3RF to a single prompt's signals.

    Parameters
    ----------
    s_I  : float in [0,1]  injection-detector risk score
    s_L  : float in [0,1]  soft Llama-Guard unsafe probability
    category : str         Llama-Guard category code ("S6", "S9", "NONE", ...)
    rho  : float in [0,1]  contextual risk (e.g. RAG context, agent-tool prompt)
    """
    cfg = cfg or FusionConfig()
    w_c = CATEGORY_SEVERITY.get(category.upper(), 0.05)
    u = 1.0 - abs(2.0 * s_L - 1.0)                       # uncertainty

    # ---------------- Hard BLOCK overrides (safety-first) -------------------- #
    if s_I >= cfg.inj_hard_block:
        return FusionResult(R=1.0, decision="BLOCK",
                            reason=f"injection_hard_override (s_I={s_I:.2f})",
                            s_I=s_I, s_L=s_L, w_c=w_c, category=category, rho=rho,
                            u=u, override="inj_hard_block")
    if category.upper() in SEVERE_CATEGORIES and s_L >= cfg.severe_lg_block:
        return FusionResult(R=1.0, decision="BLOCK",
                            reason=f"severe_category_override ({category}, s_L={s_L:.2f})",
                            s_I=s_I, s_L=s_L, w_c=w_c, category=category, rho=rho,
                            u=u, override="severe_lg_block")
    # Llama-Guard hard BLOCK for any high-severity category at high confidence
    # (covers S1 violent crimes, S10 hate, S13 elections/misinfo when Llama-Guard
    # is very confident -- these aren't in SEVERE_CATEGORIES but still warrant
    # auto-block when LG is sure).
    if s_L >= cfg.lg_hard_block_s and w_c >= cfg.lg_hard_block_w:
        return FusionResult(R=1.0, decision="BLOCK",
                            reason=f"lg_hard_block ({category}, s_L={s_L:.2f}, w_c={w_c:.2f})",
                            s_I=s_I, s_L=s_L, w_c=w_c, category=category, rho=rho,
                            u=u, override="lg_hard_block")
    # Fast-path ALLOW for prompts where both detectors are clearly clean
    if (s_I < cfg.fastpath_inj and s_L < cfg.fastpath_lg
            and rho < cfg.fastpath_rho):
        return FusionResult(R=0.0, decision="ALLOW",
                            reason="hard_allow_fastpath",
                            s_I=s_I, s_L=s_L, w_c=w_c, category=category, rho=rho,
                            u=u, override="fastpath_allow")

    # ---------------- Calibrated risk score ---------------------------------- #
    R = (cfg.alpha * s_I
         + cfg.beta  * w_c * s_L
         + cfg.gamma * s_I * s_L
         + cfg.delta * rho
         - cfg.eta   * u * (1.0 - w_c))
    R = _clamp(R)

    if R >= cfg.tau_block:
        decision = "BLOCK"
    elif R >= cfg.tau_review:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    reason = (f"R={R:.3f} | s_I={s_I:.2f} s_L={s_L:.2f} "
              f"cat={category}(w={w_c:.2f}) rho={rho:.2f} u={u:.2f}")
    override = None

    # ---------------- Soft escalation floors (recall safety net) ------------- #
    # (a) Strong injection alone -> at least REVIEW (high-precision detector).
    if decision == "ALLOW" and s_I >= cfg.inj_soft_escalate:
        decision = "REVIEW"
        override = "inj_soft_escalate"
        reason = f"inj_soft_escalate (s_I={s_I:.2f}) | " + reason

    # (b) Llama-Guard says unsafe on a mid+ severity category -> at least REVIEW.
    #     Mirror of (a). Gated on w_c so S6/S8 (benign medical/IP overlap) stay
    #     in the ALLOW band; only fires for S1/S2/S3/S7/S9/S10/S11/S13/S14.
    #     This is the missing piece that recovers recall on cyberattack (S2),
    #     misinformation (S13), violent crimes (S1), and social engineering.
    if decision == "ALLOW" and s_L >= cfg.lg_soft_escalate_s and w_c >= cfg.lg_soft_escalate_w:
        decision = "REVIEW"
        override = "lg_soft_escalate"
        reason = f"lg_soft_escalate ({category}, s_L={s_L:.2f}, w_c={w_c:.2f}) | " + reason

    return FusionResult(R=R, decision=decision, reason=reason,
                        s_I=s_I, s_L=s_L, w_c=w_c, category=category, rho=rho,
                        u=u, override=override)


# --------------------------------------------------------------------------- #
# Pipeline integration                                                        #
# --------------------------------------------------------------------------- #
def run(state: PipelineState, cfg: Optional[FusionConfig] = None,
        threshold: float = 0.5, use_toxicity: bool = True,
        treat_review_as_red: bool = False,
        base_url: str = "http://localhost:11434") -> PipelineState:
    """C3RF gate. Sets state.flag, populates scores+meta, optionally blocks."""
    if state.blocked:
        return state
    cfg = cfg or FusionConfig()

    s_I, inj_detail = _injection_score(state.prompt, threshold=threshold,
                                       use_toxicity=use_toxicity)
    s_L, cat, raw = _llamaguard_soft(state.prompt, cfg, base_url=base_url)
    rho = float(state.meta.get("context_risk", 0.0) or 0.0)

    result = fuse(s_I=s_I, s_L=s_L, category=cat, rho=rho, cfg=cfg)

    state.scores["injection_detection"] = s_I
    state.scores["llamaguard_soft"] = s_L
    state.scores["fusion_risk"] = result.R
    state.meta["llamaguard_verdict"] = raw
    state.meta["llamaguard_category"] = cat
    state.meta["fusion_decision"] = result.decision
    state.meta["fusion_override"] = result.override
    state.meta["fusion_reason"] = result.reason
    state.log("step_03c_fusion_gate",
              s_I=round(s_I, 4), s_L=round(s_L, 4), category=cat,
              w_c=round(result.w_c, 3), rho=round(rho, 3),
              R=round(result.R, 4), decision=result.decision,
              override=result.override, inj_detail=inj_detail)

    if result.decision == "BLOCK":
        state.block("step_03c_fusion_gate",
                    f"fused risk gate BLOCK | {result.reason}")
    elif result.decision == "REVIEW":
        state.flag = "amber"          # new tri-state: pipeline can route to judge
        if treat_review_as_red:
            state.block("step_03c_fusion_gate",
                        f"fused risk gate REVIEW promoted to red | {result.reason}")
    else:
        state.flag = "green"
    return state


# --------------------------------------------------------------------------- #
# Evaluation: head-to-head against injection-only and OR-gate                 #
# --------------------------------------------------------------------------- #
@dataclass
class _Sample:
    """Cached scores for one dataset row so we can sweep thresholds cheaply."""
    gt_unsafe: bool
    attack: str
    s_I: float
    s_L: float
    category: str
    rho: float
    inj_red: bool       # raw injection-detector verdict at default threshold
    lg_unsafe: bool     # raw Llama-Guard verdict


def _collect(path: str, limit: Optional[int], threshold: float,
             cfg: FusionConfig, base_url: str) -> list[_Sample]:
    samples: list[_Sample] = []
    for i, row in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        prompt = row.get("prompt", "")
        gt = str(row.get("safety", "")).lower() == "unsafe"
        s_I, inj_detail = _injection_score(prompt, threshold=threshold)
        s_L, cat, _raw = _llamaguard_soft(prompt, cfg, base_url=base_url)
        inj_red = any(not d["valid"] for d in inj_detail.values()) or s_I >= threshold
        lg_unsafe = s_L >= 0.5
        samples.append(_Sample(gt_unsafe=gt,
                               attack=row.get("attack_type", "unknown"),
                               s_I=s_I, s_L=s_L, category=cat,
                               rho=float(row.get("context_risk", 0.0) or 0.0),
                               inj_red=inj_red, lg_unsafe=lg_unsafe))
        if (i + 1) % 100 == 0:
            print(f"  ...scored {i+1}")
    return samples


def _score_at(samples: list[_Sample], cfg: FusionConfig,
              treat_review_as_red: bool) -> dict:
    inj = s3.EvalCounts()
    org = s3.EvalCounts()        # OR-gate
    fus = s3.EvalCounts()
    per_atk_fus: dict[str, s3.EvalCounts] = {}
    for s in samples:
        # injection-only
        inj.add(s.gt_unsafe, s.inj_red)
        # OR-gate
        org.add(s.gt_unsafe, s.inj_red or s.lg_unsafe)
        # fused
        r = fuse(s.s_I, s.s_L, s.category, s.rho, cfg)
        pred_red = (r.decision == "BLOCK") or (treat_review_as_red and r.decision == "REVIEW")
        fus.add(s.gt_unsafe, pred_red)
        per_atk_fus.setdefault(s.attack, s3.EvalCounts()).add(s.gt_unsafe, pred_red)
    return {
        "injection_only": inj.report(),
        "or_gate":        org.report(),
        "fusion":         fus.report(),
        "per_attack_fusion": {k: v.report() for k, v in sorted(per_atk_fus.items())},
    }


def evaluate(path: str, limit: Optional[int] = None, threshold: float = 0.5,
             cfg: Optional[FusionConfig] = None, base_url: str = "http://localhost:11434",
             treat_review_as_red: bool = False) -> dict:
    cfg = cfg or FusionConfig()
    samples = _collect(path, limit, threshold, cfg, base_url)
    return _score_at(samples, cfg, treat_review_as_red)


def sweep(path: str, limit: Optional[int], threshold: float, cfg: FusionConfig,
          base_url: str, fp_budget: int, treat_review_as_red: bool) -> dict:
    """Grid search over (tau_review, tau_block). Caches per-sample scores so
    the grid runs without re-querying Llama-Guard."""
    samples = _collect(path, limit, threshold, cfg, base_url)

    grid_review = [round(x * 0.05, 2) for x in range(4, 13)]   # 0.20..0.60
    grid_block  = [round(x * 0.05, 2) for x in range(8, 18)]   # 0.40..0.85
    best = None
    rows: list[dict] = []
    for tr in grid_review:
        for tb in grid_block:
            if tb <= tr:
                continue
            cfg_i = FusionConfig(**{**cfg.__dict__, "tau_review": tr, "tau_block": tb})
            res = _score_at(samples, cfg_i, treat_review_as_red)["fusion"]
            row = {"tau_review": tr, "tau_block": tb, **res}
            rows.append(row)
            ok_budget = res["fp"] <= fp_budget
            score = res["f1"] if ok_budget else res["f1"] - 0.5  # penalise budget-busters
            if best is None or score > best[0]:
                best = (score, row)
    return {"best": best[1] if best else None, "grid": rows}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _pretty_report(label: str, r: dict) -> None:
    print(f"{label:18} P={r['precision']:.3f}  R={r['recall']:.3f}  "
          f"F1={r['f1']:.3f}  FP={r['fp']:<4} FN={r['fn']:<4} "
          f"TP={r['tp']:<4} TN={r['tn']:<4} n={r['n']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3c: Balanced Risk-Fusion Gate (C3RF)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="classify a single prompt")
    g.add_argument("--eval", dest="eval_path", help="evaluate against a JSONL dataset")

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="injection-detector internal cutoff")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--tau-review", type=float, default=0.35)
    ap.add_argument("--tau-block",  type=float, default=0.60)
    ap.add_argument("--review-as-red", action="store_true",
                    help="treat REVIEW as a positive (blocked) prediction in eval")
    ap.add_argument("--sweep", action="store_true",
                    help="run a 2-D threshold sweep and pick the F1-optimal point")
    ap.add_argument("--fp-budget", type=int, default=60,
                    help="max false positives allowed during --sweep")
    args = ap.parse_args()

    cfg = FusionConfig(tau_review=args.tau_review, tau_block=args.tau_block)

    if args.prompt:
        st = PipelineState(prompt=args.prompt, raw_prompt=args.prompt)
        st = run(st, cfg=cfg, threshold=args.threshold, base_url=args.base_url)
        print(f"decision     : {st.meta.get('fusion_decision')}")
        print(f"flag         : {st.flag.upper()}")
        print(f"R (risk)     : {st.scores.get('fusion_risk', 0.0):.4f}")
        print(f"s_I / s_L    : {st.scores.get('injection_detection', 0.0):.3f} / "
              f"{st.scores.get('llamaguard_soft', 0.0):.3f}")
        print(f"category     : {st.meta.get('llamaguard_category')}")
        print(f"override     : {st.meta.get('fusion_override')}")
        print(f"reason       : {st.meta.get('fusion_reason')}")
        if st.blocked:
            print(f"blocked_reason: {st.block_reason}")
        return

    if args.sweep:
        out = sweep(args.eval_path, args.limit, args.threshold, cfg,
                    args.base_url, args.fp_budget, args.review_as_red)
        print("\n=== THRESHOLD SWEEP (fusion gate) ===")
        print(f"FP budget: {args.fp_budget}\n")
        # show top-10 by F1 within budget
        in_budget = [r for r in out["grid"] if r["fp"] <= args.fp_budget]
        in_budget.sort(key=lambda r: -r["f1"])
        print(f"{'tau_r':>6} {'tau_b':>6}  {'P':>6} {'R':>6} {'F1':>6}  {'FP':>4} {'FN':>4}")
        for r in in_budget[:10]:
            print(f"{r['tau_review']:>6.2f} {r['tau_block']:>6.2f}  "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}  "
                  f"{r['fp']:>4} {r['fn']:>4}")
        if out["best"]:
            b = out["best"]
            print("\n>>> BEST operating point within FP budget:")
            print(f"    tau_review={b['tau_review']}  tau_block={b['tau_block']}")
            print(f"    P={b['precision']:.3f}  R={b['recall']:.3f}  F1={b['f1']:.3f}  "
                  f"FP={b['fp']}  FN={b['fn']}")
        return

    res = evaluate(args.eval_path, limit=args.limit, threshold=args.threshold,
                   cfg=cfg, base_url=args.base_url,
                   treat_review_as_red=args.review_as_red)
    print("\n=== INPUT GATE COMPARISON ===")
    _pretty_report("injection-only:", res["injection_only"])
    _pretty_report("OR-gate       :", res["or_gate"])
    _pretty_report("C3RF (fused)  :", res["fusion"])
    print("\nPer-attack metrics (C3RF):")
    for atk, r in res["per_attack_fusion"].items():
        print(f"  {atk:30} n={r['n']:<5} "
              f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
