#!/usr/bin/env python3
"""
run_ado_protocolB.py -- ADO (Sentinel-Strategist) on Protocol B, through an isolated adapter.

Compared configuration: the SEVEN-LAYER, SINGLE-GENERATOR configuration of our framework under
Protocol B (llama3.2:3b). Not the complete thirteen-stage framework. Nothing in our framework is
imported for inference; only its stored inputs are read.

Official ADO code (github.com/Pranavdec/Adaptive-RAG-Orchestrator, commit b6fa5ed, MIT) is imported
UNCHANGED from --ado-repo: Sentinel, Strategist, MetricsCollector, UserTrustManager, TrustRAG,
DP-RAG, and the generator prompt (generation.py). run_single's control flow and DefenseManager are
emulated step for step, as in ../run_ado_official_on_slice.py, because the real manager eagerly loads
the Hugging Face Llama needed only by attention-variance filtering. See IMPLEMENTATION_AUDIT.md.

Protocol B adaptations (all documented in the audit):
  * evidence: ADO's candidate pool is exactly the 8 passages the undefended llama3.2:3b generator
    received (contexts_{attack,benign}.jsonl from dump_undefended_contexts.py); no poison is inserted
  * trust: a fresh user per request (trust 0.5, empty history), as for every independent-request arm
  * generator: llama3.2:3b with ADO's own generator prompt, temperature 0.0
  * controller: llama3.1:8b (the paper's default Llama-3-8B is not installed)
  * evidence size (decision E2): ADO keeps all 8 passages unless one of its defenses removes some
    (ADO's top_k=5 cut by MiniLM would drop the planted chunk in 175/184 rows before any defense)
  * attention-variance filtering (decision B2): --av hf4bit runs the OFFICIAL AttentionFilteringDefense
    and the OFFICIAL hf_model.Llama.query on a pre-quantized 4-bit Llama-3.1-8B-Instruct
    (unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit), passed in as its supported `shared_model`;
    --av off records the request without executing it
  * defenses are pre-initialized from the official config.yaml entries and then updated from the
    Strategist plan with the official setter logic (DefenseManager._update_defense_params)
  * DP-RAG sampling is seeded per row (numpy/random seed = row index) for reproducibility
  * evaluation-only fields (success_marker, true_answer, labels) are attached after generation only

    python run_ado_protocolB.py --selection pilot_selection.json --split attack --out pilot_attack_results.jsonl
Resumable: rows already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
S4 = ROOT / "step4 dataset"
CTX = S4 / "mitigation_results" / "baselines_protocolB"
TOP_K = 5   # ADO top_k: used inside the defenses; no extra post-defense top-5 cut (decision E2)
DEFENSES = ("differential_privacy", "trustrag", "attention_filtering")


def load_av(repo: Path, model_id: str):
    """Official AttentionFilteringDefense + official hf_model.Llama, with the weights loaded in 4-bit.

    Only the weight loading differs from hf_model.Llama.__init__ (fp16, which needs ~16 GB): the object
    is the official class, so its query() -- generation with output_attentions and the per-passage
    attention aggregation AV relies on -- runs unchanged.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = repo / "src"
    # Register the packages without running their __init__ (core/__init__ imports the dataset loaders,
    # which need ir_datasets); hf_model and av_defense are then imported unchanged.
    for name, folder in (("ado_src", src), ("ado_src.core", src / "core"), ("ado_src.defenses", src / "defenses")):
        p = types.ModuleType(name); p.__path__ = [str(folder)]; sys.modules[name] = p
    hf = importlib.import_module("ado_src.core.hf_model")
    av_mod = importlib.import_module("ado_src.defenses.av_defense")
    llm = object.__new__(hf.Llama)
    hf.Model.__init__(llm, {"model_info": {"provider": "llama", "name": model_id},
                            "params": {"max_output_tokens": 512, "temperature": 0.0}}, "cuda")
    llm.max_output_tokens = 512
    llm.tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", attn_implementation="eager",
                                                     torch_dtype=torch.float16)
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.model.requires_grad_(False)
    # Memory-safe, numerically identical replacement for model.generate() (see av_generate.py and
    # test_av_equivalence.py: identical text and passage scores on the official path)
    sys.path.insert(0, str(HERE))
    from av_generate import GenerateProxy
    llm.model = GenerateProxy(llm.model)
    return av_mod.AttentionFilteringDefense, llm


def official_defense_config(repo: Path) -> dict:
    import yaml
    cfg = yaml.safe_load((repo / "config" / "config.yaml").read_text(encoding="utf-8"))
    return {d["name"]: dict(d) for d in cfg.get("defenses", [])}


def update_params(defense, settings: dict) -> None:
    """DefenseManager._update_defense_params, verbatim logic."""
    defense.config.update(settings)
    for param, value in settings.items():
        if param in ("enabled", "name"):
            continue
        setter = f"set_{param}"
        if hasattr(defense, setter):
            getattr(defense, setter)(value)
        elif hasattr(defense, param):
            setattr(defense, param, value)


def load_official(repo: Path):
    """Import the official modules unchanged (same shim as ../run_ado_official_on_slice.py)."""
    src = repo / "src"
    vendor = repo.parent / "vendor"                     # rouge-score installed with pip --target
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))

    def pkg(alias, folder, mods):
        p = types.ModuleType(alias); p.__path__ = [str(folder)]; sys.modules[alias] = p
        return {m: importlib.import_module(f"{alias}.{m}") for m in mods}

    core = pkg("ado_core", src / "core", ["ado", "sensing", "persistence"])
    defs = pkg("ado_def", src / "defenses", ["base", "dp_rag", "trustrag"])
    gen_src = (src / "core" / "generation.py").read_text(encoding="utf-8")
    gen = types.ModuleType("ado_core_gen")
    exec(compile(gen_src, str(src / "core" / "generation.py"), "exec"), gen.__dict__)
    return (core["ado"].Sentinel, core["ado"].Strategist, core["sensing"].MetricsCollector,
            core["persistence"].UserTrustManager, defs["trustrag"].TrustRAGDefense,
            defs["dp_rag"].DifferentialPrivacyDefense, gen.OllamaGenerator)


def eval_fields(split: str) -> dict:
    """Scoring-only fields, keyed by row index. Never passed to ADO."""
    if split == "attack":
        rows = [json.loads(l) for l in open(S4 / "planted_attacks_200.jsonl", encoding="utf-8") if l.strip()]
        return {r.get("id", i): {"kind": r.get("kind"), "success_marker": r.get("success_marker"),
                                 "true_answer": r.get("true_answer") or r.get("ground_truth")}
                for i, r in enumerate(rows)}
    off = [json.loads(l) for l in open(CTX / "benign_off.jsonl", encoding="utf-8") if l.strip()]
    return {r["index"]: {"kind": "benign_control", "true_answer": r.get("true_answer")} for r in off}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True, help="JSON with {'attack': [ids], 'benign': [ids]}")
    ap.add_argument("--split", choices=["attack", "benign"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ado-repo", default=os.environ.get(
        "ADO_REPO", r"C:\Users\sarke\Desktop\comparison model\paper_faithful\Adaptive-RAG-Orchestrator"))
    ap.add_argument("--controller", default="llama3.1:8b")
    ap.add_argument("--generator", default="llama3.2:3b")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--av", choices=["hf4bit", "off"], default="hf4bit")
    ap.add_argument("--av-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    args = ap.parse_args()

    out = HERE / args.out if not os.path.isabs(args.out) else Path(args.out)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True,
                        handlers=[logging.FileHandler(str(out).replace(".jsonl", ".log"), encoding="utf-8")])
    Sentinel, Strategist, MetricsCollector, UserTrustManager, TrustRAG, DPRAG, OllamaGenerator = \
        load_official(Path(args.ado_repo))

    ids = json.loads((HERE / args.selection).read_text(encoding="utf-8"))[args.split]
    ctx = {r["index"]: r for r in (json.loads(l) for l in open(CTX / f"contexts_{args.split}.jsonl", encoding="utf-8"))}
    ev = eval_fields(args.split)
    done = {json.loads(l)["index"] for l in open(out, encoding="utf-8")} if out.exists() else set()

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # ADO's embedder (config.yaml)
    sentinel = Sentinel(model_name=args.controller, provider="ollama", timeout_seconds=args.timeout)
    strategist = Strategist({"strategist_mode": "llm"}, model_name=args.controller, provider="ollama",
                            timeout_seconds=args.timeout)
    mc = MetricsCollector()
    trust = UserTrustManager(storage_dir=str(out.parent / f"ado_users_{out.stem}"))
    gen = OllamaGenerator(model_name=args.generator, temperature=0.0)
    run_info = {"controller": args.controller, "generator": args.generator, "embedder": "all-MiniLM-L6-v2",
                "ado_commit": "b6fa5eddd449a9bc325245bcecdad400f3dd72ee", "python": platform.python_version(),
                "av": args.av, "av_model": args.av_model if args.av == "hf4bit" else None,
                "evidence": "E2: all generator passages kept unless an ADO defense removes them"}
    dcfg = official_defense_config(Path(args.ado_repo))
    AV, av_llm = load_av(Path(args.ado_repo), args.av_model) if args.av == "hf4bit" else (None, None)

    with out.open("a", encoding="utf-8") as fh:
        for n, idx in enumerate(ids, 1):
            if idx in done:
                continue
            c = ctx[idx]
            q = c["question"]                                  # as every Protocol B arm received it
            user = f"{args.split}_{idx}"                       # fresh user per request: trust 0.5, no history
            t0 = time.time()
            rec = {"index": idx, "split": args.split, "question": q, **run_info}
            np.random.seed(idx); import random as _r; _r.seed(idx)   # DP-RAG samples with np.random
            try:
                uc = trust.get_user_context(user)
                ts, th = uc.global_trust_score, uc.trust_history or []
                qh, mh = uc.query_history or [], uc.metrics_history or []
                # ---- Pass 1 (pipeline.run_single order) ----
                pre = mc.calculate_pre_retrieval(q, history=qh[-5:])
                combined = {**pre, **(mh[-1].get("post_retrieval", {}) if mh else {})}
                rp = sentinel.analyze_pre_retrieval(query=q, trust_score=ts, metrics=combined,
                                                    history_window=qh[-5:], trust_history=th)
                trend = "DECLINING" if len(th) >= 2 and th[-1].get("delta", 0) < 0 else "STABLE"
                rp.specific_threats["_trust_score"] = ts
                rp.specific_threats["_trust_trend"] = trend
                plan_pre = strategist.generate_defense_plan(rp)
                trust.update_trust_score(user, rp.new_global_score_delta, reason=rp.reasoning_trace)
                # ---- Evidence: exactly the undefended generator's passages (no retrieval, no poison) ----
                passages = list(c["ranked_context"])
                qv = emb.encode([q], normalize_embeddings=True)[0]
                E = emb.encode(passages, normalize_embeddings=True)
                sims = E @ qv
                docs = [{"content": p, "distance": float(1 - s), "embedding": e.tolist()}
                        for p, s, e in zip(passages, sims, E)]
                docs.sort(key=lambda d: d["distance"])
                # ---- Pass 2 ----
                post = mc.calculate_retrieval([1 - d["distance"] for d in docs], [d["embedding"] for d in docs])
                pa = sentinel.analyze_post_retrieval(risk_profile=rp, post_metrics=post, trust_score=ts)
                pa.specific_threats["_trust_score"] = ts
                pa.specific_threats["_trust_trend"] = trend
                plan_post = strategist.generate_defense_plan(pa, stage="post_retrieval")
                tr_new = plan_post["trustrag"].get("enabled") and not plan_pre["trustrag"].get("enabled")
                av_new = plan_post["attention_filtering"].get("enabled") and not plan_pre["attention_filtering"].get("enabled")
                plan = plan_post if (tr_new or av_new) else plan_pre
                active = [d for d in DEFENSES if plan[d].get("enabled")]
                keep_k = len(passages)                         # decision E2 (audit section 7)
                executed, av_trace = [], None
                for name in active:                            # DefenseManager.apply_post_retrieval order
                    if name == "attention_filtering":
                        if AV is None:
                            continue                           # --av off: requested, not executed
                        d = AV({**dcfg.get(name, {"name": name}), "shared_model": av_llm})
                    else:
                        d = (DPRAG if name == "differential_privacy" else TrustRAG)(dict(dcfg.get(name, {"name": name})))
                    update_params(d, plan[name])
                    d.target_top_k = TOP_K                      # defenses use ADO top_k (5), as the manager sets it
                    n_before = len(docs)
                    av_err = []
                    if name == "attention_filtering" and docs:  # free the GPU: unload Ollama's resident models
                        import ollama as _ol                    # (they reload on their next call)
                        for _m in {args.controller, args.generator}:
                            try:
                                _ol.generate(model=_m, prompt="", keep_alive=0)
                            except Exception:
                                pass
                    if name == "attention_filtering":           # surface errors ADO's fail-safe would hide
                        _orig = d.filter_by_attention_score
                        def _traced(*a, _o=_orig, **k):
                            try:
                                return _o(*a, **k)
                            except Exception as e:
                                av_err.append(f"{type(e).__name__}: {str(e)[:200]}")
                                raise
                        d.filter_by_attention_score = _traced
                    docs = d.post_retrieval(docs, q)
                    if name == "attention_filtering":
                        import torch
                        torch.cuda.empty_cache()
                        av_trace = {"threshold": d.threshold, "top_tokens": d.top_tokens,
                                    "max_corruptions": d.max_corruptions, "n_in": n_before,
                                    "removed": n_before - len(docs), "error": av_err[0] if av_err else None}
                    executed.append(name)
                docs = sorted(docs, key=lambda d: d.get("distance", float("inf")))[:keep_k]
                contexts = [d["content"] for d in docs]
                answer = gen.generate(question=q, contexts=contexts)["answer"]
                trust.update_query_history(user, q, {"pre_retrieval": pre, "post_retrieval": post})
                rec.update({
                    "final_response": answer, "blocked": False, "abstained": False, "generation_error": None,
                    "pre_threat": rp.overall_threat_level, "post_threat": pa.overall_threat_level,
                    "pre_threats": {k: v for k, v in rp.specific_threats.items() if not k.startswith("_")},
                    "post_threats": {k: v for k, v in pa.specific_threats.items() if not k.startswith("_")},
                    "sentinel_fallback": rp.reasoning_trace == "System Failure Fallback",
                    "trust_before": ts, "trust_after": trust.get_user_context(user).global_trust_score,
                    "defenses_requested": active, "defenses_executed": executed,
                    "av_requested_not_executed": "attention_filtering" in active and AV is None,
                    "av_trace": av_trace,
                    "dp_epsilon": plan["differential_privacy"].get("epsilon"),
                    "trustrag_threshold": plan["trustrag"].get("similarity_threshold"),
                    "n_candidates": len(passages), "final_context": contexts,
                    "poison_in_candidates": bool(c.get("poison_chunk")) and c["poison_chunk"] in passages,
                    "poison_in_final_context": bool(c.get("poison_chunk")) and c["poison_chunk"] in contexts,
                })
            except Exception as e:                             # defined treatment: no answer delivered
                rec.update({"final_response": "", "blocked": False, "abstained": False,
                            "generation_error": f"{type(e).__name__}: {e}"})
            rec["latency_s"] = round(time.time() - t0, 2)
            rec.update(ev[idx])                                 # scoring-only fields, attached after the run
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[{args.split} {n}/{len(ids)}] idx={idx} pre={rec.get('pre_threat')} post={rec.get('post_threat')} "
                  f"defenses={rec.get('defenses_executed')} av_req={rec.get('av_requested_not_executed')} "
                  f"{rec['latency_s']}s err={rec.get('generation_error')}", flush=True)


if __name__ == "__main__":
    main()
