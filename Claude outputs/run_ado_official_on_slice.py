"""
run_ado_official_on_slice.py -- Sentinel-Strategist (ADO), OFFICIAL code, on the thesis slice.

Official implementation: github.com/Pranavdec/Adaptive-RAG-Orchestrator
(linked in the paper's Open Science appendix, arXiv 2604.20932v1). This script
imports the authors' own modules UNCHANGED:
    src/core/ado.py         Sentinel + Strategist (LLM agents, prompts, overrides)
    src/core/sensing.py     MetricsCollector (M_LEX, M_CMP, M_INT, M_DIS, M_DRP)
    src/core/persistence.py UserTrustManager (Global Trust Score, Eq. 3)
    src/defenses/trustrag.py, dp_rag.py   (Algorithm 1, DP-RAG)
    src/core/generation.py  OllamaGenerator (generator prompt)
and re-implements only ModularRAG.run_single's control flow (pipeline.py imports
ChromaDB/LangChain, which are not needed here), step for step.

Paper configuration kept: all-MiniLM-L6-v2 embedder, top_k=5, TrustRAG sim 0.88 /
ROUGE 0.25 / x3 candidates, DP-RAG eps 3 (dp_approx) as the registry default,
Strategist in 'llm' mode (repo config.yaml), trust initialised at 0.5, one user,
benign + adversarial rows interleaved in a seeded random order (paper §5.4).
Controller + generator: Llama-3.1-8B-Instruct via Ollama (`llama3.1:8b`), the
paper's generator model; the paper's default controller is Llama-3-8B.

Adaptation to the thesis slice (documented):
  * Knowledge base = thesis kb_wiki.texts (3,200 passages), embedded with MiniLM.
  * Poisoning follows PoisonedRAG's successful-retrieval assumption and the thesis
    injection: the slice's poison chunk is placed in the retrieved set at top-1
    similarity (its embedding is its real MiniLM embedding).
  * Attention-variance filtering needs Llama attentions via HF weights (not
    available through Ollama); its ACTIVATION is recorded but it is not executed.
Resumable: rows already in the output file are skipped.
"""
import argparse
import importlib
import json
import logging
import os
import random
import re
import sys
import time
import types
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "Adaptive-RAG-Orchestrator", "src")
sys.path.insert(0, os.path.join(HERE, "vendor"))          # rouge-score (pip --target)
THESIS = os.environ.get("THESIS_DIR", r"C:\Users\sarke\Desktop\Multivector methodology-local api extension")


def load_pkg(alias, folder, modules):
    pkg = types.ModuleType(alias)
    pkg.__path__ = [folder]
    sys.modules[alias] = pkg
    return {m: importlib.import_module(f"{alias}.{m}") for m in modules}


core = load_pkg("ado_core", os.path.join(REPO, "core"), ["ado", "sensing", "persistence"])
defs = load_pkg("ado_def", os.path.join(REPO, "defenses"), ["base", "dp_rag", "trustrag"])
Sentinel, Strategist = core["ado"].Sentinel, core["ado"].Strategist
MetricsCollector = core["sensing"].MetricsCollector
UserTrustManager = core["persistence"].UserTrustManager
TrustRAGDefense = defs["trustrag"].TrustRAGDefense
DPDefense = defs["dp_rag"].DifferentialPrivacyDefense

TOP_K = 5


class MiniLMStore:
    def __init__(self, texts):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.texts = texts
        self.E = self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    def query(self, q, top_k, poison=None):
        qv = self.model.encode([q], normalize_embeddings=True)[0]
        sims = self.E @ qv
        idx = np.argsort(-sims)[:top_k]
        docs = [{"content": self.texts[i], "distance": float(1 - sims[i]), "embedding": self.E[i].tolist()} for i in idx]
        if poison:
            pv = self.model.encode([poison], normalize_embeddings=True)[0]
            docs = docs[:-1]
            docs.insert(0, {"content": poison, "distance": docs[0]["distance"] - 1e-6,
                            "embedding": pv.tolist(), "is_poison": True})
        return docs


def marker_for(row):
    if row.get("success_marker"):
        return row["success_marker"]
    m = re.findall(r"'([^']{2,40})'", row.get("poison_chunk") or "")
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ado_official_slice.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows of the shuffled order")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per LLM call (CPU inference is slow)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fh = logging.FileHandler(args.out.replace(".jsonl", ".log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.ERROR)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=[fh, sh], force=True)

    rows = [json.loads(l) for l in open(os.path.join(THESIS, "step4 dataset", "adversarial_slice.jsonl"), encoding="utf-8")]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)          # interleaved benign + adversarial (paper §5.4)
    if args.limit:
        order = order[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}

    texts = [l.rstrip("\n") for l in open(os.path.join(THESIS, "step4 dataset", "kb_wiki.texts"), encoding="utf-8") if l.strip()]
    store = MiniLMStore(texts)
    cfg = {"strategist_mode": "llm"}
    sentinel = Sentinel(model_name=args.model, provider="ollama", timeout_seconds=args.timeout)
    strategist = Strategist(cfg, model_name=args.model, provider="ollama", timeout_seconds=args.timeout)
    metrics_collector = MetricsCollector()
    user_dir = os.path.join(os.path.dirname(args.out), "ado_users")
    trust = UserTrustManager(storage_dir=user_dir)
    user = f"slice_user_seed{args.seed}"
    from ado_core_gen import OllamaGenerator  # noqa: E402  (see shim below)
    gen = OllamaGenerator(model_name=args.model, temperature=0.0)

    with open(args.out, "a", encoding="utf-8") as fout:
        for pos, i in enumerate(order):
            row = rows[i]
            if row["id"] in done:
                continue
            t0 = time.time()
            q = row["question"]
            ctx = trust.get_user_context(user)
            trust_score, trust_hist = ctx.global_trust_score, ctx.trust_history or []
            q_hist, m_hist = ctx.query_history or [], ctx.metrics_history or []
            # --- Stage 1 (pipeline.py run_single, verbatim order) ---
            pre = metrics_collector.calculate_pre_retrieval(q, history=q_hist[-5:])
            prev_post = m_hist[-1].get("post_retrieval", {}) if m_hist else {}
            combined = {**pre, **prev_post}
            rp = sentinel.analyze_pre_retrieval(query=q, trust_score=trust_score, metrics=combined,
                                                history_window=q_hist[-5:], trust_history=trust_hist)
            trend = "DECLINING" if len(trust_hist) >= 2 and trust_hist[-1].get("delta", 0) < 0 else "STABLE"
            rp.specific_threats["_trust_score"] = trust_score
            rp.specific_threats["_trust_trend"] = trend
            plan_pre = strategist.generate_defense_plan(rp)
            trust.update_trust_score(user, rp.new_global_score_delta, reason=rp.reasoning_trace)
            # --- DefenseManager.set_dynamic_config(plan_pre) + apply_pre_retrieval (manager.py, emulated
            #     exactly; the real manager eagerly loads the HF Llama for AV, which is not available) ---
            active_pre = [n for n in ("differential_privacy", "trustrag", "attention_filtering") if plan_pre[n].get("enabled")]
            fetch_k = TOP_K
            if len(active_pre) == 1:
                fetch_k = TOP_K * plan_pre[active_pre[0]].get("candidate_multiplier", 1)
            elif len(active_pre) > 1:
                for j, n in enumerate(active_pre):
                    fetch_k = TOP_K * plan_pre[n].get("candidate_multiplier", 1) / (j + 1)
            fetch_k = int(fetch_k)
            poison = row.get("poison_chunk")
            docs = store.query(q, fetch_k, poison=poison)
            # --- Stage 2 (Sentinel post pass + Strategist post plan, as in run_single) ---
            scores = [1.0 - d["distance"] for d in docs]
            post = metrics_collector.calculate_retrieval(scores, [d["embedding"] for d in docs])
            pa = sentinel.analyze_post_retrieval(risk_profile=rp, post_metrics=post, trust_score=trust_score)
            pa.specific_threats["_trust_score"] = trust_score
            pa.specific_threats["_trust_trend"] = trend
            plan_post = strategist.generate_defense_plan(pa, stage="post_retrieval")
            tr_new = plan_post["trustrag"].get("enabled") and not plan_pre["trustrag"].get("enabled")
            av_new = plan_post["attention_filtering"].get("enabled") and not plan_pre["attention_filtering"].get("enabled")
            plan_eff = plan_post if (tr_new or av_new) else plan_pre
            active = [n for n in ("differential_privacy", "trustrag", "attention_filtering") if plan_eff[n].get("enabled")]
            av_skipped = False
            for n in active:                                   # manager.apply_post_retrieval
                if n == "differential_privacy":
                    d = DPDefense({"name": n, **plan_eff[n]})
                elif n == "trustrag":
                    d = TrustRAGDefense({"name": n, **plan_eff[n]})
                else:
                    av_skipped = True                          # needs HF Llama attentions
                    continue
                d.target_top_k = TOP_K
                docs = d.post_retrieval(docs, q)
            docs = sorted(docs, key=lambda d: d.get("distance", float("inf")))[:TOP_K]
            tr_on = "trustrag" in active
            av_on = "attention_filtering" in active
            contexts = [d["content"] for d in docs]
            poison_in_ctx = bool(poison) and poison in contexts
            answer = None
            marker = marker_for(row)
            if row["kind"] in ("poisoned_context", "multivector_attack"):
                answer = gen.generate(question=q, contexts=contexts)["answer"]
            trust.update_query_history(user, q, {"pre_retrieval": pre, "post_retrieval": post})
            rec = {
                "id": row["id"], "order": pos, "kind": row["kind"], "question": q,
                "pre_threat": rp.overall_threat_level, "pre_threats": {k: v for k, v in rp.specific_threats.items() if not k.startswith("_")},
                "post_threat": pa.overall_threat_level, "post_threats": {k: v for k, v in pa.specific_threats.items() if not k.startswith("_")},
                "trust_before": trust_score, "metrics_pre": pre, "metrics_post": post,
                "dp_enabled": "differential_privacy" in active,
                "dp_epsilon": plan_eff["differential_privacy"]["epsilon"], "fetch_k": fetch_k,
                "trustrag_enabled": tr_on, "trustrag_threshold": plan_eff["trustrag"]["similarity_threshold"],
                "av_enabled_not_executed": av_on,
                "n_context": len(contexts), "poison": poison, "poison_in_final_context": poison_in_ctx,
                "success_marker": marker, "answer": answer,
                "attack_succeeded": (bool(answer) and bool(marker) and marker.lower() in answer.lower()) if answer is not None else None,
                "sentinel_fallback": rp.reasoning_trace == "System Failure Fallback",
                "seconds": round(time.time() - t0, 1), "controller_model": args.model,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{pos+1}/{len(order)}] id={row['id']} {row['kind']:<19} pre={rec['pre_threat']:<9} "
                  f"post={rec['post_threat']:<9} TR={tr_on} DP={rec['dp_enabled']} poison_ctx={poison_in_ctx} "
                  f"{rec['seconds']}s", flush=True)


# --- shim: load OllamaGenerator from the official generation.py without importing hf_model ---
_gen_src = open(os.path.join(REPO, "core", "generation.py"), encoding="utf-8").read()
_gen_mod = types.ModuleType("ado_core_gen")
exec(compile(_gen_src, os.path.join(REPO, "core", "generation.py"), "exec"), _gen_mod.__dict__)
sys.modules["ado_core_gen"] = _gen_mod

if __name__ == "__main__":
    main()
