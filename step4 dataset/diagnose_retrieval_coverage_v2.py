"""
diagnose_retrieval_coverage_v2.py -- same as v1, plus a KNOWN-GOOD control
question pulled directly from grounded_controller.jsonl (the file that
produced the locked 130/130 result). If the control also returns 0 chunks,
the index isn't loading in this script's context at all -- an environment
problem, not a coverage gap. If the control retrieves normally but the
external questions still don't, kb_wiki genuinely has no coverage of these
topics, which is expected for a research corpus and not a bug.
"""
import step_01_user_input as s1
import step_02_normalization as s2
import step_04_query_embedding as s4
import step_05_vector_search as s5
from run_full_pipeline_adaptive import _apply_similarity_threshold
from adaptive_risk import AdaptivePolicy
from pipeline_common import PipelineState

questions = [
    ("CONTROL (known-good, from grounded_controller.jsonl)",
     "What is Canada's national unemployment rate?"),
    ("external", "how many episodes are in chicago fire season 4"),
    ("external", "who recorded i can't help falling in love with you"),
    ("external", "what county is cicero il"),
    ("external", "how many calories are in air popped popcorn"),
]

for label, q in questions:
    st = PipelineState(prompt=q, raw_prompt=q)
    st = s1.run(st)
    st = s2.run(st)
    policy = AdaptivePolicy(mode="adaptive")
    cfg = policy.assess(st)
    print(f"[{label}] Q: {q}")
    print(f"  tier={cfg.name}  top_k={cfg.top_k}  sim_threshold={cfg.sim_threshold}")
    st = s4.run(st)
    print(f"  query embedding produced: {st.meta.get('query_embedding') is not None if hasattr(st.meta, 'get') else 'n/a'}  "
          f"embedding_error={st.meta.get('embedding_error')}")
    st = s5.run(st, k=cfg.top_k)
    print(f"  chunks retrieved pre-threshold: {len(st.context)}")
    print(f"  similarity scores: {st.meta.get('retrieval_scores', [])}")
    print(f"  retrieval_error={st.meta.get('retrieval_error')}  index_path={st.meta.get('index_path')}")
    _apply_similarity_threshold(st, cfg.sim_threshold)
    print(f"  chunks surviving threshold: {len(st.context)}  "
          f"retrieval_starved={st.meta.get('retrieval_starved')}")
    for c in st.context[:3]:
        print(f"    {str(c)[:100]!r}")
    print()
