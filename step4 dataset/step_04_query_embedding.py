"""
step_04_query_embedding.py  —  METHODOLOGY STEP 4: Query Embedding
==================================================================

"...assign numerical vector values to the prompt that capture semantic meaning.
text-embedding-3-large ... uses Matryoshka embedding ... prioritize the most
critical features in the earliest dimensions."

FREE/LOCAL DEFAULT: BAAI/bge-m3 via sentence-transformers. It is Matryoshka-
capable (you can truncate the vector to a prefix length and renormalize), which
lets you keep the methodology's Matryoshka description truthfully. Runs locally,
no API key.

PAID SWAP (commented): OpenAI text-embedding-3-large with the `dimensions`
parameter for native Matryoshka truncation.

NOTE for the thesis: text-embedding-3-large is natively 3072-dim and supports a
`dimensions` parameter; the "64/128/512" figures in the methodology come from
the Matryoshka *paper*, not this specific model — describe Matryoshka as the
technique and cite the chosen dimension you actually use.

Install (local): pip install sentence-transformers
Run standalone:
    python step_04_query_embedding.py --prompt "how do I reset my password" --dim 512
"""

from __future__ import annotations

import argparse
import numpy as np

from pipeline_common import PipelineState

_MODEL = None
_MODEL_NAME = "BAAI/bge-m3"


def _get_model(name: str = _MODEL_NAME):
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e
        _MODEL = SentenceTransformer(name)
    return _MODEL


def embed(text: str, dim: int | None = None,
          model_name: str = _MODEL_NAME) -> np.ndarray:
    """Return a (possibly Matryoshka-truncated, renormalized) embedding."""
    model = _get_model(model_name)
    vec = model.encode([text], normalize_embeddings=True)[0]
    if dim is not None and dim < len(vec):
        vec = vec[:dim]                       # Matryoshka prefix truncation
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm                  # renormalize after truncation
    return vec.astype("float32")


# ---- PAID SWAP: OpenAI text-embedding-3-large -----------------------------
# import os
# from openai import OpenAI
# _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# def embed(text: str, dim: int | None = 512, model_name="text-embedding-3-large"):
#     resp = _client.embeddings.create(model=model_name, input=text,
#                                      dimensions=dim)   # native Matryoshka
#     return np.array(resp.data[0].embedding, dtype="float32")
# ---------------------------------------------------------------------------


def run(state: PipelineState, dim: int | None = 512) -> PipelineState:
    if state.blocked:
        return state
    vec = embed(state.prompt, dim=dim)
    state.meta["embedding"] = vec
    state.meta["embedding_dim"] = int(len(vec))
    state.log("step_04_query_embedding", dim=int(len(vec)), model=_MODEL_NAME)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4: Query Embedding")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--dim", type=int, default=512)
    args = ap.parse_args()
    st = PipelineState(prompt=args.prompt)
    st = run(st, dim=args.dim)
    v = st.meta["embedding"]
    print(f"dim={st.meta['embedding_dim']}  first8={np.round(v[:8],4).tolist()}")


if __name__ == "__main__":
    main()
