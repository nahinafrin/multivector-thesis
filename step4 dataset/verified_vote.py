"""
verified_vote.py -- Step 9a v3: isolate-and-vote with fuzzy vote matching and a
closed-book verification tie-break.  Drop-in replacement for step_09a.aggregate().

Why: strict isolate-and-vote (min_agreement=2, exact string match) stops planted-fact
poison but abstains on ~70% of benign questions, because (a) the true answer often
sits in only ONE retrieved passage and (b) paraphrased votes ("Henri Becquerel" vs
"Antoine Henri Becquerel") are counted as different answers.

Rule (frozen before any test run):
  1. Drop abstaining passages (INSUFFICIENT etc., step_09a's own markers).
  2. Cluster the remaining short answers with a FUZZY match: equal after step_09a
     normalisation, or one's content-token set is contained in the other's, or the
     same number.  yes / no are their own clusters.
  3. MAJORITY PATH (unchanged guarantee): if the largest cluster has >= 2 votes and
     > 50% of voting passages, deliver it.  A single planted passage can never
     reach this path on its own.
  4. VERIFY PATH (new): otherwise (single-source answers, ties, conflicts) ask a
     closed-book verifier LLM -- which never sees the retrieved passages, so the
     attacker cannot influence it -- "is <candidate> a correct answer to <question>?"
     for each candidate cluster, and read P(Yes) from its first-token distribution.
     Deliver the best candidate iff P(Yes) >= TAU and it beats the runner-up by
     >= DELTA; otherwise abstain.
The poisoned passage can only win through step 4, and only if a model that has
never read it independently believes the planted value.  The query itself (which
may carry a "trust the notes" suffix) is stripped of everything after the first
question mark before verification, so the attacker's query vector is also removed.
"""
from __future__ import annotations

import re

_ABSTAIN = re.compile(
    r"\b(insufficient|don'?t know|do not know|not (sure|certain|specified|stated|"
    r"mentioned|provided|available|in the context)|no information|cannot "
    r"(determine|tell|say)|unable to|unknown|unclear)\b", re.IGNORECASE)
_STRIP = re.compile(r"[^\w\s.\-]")
_STOP = {"the", "a", "an", "of", "is", "are", "was", "were", "in", "on", "at", "to", "and", "or",
         "by", "for", "with", "as", "his", "her", "its", "their", "he", "she", "it", "they", "from"}
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "twenty": "20", "hundred": "100"}


def is_abstention(a: str) -> bool:
    return not (a or "").strip() or bool(_ABSTAIN.search(a))


def toks(a: str) -> set[str]:
    t = _STRIP.sub(" ", (a or "").lower()).replace(".", " ")
    return {(_WORDNUM.get(w, w)) for w in t.split() if w and w not in _STOP}


def nums(a: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(a or "")} | {w for w in toks(a) if w in _WORDNUM.values()}


def yn(a: str):
    t = (a or "").strip().lower().rstrip(".!")
    return t if t in ("yes", "no") else None


def same(a: str, b: str) -> bool:
    ya, yb = yn(a), yn(b)
    if ya or yb:
        return ya == yb
    na, nb = nums(a), nums(b)
    if na or nb:                      # numbers must agree exactly (subset allowed: '1002 kg' ~ '1002 kg (2200 lb)')
        return bool(na) and bool(nb) and (na <= nb or nb <= na) and (toks(a) - na <= toks(b) or toks(b) - nb <= toks(a))
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


def cluster(answers: list[str]) -> list[list[str]]:
    cl: list[list[str]] = []
    for a in answers:
        for c in cl:
            if any(same(a, x) for x in c):
                c.append(a)
                break
        else:
            cl.append([a])
    return sorted(cl, key=len, reverse=True)   # stable: ties keep passage (rerank) order


def core_question(q: str) -> str:
    q = (q or "").strip()
    i = q.find("?")
    return q[: i + 1] if i >= 0 else q


VERIFY_PROMPT = (
    "You are checking a proposed answer using only your own general knowledge.\n"
    "Question: {q}\nProposed answer: {a}\n"
    "Is the proposed answer correct? Answer Yes or No.")

TAU, DELTA = 0.8, 0.3     # frozen 2026-09-25 on the dev slices (fresh + heldout, see dev_results_*); never retune on test


def aggregate(question: str, per_passage: list[str], scorer, tau: float = TAU, delta: float = DELTA) -> dict:
    """scorer(prompt) -> P(Yes) in [0,1] from a closed-book LLM.  Returns a step_09a-style decision."""
    voting = [a.strip() for a in per_passage if not is_abstention(a)]
    base = {"n_total": len(per_passage), "n_voting": len(voting), "verify": []}
    if not voting:
        return {**base, "abstained": True, "answer": None, "path": "none", "reason": "all passages abstained"}
    cl = cluster(voting)
    top = cl[0]
    if len(top) >= 2 and len(top) / len(voting) > 0.5:
        return {**base, "abstained": False, "answer": top[0], "path": "majority",
                "reason": f"fuzzy consensus {len(top)}/{len(voting)}", "clusters": cl}
    q = core_question(question)
    sc = [(c, float(scorer(VERIFY_PROMPT.format(q=q, a=c[0])))) for c in cl]
    base["verify"] = [(c[0], round(p, 4)) for c, p in sc]
    sc.sort(key=lambda x: -x[1])
    best, pb = sc[0]
    p2 = sc[1][1] if len(sc) > 1 else 0.0
    if pb >= tau and pb - p2 >= delta:
        return {**base, "abstained": False, "answer": best[0], "path": "verified",
                "reason": f"closed-book verifier P(yes)={pb:.2f}, runner-up {p2:.2f}", "clusters": cl}
    return {**base, "abstained": True, "answer": None, "path": "verify_abstain",
            "reason": f"no candidate verified (best P(yes)={pb:.2f}, runner-up {p2:.2f})", "clusters": cl}


def _pyes(top: dict) -> float:
    import math
    y = sum(math.exp(v) for k, v in top.items() if k.strip().lower().startswith("yes"))
    n = sum(math.exp(v) for k, v in top.items() if k.strip().lower().startswith("no"))
    return y / (y + n) if y + n > 0 else 0.5


def ollama_scorer(model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
    """Needs an Ollama server with logprobs support (>= 0.12).  Falls back to the greedy
    token (Yes -> 1.0, No -> 0.0) if logprobs are not returned."""
    import requests

    def f(prompt: str) -> float:
        r = requests.post(f"{host}/api/chat", json={
            "model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "logprobs": True, "top_logprobs": 20, "options": {"temperature": 0.0, "num_predict": 1}}, timeout=300).json()
        lp = r.get("logprobs") or []
        if lp and lp[0].get("top_logprobs"):
            return _pyes({t["token"]: t["logprob"] for t in lp[0]["top_logprobs"]})
        return 1.0 if (r.get("message", {}).get("content") or "").strip().lower().startswith("yes") else 0.0
    return f


def llamacpp_scorer(gguf: str):
    """Qwen-2.5 chat template; same prompt as ollama_scorer."""
    from llama_cpp import Llama
    m = Llama(model_path=gguf, n_ctx=512, n_threads=2, verbose=False, logits_all=True)

    def f(prompt: str) -> float:
        pr = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        r = m.create_completion(pr, max_tokens=1, logprobs=20, temperature=0.0)
        return _pyes(r["choices"][0]["logprobs"]["top_logprobs"][0])
    return f


def llamacpp_scorer_mistral(gguf: str, threads: int = 2):
    """Mistral-7B-Instruct chat template ([INST] ... [/INST]); same prompt as ollama_scorer."""
    from llama_cpp import Llama
    m = Llama(model_path=gguf, n_ctx=512, n_threads=threads, verbose=False, logits_all=True)

    def f(prompt: str) -> float:
        r = m.create_completion(f"[INST] {prompt} [/INST]", max_tokens=1, logprobs=20, temperature=0.0)
        return _pyes(r["choices"][0]["logprobs"]["top_logprobs"][0])
    return f
