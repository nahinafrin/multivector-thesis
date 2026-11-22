# Routed generator (v2 / v3 switch): install into the thesis pipeline

I had read-only access to the thesis folder, so these files are staged here for you to copy in.

## Files
| file | what it is |
|---|---|
| `step_09c_routed_generator.py` | new Step 9 option: routes each query to v2 (fused) or v3 (verified isolate-and-vote) |
| `router.py` | the routing rule: do the retrieved passages disagree about the answer? |
| `verified_vote.py` | v3: fuzzy per-passage vote + closed-book Mistral-7B verifier (frozen, sha256 e4522924…) |
| `run_full_pipeline.py` | your `run_full_pipeline.py` with the new options added (a full copy) |
| `run_full_pipeline.diff` | exactly what changed in it (7 small edits) |

## Install
1. Back up `step4 dataset\run_full_pipeline.py`.
2. Copy all four `.py` files into `C:\Users\sarke\Desktop\Multivector methodology-local api extension\step4 dataset\`.
   The patched `run_full_pipeline.py` replaces the old one.
3. Make sure Ollama has `mistral:7b` (the verifier) and `llama3.2:3b` (the per-passage extractor). The thesis ensemble models are needed too, as before.

## Use
```
python run_full_pipeline.py --question "..." --generator routed                      # default: balanced
python run_full_pipeline.py --question "..." --generator routed --routed-mode safe
python run_full_pipeline.py --question "..." --generator verified                    # always v3
python run_full_pipeline.py --question "..."                                         # v2, unchanged default
```
Every output row gets a `generation.routed_generator` field. It records:
- the per-passage answers,
- the route taken (`v2`, `v3`, or `v3->v2`),
- v3's decision path and the verifier scores.

## How it routes
1. Every query first gets step_09a's per-passage short answers (one small LLM call per passage).
2. If the passages give 0 or 1 distinct answer, they agree. The query goes to **v2**: the normal fused answer.
3. If they give 2 or more distinct answers, they disagree. The query goes to **v3**, where the closed-book verifier picks the answer it can confirm or abstains.
4. When v3 abstains, the mode decides what happens:
   - **safe:** the answer is refused.
   - **balanced:** if the verifier endorsed two or more of the answers, nothing suggests a false planted value, so the v2 answer is given instead. Otherwise the answer is refused.

## Results (held-out test 2; the router was designed on held-out 1 only)
| mode | plain planted-fact attack success (50) | benign refused (40) |
|---|---|---|
| v2 only (current default) | 21/50 | 1/40 |
| **routed, balanced** | **9/50** | **1/40** |
| routed, safe | 5/50 | 6/40 |
| v3 only | 3/50 | 14/40 |
| Sentinel-Strategist | 11/50 | flags all 40 |
| ControlNET Llama / Mistral | blocks 17/50 / 16/50 | blocks 10/40 / 11/40 |

- Balanced mode keeps v2's refusal rate and lets through fewer than half as many planted-fact attacks as v2 (p = 0.002). Against Sentinel-Strategist the difference is not significant (9 vs 11, p = 0.79).
- Safe mode: 5 vs 11 against Sentinel-Strategist, p = 0.18.

**Blind spot:** a planted fact that is the only answer any passage gives causes no disagreement, so it goes through v2.

**Cost:** 5 extra small-model calls per query, plus up to a few verifier calls when the passages disagree.

**Not yet run end to end:** the test re-applied the router to per-passage answers stored from the pipeline. The integrated version adds Step 10 grounding after a v3 answer, as step_09a already does, which can only refuse more, never leak more. Run a short slice once after installing to confirm it works on your machine.
