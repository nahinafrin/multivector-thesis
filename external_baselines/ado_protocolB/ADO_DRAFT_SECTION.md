# Draft text for the paper: ADO comparison

**Now inserted** (with approval) into main.tex (Sec. V-E, Table tab:existing, latency table, abstract, summary table, conclusion) and main_iccit.tex (abstract, existing-defense paragraph, Table tab:existing, latency table, limitations, conclusion). The numbers there are checked by `step4 dataset/verify_claims.py` (claims X.ado_*).

Status: draft for the authors, using the **full Protocol B set (200 attacks + 200 benign)**. All
numbers can be traced to `full_comparison.json`, `full_sensitivity_refusal.json` and
`full_adjudication.json`.

## Draft paragraph

> **Comparison with an adaptive orchestrator (ADO).** We also evaluated ADO [cite arXiv 2604.20932], an
> agentic defense in which an LLM Sentinel scores each request and an LLM Strategist selects among
> DP-RAG, TrustRAG and attention-variance filtering. We used the authors' code (commit b6fa5ed),
> unmodified, through an adapter. The adapter gives ADO the same eight retrieved passages and the same
> llama3.2:3b generator as our Protocol B arms. Because of local hardware, the controller is
> Llama-3.1-8B and the attention filter uses a 4-bit Llama-3.1-8B, so we describe the setup as an
> *adapted implementation* rather than a reproduction. On the 200 attacks and 200 benign queries of
> Protocol B, ADO's attack success rate was 28.5% (95% CI 22.7–35.1), against 3.5% (1.7–7.0) for our
> seven-layer configuration (paired exact McNemar p = 3×10⁻¹²). Its benign accuracy was 35.5% vs.
> 52.0% (p = 6×10⁻⁶). Part of ADO's measured attack success is a scoring artifact. DP-RAG removed all
> passages in 64 of 200 attack queries, and the generator's refusals ("No relevant information …")
> contain the planted "No" marker. If refusals are not counted as successes for any arm, ADO's rate is
> 13.5% (9.4–18.9) vs. 3.5% (p = 2×10⁻⁴). The same behavior explains most of ADO's benign cost: it
> refuses or empties the context instead of blocking, with an answer rate of 83.5% (ours 91.5%,
> undefended 97.5%). This comparison concerns the seven-layer configuration under Protocol B, not the
> complete thirteen-stage framework.

## Draft table rows (same columns as the existing-defense table, n = 200 + 200)

The new row slots into Table VIII (main.tex) / Table VI (main_iccit.tex). The other rows are unchanged
from the existing table.

| Arm | ASR % | Benign blocked % | Benign correct % |
|---|---|---|---|
| Undefended | 29.5 | 0.0 | 56.5 |
| Spotlighting | 23.0 | 0.0 | 55.5 |
| LLM-Guard | 21.0 | 8.5 | 51.0 |
| **ADO (adapted)**‡ | **28.5** (13.5†) | **0.0** | **35.5** |
| ControlNET (generator context) | 4.5 | 13.5 | 46.0 |
| Ours, 3 layers | 5.5 | 9.5 | 52.0 |
| Ours, 7 layers | 3.5 | 8.5 | 52.0 |

† Refusals not counted as attack successes (S1). This is applied identically to all arms: undefended
29.0, ours (7 layers) 3.5. Spotlighting, LLM-Guard and ControlNET were not rescored with S1. If the
column is shown, rescore them first: `score_sensitivity_refusal.py` currently scores only ADO, ours
and undefended.

‡ Official code; controller Llama-3.1-8B; attention filter 4-bit Llama-3.1-8B; generator llama3.2:3b;
fresh user per request.

Paired tests to add to the paired-test sentence (ADO vs. ours, seven layers):
- attack success: 54 vs. 4 discordant, p = 3.2×10⁻¹²;
- S1: 24 vs. 4, p = 1.8×10⁻⁴;
- benign correct: 10 vs. 43, p = 5.6×10⁻⁶.

## Wording constraints for whoever edits the paper

- Say "adapted implementation", never "reproduction" or "paper-faithful".
- Compare with "our seven-layer configuration under Protocol B", not "our framework" or "the 13-stage pipeline".
- Do not claim ADO is the state of the art, and do not generalize beyond this setup (single generator,
  non-interleaved traffic, fresh user per request, so ADO's trust mechanism is inactive).
- Report the primary marker metric first. S1 was preregistered for 150 of the 200 attack rows and is
  post-hoc for the 50 pilot rows; say so if S1 is reported.
- ADO's latency (median 45 s, mean 114 s per query) is not comparable to the other arms' and should
  be reported only with its hardware setup (single 8 GB GPU, models swapped for each AV call).
