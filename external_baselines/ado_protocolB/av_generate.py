"""
av_generate.py -- memory-safe drop-in for the model.generate() call inside ADO's hf_model.Llama.query.

ADO's AV filter calls
    model.generate(input_ids, attention_mask=..., max_new_tokens=512, output_attentions=True,
                   return_dict_in_generate=True, use_cache=True, do_sample=False)
and reads outputs.sequences and outputs.attentions[i] for generated token i. With output_attentions,
HF also returns the PREFILL attentions (every layer x head x prompt^2), ~5-13 GB for an 8-passage
prompt, which does not fit an 8 GB GPU. ADO never uses them: for i = 0 its code raises on that
prefill tensor and skips the token (logged "Error calculating attention for token 0").

GenerateProxy wraps the model and replaces only generate(): greedy decoding with a KV cache, the
prefill run WITHOUT attention outputs, and each subsequent decode step WITH attention outputs --
the same tensors HF generate returns for steps >= 1. attentions[0] is an empty tuple, so ADO's
loop takes the same exception-and-skip path for token 0 as with HF generate. Everything else in
query() (token filtering, layer/head averaging, passage mapping) runs unchanged.

Equivalence with the official path is tested in test_av_equivalence.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch


class GenerateProxy:
    def __init__(self, model):
        object.__setattr__(self, "_m", model)

    def __getattr__(self, name):
        return getattr(self._m, name)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=512, pad_token_id=None, **_):
        m = self._m
        eos = m.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        out = m(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_attentions=False)
        past, mask = out.past_key_values, attention_mask
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        seq, atts = [nxt], [()]                      # step 0 (prefill): no attentions, as ADO skips it
        for _step in range(1, max_new_tokens):
            if nxt.item() in eos:
                break
            mask = torch.cat([mask, torch.ones_like(nxt)], dim=1) if mask is not None else None
            out = m(input_ids=nxt, attention_mask=mask, past_key_values=past, use_cache=True,
                    output_attentions=True)
            past = out.past_key_values
            atts.append(tuple(a.detach() for a in out.attentions))
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            seq.append(nxt)
        return SimpleNamespace(sequences=torch.cat([input_ids] + seq, dim=1), attentions=tuple(atts))
