"""Recompute manuscript numbers that verify_claims.py does not check, from saved outputs. No inference."""
from common import MR, attack_succeeded, dump, load, paired, wilson

AB, FU, SEM = MR / "ablation200", MR / "followup200", MR / "followup_nonyesno40"


def asr(d):
    return {i: attack_succeeded(r) for i, r in d.items()}


def main():
    out = {}
    full = asr(load(AB / "on_full.jsonl"))
    ids = sorted(full)
    for name in ("combo_grounding_refuse", "combo_sanitize_refuse", "combo_sanitize_grounding"):
        sub = asr(load(FU / f"on_{name}.jsonl"))
        out[f"two-layer subset {name} vs full"] = paired(sub, full, ids)
    sem = {f.stem: asr(load(f)) for f in sorted(SEM.glob("*.jsonl"))}
    sids = sorted(sem["off"])
    out["semantic40_asr"] = {k: wilson(sum(v[i] for i in sids), len(sids)) + [sum(v[i] for i in sids)] for k, v in sem.items()}
    out["semantic40_full_vs_off"] = paired(sem["off"], sem["on_full"], sids)
    out["semantic40_subset3_vs_full"] = paired(sem["on_combo_sanitize_grounding_refuse"], sem["on_full"], sids)
    out["semantic40_subset3_vs_off"] = paired(sem["off"], sem["on_combo_sanitize_grounding_refuse"], sids)
    for k in ("on_only_prompt", "on_only_refuse", "on_only_sanitize", "on_only_grounding"):
        out[f"semantic40_off_vs_{k}"] = paired(sem["off"], sem[k], sids)
    for k in ("on_noprompt", "on_norefuse", "on_nogrounding"):
        out[f"semantic40_{k}_vs_full"] = paired(sem[k], sem["on_full"], sids)
    dump("verify_unchecked_claims.json", out)


if __name__ == "__main__":
    main()
