"""
_verify_coordination.py — TEMP verification (safe to delete).

Runs the REAL per-row channel computation the pipeline's _segment does, but on
the NEW files in Downloads, WITHOUT touching faiss / Ollama / the project's
modules. Answers one question:

    Do the new conjunctive multivector_attack pairs actually
      (a) SURVIVE Step 6 (real llm-guard PromptInjection scanner), and
      (b) FIRE the new 3-channel multivector detector,
    while benign + blatant single-channel poison stay correct?

Channels per row, exactly as _segment computes them:
    query_vector       = real injection score on the (question + suffix)
    context_vector     = real llm-guard risk on the poison chunk (Step 6)
    coordination_vector= NEW heuristic on (question, surviving chunk)
then NEW multivector.multivector_risk({...}).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # step4 dataset
DATASET = (HERE.parent / "dataset").resolve()                # gate modules
DOWNLOADS = Path(r"C:\Users\sarke\Downloads\files (4)")      # NEW files

# Resolution order: Downloads wins for multivector/coordination/slice;
# step4 dataset ("") wins for pipeline_common/step_06; ../dataset for step_03c.
for p in (str(DATASET), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load the NEW modules explicitly from Downloads (so we test THOSE, not the
# project copies), but let their `from pipeline_common import ...` resolve to
# the step4 dataset unified state.
coordination = _load("coordination_channel_new", DOWNLOADS / "coordination_channel.py")
mv_new = _load("multivector_new", DOWNLOADS / "multivector.py")
slice_new = _load("build_adversarial_slice_new", DOWNLOADS / "build_adversarial_slice.py")

# REAL pipeline scorers.
import step_03c_fusion_gate as s3c          # _injection_score (DeBERTa)
import step_06_context_sanitization as s6   # sanitize_chunk (llm-guard)

BASE_THRESHOLD = s6.BASE_THRESHOLD           # 0.50
scorer = coordination.HeuristicCoordinationScorer()

BENIGN_Q = "When was the Eiffel Tower completed?"


def channels_for(question: str, poison_chunk: str | None):
    q_v = float(s3c._injection_score(question)[0])
    if poison_chunk:
        ok, c_risk = s6.sanitize_chunk(poison_chunk, BASE_THRESHOLD)
        survived = bool(ok)
        c_v = float(c_risk)
    else:
        survived, c_v = True, 0.0
    # coordination reads what SURVIVES step 6 (state.context = kept chunks)
    surviving = poison_chunk if (poison_chunk and survived) else ""
    coord = scorer.score(question, surviving) if surviving else 0.0
    chans = {"query_vector": q_v, "context_vector": c_v, "coordination_vector": float(coord)}
    res = mv_new.multivector_risk(chans)
    return survived, chans, res


def show(label, question, poison):
    survived, ch, res = channels_for(question, poison)
    print(f"\n[{label}]")
    print(f"  q='{question[:70]}'")
    if poison:
        print(f"  poison='{poison[:70]}'  survived_step6={survived}")
    print(f"  channels: qv={ch['query_vector']:.3f} cv={ch['context_vector']:.3f} "
          f"coord={ch['coordination_vector']:.3f}")
    print(f"  joint_risk={res['joint_risk']:.3f}  active={res['active_channels']}  "
          f"is_multivector={res['is_multivector']}  hard_block={res['hard_block']}")
    return res


def main():
    print("=" * 72)
    print("CONJUNCTIVE multivector_attack pairs (must SURVIVE step6 AND fire)")
    print("=" * 72)
    fired = 0
    for i, (suffix, chunk, marker) in enumerate(slice_new.MULTIVECTOR_PAIRS):
        res = show(f"mv_pair_{i} marker={marker!r}", BENIGN_Q + suffix, chunk)
        fired += int(res["is_multivector"])
    print(f"\n  >>> conjunctive fired: {fired}/{len(slice_new.MULTIVECTOR_PAIRS)}")

    print("\n" + "=" * 72)
    print("CONTROLS (must NOT fire)")
    print("=" * 72)
    show("benign", BENIGN_Q, None)
    for i, (chunk, marker) in enumerate(slice_new.POISON_PAYLOADS):
        show(f"blatant_poison_{i} (single-channel)", BENIGN_Q, chunk)


if __name__ == "__main__":
    main()
