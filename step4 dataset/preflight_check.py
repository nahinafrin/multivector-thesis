#!/usr/bin/env python3
"""preflight_check.py — verify environment before a long mitigation run."""
from __future__ import annotations
import argparse
import importlib
import json
import sys
import urllib.request


def _bootstrap_path() -> None:
    from pathlib import Path
    import sys
    here = Path(__file__).resolve().parent
    dataset = (here.parent / "dataset").resolve()
    if str(dataset) not in sys.path:
        sys.path.insert(1, str(dataset))


def _check_import(module: str) -> tuple[bool, str]:
    try:
        if module == "mitigation_pipeline":
            _bootstrap_path()
        importlib.import_module(module)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_ollama(base_url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode())
        models = {m.get("name", "") for m in data.get("models", [])}
        need = "llama3.2:3b"
        if any(need in m for m in models):
            return True, f"ollama ok ({len(models)} models)"
        return False, f"ollama up but {need!r} not found in {sorted(models)[:5]}"
    except Exception as e:
        return False, f"ollama unreachable at {base_url}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight for mitigation A/B runs")
    ap.add_argument("--slice", default=None, help="optional attack slice JSONL")
    ap.add_argument("--qa", default="data/question-answer/test.jsonl")
    ap.add_argument("--base-url", default="http://localhost:11434")
    args = ap.parse_args()

    ok = True
    print("=== imports ===")
    for mod in ("graded_channels", "mitigation_pipeline", "detector_interface",
                "run_mitigation_ab", "score_mitigation_ab"):
        good, msg = _check_import(mod)
        print(f"  {'PASS' if good else 'FAIL'}  {mod}: {msg}")
        ok &= good

    print("\n=== ollama ===")
    good, msg = _check_ollama(args.base_url)
    print(f"  {'PASS' if good else 'FAIL'}  {msg}")
    ok &= good

    print("\n=== data ===")
    from pathlib import Path
    qa = Path(args.qa)
    if qa.is_file():
        print(f"  PASS  qa file: {qa}")
    else:
        print(f"  FAIL  qa file missing: {qa}")
        ok = False

    if args.slice:
        sl = Path(args.slice)
        if sl.is_file():
            n = sum(1 for _ in open(sl, encoding="utf-8") if _.strip())
            print(f"  PASS  slice: {sl} ({n} lines)")
        else:
            print(f"  FAIL  slice missing: {sl}")
            ok = False

    print("\n=== step hooks ===")
    import step_09_generator_llm as s9
    print(f"  {'PASS' if hasattr(s9, 'generate_single') else 'WARN'}  generate_single in step_09")
    import step_07_context_ranking as s7
    src = open(s7.__file__, encoding="utf-8").read()
    print(f"  {'PASS' if 'tier_rerank_min_score' in src else 'WARN'}  tier_rerank_min_score in step_07")
    import step_08_augmented_prompt as s8
    src8 = open(s8.__file__, encoding="utf-8").read()
    print(f"  {'PASS' if 'tier_prompt_profile' in src8 else 'WARN'}  tier_prompt_profile in step_08")

    if not ok:
        sys.exit(1)
    print("\nPreflight passed.")


if __name__ == "__main__":
    main()
