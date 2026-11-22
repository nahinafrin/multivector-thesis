from pathlib import Path

for name in ("fusion_default_test.log", "fusion_candidate_test.log"):
    path = Path(name)
    print(f"=== {name} exists={path.exists()} size={path.stat().st_size if path.exists() else 0} ===")
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-16", errors="replace")
    matches = [
        line for line in text.splitlines()
        if any(token in line for token in ("INPUT GATE COMPARISON", "injection-only:", "OR-gate", "C3RF (fused)", '"precision"', '"recall"', '"f1"'))
    ]
    print("\n".join(matches[-30:]))
