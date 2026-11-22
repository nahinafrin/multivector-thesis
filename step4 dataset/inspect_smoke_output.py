import json
from pathlib import Path

input_path = Path('external_validation_smoketest.jsonl')
output_path = Path('ext_smoketest_results.jsonl')

input_rows = [json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines() if line.strip()]
output_rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines() if line.strip()]

print(f'input rows: {len(input_rows)}  output rows: {len(output_rows)}')
print('input order (source, id):', [(r.get('source'), r.get('id')) for r in input_rows])
print('output (source, id):     ', [(r.get('source'), r.get('id')) for r in output_rows])

for r in output_rows:
    fr = r.get('final_response') or (r.get('generation') or {}).get('answer') or ''
    print(f"kind={r.get('kind')!r} blocked={r.get('blocked')!r} final_response_len={len(fr)} preview={fr[:120]!r}")
