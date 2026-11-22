import argparse
import json


def get_field(row, field):
    if field in row:
        return row[field]
    cur = row
    for part in field.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--field", required=True, help="exact key or dotted key holding the raw classifier probability")
ap.add_argument("--tau", type=float, default=0.60, help="single-channel block threshold")
ap.add_argument("--eps", type=float, default=0.10, help="lower edge of the 'quiet' sub-threshold band")
args = ap.parse_args()

vals = []
skipped = 0
with open(args.scores, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        val = get_field(row, args.field)
        if val is None:
            skipped += 1
            continue
        vals.append(float(val))

n = len(vals)
if n == 0:
    raise SystemExit(f"No values found under field '{args.field}' — re-check the field name against a real sample row.")

low = sum(1 for v in vals if v <= args.eps)
high = sum(1 for v in vals if v >= args.tau)
mid = n - low - high

print(f"n={n} (skipped {skipped} rows missing '{args.field}')")
print(f"  <= {args.eps}      : {low:5d}  ({100*low/n:5.1f}%)")
print(f"  ({args.eps}, {args.tau}) 'quiet band': {mid:5d}  ({100*mid/n:5.1f}%)")
print(f"  >= {args.tau}      : {high:5d}  ({100*high/n:5.1f}%)")
print()
if (low + high) / n >= 0.90:
    print(f"SUPPORTS the saturation premise: {100*(low+high)/n:.1f}% of mass sits outside the quiet band.")
else:
    print(f"DOES NOT cleanly support saturation as stated: only {100*(low+high)/n:.1f}% outside the quiet band — revise Proposition 1's premise or its wording before citing it as empirically confirmed.")
