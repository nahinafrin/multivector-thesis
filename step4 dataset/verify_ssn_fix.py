import sys

sys.path.insert(0, ".")

from step_11_output_sanitization import mask_pii

cases = [
    "John Smith's social security number is 123-45-6789 and he lives at 42 Main Street.",
    "World War I began in 1914 and ended on 11-11-1918.",
]

for text in cases:
    masked, entities = mask_pii(text)
    print("IN :", text)
    print("OUT:", masked)
    print("ENT:", entities)
    print("-" * 60)