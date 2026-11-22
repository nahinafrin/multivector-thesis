from step_11_output_sanitization import _get_presidio

analyzer, _ = _get_presidio()
text = "John Smith's social security number is 123-45-6789 and he lives at 42 Main Street."
for entities in (None, ["US_SSN"], ["DATE_TIME"], ["US_SSN", "DATE_TIME"]):
    kwargs = {"text": text, "language": "en", "score_threshold": 0.0}
    if entities is not None:
        kwargs["entities"] = entities
    results = analyzer.analyze(**kwargs)
    print("entities=", entities, [(r.entity_type, r.start, r.end, r.score, text[r.start:r.end]) for r in results])
