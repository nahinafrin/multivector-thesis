import unittest

from run_external_guard_baseline import score_rows


class KeywordScanner:
    def scan(self, text):
        flagged = "injection" in text.lower()
        return text, not flagged, 0.9 if flagged else 0.0


class ExternalGuardBaselineTests(unittest.TestCase):
    def test_scans_query_and_planted_chunk_and_reports_detector_metrics(self):
        rows = [
            {"id": 1, "kind": "adversarial_query",
             "question": "This is an injection request"},
            {"id": 2, "kind": "poisoned_context",
             "question": "Summarize this source",
             "poison_chunk": "An injection payload is planted here."},
            {"id": 3, "kind": "benign_control",
             "question": "What is in this source?"},
        ]
        report = score_rows(rows, KeywordScanner(), 0.5)
        self.assertEqual(report["metric_scope"],
                         "detector-only; not end-to-end attack success")
        self.assertEqual(report["overall"]["true_positive"], 2)
        self.assertEqual(report["overall"]["true_negative"], 1)
        self.assertEqual(report["overall"]["attack_recall"], 1.0)
        self.assertEqual(report["overall"]["benign_false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
