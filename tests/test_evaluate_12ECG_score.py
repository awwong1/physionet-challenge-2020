import unittest

from util.evaluate_12ECG_score import evaluate_12ECG_score


class TestEvaluate12ECGScore(unittest.TestCase):
    @unittest.skip("ETEST record not supported")
    def test_evaluate_score(self):
        evaluation = evaluate_12ECG_score("tests/data", "tests/output")

        (
            auroc,
            auprc,
            accuracy,
            f_measure,
            f_beta_measure,
            g_beta_measure,
            challenge_metric,
        ) = evaluation

        self.assertEqual(f"{auroc:.3f}", "0.918")
        self.assertEqual(f"{auprc:.3f}", "0.741")
        self.assertEqual(f"{accuracy:.3f}", "0.393")
        self.assertEqual(f"{f_measure:.3f}", "0.649")
        self.assertEqual(f"{f_beta_measure:.3f}", "0.700")
        self.assertEqual(f"{g_beta_measure:.3f}", "0.569")
        self.assertEqual(f"{challenge_metric:.3f}", "0.682")
