"""Regression checks for acceptance failures that must never be hidden by rounding."""
import unittest

from compare_service import compare, distribution, labeled_metrics, question
from service_acceptance import GATES


class ServiceAcceptanceTests(unittest.TestCase):
    def test_score_can_fail_while_every_probability_passes(self):
        p, n = [0.0]*255, [0.0]*255
        p[0] = 1.0
        n[0], n[-1] = 0.9995, 0.0005
        result = question({'type': 'score', 'keys': list(range(255))}, p, n, {'score': 0}, {'score': 0.127})
        self.assertEqual(result['failures'], ['score'])
        self.assertLess(result['probability_max_abs'], GATES['probability_max_abs'])

    def test_raw_score_failure_not_hidden_by_wire(self):
        result = question({'type': 'score', 'keys': ['0', '1']}, [1., 0.], [.89999, .10001], {'score': 0}, {'score': .1})
        self.assertIn('score', result['failures'])

    def test_wire_score_gate_is_independent(self):
        result = question({'type': 'score', 'keys': ['0', '1']}, [.5, .5], [.5, .5], {'score': .5}, {'score': .6001})
        self.assertEqual(result['failures'], ['score'])

    def test_probability_gate_not_rounded_away(self):
        result = question({'type': 'noul', 'keys': ['false', 'true']}, [.8, .2], [.78999, .21001], {'noul': .2}, {'noul': .21})
        self.assertEqual(result['failures'], ['probability'])

    def test_boundary_flip_retained(self):
        result = question({'type': 'noul', 'keys': ['false', 'true']}, [.501, .499], [.499, .501], {'noul': .499}, {'noul': .501})
        self.assertTrue(result['near_boundary'] and result['raw_flip'] and result['wire_flip'])
        self.assertFalse(result['failures'])

    def test_confident_flip_fails(self):
        result = question({'type': 'choice', 'keys': ['a', 'b']}, [.52, .48], [.49, .51], {'choice': 'a'}, {'choice': 'b'})
        self.assertIn('decision', result['failures'])

    def test_nonfinite_and_bad_distributions_rejected(self):
        for p in ([float('nan'), 0], [float('inf'), 0], [-.1, 1.1], [.6, .6], [1]):
            self.assertFalse(distribution(p, 2))

    def test_missing_duplicate_reordered_cases_rejected(self):
        corpus = {'cases': [{'name': 'a'}, {'name': 'b'}], 'gates': GATES}
        for rows in ([], [{'name': 'b'}, {'name': 'a'}], [{'name': 'a'}, {'name': 'a'}]):
            with self.assertRaisesRegex(ValueError, 'cases'):
                compare(corpus, rows, rows)

    def test_quality_uses_labels_and_normalizes_score_range(self):
        metrics = labeled_metrics('score', [.25, .5, .25], 2)
        self.assertEqual(metrics['accuracy'], 0)
        self.assertEqual(metrics['score_mae'], 1)
        self.assertEqual(metrics['normalized_score_mae'], .5)
        self.assertEqual(metrics['brier'], .875)


if __name__ == '__main__':
    unittest.main()
