"""Regression checks for the fixed acceptance gates; synthetic data, no inference."""
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('kev_validate', ROOT / 'recipe/kev/validate.py')
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.corpus = json.loads((ROOT / 'tests/kev/data/processing.json').read_text())['cases']
        cases = [{'name': c['name'], 'probabilities': c['probabilities'],
                  'body': {'usage': {'input_tokens': c['input_tokens']}}} for c in self.corpus]
        self.data = {kind: {'cases': copy.deepcopy(cases)} for kind in ('official', 'fp32-merged', 'merged', 'native')}

    def run_compare(self):
        for kind, value in self.data.items():
            value = copy.deepcopy(value)
            if kind == 'native':
                value['cases'] += [copy.deepcopy(value['cases'][8]) for _ in range(3)]
            (self.out / f'{kind}.json').write_text(json.dumps(value))
        with contextlib.redirect_stdout(io.StringIO()):
            return validator.compare(SimpleNamespace(out=self.out))

    def test_probability_gate_is_not_widened(self):
        self.data['native']['cases'][1]['probabilities'][0] = [1/3+0.0101, 2/3-0.0101]
        self.assertEqual(self.run_compare(), 1)
        report = json.loads((self.out / 'comparison.json').read_text())
        failures = report['comparisons']['official-to-native']['failures']
        self.assertTrue(any(f['gate'] == 'probability' for f in failures))

    def test_near_boundary_flips_are_reported_without_failing_margin_gate(self):
        for value in self.data.values():
            value['cases'][1]['probabilities'][0] = [0.499, 0.501]
            value['cases'][8]['probabilities'][1] = [0.501, 0.499]
        self.data['native']['cases'][1]['probabilities'][0] = [0.501, 0.499]
        self.data['native']['cases'][8]['probabilities'][1] = [0.499, 0.501]
        self.assertEqual(self.run_compare(), 0)
        report = json.loads((self.out / 'comparison.json').read_text())
        boundary = report['comparisons']['official-to-native']['near_boundary_disagreements']
        self.assertEqual({b['gate'] for b in boundary}, {'choice', 'noul'})

    def test_nonfinite_cannot_pass_comparison(self):
        self.data['native']['cases'][1]['probabilities'][0][0] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'invalid probabilities'):
            self.run_compare()

    def test_missing_case_cannot_pass_comparison(self):
        self.data['official']['cases'].pop()
        with self.assertRaisesRegex(AssertionError, 'incomplete corpus'):
            self.run_compare()


if __name__ == '__main__':
    unittest.main()
