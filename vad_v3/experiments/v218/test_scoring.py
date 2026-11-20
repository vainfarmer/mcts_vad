import math
import unittest
from scoring import extract_score, windows


def response(candidates, answer='Yes'):
    rows = [dict(token=k, logprob=v) for k, v in candidates]
    return {'choices': [{'logprobs': {'content': [dict(token=answer, logprob=dict(candidates)[answer], top_logprobs=rows)]}}]}


class ScoreTests(unittest.TestCase):
    def test_same_position_normalization(self):
        result = extract_score(response([('Yes', math.log(.2)), ('No', math.log(.6)), ('Other', math.log(.2))], 'No'))
        self.assertAlmostEqual(result['score'], .25)
        self.assertEqual(result['hard_score'], 0)
        self.assertEqual(result['kind'], 'exact_pair')

    def test_missing_alternative_is_bounded_not_zero(self):
        result = extract_score(response([('Yes', -.00001), (' YES', -12), ('yes', -13), ('Y', -14), (' Yes', -15)]))
        self.assertEqual(result['kind'], 'top5_bounded_midpoint')
        self.assertLess(result['score'], 1)
        self.assertIsNone(result['no_logprob'])
        self.assertLess(result['upper'] - result['lower'], .001)

    def test_unreliable_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_score(response([('Yes', -.7), (' YES', -1), ('yes', -2), ('Y', -3), (' Yes', -4)]))
        with self.assertRaises(ValueError):
            extract_score({'choices': [{'logprobs': {'content': []}}]})

    def test_clip_coverage(self):
        rows = windows(12)
        self.assertEqual([(r['start'], r['end']) for r in rows], [(0, 5), (5, 10), (10, 12)])
        self.assertEqual(rows[-1]['center'], 11)
        self.assertTrue(all(0 <= r['context_start'] < r['context_end'] <= 12 for r in rows))


if __name__ == '__main__':
    unittest.main()
