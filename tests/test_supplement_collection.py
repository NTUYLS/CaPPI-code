import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from collect_supplement_results import calculate, verify_frame, precision_steps


class CollectionChecks(unittest.TestCase):
    def test_ties_use_inclusive_threshold_and_ranking_metrics(self):
        result = calculate([0, 0, 1, 1], [.1, .5, .5, .9])
        for key, expected in {'accuracy': .75, 'sensitivity': 1., 'specificity': .5,
                              'precision': 2/3, 'f1': .8, 'mcc': 1/np.sqrt(3),
                              'auroc': .875, 'average_precision': 5/6}.items():
            self.assertAlmostEqual(result[key], expected)

    def test_invalid_scores_rejected(self):
        for scores in ([np.nan, .2], [-.1, .2], [.5, 1.1]):
            with self.assertRaises(ValueError):
                calculate([0, 1], scores)

    def test_precision_plot_respects_average_precision_with_tied_scores(self):
        grid = np.linspace(0, 1, 101)
        steps = precision_steps([0, 0, 1, 1], [.1, .5, .5, .9], grid)
        area = np.sum(np.diff(grid) * steps[1:])
        self.assertAlmostEqual(area, 5/6)
        self.assertAlmostEqual(steps[75], 2/3)

    def test_one_class_rejected(self):
        with self.assertRaises(ValueError):
            calculate([1, 1], [.4, .5])

    def test_csv_roundtrip_keeps_distinct_probability_ranks(self):
        data = {'node_ids': np.array([10, 20, 30]), 'val_pairs': np.array([[0, 1], [0, 2]]),
                'val_labels': np.array([0, 1]), 'split_hash': 'known'}
        scores = [.16325, .16325000000000003]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            pd.DataFrame({'protein_u': [10, 10], 'protein_v': [20, 30],
                          'label': [0, 1], 'probability': scores}).to_csv(path, index=False)
            frame, result = verify_frame(path, data, 'val')
            np.testing.assert_array_equal(frame.probability.to_numpy(), scores)
            self.assertEqual(result['auroc'], 1.)
            self.assertEqual(result['average_precision'], 1.)

    def test_prediction_membership_is_checked_against_authoritative_ids(self):
        data = {'node_ids': np.array([10, 20, 30]), 'val_pairs': np.array([[0, 1], [0, 2]]),
                'val_labels': np.array([1, 0]), 'split_hash': 'known'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            frame = pd.DataFrame({'protein_u': [10, 10], 'protein_v': [20, 30],
                                  'label': [1, 0], 'probability': [.8, .2], 'prediction': [1, 0]})
            frame.to_csv(path, index=False)
            _, result = verify_frame(path, data, 'val')
            self.assertEqual(result['accuracy'], 1.)
            frame.loc[1, 'protein_v'] = 40
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                verify_frame(path, data, 'val')


if __name__ == '__main__':
    unittest.main()
