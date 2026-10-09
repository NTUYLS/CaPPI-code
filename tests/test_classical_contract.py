"""Synthetic feature-control tests; no real inputs, GPU work, or real graph fitting."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_classical_baselines as controls  # noqa: E402
from cappi_data import load_fold, prepare_dataset  # noqa: E402


class ClassicalFeatureContractTests(unittest.TestCase):
    def test_aa20_normalization_unknowns_and_full_sequence(self):
        features = controls.amino_acid_composition(['AACX', 'a c\n', '', 'A' * 1500 + 'C'])
        self.assertEqual(features.shape, (4, 20))
        np.testing.assert_allclose(features[0, :2], [.5, .25])
        np.testing.assert_allclose(features[1, :2], [.5, .5])
        self.assertEqual(features[2].sum(), 0)
        self.assertGreater(features[3, 1], 0, 'Full-sequence composition must include the last C')

    def test_symmetric_pair_features_are_endpoint_order_invariant(self):
        nodes = np.asarray([[1., 2.], [3., 5.], [7., 11.]], dtype=np.float32)
        pairs = np.asarray([[0, 1], [1, 2]])
        features = controls.symmetric_pair_features(nodes, pairs)
        np.testing.assert_array_equal(features, controls.symmetric_pair_features(nodes, pairs[:, ::-1]))
        np.testing.assert_array_equal(features[0], [2., 3., 3., 10.])

    def test_heuristics_use_training_adjacency_and_give_unseen_nodes_zero(self):
        edges = np.asarray([[0, 1], [1, 2], [1, 3], [2, 3]])
        pairs = np.asarray([[0, 2], [2, 0], [0, 4], [4, 5]])
        np.testing.assert_array_equal(controls.graph_scores(edges, 6, pairs, 'degree_product'),
                                      [2., 2., 0., 0.])
        np.testing.assert_array_equal(controls.graph_scores(edges, 6, pairs, 'common_neighbor'),
                                      [1., 1., 0., 0.])
        self.assertNotIn((0, 2), set(map(tuple, edges)))

    def test_heuristic_metrics_have_no_fake_probabilities_or_threshold(self):
        result = controls.compute_metrics([0, 1, 0, 1], [0., 4., 1., 2.], False)
        self.assertEqual(result['auroc'], 1.)
        self.assertEqual(result['average_precision'], 1.)
        for key in ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1',
                    'mcc', 'threshold', 'pr_trapezoid_area'):
            self.assertIsNone(result[key])
        probability = controls.compute_metrics([0, 1, 0, 1], [.1, .8, .2, .9], True)
        self.assertEqual(probability['threshold'], .5)
        self.assertEqual(probability['accuracy'], 1.)

    def test_scaler_is_fit_only_on_training_pairs_and_hyperparameters_are_fixed(self):
        train = np.asarray([[0., 2.], [1., 3.], [2., 5.], [3., 8.]])
        labels = np.asarray([0, 0, 1, 1])
        model = controls.fixed_estimator('aa20_drw_lr', 17)
        model.fit(train, labels)
        original_mean = model.named_steps['scaler'].mean_.copy()
        model.predict_proba(np.asarray([[10000., -10000.], [-50000., 90000.]]))
        np.testing.assert_allclose(original_mean, train.mean(axis=0))
        np.testing.assert_array_equal(original_mean, model.named_steps['scaler'].mean_)
        self.assertEqual(model.named_steps['classifier'].C, 1.)
        forest = controls.fixed_estimator('aa20_drw_rf', 17)
        self.assertEqual((forest.n_estimators, forest.min_samples_leaf, forest.max_features,
                          forest.n_jobs, forest.random_state), (200, 2, 'sqrt', 4, 17))


class ClassicalPipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='cappi_classical_synthetic_')
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root / 'raw'
        cls.raw.mkdir()
        cls.ids = np.arange(80) * 31 + 103
        rng = np.random.default_rng(123)
        with (cls.raw / '肝癌 sequence.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['name', 'sequence'])
            for index, identifier in enumerate(cls.ids):
                sequence = ''.join(rng.choice(list(controls.AA20), size=25 + index))
                writer.writerow([int(identifier), sequence])
        with (cls.raw / '肝癌 interact.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['node1', 'node2'])
            for u in range(len(cls.ids)):
                for v in range(u + 1, len(cls.ids)):
                    if rng.random() < .2:
                        writer.writerow([int(cls.ids[u]), int(cls.ids[v])])
        cls.prepared = cls.root / 'prepared'
        prepare_dataset(cls.raw, cls.prepared, seed=23)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fake_topology(self, edges, n_nodes, config, seed, device, cache_dir):
        # Synthetic fixed vectors deliberately avoid running gensim or any GPU.
        self.assertEqual(device, 'cpu')
        self.assertEqual(config.dim, 64)
        present = np.zeros(n_nodes, dtype=bool)
        present[np.asarray(edges).reshape(-1)] = True
        embedding = np.random.default_rng(seed).normal(size=(n_nodes, 64)).astype(np.float32)
        embedding[~present] = 0
        return embedding, present, controls.fingerprint(edges, n_nodes, config, seed)

    def test_pair_and_protein_outputs_and_provenance(self):
        for protocol in ('pair', 'protein'):
            with self.subTest(protocol=protocol):
                output = self.root / ('controls_' + protocol)
                data = load_fold(self.prepared, 'random', protocol, 0)
                with patch.object(controls, 'train_topology', side_effect=self.fake_topology) as graph:
                    with redirect_stdout(io.StringIO()):
                        results = controls.run_fold(self.prepared, output, protocol, 0, seed=23)
                self.assertEqual(len(results), 4)
                self.assertEqual(graph.call_count, 1)
                np.testing.assert_array_equal(graph.call_args.args[0], data['graph_edges'])
                for result in results:
                    method = result['method']
                    folder = output / result['name']
                    with (folder / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), len(data['test_pairs']))
                    self.assertEqual({row['split_hash'] for row in rows}, {data['split_hash']})
                    for split in ('train', 'val', 'test'):
                        expected_hash = controls.array_hash(data[split + '_pairs'], data[split + '_labels'])
                        self.assertEqual(result['split_array_hashes'][split], expected_hash)
                        self.assertEqual({row[split + '_array_hash'] for row in rows}, {expected_hash})
                    expected_ids = {(int(data['node_ids'][u]), int(data['node_ids'][v]))
                                    for u, v in data['test_pairs']}
                    self.assertEqual({(int(row['protein_u']), int(row['protein_v'])) for row in rows}, expected_ids)
                    if method in controls.LEARNED:
                        scores = [float(row['probability']) for row in rows]
                        self.assertTrue(all(row['raw_score'] == '' for row in rows))
                        self.assertTrue(all(int(row['prediction']) == int(float(row['probability']) >= .5) for row in rows))
                        loaded = joblib.load(folder / 'model.joblib')
                        self.assertEqual(set(loaded.classes_), {0, 1})
                    else:
                        scores = [float(row['raw_score']) for row in rows]
                        self.assertTrue(all(row['probability'] == row['prediction'] == '' for row in rows))
                        if protocol == 'protein':
                            self.assertTrue(all(score == 0 for score in scores))
                    recomputed = controls.compute_metrics(data['test_labels'], scores, method in controls.LEARNED)
                    self.assertEqual(recomputed, result['test_metrics'])
                self.assertEqual(len(controls.summarize(output)), 4)
                # Completed controls are reused only after hashes are checked.
                with patch.object(controls, 'train_topology', side_effect=AssertionError('Unexpected recomputation')):
                    repeated = controls.run_fold(self.prepared, output, protocol, 0, seed=23)
                self.assertEqual(len(repeated), 4)
                first_path = output / results[0]['name'] / 'test_predictions.csv'
                first_path.write_text(first_path.read_text(encoding='utf-8-sig') + '\n', encoding='utf-8-sig')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    controls.run_fold(self.prepared, output, protocol, 0, seed=23)

    def test_injected_holdout_edge_is_rejected(self):
        data = load_fold(self.prepared, 'random', 'pair', 1)
        data['graph_edges'] = np.concatenate([data['graph_edges'], data['test_pairs'][:1]])
        with self.assertRaises(ValueError):
            controls.validate_graph_contract(data)

    def test_summary_rejects_incompatible_fold_campaigns(self):
        output = self.root / 'mismatched_summary'
        for fold, campaign in [(0, 'old'), (1, 'new')]:
            folder = output / ('fake{}'.format(fold))
            folder.mkdir(parents=True)
            result = {'protocol': 'pair', 'method': 'degree_product', 'fold': fold,
                      'campaign_signature': campaign, 'run_signature': 'synthetic',
                      'artifacts_sha256': {}, 'test_metrics': {}}
            (folder / 'completed.json').write_text(json.dumps(result), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            controls.summarize(output)


if __name__ == '__main__':
    unittest.main()
