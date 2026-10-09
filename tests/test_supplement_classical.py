"""Focused additive checks for scaler leakage, fixed estimators, and graph cache rejection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_supplement_classical as supplement
import numpy as np
from sklearn.pipeline import Pipeline


class SupplementClassicalContractTests(unittest.TestCase):
    def setUp(self):
        self.features = np.random.default_rng(15).normal(size=(40, 168)).astype(np.float32)
        self.labels = np.tile([0, 1], 20)

    def test_all_fixed_classifiers_fit_and_roundtrip_cpu_probabilities(self):
        for method in supplement.METHODS:
            with self.subTest(method=method):
                model = supplement.fixed_estimator(method, 19)
                model.fit(self.features, self.labels)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder)/'model.joblib'
                    supplement.joblib.dump(model, path)
                    loaded = supplement.joblib.load(path)
                    np.testing.assert_allclose(model.predict_proba(self.features), loaded.predict_proba(self.features), atol=1e-12)
                scores = loaded.predict_proba(self.features)[:,1]
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0)&(scores <= 1)).all())

    def test_knn_svm_scalers_never_fit_holdout(self):
        for method in ('aa20_drw_knn','aa20_drw_svm'):
            model = supplement.fixed_estimator(method, 17)
            self.assertIsInstance(model,Pipeline)
            model.fit(self.features,self.labels)
            scaler = model.named_steps['scaler']
            np.testing.assert_allclose(scaler.mean_, self.features.mean(axis=0,dtype=np.float64))
            original = scaler.mean_.copy()
            model.predict_proba(np.full((2,168),1000000.,dtype=np.float32))
            np.testing.assert_array_equal(scaler.mean_,original)
            self.assertEqual(scaler.n_samples_seen_,40)

    def test_fixed_svm_calibration_and_xgboost_cpu_no_holdout_selection(self):
        svm = supplement.fixed_estimator('aa20_drw_svm',20260930).named_steps['classifier']
        self.assertTrue(svm.probability)
        self.assertEqual((svm.C,svm.kernel,svm.gamma,svm.random_state),(1.,'rbf','scale',20260930))
        params = supplement.fixed_estimator('aa20_drw_xgboost',20260930).get_params()
        self.assertEqual((params['device'],params['n_jobs'],params['tree_method']),('cpu',2,'hist'))
        self.assertIsNone(params.get('early_stopping_rounds'))

    def test_missing_cache_is_error_and_never_trains(self):
        data = dict(graph_edges=np.array([[0,1]],dtype=np.int64),n_nodes=3)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError,'refusing to regenerate'):
                supplement.load_existing_topology(data,folder,20260930)
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_graph_cache_content_tamper_is_rejected(self):
        data = dict(graph_edges=np.array([[0,1]],dtype=np.int64),n_nodes=3)
        identity = supplement.fingerprint(data['graph_edges'],3,supplement.GraphConfig(),20260930)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/(identity+'.npz')
            path.write_bytes(b'changed')
            path.with_suffix('.json').write_text(json.dumps(dict(npz_sha256='not-the-file-hash',fingerprint=identity)))
            with self.assertRaisesRegex(ValueError,'content changed'):
                supplement.load_existing_topology(data,folder,20260930)

    def test_pair_features_preserve_168d_and_endpoint_symmetry(self):
        node = np.random.default_rng(9).normal(size=(7,84)).astype(np.float32)
        pairs = np.array([[0,3],[2,5]])
        features = supplement.base.symmetric_pair_features(node,pairs)
        self.assertEqual(features.shape,(2,168))
        np.testing.assert_array_equal(features,supplement.base.symmetric_pair_features(node,pairs[:,::-1]))

    def test_fixed_threshold_does_not_optimize_holdout(self):
        metrics = supplement.base.compute_metrics([0,1,0,1],[.1,.3,.2,.4],True)
        self.assertEqual(metrics['auroc'],1.)
        self.assertEqual(metrics['threshold'],.5)
        self.assertEqual(metrics['mcc'],0.)


if __name__=='__main__':
    unittest.main()
