"""Scientific contract checks for the supplement; no training campaign at import."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_supplement_neural as s


class SupplementContractTests(unittest.TestCase):
    def test_grid_is_exactly_28_with_shared_base(self):
        configs, groups = s.build_grid()
        self.assertEqual(len(configs), 28)
        self.assertEqual(len(set(configs)), 28)
        self.assertEqual(configs[0], s.BASE)
        self.assertEqual([len(x) for x in groups.values()], [16, 9, 5])
        self.assertTrue(all(s.BASE.name in group for group in groups.values()))
        self.assertEqual(sum(c == s.BASE for c in configs), 1)
        self.assertEqual((len(configs) - 1) * 5, 135)

    def test_only_declared_hyperparameters_change(self):
        configs, _ = s.build_grid()
        for config in configs:
            self.assertEqual(config.model.topo_dim, config.graph.dim)
            self.assertEqual(config.model.filters, 64)
            self.assertEqual(config.model.seq_dim, 256)
            self.assertEqual(config.model.fusion, 'residual')
            self.assertEqual(config.graph.method, 'drw')
            self.assertEqual(config.graph.epochs, 5)
            self.assertEqual(config.graph.walks_per_node, 10)
            changed = {k for k, v in s.asdict(config).items() if v != s.asdict(s.BASE)[k]}
            self.assertTrue(changed <= {'dim', 'walk_length'} or changed <= {'kernel', 'max_length'}
                            or changed <= {'learning_rate'})

    def test_original_contract_still_intact(self):
        s.verify_contract(s.HERE / 'prepared', s.HERE / 'runs/reviewer_v1')

    def test_loader_never_reads_test_labels(self):
        original = np.load
        accesses = []

        class TestGuard:
            def __init__(self, archive): self.archive = archive
            def __enter__(self): return self
            def __exit__(self, *args): self.archive.close()
            def __getitem__(self, key):
                accesses.append(key)
                if key != 'pairs':
                    raise AssertionError('Attempt to read held-out test outcomes')
                return self.archive[key]

        def guarded(path, *args, **kwargs):
            archive = original(path, *args, **kwargs)
            return TestGuard(archive) if Path(path).name == 'test.npz' else archive

        with patch.object(s.np, 'load', side_effect=guarded):
            data = s.load_validation_fold(s.HERE / 'prepared', 0)
        self.assertEqual(accesses, ['pairs'])
        self.assertNotIn('test_labels', data)
        self.assertNotIn('test_pairs', data)
        self.assertEqual(len(data['val_labels']), len(data['val_pairs']))

    def test_stale_checkpoint_or_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'best.pt').write_bytes(b'original-checkpoint')
            content = {'run_signature': 'fixed', 'artifact_sha256': {'best.pt': s.sha(out / 'best.pt')},
                       'test_evaluated': False, 'evaluation_partition': 'validation'}
            (out / 'completed.json').write_text(json.dumps(content), encoding='utf-8')
            s.verify_completed(out, 'fixed')
            with self.assertRaises(ValueError): s.verify_completed(out, 'changed')
            (out / 'best.pt').write_bytes(b'modified-checkpoint')
            with self.assertRaises(ValueError): s.verify_completed(out, 'fixed')

    def test_all_metrics_include_ap_and_trapezoid_separately(self):
        result = s.metrics([0, 0, 1, 1], [.2, .8, .4, .9])
        self.assertTrue(set(s.METRICS) <= set(result))
        self.assertNotEqual(result['average_precision'], result['pr_trapezoid_area'])
        self.assertEqual(result['threshold'], .5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
