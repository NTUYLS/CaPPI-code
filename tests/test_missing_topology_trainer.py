"""Bounded synthetic checks for label isolation, resume and selected evaluation."""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_missing_topology as runner
from cappi_data import _fold_signature, _json_hash


def fixture(prepared):
    prepared.mkdir()
    with (prepared / 'nodes.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['index', 'node_id', 'sequence'])
        writer.writerows((i, 100 + i, 'ACDEFGHIKLMNPQRSTVWY' * (i % 3 + 1)) for i in range(12))
    splits = {'train': ([(0, 1), (1, 2), (0, 3), (2, 3)], [1, 1, 0, 0]),
              'validation': ([(4, 5), (4, 6)], [1, 0]),
              'test': ([(7, 8), (7, 9)], [1, 0])}
    relative = 'random/protein/fold_0'
    folder = prepared / relative
    folder.mkdir(parents=True)
    for name, (pairs, labels) in splits.items():
        np.savez_compressed(folder / f'{name}.npz', pairs=np.asarray(pairs, dtype=np.int64),
                            labels=np.asarray(labels, dtype=np.float32))
    positives = np.asarray([(0, 1), (1, 2), (4, 5), (7, 8)], dtype=np.int64)
    np.savez_compressed(prepared / 'positives.npz', pairs=positives)
    np.savez_compressed(folder / 'graph.npz', graph_edges=positives[:2],
                        train_nodes=np.arange(4, dtype=np.int32), val_nodes=np.arange(4, 7, dtype=np.int32),
                        test_nodes=np.arange(7, 12, dtype=np.int32))
    files = {str(p.relative_to(prepared)).replace('\\', '/'): runner.sha(p)
             for p in prepared.rglob('*') if p.is_file()}
    manifest = {'schema_version': 'cappi-data-v1', 'seed': 20260930, 'n_folds': 5,
                'source_sha256': {}, 'n_nodes': 12, 'n_positives': len(positives), 'files': files,
                'folds': {relative: {'split_hash': _fold_signature(prepared, relative, files)}}}
    request = {k: manifest[k] for k in ('schema_version', 'seed', 'n_folds', 'source_sha256')}
    manifest['request_hash'] = _json_hash(request)
    manifest['dataset_hash'] = _json_hash({'request': request, 'files': files})
    runner.write_json(prepared / 'manifest.json', manifest)
    return prepared


def graph_stub(edges, n_nodes, config, seed, device, cache):
    present = np.zeros(n_nodes, dtype=bool)
    present[edges.reshape(-1)] = True
    raw = np.random.RandomState(seed).normal(size=(n_nodes, config.dim)).astype(np.float32)
    raw[~present] = 0
    return raw, present, runner.fingerprint(edges, n_nodes, config, seed)


class GuardNpz:
    def __init__(self, obj, is_test, accesses, deny=True):
        self.obj, self.is_test, self.accesses, self.deny = obj, is_test, accesses, deny

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.obj.close()

    def __getitem__(self, key):
        if self.is_test:
            self.accesses.append(key)
            if key == 'labels' and self.deny:
                raise AssertionError('Test labels accessed during training')
        return self.obj[key]

    def __getattr__(self, key):
        return getattr(self.obj, key)


class TrainerContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cappi_missing_topology_test_')
        self.root = Path(self.temp.name)
        self.prepared = fixture(self.root / 'prepared')
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.patches = [mock.patch.object(runner, 'CORE_SHA', runner.code_hash()),
                        mock.patch.object(runner, 'PREPARED_SHA', runner.sha(self.prepared / 'manifest.json')),
                        mock.patch.object(runner, 'train_topology', side_effect=graph_stub)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        torch.set_num_threads(self.old_threads)
        self.temp.cleanup()

    def run_one(self, out, family='topology_dropout_bypass', epochs=1):
        return runner.run_repair_fold(self.prepared, out, runner.RepairConfig(family=family),
            'protein', 'random', 0, 20260930, 20260930, device='cpu', epochs=epochs, patience=20)

    def test_family_settings_are_explicit_and_controlled(self):
        for family in runner.FAMILIES:
            cfg = runner.RepairConfig(family=family)
            self.assertEqual(asdict(cfg)['topology_dropout'], .5 if 'dropout' in family else 0.)
            self.assertEqual(cfg.missing_policy, 'bypass' if family.endswith('bypass') else 'original')
        for settings in ({'learning_rate': .005}, {'family': 'residual', 'topology_dropout': .5},
                         {'family': 'topology_dropout', 'topology_dropout': .1},
                         {'family': 'sequence_only', 'missing_policy': 'bypass'}):
            with self.assertRaises(ValueError):
                runner.RepairConfig(**settings)

    def test_loader_and_training_never_decompress_test_labels(self):
        real_load, accesses = np.load, []

        def guarded(path, *args, **kwargs):
            return GuardNpz(real_load(path, *args, **kwargs), Path(path).name == 'test.npz', accesses)

        out = self.root / 'label_isolation'
        with mock.patch.object(runner.np, 'load', side_effect=guarded):
            data = runner.load_validation_only(self.prepared, 'protein', 'random', 0)
            self.assertNotIn('test_labels', data)
            result = self.run_one(out)
        self.assertTrue(accesses)
        self.assertEqual(set(accesses), {'pairs'})
        self.assertFalse(result['test_evaluated'])
        self.assertFalse((out / 'test_predictions.csv').exists())
        with self.assertRaises(FileNotFoundError):
            runner.evaluate_selected_run(out, self.prepared, device='cpu')
        with self.assertRaises(ValueError):
            runner.run_repair_fold(self.prepared, out, runner.RepairConfig(), 'protein', 'random',
                                   0, 0, 0, device='cpu', epochs=1, no_test=False)

    def test_selected_evaluation_has_an_independent_completion_record(self):
        out = self.root / 'selected'
        result = self.run_one(out)
        completed_hash = runner.sha(out / 'completed.json')
        declaration = {'selected_before_test': True, 'run_signature': result['run_signature'],
                       'checkpoint_sha256': result['checkpoint_sha256'],
                       'selection_rule': 'Synthetic test configuration frozen before test evaluation'}
        runner.write_json(out / 'selection_frozen.json', declaration)
        test_result = runner.evaluate_selected_run(out, self.prepared, device='cpu')
        self.assertTrue(test_result['test_evaluated'])
        self.assertEqual(completed_hash, runner.sha(out / 'completed.json'))
        self.assertEqual(test_result, runner.evaluate_selected_run(out, self.prepared, device='cpu'))
        declaration['checkpoint_sha256'] = 'wrong'
        runner.write_json(out / 'selection_frozen.json', declaration)
        with self.assertRaises(ValueError):
            runner.evaluate_selected_run(out, self.prepared, device='cpu')

    def test_resume_restores_mask_and_global_rng_exactly(self):
        reference, resumed = self.root / 'reference', self.root / 'resumed'
        self.run_one(reference, epochs=6)
        real_save = runner.save_checkpoint

        def interrupt_after_five(value, path):
            real_save(value, path)
            if Path(path).name == 'last.pt' and value['epoch'] == 4:
                raise RuntimeError('synthetic interruption after saved checkpoint')

        with mock.patch.object(runner, 'save_checkpoint', side_effect=interrupt_after_five):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                self.run_one(resumed, epochs=6)
        self.run_one(resumed, epochs=6)
        a = runner._load(reference / 'last.pt')
        b = runner._load(resumed / 'last.pt')
        self.assertEqual(a['history'], b['history'])
        self.assertEqual(a['mask_records'], b['mask_records'])
        for name in a['model']:
            self.assertTrue(torch.equal(a['model'][name], b['model'][name]), name)
        self.assertTrue(torch.equal(a['mask_generator_state'], b['mask_generator_state']))
        self.assertTrue(torch.equal(a['rng_torch'], b['rng_torch']))
        with np.load(reference / 'mask_trace.npz') as x, np.load(resumed / 'mask_trace.npz') as y:
            self.assertTrue(np.array_equal(x['effective_present'], y['effective_present']))

    def test_mask_coverage_separates_labels_without_guiding_mask(self):
        base = np.asarray([True, True, True, False])
        effective = np.asarray([False, True, False, False])
        pairs = np.asarray([(0, 1), (1, 2), (0, 3), (2, 3)], dtype=np.int64)
        labels = np.asarray([1, 1, 0, 0], dtype=np.float32)
        result = runner.mask_coverage(base, effective, pairs, labels, 1, .5)
        self.assertEqual(result['by_label']['positive']['effective_0_present'], 0)
        self.assertEqual(result['by_label']['negative']['effective_0_present'], 2)
        self.assertEqual(result['artificially_masked_nodes'], 2)


if __name__ == '__main__':
    unittest.main()
