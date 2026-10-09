"""Runner provenance, resume and held-out evaluation checks on a tiny CPU fixture.

No manuscript data, graph embedding training or CUDA is used. The actual CaPPI
sequence encoder, pair classifier, AdamW, scheduler and checkpoints are exercised.
"""

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiments as runner  # noqa: E402


class SimulatedInterruption(RuntimeError):
    pass


class GuardedFold(dict):
    """Permit held-out metadata auditing but forbid early test-label consumption."""

    def __init__(self, source, events):
        super().__init__(source)
        self.events = events

    def __getitem__(self, key):
        if key == "test_labels":
            if "best_loaded" not in self.events:
                raise AssertionError("test labels consumed before checkpoint selection")
            self.events.append("test_labels")
        return super().__getitem__(key)


class RunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cappi_runner_contract_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {
            "n_nodes": 6,
            "node_ids": np.arange(100, 106, dtype=np.int64),
            "sequences": ["ACDX", "CEFG", "GHIK", "LMNP", "QRST", "VWYA"],
            "train_nodes": np.array([0, 1, 2, 3], dtype=np.int64),
            "val_nodes": np.array([1, 2, 3], dtype=np.int64),
            "test_nodes": np.array([3, 4, 5], dtype=np.int64),
            "train_pairs": np.array([[0, 1], [1, 2], [0, 2], [0, 3]], dtype=np.int64),
            "train_labels": np.array([1, 1, 0, 0], dtype=np.float32),
            "val_pairs": np.array([[1, 3], [2, 3]], dtype=np.int64),
            "val_labels": np.array([1, 0], dtype=np.float32),
            "test_pairs": np.array([[3, 4], [4, 5]], dtype=np.int64),
            "test_labels": np.array([1, 0], dtype=np.float32),
            "graph_edges": np.array([[0, 1], [1, 2]], dtype=np.int64),
            "split_hash": "toy-immutable-split-v1",
        }
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.loader = self.stack.enter_context(mock.patch.object(runner, "load_fold", return_value=self.data))
        self.graph = self.stack.enter_context(mock.patch.object(runner, "train_topology", side_effect=self.mock_graph))
        # Tests change no scientific code. Fix this identity to isolate the test
        # from another agent concurrently editing source comments or reporting.
        self.stack.enter_context(mock.patch.object(runner, "code_hash", return_value="toy-code-v1"))
        self.stdout = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.stdout))

    def mock_graph(self, edges, n_nodes, config, seed, device, cache_dir):
        np.testing.assert_array_equal(edges, self.data["graph_edges"])
        self.assertEqual(n_nodes, 6)
        self.assertEqual(device, "cpu")
        generator = np.random.default_rng(seed)
        embedding = generator.normal(size=(6, config.dim)).astype(np.float32)
        present = np.array([True, True, True, False, False, False])
        embedding[~present] = 0
        return embedding, present, "toy-train-graph-v1"

    def args(self, label="run", **overrides):
        values = dict(output=str(self.root / label), prepared=str(self.root / "prepared"),
                      max_length=8, graph_epochs=1, seed=3107, epochs=2, lr=0.001,
                      patience=20, weight_decay=0.0001, device="cpu", no_test=False,
                      resume=False, pair_chunk=2)
        values.update(overrides)
        return argparse.Namespace(**values)

    def run_fold(self, args):
        return runner.run_fold(args, "sequence_only", "random", "pair", "drw", 0)

    def fold_dir(self, args):
        return Path(args.output) / "pair__random__drw__sequence_only__fold0"

    def interrupt_before_final_evaluation(self, args):
        original = runner._load

        def stop_at_best(path, device="cpu"):
            if Path(path).name == "best.pt":
                raise SimulatedInterruption("stopped after training, before final evaluation")
            return original(path, device)

        with mock.patch.object(runner, "_load", side_effect=stop_at_best):
            with self.assertRaises(SimulatedInterruption):
                self.run_fold(args)
        self.assertTrue((self.fold_dir(args) / "last.pt").exists())
        self.assertFalse((self.fold_dir(args) / "completed.json").exists())

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        complete = self.args("uninterrupted", epochs=7)
        expected = self.run_fold(complete)
        interrupted = self.args("interrupted", epochs=7)
        original_save = runner.save_checkpoint

        def stop_after_fifth_checkpoint(checkpoint, path):
            original_save(checkpoint, path)
            if Path(path).name == "last.pt" and checkpoint["epoch"] == 4:
                raise SimulatedInterruption("checkpoint committed after epoch five")

        with mock.patch.object(runner, "save_checkpoint", side_effect=stop_after_fifth_checkpoint):
            with self.assertRaises(SimulatedInterruption):
                self.run_fold(interrupted)
        interrupted.resume = True
        actual = self.run_fold(interrupted)
        expected_ck = runner._load(self.fold_dir(complete) / "last.pt")
        actual_ck = runner._load(self.fold_dir(interrupted) / "last.pt")
        self.assertEqual(expected_ck["history"], actual_ck["history"])
        self.assertEqual(expected_ck["scheduler"], actual_ck["scheduler"])
        self.assertTrue(torch.equal(expected_ck["rng_torch"], actual_ck["rng_torch"]))
        for name, tensor in expected_ck["model"].items():
            torch.testing.assert_close(tensor, actual_ck["model"][name], atol=0, rtol=0,
                                       msg=f"resume changed {name}")
        for index, state in expected_ck["optimizer"]["state"].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, actual_ck["optimizer"]["state"][index][name], atol=0, rtol=0)
        self.assertEqual(expected["test_metrics"], actual["test_metrics"])
        self.assertEqual(expected["best_epoch"], actual["best_epoch"])
        self.assertEqual(actual["epochs_completed"], 7)

    def test_resume_after_early_stopping_performs_no_more_updates(self):
        args = self.args("early_stop", epochs=8, patience=2)
        original_loss = torch.nn.functional.binary_cross_entropy_with_logits

        def constant_validation_loss(logits, target, *positional, **kwargs):
            loss = original_loss(logits, target, *positional, **kwargs)
            if kwargs.get("reduction", "mean") == "mean":
                return loss.detach().new_tensor(0.7)
            return loss

        with mock.patch.object(torch.nn.functional, "binary_cross_entropy_with_logits", side_effect=constant_validation_loss):
            self.interrupt_before_final_evaluation(args)
        before = runner._load(self.fold_dir(args) / "last.pt")
        self.assertEqual(before["stale"], 2)
        self.assertEqual(before["epoch"], 2)
        args.resume = True
        with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("early-stopped run updated again")):
            result = self.run_fold(args)
        after = runner._load(self.fold_dir(args) / "last.pt")
        self.assertEqual(before["history"], after["history"])
        self.assertEqual(result["epochs_completed"], 3)
        self.assertEqual(result["best_epoch"], 1)
        self.assertTrue(result["test_evaluated"])

    def test_resume_rejects_checkpoint_signature_and_graph_mismatch(self):
        args = self.args("checkpoint_identity", epochs=1)
        self.interrupt_before_final_evaluation(args)
        last = self.fold_dir(args) / "last.pt"
        pristine = runner._load(last)
        args.resume = True
        for field in ("run_signature", "graph_hash"):
            with self.subTest(field=field):
                changed = copy.deepcopy(pristine)
                changed[field] = "incorrect-provenance"
                runner.save_checkpoint(changed, last)
                with self.assertRaisesRegex(ValueError, "Checkpoint provenance mismatch"):
                    self.run_fold(args)

    def test_resume_rejects_incompatible_best_checkpoint(self):
        args = self.args("best_identity", epochs=1)
        self.interrupt_before_final_evaluation(args)
        best = self.fold_dir(args) / "best.pt"
        changed = runner._load(best)
        changed["run_signature"] = "wrong-best-run"
        runner.save_checkpoint(changed, best)
        args.resume = True
        with self.assertRaisesRegex(ValueError, "Best checkpoint provenance mismatch"):
            self.run_fold(args)

    def test_resume_rejects_changed_pair_chunk(self):
        args = self.args("chunk_identity", epochs=1)
        self.interrupt_before_final_evaluation(args)
        args.resume = True
        args.pair_chunk = 4
        with self.assertRaisesRegex(ValueError, "Cannot resume incompatible run"):
            self.run_fold(args)

    def test_completed_artifacts_are_verified_before_skipping(self):
        args = self.args("completed", epochs=1)
        expected = self.run_fold(args)
        graph_calls = self.graph.call_count
        # JSON persistence legitimately normalizes tuple-valued configuration.
        self.assertEqual(json.loads(json.dumps(expected)), self.run_fold(args))
        self.assertEqual(graph_calls, self.graph.call_count)
        best = self.fold_dir(args) / "best.pt"
        pristine = best.read_bytes()
        best.write_bytes(pristine + b"changed-checkpoint")
        with self.assertRaisesRegex(ValueError, "Completed checkpoint content changed"):
            self.run_fold(args)
        best.write_bytes(pristine)
        predictions = self.fold_dir(args) / "test_predictions.csv"
        with predictions.open("a", encoding="utf-8") as handle:
            handle.write("tampered-prediction\n")
        with self.assertRaisesRegex(ValueError, "Completed predictions content changed"):
            self.run_fold(args)

    def test_test_labels_and_scores_are_used_only_after_best_checkpoint_load(self):
        original_load = runner._load
        original_score = runner.CaPPI.score_pairs
        original_step = torch.optim.AdamW.step
        test_pairs = {tuple(pair) for pair in self.data["test_pairs"].tolist()}
        for no_test in (False, True):
            with self.subTest(no_test=no_test):
                events = []
                self.loader.return_value = GuardedFold(self.data, events)
                args = self.args(f"held_out_{no_test}", epochs=2, no_test=no_test)

                def observe_load(path, device="cpu"):
                    checkpoint = original_load(path, device)
                    if Path(path).name == "best.pt":
                        events.append("best_loaded")
                    return checkpoint

                def observe_score(model, node_embeddings, pairs):
                    if any(tuple(pair) in test_pairs for pair in pairs.detach().cpu().tolist()):
                        self.assertIn("best_loaded", events)
                        self.assertFalse(model.training)
                        self.assertFalse(torch.is_grad_enabled())
                        events.append("test_score")
                    return original_score(model, node_embeddings, pairs)

                def observe_step(optimizer, *positional, **kwargs):
                    self.assertNotIn("test_score", events)
                    events.append("optimizer_step")
                    return original_step(optimizer, *positional, **kwargs)

                with mock.patch.object(runner, "_load", side_effect=observe_load), \
                     mock.patch.object(runner.CaPPI, "score_pairs", new=observe_score), \
                     mock.patch.object(torch.optim.AdamW, "step", new=observe_step):
                    result = self.run_fold(args)
                self.assertEqual(events.count("optimizer_step"), 2)
                self.assertEqual(events.count("best_loaded"), 1)
                self.assertEqual(result["test_evaluated"], not no_test)
                if no_test:
                    self.assertNotIn("test_score", events)
                    self.assertNotIn("test_labels", events)
                    self.assertFalse((self.fold_dir(args) / "test_predictions.csv").exists())
                else:
                    self.assertEqual(events.count("test_score"), 1)
                    self.assertGreater(events.count("test_labels"), 0)
                    self.assertLess(events.index("best_loaded"), events.index("test_score"))

    def test_checkpoint_replacement_failure_preserves_original_file(self):
        target = self.root / "atomic.pt"
        runner.save_checkpoint({"marker": "original"}, target)
        original = target.read_bytes()

        def failed_save(value, path):
            Path(path).write_bytes(b"incomplete temporary checkpoint")
            raise OSError("simulated interrupted checkpoint write")

        with mock.patch.object(torch, "save", side_effect=failed_save):
            with self.assertRaises(OSError):
                runner.save_checkpoint({"marker": "new"}, target)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(runner._load(target)["marker"], "original")


if __name__ == "__main__":
    unittest.main(verbosity=2)
