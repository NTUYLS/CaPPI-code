"""Bounded synthetic checks for the controlled missing-topology experiment."""

import dataclasses
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cappi_model import CaPPI, ModelConfig, tokenize_sequences
from cappi_missing_topology import MissingTopologyCaPPI, apply_topology_dropout


class MissingTopologyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7107)
        self.config = ModelConfig(topo_dim=4, seq_dim=16, filters=4, kernel=3,
                                  max_length=16, dropout=0.3)
        self.tokens = tokenize_sequences(["ACDE", "YYWQ", "MXN", "GHHI", "KLM", "PQ"], 16)
        self.topology = torch.randn(6, 4)
        self.present = torch.tensor([True, False, True, False, True, False])
        self.pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [1, 3]], dtype=torch.long)

    def identical_models(self, policy="bypass", config=None):
        config = self.config if config is None else config
        torch.manual_seed(913)
        base = CaPPI(config)
        base_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(913)
        adapted = MissingTopologyCaPPI(config, policy)
        self.assertTrue(torch.equal(base_rng, torch.random.get_rng_state()))
        self.assertEqual(list(base.state_dict()), list(adapted.state_dict()))
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)
        return base, adapted

    def test_original_is_exact_for_all_fusion_modes_and_training_rng(self):
        for fusion in ("residual", "concat_matched", "concat_linear", "sequence_only", "topology_only"):
            config = dataclasses.replace(self.config, fusion=fusion)
            base, original = self.identical_models("original", config)
            for training in (False, True):
                with self.subTest(fusion=fusion, training=training):
                    base.train(training)
                    original.train(training)
                    torch.manual_seed(2309)
                    expected = base(self.tokens, self.topology, self.present, self.pairs)
                    expected_rng = torch.random.get_rng_state().clone()
                    torch.manual_seed(2309)
                    actual = original(self.tokens, self.topology, self.present, self.pairs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))

    def test_bypass_adds_no_parameters_and_all_present_path_is_exact(self):
        base, bypass = self.identical_models()
        self.assertEqual(sum(p.numel() for p in base.parameters()),
                         sum(p.numel() for p in bypass.parameters()))
        present = torch.ones(6, dtype=torch.bool)
        for training in (False, True):
            base.train(training)
            bypass.train(training)
            torch.manual_seed(33)
            expected = base(self.tokens, self.topology, present, self.pairs)
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(33)
            actual = bypass(self.tokens, self.topology, present, self.pairs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))

    def test_mixed_masks_keep_available_nodes_and_bypass_missing_nodes_exactly(self):
        base, bypass = self.identical_models()
        for training in (False, True):
            base.train(training)
            bypass.train(training)
            torch.manual_seed(550)
            original = base.encode_nodes(self.tokens, self.topology, self.present)
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(550)
            actual = bypass.encode_nodes(self.tokens, self.topology, self.present)
            torch.testing.assert_close(actual[self.present], original[self.present], rtol=0, atol=0)
            self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
            sequence = bypass.sequence_encoder(self.tokens)
            torch.testing.assert_close(actual[~self.present], sequence[~self.present], rtol=0, atol=0)

    def test_missing_nan_and_projection_bias_do_not_reach_output_or_gradients(self):
        model = MissingTopologyCaPPI(self.config, "bypass").eval()
        with torch.no_grad():
            model.topology_projection.bias.fill_(50)
            model.fusion.a_from_b_value.bias.fill_(30)
        topology = self.topology.clone()
        topology[~self.present] = float("nan")
        topology.requires_grad_(True)
        actual = model.encode_nodes(self.tokens, topology, self.present)
        self.assertTrue(torch.isfinite(actual).all())
        expected = model.sequence_encoder(self.tokens)
        torch.testing.assert_close(actual[~self.present], expected[~self.present], rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(topology.grad).all())
        self.assertEqual(torch.count_nonzero(topology.grad[~self.present]).item(), 0)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_all_missing_uses_shared_pair_head_and_no_fusion_gradient(self):
        model = MissingTopologyCaPPI(self.config, "bypass").eval()
        present = torch.zeros(6, dtype=torch.bool)
        topology = torch.full_like(self.topology, float("nan"))
        actual = model(self.tokens, topology, present, self.pairs)
        expected = model.score_pairs(model.sequence_encoder(self.tokens), self.pairs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        nn.BCEWithLogitsLoss()(actual, torch.tensor([1., 0., 1., 0.])).backward()
        for module in (model.sequence_encoder, model.pair_head):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        for module in (model.topology_projection, model.fusion):
            self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0
                                for p in module.parameters()))

    def test_bounded_update_is_finite_on_intended_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MissingTopologyCaPPI(self.config, "bypass").to(device).train()
        tokens, pairs, present = self.tokens.to(device), self.pairs.to(device), self.present.to(device)
        topology = self.topology.to(device).clone()
        topology[~present] = float("nan")
        before = model.sequence_encoder.conv.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = nn.BCEWithLogitsLoss()(model(tokens, topology, present, pairs),
                                    torch.tensor([1., 0., 1., 0.], device=device))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.sequence_encoder.conv.weight.detach()))

    def test_invalid_policy_config_and_input_contracts(self):
        with self.assertRaises(ValueError):
            MissingTopologyCaPPI(self.config, "unknown")
        for fusion in ("concat_matched", "concat_linear", "sequence_only", "topology_only"):
            with self.assertRaises(ValueError):
                MissingTopologyCaPPI(dataclasses.replace(self.config, fusion=fusion), "bypass")
        model = MissingTopologyCaPPI(self.config, "bypass")
        invalid = [(self.tokens.float(), self.topology, self.present),
                   (self.tokens, self.topology[:, :3], self.present),
                   (self.tokens, self.topology.long(), self.present),
                   (self.tokens, self.topology, self.present.float()),
                   (self.tokens, self.topology, self.present[:, None])]
        for inputs in invalid:
            with self.assertRaises(ValueError):
                model.encode_nodes(*inputs)

    def test_dropout_zero_copies_mask_without_consuming_rng(self):
        generator = torch.Generator().manual_seed(21)
        state, global_state = generator.get_state().clone(), torch.random.get_rng_state().clone()
        actual = apply_topology_dropout(self.present, 0, generator)
        self.assertTrue(torch.equal(actual, self.present))
        self.assertNotEqual(actual.data_ptr(), self.present.data_ptr())
        self.assertTrue(torch.equal(state, generator.get_state()))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_dropout_one_all_missing_without_consuming_rng(self):
        generator = torch.Generator().manual_seed(21)
        state = generator.get_state().clone()
        self.assertFalse(apply_topology_dropout(self.present, 1, generator).any())
        self.assertTrue(torch.equal(state, generator.get_state()))

    def test_dropout_is_nodewise_reproducible_and_cannot_restore_unavailable_nodes(self):
        present = torch.ones(1024, dtype=torch.bool)
        present[::5] = False
        generator = torch.Generator().manual_seed(701)
        actual = apply_topology_dropout(present, 0.5, generator)
        reference_generator = torch.Generator().manual_seed(701)
        expected = present & (torch.rand(present.shape, generator=reference_generator) >= 0.5)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(generator.get_state(), reference_generator.get_state()))
        self.assertFalse(actual[~present].any())
        self.assertTrue(torch.equal(actual, apply_topology_dropout(
            present, 0.5, torch.Generator().manual_seed(701))))
        self.assertGreater(float(actual[present].float().mean()), 0.4)
        self.assertLess(float(actual[present].float().mean()), 0.6)

    def test_dropout_probability_and_mask_validation(self):
        for probability in (-0.01, 1.01, float("nan"), float("inf"), -float("inf"), True, "0.5", None):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                apply_topology_dropout(self.present, probability)
        for present in ([True], self.present.float(), self.present[:, None]):
            with self.assertRaises(ValueError):
                apply_topology_dropout(present, 0.2)
        self.assertEqual(apply_topology_dropout(torch.empty(0, dtype=torch.bool), 0.5).shape, (0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
