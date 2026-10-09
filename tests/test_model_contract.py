"""Bounded scientific-contract checks; no data download or experiment training.

Run with the intended interpreter from the revised project directory:
    python -m unittest discover -s tests -p test_model_contract.py -v
The gradient/update check uses CUDA when available; all other tests use CPU.
"""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cappi_model import (  # noqa: E402
    AMINO_ACIDS,
    CaPPI,
    ModelConfig,
    ResidualCrossModalFusion,
    get_parameter_report,
    tokenize_sequences,
)


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3107)

    def test_canonical_unknown_and_padding_are_distinct(self):
        tokens = tokenize_sequences([AMINO_ACIDS, "a xBzjuo*- \n", ""], max_length=24)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens[0, :20].tolist(), list(range(1, 21)))
        self.assertEqual(tokens[1, :9].tolist(), [1] + [21] * 8)
        self.assertTrue(torch.equal(tokens[2], torch.zeros(24, dtype=torch.long)))
        self.assertEqual(tokenize_sequences(["ACDE"], 2).tolist(), [[1, 2]])

    def test_trailing_padding_invariance_and_empty_sequence(self):
        model = CaPPI(ModelConfig(fusion="sequence_only", max_length=48)).eval()
        short = tokenize_sequences(["ACDXEFGH", "", "X"], 8)
        long = tokenize_sequences(["ACDXEFGH", "", "X"], 48)
        with torch.no_grad():
            short_output = model.sequence_encoder(short)
            long_output = model.sequence_encoder(long)
        torch.testing.assert_close(short_output, long_output, atol=1e-7, rtol=1e-6)
        self.assertEqual(torch.count_nonzero(long_output[1]).item(), 0)
        self.assertGreater(torch.count_nonzero(long_output[2]).item(), 0)

    def test_missing_topology_masks_values_and_projection_bias(self):
        config = ModelConfig(fusion="topology_only", max_length=8)
        model = CaPPI(config).eval()
        with torch.no_grad():
            model.topology_projection.bias.fill_(5)
        tokens = tokenize_sequences(["ACD", "EFG"], 8)
        topology = torch.randn(2, 64)
        topology[1] = float("nan")
        encoded = model.encode_nodes(tokens, topology, torch.tensor([True, False]))
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertEqual(torch.count_nonzero(encoded[1]).item(), 0)
        self.assertGreater(torch.count_nonzero(encoded[0]).item(), 0)

    def test_pair_score_is_strictly_symmetric_at_evaluation(self):
        model = CaPPI().eval()
        embeddings = torch.randn(8, 256)
        pairs = torch.tensor([[0, 1], [2, 7], [4, 4], [1, 6]], dtype=torch.long)
        with torch.no_grad():
            forward = model.score_pairs(embeddings, pairs)
            reverse = model.score_pairs(embeddings, pairs.flip(1))
        torch.testing.assert_close(forward, reverse, rtol=0, atol=0)
        self.assertEqual(tuple(forward.shape), (4,))

    def test_capacity_match_is_actual_and_report_does_not_change_rng(self):
        base = CaPPI(ModelConfig(fusion="residual"))
        matched = CaPPI(ModelConfig(fusion="concat_matched"))
        state = torch.random.get_rng_state().clone()
        residual_report = get_parameter_report(base)
        matched_report = get_parameter_report(matched)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        actual_base = sum(parameter.numel() for parameter in base.fusion.parameters())
        actual_matched = sum(parameter.numel() for parameter in matched.fusion.parameters())
        self.assertEqual(actual_base, residual_report["reference_residual_fusion_trainable"])
        self.assertEqual(actual_matched, matched_report["modules"]["fusion"]["trainable"])
        self.assertLessEqual(abs(actual_matched - actual_base) / actual_base, 0.01)
        self.assertEqual(matched_report["matched_hidden_width"], 1881)
        self.assertEqual(residual_report["modules"]["sequence_encoder"],
                         matched_report["modules"]["sequence_encoder"])
        self.assertEqual(residual_report["modules"]["pair_head"],
                         matched_report["modules"]["pair_head"])

    def test_ablations_remove_modules_and_all_remaining_parameters_participate(self):
        tokens = tokenize_sequences(["ACDF", "WXQ", "MNK", "EHG"], 8)
        topology = torch.randn(4, 64)
        present = torch.ones(4, dtype=torch.bool)
        pairs = torch.tensor([[0, 1], [0, 2], [2, 3], [1, 3]], dtype=torch.long)
        for kind in ("residual", "concat_matched", "concat_linear", "sequence_only", "topology_only"):
            with self.subTest(fusion=kind):
                model = CaPPI(ModelConfig(fusion=kind, max_length=8))
                logits = model(tokens, topology, present, pairs)
                self.assertTrue(torch.isfinite(logits).all())
                nn.BCEWithLogitsLoss()(logits, torch.tensor([1., 0., 1., 0.])).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, msg=f"unused parameter {kind}/{name}")
                if kind == "sequence_only":
                    self.assertIsNone(model.topology_projection)
                    self.assertIsNone(model.fusion)
                if kind == "topology_only":
                    self.assertIsNone(model.sequence_encoder)
                    self.assertIsNone(model.fusion)

    def test_joint_training_gradients_and_one_optimizer_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CaPPI(ModelConfig(max_length=32)).to(device).train()
        sequences = ["ACDEFGHIKL", "MNOPQRSTVWY", "WYYXKLMNPQ", "AACCFGY"] * 2
        tokens = tokenize_sequences(sequences, 32).to(device)
        topology = torch.randn(8, 64, device=device)
        present = torch.ones(8, dtype=torch.bool, device=device)
        pairs = torch.tensor([[i, (i + 1) % 8] for i in range(8)], device=device)
        labels = torch.tensor([1., 0., 1., 0., 0., 1., 0., 1.], device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.sequence_encoder.conv.weight.detach().clone()
        loss = nn.BCEWithLogitsLoss()(model(tokens, topology, present, pairs), labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        norms = {}
        for name in ("sequence_encoder", "topology_projection", "fusion", "pair_head"):
            module = getattr(model, name)
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            norms[name] = sum(float(gradient.detach().abs().sum().cpu()) for gradient in gradients)
            self.assertGreater(norms[name], 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.sequence_encoder.conv.weight.detach()))
        print(f"\nBounded update: device={device}, torch={torch.__version__}, "
              f"loss={float(loss.detach().cpu()):.6f}, absolute_gradient_sums={norms}")

    def test_residual_matches_original_singleton_mha_effective_function(self):
        # Copy original MHA V/O, FFNs, shared norms and final projection to verify
        # removing Q/K preserves the effective original deterministic function.
        dim = 16
        reference_a = nn.MultiheadAttention(dim, 4, dropout=0.0, batch_first=True)
        reference_b = nn.MultiheadAttention(dim, 4, dropout=0.0, batch_first=True)
        revised = ResidualCrossModalFusion(dim, 0.0).eval()
        with torch.no_grad():
            for attention, value, output in (
                (reference_a, revised.a_from_b_value, revised.a_from_b_output),
                (reference_b, revised.b_from_a_value, revised.b_from_a_output),
            ):
                value.weight.copy_(attention.in_proj_weight[2 * dim:])
                value.bias.copy_(attention.in_proj_bias[2 * dim:])
                output.weight.copy_(attention.out_proj.weight)
                output.bias.copy_(attention.out_proj.bias)
            a, b = torch.randn(5, dim), torch.randn(5, dim)
            a_message, _ = reference_a(a[:, None], b[:, None], b[:, None], need_weights=False)
            b_message, _ = reference_b(b[:, None], a[:, None], a[:, None], need_weights=False)
            a_out = revised.norm_a(a + a_message[:, 0])
            a_out = revised.norm_a(a_out + revised.ffn_a(a_out))
            b_out = revised.norm_b(b + b_message[:, 0])
            b_out = revised.norm_b(b_out + revised.ffn_b(b_out))
            expected = revised.projection(torch.cat((a_out, b_out), dim=-1))
            actual = revised(a, b)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)

    def test_shape_and_index_contracts_fail_explicitly(self):
        with self.assertRaises(ValueError):
            ModelConfig(kernel=4)
        model = CaPPI(ModelConfig(max_length=8))
        with self.assertRaises(ValueError):
            model.sequence_encoder(torch.tensor([[22]], dtype=torch.long))
        with self.assertRaises(ValueError):
            model.score_pairs(torch.randn(2, 256), torch.tensor([[0, -1]]))
        with self.assertRaises(ValueError):
            model.encode_nodes(torch.ones(2, 8, dtype=torch.long), torch.randn(2, 64), torch.ones(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
