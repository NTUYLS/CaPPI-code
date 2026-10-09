"""Trainable CaPPI sequence/topology model with an explicit undirected-pair contract.

Scientific changes relative to the archived scripts:
* Jointly optimize GCNN, topology projection, fusion and the pair classifier using
  training labels. Never reuse the archived random, untrained feature CSVs.
* Encode canonical amino acids as 20 separate channels plus one unknown channel;
  padding has no channel and is excluded from the convolutional mean pool.
* Replace singleton multihead attention with its effective value/output linear
  maps. With one key token, the attention weight is always one: query/key weights
  cannot affect its output. This module is residual cross-modal fusion, NOT
  token-to-token attention. The original shared LayerNorm within each direction,
  two FFNs and final LayerNorm/projection are retained. The configurable dropout
  here acts in the FFNs (default 0.3); the archived CAM used dropout 0.0.
* Average both endpoint orders at the logit level. Retain the original pair-head
  widths and dropout rates, but return a single binary logit for BCEWithLogitsLoss
  instead of two softmax scores. Old classifier activity regularization is not
  silently reproduced here; the training configuration must disclose its loss
  and optimizer regularization separately.

Topology vectors are fixed INPUTS fitted on the training graph only. This module
does not construct a graph, fit Skip-gram, or inspect validation/test edges. A
caller must validate graph provenance and provide an explicit presence mask;
unavailable topology is zeroed after projection, including the projection bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = 0
UNKNOWN_TOKEN = 21
AMINO_TO_TOKEN = {residue: index + 1 for index, residue in enumerate(AMINO_ACIDS)}
FusionKind = Literal[
    "residual", "concat_matched", "concat_linear", "sequence_only", "topology_only"
]


@dataclass(frozen=True)
class ModelConfig:
    topo_dim: int = 64
    seq_dim: int = 256
    filters: int = 64
    kernel: int = 7
    max_length: int = 1500
    fusion: FusionKind = "residual"
    dropout: float = 0.3
    pair_dropout: tuple[float, float, float] = (0.4, 0.4, 0.1)

    def __post_init__(self) -> None:
        for name in ("topo_dim", "seq_dim", "filters", "kernel", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.kernel % 2 != 1:
            raise ValueError("kernel must be odd for length-preserving convolution")
        if self.fusion not in {
            "residual", "concat_matched", "concat_linear", "sequence_only", "topology_only"
        }:
            raise ValueError(f"unsupported fusion: {self.fusion!r}")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if len(self.pair_dropout) != 3 or any(not 0 <= p < 1 for p in self.pair_dropout):
            raise ValueError("pair_dropout must contain three probabilities in [0, 1)")


def tokenize_sequences(sequences: Iterable[str], max_length: int = 1500) -> Tensor:
    """Return CPU int64 [N, max_length]; truncate on the right, then pad with 0.

    Whitespace is removed and letter case normalized. Every other noncanonical
    character (including X, B, Z, J, U, O, '*' and '-') maps to token 21. Empty
    sequences become all padding; upstream data validation should decide whether
    an empty sequence is permissible in a scientific experiment.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if isinstance(sequences, str):
        raise TypeError("sequences must be an iterable of sequence strings, not one string")
    sequence_list = list(sequences)
    tokens = torch.zeros((len(sequence_list), max_length), dtype=torch.long)
    for row, sequence in enumerate(sequence_list):
        if not isinstance(sequence, str):
            raise TypeError(f"sequence {row} must be a string")
        normalized = "".join(sequence.split()).upper()[:max_length]
        if normalized:
            tokens[row, :len(normalized)] = torch.tensor(
                [AMINO_TO_TOKEN.get(residue, UNKNOWN_TOKEN) for residue in normalized],
                dtype=torch.long,
            )
    return tokens


class GatedSequenceEncoder(nn.Module):
    """One gated convolution, padding-masked global mean, trainable projection."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_length = config.max_length
        self.output_dim = config.seq_dim
        self.conv = nn.Conv1d(21, config.filters, config.kernel, padding=config.kernel // 2)
        self.gate = nn.Conv1d(21, config.filters, config.kernel, padding=config.kernel // 2)
        self.projection = nn.Linear(config.filters, config.seq_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dtype != torch.long or tokens.ndim != 2:
            raise ValueError("tokens must have dtype torch.long and shape [N, L]")
        if not 1 <= tokens.shape[1] <= self.max_length:
            raise ValueError(f"token length must be in [1, {self.max_length}]")
        if bool(((tokens < PAD_TOKEN) | (tokens > UNKNOWN_TOKEN)).any()):
            raise ValueError("token IDs must be in [0, 21]")
        if tokens.shape[0] == 0:
            return self.projection.weight.new_empty((0, self.output_dim))
        mask = tokens.ne(PAD_TOKEN).unsqueeze(1)
        # Drop the padding channel entirely: all padded inputs are exactly zero.
        one_hot = F.one_hot(tokens, num_classes=22)[..., 1:].transpose(1, 2)
        one_hot = one_hot.to(dtype=self.conv.weight.dtype)
        features = self.conv(one_hot) * torch.sigmoid(self.gate(one_hot))
        # Convolution biases/neighbouring residues can activate padded positions;
        # masking AFTER the convolution prevents these positions entering the pool.
        features = features * mask.to(dtype=features.dtype)
        denominator = mask.sum(dim=-1).clamp_min(1).to(dtype=features.dtype)
        pooled = features.sum(dim=-1) / denominator
        encoded = self.projection(pooled)
        return encoded * mask.any(dim=-1).to(dtype=encoded.dtype)


class ResidualCrossModalFusion(nn.Module):
    """Effective bidirectional singleton CAM, with no inert query/key parameters."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.a_from_b_value = nn.Linear(dim, dim)
        self.a_from_b_output = nn.Linear(dim, dim)
        self.b_from_a_value = nn.Linear(dim, dim)
        self.b_from_a_output = nn.Linear(dim, dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.ffn_a = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.projection = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim))

    def forward(self, sequence: Tensor, topology: Tensor) -> Tensor:
        a = self.norm_a(sequence + self.a_from_b_output(self.a_from_b_value(topology)))
        a = self.norm_a(a + self.ffn_a(a))
        b = self.norm_b(topology + self.b_from_a_output(self.b_from_a_value(sequence)))
        b = self.norm_b(b + self.ffn_b(b))
        return self.projection(torch.cat((a, b), dim=-1))


def residual_fusion_parameter_count(dim: int) -> int:
    """Exact effective residual-fusion count, excluding sequence/topology encoders."""
    return 22 * dim * dim + 23 * dim


class CapacityMatchedConcatFusion(nn.Module):
    """Concatenation MLP whose actual used parameters match residual fusion."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        target = residual_fusion_parameter_count(dim)
        # LN(2d) + Linear(2d,h) + Linear(h,d) = (3d+1)h + 5d.
        self.hidden_width = max(1, round((target - 5 * dim) / (3 * dim + 1)))
        self.network = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, self.hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_width, dim),
        )
        actual = sum(parameter.numel() for parameter in self.parameters())
        if abs(actual - target) / target > 0.01:
            raise ValueError("cannot match fusion capacity within 1% at this dimension")

    def forward(self, sequence: Tensor, topology: Tensor) -> Tensor:
        return self.network(torch.cat((sequence, topology), dim=-1))


class LinearConcatFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * dim, dim)

    def forward(self, sequence: Tensor, topology: Tensor) -> Tensor:
        return self.projection(torch.cat((sequence, topology), dim=-1))


class PairHead(nn.Module):
    """Archived endpoint-specific widths, symmetrized at the binary-logit level."""

    def __init__(self, dim: int, dropout: tuple[float, float, float]) -> None:
        super().__init__()
        self.left = nn.Sequential(nn.Linear(dim, 64), nn.ReLU())
        self.right = nn.Sequential(nn.Linear(dim, 128), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, 128), nn.ReLU(), nn.Dropout(dropout[0]),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout[1]),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout[2]),
            nn.Linear(32, 1),
        )

    def ordered_logits(self, left: Tensor, right: Tensor) -> Tensor:
        return self.classifier(torch.cat((self.left(left), self.right(right)), dim=-1)).squeeze(-1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        # Strictly symmetric at eval; independent training dropout is symmetric
        # in distribution. Do not report train-mode equality as an invariant.
        return (self.ordered_logits(left, right) + self.ordered_logits(right, left)) * 0.5


class CaPPI(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.sequence_encoder = (
            None if self.config.fusion == "topology_only" else GatedSequenceEncoder(self.config)
        )
        self.topology_projection = (
            None if self.config.fusion == "sequence_only"
            else nn.Linear(self.config.topo_dim, self.config.seq_dim)
        )
        if self.config.fusion == "residual":
            self.fusion = ResidualCrossModalFusion(self.config.seq_dim, self.config.dropout)
        elif self.config.fusion == "concat_matched":
            self.fusion = CapacityMatchedConcatFusion(self.config.seq_dim, self.config.dropout)
        elif self.config.fusion == "concat_linear":
            self.fusion = LinearConcatFusion(self.config.seq_dim)
        else:
            self.fusion = None
        self.pair_head = PairHead(self.config.seq_dim, self.config.pair_dropout)

    def encode_nodes(self, tokens: Tensor, topology: Tensor, topology_present: Tensor) -> Tensor:
        """Encode N nodes; topology_present=False means no permitted train-graph vector.

        Protein-disjoint nodes may use sequence features with missing topology.
        This is an explicit missing-topology adaptation; the fixed Skip-gram
        representation itself remains transductive.
        """
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must have dtype torch.long and shape [N, L]")
        n_nodes = tokens.shape[0]
        if topology.shape != (n_nodes, self.config.topo_dim) or not topology.is_floating_point():
            raise ValueError(f"topology must be floating point [N, {self.config.topo_dim}]")
        if topology_present.shape != (n_nodes,) or topology_present.dtype != torch.bool:
            raise ValueError("topology_present must have dtype torch.bool and shape [N]")
        if topology.device != tokens.device or topology_present.device != tokens.device:
            raise ValueError("tokens, topology and topology_present must be on the same device")
        sequence = self.sequence_encoder(tokens) if self.sequence_encoder is not None else None
        if self.topology_projection is not None:
            # Mask before AND after projection so arbitrary/NaN unavailable input
            # cannot enter the model, and a learned bias cannot imitate topology.
            safe_topology = topology.masked_fill(~topology_present[:, None], 0)
            projected = self.topology_projection(safe_topology)
            projected = projected.masked_fill(~topology_present[:, None], 0)
        else:
            projected = None
        if self.config.fusion == "sequence_only":
            return sequence
        if self.config.fusion == "topology_only":
            return projected
        return self.fusion(sequence, projected)

    def score_pairs(self, node_embeddings: Tensor, pairs: Tensor) -> Tensor:
        """Return raw binary logits [B], indexed using node rows (no graph access)."""
        if node_embeddings.ndim != 2 or node_embeddings.shape[1] != self.config.seq_dim:
            raise ValueError(f"node_embeddings must have shape [N, {self.config.seq_dim}]")
        if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have dtype torch.long and shape [B, 2]")
        if pairs.device != node_embeddings.device:
            raise ValueError("pairs and node_embeddings must be on the same device")
        if bool(((pairs < 0) | (pairs >= node_embeddings.shape[0])).any()):
            raise ValueError("pair endpoint outside node_embeddings row range")
        return self.pair_head(node_embeddings[pairs[:, 0]], node_embeddings[pairs[:, 1]])

    def forward(self, tokens: Tensor, topology: Tensor, topology_present: Tensor, pairs: Tensor) -> Tensor:
        return self.score_pairs(self.encode_nodes(tokens, topology, topology_present), pairs)


def get_parameter_report(model: CaPPI) -> dict:
    """Report actual trainable counts and the specified capacity-matching reference."""
    modules = {}
    for name in ("sequence_encoder", "topology_projection", "fusion", "pair_head"):
        module = getattr(model, name)
        parameters = list(module.parameters()) if module is not None else []
        modules[name] = {
            "total": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        }
    actual = modules["fusion"]["trainable"]
    target = residual_fusion_parameter_count(model.config.seq_dim)
    return {
        "config": asdict(model.config),
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "modules": modules,
        "reference_residual_fusion_trainable": target,
        "fusion_capacity_relative_difference": abs(actual - target) / target,
        "fusion_capacity_signed_difference": actual - target,
        "matched_hidden_width": getattr(model.fusion, "hidden_width", None),
        "capacity_match_applies": model.config.fusion in {"residual", "concat_matched"},
        "pair_output": "mean_of_forward_and_reverse_binary_logits",
        "topology": "fixed_train_graph_input_with_trainable_masked_projection",
    }
