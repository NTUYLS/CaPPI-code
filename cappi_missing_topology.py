"""Controlled, parameter-free missing-topology adaptation for CaPPI.

This module leaves the archived experiment model untouched. ``original`` uses
CaPPI's existing path exactly. ``bypass`` retains the residual model's parameters
and shared pair head, but uses the sequence encoder output directly for nodes
whose topology is unavailable. Topology dropout is an explicit caller operation;
this model never samples a modality mask during training or evaluation.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor

from cappi_model import CaPPI, ModelConfig


def apply_topology_dropout(
    present: Tensor, probability: float, generator: torch.Generator | None = None
) -> Tensor:
    """Independently hide whole node topology vectors, without inspecting labels.

    ``present`` is a boolean vector on the target device. Already unavailable
    topology cannot become available. At probability zero, return a copy without
    advancing any RNG; at probability one, return an all-false mask. Otherwise
    sample one uniform draw per node using the caller's generator (or the device
    default generator). The caller supplies a generator for the same device.
    """
    if not isinstance(present, Tensor) or present.ndim != 1 or present.dtype != torch.bool:
        raise ValueError("present must have dtype torch.bool and shape [N]")
    if isinstance(probability, bool) or not isinstance(probability, Real):
        raise ValueError("probability must be a finite number in [0, 1]")
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be a finite number in [0, 1]")
    if probability == 0:
        return present.clone()
    if probability == 1:
        return torch.zeros_like(present)
    keep = torch.rand(present.shape, device=present.device, generator=generator) >= probability
    return present & keep


class MissingTopologyCaPPI(CaPPI):
    """CaPPI with an explicit ``original`` or residual-only ``bypass`` policy.

    The constructor registers exactly the original CaPPI parameters, with the
    same names, initialization, and RNG consumption. Available nodes keep the
    original fusion path; missing nodes use the sequence encoding with the same
    ``pair_head``. Both branches are in one model and optimized jointly.
    """

    def __init__(self, config: ModelConfig, missing_policy: str = "original") -> None:
        if missing_policy not in {"original", "bypass"}:
            raise ValueError("missing_policy must be 'original' or 'bypass'")
        if missing_policy == "bypass" and config.fusion != "residual":
            raise ValueError("missing_policy='bypass' requires fusion='residual'")
        super().__init__(config)
        self.missing_policy = missing_policy

    def encode_nodes(self, tokens: Tensor, topology: Tensor, topology_present: Tensor) -> Tensor:
        if self.missing_policy == "original":
            return super().encode_nodes(tokens, topology, topology_present)

        # Mirror the original tensor contract. Mask BEFORE projection to stop
        # arbitrary/NaN missing inputs entering the graph, and AFTER projection
        # to remove its bias. The final bypass also excludes all fusion biases,
        # normalization, and topology-to-sequence messages for missing nodes.
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must have dtype torch.long and shape [N, L]")
        n_nodes = tokens.shape[0]
        if topology.shape != (n_nodes, self.config.topo_dim) or not topology.is_floating_point():
            raise ValueError(f"topology must be floating point [N, {self.config.topo_dim}]")
        if topology_present.shape != (n_nodes,) or topology_present.dtype != torch.bool:
            raise ValueError("topology_present must have dtype torch.bool and shape [N]")
        if topology.device != tokens.device or topology_present.device != tokens.device:
            raise ValueError("tokens, topology and topology_present must be on the same device")

        sequence = self.sequence_encoder(tokens)
        safe_topology = topology.masked_fill(~topology_present[:, None], 0)
        projected = self.topology_projection(safe_topology)
        projected = projected.masked_fill(~topology_present[:, None], 0)
        # Keep the full-batch fusion forward, so dropout RNG consumption and
        # available-node values remain identical to the original model even in
        # train mode with mixed masks. torch.where discards its missing-node
        # outputs and their gradient contribution; no extra RNG is introduced.
        fused = self.fusion(sequence, projected)
        return torch.where(topology_present[:, None], fused, sequence)
