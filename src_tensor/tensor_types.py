"""Tensor graph data structures (GPU-ready)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


@dataclass
class LayerEdgesTensor:
    """Edges grouped by topological level (tensor form)."""
    level: int
    parent_idx: torch.Tensor
    child_idx: torch.Tensor
    edge_sign: torch.Tensor
    edge_trig: torch.Tensor
    edge_param: torch.Tensor


@dataclass
class TensorDAGEdges:
    """Edge list for DAG evaluation (tensor form)."""
    parent_idx: torch.Tensor
    child_idx: torch.Tensor
    edge_sign: torch.Tensor
    edge_trig: torch.Tensor
    edge_param: torch.Tensor


@dataclass
class TensorDAGGraph:
    """Compact DAG representation with tensors (GPU-ready)."""
    num_nodes: int
    edges: TensorDAGEdges
    node_value_init: torch.Tensor
    topo_order: torch.Tensor
    final_nodes: torch.Tensor
    final_coeff: torch.Tensor
    layers: Optional[List[LayerEdgesTensor]] = None


@dataclass
class TensorPauliSum:
    """Tensor-native PauliSum representation (GPU-ready)."""
    n_qubits: int
    x_mask: torch.Tensor
    z_mask: torch.Tensor
    coeff_init: torch.Tensor
    steps: List["TensorSparseStep"]


@dataclass
class TensorSparseStep:
    """Sparse linear maps for one rotation/selection step.

    Each step represents:
        v_out = M_const v_in + cos(theta) M_cos v_in + sin(theta) M_sin v_in
    where M_* are sparse matrices with embedded phase signs.

    For large Pauli rotation steps, unchanged branches may be stored implicitly
    via `same_cols` / `anti_same_pos` instead of materializing `mat_const` and
    `mat_cos`.
    """
    mat_const: torch.Tensor
    mat_cos: torch.Tensor
    mat_sin: torch.Tensor
    param_idx: int
    shape: Tuple[int, int]  # [수정] 기본값이 없는 필드를 위로 올림
    emb_idx: int = -1       # [수정] 기본값이 있는 필드를 가장 아래로 배치
    same_cols: Optional[torch.Tensor] = None
    anti_same_pos: Optional[torch.Tensor] = None

    def same_nnz(self) -> int:
        if self.same_cols is None:
            return 0
        return int(self.same_cols.numel())

    def to(self, device: Any) -> "TensorSparseStep":
        return TensorSparseStep(
            mat_const=self.mat_const.to(device),
            mat_cos=self.mat_cos.to(device),
            mat_sin=self.mat_sin.to(device),
            param_idx=self.param_idx,
            shape=self.shape,
            emb_idx=self.emb_idx,
            same_cols=None if self.same_cols is None else self.same_cols.to(device),
            anti_same_pos=None if self.anti_same_pos is None else self.anti_same_pos.to(device),
        )


def tensor_psum_to_device(psum: "TensorPauliSum", device: str) -> "TensorPauliSum":
    """Move a TensorPauliSum and its steps to a device."""
    steps = [s.to(device) for s in psum.steps]
    return TensorPauliSum(
        n_qubits=psum.n_qubits,
        x_mask=psum.x_mask.to(device),
        z_mask=psum.z_mask.to(device),
        coeff_init=psum.coeff_init.to(device),
        steps=steps,
    )


__all__ = [
    "LayerEdgesTensor",
    "TensorDAGEdges",
    "TensorDAGGraph",
    "TensorPauliSum",
    "TensorSparseStep",
    "tensor_psum_to_device",
]
