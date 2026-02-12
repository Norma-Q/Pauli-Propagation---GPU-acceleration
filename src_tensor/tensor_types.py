"""Tensor graph data structures (GPU-ready)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    """
    mat_const: torch.Tensor
    mat_cos: torch.Tensor
    mat_sin: torch.Tensor
    param_idx: int
    shape: Tuple[int, int]  # [수정] 기본값이 없는 필드를 위로 올림
    emb_idx: int = -1       # [수정] 기본값이 있는 필드를 가장 아래로 배치


def tensor_psum_to_device(psum: "TensorPauliSum", device: str) -> "TensorPauliSum":
    """Move a TensorPauliSum and its steps to a device."""
    steps = [
        TensorSparseStep(
            mat_const=s.mat_const.to(device),
            mat_cos=s.mat_cos.to(device),
            mat_sin=s.mat_sin.to(device),
            param_idx=s.param_idx,
            emb_idx=s.emb_idx,  # [추가] 장치 이동 시에도 임베딩 인덱스 유지
            shape=s.shape,
        )
        for s in psum.steps
    ]
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