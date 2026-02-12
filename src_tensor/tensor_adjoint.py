"""Adjoint + K-large helpers for tensor backend.

This module intentionally keeps higher-level convenience functions out of
`tensor_propagate.py` to avoid expanding the propagation API surface.

Primary primitive lives in `TensorSparseEvaluator.evaluate_adjoint`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, cast

torch: Any
try:
    import torch as _torch

    torch = _torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = cast(Any, None)
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from torch import Tensor
else:  # pragma: no cover - typing only
    Tensor = Any

from .tensor_eval import TensorSparseEvaluator
from .tensor_propagate import propagate_surrogate_tensor
from .tensor_types import TensorPauliSum


@dataclass
class UnionBasis:
    """Ordered union basis description."""

    n_qubits: int
    pstrs: List[Any]
    index: Dict[Any, int]


@dataclass
class UnionBasisObservable:
    """Minimal observable-like wrapper for propagate_surrogate_tensor.

    It must provide:
      - n_qubits
      - terms: Dict[PauliString, float]

    Coefficients are irrelevant when min_abs is None.
    """

    n_qubits: int
    terms: Dict[Any, float]


def build_union_basis(observables: Sequence) -> UnionBasis:
    if len(observables) == 0:
        raise ValueError("observables must be a non-empty sequence")

    n_qubits = int(observables[0].n_qubits)
    for obs in observables:
        if int(obs.n_qubits) != n_qubits:
            raise ValueError("All observables must have the same n_qubits")

    index: Dict[Any, int] = {}
    pstrs: List[Any] = []
    for obs in observables:
        for p in obs.terms.keys():
            if p not in index:
                index[p] = len(pstrs)
                pstrs.append(p)

    return UnionBasis(n_qubits=n_qubits, pstrs=pstrs, index=index)


def build_union_observable(observables: Sequence) -> Tuple[UnionBasisObservable, UnionBasis]:
    basis = build_union_basis(observables)
    # Preserve order by inserting keys in basis order.
    # Use a non-zero seed per basis term so theta-aware min_abs pruning can
    # estimate path magnitudes during compile. Zero seeds would prune
    # everything immediately whenever min_abs is enabled.
    max_abs_coeff: Dict[Any, float] = {p: 0.0 for p in basis.pstrs}
    for obs in observables:
        for p, c in obs.terms.items():
            c_abs = abs(float(c))
            if c_abs > max_abs_coeff[p]:
                max_abs_coeff[p] = c_abs
    terms = {p: (v if v > 0.0 else 1.0) for p, v in max_abs_coeff.items()}
    return UnionBasisObservable(n_qubits=basis.n_qubits, terms=terms), basis


def propagate_union_basis_psum(
    *,
    circuit,
    observables: Sequence,
    max_weight: int = 50,
    max_xy: int = 50,
    device: str = "cuda",
    dtype: str = "float32",
    offload_steps: bool = True,
    offload_keep: int = 1,
    step_device: str = "cpu",
    thetas=None,
    min_abs: Optional[float] = None,
    min_mat_abs: Optional[float] = None,
) -> Tuple[TensorPauliSum, UnionBasis]:
    """Build a TensorPauliSum using the ordered union of observable terms.

    Uses the existing `propagate_surrogate_tensor` entrypoint by supplying a
    dummy observable with the union term set.
    """

    union_obs, basis = build_union_observable(observables)
    psum = propagate_surrogate_tensor(
        circuit=circuit,
        observable=union_obs,
        max_weight=max_weight,
        max_xy=max_xy,
        device=device,
        dtype=dtype,
        thetas=thetas,
        min_abs=min_abs,
        min_mat_abs=min_mat_abs,
        offload_steps=offload_steps,
        offload_keep=offload_keep,
        step_device=step_device,
    )
    return psum, basis


def adjoint_weights_on_zero(
    psum: TensorPauliSum,
    thetas,
    *,
    priors=None,
    stream_device: Optional[str] = None,
    offload_back: bool = False,
) -> Tensor:
    """Compute w = M(theta, priors)^T s for |0...0> expectation."""

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")

    # 1. Evaluator 생성 (기존 구조 유지)
    evaluator = TensorSparseEvaluator(psum, stream_device=stream_device, offload_back=offload_back)
    
    n_out = int(psum.x_mask.shape[0])
    if n_out == 0:
        n_in = int(psum.coeff_init.shape[0])
        # [수정] 배치가 있을 경우 zeros도 (Batch, n_in) 형태여야 함
        batch_size = priors.shape[0] if priors is not None else 1
        return torch.zeros((batch_size, n_in), dtype=psum.coeff_init.dtype, device=psum.coeff_init.device)

    # 2. 초기 상태 |0...0>에 대응하는 마스크 생성
    diag_mask = (psum.x_mask == 0) if psum.x_mask.dim() == 1 else (psum.x_mask == 0).all(dim=1)
    s = diag_mask.to(dtype=psum.coeff_init.dtype, device=psum.coeff_init.device)
    
    # 3. [핵심 수정] evaluate_adjoint 호출
    # 이제 priors는 (Batch, Dim) 형태이며, evaluator 내부에서 이를 이용해 
    # (Batch, Num_Paulis) 형태의 w를 반환하도록 로직이 확장되어야 합니다.
    w = evaluator.evaluate_adjoint(thetas, s, priors=priors)
    
    return w

def expvals_from_w_and_coeff_matrix(w: Tensor, V0: Tensor) -> Tensor:
    """Compute expectation values from weights 'w' and observable matrix 'V0'.
    
    Args:
        w: Pauli weights of shape (Batch, Num_Paulis) or (Num_Paulis,)
        V0: Observable coefficient matrix of shape (Num_Paulis, Num_Observables)
        
    Returns:
        Tensor: Expectation values of shape (Batch, Num_Observables)
    """
    # 1. w가 1차원(배치 없음)으로 들어온 경우를 대비한 처리
    if w.dim() == 1:
        # (Num_Paulis,) @ (Num_Paulis, Num_Obs) -> (Num_Obs,)
        return torch.matmul(w, V0)

    # 2. [핵심] 배치 행렬 곱 수행
    # w: (Batch, Num_Paulis), V0: (Num_Paulis, Num_Observables)
    # 결과: (Batch, Num_Observables)
    # torch.mm은 2D @ 2D 연산을 지원하며, GPU에서 최적화된 커널을 사용합니다.
    res = torch.mm(w, V0)
    
    return res


def coeff_matrix_from_observables(
    *,
    index: Dict[Any, int],
    observables: Sequence,
    device: str,
    dtype: Any,
) -> Tensor:
    """Build dense V0 = (N, K) coefficient matrix from PauliSum-like observables."""

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    if len(observables) == 0:
        raise ValueError("observables must be non-empty")

    V0 = torch.zeros((len(index), len(observables)), dtype=dtype, device=device)
    for j, obs in enumerate(observables):
        for p, c in obs.terms.items():
            i = index.get(p)
            if i is None:
                continue
            if not isinstance(c, (int, float)):
                raise TypeError("All observable coefficients must be numeric (int/float) for tensor backend")
            V0[i, j] = float(c)
    return V0


__all__ = [
    "UnionBasis",
    "UnionBasisObservable",
    "build_union_basis",
    "build_union_observable",
    "propagate_union_basis_psum",
    "adjoint_weights_on_zero",
    "coeff_matrix_from_observables",
    "expvals_from_w_and_coeff_matrix",
]
