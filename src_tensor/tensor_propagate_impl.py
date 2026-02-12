"""Thin wrapper around the compiled tensor propagation backend.

This module intentionally contains *minimal* Python logic so the core
propagation engine can be distributed as a compiled extension.

Dev-only pure Python reference code lives outside the package in:
- dev_backend/tensor_propagate_impl_py.py
- dev_backend/tensor_propagate_impl_full_dev.py

Public API entrypoints remain in `src_tensor/tensor_propagate.py`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, TYPE_CHECKING, cast

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

torch = cast(Any, torch)

from .tensor_types import TensorSparseStep

_CPP_BACKEND: Any
try:
    # Prefer a locally-built backend (for dev / extended-qubit experiments).
    try:
        from . import _pps_tensor_backend_local as _CPP_BACKEND  # type: ignore
    except Exception:  # pragma: no cover
        from . import _pps_tensor_backend as _CPP_BACKEND  # type: ignore

    _CPP_AVAILABLE = True
except Exception:  # pragma: no cover
    _CPP_BACKEND = cast(Any, None)
    _CPP_AVAILABLE = False


class CPPBackendUnavailableError(RuntimeError):
    pass


def _require_backend() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor propagation.")
    if not _CPP_AVAILABLE:
        raise CPPBackendUnavailableError(
            "Compiled backend is not available. Install an official wheel/binary that includes "
            "`src_tensor._pps_tensor_backend` for your Python/PyTorch version."
        )


def _make_sparse(row_idx, col_idx, values, shape, device, dtype):
    if row_idx.numel() == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64, device=device),
            torch.zeros((0,), dtype=dtype, device=device),
            size=shape,
            device=device,
            dtype=dtype,
        )
    if str(row_idx.device) != str(torch.device(device)):
        row_idx = row_idx.to(device)
        col_idx = col_idx.to(device)
        values = values.to(device)
    indices = torch.stack([row_idx, col_idx], dim=0)
    return torch.sparse_coo_tensor(indices, values, size=shape, device=device, dtype=dtype)


def _gate_masks(gate) -> Tuple[int, int]:
    gx = 0
    gz = 0
    for q, p in zip(gate.qubits, gate.pauli):
        bit = 1 << int(q)
        if p == "X":
            gx |= bit
        elif p == "Z":
            gz |= bit
        elif p == "Y":
            gx |= bit
            gz |= bit
    return gx, gz


def _gate_masks_words(gate, *, n_words: int, device) -> Tuple[Tensor, Tensor]:
    """Return (gx_words, gz_words) packed into int64 limbs (63 bits/word)."""
    gx = torch.zeros((n_words,), dtype=torch.int64, device=device)
    gz = torch.zeros((n_words,), dtype=torch.int64, device=device)
    for q, p in zip(gate.qubits, gate.pauli):
        qi = int(q)
        w = qi // 63
        b = qi % 63
        if w < 0 or w >= n_words:
            raise ValueError("gate qubit index exceeds mask word dimension")
        bit = int(1) << b
        if p == "X":
            gx[w] |= bit
        elif p == "Z":
            gz[w] |= bit
        elif p == "Y":
            gx[w] |= bit
            gz[w] |= bit
    return gx, gz


def build_clifford_step(
    *,
    gate,
    x_mask: Tensor,
    z_mask: Tensor,
    t_dtype,
    device: str,
    step_device: Optional[str] = None,
    min_abs: Optional[float],
    coeffs_cache: Optional[Tensor],
    max_weight: int,
    max_xy: int,
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor]]:
    _require_backend()

    symbol = str(gate.symbol).upper()
    qubits = [int(q) for q in gate.qubits]
    is_multiword = x_mask.dim() == 2

    if is_multiword:
        if not hasattr(_CPP_BACKEND, "build_clifford_step_mw_cpp"):
            raise CPPBackendUnavailableError(
                "Compiled backend does not expose multiword APIs. Build/install a backend that includes "
                "`build_clifford_step_mw_cpp` (see src_tensor/cpp_backend/build_local_backend.py)."
            )
        new_x, new_z, row, col, val, coeffs_cache_out = _CPP_BACKEND.build_clifford_step_mw_cpp(
            symbol,
            qubits,
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            int(max_weight),
            int(max_xy),
        )
    else:
        new_x, new_z, row, col, val, coeffs_cache_out = _CPP_BACKEND.build_clifford_step_cpp(
            symbol,
            qubits,
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            int(max_weight),
            int(max_xy),
        )

    if step_device is None:
        step_device = "cpu" if device != "cpu" else device

    n_out = int(new_x.shape[0])
    n_in = int(x_mask.shape[0])
    mat_const = _make_sparse(row, col, val, (n_out, n_in), step_device, t_dtype)
    mat_empty = _make_sparse(
        torch.zeros(0, dtype=torch.int64, device=step_device),
        torch.zeros(0, dtype=torch.int64, device=step_device),
        torch.zeros(0, dtype=t_dtype, device=step_device),
        (n_out, n_in),
        step_device,
        t_dtype,
    )

    step = TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_empty,
        mat_sin=mat_empty,
        param_idx=-1,
        shape=(n_out, n_in),
    )

    out_cache: Optional[Tensor] = None
    if min_abs is not None and coeffs_cache_out is not None:
        out_cache = cast(Tensor, coeffs_cache_out)

    return new_x, new_z, step, out_cache


def build_pauli_rotation_step(
    *,
    gate,
    x_mask: Tensor,
    z_mask: Tensor,
    t_dtype,
    device: str,
    step_device: Optional[str] = None,
    min_abs: Optional[float],
    coeffs_cache: Optional[Tensor],
    thetas_t: Optional[Tensor],
    max_weight: int,
    max_xy: int,
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor]]:
    _require_backend()

    # [변경 사항] 게이트에서 두 종류의 인덱스를 모두 가져옵니다.
    p_idx = int(getattr(gate, "param_idx", -1))
    e_idx = int(getattr(gate, "embedding_idx", -1))

    # NOTE:
    # The compiled backend historically expects a non-negative parameter index
    # to build the rotation step (even when we later drive it via priors).
    # If we pass p_idx < 0 for an embedding gate, some backend builds will
    # silently produce an identity-like step (cos/sin parts missing), which
    # makes priors have no effect and yields incorrect expvals (e.g., always 1).
    #
    # We therefore pass a valid index to the backend when embedding_idx is set,
    # without changing the semantic meaning of step.param_idx used by evaluators.
    backend_p_idx = p_idx if p_idx >= 0 else (e_idx if e_idx >= 0 else -1)

    is_multiword = x_mask.dim() == 2
    if is_multiword:
        if not hasattr(_CPP_BACKEND, "build_pauli_rotation_step_mw_cpp"):
            raise CPPBackendUnavailableError(
                "Compiled backend does not expose multiword APIs."
            )
        gx_words, gz_words = _gate_masks_words(gate, n_words=int(x_mask.shape[1]), device=x_mask.device)
        (
            new_x,
            new_z,
            r0, c0, v0,
            r1, c1, v1,
            r2, c2, v2,
            coeffs_cache_out,
        ) = _CPP_BACKEND.build_pauli_rotation_step_mw_cpp(
            gx_words,
            gz_words,
            backend_p_idx,  # backend requires non-negative index for rotation steps
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            thetas_t,
            int(max_weight),
            int(max_xy),
        )
    else:
        gx, gz = _gate_masks(gate)
        (
            new_x,
            new_z,
            r0, c0, v0,
            r1, c1, v1,
            r2, c2, v2,
            coeffs_cache_out,
        ) = _CPP_BACKEND.build_pauli_rotation_step_cpp(
            int(gx),
            int(gz),
            backend_p_idx,
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            thetas_t,
            int(max_weight),
            int(max_xy),
        )

    if step_device is None:
        step_device = "cpu" if device != "cpu" else device

    n_out = int(new_x.shape[0])
    n_in = int(x_mask.shape[0])
    mat_const = _make_sparse(r0, c0, v0, (n_out, n_in), step_device, t_dtype)
    mat_cos = _make_sparse(r1, c1, v1, (n_out, n_in), step_device, t_dtype)
    mat_sin = _make_sparse(r2, c2, v2, (n_out, n_in), step_device, t_dtype)

    # [중요] TensorSparseStep 생성 시, 2단계에서 정한 필드 순서 준수
    # shape (기본값 없음) -> emb_idx (기본값 있음) 순서
    step = TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_cos,
        mat_sin=mat_sin,
        param_idx=p_idx,
        shape=(n_out, n_in),
        emb_idx=e_idx  # 새로 추가된 인덱스 저장
    )

    out_cache: Optional[Tensor] = None
    if min_abs is not None and coeffs_cache_out is not None:
        out_cache = cast(Tensor, coeffs_cache_out)

    return new_x, new_z, step, out_cache


__all__ = [
    "CPPBackendUnavailableError",
    "build_clifford_step",
    "build_pauli_rotation_step",
]
