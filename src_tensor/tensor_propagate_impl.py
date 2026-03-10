"""Thin wrapper around the compiled tensor propagation backend.

This module intentionally contains *minimal* Python logic so the core
propagation engine can be distributed as a compiled extension.

Dev-only pure Python reference code lives outside the package in:
- dev_backend/tensor_propagate_impl_py.py
- dev_backend/tensor_propagate_impl_full_dev.py

Public API entrypoints remain in `src_tensor/tensor_propagate.py`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, TYPE_CHECKING, cast, List

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


def _destructive_cat(tensor_list: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors while freeing original tensors immediately to reduce peak memory.
    
    Equivalent to torch.cat(tensor_list, dim=dim) but avoids holding both input list
    and output tensor in memory simultaneously.
    """
    if not tensor_list:
        raise ValueError("Empty tensor list")
    
    # Calculate total size
    total_size = sum(t.shape[dim] for t in tensor_list)
    out_shape = list(tensor_list[0].shape)
    out_shape[dim] = total_size
    
    # Pre-allocate output tensor
    out = torch.empty(out_shape, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
    
    offset = 0
    for i in range(len(tensor_list)):
        t = tensor_list[i]
        size = t.shape[dim]
        # Slice assignment
        if dim == 0:
            out[offset:offset+size] = t
        elif dim == 1:
            out[:, offset:offset+size] = t
        else:
            # Fallback for other dims (rarely used here)
            idx = [slice(None)] * out.ndim
            idx[dim] = slice(offset, offset+size)
            out[tuple(idx)] = t
            
        offset += size
        tensor_list[i] = None  # Release reference immediately
        
    return out


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


def _make_empty_sparse(shape, device, dtype):
    return torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.int64, device=device),
        torch.zeros((0,), dtype=dtype, device=device),
        size=shape,
        device=device,
        dtype=dtype,
    )


def _empty_i64(device):
    return torch.zeros((0,), dtype=torch.int64, device=device)


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
    weight_x: float,
    weight_y: float,
    weight_z: float,
    **kwargs,
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
            float(weight_x),
            float(weight_y),
            float(weight_z),
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
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )

    if step_device is None:
        step_device = "cpu" if device != "cpu" else device

    n_out = int(new_x.shape[0])
    n_in = int(x_mask.shape[0])
    mat_const = _make_sparse(row, col, val, (n_out, n_in), step_device, t_dtype)
    mat_empty = _make_empty_sparse((n_out, n_in), step_device, t_dtype)

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
    weight_x: float,
    weight_y: float,
    weight_z: float,
    **kwargs,
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
        if hasattr(_CPP_BACKEND, "build_pauli_rotation_step_implicit_mw_cpp"):
            gx_words, gz_words = _gate_masks_words(gate, n_words=int(x_mask.shape[1]), device=x_mask.device)
            (
                new_x,
                new_z,
                same_cols,
                anti_same_pos,
                r2,
                c2,
                v2,
                coeffs_cache_out,
            ) = _CPP_BACKEND.build_pauli_rotation_step_implicit_mw_cpp(
                gx_words,
                gz_words,
                backend_p_idx,
                x_mask,
                z_mask,
                t_dtype,
                min_abs,
                coeffs_cache,
                thetas_t,
                int(max_weight),
                float(weight_x),
                float(weight_y),
                float(weight_z),
            )

            if step_device is None:
                step_device = "cpu" if device != "cpu" else device

            n_out = int(new_x.shape[0])
            n_in = int(x_mask.shape[0])
            step = TensorSparseStep(
                mat_const=_make_empty_sparse((n_out, n_in), step_device, t_dtype),
                mat_cos=_make_empty_sparse((n_out, n_in), step_device, t_dtype),
                mat_sin=_make_sparse(r2, c2, v2, (n_out, n_in), step_device, t_dtype),
                param_idx=p_idx,
                shape=(n_out, n_in),
                emb_idx=e_idx,
                same_cols=same_cols.to(step_device),
                anti_same_pos=anti_same_pos.to(step_device),
            )

            out_cache: Optional[Tensor] = None
            if min_abs is not None and coeffs_cache_out is not None:
                out_cache = cast(Tensor, coeffs_cache_out)
            return new_x, new_z, step, out_cache

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
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )
    else:
        if hasattr(_CPP_BACKEND, "build_pauli_rotation_step_implicit_cpp"):
            gx, gz = _gate_masks(gate)
            (
                new_x,
                new_z,
                same_cols,
                anti_same_pos,
                r2,
                c2,
                v2,
                coeffs_cache_out,
            ) = _CPP_BACKEND.build_pauli_rotation_step_implicit_cpp(
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
                float(weight_x),
                float(weight_y),
                float(weight_z),
            )

            if step_device is None:
                step_device = "cpu" if device != "cpu" else device

            n_out = int(new_x.shape[0])
            n_in = int(x_mask.shape[0])
            step = TensorSparseStep(
                mat_const=_make_empty_sparse((n_out, n_in), step_device, t_dtype),
                mat_cos=_make_empty_sparse((n_out, n_in), step_device, t_dtype),
                mat_sin=_make_sparse(r2, c2, v2, (n_out, n_in), step_device, t_dtype),
                param_idx=p_idx,
                shape=(n_out, n_in),
                emb_idx=e_idx,
                same_cols=same_cols.to(step_device),
                anti_same_pos=anti_same_pos.to(step_device),
            )

            out_cache: Optional[Tensor] = None
            if min_abs is not None and coeffs_cache_out is not None:
                out_cache = cast(Tensor, coeffs_cache_out)
            return new_x, new_z, step, out_cache

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
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )

    if step_device is None:
        step_device = "cpu" if device != "cpu" else device

    work_device = new_x.device
    n_out_backend = int(new_x.shape[0])
    n_in = int(x_mask.shape[0])

    c0_w = c0.to(work_device, dtype=torch.int64)
    c1_w = c1.to(work_device, dtype=torch.int64)
    c2_w = c2.to(work_device, dtype=torch.int64)
    r0_w = r0.to(work_device, dtype=torch.int64)
    r1_w = r1.to(work_device, dtype=torch.int64)
    r2_w = r2.to(work_device, dtype=torch.int64)
    v2_w = v2.to(work_device)

    same_backend_mask = torch.zeros((n_out_backend,), dtype=torch.bool, device=work_device)
    if r0_w.numel() > 0:
        same_backend_mask[r0_w] = True
    if r1_w.numel() > 0:
        same_backend_mask[r1_w] = True

    same_backend_rows = torch.nonzero(same_backend_mask, as_tuple=False).flatten().to(torch.int64)
    n_same = int(same_backend_rows.numel())
    backend_to_same_pos = torch.full((n_out_backend,), -1, dtype=torch.int64, device=work_device)
    if n_same > 0:
        backend_to_same_pos[same_backend_rows] = torch.arange(n_same, dtype=torch.int64, device=work_device)

    same_cols = torch.full((n_same,), -1, dtype=torch.int64, device=work_device)
    if c0_w.numel() > 0:
        pos0 = backend_to_same_pos.index_select(0, r0_w)
        same_cols[pos0] = c0_w
    if c1_w.numel() > 0:
        pos1 = backend_to_same_pos.index_select(0, r1_w)
        prev = same_cols.index_select(0, pos1)
        if bool(((prev >= 0) & (prev != c1_w)).any().item()):
            raise RuntimeError("Rotation same-branch row merged from multiple input cols; implicit one-to-one model violated.")
        same_cols[pos1] = c1_w
        anti_same_pos = pos1
    else:
        anti_same_pos = _empty_i64(work_device)

    if n_same > 0 and bool((same_cols < 0).any().item()):
        raise RuntimeError("Rotation same-branch reconstruction produced unassigned same rows.")

    backend_to_final = backend_to_same_pos.clone()

    novel_backend_rows = _empty_i64(work_device)
    sin_row_final = _empty_i64(work_device)
    if r2_w.numel() > 0:
        sin_row_final = backend_to_final.index_select(0, r2_w)
        novel_edge_mask = sin_row_final.lt(0)
        if bool(novel_edge_mask.any().item()):
            novel_row_mask = torch.zeros((n_out_backend,), dtype=torch.bool, device=work_device)
            novel_row_mask[r2_w[novel_edge_mask]] = True
            novel_row_mask &= backend_to_final.lt(0)
            novel_backend_rows = torch.nonzero(novel_row_mask, as_tuple=False).flatten().to(torch.int64)
            novel_pos_map = torch.full((n_out_backend,), -1, dtype=torch.int64, device=work_device)
            novel_pos_map[novel_backend_rows] = torch.arange(
                n_same,
                n_same + int(novel_backend_rows.numel()),
                dtype=torch.int64,
                device=work_device,
            )
            sin_row_final = torch.where(
                novel_edge_mask,
                novel_pos_map.index_select(0, r2_w),
                sin_row_final,
            )

    final_backend_rows = torch.cat([same_backend_rows, novel_backend_rows], dim=0)
    new_x = new_x.index_select(0, final_backend_rows) if final_backend_rows.numel() > 0 else new_x[:0]
    new_z = new_z.index_select(0, final_backend_rows) if final_backend_rows.numel() > 0 else new_z[:0]

    n_out = int(new_x.shape[0])
    mat_const = _make_empty_sparse((n_out, n_in), step_device, t_dtype)
    mat_cos = _make_empty_sparse((n_out, n_in), step_device, t_dtype)
    mat_sin = _make_sparse(sin_row_final, c2_w, v2_w, (n_out, n_in), step_device, t_dtype)

    # [중요] TensorSparseStep 생성 시, 2단계에서 정한 필드 순서 준수
    # shape (기본값 없음) -> emb_idx (기본값 있음) 순서
    step = TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_cos,
        mat_sin=mat_sin,
        param_idx=p_idx,
        shape=(n_out, n_in),
        emb_idx=e_idx,  # 새로 추가된 인덱스 저장
        same_cols=same_cols.to(step_device),
        anti_same_pos=anti_same_pos.to(step_device),
    )

    out_cache: Optional[Tensor] = None
    if min_abs is not None and coeffs_cache_out is not None:
        out_cache = cast(Tensor, coeffs_cache_out).index_select(0, final_backend_rows.to(cast(Tensor, coeffs_cache_out).device))

    return new_x, new_z, step, out_cache


def build_depolarizing_step(
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
    weight_x: float,
    weight_y: float,
    weight_z: float,
    **kwargs,
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor]]:
    _require_backend()

    qubit = int(gate.qubit)
    px = float(gate.px)
    py = float(gate.py)
    pz = float(gate.pz)

    is_multiword = x_mask.dim() == 2
    if is_multiword:
        if not hasattr(_CPP_BACKEND, "build_depolarizing_step_mw_cpp"):
            raise CPPBackendUnavailableError(
                "Compiled backend does not expose multiword depolarizing API."
            )
        new_x, new_z, row, col, val, coeffs_cache_out = _CPP_BACKEND.build_depolarizing_step_mw_cpp(
            qubit,
            px,
            py,
            pz,
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            int(max_weight),
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )
    else:
        new_x, new_z, row, col, val, coeffs_cache_out = _CPP_BACKEND.build_depolarizing_step_cpp(
            qubit,
            px,
            py,
            pz,
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            int(max_weight),
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )

    if step_device is None:
        step_device = "cpu" if device != "cpu" else device

    n_out = int(new_x.shape[0])
    n_in = int(x_mask.shape[0])
    mat_const = _make_sparse(row, col, val, (n_out, n_in), step_device, t_dtype)
    mat_empty = _make_empty_sparse((n_out, n_in), step_device, t_dtype)

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


def build_amplitude_damping_step(
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
    weight_x: float,
    weight_y: float,
    weight_z: float,
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor]]:
    _require_backend()

    qubit = int(gate.qubit)
    gamma = float(gate.gamma)

    is_multiword = x_mask.dim() == 2
    if is_multiword:
        if not hasattr(_CPP_BACKEND, "build_amplitude_damping_step_mw_cpp"):
            raise CPPBackendUnavailableError(
                "Compiled backend does not expose multiword amplitude-damping API."
            )
        new_x, new_z, row, col, val, coeffs_cache_out = _CPP_BACKEND.build_amplitude_damping_step_mw_cpp(
            qubit,
            gamma,
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            int(max_weight),
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )
    else:
        new_x, new_z, row, col, val, coeffs_cache_out = _CPP_BACKEND.build_amplitude_damping_step_cpp(
            qubit,
            gamma,
            x_mask,
            z_mask,
            t_dtype,
            min_abs,
            coeffs_cache,
            int(max_weight),
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )

    if step_device is None:
        step_device = "cpu" if device != "cpu" else device

    n_out = int(new_x.shape[0])
    n_in = int(x_mask.shape[0])
    mat_const = _make_sparse(row, col, val, (n_out, n_in), step_device, t_dtype)
    mat_empty = _make_empty_sparse((n_out, n_in), step_device, t_dtype)

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


def compact_sparse_step_chunked(
    step: TensorSparseStep,
    keep_row_mask: Tensor,
    keep_col_mask: Optional[Tensor],
    device: str,
    chunk_size: int = 1_000_000,
    direct_threshold: int = 20_000_000,
) -> TensorSparseStep:
    """Filter and remap a TensorSparseStep using GPU acceleration with minimal memory footprint.

    This function solves the OOM issue when filtering large sparse matrices by:
    1. Creating a remapping table (cumsum) on GPU (or CPU if too large, but usually GPU fits).
    2. Streaming indices of the sparse matrices in chunks to the GPU.
    3. Applying the mask and remapping indices on the GPU.
    4. Assembling the result back on the CPU.

    Args:
        step: The original step with potentially large sparse matrices (on CPU).
        keep_row_mask: Boolean tensor indicating which rows to keep (size: n_old_rows).
        keep_col_mask: Optional boolean tensor indicating which cols to keep (size: n_old_cols).
        device: Compute device (e.g., "cuda") for acceleration.
        chunk_size: Number of NNZ to process at a time.
        direct_threshold: If nnz < direct_threshold, process entirely on GPU for speed.

    Returns:
        A new TensorSparseStep with filtered rows and remapped indices.
    """
    if step.same_cols is not None:
        same_cols = step.same_cols
        anti_same_pos = step.anti_same_pos if step.anti_same_pos is not None else _empty_i64(same_cols.device)
        n_same_old = int(same_cols.numel())
        keep_row_same = keep_row_mask[:n_same_old].to(same_cols.device)
        keep_col_mask_dev = keep_col_mask.to(same_cols.device) if keep_col_mask is not None else None

        if keep_col_mask_dev is None:
            keep_col_mask_dev = torch.ones((int(step.shape[1]),), dtype=torch.bool, device=same_cols.device)

        col_cumsum = torch.cumsum(keep_col_mask_dev.to(torch.int64), dim=0) - 1
        kept_same_cols = same_cols[keep_row_same]
        new_same_cols = col_cumsum.index_select(0, kept_same_cols) if kept_same_cols.numel() > 0 else _empty_i64(same_cols.device)

        same_row_cumsum = torch.cumsum(keep_row_same.to(torch.int64), dim=0) - 1
        if anti_same_pos.numel() > 0:
            kept_anti_mask = keep_row_same.index_select(0, anti_same_pos)
            kept_anti_pos = anti_same_pos[kept_anti_mask]
            new_anti_same_pos = same_row_cumsum.index_select(0, kept_anti_pos) if kept_anti_pos.numel() > 0 else _empty_i64(same_cols.device)
        else:
            new_anti_same_pos = _empty_i64(same_cols.device)

        n_new_rows = int(keep_row_mask.sum().item())
        n_new_cols = int(keep_col_mask_dev.sum().item())

        def _process_matrix(mat: Tensor) -> Tensor:
            if mat._nnz() == 0:
                return torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
                    torch.zeros((0,), dtype=mat.dtype, device=mat.device),
                    size=(n_new_rows, n_new_cols),
                    device=mat.device,
                    dtype=mat.dtype,
                )

            return compact_sparse_step_chunked(
                TensorSparseStep(
                    mat_const=mat,
                    mat_cos=_make_empty_sparse(mat.shape, mat.device, mat.dtype),
                    mat_sin=_make_empty_sparse(mat.shape, mat.device, mat.dtype),
                    param_idx=step.param_idx,
                    emb_idx=step.emb_idx,
                    shape=step.shape,
                ),
                keep_row_mask=keep_row_mask,
                keep_col_mask=keep_col_mask,
                device=device,
                chunk_size=chunk_size,
            ).mat_const

        new_const = _process_matrix(step.mat_const)
        new_cos = _process_matrix(step.mat_cos)
        new_sin = _process_matrix(step.mat_sin)

        return TensorSparseStep(
            mat_const=new_const,
            mat_cos=new_cos,
            mat_sin=new_sin,
            param_idx=step.param_idx,
            emb_idx=step.emb_idx,
            shape=(n_new_rows, n_new_cols),
            same_cols=new_same_cols,
            anti_same_pos=new_anti_same_pos,
        )

    # 1. Prepare Remap Table
    # Calculate new row indices: cumsum of the mask gives the new index for kept rows.
    # We use -1 to indicate dropped rows.
    n_old_rows = keep_row_mask.shape[0]
    n_new_rows = int(keep_row_mask.sum().item())
    n_new_cols = int(keep_col_mask.sum().item()) if keep_col_mask is not None else int(step.shape[1])

    # Move mask to GPU for fast cumsum and lookup
    # Boolean mask for 100M rows is ~100MB, fits easily in VRAM.
    mask_gpu = keep_row_mask.to(device=device, dtype=torch.bool)
    
    # new_idx_map[old_idx] = new_idx (if kept)
    # We calculate cumsum on the mask (treated as int).
    # cumsum starts at 1 for the first kept element, so we subtract 1.
    # We only care about indices where mask is True.
    cumsum_gpu = torch.cumsum(mask_gpu.to(torch.int64), dim=0) - 1

    # Prepare col map if exists
    col_mask_gpu = None
    col_cumsum_gpu = None
    if keep_col_mask is not None:
        col_mask_gpu = keep_col_mask.to(device=device, dtype=torch.bool)
        col_cumsum_gpu = torch.cumsum(col_mask_gpu.to(torch.int64), dim=0) - 1
    
    # 2. Helper to process one matrix
    def _process_matrix(mat: Tensor) -> Tensor:
        nnz = mat._nnz()
        if nnz == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
                torch.zeros((0,), dtype=mat.dtype, device=mat.device),
                size=(n_new_rows, n_new_cols),
                device=mat.device,
                dtype=mat.dtype
            )

        # We iterate over the coalesced indices (assuming mat is coalesced or we coalesce it)
        if not mat.is_coalesced():
            mat = mat.coalesce()
        
        # [Strategy 1] Fast Path: Small enough to fit in GPU memory entirely
        # T4 16GB can comfortably handle ~20M-30M terms with full overhead.
        if nnz < direct_threshold:
            # Move everything to GPU
            mat_gpu = mat.to(device)
            indices = mat_gpu.indices()
            values = mat_gpu.values()
            
            row_indices = indices[0]
            keep = mask_gpu[row_indices]
            
            if col_mask_gpu is not None:
                keep = keep & col_mask_gpu[indices[1]]
            
            valid_indices = indices[:, keep]
            valid_values = values[keep]
            
            # Remap
            valid_indices[0] = cumsum_gpu[valid_indices[0]]
            if col_cumsum_gpu is not None:
                valid_indices[1] = col_cumsum_gpu[valid_indices[1]]
            
            return torch.sparse_coo_tensor(
                valid_indices.cpu(), # Move result back to storage device (CPU)
                valid_values.cpu(),
                size=(n_new_rows, n_new_cols),
                device=mat.device,
                dtype=mat.dtype
            )

        # [Strategy 2] Hybrid Path: Large matrix, process in chunks to avoid OOM
        # Indices -> GPU, Values -> CPU (Slicing)
        indices = mat.indices() # (2, NNZ)
        values = mat.values()   # (NNZ,)

        # [Optimization] Pin memory for faster H2D transfer
        # Only pin if tensor is large enough (> 1MB approx) to justify allocation overhead.
        if device.startswith("cuda") and not indices.is_pinned() and indices.numel() > 100_000:
            try:
                indices = indices.pin_memory()
            except Exception:
                pass # Fallback to pageable memory if RAM is full

        # [Optimization] Dynamic chunking based on available VRAM (bounded).
        current_chunk_size = chunk_size
        if device.startswith("cuda"):
            try:
                free_mem, _ = torch.cuda.mem_get_info(device)
                # Heuristic: ~128 bytes per NNZ for indices(16B) + remapped(16B) + masks + overhead
                # Reserve 500MB buffer for safety.
                safe_mem = max(0, free_mem - 500 * 1024 * 1024)
                estimated_capacity = int(safe_mem // 128)

                # Keep chunk size bounded: do not exceed user-configured chunk_size.
                current_chunk_size = max(100_000, min(int(chunk_size), max(1, estimated_capacity)))
            except Exception:
                pass # Fallback to default if mem_get_info fails
        
        new_indices_list = []
        new_values_list = []
        
        # Chunked processing
        for start in range(0, nnz, current_chunk_size):
            end = min(start + current_chunk_size, nnz)
            
            # Move chunk to GPU
            # indices_chunk: (2, chunk_len)
            indices_chunk = indices[:, start:end].to(device, non_blocking=True)
            # [Optimization] Values are NOT moved to GPU. 
            # We only need indices on GPU to decide what to keep.
            
            row_indices = indices_chunk[0]
            col_indices = indices_chunk[1]
            
            # Check which elements to keep based on row index
            # mask_gpu is (N_rows,), row_indices are in [0, N_rows)
            keep_chunk = mask_gpu[row_indices]

            # Check cols if needed
            if col_mask_gpu is not None:
                keep_col_chunk = col_mask_gpu[col_indices]
                keep_chunk = keep_chunk & keep_col_chunk
            
                        
            # Filter
            if not keep_chunk.any():
                del indices_chunk, keep_chunk, row_indices, col_indices
                continue
                
            valid_indices = indices_chunk[:, keep_chunk]
            
            # Remap row indices
            # valid_indices[0] contains old row indices.
            
            # Remap row indices
            # valid_indices[0] contains old row indices.
            # We replace them with new indices from cumsum_gpu.
            old_rows = valid_indices[0]
            new_rows = cumsum_gpu[old_rows]
            valid_indices[0] = new_rows
            
            # Remap col indices if needed
            if col_cumsum_gpu is not None:
                old_cols = valid_indices[1]
                new_cols = col_cumsum_gpu[old_cols]
                valid_indices[1] = new_cols

            
            # Move back to CPU to save VRAM
            new_indices_list.append(valid_indices.cpu())
            
            # [Optimization] Filter values on CPU using the mask computed on GPU
            # keep_chunk is on GPU, move to CPU to slice the CPU tensor 'values'
            keep_chunk_cpu = keep_chunk.cpu()
            values_slice = values[start:end]
            new_values_list.append(values_slice[keep_chunk_cpu])
            
            # Explicit delete to free GPU memory immediately
            del indices_chunk, keep_chunk, valid_indices, old_rows, new_rows, keep_chunk_cpu


        # Assemble result on CPU
        if not new_indices_list:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
                torch.zeros((0,), dtype=mat.dtype, device=mat.device),
                size=(n_new_rows, n_new_cols),
                device=mat.device,
                dtype=mat.dtype
            )
            
        final_indices = _destructive_cat(new_indices_list, dim=1)
        final_values = _destructive_cat(new_values_list, dim=0)
        
        return torch.sparse_coo_tensor(
            final_indices,
            final_values,
            size=(n_new_rows, n_new_cols),
            device=mat.device,
            dtype=mat.dtype
        )

    # 3. Process all matrices sequentially to minimize peak memory
    new_const = _process_matrix(step.mat_const)
    # Explicitly delete intermediate GPU tensors if any remain (handled in loop)
    new_cos = _process_matrix(step.mat_cos)
    new_sin = _process_matrix(step.mat_sin)

    # Clean up GPU map
    del mask_gpu, cumsum_gpu, col_mask_gpu, col_cumsum_gpu
    # torch.cuda.empty_cache() # Optional, depending on how tight memory is

    return TensorSparseStep(
        mat_const=new_const,
        mat_cos=new_cos,
        mat_sin=new_sin,
        param_idx=step.param_idx,
        emb_idx=step.emb_idx,
        shape=(n_new_rows, n_new_cols),
    )


__all__ = [
    "CPPBackendUnavailableError",
    "build_clifford_step",
    "build_pauli_rotation_step",
    "compact_sparse_step_chunked",
]
