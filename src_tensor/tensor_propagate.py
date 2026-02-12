"""Tensor-native surrogate propagation (GPU-first, WIP).

This module also provides a *back-propagating zero-filter* that prunes the
sparse-step graph for the common case where the initial state is |0..0>.

For <0|U† O U|0>, only diagonal output terms (I/Z-only, i.e. x_mask == 0)
contribute. `zero_filter_tensor_backprop` keeps only those output rows and
then walks steps backwards to remove any input columns that cannot reach them.
"""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING, Tuple, cast

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

from .tensor_types import TensorPauliSum, TensorSparseStep
from tqdm import tqdm

from .tensor_propagate_impl import (
    build_clifford_step,
    build_pauli_rotation_step,
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
    indices = torch.stack([row_idx, col_idx], dim=0)
    sp = torch.sparse_coo_tensor(indices, values, size=shape, device=device, dtype=dtype)
    # Memory-first default: avoid coalesce() (it can allocate large temporary buffers).
    return sp


def _xmask_is_zero_rows(x_mask: Tensor) -> Tensor:
    """Row mask for diagonal terms (x_mask == 0) supporting packed limbs."""

    if x_mask.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=x_mask.device)
    if x_mask.dim() == 1:
        return x_mask == 0
    return (x_mask == 0).all(dim=1)


def make_selection_step(row_mask: Tensor, n_in: int, device: str, dtype: Any) -> TensorSparseStep:
    """Append-only selection step: keeps only rows where row_mask is True."""

    col_idx = torch.nonzero(row_mask, as_tuple=False).flatten().to(torch.int64)
    row_idx = torch.arange(col_idx.numel(), dtype=torch.int64, device=col_idx.device)
    val = torch.ones((row_idx.numel(),), dtype=dtype, device=col_idx.device)

    mat_const = _make_sparse(row_idx, col_idx, val, (int(row_idx.numel()), int(n_in)), col_idx.device, dtype)
    mat_empty = _make_sparse(
        torch.zeros((0,), dtype=torch.int64, device=col_idx.device),
        torch.zeros((0,), dtype=torch.int64, device=col_idx.device),
        torch.zeros((0,), dtype=dtype, device=col_idx.device),
        (int(row_idx.numel()), int(n_in)),
        col_idx.device,
        dtype,
    )

    return TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_empty,
        mat_sin=mat_empty,
        param_idx=-1,
        shape=(int(row_idx.numel()), int(n_in)),
    )


def _sparse_used_cols(mat: Tensor, row_mask: Tensor, n_cols: int) -> Tensor:
    """Given a sparse COO mat (n_out,n_in), return which input cols connect to kept rows."""

    if mat._nnz() == 0 or int(row_mask.sum().item()) == 0:
        return torch.zeros((int(n_cols),), dtype=torch.bool, device=mat.device)

    # Need indices; some sparse tensors require coalesce to expose them.
    try:
        row_idx, col_idx = mat.indices()
    except Exception:
        mat = mat.coalesce()
        row_idx, col_idx = mat.indices()

    keep = row_mask.to(mat.device)[row_idx]
    if not bool(keep.any().item()):
        return torch.zeros((int(n_cols),), dtype=torch.bool, device=mat.device)

    used = torch.zeros((int(n_cols),), dtype=torch.bool, device=mat.device)
    used[col_idx[keep]] = True
    return used


def _filter_sparse_rows_cols(mat: Tensor, row_mask: Tensor, col_mask: Tensor) -> Tensor:
    """Filter a sparse COO matrix to only masked rows/cols, remapping indices."""

    new_shape = (int(row_mask.sum().item()), int(col_mask.sum().item()))
    if mat._nnz() == 0 or new_shape[0] == 0 or new_shape[1] == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
            torch.zeros((0,), dtype=mat.dtype, device=mat.device),
            size=new_shape,
            device=mat.device,
            dtype=mat.dtype,
        )

    try:
        row_idx, col_idx = mat.indices()
        val = mat.values()
    except Exception:
        mat2 = mat.coalesce()
        row_idx, col_idx = mat2.indices()
        val = mat2.values()

    row_mask_d = row_mask.to(mat.device)
    col_mask_d = col_mask.to(mat.device)
    keep = row_mask_d[row_idx] & col_mask_d[col_idx]
    row_idx = row_idx[keep]
    col_idx = col_idx[keep]
    val = val[keep]

    if row_idx.numel() == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
            torch.zeros((0,), dtype=mat.dtype, device=mat.device),
            size=new_shape,
            device=mat.device,
            dtype=mat.dtype,
        )

    row_map = torch.cumsum(row_mask_d.to(torch.int64), dim=0) - 1
    col_map = torch.cumsum(col_mask_d.to(torch.int64), dim=0) - 1
    row_idx = row_map[row_idx]
    col_idx = col_map[col_idx]

    return torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx], dim=0),
        val,
        size=new_shape,
        device=mat.device,
        dtype=mat.dtype,
    )


def _filter_step_rows_cols(step: TensorSparseStep, row_mask: Tensor, col_mask: Tensor) -> TensorSparseStep:
    new_const = _filter_sparse_rows_cols(step.mat_const, row_mask, col_mask)
    new_cos = _filter_sparse_rows_cols(step.mat_cos, row_mask, col_mask)
    new_sin = _filter_sparse_rows_cols(step.mat_sin, row_mask, col_mask)
    return TensorSparseStep(
        mat_const=new_const,
        mat_cos=new_cos,
        mat_sin=new_sin,
        param_idx=step.param_idx,
        shape=(int(row_mask.sum().item()), int(col_mask.sum().item())),
    )

def _step_to_device(step: TensorSparseStep, device: str) -> TensorSparseStep:
    return TensorSparseStep(
        mat_const=step.mat_const.to(device),
        mat_cos=step.mat_cos.to(device),
        mat_sin=step.mat_sin.to(device),
        param_idx=step.param_idx,
        shape=step.shape,
    )

def _prune_sparse_by_abs(mat: Tensor, min_mat_abs: float) -> Tensor:
    if min_mat_abs <= 0.0 or mat._nnz() == 0:
        return mat
    mat_c = mat.coalesce()
    row_idx, col_idx = mat_c.indices()
    val = mat_c.values()
    keep = torch.abs(val) >= float(min_mat_abs)
    if bool(torch.all(keep).item()):
        return mat_c
    if int(torch.count_nonzero(keep).item()) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
            torch.zeros((0,), dtype=mat.dtype, device=mat.device),
            size=mat.shape,
            device=mat.device,
            dtype=mat.dtype,
        )
    return torch.sparse_coo_tensor(
        torch.stack([row_idx[keep], col_idx[keep]], dim=0),
        val[keep],
        size=mat.shape,
        device=mat.device,
        dtype=mat.dtype,
    )


def _prune_step_by_abs(step: TensorSparseStep, min_mat_abs: float) -> TensorSparseStep:
    return TensorSparseStep(
        mat_const=_prune_sparse_by_abs(step.mat_const, min_mat_abs),
        mat_cos=_prune_sparse_by_abs(step.mat_cos, min_mat_abs),
        mat_sin=_prune_sparse_by_abs(step.mat_sin, min_mat_abs),
        param_idx=step.param_idx,
        shape=step.shape,
    )


def propagate_surrogate_tensor(
    circuit,
    observable,
    max_weight: int = 50,
    max_xy: int = 50,
    device: str = "cuda",
    dtype: str = "float32",
    thetas: Optional[Tensor] = None,
    min_abs: Optional[float] = None,
    min_mat_abs: Optional[float] = None,
    offload_steps: bool = True,
    offload_keep: int = 1,
    step_device: str = "cpu",
    debug_cuda_mem: bool = False,
    debug_every: int = 25,
) -> TensorPauliSum:
    """Tensor-native propagation to a tensor PauliSum.

    NOTE: For n_qubits > 63 this uses packed int64 limbs (63 bits/word).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor propagation.")
    if min_abs is not None and thetas is None:
        raise ValueError("min_abs requires thetas to be provided.")
    if min_mat_abs is not None and float(min_mat_abs) < 0.0:
        raise ValueError("min_mat_abs must be >= 0")

    # Debug flags are intentionally ignored in the memory-first default.
    _ = (debug_cuda_mem, debug_every)

    n_qubits = observable.n_qubits

    # Build initial tensors from observable
    pstrs = list(observable.terms.keys())
    coeffs = [float(observable.terms[p]) for p in pstrs]

    def _pack_masks_63(mask_int: int, n_words: int) -> List[int]:
        word_mask = (1 << 63) - 1
        return [int((mask_int >> (63 * i)) & word_mask) for i in range(n_words)]

    if n_qubits <= 63:
        x_mask = torch.as_tensor([p.x_mask for p in pstrs], dtype=torch.int64, device=device)
        z_mask = torch.as_tensor([p.z_mask for p in pstrs], dtype=torch.int64, device=device)
    else:
        n_words = (int(n_qubits) + 62) // 63
        x_words = [_pack_masks_63(int(p.x_mask), n_words) for p in pstrs]
        z_words = [_pack_masks_63(int(p.z_mask), n_words) for p in pstrs]
        x_mask = torch.as_tensor(x_words, dtype=torch.int64, device=device)
        z_mask = torch.as_tensor(z_words, dtype=torch.int64, device=device)

    t_dtype = torch.float64 if dtype == "float64" else torch.float32
    coeff_init = torch.as_tensor(coeffs, dtype=t_dtype, device=device)
    steps: List[TensorSparseStep] = []

    coeffs_cache: Optional[Tensor] = None
    thetas_t: Optional[Tensor] = None
    min_abs_internal: Optional[float] = min_abs

    if min_abs is not None:
        coeff_max = float(torch.max(torch.abs(coeff_init)).detach().cpu().item()) if coeff_init.numel() > 0 else 0.0
        coeff_scale = coeff_max if coeff_max > 0.0 else 1.0
        coeffs_cache = coeff_init / coeff_scale
        min_abs_internal = float(min_abs) / coeff_scale
        thetas_t = torch.as_tensor(thetas, dtype=coeff_init.dtype, device=device)
        if thetas_t.numel() == 0:
            thetas_t = torch.zeros(1, dtype=coeff_init.dtype, device=device)

    total_gates = len(circuit)
    for gate in tqdm(reversed(circuit), total=total_gates, desc="propagate", dynamic_ncols=True):
        gate_name = gate.__class__.__name__

        if gate_name == "CliffordGate":
            new_x, new_z, step, coeffs_cache = build_clifford_step(
                gate=gate,
                x_mask=x_mask,
                z_mask=z_mask,
                t_dtype=t_dtype,
                device=device,
                step_device=step_device,
                min_abs=min_abs_internal,
                coeffs_cache=coeffs_cache,
                max_weight=max_weight,
                max_xy=max_xy,
            )
            if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                step = _prune_step_by_abs(step, float(min_mat_abs))
            steps.append(step)
            if offload_steps and device != "cpu" and len(steps) > offload_keep:
                idx = len(steps) - offload_keep - 1
                if steps[idx].mat_const.device.type != "cpu":
                    steps[idx] = _step_to_device(steps[idx], "cpu")
            x_mask, z_mask = new_x, new_z
            continue

        if gate_name == "PauliRotation":
            new_x, new_z, step, coeffs_cache = build_pauli_rotation_step(
                gate=gate,
                x_mask=x_mask,
                z_mask=z_mask,
                t_dtype=t_dtype,
                device=device,
                step_device=step_device,
                min_abs=min_abs_internal,
                coeffs_cache=coeffs_cache,
                thetas_t=thetas_t,
                max_weight=max_weight,
                max_xy=max_xy,
            )
            if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                step = _prune_step_by_abs(step, float(min_mat_abs))
            steps.append(step)
            if offload_steps and device != "cpu" and len(steps) > offload_keep:
                idx = len(steps) - offload_keep - 1
                if steps[idx].mat_const.device.type != "cpu":
                    steps[idx] = _step_to_device(steps[idx], "cpu")
            x_mask, z_mask = new_x, new_z
            continue

        raise TypeError(
            f"Unsupported gate type in propagate_surrogate_tensor: {gate_name}. "
            "Expected CliffordGate or PauliRotation."
        )

    psum = TensorPauliSum(
        n_qubits=n_qubits,
        x_mask=x_mask,
        z_mask=z_mask,
        coeff_init=coeff_init,
        steps=steps,
    )

    return psum


def zero_filter_tensor(psum: TensorPauliSum) -> TensorPauliSum:
    """Forward zero-filter: keep only diagonal output rows (x_mask == 0).

    This does *not* prune input columns; for that, use `zero_filter_tensor_backprop`.
    """

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    if psum.x_mask.numel() == 0:
        return psum

    out_mask = _xmask_is_zero_rows(psum.x_mask)
    steps = list(psum.steps)
    if not bool(torch.all(out_mask).item()):
        steps.append(make_selection_step(out_mask, int(psum.x_mask.shape[0]), str(psum.x_mask.device), psum.coeff_init.dtype))

    return TensorPauliSum(
        n_qubits=psum.n_qubits,
        x_mask=psum.x_mask[out_mask],
        z_mask=psum.z_mask[out_mask],
        coeff_init=psum.coeff_init,
        steps=steps,
    )


def zero_filter_tensor_backprop_with_keep_mask(
    psum: TensorPauliSum,
    stream_device: Optional[str] = None,
    offload_back: Optional[bool] = None,
) -> Tuple[TensorPauliSum, Tensor]:
    """Back-propagating zero-filter.

    Returns:
      (filtered_psum, keep_mask_in)
    where keep_mask_in is a boolean mask over the *input coefficient dimension*
    after pruning.
    """

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    if psum.x_mask.numel() == 0:
        keep_mask = torch.zeros((int(psum.coeff_init.shape[0]),), dtype=torch.bool, device=psum.coeff_init.device)
        filtered = TensorPauliSum(
            n_qubits=psum.n_qubits,
            x_mask=psum.x_mask,
            z_mask=psum.z_mask,
            coeff_init=psum.coeff_init[keep_mask],
            steps=list(psum.steps),
        )
        return filtered, keep_mask

    out_mask = _xmask_is_zero_rows(psum.x_mask)
    steps = list(psum.steps)

    if stream_device is None:
        stream_device = psum.x_mask.device.type if psum.x_mask.device.type != "cpu" else None
    if offload_back is None:
        offload_back = stream_device is not None

    row_mask = out_mask
    for i in reversed(range(len(steps))):
        step = steps[i]
        if stream_device is not None and step.mat_const.device.type != stream_device:
            step = _step_to_device(step, stream_device)

        n_cols = int(step.shape[1])
        row_mask_device = row_mask.to(step.mat_const.device)
        col_mask = _sparse_used_cols(step.mat_const, row_mask_device, n_cols)
        col_mask |= _sparse_used_cols(step.mat_cos, row_mask_device, n_cols)
        col_mask |= _sparse_used_cols(step.mat_sin, row_mask_device, n_cols)

        steps[i] = _filter_step_rows_cols(step, row_mask_device, col_mask)
        row_mask = col_mask.to(out_mask.device)

        if stream_device is not None and offload_back:
            steps[i] = _step_to_device(steps[i], "cpu")

    coeff_init = psum.coeff_init[row_mask]

    filtered = TensorPauliSum(
        n_qubits=psum.n_qubits,
        x_mask=psum.x_mask[out_mask],
        z_mask=psum.z_mask[out_mask],
        coeff_init=coeff_init,
        steps=steps,
    )
    return filtered, row_mask


def zero_filter_tensor_backprop(
    psum: TensorPauliSum,
    stream_device: Optional[str] = None,
    offload_back: Optional[bool] = None,
) -> TensorPauliSum:
    """Back-propagating zero-filter (prune rows/cols through sparse steps)."""

    filtered, _keep = zero_filter_tensor_backprop_with_keep_mask(
        psum,
        stream_device=stream_device,
        offload_back=offload_back,
    )
    return filtered


__all__ = [
    "propagate_surrogate_tensor",
    "zero_filter_tensor",
    "zero_filter_tensor_backprop",
    "zero_filter_tensor_backprop_with_keep_mask",
]
