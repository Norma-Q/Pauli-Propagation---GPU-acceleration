"""Tensor-native surrogate propagation (GPU-first, WIP).

This module also provides a *back-propagating zero-filter* that prunes the
sparse-step graph for the common case where the initial state is |0..0>.

For <0|U† O U|0>, only diagonal output terms (I/Z-only, i.e. x_mask == 0)
contribute. `zero_filter_tensor_backprop` keeps only those output rows and
then walks steps backwards to remove any input columns that cannot reach them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple, cast
import math
import os
import time
import torch.multiprocessing as mp

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
    build_depolarizing_step,
    build_amplitude_damping_step,
    compact_sparse_step_chunked,
)


_PARALLEL_USABLE_VRAM_RATIO = 0.875
_PARALLEL_TARGET_USAGE_RATIO = 0.50
_PARALLEL_BYTES_PER_TERM_EST = 120
_PARALLEL_SOFT_VRAM_GB = 9.5
_PARALLEL_HARD_VRAM_GB = 11.0
_PARALLEL_MIN_TERMS_SOFT = 10_000_000
_PARALLEL_MIN_TERMS_HARD = 5_000_000
_PARALLEL_CONSEC_STEPS_SOFT = 2


def _zero_filter_used_cols_mode() -> str:
    # Stability-first default: force used-cols accumulation on CPU.
    return "cpu"


def _zero_filter_compact_device(compute_device: str) -> str:
    mode = str(os.environ.get("PPS_ZERO_FILTER_COMPACT_DEVICE", "auto")).strip().lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return str(compute_device)
    return str(compute_device)


def _rotation_anti_merge_mode() -> str:
    # Default to "off": global canonicalize remains the authoritative dedup step,
    # and the extra anti-merge pass can dominate runtime on modest chunk counts.
    #
    # Keep env overrides for experiments / safety valves:
    # - off: skip the extra anti-merge pass and go straight to canonicalize.
    # - lookup_only: useful when chunk count is large and peak CPU RAM around
    #   canonicalize becomes the practical bottleneck.
    # - full: legacy/debug mode; usually slower than lookup_only.
    mode = str(os.environ.get("PPS_ROTATION_ANTI_MERGE_MODE", "off")).strip().lower()
    if mode in {"full", "lookup_only", "off"}:
        return mode
    return "off"


def _rotation_anti_merge_profile_enabled() -> bool:
    value = str(os.environ.get("PPS_ROTATION_ANTI_MERGE_PROFILE", "0")).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _rotation_canonicalize_keep_implicit_enabled() -> bool:
    # Default to keeping implicit rotation structure through canonicalize when
    # it is provably safe. This reduces unnecessary explicit materialization
    # after anti-merge is skipped, while the downstream fallback path below
    # still preserves correctness if the implicit layout cannot be kept.
    value = str(os.environ.get("PPS_ROTATION_CANONICALIZE_KEEP_IMPLICIT", "1")).strip().lower()
    return value not in {"", "0", "false", "no", "off"}

def _auto_parallel_threshold_from_device(compute_device: str) -> int:
    if (not torch.cuda.is_available()) or (not str(compute_device).startswith("cuda")):
        return 1_000_000

    try:
        if ":" in str(compute_device):
            dev_idx = int(str(compute_device).split(":", 1)[1])
        else:
            dev_idx = 0

        props = torch.cuda.get_device_properties(dev_idx)
        total_bytes = int(props.total_memory)
        usable_bytes = int(total_bytes * _PARALLEL_USABLE_VRAM_RATIO)
        budget_bytes = int(usable_bytes * _PARALLEL_TARGET_USAGE_RATIO)

        threshold = int(budget_bytes // _PARALLEL_BYTES_PER_TERM_EST)
        return max(1_000_000, threshold)
    except Exception:
        return 1_000_000


def _current_vram_allocated_gb(device: str) -> float:
    if (not torch.cuda.is_available()) or (not str(device).startswith("cuda")):
        return 0.0
    try:
        return float(torch.cuda.memory_allocated(device) / (1024 ** 3))
    except Exception:
        return 0.0


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


def _parallel_propagate_worker(args: Tuple) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor], Dict[str, Any]]:
    (
        device, builder_fn, x_mask_chunk, z_mask_chunk,
        chunk_size, step_device, output_device, kwargs
    ) = args

    return _process_step_chunked(
        builder_fn,
        x_mask_chunk,
        z_mask_chunk,
        chunk_size,
        device,
        step_device,
        output_device,
        defer_postprocess=True,
        **kwargs,
    )


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


def _implicit_same_used_cols(step: TensorSparseStep, row_mask: Tensor, n_cols: int) -> Tensor:
    if step.same_cols is None or step.same_cols.numel() == 0 or int(row_mask.sum().item()) == 0:
        return torch.zeros((int(n_cols),), dtype=torch.bool, device=row_mask.device)

    n_same = int(step.same_cols.numel())
    keep_same = row_mask[:n_same]
    if not bool(keep_same.any().item()):
        return torch.zeros((int(n_cols),), dtype=torch.bool, device=row_mask.device)

    used = torch.zeros((int(n_cols),), dtype=torch.bool, device=row_mask.device)
    same_cols = step.same_cols.to(row_mask.device)
    keep = keep_same.to(row_mask.device)
    used[same_cols[keep]] = True
    return used


def _accumulate_used_cols_chunked(mat: Tensor, row_mask_d: Tensor, col_mask_d: Tensor, chunk_size: int) -> None:
    """Accumulate used columns from a CPU sparse matrix into a GPU mask using chunks."""
    if mat._nnz() == 0:
        return

    try:
        indices = mat.indices()
    except Exception:
        mat = mat.coalesce()
        indices = mat.indices()
    nnz = indices.shape[1]
    device = row_mask_d.device

    if str(device).startswith("cuda") and not indices.is_pinned() and indices.numel() > 100_000:
        try:
            indices = indices.pin_memory()
        except Exception:
            pass

    current_chunk_size = int(chunk_size)
    if str(device).startswith("cuda"):
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            safe_mem = max(0, free_mem - 500 * 1024 * 1024)
            estimated_capacity = int(max(1, safe_mem // 64))
            # NOTE: keep chunk bounded (old max() could overshoot badly)
            current_chunk_size = max(100_000, min(int(chunk_size), estimated_capacity))
        except Exception:
            pass

    for start in range(0, nnz, current_chunk_size):
        end = min(start + current_chunk_size, nnz)
        idx_chunk = indices[:, start:end].to(device, non_blocking=True)
        keep = row_mask_d[idx_chunk[0]]
        col_mask_d[idx_chunk[1][keep]] = True


def _accumulate_used_cols_same_chunked(
    same_cols: Optional[Tensor],
    row_mask_d: Tensor,
    col_mask_d: Tensor,
    chunk_size: int,
) -> None:
    if same_cols is None or same_cols.numel() == 0:
        return

    device = row_mask_d.device
    n_same = int(same_cols.numel())
    keep_same = row_mask_d[:n_same]
    if int(keep_same.sum().item()) == 0:
        return

    nnz = n_same
    current_chunk_size = int(chunk_size)
    if str(device).startswith("cuda"):
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            safe_mem = max(0, free_mem - 500 * 1024 * 1024)
            estimated_capacity = int(max(1, safe_mem // 32))
            current_chunk_size = max(100_000, min(int(chunk_size), estimated_capacity))
        except Exception:
            pass

    if str(device).startswith("cuda") and nnz > 100_000:
        try:
            if not same_cols.is_pinned():
                same_cols = same_cols.pin_memory()
        except Exception:
            pass

    for start in range(0, nnz, current_chunk_size):
        end = min(start + current_chunk_size, nnz)
        keep_chunk = keep_same[start:end]
        if not bool(keep_chunk.any().item()):
            continue
        col_chunk = same_cols[start:end].to(device, non_blocking=True)
        col_mask_d[col_chunk[keep_chunk]] = True


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
    same_cols = None
    anti_same_pos = None
    if step.same_cols is not None:
        same_cols_dev = step.same_cols
        keep_same = row_mask[: int(same_cols_dev.numel())].to(same_cols_dev.device)
        col_mask_dev = col_mask.to(same_cols_dev.device)
        col_map = torch.cumsum(col_mask_dev.to(torch.int64), dim=0) - 1
        kept_same_cols = same_cols_dev[keep_same]
        same_cols = col_map[kept_same_cols] if kept_same_cols.numel() > 0 else same_cols_dev[:0]

        anti_pos_old = step.anti_same_pos if step.anti_same_pos is not None else same_cols_dev[:0]
        if anti_pos_old.numel() > 0:
            row_map_same = torch.cumsum(keep_same.to(torch.int64), dim=0) - 1
            anti_keep = keep_same.index_select(0, anti_pos_old)
            kept_anti_old = anti_pos_old[anti_keep]
            anti_same_pos = row_map_same.index_select(0, kept_anti_old) if kept_anti_old.numel() > 0 else anti_pos_old[:0]
        else:
            anti_same_pos = same_cols_dev[:0]

    new_const = _filter_sparse_rows_cols(step.mat_const, row_mask, col_mask)
    new_cos = _filter_sparse_rows_cols(step.mat_cos, row_mask, col_mask)
    new_sin = _filter_sparse_rows_cols(step.mat_sin, row_mask, col_mask)
    return TensorSparseStep(
        mat_const=new_const,
        mat_cos=new_cos,
        mat_sin=new_sin,
        param_idx=step.param_idx,
        emb_idx=step.emb_idx,
        shape=(int(row_mask.sum().item()), int(col_mask.sum().item())),
        same_cols=same_cols,
        anti_same_pos=anti_same_pos,
    )

def _step_to_device(step: TensorSparseStep, device: str) -> TensorSparseStep:
    return TensorSparseStep(
        mat_const=step.mat_const.to(device),
        mat_cos=step.mat_cos.to(device),
        mat_sin=step.mat_sin.to(device),
        param_idx=step.param_idx,
        emb_idx=step.emb_idx,
        shape=step.shape,
        same_cols=None if step.same_cols is None else step.same_cols.to(device),
        anti_same_pos=None if step.anti_same_pos is None else step.anti_same_pos.to(device),
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
        emb_idx=step.emb_idx,
        shape=step.shape,
        same_cols=step.same_cols,
        anti_same_pos=step.anti_same_pos,
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
        
    return out


def _canonicalize_pauli_rows(x_mask: Tensor, z_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Canonicalize Pauli rows and return (unique_x, unique_z, inverse_map).

    `inverse_map` maps each old row index to the corresponding unique row index.
    """
    n_rows = int(x_mask.shape[0])
    if n_rows == 0:
        return x_mask, z_mask, torch.zeros((0,), dtype=torch.int64, device=x_mask.device)

    if x_mask.dim() != z_mask.dim():
        raise RuntimeError("x_mask/z_mask dim mismatch during row canonicalization")

    # Build row keys as [x | z] so duplicates across chunk boundaries can be merged.
    if x_mask.dim() == 1:
        keys = torch.stack([x_mask, z_mask], dim=1)
    else:
        keys = torch.cat([x_mask, z_mask], dim=1)

    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    if x_mask.dim() == 1:
        unique_x = unique_keys[:, 0]
        unique_z = unique_keys[:, 1]
    else:
        n_words = int(x_mask.shape[1])
        unique_x = unique_keys[:, :n_words]
        unique_z = unique_keys[:, n_words:]

    return unique_x, unique_z, inverse.to(torch.int64)


def _canonicalize_pauli_rows_stable(x_mask: Tensor, z_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Canonicalize rows while preserving first-occurrence order."""
    n_rows = int(x_mask.shape[0])
    if n_rows == 0:
        return x_mask, z_mask, torch.zeros((0,), dtype=torch.int64, device=x_mask.device)

    if x_mask.dim() != z_mask.dim():
        raise RuntimeError("x_mask/z_mask dim mismatch during stable row canonicalization")

    if x_mask.dim() == 1:
        keys = torch.stack([x_mask, z_mask], dim=1)
    else:
        keys = torch.cat([x_mask, z_mask], dim=1)

    unique_keys_sorted, inverse_sorted = torch.unique(keys, dim=0, return_inverse=True)
    n_unique = int(unique_keys_sorted.shape[0])
    if n_unique == n_rows:
        return x_mask, z_mask, torch.arange(n_rows, dtype=torch.int64, device=x_mask.device)

    perm = torch.argsort(inverse_sorted, stable=True)
    inverse_perm = inverse_sorted.index_select(0, perm)
    is_first = torch.ones((int(perm.numel()),), dtype=torch.bool, device=perm.device)
    if int(perm.numel()) > 1:
        is_first[1:] = inverse_perm[1:] != inverse_perm[:-1]

    first_pos_by_sorted_id = perm[is_first]
    sorted_id_by_first = inverse_perm[is_first]
    stable_order = torch.argsort(first_pos_by_sorted_id, stable=True)
    kept_rows = first_pos_by_sorted_id.index_select(0, stable_order)
    sorted_id_stable = sorted_id_by_first.index_select(0, stable_order)

    sorted_to_stable = torch.empty((n_unique,), dtype=torch.int64, device=inverse_sorted.device)
    sorted_to_stable[sorted_id_stable] = torch.arange(n_unique, dtype=torch.int64, device=inverse_sorted.device)
    row_inverse = sorted_to_stable.index_select(0, inverse_sorted)

    unique_keys = keys.index_select(0, kept_rows.to(keys.device))
    if x_mask.dim() == 1:
        unique_x = unique_keys[:, 0]
        unique_z = unique_keys[:, 1]
    else:
        n_words = int(x_mask.shape[1])
        unique_x = unique_keys[:, :n_words]
        unique_z = unique_keys[:, n_words:]

    return unique_x, unique_z, row_inverse


def _remap_sparse_rows(mat: Tensor, row_inverse: Tensor, new_n_rows: int) -> Tensor:
    """Remap sparse matrix rows by row_inverse and coalesce duplicates."""
    if mat._nnz() == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
            torch.zeros((0,), dtype=mat.dtype, device=mat.device),
            size=(int(new_n_rows), int(mat.shape[1])),
            device=mat.device,
            dtype=mat.dtype,
        )

    mat_c = mat.coalesce()
    idx = mat_c.indices()
    val = mat_c.values()

    row_inverse_dev = row_inverse.to(idx.device)
    new_row = row_inverse_dev.index_select(0, idx[0])
    new_idx = torch.stack([new_row, idx[1]], dim=0)

    remapped = torch.sparse_coo_tensor(
        new_idx,
        val,
        size=(int(new_n_rows), int(mat.shape[1])),
        device=mat.device,
        dtype=mat.dtype,
    )
    return remapped.coalesce()


def _merge_rotation_anti_rows_only(
    full_new_x: Tensor,
    full_new_z: Tensor,
    full_step: TensorSparseStep,
    full_cache: Optional[Tensor],
    profile_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor]]:
    """Merge only anti-commute-expanded rows for implicit Pauli-rotation steps.

    This keeps same-branch rows fixed and only deduplicates novel rows against
    anti-same rows. Legacy full dedup remains available via env toggle.
    """
    if full_step.same_cols is None:
        return full_new_x, full_new_z, full_step, full_cache

    anti_pos = full_step.anti_same_pos
    if anti_pos is None or anti_pos.numel() == 0:
        return full_new_x, full_new_z, full_step, full_cache

    n_rows = int(full_new_x.shape[0])
    n_cols = int(full_step.shape[1])
    n_same = int(full_step.same_cols.numel())
    if n_rows <= n_same:
        return full_new_x, full_new_z, full_step, full_cache

    merge_mode = _rotation_anti_merge_mode()
    if merge_mode == "off":
        if profile_entry is not None:
            anti_rows = int(anti_pos.numel())
            profile_entry["anti_merge_mode"] = "off"
            profile_entry["anti_merge_anti_same_rows"] = anti_rows
            profile_entry["anti_merge_novel_rows"] = int(n_rows - n_same)
            profile_entry["anti_merge_hits"] = 0
            profile_entry["anti_merge_kept_novel"] = int(n_rows - n_same)
            profile_entry["anti_merge_s"] = 0.0
            profile_entry["anti_merge_remap_s"] = 0.0
            profile_entry["anti_merge_changed"] = False
        return full_new_x, full_new_z, full_step, full_cache

    profile_enabled = _rotation_anti_merge_profile_enabled()
    collect_stats = profile_enabled or (profile_entry is not None)
    merge_t0 = time.perf_counter() if collect_stats else 0.0

    anti_pos_cpu = anti_pos.to(dtype=torch.int64, device="cpu")
    x_cpu = full_new_x if str(full_new_x.device) == "cpu" else full_new_x.to("cpu")
    z_cpu = full_new_z if str(full_new_z.device) == "cpu" else full_new_z.to("cpu")

    key_to_row = {}
    if x_cpu.dim() == 1:
        for r in anti_pos_cpu.tolist():
            key = (int(x_cpu[r].item()), int(z_cpu[r].item()))
            if key not in key_to_row:
                key_to_row[key] = int(r)
    else:
        for r in anti_pos_cpu.tolist():
            key = tuple(torch.cat([x_cpu[r], z_cpu[r]], dim=0).tolist())
            if key not in key_to_row:
                key_to_row[key] = int(r)

    row_map_cpu = torch.arange(n_rows, dtype=torch.int64, device="cpu")
    keep_novel_rows: List[int] = []
    next_row = int(n_same)
    merge_hits = 0
    anti_same_rows = int(anti_pos_cpu.numel())
    novel_rows = int(n_rows - n_same)

    if x_cpu.dim() == 1:
        for old_r in range(n_same, n_rows):
            key = (int(x_cpu[old_r].item()), int(z_cpu[old_r].item()))
            target = key_to_row.get(key)
            if target is None:
                row_map_cpu[old_r] = next_row
                keep_novel_rows.append(old_r)
                if merge_mode == "full":
                    key_to_row[key] = next_row
                next_row += 1
            else:
                row_map_cpu[old_r] = int(target)
                merge_hits += 1
    else:
        for old_r in range(n_same, n_rows):
            key = tuple(torch.cat([x_cpu[old_r], z_cpu[old_r]], dim=0).tolist())
            target = key_to_row.get(key)
            if target is None:
                row_map_cpu[old_r] = next_row
                keep_novel_rows.append(old_r)
                if merge_mode == "full":
                    key_to_row[key] = next_row
                next_row += 1
            else:
                row_map_cpu[old_r] = int(target)
                merge_hits += 1

    new_n_rows = int(next_row)
    merge_time_s = (time.perf_counter() - merge_t0) if collect_stats else 0.0
    if profile_entry is not None:
        profile_entry["anti_merge_mode"] = merge_mode
        profile_entry["anti_merge_anti_same_rows"] = anti_same_rows
        profile_entry["anti_merge_novel_rows"] = novel_rows
        profile_entry["anti_merge_hits"] = int(merge_hits)
        profile_entry["anti_merge_kept_novel"] = int(len(keep_novel_rows))
        profile_entry["anti_merge_s"] = float(merge_time_s)
        profile_entry["anti_merge_changed"] = bool(new_n_rows != n_rows)
    if new_n_rows == n_rows:
        if profile_enabled:
            print(
                "[PPS AntiMerge] "
                f"mode={merge_mode} anti_same_rows={anti_same_rows} novel_rows={novel_rows} "
                f"merge_hits={merge_hits} kept_novel={len(keep_novel_rows)} "
                f"merge_time_s={merge_time_s:.6f} remap_time_s=0.000000 changed=0"
            )
        return full_new_x, full_new_z, full_step, full_cache

    keep_rows_cpu = torch.cat(
        [
            torch.arange(n_same, dtype=torch.int64, device="cpu"),
            torch.as_tensor(keep_novel_rows, dtype=torch.int64, device="cpu"),
        ],
        dim=0,
    )
    keep_rows_dev = keep_rows_cpu.to(full_new_x.device)
    remap_t0 = time.perf_counter() if collect_stats else 0.0

    merged_x = full_new_x.index_select(0, keep_rows_dev)
    merged_z = full_new_z.index_select(0, keep_rows_dev)

    def _remap_sparse_rows_by_map(mat: Tensor, row_map: Tensor, out_rows: int, out_cols: int) -> Tensor:
        if mat._nnz() == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.int64, device=mat.device),
                torch.zeros((0,), dtype=mat.dtype, device=mat.device),
                size=(int(out_rows), int(out_cols)),
                device=mat.device,
                dtype=mat.dtype,
            )
        mat_c = mat if mat.is_coalesced() else mat.coalesce()
        idx = mat_c.indices()
        val = mat_c.values()
        row_map_dev = row_map.to(idx.device)
        new_rows = row_map_dev.index_select(0, idx[0])
        new_idx = torch.stack([new_rows, idx[1]], dim=0)
        return torch.sparse_coo_tensor(
            new_idx,
            val,
            size=(int(out_rows), int(out_cols)),
            device=mat.device,
            dtype=mat.dtype,
        ).coalesce()

    merged_step = TensorSparseStep(
        mat_const=_remap_sparse_rows_by_map(full_step.mat_const, row_map_cpu, new_n_rows, n_cols),
        mat_cos=_remap_sparse_rows_by_map(full_step.mat_cos, row_map_cpu, new_n_rows, n_cols),
        mat_sin=_remap_sparse_rows_by_map(full_step.mat_sin, row_map_cpu, new_n_rows, n_cols),
        param_idx=full_step.param_idx,
        emb_idx=full_step.emb_idx,
        shape=(new_n_rows, n_cols),
        same_cols=full_step.same_cols,
        anti_same_pos=full_step.anti_same_pos,
    )

    merged_cache = full_cache
    if full_cache is not None:
        row_map_cache = row_map_cpu.to(full_cache.device)
        merged_cache = torch.zeros((new_n_rows,), dtype=full_cache.dtype, device=full_cache.device)
        merged_cache.index_add_(0, row_map_cache, full_cache)

    remap_time_s = (time.perf_counter() - remap_t0) if collect_stats else 0.0
    if profile_entry is not None:
        profile_entry["anti_merge_remap_s"] = float(remap_time_s)
    if profile_enabled:
        print(
            "[PPS AntiMerge] "
            f"mode={merge_mode} anti_same_rows={anti_same_rows} novel_rows={novel_rows} "
            f"merge_hits={merge_hits} kept_novel={len(keep_novel_rows)} "
            f"merge_time_s={merge_time_s:.6f} remap_time_s={remap_time_s:.6f} changed=1"
        )

    return merged_x, merged_z, merged_step, merged_cache


def _validate_sparse_indices_bounds(mat: Tensor, shape: Tuple[int, int], name: str) -> None:
    """Fail fast if sparse indices are outside declared shape bounds."""
    if int(mat.shape[0]) != int(shape[0]) or int(mat.shape[1]) != int(shape[1]):
        raise RuntimeError(
            f"{name} shape mismatch: tensor shape={tuple(mat.shape)} declared shape={tuple(shape)}"
        )
    if mat._nnz() == 0:
        return
    mat_c = mat if mat.is_coalesced() else mat.coalesce()
    idx = mat_c.indices()
    rows = idx[0]
    cols = idx[1]
    max_row = int(rows.max().item())
    max_col = int(cols.max().item())
    if max_row >= int(shape[0]) or max_col >= int(shape[1]):
        raise RuntimeError(
            f"{name} index out of bounds: max_row={max_row} max_col={max_col} shape={tuple(shape)}"
        )


def _validate_assembled_step(
    *,
    full_new_x: Tensor,
    full_new_z: Tensor,
    full_step: TensorSparseStep,
    full_cache: Optional[Tensor],
    expected_input_cols: int,
    stage: str,
) -> None:
    """Validate chunk/parallel-assembled outputs to avoid silent merge/index corruption."""
    n_rows = int(full_step.shape[0])
    n_cols = int(full_step.shape[1])

    if int(full_new_x.shape[0]) != n_rows or int(full_new_z.shape[0]) != n_rows:
        raise RuntimeError(
            f"[{stage}] row mismatch: step_rows={n_rows} x_rows={int(full_new_x.shape[0])} z_rows={int(full_new_z.shape[0])}"
        )
    if full_new_x.shape != full_new_z.shape:
        raise RuntimeError(
            f"[{stage}] x/z shape mismatch: x_shape={tuple(full_new_x.shape)} z_shape={tuple(full_new_z.shape)}"
        )
    if n_cols != int(expected_input_cols):
        raise RuntimeError(
            f"[{stage}] input-col mismatch: step_cols={n_cols} expected_cols={int(expected_input_cols)}"
        )

    if full_cache is not None and int(full_cache.shape[0]) != n_rows:
        raise RuntimeError(
            f"[{stage}] coeff cache row mismatch: cache_rows={int(full_cache.shape[0])} step_rows={n_rows}"
        )

    if full_step.same_cols is not None:
        n_same = int(full_step.same_cols.numel())
        if n_same > n_rows:
            raise RuntimeError(f"[{stage}] same_cols size exceeds row count: n_same={n_same} n_rows={n_rows}")
        if n_same > 0:
            same_cols = full_step.same_cols
            max_same_col = int(same_cols.max().item())
            if max_same_col >= n_cols:
                raise RuntimeError(
                    f"[{stage}] same_cols out of bounds: max_col={max_same_col} step_cols={n_cols}"
                )
        anti = full_step.anti_same_pos
        if anti is not None and anti.numel() > 0:
            max_anti = int(anti.max().item())
            if max_anti >= n_same:
                raise RuntimeError(
                    f"[{stage}] anti_same_pos out of bounds: max_pos={max_anti} n_same={n_same}"
                )

    declared_shape = (n_rows, n_cols)
    _validate_sparse_indices_bounds(full_step.mat_const, declared_shape, f"[{stage}] mat_const")
    _validate_sparse_indices_bounds(full_step.mat_cos, declared_shape, f"[{stage}] mat_cos")
    _validate_sparse_indices_bounds(full_step.mat_sin, declared_shape, f"[{stage}] mat_sin")


def _implicit_step_to_explicit(step: TensorSparseStep) -> TensorSparseStep:
    """Materialize implicit same-branch representation into explicit const/cos matrices."""
    if step.same_cols is None:
        return step

    n_out, n_in = int(step.shape[0]), int(step.shape[1])
    device = step.mat_sin.device
    dtype = step.mat_sin.dtype

    same_cols = step.same_cols.to(device)
    n_same = int(same_cols.numel())

    anti_mask = torch.zeros((n_same,), dtype=torch.bool, device=device)
    if step.anti_same_pos is not None and step.anti_same_pos.numel() > 0:
        anti_pos = step.anti_same_pos.to(device)
        anti_mask[anti_pos] = True

    rows = torch.arange(n_same, dtype=torch.int64, device=device)
    ones = torch.ones((n_same,), dtype=dtype, device=device)

    const_rows = rows[~anti_mask]
    const_cols = same_cols[~anti_mask]
    const_vals = ones[~anti_mask]

    cos_rows = rows[anti_mask]
    cos_cols = same_cols[anti_mask]
    cos_vals = ones[anti_mask]

    implicit_const = _make_sparse(const_rows, const_cols, const_vals, (n_out, n_in), device, dtype)
    implicit_cos = _make_sparse(cos_rows, cos_cols, cos_vals, (n_out, n_in), device, dtype)

    mat_const = (step.mat_const + implicit_const).coalesce() if step.mat_const._nnz() > 0 else implicit_const
    mat_cos = (step.mat_cos + implicit_cos).coalesce() if step.mat_cos._nnz() > 0 else implicit_cos

    return TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_cos,
        mat_sin=step.mat_sin,
        param_idx=step.param_idx,
        emb_idx=step.emb_idx,
        shape=step.shape,
        same_cols=None,
        anti_same_pos=None,
    )


def _canonicalize_rows_and_remap_step(
    full_new_x: Tensor,
    full_new_z: Tensor,
    full_step: TensorSparseStep,
    full_cache: Optional[Tensor],
    profile_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor]]:
    """Merge duplicate Pauli rows and remap sparse step/cache consistently."""
    t0 = time.perf_counter() if profile_entry is not None else 0.0
    rows_before = int(full_new_x.shape[0])

    if full_step.same_cols is not None and _rotation_canonicalize_keep_implicit_enabled():
        n_same = int(full_step.same_cols.numel())
        unique_x, unique_z, row_inverse = _canonicalize_pauli_rows_stable(full_new_x, full_new_z)
        n_unique_rows = int(unique_x.shape[0])
        if profile_entry is not None:
            profile_entry["canonicalize_rows_before"] = rows_before
            profile_entry["canonicalize_rows_after"] = n_unique_rows
            profile_entry["canonicalize_changed"] = bool(n_unique_rows != rows_before)
            profile_entry["canonicalize_keep_implicit"] = True

        if n_unique_rows == rows_before:
            if profile_entry is not None:
                profile_entry["canonicalize_s"] = time.perf_counter() - t0
                profile_entry["canonicalize_implicit_preserved"] = True
            return full_new_x, full_new_z, full_step, full_cache

        expected_same = torch.arange(n_same, dtype=torch.int64, device=row_inverse.device)
        same_identity_ok = bool(torch.equal(row_inverse[:n_same], expected_same))
        if same_identity_ok:
            remapped_step = TensorSparseStep(
                mat_const=_remap_sparse_rows(full_step.mat_const, row_inverse, n_unique_rows),
                mat_cos=_remap_sparse_rows(full_step.mat_cos, row_inverse, n_unique_rows),
                mat_sin=_remap_sparse_rows(full_step.mat_sin, row_inverse, n_unique_rows),
                param_idx=full_step.param_idx,
                emb_idx=full_step.emb_idx,
                shape=(n_unique_rows, int(full_step.shape[1])),
                same_cols=full_step.same_cols,
                anti_same_pos=full_step.anti_same_pos,
            )
            if full_cache is not None:
                cache_dev = full_cache.device
                row_inverse_dev = row_inverse.to(cache_dev)
                merged_cache = torch.zeros((n_unique_rows,), dtype=full_cache.dtype, device=cache_dev)
                merged_cache.index_add_(0, row_inverse_dev, full_cache)
                full_cache = merged_cache
            if profile_entry is not None:
                profile_entry["canonicalize_s"] = time.perf_counter() - t0
                profile_entry["canonicalize_implicit_preserved"] = True
            return unique_x, unique_z, remapped_step, full_cache

        if profile_entry is not None:
            profile_entry["canonicalize_implicit_preserved"] = False
            profile_entry["canonicalize_keep_implicit_fallback"] = True

    unique_x, unique_z, row_inverse = _canonicalize_pauli_rows(full_new_x, full_new_z)
    n_unique_rows = int(unique_x.shape[0])
    if profile_entry is not None:
        profile_entry["canonicalize_rows_before"] = rows_before
        profile_entry["canonicalize_rows_after"] = n_unique_rows
        profile_entry["canonicalize_changed"] = bool(n_unique_rows != rows_before)
        profile_entry.setdefault("canonicalize_keep_implicit", False)
        profile_entry.setdefault("canonicalize_implicit_preserved", False)
    if n_unique_rows == int(full_new_x.shape[0]):
        if profile_entry is not None:
            profile_entry["canonicalize_s"] = time.perf_counter() - t0
        return full_new_x, full_new_z, full_step, full_cache

    if full_step.same_cols is not None:
        full_step = _implicit_step_to_explicit(full_step)
    full_step = TensorSparseStep(
        mat_const=_remap_sparse_rows(full_step.mat_const, row_inverse, n_unique_rows),
        mat_cos=_remap_sparse_rows(full_step.mat_cos, row_inverse, n_unique_rows),
        mat_sin=_remap_sparse_rows(full_step.mat_sin, row_inverse, n_unique_rows),
        param_idx=full_step.param_idx,
        emb_idx=full_step.emb_idx,
        shape=(n_unique_rows, int(full_step.shape[1])),
        same_cols=None,
        anti_same_pos=None,
    )
    if full_cache is not None:
        cache_dev = full_cache.device
        row_inverse_dev = row_inverse.to(cache_dev)
        merged_cache = torch.zeros((n_unique_rows,), dtype=full_cache.dtype, device=cache_dev)
        merged_cache.index_add_(0, row_inverse_dev, full_cache)
        full_cache = merged_cache
    if profile_entry is not None:
        profile_entry["canonicalize_s"] = time.perf_counter() - t0
    return unique_x, unique_z, full_step, full_cache


def _process_step_chunked(
    builder_fn,
    x_mask_in: Tensor,
    z_mask_in: Tensor,
    chunk_size: int,
    device: str,
    step_device: str,
    output_device: str = "cpu",
    defer_postprocess: bool = False,
    **kwargs
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor], Dict[str, Any]]:
    """
    Process a propagation step in chunks to avoid GPU OOM.
    Keeps the main masks on CPU, moves chunks to GPU for processing,
    then assembles the result back on CPU (or step_device).
    """
    n_terms = x_mask_in.shape[0]
    profile_entry: Dict[str, Any] = {
        "assembly_mode": "direct" if n_terms <= chunk_size else "chunked",
        "n_chunks": 1 if n_terms <= chunk_size else int(math.ceil(n_terms / chunk_size)),
        "canonicalize_s": 0.0,
        "anti_merge_s": 0.0,
        "anti_merge_remap_s": 0.0,
    }

    # Fast path: if small enough, run directly
    if n_terms <= chunk_size:
        x_gpu = x_mask_in.to(device)
        z_gpu = z_mask_in.to(device)
        # Handle coeffs_cache if present
        if 'coeffs_cache' in kwargs and kwargs['coeffs_cache'] is not None:
            kwargs['coeffs_cache'] = kwargs['coeffs_cache'].to(device)
            
        new_x, new_z, step, cache = builder_fn(x_mask=x_gpu, z_mask=z_gpu, device=device, step_device=step_device, **kwargs)
        profile_entry["output_terms"] = int(new_x.shape[0])
        return new_x.to(output_device), new_z.to(output_device), step, (cache.to(output_device) if cache is not None else None), profile_entry

    # Chunked processing
    new_x_parts = []
    new_z_parts = []
    cache_parts = []
    same_x_parts = []
    same_z_parts = []
    novel_x_parts = []
    novel_z_parts = []
    same_cache_parts = []
    novel_cache_parts = []
    
    # Accumulators for sparse indices
    const_indices, const_values = [], []
    cos_indices, cos_values = [], []
    sin_indices, sin_values = [], []
    same_cols_parts = []
    anti_same_pos_parts = []
    implicit_sin_parts = []
    last_step: Optional[TensorSparseStep] = None
    implicit_mode: Optional[bool] = None
    
    row_offset = 0
    col_offset = 0
    same_row_offset = 0
    novel_row_offset = 0
    
    n_chunks = math.ceil(n_terms / chunk_size)
    shared_thetas_t = None
    if "thetas_t" in kwargs and kwargs["thetas_t"] is not None:
        theta_obj = kwargs["thetas_t"]
        if torch is not None and isinstance(theta_obj, torch.Tensor):
            shared_thetas_t = theta_obj.to(device)
        else:
            shared_thetas_t = theta_obj
    
    # Helper to offset and append sparse parts
    def _append_sparse(mat, indices_list, values_list, r_off, c_off):
        if mat._nnz() > 0:
            # Ensure we have indices (coalesce if needed)
            if not mat.is_coalesced():
                mat = mat.coalesce()
            idx = mat.indices().to("cpu")
            val = mat.values().to("cpu")
            # Apply offsets
            idx[0] += r_off
            idx[1] += c_off
            indices_list.append(idx)
            values_list.append(val)

    def _append_implicit_same(step, same_cols_list, anti_pos_list, r_off, c_off):
        if step.same_cols is None or step.same_cols.numel() == 0:
            return
        same_cols_list.append(step.same_cols.to("cpu") + int(c_off))
        anti_pos = step.anti_same_pos
        if anti_pos is not None and anti_pos.numel() > 0:
            anti_pos_list.append(anti_pos.to("cpu") + int(r_off))

    def _append_sparse_local(mat, indices_list, values_list, c_off, n_same_local, same_off, novel_off):
        if mat._nnz() == 0:
            return
        if not mat.is_coalesced():
            mat = mat.coalesce()
        idx = mat.indices().to("cpu")
        val = mat.values().to("cpu")
        indices_list.append((idx, val, int(c_off), int(n_same_local), int(same_off), int(novel_off)))
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_terms)
        
        # 1. Move chunk to GPU
        x_chunk = x_mask_in[start:end].to(device)
        z_chunk = z_mask_in[start:end].to(device)
        
        chunk_kwargs = kwargs.copy()
        if shared_thetas_t is not None:
            chunk_kwargs["thetas_t"] = shared_thetas_t
        if 'coeffs_cache' in chunk_kwargs and chunk_kwargs['coeffs_cache'] is not None:
            chunk_kwargs['coeffs_cache'] = chunk_kwargs['coeffs_cache'][start:end].to(device)
            
        nx, nz, step, cache = builder_fn(x_mask=x_chunk, z_mask=z_chunk, device=device, step_device=device, **chunk_kwargs)
        last_step = step

        step_is_implicit = step.same_cols is not None
        if implicit_mode is None:
            implicit_mode = bool(step_is_implicit)
        elif bool(implicit_mode) != bool(step_is_implicit):
            raise RuntimeError("Chunked step assembly cannot mix implicit and explicit step formats.")
        
        # 2. Collect Results (Move back to CPU immediately)
        nx_cpu = nx.to(output_device)
        nz_cpu = nz.to(output_device)
        cache_cpu = cache.to(output_device) if cache is not None else None

        if step_is_implicit:
            n_same_local = int(step.same_cols.numel())
            same_x_parts.append(nx_cpu[:n_same_local])
            same_z_parts.append(nz_cpu[:n_same_local])
            novel_x_parts.append(nx_cpu[n_same_local:])
            novel_z_parts.append(nz_cpu[n_same_local:])
            if cache_cpu is not None:
                same_cache_parts.append(cache_cpu[:n_same_local])
                novel_cache_parts.append(cache_cpu[n_same_local:])
        else:
            new_x_parts.append(nx_cpu)
            new_z_parts.append(nz_cpu)
            if cache_cpu is not None:
                cache_parts.append(cache_cpu)
        
        # 3. Accumulate Sparse Matrices
        if step_is_implicit:
            if step.mat_const._nnz() > 0 or step.mat_cos._nnz() > 0:
                raise RuntimeError("Implicit rotation step unexpectedly contains explicit const/cos matrices.")
            n_same_local = int(step.same_cols.numel())
            _append_sparse_local(
                step.mat_sin,
                implicit_sin_parts,
                sin_values,
                col_offset,
                n_same_local,
                same_row_offset,
                novel_row_offset,
            )
            _append_implicit_same(step, same_cols_parts, anti_same_pos_parts, same_row_offset, col_offset)
            same_row_offset += n_same_local
            novel_row_offset += int(nx.shape[0]) - n_same_local
        else:
            _append_sparse(step.mat_const, const_indices, const_values, row_offset, col_offset)
            _append_sparse(step.mat_cos, cos_indices, cos_values, row_offset, col_offset)
            _append_sparse(step.mat_sin, sin_indices, sin_values, row_offset, col_offset)
            row_offset += nx.shape[0]
        col_offset += (end - start)

    # 4. Assemble Final Tensors
    if bool(implicit_mode):
        x_parts = [*same_x_parts, *novel_x_parts]
        z_parts = [*same_z_parts, *novel_z_parts]
        full_new_x = _destructive_cat(x_parts, dim=0)
        full_new_z = _destructive_cat(z_parts, dim=0)
        full_cache = None
        if same_cache_parts or novel_cache_parts:
            cache_merge = [*same_cache_parts, *novel_cache_parts]
            full_cache = _destructive_cat(cache_merge, dim=0)
        same_x_parts = same_z_parts = novel_x_parts = novel_z_parts = None
        same_cache_parts = novel_cache_parts = None
    else:
        full_new_x = _destructive_cat(new_x_parts, dim=0)
        new_x_parts = None
        full_new_z = _destructive_cat(new_z_parts, dim=0)
        new_z_parts = None
        full_cache = _destructive_cat(cache_parts, dim=0) if cache_parts else None
        cache_parts = None
    
    def _concat_sparse(indices_list, values_list, shape):
        if not indices_list:
            return _make_sparse(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=kwargs['t_dtype']), shape, step_device, kwargs['t_dtype'])
        
        # [Memory Optimization] Clear lists immediately after cat to reduce peak memory
        all_indices = _destructive_cat(indices_list, dim=1).to(step_device)
        indices_list.clear()
        
        all_values = _destructive_cat(values_list, dim=0).to(step_device)
        values_list.clear()
        
        return torch.sparse_coo_tensor(all_indices, all_values, size=shape, device=step_device)

    full_shape = (same_row_offset + novel_row_offset, col_offset) if bool(implicit_mode) else (row_offset, col_offset)

    mat_const = _concat_sparse(const_indices, const_values, full_shape)
    const_indices, const_values = None, None

    mat_cos = _concat_sparse(cos_indices, cos_values, full_shape)
    cos_indices, cos_values = None, None

    if bool(implicit_mode):
        remapped_sin_indices = []
        remapped_sin_values = []
        total_same = int(same_row_offset)
        for idx, val, c_off, n_same_local, same_off, novel_off in implicit_sin_parts:
            local_idx = idx.clone()
            local_rows = local_idx[0]
            same_mask_local = local_rows < int(n_same_local)
            if bool(same_mask_local.any().item()):
                local_idx[0][same_mask_local] = local_rows[same_mask_local] + int(same_off)
            if bool((~same_mask_local).any().item()):
                local_idx[0][~same_mask_local] = (
                    (local_rows[~same_mask_local] - int(n_same_local)) + total_same + int(novel_off)
                )
            local_idx[1] += int(c_off)
            remapped_sin_indices.append(local_idx)
            remapped_sin_values.append(val)
        mat_sin = _concat_sparse(remapped_sin_indices, remapped_sin_values, full_shape)
        implicit_sin_parts = None
    else:
        mat_sin = _concat_sparse(sin_indices, sin_values, full_shape)
    sin_indices, sin_values = None, None

    full_same_cols = _destructive_cat(same_cols_parts, dim=0).to(step_device) if same_cols_parts else None
    full_anti_same_pos = _destructive_cat(anti_same_pos_parts, dim=0).to(step_device) if anti_same_pos_parts else None
    same_cols_parts, anti_same_pos_parts = None, None
    if last_step is None:
        raise RuntimeError("Chunked step assembly produced no chunks.")

    full_step = TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_cos,
        mat_sin=mat_sin,
        param_idx=last_step.param_idx,
        emb_idx=last_step.emb_idx,
        shape=full_shape,
        same_cols=full_same_cols,
        anti_same_pos=full_anti_same_pos,
    )

    if (not defer_postprocess) and getattr(builder_fn, "__name__", "") == "build_pauli_rotation_step" and full_step.same_cols is not None:
        full_new_x, full_new_z, full_step, full_cache = _merge_rotation_anti_rows_only(
            full_new_x,
            full_new_z,
            full_step,
            full_cache,
            profile_entry=profile_entry,
        )

    gate = kwargs.get("gate", None)
    gate_name = gate.__class__.__name__ if gate is not None else ""

    # Deduplicate expanding-gate rows at chunk boundaries and keep step/cache remapping consistent.
    if (not defer_postprocess) and gate_name != "CliffordGate":
        full_new_x, full_new_z, full_step, full_cache = _canonicalize_rows_and_remap_step(
            full_new_x,
            full_new_z,
            full_step,
            full_cache,
            profile_entry=profile_entry,
        )

    if defer_postprocess and profile_entry is not None:
        profile_entry["defer_postprocess"] = True

    _validate_assembled_step(
        full_new_x=full_new_x,
        full_new_z=full_new_z,
        full_step=full_step,
        full_cache=full_cache,
        expected_input_cols=n_terms,
        stage="chunked",
    )
    
    profile_entry["output_terms"] = int(full_new_x.shape[0])
    return full_new_x, full_new_z, full_step, full_cache, profile_entry


def _process_step_parallel(
    builder_fn,
    x_mask_in: Tensor,
    z_mask_in: Tensor,
    chunk_size: int,
    devices: List[str],
    step_device: str,
    output_device: str = "cpu",
    parallel_threshold: int = 1_000_000,
    pools: Optional[List[Any]] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor], Dict[str, Any]]:
    n_terms = int(x_mask_in.shape[0])
    n_gpus = len(devices)
    profile_entry: Dict[str, Any] = {
        "assembly_mode": "parallel",
        "n_chunks": int(n_gpus),
        "canonicalize_s": 0.0,
        "anti_merge_s": 0.0,
        "anti_merge_remap_s": 0.0,
    }

    if n_terms < int(parallel_threshold) or n_gpus < 2:

        return _process_step_chunked(
            builder_fn,
            x_mask_in,
            z_mask_in,
            chunk_size,
            devices[0],
            step_device,
            output_device,
            **kwargs,
        )

    chunk_bounds: List[Tuple[int, int]] = []
    base_chunk = n_terms // n_gpus
    remainder = n_terms % n_gpus
    start = 0
    for i in range(n_gpus):
        end = start + base_chunk + (1 if i < remainder else 0)
        chunk_bounds.append((start, end))
        start = end

    shared_thetas_t = kwargs.get("thetas_t", None)
    if torch is not None and isinstance(shared_thetas_t, torch.Tensor) and shared_thetas_t.device.type != "cpu":
        # Send a CPU copy through multiprocessing, then let each worker move it
        # once to its assigned GPU. Passing a cuda:0 tensor directly can cause
        # mixed-device backend calls on non-primary workers.
        shared_thetas_t = shared_thetas_t.cpu()
    worker_args = []
    for i in range(n_gpus):
        start, end = chunk_bounds[i]
        worker_kwargs = kwargs.copy()
        if shared_thetas_t is not None:
            worker_kwargs["thetas_t"] = shared_thetas_t
        if "coeffs_cache" in kwargs and kwargs["coeffs_cache"] is not None:
            worker_kwargs["coeffs_cache"] = kwargs["coeffs_cache"][start:end]
        worker_args.append(
            (
                devices[i],
                builder_fn,
                x_mask_in[start:end],
                z_mask_in[start:end],
                chunk_size,
                step_device,
                output_device,
                worker_kwargs,
            )
        )

    if pools is None:
        ctx = mp.get_context("spawn")
        local_pools = [ctx.Pool(processes=1) for _ in range(n_gpus)]
        try:
            async_results = [
                local_pools[i].apply_async(_parallel_propagate_worker, (worker_args[i],))
                for i in range(n_gpus)
            ]
            results = [res.get() for res in async_results]
        finally:
            for local_pool in local_pools:
                local_pool.close()
            for local_pool in local_pools:
                local_pool.join()
    else:
        async_results = [
            pools[i].apply_async(_parallel_propagate_worker, (worker_args[i],))
            for i in range(n_gpus)
        ]
        results = [res.get() for res in async_results]

    step_parts = [r[2] for r in results]
    if not step_parts:
        raise RuntimeError("Parallel step assembly produced no worker results.")

    implicit_flags = [s.same_cols is not None for s in step_parts]
    implicit_mode = bool(implicit_flags[0])
    if any(bool(flag) != implicit_mode for flag in implicit_flags[1:]):
        raise RuntimeError("Parallel step assembly cannot mix implicit and explicit step formats.")

    if implicit_mode:
        same_x_parts: List[Tensor] = []
        same_z_parts: List[Tensor] = []
        novel_x_parts: List[Tensor] = []
        novel_z_parts: List[Tensor] = []
        same_cache_parts: List[Tensor] = []
        novel_cache_parts: List[Tensor] = []
        same_cols_parts: List[Tensor] = []
        anti_same_pos_parts: List[Tensor] = []
        implicit_sin_parts = []
        same_row_offset = 0
        novel_row_offset = 0
        col_offset = 0

        def _append_implicit_same(step, same_cols_list, anti_pos_list, r_off, c_off):
            if step.same_cols is None or step.same_cols.numel() == 0:
                return
            same_cols_list.append(step.same_cols.to("cpu") + int(c_off))
            anti_pos = step.anti_same_pos
            if anti_pos is not None and anti_pos.numel() > 0:
                anti_pos_list.append(anti_pos.to("cpu") + int(r_off))

        def _append_sparse_local(mat, parts_list, c_off, n_same_local, same_off, novel_off):
            if mat._nnz() == 0:
                return
            if not mat.is_coalesced():
                mat = mat.coalesce()
            idx = mat.indices().to("cpu")
            val = mat.values().to("cpu")
            parts_list.append((idx, val, int(c_off), int(n_same_local), int(same_off), int(novel_off)))

        for nx, nz, step, cache, _worker_profile in results:
            n_same_local = int(step.same_cols.numel()) if step.same_cols is not None else 0
            same_x_parts.append(nx[:n_same_local])
            same_z_parts.append(nz[:n_same_local])
            novel_x_parts.append(nx[n_same_local:])
            novel_z_parts.append(nz[n_same_local:])
            if cache is not None:
                same_cache_parts.append(cache[:n_same_local])
                novel_cache_parts.append(cache[n_same_local:])
            _append_sparse_local(
                step.mat_sin,
                implicit_sin_parts,
                col_offset,
                n_same_local,
                same_row_offset,
                novel_row_offset,
            )
            _append_implicit_same(step, same_cols_parts, anti_same_pos_parts, same_row_offset, col_offset)
            same_row_offset += n_same_local
            novel_row_offset += int(nx.shape[0]) - n_same_local
            col_offset += int(step.shape[1])

        full_new_x = _destructive_cat([*same_x_parts, *novel_x_parts], dim=0)
        full_new_z = _destructive_cat([*same_z_parts, *novel_z_parts], dim=0)
        full_cache = None
        if same_cache_parts or novel_cache_parts:
            full_cache = _destructive_cat([*same_cache_parts, *novel_cache_parts], dim=0)
    else:
        new_x_parts = [r[0] for r in results]
        new_z_parts = [r[1] for r in results]
        cache_parts = [r[3] for r in results if r[3] is not None]
        full_new_x = _destructive_cat(new_x_parts, dim=0)
        full_new_z = _destructive_cat(new_z_parts, dim=0)
        full_cache = _destructive_cat(cache_parts, dim=0) if cache_parts else None

    gate = kwargs.get("gate", None)
    gate_name = gate.__class__.__name__ if gate is not None else ""
    builder_name = getattr(builder_fn, "__name__", "")
    parallel_canonicalize_done = False
    row_inverse_fast: Optional[Tensor] = None

    if (
        implicit_mode
        and gate_name != "CliffordGate"
        and builder_name == "build_pauli_rotation_step"
        and _rotation_anti_merge_mode() == "off"
        and _rotation_canonicalize_keep_implicit_enabled()
    ):
        canonicalize_t0 = time.perf_counter() if profile_entry is not None else 0.0
        rows_before = int(full_new_x.shape[0])
        unique_x_fast, unique_z_fast, row_inverse_candidate = _canonicalize_pauli_rows_stable(full_new_x, full_new_z)
        n_unique_rows_fast = int(unique_x_fast.shape[0])
        if profile_entry is not None:
            profile_entry["canonicalize_rows_before"] = rows_before
            profile_entry["canonicalize_rows_after"] = n_unique_rows_fast
            profile_entry["canonicalize_changed"] = bool(n_unique_rows_fast != rows_before)
            profile_entry["canonicalize_keep_implicit"] = True
        expected_same = torch.arange(int(same_row_offset), dtype=torch.int64, device=row_inverse_candidate.device)
        same_identity_ok = bool(torch.equal(row_inverse_candidate[: int(same_row_offset)], expected_same))
        if same_identity_ok:
            if full_cache is not None and n_unique_rows_fast != rows_before:
                cache_dev = full_cache.device
                row_inverse_dev = row_inverse_candidate.to(cache_dev)
                merged_cache = torch.zeros((n_unique_rows_fast,), dtype=full_cache.dtype, device=cache_dev)
                merged_cache.index_add_(0, row_inverse_dev, full_cache)
                full_cache = merged_cache
            full_new_x = unique_x_fast
            full_new_z = unique_z_fast
            row_inverse_fast = row_inverse_candidate
            parallel_canonicalize_done = True
            if profile_entry is not None:
                profile_entry["canonicalize_implicit_preserved"] = True
                profile_entry["canonicalize_parallel_fastpath"] = True
                profile_entry["canonicalize_s"] = time.perf_counter() - canonicalize_t0
        else:
            if profile_entry is not None:
                profile_entry["canonicalize_implicit_preserved"] = False
                profile_entry["canonicalize_keep_implicit_fallback"] = True
                profile_entry["canonicalize_parallel_fastpath"] = False

    const_indices, const_values = [], []
    cos_indices, cos_values = [], []
    sin_indices, sin_values = [], []
    row_offset = 0
    explicit_col_offset = 0

    def _append_sparse(mat, indices_list, values_list, r_off, c_off):
        if mat._nnz() > 0:
            if not mat.is_coalesced():
                mat = mat.coalesce()
            idx = mat.indices()
            val = mat.values()
            idx[0] += r_off
            idx[1] += c_off
            indices_list.append(idx)
            values_list.append(val)

    ref_step = step_parts[0]
    if implicit_mode:
        for step in step_parts:
            if step.mat_const._nnz() > 0 or step.mat_cos._nnz() > 0:
                raise RuntimeError("Implicit parallel step unexpectedly contains explicit const/cos matrices.")
        n_parallel_rows = int(full_new_x.shape[0]) if parallel_canonicalize_done else int(same_row_offset + novel_row_offset)
        full_shape = (n_parallel_rows, col_offset)
    else:
        for step in step_parts:
            _append_sparse(step.mat_const, const_indices, const_values, row_offset, explicit_col_offset)
            _append_sparse(step.mat_cos, cos_indices, cos_values, row_offset, explicit_col_offset)
            _append_sparse(step.mat_sin, sin_indices, sin_values, row_offset, explicit_col_offset)
            row_offset += int(step.shape[0])
            explicit_col_offset += int(step.shape[1])
        full_shape = (row_offset, explicit_col_offset)

    def _concat_sparse(indices_list, values_list, shape, t_dtype):
        if not indices_list:
            return _make_sparse(
                torch.tensor([], dtype=torch.int64),
                torch.tensor([], dtype=torch.int64),
                torch.tensor([], dtype=t_dtype),
                shape,
                step_device,
                t_dtype,
            )
        all_indices = _destructive_cat(indices_list, dim=1).to(step_device)
        all_values = _destructive_cat(values_list, dim=0).to(step_device)
        return torch.sparse_coo_tensor(all_indices, all_values, size=shape, device=step_device)

    t_dtype = kwargs["t_dtype"]
    mat_const = _concat_sparse(const_indices, const_values, full_shape, t_dtype)
    mat_cos = _concat_sparse(cos_indices, cos_values, full_shape, t_dtype)
    if implicit_mode:
        remapped_sin_indices = []
        remapped_sin_values = []
        total_same = int(same_row_offset)
        for idx, val, c_off, n_same_local, same_off, novel_off in implicit_sin_parts:
            local_idx = idx.clone()
            local_rows = local_idx[0]
            same_mask_local = local_rows < int(n_same_local)
            if bool(same_mask_local.any().item()):
                local_idx[0][same_mask_local] = local_rows[same_mask_local] + int(same_off)
            if bool((~same_mask_local).any().item()):
                local_idx[0][~same_mask_local] = (
                    (local_rows[~same_mask_local] - int(n_same_local)) + total_same + int(novel_off)
                )
            if row_inverse_fast is not None:
                local_idx[0] = row_inverse_fast.index_select(0, local_idx[0].to(row_inverse_fast.device)).to(local_idx.device)
            local_idx[1] += int(c_off)
            remapped_sin_indices.append(local_idx)
            remapped_sin_values.append(val)
        mat_sin = _concat_sparse(remapped_sin_indices, remapped_sin_values, full_shape, t_dtype)
        full_same_cols = _destructive_cat(same_cols_parts, dim=0).to(step_device) if same_cols_parts else None
        full_anti_same_pos = _destructive_cat(anti_same_pos_parts, dim=0).to(step_device) if anti_same_pos_parts else None
    else:
        mat_sin = _concat_sparse(sin_indices, sin_values, full_shape, t_dtype)
        full_same_cols = None
        full_anti_same_pos = None

    full_step = TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_cos,
        mat_sin=mat_sin,
        param_idx=ref_step.param_idx,
        emb_idx=ref_step.emb_idx,
        shape=full_shape,
        same_cols=full_same_cols,
        anti_same_pos=full_anti_same_pos,
    )

    if builder_name == "build_pauli_rotation_step" and full_step.same_cols is not None:
        full_new_x, full_new_z, full_step, full_cache = _merge_rotation_anti_rows_only(
            full_new_x,
            full_new_z,
            full_step,
            full_cache,
            profile_entry=profile_entry,
        )
    if gate_name != "CliffordGate" and not parallel_canonicalize_done:
        full_new_x, full_new_z, full_step, full_cache = _canonicalize_rows_and_remap_step(
            full_new_x,
            full_new_z,
            full_step,
            full_cache,
            profile_entry=profile_entry,
        )

    _validate_assembled_step(
        full_new_x=full_new_x,
        full_new_z=full_new_z,
        full_step=full_step,
        full_cache=full_cache,
        expected_input_cols=n_terms,
        stage="parallel",
    )

    profile_entry["output_terms"] = int(full_new_x.shape[0])
    return full_new_x, full_new_z, full_step, full_cache, profile_entry


def propagate_surrogate_tensor(
    circuit,
    observable,
    max_weight: int = 50,
    weight_x: float = 1.0,
    weight_y: float = 1.0,
    weight_z: float = 1.0,
    memory_device: str = "cpu",
    compute_device: str = "cuda",
    dtype: str = "float32",
    thetas: Optional[Tensor] = None,
    min_abs: Optional[float] = None,
    min_mat_abs: Optional[float] = None,
    chunk_size: int = 1_000_000,
    parallel_compile: bool = False,
    parallel_threshold: int = -1,
    parallel_devices: Optional[Sequence[int]] = None,
    profile: Optional[Dict[str, Any]] = None,
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

    n_qubits = observable.n_qubits

    # Build initial tensors from observable
    pstrs = list(observable.terms.keys())
    coeffs = [float(observable.terms[p]) for p in pstrs]

    def _pack_masks_63(mask_int: int, n_words: int) -> List[int]:
        word_mask = (1 << 63) - 1
        return [int((mask_int >> (63 * i)) & word_mask) for i in range(n_words)]

    # [Optimization] Initialize masks on CPU to support huge numbers of terms
    if n_qubits <= 63:
        x_mask = torch.as_tensor([p.x_mask for p in pstrs], dtype=torch.int64, device=memory_device)
        z_mask = torch.as_tensor([p.z_mask for p in pstrs], dtype=torch.int64, device=memory_device)
    else:
        n_words = (int(n_qubits) + 62) // 63
        x_words = [_pack_masks_63(int(p.x_mask), n_words) for p in pstrs]
        z_words = [_pack_masks_63(int(p.z_mask), n_words) for p in pstrs]
        x_mask = torch.as_tensor(x_words, dtype=torch.int64, device=memory_device)
        z_mask = torch.as_tensor(z_words, dtype=torch.int64, device=memory_device)

    t_dtype = torch.float64 if dtype == "float64" else torch.float32
    coeff_init = torch.as_tensor(coeffs, dtype=t_dtype, device=memory_device)
    steps: List[TensorSparseStep] = []

    coeffs_cache: Optional[Tensor] = None
    thetas_t: Optional[Tensor] = None
    min_abs_internal: Optional[float] = min_abs

    if min_abs is not None:
        coeff_max = float(torch.max(torch.abs(coeff_init)).detach().cpu().item()) if coeff_init.numel() > 0 else 0.0
        coeff_scale = coeff_max if coeff_max > 0.0 else 1.0
        coeffs_cache = (coeff_init / coeff_scale) # Keep on CPU
        min_abs_internal = float(min_abs) / coeff_scale
        thetas_t = torch.as_tensor(thetas, dtype=coeff_init.dtype, device=compute_device)
        assert thetas_t is not None
        if thetas_t.numel() == 0:
            thetas_t = torch.zeros(1, dtype=coeff_init.dtype, device=compute_device)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    selected_device_ids: List[int] = []
    if bool(parallel_compile) and str(compute_device).startswith("cuda") and num_gpus > 0:
        if parallel_devices is None:
            selected_device_ids = list(range(num_gpus))
        else:
            selected_device_ids = sorted({int(i) for i in parallel_devices})
            if len(selected_device_ids) == 0:
                raise ValueError("parallel_devices must not be empty when provided")
            invalid_ids = [i for i in selected_device_ids if i < 0 or i >= num_gpus]
            if invalid_ids:
                raise ValueError(
                    f"parallel_devices contains invalid GPU id(s): {invalid_ids}. available ids: 0..{num_gpus-1}"
                )

    use_parallel = bool(parallel_compile) and len(selected_device_ids) > 1 and str(compute_device).startswith("cuda")
    parallel_started_logged = False
    parallel_ctx = None
    parallel_pools: Optional[List[Any]] = None

    if use_parallel:
        devices = [f"cuda:{i}" for i in selected_device_ids]
        parallel_ctx = mp.get_context("spawn")

        manual_parallel_threshold = int(parallel_threshold) if int(parallel_threshold) > 0 else None

        def process_step_fn(builder_fn, x_mask, z_mask, chunk_size_local, **kwargs):
            nonlocal parallel_started_logged, parallel_pools
            n_terms_local = int(x_mask.shape[0])

            # Simplified trigger: terms >= max(2 * chunk_size, T_min)
            t_min = manual_parallel_threshold if manual_parallel_threshold is not None else _PARALLEL_MIN_TERMS_HARD
            effective_parallel_threshold = max(2 * int(chunk_size_local), int(t_min))
            use_parallel_this_step = n_terms_local >= effective_parallel_threshold

            if use_parallel_this_step:
                if not parallel_started_logged:
                    print(
                        f"[PPS Info] Starting multi-GPU parallel compile across {num_gpus} GPUs ({devices}) "
                        f"at terms={n_terms_local:,}."
                    )
                    parallel_started_logged = True
                if parallel_pools is None:
                    parallel_pools = [parallel_ctx.Pool(processes=1) for _ in devices]
                return _process_step_parallel(
                    builder_fn,
                    x_mask,
                    z_mask,
                    chunk_size_local,
                    devices,
                    memory_device,
                    memory_device,
                    parallel_threshold=effective_parallel_threshold,
                    pools=parallel_pools,
                    **kwargs,
                )

            return _process_step_chunked(
                builder_fn,
                x_mask,
                z_mask,
                chunk_size_local,
                compute_device,
                memory_device,
                output_device=memory_device,
                **kwargs,
            )
    else:
        def process_step_fn(builder_fn, x_mask, z_mask, chunk_size_local, **kwargs):
            return _process_step_chunked(
                builder_fn,
                x_mask,
                z_mask,
                chunk_size_local,
                compute_device,
                memory_device,
                output_device=memory_device,
                **kwargs,
            )

    total_gates = len(circuit)
    propagate_stats: Optional[Dict[str, Any]] = None
    if profile is not None:
        propagate_stats = cast(Dict[str, Any], profile.setdefault("propagate", {}))
        propagate_stats.setdefault("steps", [])
        propagate_stats["total_gates"] = int(total_gates)
    propagate_t0 = time.perf_counter() if profile is not None else 0.0

    def _record_step(
        *,
        loop_idx: int,
        gate_name: str,
        n_terms_in: int,
        new_x: Tensor,
        step: TensorSparseStep,
        step_profile: Dict[str, Any],
        process_s: float,
        prune_s: float,
    ) -> None:
        if propagate_stats is None:
            return
        cast(List[Dict[str, Any]], propagate_stats["steps"]).append(
            {
                **step_profile,
                "propagate_step_index": int(loop_idx),
                "circuit_gate_index": int(total_gates - 1 - loop_idx),
                "gate_name": gate_name,
                "input_terms": int(n_terms_in),
                "output_terms": int(new_x.shape[0]),
                "process_s": float(process_s),
                "prune_s": float(prune_s),
                "step_total_s": float(process_s + prune_s),
                "step_rows": int(step.shape[0]),
                "step_cols": int(step.shape[1]),
            }
        )

    try:
        for loop_idx, gate in enumerate(tqdm(reversed(circuit), total=total_gates, desc="propagate", dynamic_ncols=True)):
            gate_name = gate.__class__.__name__
            n_terms_in = int(x_mask.shape[0])

            if gate_name == "CliffordGate":
                step_t0 = time.perf_counter() if profile is not None else 0.0
                new_x, new_z, step, coeffs_cache, step_profile = process_step_fn(
                    build_clifford_step,
                    x_mask,
                    z_mask,
                    chunk_size,
                    gate=gate,
                    t_dtype=t_dtype,
                    min_abs=min_abs_internal,
                    coeffs_cache=coeffs_cache,
                    max_weight=max_weight,
                    weight_x=weight_x,
                    weight_y=weight_y,
                    weight_z=weight_z,
                )
                process_s = (time.perf_counter() - step_t0) if profile is not None else 0.0
                prune_t0 = time.perf_counter() if profile is not None else 0.0
                if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                    step = _prune_step_by_abs(step, float(min_mat_abs))
                prune_s = (time.perf_counter() - prune_t0) if profile is not None else 0.0
                steps.append(step)
                x_mask, z_mask = new_x, new_z
                _record_step(
                    loop_idx=loop_idx,
                    gate_name=gate_name,
                    n_terms_in=n_terms_in,
                    new_x=new_x,
                    step=step,
                    step_profile=step_profile,
                    process_s=process_s,
                    prune_s=prune_s,
                )
                continue

            if gate_name == "PauliRotation":
                step_t0 = time.perf_counter() if profile is not None else 0.0
                new_x, new_z, step, coeffs_cache, step_profile = process_step_fn(
                    build_pauli_rotation_step,
                    x_mask,
                    z_mask,
                    chunk_size,
                    gate=gate,
                    t_dtype=t_dtype,
                    min_abs=min_abs_internal,
                    coeffs_cache=coeffs_cache,
                    thetas_t=thetas_t,
                    max_weight=max_weight,
                    weight_x=weight_x,
                    weight_y=weight_y,
                    weight_z=weight_z,
                )
                process_s = (time.perf_counter() - step_t0) if profile is not None else 0.0
                prune_t0 = time.perf_counter() if profile is not None else 0.0
                if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                    step = _prune_step_by_abs(step, float(min_mat_abs))
                prune_s = (time.perf_counter() - prune_t0) if profile is not None else 0.0
                steps.append(step)
                x_mask, z_mask = new_x, new_z
                _record_step(
                    loop_idx=loop_idx,
                    gate_name=gate_name,
                    n_terms_in=n_terms_in,
                    new_x=new_x,
                    step=step,
                    step_profile=step_profile,
                    process_s=process_s,
                    prune_s=prune_s,
                )
                continue

            if gate_name == "DepolarizingNoise":
                step_t0 = time.perf_counter() if profile is not None else 0.0
                new_x, new_z, step, coeffs_cache, step_profile = process_step_fn(
                    build_depolarizing_step,
                    x_mask,
                    z_mask,
                    chunk_size,
                    gate=gate,
                    t_dtype=t_dtype,
                    min_abs=min_abs_internal,
                    coeffs_cache=coeffs_cache,
                    max_weight=max_weight,
                    weight_x=weight_x,
                    weight_y=weight_y,
                    weight_z=weight_z,
                )
                process_s = (time.perf_counter() - step_t0) if profile is not None else 0.0
                prune_t0 = time.perf_counter() if profile is not None else 0.0
                if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                    step = _prune_step_by_abs(step, float(min_mat_abs))
                prune_s = (time.perf_counter() - prune_t0) if profile is not None else 0.0
                steps.append(step)
                x_mask, z_mask = new_x, new_z
                _record_step(
                    loop_idx=loop_idx,
                    gate_name=gate_name,
                    n_terms_in=n_terms_in,
                    new_x=new_x,
                    step=step,
                    step_profile=step_profile,
                    process_s=process_s,
                    prune_s=prune_s,
                )
                continue

            if gate_name == "AmplitudeDampingNoise":
                step_t0 = time.perf_counter() if profile is not None else 0.0
                new_x, new_z, step, coeffs_cache, step_profile = process_step_fn(
                    build_amplitude_damping_step,
                    x_mask,
                    z_mask,
                    chunk_size,
                    gate=gate,
                    t_dtype=t_dtype,
                    min_abs=min_abs_internal,
                    coeffs_cache=coeffs_cache,
                    max_weight=max_weight,
                    weight_x=weight_x,
                    weight_y=weight_y,
                    weight_z=weight_z,
                )
                process_s = (time.perf_counter() - step_t0) if profile is not None else 0.0
                prune_t0 = time.perf_counter() if profile is not None else 0.0
                if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                    step = _prune_step_by_abs(step, float(min_mat_abs))
                prune_s = (time.perf_counter() - prune_t0) if profile is not None else 0.0
                steps.append(step)
                x_mask, z_mask = new_x, new_z
                _record_step(
                    loop_idx=loop_idx,
                    gate_name=gate_name,
                    n_terms_in=n_terms_in,
                    new_x=new_x,
                    step=step,
                    step_profile=step_profile,
                    process_s=process_s,
                    prune_s=prune_s,
                )
                continue

            raise TypeError(
                f"Unsupported gate type in propagate_surrogate_tensor: {gate_name}. "
                "Expected CliffordGate, PauliRotation, DepolarizingNoise, or AmplitudeDampingNoise."
            )

        psum = TensorPauliSum(
            n_qubits=n_qubits,
            x_mask=x_mask,
            z_mask=z_mask,
            coeff_init=coeff_init,
            steps=steps,
        )
        if propagate_stats is not None:
            propagate_stats["total_s"] = time.perf_counter() - propagate_t0
            propagate_stats["output_terms"] = int(x_mask.shape[0])

        return psum
    finally:
        if parallel_pools is not None:
            for parallel_pool in parallel_pools:
                parallel_pool.close()
            for parallel_pool in parallel_pools:
                parallel_pool.join()


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
    compute_device: str = "cuda",
    chunk_size: int = 1_000_000,
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[TensorPauliSum, Tensor]:
    """Back-propagating zero-filter.

    Memory note:
      This function may consume (clear) internals of the input `psum` while
      constructing the filtered result to reduce CPU memory peak.

    Returns:
      (filtered_psum, keep_mask_in)
    where keep_mask_in is a boolean mask over the *input coefficient dimension*
    after pruning.
    """
    print(f"Starting zero-filtering on {compute_device}...")

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")

    n_qubits = int(psum.n_qubits)
    x_mask_src = psum.x_mask
    z_mask_src = psum.z_mask
    coeff_init_src = psum.coeff_init
    steps = list(psum.steps)
    zero_filter_stats: Optional[Dict[str, Any]] = None
    if profile is not None:
        zero_filter_stats = cast(Dict[str, Any], profile.setdefault("zero_filter", {}))
        zero_filter_stats.setdefault("steps", [])
    zero_filter_t0 = time.perf_counter() if profile is not None else 0.0

    # Release references from input object early (consuming behavior).
    psum.steps = []

    if x_mask_src.numel() == 0:
        print("[ZeroFilter] No terms generated (empty mask). Skipping back-propagation.")
        keep_mask = torch.zeros((int(coeff_init_src.shape[0]),), dtype=torch.bool, device=coeff_init_src.device)
        filtered = TensorPauliSum(
            n_qubits=n_qubits,
            x_mask=x_mask_src,
            z_mask=z_mask_src,
            coeff_init=coeff_init_src[keep_mask],
            steps=steps,
        )
        if zero_filter_stats is not None:
            zero_filter_stats["initial_terms"] = 0
            zero_filter_stats["diagonal_terms"] = 0
            zero_filter_stats["final_terms"] = 0
            zero_filter_stats["total_s"] = time.perf_counter() - zero_filter_t0
        return filtered, keep_mask

    out_mask = _xmask_is_zero_rows(x_mask_src)
    
    # [Log] Show initial reduction from total propagated terms to diagonal terms
    n_total = int(x_mask_src.shape[0])
    n_diag = int(out_mask.sum().item())
    pct = 100.0 * n_diag / n_total if n_total > 0 else 0.0
    print(f"[ZeroFilter] Initial pruning (diagonal terms only): {n_total:,} -> {n_diag:,} terms kept ({pct:.8f}%)")
    print(f"[ZeroFilter] Zero-filtering done. Starting back-propagation of keep mask through {len(steps)} steps...")
    if zero_filter_stats is not None:
        zero_filter_stats["initial_terms"] = int(n_total)
        zero_filter_stats["diagonal_terms"] = int(n_diag)

    # Materialize filtered output masks once, then clear original references.
    x_mask_out = x_mask_src[out_mask]
    z_mask_out = z_mask_src[out_mask]
    psum.x_mask = x_mask_out[:0]
    psum.z_mask = z_mask_out[:0]

    # [Memory Optimization] Clear GPU cache from forward pass to free up VRAM for filtering
    if torch.cuda.is_available() and "cuda" in compute_device:
        torch.cuda.empty_cache()

    row_mask = out_mask
    pbar = tqdm(reversed(range(len(steps))), total=len(steps), desc="zero-filter", dynamic_ncols=True)

    # used-cols accumulation is fixed to CPU for stability.
    used_cols_mode = _zero_filter_used_cols_mode()
    compact_device = _zero_filter_compact_device(compute_device)

    if compact_device != str(compute_device):
        print(f"[ZeroFilter] compaction device override: {compact_device}")

    for loop_idx, i in enumerate(pbar):
        step_t0 = time.perf_counter() if zero_filter_stats is not None else 0.0
        old_step = steps[i]
        n_cols = int(old_step.shape[1])

        # Coalesce each sparse matrix at most once per step and reuse.
        mat_const = old_step.mat_const if old_step.mat_const.is_coalesced() else old_step.mat_const.coalesce()
        mat_cos = old_step.mat_cos if old_step.mat_cos.is_coalesced() else old_step.mat_cos.coalesce()
        mat_sin = old_step.mat_sin if old_step.mat_sin.is_coalesced() else old_step.mat_sin.coalesce()

        nnz_const = int(mat_const._nnz())
        nnz_cos = int(mat_cos._nnz())
        nnz_sin = int(mat_sin._nnz())
        nnz_total = nnz_const + nnz_cos + nnz_sin
        same_nnz = int(old_step.same_nnz())

        if int(row_mask.sum().item()) == 0:
            col_mask = torch.zeros((n_cols,), dtype=torch.bool, device=row_mask.device)
            used_cols_s = 0.0
        elif nnz_total == 0 and same_nnz == 0:
            col_mask = torch.zeros((n_cols,), dtype=torch.bool, device=row_mask.device)
            used_cols_s = 0.0
        else:
            used_cols_t0 = time.perf_counter() if zero_filter_stats is not None else 0.0
            col_mask = _implicit_same_used_cols(old_step, row_mask, n_cols)
            if int(mat_const._nnz()) > 0:
                col_mask |= _sparse_used_cols(mat_const, row_mask, n_cols)
            if nnz_cos > 0:
                col_mask |= _sparse_used_cols(mat_cos, row_mask, n_cols)
            if nnz_sin > 0:
                col_mask |= _sparse_used_cols(mat_sin, row_mask, n_cols)
            used_cols_s = (time.perf_counter() - used_cols_t0) if zero_filter_stats is not None else 0.0

        if (loop_idx % 8 == 0) or (i == 0):
            pbar.set_postfix_str(
                f"step={i} rows={int(row_mask.sum().item())} cols={int(col_mask.sum().item())} nnz={nnz_total}",
                refresh=False,
            )

        step_for_compact = TensorSparseStep(
            mat_const=mat_const,
            mat_cos=mat_cos,
            mat_sin=mat_sin,
            param_idx=old_step.param_idx,
            emb_idx=old_step.emb_idx,
            shape=old_step.shape,
            same_cols=old_step.same_cols,
            anti_same_pos=old_step.anti_same_pos,
        )

        compact_t0 = time.perf_counter() if zero_filter_stats is not None else 0.0
        steps[i] = compact_sparse_step_chunked(
            step_for_compact,
            keep_row_mask=row_mask,
            keep_col_mask=col_mask,
            device=compact_device,
            chunk_size=chunk_size,
        )
        compact_s = (time.perf_counter() - compact_t0) if zero_filter_stats is not None else 0.0
        del old_step, step_for_compact
        if zero_filter_stats is not None:
            cast(List[Dict[str, Any]], zero_filter_stats["steps"]).append(
                {
                    "zero_filter_step_index": int(loop_idx),
                    "propagate_step_index": int(i),
                    "rows_kept_in": int(row_mask.sum().item()),
                    "cols_kept_out": int(col_mask.sum().item()),
                    "nnz_total": int(nnz_total),
                    "same_nnz": int(same_nnz),
                    "used_cols_s": float(used_cols_s),
                    "compact_s": float(compact_s),
                    "step_total_s": float(time.perf_counter() - step_t0),
                }
            )
        row_mask = col_mask

    coeff_init = coeff_init_src[row_mask]
    psum.coeff_init = coeff_init[:0]

    filtered = TensorPauliSum(
        n_qubits=n_qubits,
        x_mask=x_mask_out,
        z_mask=z_mask_out,
        coeff_init=coeff_init,
        steps=steps,
    )
    if zero_filter_stats is not None:
        zero_filter_stats["total_s"] = time.perf_counter() - zero_filter_t0
        zero_filter_stats["final_terms"] = int(x_mask_out.shape[0])
    return filtered, row_mask


def zero_filter_tensor_backprop(
    psum: TensorPauliSum,
    compute_device: str = "cuda",
    chunk_size: int = 1_000_000,
    profile: Optional[Dict[str, Any]] = None,
) -> TensorPauliSum:
    """Back-propagating zero-filter (prune rows/cols through sparse steps)."""

    filtered, _keep = zero_filter_tensor_backprop_with_keep_mask(
        psum,
        compute_device=compute_device,
        chunk_size=chunk_size,
        profile=profile,
    )
    return filtered


__all__ = [
    "propagate_surrogate_tensor",
    "zero_filter_tensor",
    "zero_filter_tensor_backprop",
    "zero_filter_tensor_backprop_with_keep_mask",
]
