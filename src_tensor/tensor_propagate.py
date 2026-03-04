"""Tensor-native surrogate propagation (GPU-first, WIP).

This module also provides a *back-propagating zero-filter* that prunes the
sparse-step graph for the common case where the initial state is |0..0>.

For <0|U† O U|0>, only diagonal output terms (I/Z-only, i.e. x_mask == 0)
contribute. `zero_filter_tensor_backprop` keeps only those output rows and
then walks steps backwards to remove any input columns that cannot reach them.
"""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING, Tuple, cast
import math

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
    compact_sparse_step_chunked,
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


def _accumulate_used_cols_chunked(mat: Tensor, row_mask_d: Tensor, col_mask_d: Tensor, chunk_size: int) -> None:
    """Accumulate used columns from a CPU sparse matrix into a GPU mask using chunks."""
    if mat._nnz() == 0:
        return
    
    # Coalesce the tensor to ensure indices are accessible and unique.
    # This is required before calling .indices() on a sparse tensor that may
    # have been constructed from non-unique/unsorted indices.
    if not mat.is_coalesced():
        mat = mat.coalesce()
    # Access indices on CPU (Storage Device)
    indices = mat.indices()
    nnz = indices.shape[1]
    device = row_mask_d.device
    
    # [Optimization] Pin memory for faster H2D transfer
    # This enables direct DMA transfer and better PCIe saturation.
    # Only pin if tensor is large enough (> 1MB approx) to justify allocation overhead.
    # 1MB ~ 128k int64 elements.
    if str(device).startswith("cuda") and not indices.is_pinned() and indices.numel() > 100_000:
        try:
            indices = indices.pin_memory()
        except Exception:
            pass # Fallback to pageable memory if RAM is full
    
    # [Optimization] Dynamic Chunking for mask accumulation
    current_chunk_size = chunk_size
    if str(device).startswith("cuda"):
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            # Heuristic: indices (16B) + overhead.
            # Reserve 500MB buffer.
            safe_mem = max(0, free_mem - 500 * 1024 * 1024)
            estimated_capacity = int(safe_mem // 64)
            current_chunk_size = max(chunk_size, estimated_capacity)
        except Exception:
            pass

    for start in range(0, nnz, current_chunk_size):
        end = min(start + current_chunk_size, nnz)
        # Move only indices to GPU (Compute Device)
        idx_chunk = indices[:, start:end].to(device, non_blocking=True)
        
        # Update mask on GPU
        keep = row_mask_d[idx_chunk[0]]
        col_mask_d[idx_chunk[1][keep]] = True


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
        emb_idx=step.emb_idx,
        shape=(int(row_mask.sum().item()), int(col_mask.sum().item())),
    )

def _step_to_device(step: TensorSparseStep, device: str) -> TensorSparseStep:
    return TensorSparseStep(
        mat_const=step.mat_const.to(device),
        mat_cos=step.mat_cos.to(device),
        mat_sin=step.mat_sin.to(device),
        param_idx=step.param_idx,
        emb_idx=step.emb_idx,
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
        emb_idx=step.emb_idx,
        shape=step.shape,
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


def _process_step_chunked(
    builder_fn,
    x_mask_in: Tensor,
    z_mask_in: Tensor,
    chunk_size: int,
    device: str,
    step_device: str,
    output_device: str = "cpu",
    **kwargs
) -> Tuple[Tensor, Tensor, TensorSparseStep, Optional[Tensor]]:
    """
    Process a propagation step in chunks to avoid GPU OOM.
    Keeps the main masks on CPU, moves chunks to GPU for processing,
    then assembles the result back on CPU (or step_device).
    """
    n_terms = x_mask_in.shape[0]
    
    # Fast path: if small enough, run directly
    if n_terms <= chunk_size:
        x_gpu = x_mask_in.to(device)
        z_gpu = z_mask_in.to(device)
        # Handle coeffs_cache if present
        if 'coeffs_cache' in kwargs and kwargs['coeffs_cache'] is not None:
            kwargs['coeffs_cache'] = kwargs['coeffs_cache'].to(device)
            
        new_x, new_z, step, cache = builder_fn(x_mask=x_gpu, z_mask=z_gpu, device=device, step_device=step_device, **kwargs)
        return new_x.to(output_device), new_z.to(output_device), step, (cache.to(output_device) if cache is not None else None)

    # Chunked processing
    new_x_parts = []
    new_z_parts = []
    cache_parts = []
    
    # Accumulators for sparse indices
    const_indices, const_values = [], []
    cos_indices, cos_values = [], []
    sin_indices, sin_values = [], []
    
    row_offset = 0
    col_offset = 0
    
    n_chunks = math.ceil(n_terms / chunk_size)
    
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

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_terms)
        
        # 1. Move chunk to GPU
        x_chunk = x_mask_in[start:end].to(device)
        z_chunk = z_mask_in[start:end].to(device)
        
        chunk_kwargs = kwargs.copy()
        if 'coeffs_cache' in chunk_kwargs and chunk_kwargs['coeffs_cache'] is not None:
            chunk_kwargs['coeffs_cache'] = chunk_kwargs['coeffs_cache'][start:end].to(device)
            
        nx, nz, step, cache = builder_fn(x_mask=x_chunk, z_mask=z_chunk, device=device, step_device=device, **chunk_kwargs)
        
        # 2. Collect Results (Move back to CPU immediately)
        new_x_parts.append(nx.to(output_device))
        new_z_parts.append(nz.to(output_device))
        if cache is not None:
            cache_parts.append(cache.to(output_device))
        
        # 3. Accumulate Sparse Matrices
        _append_sparse(step.mat_const, const_indices, const_values, row_offset, col_offset)
        _append_sparse(step.mat_cos, cos_indices, cos_values, row_offset, col_offset)
        _append_sparse(step.mat_sin, sin_indices, sin_values, row_offset, col_offset)
        
        row_offset += nx.shape[0]
        col_offset += (end - start)

    # 4. Assemble Final Tensors
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

    full_shape = (row_offset, col_offset)

    mat_const = _concat_sparse(const_indices, const_values, full_shape)
    const_indices, const_values = None, None

    mat_cos = _concat_sparse(cos_indices, cos_values, full_shape)
    cos_indices, cos_values = None, None

    mat_sin = _concat_sparse(sin_indices, sin_values, full_shape)
    sin_indices, sin_values = None, None

    full_step = TensorSparseStep(
        mat_const=mat_const,
        mat_cos=mat_cos,
        mat_sin=mat_sin,
        param_idx=step.param_idx,
        emb_idx=step.emb_idx,
        shape=full_shape
    )
    
    return full_new_x, full_new_z, full_step, full_cache


def propagate_surrogate_tensor(
    circuit,
    observable,
    max_weight: int = 50,
    max_xy: int = 50,
    memory_device: str = "cpu",
    compute_device: str = "cuda",
    dtype: str = "float32",
    thetas: Optional[Tensor] = None,
    min_abs: Optional[float] = None,
    min_mat_abs: Optional[float] = None,
    chunk_size: int = 1_000_000,
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
        if thetas_t.numel() == 0:
            thetas_t = torch.zeros(1, dtype=coeff_init.dtype, device=compute_device)

    total_gates = len(circuit)
    for gate in tqdm(reversed(circuit), total=total_gates, desc="propagate", dynamic_ncols=True):
        gate_name = gate.__class__.__name__

        if gate_name == "CliffordGate":
            new_x, new_z, step, coeffs_cache = _process_step_chunked(
                build_clifford_step,
                x_mask, z_mask, chunk_size, compute_device, memory_device,
                output_device=memory_device,
                gate=gate,
                t_dtype=t_dtype,
                min_abs=min_abs_internal,
                coeffs_cache=coeffs_cache,
                max_weight=max_weight,
                max_xy=max_xy,
            )
            if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                step = _prune_step_by_abs(step, float(min_mat_abs))
            steps.append(step)
            x_mask, z_mask = new_x, new_z
            continue

        if gate_name == "PauliRotation":
            new_x, new_z, step, coeffs_cache = _process_step_chunked(
                build_pauli_rotation_step,
                x_mask, z_mask, chunk_size, compute_device, memory_device,
                output_device=memory_device,
                gate=gate,
                t_dtype=t_dtype,
                min_abs=min_abs_internal,
                coeffs_cache=coeffs_cache,
                thetas_t=thetas_t,
                max_weight=max_weight,
                max_xy=max_xy,
            )
            if min_mat_abs is not None and float(min_mat_abs) > 0.0:
                step = _prune_step_by_abs(step, float(min_mat_abs))
            steps.append(step)
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
    compute_device: str = "cuda",
    chunk_size: int = 1_000_000,
) -> Tuple[TensorPauliSum, Tensor]:
    """Back-propagating zero-filter.

    Returns:
      (filtered_psum, keep_mask_in)
    where keep_mask_in is a boolean mask over the *input coefficient dimension*
    after pruning.
    """
    print(f"Starting zero-filtering on {compute_device}...")

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    if psum.x_mask.numel() == 0:
        print("[ZeroFilter] No terms generated (empty mask). Skipping back-propagation.")
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
    
    # [Log] Show initial reduction from total propagated terms to diagonal terms
    n_total = psum.x_mask.shape[0]
    n_diag = int(out_mask.sum().item())
    pct = 100.0 * n_diag / n_total if n_total > 0 else 0.0
    print(f"[ZeroFilter] Initial pruning (diagonal terms only): {n_total:,} -> {n_diag:,} terms kept ({pct:.8f}%)")
    print(f"[ZeroFilter] Zero-filtering done. Starting back-propagation of keep mask through {len(psum.steps)} steps...")

    steps = list(psum.steps)

    # [Memory Optimization] Clear GPU cache from forward pass to free up VRAM for filtering
    if torch.cuda.is_available() and "cuda" in compute_device:
        torch.cuda.empty_cache()

    row_mask = out_mask
    for i in reversed(range(len(steps))):
        # [Memory Optimization] Decompose step to release old tensors immediately after filtering
        old_step = steps[i]
        steps[i] = None  # Clear reference from list to allow GC
        
        n_cols = int(old_step.shape[1])
        
        # 1. Calculate used columns (Chunked GPU acceleration)
        # Move masks to GPU, stream indices in chunks
        try:
            row_mask_d = row_mask.to(compute_device)
            col_mask_d = torch.zeros((n_cols,), dtype=torch.bool, device=compute_device)
            
            _accumulate_used_cols_chunked(old_step.mat_const, row_mask_d, col_mask_d, chunk_size)
            _accumulate_used_cols_chunked(old_step.mat_cos, row_mask_d, col_mask_d, chunk_size)
            _accumulate_used_cols_chunked(old_step.mat_sin, row_mask_d, col_mask_d, chunk_size)
            
            col_mask = col_mask_d.to(row_mask.device) # Back to CPU
        except RuntimeError as e: # Fallback to CPU if GPU OOM (e.g. masks too big)
            print(f"Warning: GPU acceleration failed in zero-filter, falling back to CPU. Error: {e}")
            col_mask = _sparse_used_cols(old_step.mat_const, row_mask, n_cols)
            if old_step.mat_cos._nnz() > 0:
                col_mask |= _sparse_used_cols(old_step.mat_cos, row_mask, n_cols)
            if old_step.mat_sin._nnz() > 0:
                col_mask |= _sparse_used_cols(old_step.mat_sin, row_mask, n_cols)
            
        # [Log] Print reduction stats
        if len(str(len(steps))) < 3:
            if (i % 10) == 0 or (i == len(steps) - 1):
                # Note: cols can be > rows because a single output term (row) may depend on
                # multiple input terms (cols) due to rotation mixing (branching).
                print(f"[ZeroFilter] Step {i}: {int(row_mask.sum().item())} rows -> {int(col_mask.sum().item())} cols kept")
        else:
            if (i % 100) == 0 or (i == len(steps) - 1):
                # Note: cols can be > rows because a single output term (row) may depend on
                # multiple input terms (cols) due to rotation mixing (branching).
                print(f"[ZeroFilter] Step {i}: {int(row_mask.sum().item())} rows -> {int(col_mask.sum().item())} cols kept")

        # 2. Filter matrices using optimized chunked kernel
        # This replaces the old sequential _filter_sparse_rows_cols calls
        # and handles GPU streaming + memory cleanup internally.
        steps[i] = compact_sparse_step_chunked(
            old_step,
            keep_row_mask=row_mask,
            keep_col_mask=col_mask,
            device=compute_device,
            chunk_size=chunk_size
        )
        
        # Release old step explicitly (though GC should handle it as steps[i] was cleared)
        del old_step
        
        row_mask = col_mask

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
    compute_device: str = "cuda",
    chunk_size: int = 1_000_000,
) -> TensorPauliSum:
    """Back-propagating zero-filter (prune rows/cols through sparse steps)."""

    filtered, _keep = zero_filter_tensor_backprop_with_keep_mask(
        psum,
        compute_device=compute_device,
        chunk_size=chunk_size,
    )
    return filtered


__all__ = [
    "propagate_surrogate_tensor",
    "zero_filter_tensor",
    "zero_filter_tensor_backprop",
    "zero_filter_tensor_backprop_with_keep_mask",
]
