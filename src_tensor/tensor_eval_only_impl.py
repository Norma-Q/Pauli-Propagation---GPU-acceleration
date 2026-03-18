"""Direct evaluate-only tensor backend.

This path skips TensorSparseStep construction entirely and propagates
observable coefficients backward through the circuit for a single theta point.
It is intended for forward-only evaluation workloads where we want to avoid
holding all per-step sparse matrices in memory.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING, Optional, Tuple, cast
from tqdm import tqdm

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

_CPP_EVAL_BACKEND: Any
try:
    from . import _pps_tensor_eval_only_backend_local as _CPP_EVAL_BACKEND  # type: ignore

    _CPP_EVAL_AVAILABLE = True
except Exception:  # pragma: no cover
    _CPP_EVAL_BACKEND = cast(Any, None)
    _CPP_EVAL_AVAILABLE = False


class CPPEvalOnlyBackendUnavailableError(RuntimeError):
    pass


def _require_eval_backend() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for the direct evaluate-only backend.")
    if not _CPP_EVAL_AVAILABLE:
        raise CPPEvalOnlyBackendUnavailableError(
            "Direct evaluate-only backend is not available. "
            "Build it with `QAOA/rebuild_eval_only_backend.py`."
        )


def _resolve_t_dtype(dtype: str | Any):
    if isinstance(dtype, str):
        key = str(dtype).strip().lower()
        if key in ("float64", "torch.float64", "torch.double", "double"):
            return torch.float64
        if key in ("float32", "torch.float32", "torch.float", "float"):
            return torch.float32
        raise ValueError(f"Unsupported direct-eval dtype: {dtype}")
    return dtype


def _device_matches(requested: str | Any, actual: str | Any) -> bool:
    req = torch.device(requested)
    act = torch.device(actual)
    if req.type != act.type:
        return False
    if req.index is None:
        return True
    return req.index == act.index


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


def _pack_masks_63(mask_int: int, n_words: int) -> list[int]:
    word_mask = (1 << 63) - 1
    return [int((mask_int >> (63 * i)) & word_mask) for i in range(n_words)]


def _observable_to_state(*, observable, device: str, dtype) -> Tuple[Tensor, Tensor, Tensor]:
    pstrs = list(observable.terms.keys())
    coeffs = [float(observable.terms[p]) for p in pstrs]
    n_qubits = int(observable.n_qubits)

    if n_qubits <= 63:
        x_mask = torch.as_tensor([int(p.x_mask) for p in pstrs], dtype=torch.int64, device=device)
        z_mask = torch.as_tensor([int(p.z_mask) for p in pstrs], dtype=torch.int64, device=device)
    else:
        n_words = (n_qubits + 62) // 63
        x_words = [_pack_masks_63(int(p.x_mask), n_words) for p in pstrs]
        z_words = [_pack_masks_63(int(p.z_mask), n_words) for p in pstrs]
        x_mask = torch.as_tensor(x_words, dtype=torch.int64, device=device)
        z_mask = torch.as_tensor(z_words, dtype=torch.int64, device=device)

    coeff_t = torch.as_tensor(coeffs, dtype=dtype, device=device)
    return x_mask, z_mask, coeff_t


def _popcount_u64(x: Tensor) -> Tensor:
    count = torch.zeros_like(x, dtype=torch.int64)
    for i in range(64):
        count = count + torch.bitwise_and(torch.bitwise_right_shift(x, i), 1)
    return count


def _popcount_sum_words(x: Tensor) -> Tensor:
    c = _popcount_u64(x)
    if x.dim() == 2:
        return c.sum(1)
    return c


def _truncation_is_effective(
    x_mask: Tensor,
    *,
    max_weight: int,
    weight_x: float,
    weight_y: float,
    weight_z: float,
) -> bool:
    weights = (float(weight_x), float(weight_y), float(weight_z))
    if any(w < 0.0 for w in weights):
        return True
    max_axis_weight = max(weights)
    if max_axis_weight <= 0.0:
        return False
    n_words = int(x_mask.shape[1]) if x_mask.dim() == 2 else 1
    max_possible_weight = float(n_words * 63) * max_axis_weight
    return float(max_weight) < max_possible_weight


def _truncate_terms_mask(x_mask: Tensor, z_mask: Tensor, *, max_weight: int, weight_x: float, weight_y: float, weight_z: float) -> Tensor:
    if not _truncation_is_effective(
        x_mask,
        max_weight=max_weight,
        weight_x=weight_x,
        weight_y=weight_y,
        weight_z=weight_z,
    ):
        return torch.ones((int(x_mask.shape[0]),), dtype=torch.bool, device=x_mask.device)
    x_cnt = _popcount_sum_words(x_mask & (~z_mask)).to(torch.float64)
    y_cnt = _popcount_sum_words(x_mask & z_mask).to(torch.float64)
    z_cnt = _popcount_sum_words((~x_mask) & z_mask).to(torch.float64)
    weighted = x_cnt * float(weight_x) + y_cnt * float(weight_y) + z_cnt * float(weight_z)
    return weighted <= float(max_weight)


def _apply_state_filters(
    x_mask: Tensor,
    z_mask: Tensor,
    coeffs: Tensor,
    *,
    min_abs: Optional[float],
    max_weight: int,
    weight_x: float,
    weight_y: float,
    weight_z: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    if int(coeffs.numel()) == 0:
        return x_mask[:0], z_mask[:0], coeffs[:0]

    if min_abs is not None:
        keep = coeffs.abs() >= float(min_abs)
        keep_idx = torch.nonzero(keep, as_tuple=False).flatten().to(torch.int64)
        x_mask = x_mask.index_select(0, keep_idx)
        z_mask = z_mask.index_select(0, keep_idx)
        coeffs = coeffs.index_select(0, keep_idx)
        if int(coeffs.numel()) == 0:
            return x_mask, z_mask, coeffs

    mask = _truncate_terms_mask(
        x_mask,
        z_mask,
        max_weight=int(max_weight),
        weight_x=float(weight_x),
        weight_y=float(weight_y),
        weight_z=float(weight_z),
    )
    if not bool(mask.all().item()):
        out_idx = torch.nonzero(mask, as_tuple=False).flatten().to(torch.int64)
        x_mask = x_mask.index_select(0, out_idx)
        z_mask = z_mask.index_select(0, out_idx)
        coeffs = coeffs.index_select(0, out_idx)
    return x_mask, z_mask, coeffs


def _merge_state_pair(
    lhs: Tuple[Tensor, Tensor, Tensor],
    rhs: Tuple[Tensor, Tensor, Tensor],
    *,
    min_abs: Optional[float],
    max_weight: int,
    weight_x: float,
    weight_y: float,
    weight_z: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    x_l, z_l, c_l = lhs
    x_r, z_r, c_r = rhs

    if int(c_l.numel()) == 0:
        return _apply_state_filters(
            x_r,
            z_r,
            c_r,
            min_abs=min_abs,
            max_weight=max_weight,
            weight_x=weight_x,
            weight_y=weight_y,
            weight_z=weight_z,
        )
    if int(c_r.numel()) == 0:
        return _apply_state_filters(
            x_l,
            z_l,
            c_l,
            min_abs=min_abs,
            max_weight=max_weight,
            weight_x=weight_x,
            weight_y=weight_y,
            weight_z=weight_z,
        )

    x_cat = torch.cat([x_l, x_r], dim=0)
    z_cat = torch.cat([z_l, z_r], dim=0)
    c_cat = torch.cat([c_l, c_r], dim=0)

    if x_cat.dim() == 1:
        keys = torch.stack([x_cat, z_cat], dim=1)
        uniq, inv = torch.unique(keys, dim=0, return_inverse=True, sorted=False)
        new_x = uniq[:, 0]
        new_z = uniq[:, 1]
    else:
        keys = torch.cat([x_cat, z_cat], dim=1)
        uniq, inv = torch.unique(keys, dim=0, return_inverse=True, sorted=False)
        n_words = int(x_cat.shape[1])
        new_x = uniq[:, :n_words]
        new_z = uniq[:, n_words:]

    inv = inv.to(torch.int64)
    new_coeffs = torch.zeros((int(new_x.shape[0]),), dtype=c_cat.dtype, device=c_cat.device)
    if int(c_cat.numel()) > 0:
        new_coeffs.index_add_(0, inv, c_cat)

    return _apply_state_filters(
        new_x,
        new_z,
        new_coeffs,
        min_abs=min_abs,
        max_weight=max_weight,
        weight_x=weight_x,
        weight_y=weight_y,
        weight_z=weight_z,
    )


def _move_state(
    state: Tuple[Tensor, Tensor, Tensor],
    *,
    device: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    x_mask, z_mask, coeffs = state
    return x_mask.to(device), z_mask.to(device), coeffs.to(device)


def _rotation_anti_local_rows(
    *,
    gate,
    x_mask: Tensor,
    z_mask: Tensor,
) -> Tensor:
    if x_mask.dim() == 2:
        gx_words, gz_words = _gate_masks_words(gate, n_words=int(x_mask.shape[1]), device=x_mask.device)
        gx_t = gx_words.unsqueeze(0)
        gz_t = gz_words.unsqueeze(0)
        symp = _popcount_u64((torch.bitwise_and(x_mask, gz_t)) ^ (torch.bitwise_and(z_mask, gx_t))).sum(1) & 1
    else:
        gx, gz = _gate_masks(gate)
        gx_t = torch.scalar_tensor(int(gx), dtype=torch.int64, device=x_mask.device)
        gz_t = torch.scalar_tensor(int(gz), dtype=torch.int64, device=x_mask.device)
        symp = _popcount_u64((x_mask & gz_t) ^ (z_mask & gx_t)) & 1
    return torch.nonzero(symp.eq(1), as_tuple=False).flatten().to(torch.int64)


def _cat_states(parts: list[Tensor], *, dim: int, like: Tensor) -> Tensor:
    if not parts:
        return like[:0] if dim == 0 else like[:, :0]
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=dim)


def _apply_gate_state_direct(
    *,
    gate,
    x_mask: Tensor,
    z_mask: Tensor,
    coeffs: Tensor,
    thetas_t: Tensor,
    min_abs: Optional[float],
    max_weight: int,
    weight_x: float,
    weight_y: float,
    weight_z: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    gate_name = gate.__class__.__name__
    if gate_name == "CliffordGate":
        symbol = str(gate.symbol).upper()
        qubits = [int(q) for q in gate.qubits]
        if x_mask.dim() == 2:
            return _CPP_EVAL_BACKEND.apply_clifford_state_mw_cpp(
                symbol,
                qubits,
                x_mask,
                z_mask,
                coeffs,
                min_abs,
                int(max_weight),
                float(weight_x),
                float(weight_y),
                float(weight_z),
            )
        return _CPP_EVAL_BACKEND.apply_clifford_state_cpp(
            symbol,
            qubits,
            x_mask,
            z_mask,
            coeffs,
            min_abs,
            int(max_weight),
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )

    if gate_name == "PauliRotation":
        p_idx = int(getattr(gate, "param_idx", -1))
        if p_idx < 0 or p_idx >= int(thetas_t.shape[0]):
            raise ValueError(f"Invalid param_idx={p_idx} for theta length {int(thetas_t.shape[0])}")
        theta = thetas_t[p_idx]
        if x_mask.dim() == 2:
            gx_words, gz_words = _gate_masks_words(gate, n_words=int(x_mask.shape[1]), device=x_mask.device)
            return _CPP_EVAL_BACKEND.apply_pauli_rotation_state_mw_cpp(
                gx_words,
                gz_words,
                theta,
                x_mask,
                z_mask,
                coeffs,
                min_abs,
                int(max_weight),
                float(weight_x),
                float(weight_y),
                float(weight_z),
            )
        gx, gz = _gate_masks(gate)
        return _CPP_EVAL_BACKEND.apply_pauli_rotation_state_cpp(
            int(gx),
            int(gz),
            theta,
            x_mask,
            z_mask,
            coeffs,
            min_abs,
            int(max_weight),
            float(weight_x),
            float(weight_y),
            float(weight_z),
        )

    raise NotImplementedError(
        "Direct evaluate-only MVP currently supports only CliffordGate and PauliRotation."
    )


def _apply_pauli_rotation_anti_sin(
    *,
    gate,
    x_mask: Tensor,
    z_mask: Tensor,
    coeffs: Tensor,
    thetas_t: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    p_idx = int(getattr(gate, "param_idx", -1))
    if p_idx < 0 or p_idx >= int(thetas_t.shape[0]):
        raise ValueError(f"Invalid param_idx={p_idx} for theta length {int(thetas_t.shape[0])}")
    theta = thetas_t[p_idx]
    if x_mask.dim() == 2:
        gx_words, gz_words = _gate_masks_words(gate, n_words=int(x_mask.shape[1]), device=x_mask.device)
        return _CPP_EVAL_BACKEND.apply_pauli_rotation_anti_sin_mw_cpp(
            gx_words,
            gz_words,
            theta,
            x_mask,
            z_mask,
            coeffs,
        )
    gx, gz = _gate_masks(gate)
    return _CPP_EVAL_BACKEND.apply_pauli_rotation_anti_sin_cpp(
        int(gx),
        int(gz),
        theta,
        x_mask,
        z_mask,
        coeffs,
    )


def _merge_pauli_query_into_base(
    *,
    base_x: Tensor,
    base_z: Tensor,
    query_x: Tensor,
    query_z: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    if base_x.dim() == 2:
        return _CPP_EVAL_BACKEND.merge_pauli_query_into_base_mw_cpp(
            base_x,
            base_z,
            query_x,
            query_z,
        )
    return _CPP_EVAL_BACKEND.merge_pauli_query_into_base_cpp(
        base_x,
        base_z,
        query_x,
        query_z,
    )


def _apply_gate_state_chunked(
    *,
    gate,
    x_mask: Tensor,
    z_mask: Tensor,
    coeffs: Tensor,
    thetas_t: Tensor,
    compute_device: str,
    memory_device: str,
    min_abs: Optional[float],
    max_weight: int,
    weight_x: float,
    weight_y: float,
    weight_z: float,
    chunk_size: int,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    n_terms = int(coeffs.shape[0])
    gate_name = gate.__class__.__name__
    state_device = str(coeffs.device)
    if not _device_matches(str(memory_device), state_device):
        raise ValueError(
            f"state tensors must live on memory_device={memory_device}, got coeffs on {state_device}"
        )

    if gate_name != "PauliRotation" and n_terms <= int(chunk_size):
        chunk_state = _move_state((x_mask, z_mask, coeffs), device=str(compute_device))
        new_x, new_z, new_coeffs = _apply_gate_state_direct(
            gate=gate,
            x_mask=chunk_state[0],
            z_mask=chunk_state[1],
            coeffs=chunk_state[2],
            thetas_t=thetas_t,
            min_abs=min_abs,
            max_weight=max_weight,
            weight_x=weight_x,
            weight_y=weight_y,
            weight_z=weight_z,
        )
        if not _device_matches(str(memory_device), str(new_coeffs.device)):
            new_x, new_z, new_coeffs = _move_state(
                (new_x, new_z, new_coeffs),
                device=str(memory_device),
            )
        return new_x, new_z, new_coeffs, 1

    if gate_name == "CliffordGate":
        x_parts: list[Tensor] = []
        z_parts: list[Tensor] = []
        coeff_parts: list[Tensor] = []
        n_chunks = int(math.ceil(float(n_terms) / float(chunk_size)))
        for chunk_idx in range(n_chunks):
            start = int(chunk_idx * chunk_size)
            end = int(min(start + int(chunk_size), n_terms))
            chunk_state_calc = _apply_gate_state_direct(
                gate=gate,
                x_mask=x_mask[start:end].to(compute_device),
                z_mask=z_mask[start:end].to(compute_device),
                coeffs=coeffs[start:end].to(compute_device),
                thetas_t=thetas_t,
                min_abs=min_abs,
                max_weight=max_weight,
                weight_x=weight_x,
                weight_y=weight_y,
                weight_z=weight_z,
            )
            chunk_state = _move_state(chunk_state_calc, device=str(memory_device))
            if int(chunk_state[2].numel()) == 0:
                continue
            x_parts.append(chunk_state[0])
            z_parts.append(chunk_state[1])
            coeff_parts.append(chunk_state[2])
        new_x = _cat_states(x_parts, dim=0, like=x_mask)
        new_z = _cat_states(z_parts, dim=0, like=z_mask)
        new_coeffs = _cat_states(coeff_parts, dim=0, like=coeffs)
        return new_x, new_z, new_coeffs, n_chunks

    if gate_name == "PauliRotation":
        outer_chunks = int(math.ceil(float(n_terms) / float(chunk_size)))
        anti_row_parts: list[Tensor] = []
        for chunk_idx in range(outer_chunks):
            start = int(chunk_idx * chunk_size)
            end = int(min(start + int(chunk_size), n_terms))
            x_chunk = x_mask[start:end].to(compute_device)
            z_chunk = z_mask[start:end].to(compute_device)
            anti_local = _rotation_anti_local_rows(
                gate=gate,
                x_mask=x_chunk,
                z_mask=z_chunk,
            )
            if int(anti_local.numel()) == 0:
                continue
            anti_row_parts.append(anti_local.to(memory_device) + int(start))

        anti_rows = _cat_states(
            anti_row_parts,
            dim=0,
            like=torch.empty((0,), dtype=torch.int64, device=coeffs.device),
        )
        if int(anti_rows.numel()) == 0:
            new_x = x_mask
            new_z = z_mask
            new_coeffs = coeffs
        else:
            p_idx = int(getattr(gate, "param_idx", -1))
            if p_idx < 0 or p_idx >= int(thetas_t.shape[0]):
                raise ValueError(f"Invalid param_idx={p_idx} for theta length {int(thetas_t.shape[0])}")
            cos_t_mem = torch.cos(
                thetas_t[p_idx].to(dtype=coeffs.dtype, device=memory_device)
            )

            same_coeffs = coeffs.clone()
            same_coeffs.index_copy_(
                0,
                anti_rows,
                coeffs.index_select(0, anti_rows) * cos_t_mem,
            )

            anti_x = x_mask.index_select(0, anti_rows)
            anti_z = z_mask.index_select(0, anti_rows)
            anti_base_x = anti_x
            anti_base_z = anti_z
            if not _device_matches(str(compute_device), str(anti_x.device)):
                anti_base_x = anti_x.to(compute_device, non_blocking=True)
                anti_base_z = anti_z.to(compute_device, non_blocking=True)

            novel_x_parts: list[Tensor] = []
            novel_z_parts: list[Tensor] = []
            novel_coeff_parts: list[Tensor] = []
            anti_terms = int(anti_rows.numel())
            n_anti_same = int(anti_terms)
            anti_chunks = int(math.ceil(float(anti_terms) / float(chunk_size)))

            for chunk_idx in range(anti_chunks):
                start = int(chunk_idx * chunk_size)
                end = int(min(start + int(chunk_size), anti_terms))
                anti_rows_chunk = anti_rows[start:end]
                anti_x_chunk = x_mask.index_select(0, anti_rows_chunk).to(compute_device)
                anti_z_chunk = z_mask.index_select(0, anti_rows_chunk).to(compute_device)
                anti_coeff_chunk = coeffs.index_select(0, anti_rows_chunk).to(compute_device)
                sin_x, sin_z, sin_coeffs = _apply_pauli_rotation_anti_sin(
                    gate=gate,
                    x_mask=anti_x_chunk,
                    z_mask=anti_z_chunk,
                    coeffs=anti_coeff_chunk,
                    thetas_t=thetas_t,
                )

                merged_anti_x, merged_anti_z, row_local = _merge_pauli_query_into_base(
                    base_x=anti_base_x,
                    base_z=anti_base_z,
                    query_x=sin_x,
                    query_z=sin_z,
                )

                if not _device_matches(str(memory_device), str(sin_coeffs.device)):
                    sin_coeffs = sin_coeffs.to(memory_device)
                if not _device_matches(str(memory_device), str(row_local.device)):
                    row_local = row_local.to(memory_device)

                hit_mask = row_local.lt(n_anti_same)
                if bool(hit_mask.any().item()):
                    hit_rows_global = anti_rows.index_select(0, row_local[hit_mask])
                    same_coeffs.index_add_(
                        0,
                        hit_rows_global,
                        sin_coeffs[hit_mask],
                    )
                miss_mask = ~hit_mask
                n_novel_local = int(merged_anti_x.shape[0]) - n_anti_same
                if n_novel_local > 0:
                    novel_x_local = merged_anti_x[n_anti_same:]
                    novel_z_local = merged_anti_z[n_anti_same:]
                    if not _device_matches(str(memory_device), str(novel_x_local.device)):
                        novel_x_local = novel_x_local.to(memory_device)
                        novel_z_local = novel_z_local.to(memory_device)
                    novel_coeff_local = torch.zeros(
                        (n_novel_local,),
                        dtype=sin_coeffs.dtype,
                        device=memory_device,
                    )
                    if bool(miss_mask.any().item()):
                        novel_local_rows = row_local[miss_mask] - int(n_anti_same)
                        novel_coeff_local.index_add_(0, novel_local_rows, sin_coeffs[miss_mask])
                    novel_x_parts.append(novel_x_local)
                    novel_z_parts.append(novel_z_local)
                    novel_coeff_parts.append(novel_coeff_local)

            new_x = _cat_states([x_mask, *novel_x_parts], dim=0, like=x_mask)
            new_z = _cat_states([z_mask, *novel_z_parts], dim=0, like=z_mask)
            new_coeffs = _cat_states([same_coeffs, *novel_coeff_parts], dim=0, like=coeffs)

        new_x, new_z, new_coeffs = _apply_state_filters(
            new_x,
            new_z,
            new_coeffs,
            min_abs=min_abs,
            max_weight=max_weight,
            weight_x=weight_x,
            weight_y=weight_y,
            weight_z=weight_z,
        )
        return new_x, new_z, new_coeffs, outer_chunks

    n_chunks = int(math.ceil(float(n_terms) / float(chunk_size)))
    merged_state: Optional[Tuple[Tensor, Tensor, Tensor]] = None

    for chunk_idx in range(n_chunks):
        start = int(chunk_idx * chunk_size)
        end = int(min(start + int(chunk_size), n_terms))
        chunk_state_calc = _apply_gate_state_direct(
            gate=gate,
            x_mask=x_mask[start:end].to(compute_device),
            z_mask=z_mask[start:end].to(compute_device),
            coeffs=coeffs[start:end].to(compute_device),
            thetas_t=thetas_t,
            min_abs=None,
            max_weight=max_weight,
            weight_x=weight_x,
            weight_y=weight_y,
            weight_z=weight_z,
        )
        chunk_state = _move_state(chunk_state_calc, device=str(memory_device))
        if merged_state is None:
            merged_state = chunk_state
        else:
            merged_state = _merge_state_pair(
                merged_state,
                chunk_state,
                min_abs=None,
                max_weight=max_weight,
                weight_x=weight_x,
                weight_y=weight_y,
                weight_z=weight_z,
            )

    if merged_state is None:
        return x_mask[:0], z_mask[:0], coeffs[:0], n_chunks

    new_x, new_z, new_coeffs = _apply_state_filters(
        merged_state[0],
        merged_state[1],
        merged_state[2],
        min_abs=min_abs,
        max_weight=max_weight,
        weight_x=weight_x,
        weight_y=weight_y,
        weight_z=weight_z,
    )
    return new_x, new_z, new_coeffs, n_chunks


def evaluate_expval_direct_observable(
    *,
    circuit,
    observable,
    thetas: Any,
    memory_device: str,
    compute_device: str,
    dtype: str | Any,
    min_abs: Optional[float],
    max_weight: int,
    weight_x: float,
    weight_y: float,
    weight_z: float,
    show_progress: bool = True,
    chunk_size: int = 1_000_000,
) -> Tensor:
    _require_eval_backend()
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    t_dtype = _resolve_t_dtype(dtype)
    thetas_t = torch.as_tensor(thetas, dtype=t_dtype, device=compute_device)
    if thetas_t.numel() == 0:
        thetas_t = torch.zeros((1,), dtype=t_dtype, device=compute_device)

    x_mask, z_mask, coeffs = _observable_to_state(observable=observable, device=memory_device, dtype=t_dtype)
    circuit_rev = list(circuit)
    circuit_rev.reverse()

    pbar = tqdm(
        circuit_rev,
        total=len(circuit_rev),
        desc="direct-eval",
        dynamic_ncols=True,
        disable=not bool(show_progress),
    )
    total_gates = len(circuit_rev)

    for step_idx, gate in enumerate(pbar):
        if int(coeffs.numel()) == 0:
            if bool(show_progress):
                pbar.set_postfix_str("terms=0", refresh=False)
            break

        gate_name = gate.__class__.__name__
        x_mask, z_mask, coeffs, n_chunks = _apply_gate_state_chunked(
            gate=gate,
            x_mask=x_mask,
            z_mask=z_mask,
            coeffs=coeffs,
            thetas_t=thetas_t,
            compute_device=str(compute_device),
            memory_device=str(memory_device),
            min_abs=min_abs,
            max_weight=max_weight,
            weight_x=weight_x,
            weight_y=weight_y,
            weight_z=weight_z,
            chunk_size=int(chunk_size),
        )

        if bool(show_progress) and ((step_idx % 8 == 0) or (step_idx == total_gates - 1)):
            postfix = f"terms={int(coeffs.numel())} gate={gate_name}"
            if int(n_chunks) > 1:
                postfix += f" chunks={int(n_chunks)}"
            pbar.set_postfix_str(postfix, refresh=False)

    if bool(show_progress) and int(coeffs.numel()) > 0:
        pbar.set_postfix_str(f"terms={int(coeffs.numel())}", refresh=False)

    if int(coeffs.numel()) == 0:
        return torch.zeros((), dtype=t_dtype, device=compute_device)

    diag_mask = (x_mask == 0) if x_mask.dim() == 1 else (x_mask == 0).all(dim=1)
    if int(diag_mask.numel()) == 0:
        return torch.zeros((), dtype=t_dtype, device=compute_device)
    return coeffs[diag_mask].sum().to(compute_device)


__all__ = [
    "CPPEvalOnlyBackendUnavailableError",
    "evaluate_expval_direct_observable",
]
