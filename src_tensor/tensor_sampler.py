"""Quasi-probability utilities built on the tensor sparse backend.

This module provides `TensorSparseSampler`, a lightweight helper to:
  1) compute requested Z-string correlators <prod_{i in S} Z_i>
     using the existing union-basis + adjoint workflow
  2) evaluate truncated quasi-probabilities q^{(k)}(x) for batches of bitstrings x
     via the Walsh/Fourier expansion over those correlators

For order k:

  q^{(k)}(x) = 2^{-n} * sum_{S: |S|<=k} (-1)^{sum_{i in S} x_i} <Z_S>

where x_i in {0,1} and <Z_∅> := 1.

Notes
-----
- This is *not* a Born-rule sampler. q^{(k)} can be negative.
- The user supplies the list of subsets S (Z-combos) required by downstream
  quasi-probability queries, including high-order subsets when needed.
- Primary input format for x is a dense 0/1 tensor of shape (B, n_qubits),
  which matches how samples appear elsewhere in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from numbers import Integral
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple, Union, cast

import numpy as np

torch: Any
try:
    import torch as _torch

    torch = _torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = cast(Any, None)
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from torch import Tensor
else:  # pragma: no cover
    Tensor = Any

from src.pauli_surrogate_python import PauliSum

from .tensor_adjoint import (
    UnionBasis,
    adjoint_weights_on_zero,
    coeff_matrix_from_observables,
    expvals_from_w_and_coeff_matrix,
    propagate_union_basis_psum,
)
from .tensor_types import TensorPauliSum


Subset = Sequence[int]
ZCombosInput = Union[
    Sequence[Subset],
    Mapping[Union[int, str], Sequence[Subset]],
    Tuple[Sequence[Subset], ...],
]


@dataclass(frozen=True)
class ZComboSpec:
    """Normalized Z-combos grouped by subset order."""

    by_order: Dict[int, List[Tuple[int, ...]]]

    @property
    def orders(self) -> List[int]:
        return sorted(int(k) for k in self.by_order.keys())

    @property
    def singles(self) -> List[Tuple[int, ...]]:
        return list(self.by_order.get(1, []))

    @property
    def pairs(self) -> List[Tuple[int, ...]]:
        return list(self.by_order.get(2, []))

    @property
    def triples(self) -> List[Tuple[int, ...]]:
        return list(self.by_order.get(3, []))

    @property
    def n_obs(self) -> int:
        return sum(len(v) for v in self.by_order.values())


def _normalize_subset(subset: Subset) -> Tuple[int, ...]:
    t = tuple(int(i) for i in subset)
    if len(t) == 0:
        raise ValueError("Empty subset is not allowed (use implicit identity term)")
    if len(set(t)) != len(t):
        raise ValueError(f"Subset has duplicate indices: {subset}")
    return t


def normalize_z_combos(z_combos: ZCombosInput, *, max_order: Optional[int] = None) -> ZComboSpec:
    """Normalize user-provided Z-combos.

    Supported input forms:
      - list of subsets: [[i], [i,j], [i,j,k], ...]
      - dict keyed by order: {1: [...], 2: [...], 3: [...], ...} (keys may be str)
      - tuple/list of per-order groups, where element i is order-(i+1) subsets

    If max_order is provided, only subset sizes 1..max_order are retained.
    """

    order_cap: Optional[int] = None
    if max_order is not None:
        if int(max_order) < 1:
            raise ValueError("max_order must be >= 1")
        order_cap = int(max_order)

    by_order: Dict[int, List[Tuple[int, ...]]] = {}

    def _add(subset: Subset) -> None:
        t = _normalize_subset(subset)
        k = len(t)
        if order_cap is not None and k > order_cap:
            return
        by_order.setdefault(k, []).append(t)

    if isinstance(z_combos, Mapping):
        for k, subsets in z_combos.items():
            kk = int(k)
            if kk < 1 or (order_cap is not None and kk > order_cap):
                continue
            for s in cast(Sequence[Subset], subsets):
                _add(s)
    elif (
        isinstance(z_combos, tuple)
        and len(z_combos) > 0
        and isinstance(z_combos[0], (list, tuple))
        and len(cast(Sequence[Any], z_combos[0])) > 0
        and not isinstance(cast(Sequence[Any], z_combos[0])[0], Integral)
    ):
        for order, subsets in enumerate(cast(Tuple[Sequence[Subset], ...], z_combos), start=1):
            if order_cap is not None and order > order_cap:
                break
            for s in subsets:
                _add(s)
    else:
        for s in cast(Sequence[Subset], z_combos):
            _add(s)

    return ZComboSpec(by_order=by_order)


def _bitstrings_to_tensor(x_batch: Any, *, n_qubits: int, device: str) -> Tensor:
    """Convert common bitstring batch formats to a (B, n_qubits) 0/1 tensor."""

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for TensorSparseSampler")

    if _TORCH_AVAILABLE and isinstance(x_batch, torch.Tensor):
        x = x_batch
    elif isinstance(x_batch, np.ndarray):
        x = torch.from_numpy(x_batch)
    elif isinstance(x_batch, (list, tuple)) and len(x_batch) > 0 and isinstance(x_batch[0], str):
        # list[str] like ["0101", ...]
        rows = []
        for s in x_batch:
            s_clean = s.strip()
            bad_chars = {ch for ch in s_clean if ch not in {"0", "1"}}
            if bad_chars:
                raise ValueError(
                    f"Bitstring contains non-binary characters {sorted(bad_chars)}: {s!r}"
                )
            rows.append([1 if ch == "1" else 0 for ch in s_clean])
        x = torch.tensor(rows, dtype=torch.uint8)
    else:
        x = torch.tensor(x_batch)

    if x.dim() == 1:
        # Single bitstring.
        x = x.unsqueeze(0)

    if x.shape[1] != int(n_qubits):
        raise ValueError(f"x_batch must have shape (B, {n_qubits}); got {tuple(x.shape)}")

    # Force to 0/1 integer-like type.
    if x.dtype == torch.bool:
        x01 = x.to(torch.uint8)
    else:
        x01 = x.to(torch.uint8)

    if not torch.all((x01 == 0) | (x01 == 1)):
        raise ValueError("x_batch must contain only 0/1 values")

    # Keep on requested device.
    return x01.to(device)


def _shrink_union_basis(basis: UnionBasis, keep_mask_in: Tensor) -> UnionBasis:
    """Shrink UnionBasis to match a pruned input coefficient dimension."""

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")

    keep_idx = torch.nonzero(torch.as_tensor(keep_mask_in, device="cpu"), as_tuple=False).flatten().tolist()
    pstrs2 = [basis.pstrs[i] for i in keep_idx]
    index2 = {p: i for i, p in enumerate(pstrs2)}
    return UnionBasis(n_qubits=basis.n_qubits, pstrs=pstrs2, index=index2)

class TensorSparseSampler:
    """Compute Z-moments and truncated quasi-probabilities q^(k)(x) for any k."""

    def __init__(
        self,
        *,
        n_qubits: int,
        circuit: Sequence,
        z_combos: ZCombosInput,
        max_order: Optional[int] = None,
        build_device: str = "cpu",
        dtype: str = "float64",
        max_weight: int = 8,
        max_xy: int = 1_000_000_000,
        step_device: str = "cpu",
        offload_steps: bool = False,
        offload_keep: int = 1,
        build_thetas: Any = None,
        build_min_abs: Optional[float] = None,
        build_min_mat_abs: Optional[float] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TensorSparseSampler")

        self.n_qubits = int(n_qubits)
        self.circuit = list(circuit)
        self.dtype_str = str(dtype).lower()
        self.max_order = int(max_order) if max_order is not None else None

        self.z_spec = normalize_z_combos(z_combos, max_order=self.max_order)
        if self.z_spec.n_obs == 0:
            raise ValueError("z_combos must include at least one subset")
        if self.max_order is None:
            self.max_order = max(self.z_spec.orders, default=0)

        self._obs: List[PauliSum] = self._build_observables(self.z_spec)

        # Build union-basis propagation once. This is the expensive part.
        self.psum_union, self.basis = propagate_union_basis_psum(
            circuit=self.circuit,
            observables=self._obs,
            device=build_device,
            dtype=self.dtype_str,
            max_weight=int(max_weight),
            max_xy=int(max_xy),
            offload_steps=bool(offload_steps),
            offload_keep=int(offload_keep),
            step_device=str(step_device),
            thetas=build_thetas,
            min_abs=build_min_abs,
            min_mat_abs=build_min_mat_abs,
        )

        # Back-propagating zero-filter: keep only output diagonal terms (x_mask==0)
        # and prune any upstream nodes that cannot contribute.
        from .tensor_propagate import zero_filter_tensor_backprop_with_keep_mask

        K0 = int(self.psum_union.coeff_init.numel())
        n_out0 = int(self.psum_union.x_mask.shape[0]) if getattr(self.psum_union.x_mask, "dim", lambda: 1)() >= 1 else 0
        t_z0 = time.perf_counter()
        self.psum_union, keep_mask_in = zero_filter_tensor_backprop_with_keep_mask(
            self.psum_union,
            stream_device=str(build_device) if str(build_device) != "cpu" else None,
            offload_back=(str(build_device) != "cpu"),
        )
        t_z1 = time.perf_counter()
        self.basis = _shrink_union_basis(self.basis, keep_mask_in)

        K1 = int(self.psum_union.coeff_init.numel())
        n_out1 = int(self.psum_union.x_mask.shape[0]) if getattr(self.psum_union.x_mask, "dim", lambda: 1)() >= 1 else 0
        if K0 > 0 and n_out0 > 0:
            print(
                "zero_filter_backprop: "
                f"K {K0}->{K1} ({K1 / K0:.3f}), "
                f"n_out {n_out0}->{n_out1} ({n_out1 / n_out0:.3f}), "
                f"dt={t_z1 - t_z0:.3f}s"
            )

        # Caches keyed by (device_str, torch_dtype)
        self._V0_cache: Dict[Tuple[str, str], Tensor] = {}
        self._idx_cache: Dict[str, Dict[int, Tensor]] = {}

    def _build_observables(self, z_spec: ZComboSpec) -> List[PauliSum]:
        obs: List[PauliSum] = []

        def _mk(sub: Tuple[int, ...]) -> PauliSum:
            ps = PauliSum(self.n_qubits)
            ps.add_from_str("Z" * len(sub), 1.0, qubits=list(sub))
            return ps

        for order in z_spec.orders:
            for s in z_spec.by_order[order]:
                obs.append(_mk(s))
        return obs

    def _get_V0(self, *, device: str, dtype: Any) -> Tensor:
        key = (str(device), str(dtype))
        V0 = self._V0_cache.get(key)
        if V0 is None:
            V0 = coeff_matrix_from_observables(index=self.basis.index, observables=self._obs, device=str(device), dtype=dtype)
            self._V0_cache[key] = V0
        return V0

    def _get_subset_indices(self, *, device: str) -> Dict[int, Tensor]:
        cache = self._idx_cache.get(str(device))
        if cache is not None:
            return cast(Dict[int, Tensor], cache)

        idx: Dict[int, Tensor] = {}
        for order in self.z_spec.orders:
            combos = self.z_spec.by_order[order]
            if len(combos) == 0:
                continue
            if order == 1:
                idx[order] = torch.tensor([s[0] for s in combos], dtype=torch.long, device=device)
            else:
                idx[order] = torch.tensor(combos, dtype=torch.long, device=device)

        self._idx_cache[str(device)] = idx
        return idx

    @property
    def n_observables(self) -> int:
        return self.z_spec.n_obs

    def compute_moments(
        self,
        thetas: Any,
        *,
        stream_device: Optional[str] = None,
        offload_back: bool = True,
    ) -> Tensor:
        """Return moments vector aligned with the normalized combo order.

        Order is by subset size (1,2,3,...) and preserves input order within
        each subset-size group.
        """

        w = adjoint_weights_on_zero(self.psum_union, thetas, stream_device=stream_device, offload_back=offload_back)
        V0 = self._get_V0(device=str(w.device), dtype=w.dtype)
        return expvals_from_w_and_coeff_matrix(w, V0)

    def quasi_prob_from_moments(
        self,
        x_batch: Any,
        moments: Tensor,
        *,
        order: int = 3,
    ) -> Tensor:
        """Compute q^(order)(x) for a batch of bitstrings x using precomputed moments."""

        if int(order) < 0:
            raise ValueError("order must be >= 0")
        if int(order) > int(self.max_order):
            raise ValueError(f"order={order} exceeds sampler max_order={self.max_order}")
        if moments.dim() != 1 or int(moments.shape[0]) != int(self.n_observables):
            raise ValueError(f"moments must be 1D of length {self.n_observables}; got {tuple(moments.shape)}")

        device = str(moments.device)
        x01 = _bitstrings_to_tensor(x_batch, n_qubits=self.n_qubits, device=device)

        # s_i = (-1)^{x_i} = 1 - 2*x_i
        s = (1.0 - 2.0 * x01.to(dtype=moments.dtype)).to(device)

        B = int(s.shape[0])
        q = torch.ones((B,), dtype=moments.dtype, device=device)

        idx = self._get_subset_indices(device=device)

        offset = 0
        for subset_order in self.z_spec.orders:
            n_terms = len(self.z_spec.by_order[subset_order])
            if n_terms == 0:
                continue
            m_k = moments[offset : offset + n_terms]
            if order >= subset_order:
                if subset_order == 1:
                    prod_k = s[:, idx[subset_order]]
                else:
                    prod_k = s[:, idx[subset_order]].prod(dim=-1)
                q = q + prod_k @ m_k
            offset += n_terms

        return q / float(2 ** self.n_qubits)

    def quasi_prob(
        self,
        x_batch: Any,
        thetas: Any,
        *,
        order: int = 3,
        stream_device: Optional[str] = None,
        offload_back: bool = True,
    ) -> Tensor:
        """Convenience: compute moments then return q^(order)(x_batch)."""

        moments = self.compute_moments(thetas, stream_device=stream_device, offload_back=offload_back)
        return self.quasi_prob_from_moments(x_batch, moments, order=order)


__all__ = [
    "ZComboSpec",
    "normalize_z_combos",
    "TensorSparseSampler",
]
