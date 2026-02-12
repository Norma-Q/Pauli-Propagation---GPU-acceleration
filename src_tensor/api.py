"""High-level user API for tensor surrogate workflows.

This module provides a fixed, memory-first execution path for common use:
  1) compile once from circuit + observables
  2) evaluate expvals for theta vectors
  3) build custom training loops with PyTorch autograd

Default preset `gpu_min` prioritizes low GPU memory pressure by building and
storing sparse steps on CPU, then streaming computation to GPU for evaluation.
Preset `gpu_full` prioritizes exact/no-truncation workflows by keeping build,
steps, and eval on GPU with conservative precision defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, cast
import numpy as np

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

from .tensor_adjoint import (
    UnionBasis,
    adjoint_weights_on_zero,
    coeff_matrix_from_observables,
    expvals_from_w_and_coeff_matrix,
    propagate_union_basis_psum,
)
from .tensor_sampler import TensorSparseSampler
from .tensor_types import TensorPauliSum


@dataclass(frozen=True)
class TensorSurrogatePreset:
    """Execution preset for high-level API."""

    build_device: str = "cpu"
    step_device: str = "cpu"
    stream_device: str = "cuda"
    dtype: str = "float32"
    max_weight: int = 8
    max_xy: int = 1_000_000_000
    offload_steps: bool = True
    offload_keep: int = 1
    offload_back: bool = True


DEFAULT_PRESETS: Dict[str, TensorSurrogatePreset] = {
    "gpu_min": TensorSurrogatePreset(
        build_device="cuda",
        step_device="cpu",
        stream_device="cuda",
        dtype="float32",
        max_weight=5,
        max_xy=1_000_000_000,
        offload_steps=True,
        offload_keep=1,
        offload_back=True,
    ),
    "gpu_full": TensorSurrogatePreset(
        build_device="cuda",
        step_device="cpu",
        stream_device="cuda",
        dtype="float64",
        max_weight=1_000_000_000,
        max_xy=1_000_000_000,
        offload_steps=True,
        offload_keep=1,
        offload_back=True,
    )
}


@dataclass
class CompiledTensorSurrogate:
    """Compiled surrogate program for repeated evaluation/training."""

    circuit: List[Any]
    psum_union: TensorPauliSum
    basis: UnionBasis
    observables: List[Any]
    preset_name: str
    preset: TensorSurrogatePreset
    _V0_cache: Dict[Tuple[str, str], Tensor]

    def _get_V0(self, *, device: str, dtype: Any) -> Tensor:
        key = (str(device), str(dtype))
        V0 = self._V0_cache.get(key)
        if V0 is None:
            V0 = coeff_matrix_from_observables(
                index=self.basis.index,
                observables=self.observables,
                device=str(device),
                dtype=dtype,
            )
            self._V0_cache[key] = V0
        return V0

    def expvals(
        self,
        thetas: Any,
        *,
        stream_device: Optional[str] = None,
        offload_back: Optional[bool] = None,
    ) -> Tensor:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor backend.")

        stream = _resolve_stream_device(
            self.preset.stream_device if stream_device is None else stream_device
        )
        back = self.preset.offload_back if offload_back is None else bool(offload_back)

        w = adjoint_weights_on_zero(
            self.psum_union,
            thetas,
            stream_device=stream,
            offload_back=back,
        )
        V0 = self._get_V0(device=str(w.device), dtype=w.dtype)
        return expvals_from_w_and_coeff_matrix(w, V0)

    def expval(
        self,
        thetas: Any,
        *,
        obs_index: int = 0,
        stream_device: Optional[str] = None,
        offload_back: Optional[bool] = None,
    ) -> Tensor:
        vals = self.expvals(thetas, stream_device=stream_device, offload_back=offload_back)
        if obs_index < 0 or obs_index >= int(vals.shape[0]):
            raise IndexError(f"obs_index={obs_index} is out of range for {int(vals.shape[0])} observables")
        return vals[obs_index]

    def expvals_pennylane(self, thetas: Any, *, max_qubits: int = 20) -> Tensor:
        """Reference expvals via PennyLane (small circuits only)."""
        return pennylane_expvals_small(
            circuit=self.circuit,
            observables=self.observables,
            thetas=thetas,
            n_qubits=int(self.basis.n_qubits),
            max_qubits=max_qubits,
        )


def _resolve_stream_device(stream_device: str) -> Optional[str]:
    dev = str(stream_device)
    if dev.startswith("cuda"):
        if _TORCH_AVAILABLE and bool(torch.cuda.is_available()):
            return dev
        return None
    return dev


def _require_pennylane():
    try:
        import pennylane as qml  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("PennyLane is required for this API path.") from e
    return qml


def _thetas_to_numpy(thetas: Any) -> np.ndarray:
    if _TORCH_AVAILABLE and isinstance(thetas, torch.Tensor):
        arr = thetas.detach().cpu().numpy()
    else:
        arr = np.asarray(thetas)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _validate_small_n(n_qubits: int, max_qubits: int) -> None:
    n = int(n_qubits)
    if n < 1:
        raise ValueError("n_qubits must be >= 1")
    if n > int(max_qubits):
        raise ValueError(
            f"PennyLane conversion path is limited to <= {int(max_qubits)} qubits; got n_qubits={n}"
        )


def _apply_circuit_pennylane(circuit: Sequence[Any], thetas_np: np.ndarray, qml: Any) -> None:
    for gate in circuit:
        gate_name = gate.__class__.__name__
        if gate_name == "CliffordGate":
            symbol = str(gate.symbol).upper()
            if symbol == "H":
                qml.Hadamard(wires=gate.qubits[0])
            elif symbol == "S":
                qml.S(wires=gate.qubits[0])
            elif symbol == "X":
                qml.PauliX(wires=gate.qubits[0])
            elif symbol == "Y":
                qml.PauliY(wires=gate.qubits[0])
            elif symbol == "Z":
                qml.PauliZ(wires=gate.qubits[0])
            elif symbol == "SX":
                qml.SX(wires=gate.qubits[0])
            elif symbol == "CNOT":
                qml.CNOT(wires=gate.qubits)
            elif symbol == "CZ":
                qml.CZ(wires=gate.qubits)
            elif symbol == "SWAP":
                qml.SWAP(wires=gate.qubits)
            else:
                raise ValueError(f"Unsupported CliffordGate symbol for PennyLane conversion: {symbol}")
            continue

        if gate_name == "PauliRotation":
            pidx = int(gate.param_idx)
            if pidx < 0 or pidx >= int(thetas_np.shape[0]):
                raise ValueError(
                    f"Invalid param_idx={pidx} for theta length {int(thetas_np.shape[0])}"
                )
            qml.PauliRot(float(thetas_np[pidx]), str(gate.pauli), wires=list(gate.qubits))
            continue

        raise TypeError(f"Unsupported gate type for PennyLane conversion: {gate_name}")


def _qml_op_from_paulistring(pstr: Any, n_qubits: int, qml: Any):
    ops: List[Any] = []
    x_mask = int(pstr.x_mask)
    z_mask = int(pstr.z_mask)
    for q in range(int(n_qubits)):
        xb = (x_mask >> q) & 1
        zb = (z_mask >> q) & 1
        if xb == 0 and zb == 0:
            continue
        if xb == 1 and zb == 0:
            ops.append(qml.X(q))
        elif xb == 0 and zb == 1:
            ops.append(qml.Z(q))
        else:
            ops.append(qml.Y(q))
    if len(ops) == 0:
        return qml.Identity(0)
    if len(ops) == 1:
        return ops[0]
    return qml.prod(*ops)


def _qml_obs_from_paulisum(obs: Any, qml: Any):
    terms = list(obs.terms.items())
    if len(terms) == 0:
        raise ValueError("Observable has no terms")
    op_sum: Any = None
    for p, c in terms:
        c_f = float(c)
        term = c_f * _qml_op_from_paulistring(p, int(obs.n_qubits), qml)
        op_sum = term if op_sum is None else (op_sum + term)
    return op_sum


def pennylane_expvals_small(
    *,
    circuit: Sequence[Any],
    observables: Sequence[Any],
    thetas: Any,
    n_qubits: int,
    max_qubits: int = 20,
) -> Tensor:
    """Reference expvals using PennyLane for small circuits (n_qubits <= max_qubits)."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    qml = _require_pennylane()
    _validate_small_n(n_qubits, max_qubits)

    thetas_np = _thetas_to_numpy(thetas)
    obs_ops = [_qml_obs_from_paulisum(obs, qml) for obs in observables]
    dev = qml.device("default.qubit", wires=int(n_qubits))

    @qml.qnode(dev)
    def qnode(params):
        _apply_circuit_pennylane(circuit, params, qml)
        return [qml.expval(op) for op in obs_ops]

    vals = np.asarray(qnode(thetas_np), dtype=np.float64)
    return torch.as_tensor(vals, dtype=torch.float64)


def pennylane_sample_small(
    *,
    circuit: Sequence[Any],
    thetas: Any,
    n_qubits: int,
    shots: int = 1000,
    max_qubits: int = 20,
    seed: Optional[int] = None,
) -> Tensor:
    """Reference sampling using PennyLane for small circuits (n_qubits <= max_qubits)."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    if int(shots) <= 0:
        raise ValueError("shots must be > 0")
    qml = _require_pennylane()
    _validate_small_n(n_qubits, max_qubits)

    thetas_np = _thetas_to_numpy(thetas)
    dev = qml.device("default.qubit", wires=int(n_qubits), shots=int(shots), seed=seed)

    @qml.qnode(dev)
    def qnode(params):
        _apply_circuit_pennylane(circuit, params, qml)
        return qml.sample(wires=list(range(int(n_qubits))))

    samples = np.asarray(qnode(thetas_np), dtype=np.uint8)
    return torch.as_tensor(samples, dtype=torch.uint8)


def _bit_reverse(x: int, n_qubits: int) -> int:
    y = 0
    for i in range(int(n_qubits)):
        y = (y << 1) | ((int(x) >> i) & 1)
    return int(y)


def pennylane_probs_small(
    *,
    circuit: Sequence[Any],
    thetas: Any,
    n_qubits: int,
    max_qubits: int = 20,
    seed: Optional[int] = None,
    bit_order: str = "le",
) -> Tensor:
    """Reference exact probabilities via PennyLane for small circuits.

    Returns a length-2**n tensor of probabilities.

    `bit_order` controls indexing convention of the returned vector:
      - "be": PennyLane basis order for wires [0..n-1]
      - "le": project convention where integer codes use little-endian bits
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    qml = _require_pennylane()
    _validate_small_n(int(n_qubits), max_qubits)

    thetas_np = _thetas_to_numpy(thetas)
    dev = qml.device("default.qubit", wires=int(n_qubits), seed=seed)

    @qml.qnode(dev)
    def qnode(params):
        _apply_circuit_pennylane(circuit, params, qml)
        return qml.probs(wires=list(range(int(n_qubits))))

    p_be = np.asarray(qnode(thetas_np), dtype=np.float64).reshape(-1)
    if str(bit_order).lower() == "be":
        p_out = p_be
    elif str(bit_order).lower() == "le":
        dim = 1 << int(n_qubits)
        perm = np.fromiter((_bit_reverse(i, int(n_qubits)) for i in range(dim)), dtype=np.int64, count=dim)
        p_out = p_be[perm]
    else:
        raise ValueError("bit_order must be 'le' or 'be'")

    return torch.as_tensor(p_out, dtype=torch.float64)


def pennylane_reference(
    program_or_sampler: Any,
    thetas: Any,
    *,
    max_qubits: int = 20,
    shots: int = 1000,
    seed: Optional[int] = None,
) -> Tensor:
    """Unified PennyLane reference path.

    - CompiledTensorSurrogate -> returns expvals.
    - TensorSparseSampler -> returns sampled bitstrings.
    """
    if isinstance(program_or_sampler, CompiledTensorSurrogate):
        return program_or_sampler.expvals_pennylane(thetas, max_qubits=max_qubits)
    if isinstance(program_or_sampler, TensorSparseSampler):
        return pennylane_sample_small(
            circuit=program_or_sampler.circuit,
            thetas=thetas,
            n_qubits=int(program_or_sampler.n_qubits),
            shots=shots,
            max_qubits=max_qubits,
            seed=seed,
        )
    raise TypeError(
        "program_or_sampler must be CompiledTensorSurrogate or TensorSparseSampler for pennylane_reference"
    )


def pennylane_reference_probs(
    program_or_sampler: Any,
    thetas: Any,
    *,
    max_qubits: int = 20,
    seed: Optional[int] = None,
    bit_order: str = "le",
) -> Tensor:
    """Unified PennyLane reference path for exact model probabilities.

    - TensorSparseSampler -> returns qml.probs for the sampler's circuit.

    Notes:
      - This is only intended for small n (<= max_qubits).
      - `bit_order="le"` matches the project's integer-code convention.
    """
    if isinstance(program_or_sampler, TensorSparseSampler):
        return pennylane_probs_small(
            circuit=program_or_sampler.circuit,
            thetas=thetas,
            n_qubits=int(program_or_sampler.n_qubits),
            max_qubits=max_qubits,
            seed=seed,
            bit_order=bit_order,
        )
    raise TypeError("program_or_sampler must be TensorSparseSampler for pennylane_reference_probs")


def _normalize_observables(observables: Any) -> List[Any]:
    if isinstance(observables, Sequence) and not isinstance(observables, (str, bytes)):
        obs_list = list(observables)
    else:
        obs_list = [observables]

    if len(obs_list) == 0:
        raise ValueError("observables must be non-empty")
    for obs in obs_list:
        if not hasattr(obs, "n_qubits") or not hasattr(obs, "terms"):
            raise TypeError("Each observable must have `n_qubits` and `terms`")
    return obs_list


def _shrink_union_basis(basis: UnionBasis, keep_mask_in: Tensor) -> UnionBasis:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")
    keep_idx = torch.nonzero(torch.as_tensor(keep_mask_in, device="cpu"), as_tuple=False).flatten().tolist()
    pstrs2 = [basis.pstrs[i] for i in keep_idx]
    index2 = {p: i for i, p in enumerate(pstrs2)}
    return UnionBasis(n_qubits=basis.n_qubits, pstrs=pstrs2, index=index2)


def resolve_preset(name: str = "gpu_min", *, overrides: Optional[Mapping[str, Any]] = None) -> TensorSurrogatePreset:
    base = DEFAULT_PRESETS.get(str(name))
    if base is None:
        allowed = ", ".join(sorted(DEFAULT_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {allowed}")

    if not overrides:
        return base

    values = dict(base.__dict__)
    for k, v in dict(overrides).items():
        if k not in values:
            raise ValueError(f"Unknown preset override key: {k}")
        values[k] = v
    return TensorSurrogatePreset(**values)


def compile_expval_program(
    *,
    circuit,
    observables: Any,
    preset: str = "gpu_min",
    preset_overrides: Optional[Mapping[str, Any]] = None,
    build_thetas: Any = None,
    build_min_abs: Optional[float] = None,
    build_min_mat_abs: Optional[float] = None,
) -> CompiledTensorSurrogate:
    """Compile a reusable expval program with fixed memory-first flow.

    Fixed internal flow:
      1) propagate union basis
      2) zero-filter with backprop pruning
      3) shrink union basis to pruned input space

    Optional `build_thetas` + `build_min_abs` enable theta-dependent pruning
    during compile. This improves size/runtime only near that theta region.
    Optional `build_min_mat_abs` prunes sparse step matrix entries by |M_ij|.
    """

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for tensor backend.")

    obs_list = _normalize_observables(observables)
    cfg = resolve_preset(preset, overrides=preset_overrides)

    psum_union, basis = propagate_union_basis_psum(
        circuit=circuit,
        observables=obs_list,
        max_weight=int(cfg.max_weight),
        max_xy=int(cfg.max_xy),
        device=str(cfg.build_device),
        dtype=str(cfg.dtype),
        offload_steps=bool(cfg.offload_steps),
        offload_keep=int(cfg.offload_keep),
        step_device=str(cfg.step_device),
        thetas=build_thetas,
        min_abs=build_min_abs,
        min_mat_abs=build_min_mat_abs,
    )

    from .tensor_propagate import zero_filter_tensor_backprop_with_keep_mask

    stream_for_prune: Optional[str] = str(cfg.build_device) if str(cfg.build_device) != "cpu" else None
    psum_union, keep_mask_in = zero_filter_tensor_backprop_with_keep_mask(
        psum_union,
        stream_device=stream_for_prune,
        offload_back=(stream_for_prune is not None),
    )
    basis = _shrink_union_basis(basis, keep_mask_in)

    return CompiledTensorSurrogate(
        circuit=list(circuit),
        psum_union=psum_union,
        basis=basis,
        observables=obs_list,
        preset_name=str(preset),
        preset=cfg,
        _V0_cache={},
    )


def build_quasi_sampler(
    *,
    n_qubits: int,
    circuit,
    z_combos,
    max_order: Optional[int] = None,
    preset: str = "gpu_min",
    preset_overrides: Optional[Mapping[str, Any]] = None,
    build_thetas: Any = None,
    build_min_abs: Optional[float] = None,
    build_min_mat_abs: Optional[float] = None,
) -> TensorSparseSampler:
    """Build TensorSparseSampler with the same preset system."""

    cfg = resolve_preset(preset, overrides=preset_overrides)
    return TensorSparseSampler(
        n_qubits=int(n_qubits),
        circuit=circuit,
        z_combos=z_combos,
        max_order=(None if max_order is None else int(max_order)),
        build_device=str(cfg.build_device),
        dtype=str(cfg.dtype),
        max_weight=int(cfg.max_weight),
        max_xy=int(cfg.max_xy),
        step_device=str(cfg.step_device),
        offload_steps=bool(cfg.offload_steps),
        offload_keep=int(cfg.offload_keep),
        build_thetas=build_thetas,
        build_min_abs=build_min_abs,
        build_min_mat_abs=build_min_mat_abs,
    )


__all__ = [
    "TensorSurrogatePreset",
    "DEFAULT_PRESETS",
    "CompiledTensorSurrogate",
    "resolve_preset",
    "compile_expval_program",
    "build_quasi_sampler",
    "pennylane_expvals_small",
    "pennylane_sample_small",
    "pennylane_reference",
]
