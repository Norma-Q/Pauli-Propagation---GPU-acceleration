"""High-level user API for tensor surrogate workflows.

This module provides a fixed, memory-first execution path for common use:
  1) compile once from circuit + observables
  2) evaluate expvals for theta vectors
  3) build custom training loops with PyTorch autograd

The `cpu` preset prioritizes scalability and safety by building and storing
sparse steps on CPU, then streaming computation to GPU for evaluation.
Preset `gpu` prioritizes maximum speed by keeping build and masks on GPU,
with conservative precision defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from .tensor_eval import TensorSparseEvaluator
from .tensor_sampler import TensorSparseSampler
from .tensor_types import TensorPauliSum


def _expvals_parallel_worker(args: Tuple) -> Tensor:
    from .tensor_adjoint import adjoint_weights_on_zero

    (
        rank, devices, psum_union, thetas, embedding_chunk, chunk_size
    ) = args

    device = devices[rank]
    w_chunk = adjoint_weights_on_zero(
        psum_union,
        thetas,
        embedding=embedding_chunk,
        compute_device=device,
        chunk_size=chunk_size,
    )
    return w_chunk.cpu()


@dataclass(frozen=True)
class TensorSurrogatePreset:
    """Execution preset for high-level API."""
    memory_device: str = "cpu"
    compute_device: str = "cuda"
    dtype: str = "float32"
    max_weight: int = 20
    weight_x: float = 1.0
    weight_y: float = 1.0
    weight_z: float = 1.0
    chunk_size: int = 1_000_000


DEFAULT_PRESETS: Dict[str, TensorSurrogatePreset] = {
    "cpu": TensorSurrogatePreset(
        memory_device="cpu",  # where pauli sums and masks are built/stored
        compute_device="cpu", # where the main matrix computation happens; can be "cuda" for GPU eval with CPU storage
        dtype="float32",
        max_weight=1_000_000_000,
        weight_x=1.0,
        weight_y=1.0,
        weight_z=1.0,
        chunk_size=10_000_000,
    ),
    "gpu": TensorSurrogatePreset(
        memory_device="cuda",
        compute_device="cuda",
        dtype="float64",
        max_weight=1_000_000_000,
        weight_x=1.0,
        weight_y=1.0,
        weight_z=1.0,
        chunk_size=10_000_000,
    ),
    "hybrid": TensorSurrogatePreset(
        memory_device="cpu",
        compute_device="cuda",
        dtype="float32",
        max_weight=1_000_000_000,
        weight_x=1.0,
        weight_y=1.0,
        weight_z=1.0,
        chunk_size=25_000_000,
    ),
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
    _V0_cache: Dict[Tuple[str, str], Tensor] = field(default_factory=dict)
    _obs_sparse_cache: Dict[Tuple[int, str, str], Tuple[Tensor, Tensor]] = field(default_factory=dict)
    _expval_evaluator: Optional[TensorSparseEvaluator] = None
    _diag_mask_cache: Dict[str, Tensor] = field(default_factory=dict)

    def _validate_obs_index(self, obs_index: int) -> int:
        n_obs = len(self.observables)
        if obs_index < 0 or obs_index >= n_obs:
            raise IndexError(f"obs_index={obs_index} is out of range for {n_obs} observables")
        return int(obs_index)

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

    def _get_obs_sparse_terms(self, *, obs_index: int, device: Any, dtype: Any) -> Tuple[Tensor, Tensor]:
        key = (int(obs_index), str(device), str(dtype))
        cached = self._obs_sparse_cache.get(key)
        if cached is not None:
            return cached

        obs = self.observables[int(obs_index)]
        idx_list: List[int] = []
        coeff_list: List[float] = []
        for p, c in obs.terms.items():
            i = self.basis.index.get(p)
            if i is None:
                continue
            c_f = float(c)
            if c_f == 0.0:
                continue
            idx_list.append(int(i))
            coeff_list.append(c_f)

        idx_t = torch.as_tensor(idx_list, dtype=torch.long, device=device)
        coeff_t = torch.as_tensor(coeff_list, dtype=dtype, device=device)
        self._obs_sparse_cache[key] = (idx_t, coeff_t)
        return idx_t, coeff_t

    def _get_expval_evaluator(self) -> TensorSparseEvaluator:
        evaluator = self._expval_evaluator
        if evaluator is None:
            evaluator = TensorSparseEvaluator(
                self.psum_union,
                compute_device=self.preset.compute_device,
                chunk_size=self.preset.chunk_size,
            )
            self._expval_evaluator = evaluator
        return evaluator

    def _get_diag_mask(self, *, device: Any) -> Tensor:
        key = str(device)
        cached = self._diag_mask_cache.get(key)
        if cached is not None:
            return cached
        diag_mask = (self.psum_union.x_mask == 0) if self.psum_union.x_mask.dim() == 1 else (self.psum_union.x_mask == 0).all(dim=1)
        diag_mask = diag_mask.to(device=device)
        self._diag_mask_cache[key] = diag_mask
        return diag_mask

    def expvals(
        self,
        thetas: Any = None,
        *,
        embedding: Any = None,
        parallel: bool = False,
    ) -> Tensor:
        """Evaluate expectation values with optional data priors (embedding).
        
        Args:
            thetas: Optional trainable parameters. Can be None for embedding-only circuits.
            embedding: Optional embedding (generative) parameters.
            parallel: Use multi-GPU evaluation if available.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor backend.")

        # [수정 포인트] embedding이 입력되었을 때만 배치 차원을 검사합니다.
        # 기존의 1D 입력을 (1, N)으로 만들어 하부 로직이 항상 2D(Batch)를 기대하게 합니다.
        if embedding is not None:
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_parallel = (
            parallel
            and embedding is not None
            and embedding.shape[0] > 1
            and num_gpus > 1
        )

        if use_parallel:
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            embedding_chunks = torch.chunk(embedding, num_gpus)
            worker_args = [
                (
                    i,
                    devices,
                    self.psum_union,
                    thetas,
                    embedding_chunks[i],
                    self.preset.chunk_size,
                )
                for i in range(num_gpus)
            ]

            import torch.multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_gpus) as pool:
                w_chunks_cpu = pool.map(_expvals_parallel_worker, worker_args)

            w = torch.cat(w_chunks_cpu, dim=0)
        else:
            w = adjoint_weights_on_zero(
                self.psum_union,
                thetas,
                embedding=embedding,
                compute_device=self.preset.compute_device,
                chunk_size=self.preset.chunk_size,
            )
        
        V0 = self._get_V0(device=str(w.device), dtype=w.dtype)
        
        # [수정 포인트] expvals_from_w_and_coeff_matrix 내부에서 Batch Matmul을 수행하게 합니다.
        res = expvals_from_w_and_coeff_matrix(w, V0)
        
        return res

    def expval(
        self,
        thetas: Any,
        *,
        obs_index: int = 0,
    ) -> Tensor:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor backend.")

        obs_index = self._validate_obs_index(obs_index)

        coeff_init = torch.zeros_like(self.psum_union.coeff_init)
        idx_t, coeff_t = self._get_obs_sparse_terms(
            obs_index=obs_index,
            device=coeff_init.device,
            dtype=coeff_init.dtype,
        )
        if idx_t.numel() > 0:
            coeff_init.index_copy_(0, idx_t, coeff_t)

        evaluator = self._get_expval_evaluator()
        coeff_out = evaluator.evaluate_coeffs(thetas, coeff_init=coeff_init)
        diag_mask = self._get_diag_mask(device=coeff_out.device)
        return torch.sum(coeff_out[diag_mask])

    def expvals_pennylane(self, thetas: Any, *, max_qubits: int = 20) -> Tensor:
        """Reference expvals via PennyLane (small circuits only)."""
        return pennylane_expvals_small(
            circuit=self.circuit,
            observables=self.observables,
            thetas=thetas,
            n_qubits=int(self.basis.n_qubits),
            max_qubits=max_qubits,
        )


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


def resolve_preset(name: str = "cpu", *, overrides: Optional[Mapping[str, Any]] = None) -> TensorSurrogatePreset:
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
    preset: str = "cpu",
    preset_overrides: Optional[Mapping[str, Any]] = None,
    build_thetas: Any = None,
    build_min_abs: Optional[float] = None,
    build_min_mat_abs: Optional[float] = None,
    parallel_compile: bool = False,
    parallel_threshold: int = -1,
    parallel_devices: Optional[Sequence[int]] = None,
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
        memory_device=str(cfg.memory_device),
        compute_device=str(cfg.compute_device),
        max_weight=int(cfg.max_weight),
        weight_x=float(cfg.weight_x),
        weight_y=float(cfg.weight_y),
        weight_z=float(cfg.weight_z),
        dtype=str(cfg.dtype),
        thetas=build_thetas,
        min_abs=build_min_abs,
        min_mat_abs=build_min_mat_abs,
        chunk_size=int(cfg.chunk_size),
        parallel_compile=parallel_compile,
        parallel_threshold=int(parallel_threshold),
        parallel_devices=parallel_devices,
    )


    # [Correction] Check x_mask size for propagated terms, not coeff_init (which is just the observable terms)
    n_terms_prop = int(psum_union.x_mask.shape[0])
    print(f"[PPS Info] Propagation complete. Terms generated: {n_terms_prop:,}")


    from .tensor_propagate import zero_filter_tensor_backprop_with_keep_mask

    psum_union, keep_mask_in = zero_filter_tensor_backprop_with_keep_mask(
        psum_union,
        compute_device=str(cfg.compute_device),
        chunk_size=int(cfg.chunk_size),
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
    preset: str = "cpu",
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
        memory_device=str(cfg.memory_device),
        compute_device=str(cfg.compute_device),
        dtype=str(cfg.dtype),
        max_weight=int(cfg.max_weight),
        weight_x=float(cfg.weight_x),
        weight_y=float(cfg.weight_y),
        weight_z=float(cfg.weight_z),
        chunk_size=int(cfg.chunk_size),
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
