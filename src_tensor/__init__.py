"""GPU-first tensor backend (WIP)."""

from .tensor_types import TensorDAGEdges, TensorDAGGraph, LayerEdgesTensor
from .tensor_types import TensorPauliSum, TensorSparseStep
from .tensor_eval import TensorDAGEvaluator, TensorSparseEvaluator
from .tensor_propagate import propagate_surrogate_tensor
from .tensor_adjoint import (
    UnionBasis,
    propagate_union_basis_psum,
    adjoint_weights_on_zero,
    coeff_matrix_from_observables,
    expvals_from_w_and_coeff_matrix,
)
from .tensor_sampler import TensorSparseSampler, ZComboSpec, normalize_z_combos
from .api import (
    TensorSurrogatePreset,
    DEFAULT_PRESETS,
    CompiledTensorSurrogate,
    resolve_preset,
    compile_expval_program,
    build_quasi_sampler,
    pennylane_expvals_small,
    pennylane_sample_small,
    pennylane_reference,
)

__all__ = [
    "TensorDAGEdges",
    "TensorDAGGraph",
    "LayerEdgesTensor",
    "TensorPauliSum",
    "TensorSparseStep",
    "TensorDAGEvaluator",
    "TensorSparseEvaluator",
    "propagate_surrogate_tensor",
    "UnionBasis",
    "propagate_union_basis_psum",
    "adjoint_weights_on_zero",
    "coeff_matrix_from_observables",
    "expvals_from_w_and_coeff_matrix",
    "TensorSparseSampler",
    "ZComboSpec",
    "normalize_z_combos",
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
