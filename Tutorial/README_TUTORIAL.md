# PadoPauli — Tutorial Notebooks

English tutorial notebooks for the PadoPauli high-level tensor API and GPU memory-first workflows.

## Notebook index

1. `01_quickstart_gpu_basics.ipynb`
   - CPU vs GPU timing comparison for expval evaluation.
   - Truncation (`max_weight`) vs exact PennyLane reference accuracy trend.

2. `02_pennylane_reference_api.ipynb`
   - How to use the small-qubit PennyLane reference API for expvals and sampling.

3. `03_training_with_compiled_program.ipynb`
   - Explicit PyTorch training loop built on `CompiledTensorSurrogate.expvals`.

4. `04_embedding_batched_inputs_basics.ipynb`
   - Embedding + batched input tutorial in three steps:
     1) 1D sin-regression,
     2) 2D linear binary classification,
     3) XOR classification.
   - Demonstrates training only on `thetas` while feeding batched embedding inputs.

5. `05_preset_tuning_gpu_budget.ipynb`
   - How to tune presets with `resolve_preset(...)` + `preset_overrides` under GPU resource constraints.

6. `06_quasi_probability_workflow.ipynb`
   - Moments and truncated quasi-probability via `build_quasi_sampler`.
   - Includes an exact check case with a high-capacity GPU preset configuration for full-order correlators.

7. `07_advanced_qcbm_tfim_ground_state.ipynb`
   - Advanced QCBM training example on 1D TFIM ground-state samples.

8. `08_advanced_maxcut_qaoa.ipynb`
   - Advanced MaxCut-QAOA example: surrogate optimization + classical optimum comparison + sampling analysis.

9. `09_gpu_multiprocessing_compile_benchmark.ipynb`
   - Benchmarks PPS compile at medium scale (10~15 qubits recommended) with `gpu_parallel=False/True`.
   - Uses `compile_expval_program(..., parallel_compile=...)` to compare propagation/compile time.
   - Logs per-GPU peak utilization/memory (via `nvidia-smi`) during compile for side-by-side notes.

## Where updated `src` / `src_tensor` features are used

The tutorials share the same pattern:
1) define circuit/observables from `src.pauli_surrogate_python`,
2) compile/build runtime objects from `src_tensor.api`,
3) evaluate/train/sample in each notebook task.

| Feature (module) | Where used | What / How it is applied |
|---|---|---|
| `PauliRotation`, `CliffordGate` (`src.pauli_surrogate_python`) | 01, 03, 04, 05, 06, 07, 08 | Core gate objects used to define surrogate circuits consistently across benchmarking, training, quasi-probability, and advanced examples. |
| `PauliSum` (`src.pauli_surrogate_python`) | 01, 03, 05, 08 | Observable container used to build expval targets (e.g., per-qubit Z observables, MaxCut objectives) before compilation. |
| `compile_expval_program(...)` (`src_tensor.api`) | 01, 02, 03, 05, 08 | Compile-once entrypoint for expval workflows. Produces a reusable `CompiledTensorSurrogate` used by `.expvals(...)` in timing checks, optimization loops, and supervised tasks. |
| `CompiledTensorSurrogate.expvals(...)` (`src_tensor.api`) | 01, 03, 04 | Main evaluation path after compile. In 03 it drives an explicit PyTorch autograd training loop; in 04 it is used with batched embedding inputs. |
| `resolve_preset(...)` + `preset_overrides` (`src_tensor.api`) | 05 | Demonstrates preset-based tuning from the current built-in presets (`cpu`, `gpu`, `hybrid`) and targeted overrides like `max_weight` / `dtype` for budget-aware execution. |
| `build_quasi_sampler(...)` (`src_tensor.api`) | 02, 06, 07 | Builds `TensorSparseSampler` for moment computation and quasi-probability reconstruction. In 06/07 this is central to moment/probability analysis; in 02 it is paired with reference sampling. |
| `pennylane_reference(...)` (`src_tensor.api`) | 02 | Unified small-qubit exact/reference path: `program -> expvals`, `sampler -> sampled bitstrings` for correctness checks. |
| `pennylane_reference_probs(...)` (`src_tensor.api`) | 07 | Exact probability baseline (`qml.probs`) used to validate QCBM moment-based reconstruction quality. |
| `pennylane_sample_small(...)` (`src_tensor.api`) | 08 | Exact small-qubit sampling helper used in MaxCut-QAOA analysis to compare surrogate-side results with reference bitstring samples. |

Notes:
- Current built-in preset names in `src_tensor.api` are `cpu`, `gpu`, and `hybrid`.
- Notebook 06 explicitly checks that local `src_tensor.tensor_sampler` is imported, helping avoid stale site-packages during iterative API updates.

## Recommended order

Run notebooks in numeric order.
