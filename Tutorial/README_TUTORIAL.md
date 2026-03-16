# Tutorial Notebooks

English tutorial notebooks for the high-level tensor API and GPU memory-first workflows.

## Notebook index

1. `01_quickstart_gpu_min.ipynb`
   - CPU vs GPU timing comparison for expval evaluation.
   - Truncation (`max_weight`) vs exact PennyLane reference accuracy trend.

2. `02_training_with_compiled_program.ipynb`
   - Explicit PyTorch training loop built on `CompiledTensorSurrogate.expvals`.

3. `03_quasi_probability_workflow.ipynb`
   - Moments and truncated quasi-probability via `build_quasi_sampler`.
   - Includes an exact check case with `preset="gpu_full"` and full-order correlators.

4. `04_preset_tuning_gpu_budget.ipynb`
   - How to tune presets with `resolve_preset(...)` + `preset_overrides` under GPU resource constraints.

5. `05_pennylane_reference_api.ipynb`
   - How to use the small-qubit PennyLane reference API for expvals and sampling.

6. `06_advanced_qcbm_tfim_ground_state.ipynb`
   - Advanced QCBM training example on 1D TFIM ground-state samples.

7. `07_advanced_maxcut_qaoa.ipynb`
   - Advanced MaxCut-QAOA example: surrogate optimization + classical optimum comparison + sampling analysis.

8. `08_embedding_batched_priors_basics.ipynb`
   - Embedding + batched priors tutorial in three steps:
     1) 1D sin-regression,
     2) 2D linear binary classification,
     3) XOR classification.
   - Demonstrates training only on `thetas` while feeding input data through `priors` batches.

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
| `PauliRotation`, `CliffordGate` (`src.pauli_surrogate_python`) | 01, 02, 03, 04, 05, 06, 07, 08 | Core gate objects used to define surrogate circuits consistently across benchmarking, training, quasi-probability, and advanced examples. |
| `PauliSum` (`src.pauli_surrogate_python`) | 01, 02, 04, 05, 07, 08 | Observable container used to build expval targets (e.g., per-qubit Z observables, MaxCut objectives) before compilation. |
| `compile_expval_program(...)` (`src_tensor.api`) | 01, 02, 04, 05, 07, 08 | Compile-once entrypoint for expval workflows. Produces a reusable `CompiledTensorSurrogate` used by `.expvals(...)` in timing checks, optimization loops, and supervised tasks. |
| `CompiledTensorSurrogate.expvals(...)` (`src_tensor.api`) | 01, 02, 05, 08 | Main evaluation path after compile. In 02 it drives an explicit PyTorch autograd training loop; in 08 it is used with batched priors/embedding inputs. |
| `resolve_preset(...)` + `preset_overrides` (`src_tensor.api`) | 04 | Demonstrates preset-based tuning from a base config (shown with `gpu_safe`) and targeted overrides like `max_weight` / `dtype` for budget-aware execution. |
| `build_quasi_sampler(...)` (`src_tensor.api`) | 03, 05, 06 | Builds `TensorSparseSampler` for moment computation and quasi-probability reconstruction. In 03/06 this is central to moment/probability analysis; in 05 it is paired with reference sampling. |
| `pennylane_reference(...)` (`src_tensor.api`) | 05 | Unified small-qubit exact/reference path: `program -> expvals`, `sampler -> sampled bitstrings` for correctness checks. |
| `pennylane_reference_probs(...)` (`src_tensor.api`) | 06 | Exact probability baseline (`qml.probs`) used to validate QCBM moment-based reconstruction quality. |
| `pennylane_sample_small(...)` (`src_tensor.api`) | 07 | Exact small-qubit sampling helper used in MaxCut-QAOA analysis to compare surrogate-side results with reference bitstring samples. |

Notes:
- Notebook 04 text mentions `gpu_min` / `gpu_full` as policy labels, while executable code demonstrates the current preset path via `gpu_safe` + overrides.
- Notebook 03 explicitly checks that local `src_tensor.tensor_sampler` is imported, helping avoid stale site-packages during iterative API updates.

## Recommended order

Run notebooks in numeric order.
