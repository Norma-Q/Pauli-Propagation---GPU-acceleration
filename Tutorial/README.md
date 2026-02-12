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
   - How to override `gpu_min` safely under GPU resource constraints.

5. `05_pennylane_reference_api.ipynb`
   - How to use the small-qubit PennyLane reference API for expvals and sampling.

6. `06_advanced_qcbm_tfim_ground_state.ipynb`
   - Advanced QCBM training example on 1D TFIM ground-state samples.

7. `07_advanced_maxcut_qaoa.ipynb`
   - Advanced MaxCut-QAOA example: surrogate optimization + classical optimum comparison + sampling analysis.

## Recommended order

Run notebooks in numeric order.

## Environment setup (conda)

This repo does not currently ship a `pyproject.toml`/`setup.py`, so the notebooks
are typically run from the repo root (so `src/` and `src_tensor/` are importable).

Create a dedicated conda env and install the Tutorial dependencies:

```bash
./scripts/create_tutorial_conda_env.sh pps-tutorial
conda activate pps-tutorial
jupyter lab
```

If you prefer manual steps:

```bash
conda create -y -n pps-tutorial python=3.11 pip
conda activate pps-tutorial
python -m pip install -r requirements-tutorial.txt
python -m ipykernel install --user --name pps-tutorial --display-name "PPS Tutorial (pps-tutorial)"
jupyter lab
```

Notes:
- The GPU tensor path uses a local compiled extension at `src_tensor/_pps_tensor_backend_local.so`.
   For best compatibility, keep Python/PyTorch close to the pinned versions in `requirements-tutorial.txt`.
