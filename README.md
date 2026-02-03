# Pauli Propagation Surrogate (Tensor)
## Overview
Prebuilt tensor backend for Pauli propagation surrogate.
This repo ships **precompiled wheels** in `dist/`.
![alt text](img/image.png)

## Summary
### Overview
A Pauli-propagation surrogate is a learned, tensor-based approximation layer that accelerates quantum circuit evaluation by turning Pauli-term rewrite rules into efficient sparse tensor operations. Instead of simulating full quantum states, the surrogate captures how observables transform through a circuit and evaluates expectations quickly on CPU/GPU.

### Why is it needed?
Conventional quantum optimization can be bottlenecked by repeated circuit executions and expensive simulations. The surrogate approach reduces runtime and cost by:
- Cutting the number of expensive evaluations needed during optimization
- Enabling fast, noise-free exploration of parameter landscapes before running on hardware
- Providing a practical path to scale up experiments while preserving accuracy

### Context: where this fits in modern QML
Many near-term quantum algorithms (e.g., VQE/QAOA-style workflows) are **hybrid**: a parameterized quantum circuit is evaluated, then a classical optimizer updates parameters to improve an objective. In practice, that means **many expectation-value evaluations** of Hamiltonians expressed as **sums of Pauli strings**. This project targets that bottleneck by providing a fast surrogate of Pauli-term propagation and evaluation so you can iterate quickly on CPU/GPU before committing to hardware runs.

### High-level idea (no heavy math)
1. Express the observable (cost/Hamiltonian) as a sum of Pauli strings.
2. Propagate those Pauli terms *backwards* through the circuit using deterministic rewrite rules.
3. Convert the rewrite rules into sparse tensor steps.
4. Evaluate coefficients efficiently, enabling fast expectation estimates for many parameter values.

### What you gain
- **Speed**: significant reduction in repeated evaluation time for parameter sweeps.
- **Scalability**: sparse tensor ops enable larger circuits than naive simulation.
- **Workflow agility**: quick iteration during model design and training.

### Typical use cases
- Variational quantum algorithms (VQE-like energy estimation, QAOA-like optimization)
- Fast prototyping of circuit ansätze and optimizer settings
- GPU-accelerated experiments where full-state simulation is too slow
- Research pipelines that require many repeated observable evaluations

### Limitations / caveats
- This is **not** a full statevector simulator.
- Accuracy depends on truncation settings (e.g., `min_abs`, `max_weight`).
- For exact hardware-level noise effects, a separate noise model or hardware evaluation is still required.

### How to decide if this repo is right for you
- You need **fast expectation-value evaluation** for Pauli-sum observables.
- You want a **GPU-ready** backend for repeated variational updates.
- You are running **many optimization steps** and need a scalable surrogate.

## Requirements
- Python **3.11**
- Linux x86_64 (for provided wheel)
- CUDA 12.x (for GPU use)

## Environment Setup
```bash
conda env create -f environment_311.yml
conda activate ENV_311
```

## Install Wheel
```bash
pip install dist/pauli_propagation_surrogate_tensor-0.1.0-cp311-cp311-linux_x86_64.whl
```

## Verify Installation
After installing the wheel, verify the backend is working correctly:
```python
python -c "from src_tensor import _pps_tensor_backend; print('Backend loaded successfully')"
```

If no errors occur, the installation is successful.

## Run Tutorial
To verify the full workflow, open and run `Tutorial.ipynb`.

Run all cells in order to confirm:
- Backend import succeeds
- Surrogate propagation runs
- Training loop completes
- (Optional) CPU/GPU timing comparison runs

**Important:** Make sure the correct kernel (conda environment) is selected in Jupyter.

## Quick Concepts (glossary-style)
- **Pauli String**: A tensor product like $X \otimes I \otimes Z \otimes X$ that defines a measurement basis for an observable term.
- **Pauli Sum**: A weighted sum of Pauli strings representing a Hamiltonian or cost function.
- **Variational Circuit**: A parameterized circuit optimized by a classical loop (common in VQE/QAOA workflows).
- **Expectation Value**: The scalar value of an observable on a prepared quantum state; the objective optimized in many QML algorithms.

## Practical tips
- Start with a **looser truncation** during early training, then tighten `min_abs` near convergence for speed/accuracy balance.
- If GPU is available, compare CPU/GPU timing in the tutorial to calibrate your workload.
- Keep `max_weight` modest to prevent term growth on large circuits.

## Project Structure
```
Pauli-propagation-surrogate/
├── README.md                    # This file (installation & usage)
├── IMPLEMENTATION_SUMMARY.md    # Technical documentation (developers)
├── LICENSE                      # GNU AGPLv3 license
├── environment_311.yml          # Conda environment (Python 3.11)
├── setup.py                     # Package configuration
├── MANIFEST.in                  # Distribution file rules
├── dist/                        # Prebuilt wheel files
│   └── pauli_propagation_surrogate_tensor-0.1.0-cp311-cp311-linux_x86_64.whl
└── Tutorial.ipynb               # Main tutorial notebook
```

**Note:** `src_tensor/` is **not** included in the repository. It is provided by the installed wheel package.

## Packaging & Distribution

This repository follows a **prebuilt-binary distribution model**:
- The compiled backend (`src_tensor._pps_tensor_backend*.so`) is included in official wheels
- Building from this repo alone produces a Python-only wheel (no backend) that will fail at runtime
- Wheels are **platform and Python-version specific** (e.g., `cp311` = Python 3.11)

For developers: see [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details.

## Notes
- This package does **not** compile C++ at install time
- If Python version differs, the wheel won't install
- Ensure you're **not** in the source directory when importing (avoids local `src_tensor` shadowing installed package)

## References (background reading)
- Quantum Surrogate Model overview and motivation: https://tzcjilwq.gensparkspace.com/
- Variational quantum eigensolver (VQE) basics: https://en.wikipedia.org/wiki/Variational_quantum_eigensolver
- Quantum approximate optimization algorithm (QAOA) overview: https://en.wikipedia.org/wiki/Quantum_optimization_algorithms
- PennyLane QML learning hub: https://pennylane.ai/qml/

## License
GNU AGPLv3 (see LICENSE)

## Copyright
Copyright (C) 2026 ys_lee@norma.co.kr

This program is licensed under the GNU Affero General Public License v3.0.

<img src="img/NORMA_CI.png" width="50%">
