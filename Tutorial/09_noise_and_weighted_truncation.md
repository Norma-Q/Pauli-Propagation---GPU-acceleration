# 09. Noise Channels and Weighted Truncation

This tutorial introduces two new noise-step gates and the new weighted truncation system in the tensor surrogate workflow.

## What is new

### 1) Depolarizing noise (single-qubit, inhomogeneous)

Use `DepolarizingNoise(qubit, px, py, pz)`.

Schrödinger-picture channel:

\[
\mathcal{E}(\rho) = p_I\rho + p_X X\rho X + p_Y Y\rho Y + p_Z Z\rho Z,
\quad p_I = 1-(p_X+p_Y+p_Z)
\]

Heisenberg-dual action on Pauli operators:

\[
\mathcal{E}^\dagger(I)=I,
\quad \mathcal{E}^\dagger(X)=(1-2p_Y-2p_Z)X,
\quad \mathcal{E}^\dagger(Y)=(1-2p_X-2p_Z)Y,
\quad \mathcal{E}^\dagger(Z)=(1-2p_X-2p_Y)Z.
\]

### 2) Amplitude damping noise (single-qubit)

Use `AmplitudeDampingNoise(qubit, gamma)` with `gamma in [0,1]`.

Exact Heisenberg-dual action:

\[
\mathcal{E}^\dagger(I)=I,
\quad \mathcal{E}^\dagger(X)=\sqrt{1-\gamma}X,
\quad \mathcal{E}^\dagger(Y)=\sqrt{1-\gamma}Y,
\quad \mathcal{E}^\dagger(Z)=(1-\gamma)Z+\gamma I.
\]

This means:
- no branching on `I`
- possible branching from `Z` to `I`

### 3) Weighted truncation (replaces `max_xy` path)

The truncation test is now:

\[
\text{keep term if } w_X\#X + w_Y\#Y + w_Z\#Z \le \text{max_weight}
\]

with user controls:
- `max_weight`
- `weight_x`
- `weight_y`
- `weight_z`

---

## Quick Start Example (2 qubits)

```python
import numpy as np
from src.pauli_surrogate_python import (
    PauliSum,
    CliffordGate,
    DepolarizingNoise,
    AmplitudeDampingNoise,
)
from src_tensor.api import compile_expval_program

# Observable: <X0>
obs = PauliSum(2)
obs.add_from_str("X", 1.0, [0])

# A tiny circuit: H on q0, then one noise step on q0
circuit_dep = [
    CliffordGate("H", [0]),
    DepolarizingNoise(qubit=0, px=0.05, py=0.07, pz=0.11),
]

prog_dep = compile_expval_program(
    circuit=circuit_dep,
    observables=[obs],
    preset="cpu",
    preset_overrides={
        "max_weight": 10,
        "weight_x": 1.0,
        "weight_y": 1.0,
        "weight_z": 1.0,
    },
)

val_dep = float(prog_dep.expvals(np.array([0.0], dtype=np.float64)).reshape(-1)[0].item())
print("Depolarizing expval:", val_dep)

circuit_ad = [
    CliffordGate("H", [0]),
    AmplitudeDampingNoise(qubit=0, gamma=0.25),
]

prog_ad = compile_expval_program(
    circuit=circuit_ad,
    observables=[obs],
    preset="cpu",
    preset_overrides={
        "max_weight": 10,
        "weight_x": 1.0,
        "weight_y": 1.0,
        "weight_z": 1.0,
    },
)

val_ad = float(prog_ad.expvals(np.array([0.0], dtype=np.float64)).reshape(-1)[0].item())
print("Amplitude damping expval:", val_ad)
```

---

## Weighted Truncation Example (3-qubit intuition)

If `Y` terms are too expensive in your workload, increase `weight_y`:

```python
preset_overrides = {
    "max_weight": 8,
    "weight_x": 1.0,
    "weight_y": 4.0,
    "weight_z": 1.0,
}
```

This keeps the same `max_weight` budget but prunes Y-heavy terms earlier.

---

## Practical Tips

1. Start with `weight_x = weight_y = weight_z = 1.0`.
2. Increase a specific weight if that Pauli component dominates memory/time.
3. Tune `max_weight` and component weights together; they are coupled.
4. For amplitude damping, expect additional branching from `Z -> I` paths.

---

## Validation Notes

A 2–3 qubit validation run was used to verify:
- depolarizing scale factors
- amplitude damping branch/scales
- weighted truncation behavior
- high-level compile/eval integration

Example run command:

```bash
PYTHONPATH=/home/quantum/ys_lee/Pauli-Propagation---GPU-acceleration \
conda run -n cudaq python /tmp/validate_noise_2q3q.py
```
