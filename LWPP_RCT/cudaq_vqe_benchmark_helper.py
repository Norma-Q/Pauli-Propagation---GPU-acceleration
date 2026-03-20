from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _require_cudaq() -> Any:
    try:
        import cudaq  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "cudaq is required for the VQE benchmark notebook. "
            "Use the pps-tutorial environment or install CUDA-Q."
        ) from exc
    return cudaq


def maybe_set_target(target_name: Optional[str]) -> str:
    cudaq = _require_cudaq()
    if target_name:
        cudaq.set_target(str(target_name))
    target = cudaq.get_target()
    target_name_attr = getattr(target, "name", None)
    return str(target_name_attr or target)


def central_difference_gradient(
    *,
    theta: np.ndarray,
    objective_fn,
    epsilon: float = 1.0e-4,
) -> np.ndarray:
    theta_np = np.asarray(theta, dtype=np.float64).reshape(-1)
    grad = np.zeros_like(theta_np, dtype=np.float64)
    eps = float(epsilon)
    for i in range(theta_np.shape[0]):
        theta_plus = theta_np.copy()
        theta_minus = theta_np.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        grad[i] = float(objective_fn(theta_plus) - objective_fn(theta_minus)) / (2.0 * eps)
    return grad


def build_ring_hea_kernel(
    *,
    n_qubits: int,
    n_layers: int,
) -> Tuple[Any, int]:
    cudaq = _require_cudaq()
    kernel, params = cudaq.make_kernel(list[float])
    qubits = kernel.qalloc(int(n_qubits))

    param_index = 0
    for _layer in range(int(n_layers)):
        for q in range(int(n_qubits)):
            kernel.ry(params[param_index], qubits[q])
            param_index += 1
            kernel.rz(params[param_index], qubits[q])
            param_index += 1

        for q in range(int(n_qubits)):
            kernel.cx(qubits[q], qubits[(q + 1) % int(n_qubits)])

    return kernel, int(param_index)


def build_mixed_spin_hamiltonian(
    *,
    n_qubits: int,
    seed: int,
) -> Tuple[Any, Dict[str, int]]:
    cudaq = _require_cudaq()
    rng = np.random.default_rng(int(seed))
    hamiltonian = None

    counts = {
        "local_z": 0,
        "local_x": 0,
        "ring_zz": 0,
        "skip_xx": 0,
    }

    for q in range(int(n_qubits)):
        coeff = float(rng.uniform(-1.0, 1.0))
        term = coeff * cudaq.spin.z(int(q))
        hamiltonian = term if hamiltonian is None else (hamiltonian + term)
        counts["local_z"] += 1

    for q in range(int(n_qubits)):
        coeff = float(0.35 * rng.uniform(-1.0, 1.0))
        hamiltonian = hamiltonian + coeff * cudaq.spin.x(int(q))
        counts["local_x"] += 1

    for q in range(int(n_qubits)):
        coeff = float(rng.uniform(-1.0, 1.0))
        hamiltonian = hamiltonian + coeff * (
            cudaq.spin.z(int(q)) * cudaq.spin.z(int((q + 1) % int(n_qubits)))
        )
        counts["ring_zz"] += 1

    for q in range(0, int(n_qubits), 2):
        coeff = float(0.25 * rng.uniform(-1.0, 1.0))
        hamiltonian = hamiltonian + coeff * (
            cudaq.spin.x(int(q)) * cudaq.spin.x(int((q + 2) % int(n_qubits)))
        )
        counts["skip_xx"] += 1

    if hamiltonian is None:
        raise ValueError("Hamiltonian construction failed.")

    return hamiltonian, counts


def run_adam_benchmark(
    *,
    n_qubits: int,
    n_layers: int,
    max_iterations: int,
    learning_rate: float,
    seed: int,
    target_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[float]]:
    cudaq = _require_cudaq()
    active_target = maybe_set_target(target_name)

    kernel, n_params = build_ring_hea_kernel(
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
    )
    hamiltonian, term_counts = build_mixed_spin_hamiltonian(
        n_qubits=int(n_qubits),
        seed=int(seed) + int(n_qubits),
    )

    rng = np.random.default_rng(int(seed) + 10_000 + int(n_qubits))
    initial_theta = rng.uniform(
        low=-0.25 * np.pi,
        high=0.25 * np.pi,
        size=(int(n_params),),
    ).astype(np.float64)

    observe_call_count = 0
    observe_seconds = 0.0
    objective_call_rows: List[Dict[str, Any]] = []

    def energy_only(theta: Sequence[float]) -> float:
        nonlocal observe_call_count, observe_seconds
        theta_np = np.asarray(theta, dtype=np.float64).reshape(-1)
        start = time.perf_counter()
        value = float(cudaq.observe(kernel, hamiltonian, theta_np.tolist()).expectation())
        observe_seconds += float(time.perf_counter() - start)
        observe_call_count += 1
        return value

    initial_energy = float(energy_only(initial_theta))

    optimizer = cudaq.optimizers.Adam()
    optimizer.max_iterations = int(max_iterations)
    optimizer.step_size = float(learning_rate)
    optimizer.initial_parameters = [float(x) for x in initial_theta.tolist()]

    outer_objective_calls = 0

    def objective(parameter_vector: Sequence[float]) -> Tuple[float, List[float]]:
        nonlocal outer_objective_calls
        theta_np = np.asarray(parameter_vector, dtype=np.float64).reshape(-1)
        current_energy = float(energy_only(theta_np))
        grad = np.asarray(
            central_difference_gradient(
                theta=theta_np,
                objective_fn=energy_only,
                epsilon=1.0e-4,
            ),
            dtype=np.float64,
        )
        objective_call_rows.append(
            {
                "theta": theta_np.tolist(),
                "energy": float(current_energy),
            }
        )
        outer_objective_calls += 1
        return current_energy, grad.tolist()

    optimizer_start = time.perf_counter()
    best_energy, final_theta_list = optimizer.optimize(int(n_params), objective)
    optimizer_seconds = float(time.perf_counter() - optimizer_start)

    final_theta = np.asarray(final_theta_list, dtype=np.float64)
    final_energy = float(energy_only(final_theta))

    post_update_rows = objective_call_rows[1:] + [
        {
            "theta": final_theta.tolist(),
            "energy": float(final_energy),
        }
    ]
    step_trajectory = [
        float(row["energy"])
        for row in post_update_rows[: int(max_iterations)]
    ]

    total_terms = int(sum(term_counts.values()))
    summary = {
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "n_params": int(n_params),
        "max_iterations": int(max_iterations),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "target": str(active_target),
        "hamiltonian_term_counts": dict(term_counts),
        "hamiltonian_total_terms": int(total_terms),
        "initial_energy": float(initial_energy),
        "best_energy_reported_by_optimizer": float(best_energy),
        "final_energy": float(final_energy),
        "optimizer_seconds": float(optimizer_seconds),
        "outer_objective_calls": int(outer_objective_calls),
        "observe_calls_total": int(observe_call_count),
        "observe_seconds_total": float(observe_seconds),
        "mean_observe_ms": float(1.0e3 * observe_seconds / max(1, observe_call_count)),
        "effective_observe_calls_per_iteration": float(
            observe_call_count / max(1, int(max_iterations))
        ),
    }
    return summary, step_trajectory


def run_suite(
    *,
    systems: Sequence[int],
    n_layers: int,
    max_iterations: int,
    learning_rate: float,
    seed: int,
    target_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    summaries: List[Dict[str, Any]] = []
    trajectories: Dict[str, List[float]] = {}

    for n_qubits in systems:
        summary, trajectory = run_adam_benchmark(
            n_qubits=int(n_qubits),
            n_layers=int(n_layers),
            max_iterations=int(max_iterations),
            learning_rate=float(learning_rate),
            seed=int(seed),
            target_name=target_name,
        )
        summaries.append(summary)
        trajectories[str(int(n_qubits))] = list(trajectory)

    return summaries, trajectories


def print_summary_table(summaries: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "Qubits",
        "Layers",
        "Params",
        "Iters",
        "FinalE",
        "OptSec",
        "ObsCalls",
        "ObsSec",
        "Obs/Iter",
    ]
    rows = []
    for row in summaries:
        rows.append(
            [
                str(int(row["n_qubits"])),
                str(int(row["n_layers"])),
                str(int(row["n_params"])),
                str(int(row["max_iterations"])),
                f"{float(row['final_energy']):.6f}",
                f"{float(row['optimizer_seconds']):.3f}",
                str(int(row["observe_calls_total"])),
                f"{float(row['observe_seconds_total']):.3f}",
                f"{float(row['effective_observe_calls_per_iteration']):.1f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def _fmt(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))


def plot_suite(
    *,
    summaries: Sequence[Dict[str, Any]],
    trajectories: Dict[str, Sequence[float]],
) -> Any:
    import matplotlib.pyplot as plt

    qubits = [int(row["n_qubits"]) for row in summaries]
    optimizer_seconds = [float(row["optimizer_seconds"]) for row in summaries]
    observe_calls = [int(row["observe_calls_total"]) for row in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    axes[0].bar([str(q) for q in qubits], optimizer_seconds, color=["#4c78a8", "#f58518"])
    axes[0].set_title("Adam Optimize Wall Time")
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar([str(q) for q in qubits], observe_calls, color=["#72b7b2", "#e45756"])
    axes[1].set_title("Total cudaq.observe Calls")
    axes[1].set_xlabel("Qubits")
    axes[1].set_ylabel("Calls")
    axes[1].grid(True, axis="y", alpha=0.3)

    for q in qubits:
        values = trajectories[str(int(q))]
        axes[2].plot(
            np.arange(1, len(values) + 1),
            values,
            marker="o",
            linewidth=2.0,
            label=f"{int(q)}Q",
        )
    axes[2].set_title("Post-Update Energy Trajectory")
    axes[2].set_xlabel("Adam Iteration")
    axes[2].set_ylabel("Energy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    return fig


def save_suite_json(
    *,
    path: Path,
    summaries: Sequence[Dict[str, Any]],
    trajectories: Dict[str, Sequence[float]],
) -> None:
    payload = {
        "summaries": list(summaries),
        "trajectories": {
            str(key): [float(x) for x in values]
            for key, values in trajectories.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
