from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from src_tensor.api import compile_expval_program

try:
    from test_qaoa.qaoa_surrogate_common import (
        build_maxcut_observable,
        build_qaoa_circuit,
        build_qaoa_theta_init_tqa,
        choose_device,
        expected_cut_from_sum_zz,
    )
except ImportError:
    from qaoa_surrogate_common import (  # type: ignore
        build_maxcut_observable,
        build_qaoa_circuit,
        build_qaoa_theta_init_tqa,
        choose_device,
        expected_cut_from_sum_zz,
    )


DEFAULT_CONFIG_PATH = _THIS_DIR / "q25_experiment_config.json"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_mapping_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "YAML config was requested but PyYAML is not installed. "
                "Use the bundled JSON config or install PyYAML."
            ) from exc
        data = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported config format: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    return data


def cleanup_memory(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def float_tag(x: float) -> str:
    text = f"{float(x):.0e}" if abs(float(x)) < 1e-2 else str(float(x))
    return text.replace("-", "m").replace(".", "p")


def seed_tag(seed: int) -> str:
    return f"seed_{int(seed):03d}"


def normalize_init_strategy(raw: str) -> str:
    key = str(raw).strip().lower().replace("-", "_")
    if key in {"near_zero", "nearidentity", "near_identity"}:
        return "near_zero"
    if key == "random":
        return "random"
    if key == "tqa":
        return "tqa"
    raise ValueError(f"Unknown init strategy: {raw}")


def build_initial_theta_np(
    *,
    init_strategy: str,
    n_layers: int,
    n_edges: int,
    n_qubits: int,
    delta_t: float,
    seed: int,
) -> np.ndarray:
    key = normalize_init_strategy(init_strategy)
    if key == "tqa":
        return build_qaoa_theta_init_tqa(
            p_layers=int(n_layers),
            n_edges=int(n_edges),
            n_qubits=int(n_qubits),
            delta_t=float(delta_t),
            dtype=np.float64,
        )

    rng = np.random.default_rng(int(seed))
    if key == "random":
        low = -0.5 * math.pi
        high = 0.5 * math.pi
        return rng.uniform(low=low, high=high, size=(2 * int(n_layers),)).astype(np.float64)
    if key == "near_zero":
        low = -0.01 * math.pi
        high = 0.01 * math.pi
        return rng.uniform(low=low, high=high, size=(2 * int(n_layers),)).astype(np.float64)
    raise ValueError(f"Unknown init strategy: {init_strategy}")


def canonical_edge(u: int, v: int) -> Tuple[int, int]:
    a = int(u)
    b = int(v)
    if a == b:
        raise ValueError("Self-loops are not allowed.")
    return (a, b) if a < b else (b, a)


def _is_connected(n_qubits: int, edges: Sequence[Tuple[int, int]]) -> bool:
    if int(n_qubits) < 1:
        return False
    if int(n_qubits) == 1:
        return True
    if len(edges) == 0:
        return False

    graph: List[List[int]] = [[] for _ in range(int(n_qubits))]
    for u, v in edges:
        uu = int(u)
        vv = int(v)
        graph[uu].append(vv)
        graph[vv].append(uu)

    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return len(seen) == int(n_qubits)


def generate_erdos_renyi_connected(
    *,
    n_qubits: int,
    edge_prob: float,
    seed: int,
    max_tries: int = 100,
) -> List[Tuple[int, int]]:
    n = int(n_qubits)
    p = float(edge_prob)
    if n < 2:
        raise ValueError("n_qubits must be at least 2.")
    if not (0.0 < p <= 1.0):
        raise ValueError("edge_prob must be in (0, 1].")

    for attempt in range(int(max_tries)):
        rng = random.Random(int(seed) + attempt)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        if _is_connected(n, edges):
            return edges
    raise RuntimeError(
        f"Could not generate a connected Erdős-Rényi graph after {int(max_tries)} attempts."
    )


def save_graph_circle(n_qubits: int, edges: Sequence[Tuple[int, int]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_qubits), endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    for u, v in edges:
        ax.plot([x[int(u)], x[int(v)]], [y[int(u)], y[int(v)]], color="gray", alpha=0.45)
    ax.scatter(x, y, s=180, c="#9ecae1", edgecolors="black", zorder=3)
    for i in range(int(n_qubits)):
        ax.text(1.11 * x[i], 1.11 * y[i], str(i), ha="center", va="center")
    ax.set_title(f"Erdos-Renyi Graph (n={int(n_qubits)})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_or_create_graph(
    *,
    graph_dir: Path,
    n_qubits: int,
    edge_prob: float,
    seed: int,
    graph_tag: str = "",
    create_if_missing: bool = True,
    max_tries: int = 100,
) -> Tuple[List[Tuple[int, int]], Path, Optional[Path], bool]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(graph_tag).strip()
    json_path = graph_dir / f"Q{int(n_qubits)}_edges{suffix}.json"
    png_path = graph_dir / f"Q{int(n_qubits)}_renyi{str(float(edge_prob))[2:]}{suffix}.png"

    created = False
    if json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_edges = raw.get("edges", None)
        else:
            raw_edges = raw
        if not isinstance(raw_edges, list):
            raise ValueError(f"Graph JSON must contain an edge list or an {{\"edges\": ...}} mapping: {json_path}")
        edges = [canonical_edge(int(u), int(v)) for u, v in raw_edges]
    else:
        if not create_if_missing:
            raise FileNotFoundError(f"Graph file not found: {json_path}")
        edges = generate_erdos_renyi_connected(
            n_qubits=int(n_qubits),
            edge_prob=float(edge_prob),
            seed=int(seed),
            max_tries=int(max_tries),
        )
        json_path.write_text(
            json.dumps([[int(u), int(v)] for (u, v) in edges], indent=2),
            encoding="utf-8",
        )
        created = True

    if created or not png_path.exists():
        try:
            save_graph_circle(int(n_qubits), edges, png_path)
        except Exception:
            png_path = None
    elif not png_path.exists():
        png_path = None
    return list(edges), json_path, png_path, created


def theta_trajectory_to_step_map(
    theta_trajectory: Sequence[Dict[str, Any]],
    *,
    start_thetas: Sequence[float],
    final_thetas: Sequence[float],
    stride: int = 1,
) -> Dict[str, List[float]]:
    keep_stride = max(1, int(stride))
    out: Dict[str, List[float]] = {
        "0": [float(x) for x in start_thetas],
    }
    last_row: Optional[Dict[str, Any]] = None
    for row in theta_trajectory:
        step = int(row["step"])
        last_row = dict(row)
        if (step % keep_stride) == 0:
            out[str(step)] = [float(x) for x in row["thetas"]]
    if last_row is not None:
        out[str(int(last_row["step"]))] = [float(x) for x in last_row["thetas"]]
    else:
        out["0"] = [float(x) for x in final_thetas]
    out["final"] = [float(x) for x in final_thetas]
    return out


def extract_compile_resources(program: Any) -> Dict[str, Any]:
    psum = program.psum_union
    total_nnz = 0
    for step in psum.steps:
        total_nnz += int(step.mat_const._nnz())
        total_nnz += int(step.mat_cos._nnz())
        total_nnz += int(step.mat_sin._nnz())
    return {
        "terms_after_zero_filter": int(psum.x_mask.shape[0]),
        "n_steps": int(len(psum.steps)),
        "nnz_total": int(total_nnz),
    }


def make_stage_compile_fn(
    *,
    stage_name: str,
    circuit: List[Any],
    zz_obj: Any,
    device: str,
    chunk_size: int,
    parallel_compile: bool,
    max_weight_override: Optional[int],
    n_qubits: int,
    preset_name: str = "hybrid",
) -> Callable[..., Tuple[Any, Dict[str, Any]]]:
    effective_max_weight = None if max_weight_override is None else int(max_weight_override)
    max_weight_mode = "coefficient_only" if max_weight_override is None else "truncated"

    def _compile(
        *,
        build_thetas: Optional[torch.Tensor],
        build_min_abs: Optional[float],
    ) -> Tuple[Any, Dict[str, Any]]:
        compile_start = time.time()
        preset_overrides: Dict[str, Any] = {
            "chunk_size": int(chunk_size),
            "compute_device": str(device) if str(device).startswith("cuda") else "cpu",
        }
        if effective_max_weight is not None:
            preset_overrides["max_weight"] = int(effective_max_weight)
        program = compile_expval_program(
            circuit=circuit,
            observables=[zz_obj],
            preset=str(preset_name),
            preset_overrides=preset_overrides,
            build_thetas=build_thetas,
            build_min_abs=build_min_abs,
            parallel_compile=bool(parallel_compile),
        )
        info = extract_compile_resources(program)
        info.update(
            {
                "stage_name": str(stage_name),
                "preset": str(preset_name),
                "preset_overrides": dict(preset_overrides),
                "compile_seconds": float(time.time() - compile_start),
                "build_min_abs": None if build_min_abs is None else float(build_min_abs),
                "effective_max_weight": None if effective_max_weight is None else int(effective_max_weight),
                "max_weight_mode": str(max_weight_mode),
            }
        )
        return program, info

    return _compile


def _history_rows(
    *,
    step: int,
    sum_zz: float,
    n_edges: int,
) -> Dict[str, Any]:
    return {
        "step": int(step),
        "sum_zz": float(sum_zz),
        "expected_cut": float(expected_cut_from_sum_zz(float(sum_zz), int(n_edges))),
    }


def train_with_fixed_program(
    *,
    program: Any,
    start_thetas_np: np.ndarray,
    device: str,
    n_edges: int,
    steps: int,
    lr: float,
    stage_name: str,
    log_every: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    theta_trajectory: List[Dict[str, Any]] = []
    thetas = torch.nn.Parameter(
        torch.tensor(np.asarray(start_thetas_np), dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam([thetas], lr=float(lr))

    print(f"[{stage_name}] training for {int(steps)} steps")
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = program.expval(thetas, obs_index=0)
        if not bool(loss.requires_grad):
            raise RuntimeError(f"{stage_name}: loss does not require gradients.")
        loss.backward()
        optimizer.step()

        sum_zz = float(loss.detach().cpu().item())
        history.append(_history_rows(step=step + 1, sum_zz=sum_zz, n_edges=n_edges))
        theta_trajectory.append(
            {
                "step": int(step + 1),
                "thetas": thetas.detach().cpu().numpy().tolist(),
            }
        )

        if (step == 0) or ((step + 1) % int(log_every) == 0) or (step + 1 == int(steps)):
            print(
                f"[{stage_name}] step={step + 1:04d} "
                f"sum<ZZ>={sum_zz:+.6f} "
                f"E[cut]={history[-1]['expected_cut']:.6f}"
            )

    return history, thetas.detach().cpu().numpy(), theta_trajectory


def train_with_periodic_rebuild(
    *,
    compile_program_fn: Callable[..., Tuple[Any, Dict[str, Any]]],
    start_thetas_np: np.ndarray,
    device: str,
    n_edges: int,
    steps: int,
    lr: float,
    build_min_abs: float,
    rebuild_interval: int,
    stage_name: str,
    log_every: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    theta_trajectory: List[Dict[str, Any]] = []
    rebuild_log: List[Dict[str, Any]] = []
    thetas = torch.nn.Parameter(
        torch.tensor(np.asarray(start_thetas_np), dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam([thetas], lr=float(lr))

    anchor = thetas.detach().clone()
    program, compile_info = compile_program_fn(
        build_thetas=anchor,
        build_min_abs=float(build_min_abs),
    )
    rebuild_log.append({"rebuild_index": 0, "after_step": 0, **compile_info})

    print(
        f"[{stage_name}] training for {int(steps)} steps "
        f"(min_abs={float(build_min_abs):.1e}, rebuild_interval={int(rebuild_interval)})"
    )
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = program.expval(thetas, obs_index=0)
        if not bool(loss.requires_grad):
            raise RuntimeError(
                f"{stage_name}: loss does not require gradients under current coefficient truncation."
            )
        loss.backward()
        optimizer.step()

        sum_zz = float(loss.detach().cpu().item())
        history.append(_history_rows(step=step + 1, sum_zz=sum_zz, n_edges=n_edges))
        theta_trajectory.append(
            {
                "step": int(step + 1),
                "thetas": thetas.detach().cpu().numpy().tolist(),
            }
        )

        if (step == 0) or ((step + 1) % int(log_every) == 0) or (step + 1 == int(steps)):
            print(
                f"[{stage_name}] step={step + 1:04d} "
                f"sum<ZZ>={sum_zz:+.6f} "
                f"E[cut]={history[-1]['expected_cut']:.6f}"
            )

        need_rebuild = (
            int(rebuild_interval) > 0
            and (step + 1) < int(steps)
            and ((step + 1) % int(rebuild_interval) == 0)
        )
        if need_rebuild:
            del program
            cleanup_memory(device)
            anchor = thetas.detach().clone()
            program, compile_info = compile_program_fn(
                build_thetas=anchor,
                build_min_abs=float(build_min_abs),
            )
            rebuild_log.append(
                {
                    "rebuild_index": int(len(rebuild_log)),
                    "after_step": int(step + 1),
                    **compile_info,
                }
            )

    del program
    cleanup_memory(device)
    return history, thetas.detach().cpu().numpy(), theta_trajectory, rebuild_log


def _require_cudaq() -> Any:
    try:
        import cudaq  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "cudaq is required for the exact branch. "
            "Use the pps-tutorial environment or install CUDA-Q."
        ) from exc
    return cudaq


def _cudaq_target_name(cudaq: Any) -> str:
    target = cudaq.get_target()
    target_name_attr = getattr(target, "name", None)
    return str(target_name_attr or target)


def _require_cudaq_gpu_target() -> Tuple[Any, str]:
    cudaq = _require_cudaq()
    active_target = _cudaq_target_name(cudaq)
    if not active_target.startswith("nvidia"):
        try:
            cudaq.set_target("nvidia")
        except Exception as exc:
            raise RuntimeError(
                "CUDA-Q exact branch is configured to require the NVIDIA GPU target, "
                "but cudaq.set_target('nvidia') failed. "
                "Check CUDA_VISIBLE_DEVICES, CUDA driver visibility, and CUDA-Q GPU support."
            ) from exc
        active_target = _cudaq_target_name(cudaq)

    if not active_target.startswith("nvidia"):
        raise RuntimeError(
            f"CUDA-Q exact branch requires an NVIDIA GPU target, but active target is {active_target!r}."
        )
    return cudaq, active_target


def central_difference_gradient(
    *,
    theta: np.ndarray,
    objective_fn: Callable[[Sequence[float]], float],
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


def build_cudaq_exact_backend(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
) -> Dict[str, Any]:
    cudaq, active_target = _require_cudaq_gpu_target()
    kernel, params = cudaq.make_kernel(list[float])
    q = kernel.qalloc(int(n_qubits))

    for i in range(int(n_qubits)):
        kernel.h(q[i])

    for layer in range(int(p_layers)):
        gamma_idx = 2 * layer
        beta_idx = 2 * layer + 1
        for (u, v) in edges:
            uu = int(u)
            vv = int(v)
            kernel.cx(q[uu], q[vv])
            kernel.rz(params[gamma_idx], q[vv])
            kernel.cx(q[uu], q[vv])
        for i in range(int(n_qubits)):
            kernel.rx(params[beta_idx], q[i])

    hamiltonian = None
    for (u, v) in edges:
        term = cudaq.spin.z(int(u)) * cudaq.spin.z(int(v))
        hamiltonian = term if hamiltonian is None else (hamiltonian + term)

    if hamiltonian is None:
        raise ValueError("Edge list must not be empty for MaxCut.")

    def _evaluate_single(thetas: Sequence[float]) -> float:
        theta_np = np.asarray(thetas, dtype=np.float64).reshape(-1)
        param_list = [float(x) for x in theta_np.tolist()]
        return float(cudaq.observe(kernel, hamiltonian, param_list).expectation())

    def _evaluate_batch(theta_batch: Sequence[Sequence[float]]) -> np.ndarray:
        theta_np = np.asarray(theta_batch, dtype=np.float64)
        if theta_np.ndim == 1:
            theta_np = theta_np.reshape(1, -1)
        results = cudaq.observe(kernel, hamiltonian, theta_np)
        if isinstance(results, list):
            return np.asarray([float(res.expectation()) for res in results], dtype=np.float64)
        return np.asarray([float(results.expectation())], dtype=np.float64)

    return {
        "backend": "cudaq.observe",
        "target": str(active_target),
        "kernel": kernel,
        "hamiltonian": hamiltonian,
        "evaluate_single": _evaluate_single,
        "evaluate_batch": _evaluate_batch,
    }


def train_with_exact_optimizer_cudaq(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    start_thetas_np: np.ndarray,
    n_edges: int,
    steps: int,
    lr: float,
    stage_name: str,
    log_every: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    cudaq, active_target = _require_cudaq_gpu_target()
    exact_backend = build_cudaq_exact_backend(
        n_qubits=int(n_qubits),
        edges=edges,
        p_layers=int(p_layers),
    )
    objective_eval = exact_backend["evaluate_single"]
    optimizer = cudaq.optimizers.Adam()
    optimizer.max_iterations = int(steps)
    optimizer.step_size = float(lr)
    optimizer.initial_parameters = [float(x) for x in np.asarray(start_thetas_np, dtype=np.float64).tolist()]

    call_rows: List[Dict[str, Any]] = []
    live_log_every = max(1, min(int(log_every), 10))

    def objective(parameter_vector: Sequence[float]) -> Tuple[float, List[float]]:
        theta_np = np.asarray(parameter_vector, dtype=np.float64).reshape(-1)
        current_sum_zz = float(objective_eval(theta_np))
        grad = np.asarray(
            central_difference_gradient(
                theta=theta_np,
                objective_fn=objective_eval,
                epsilon=1.0e-4,
            ),
            dtype=np.float64,
        )
        call_rows.append(
            {
                "thetas": theta_np.tolist(),
                "sum_zz": float(current_sum_zz),
            }
        )
        call_index = len(call_rows)
        step_hint = max(0, call_index - 1)
        if call_index == 1:
            print(
                f"[{stage_name}] initial "
                f"sum<ZZ>={float(current_sum_zz):+.6f} "
                f"E[cut]={expected_cut_from_sum_zz(float(current_sum_zz), int(n_edges)):.6f}"
            )
        elif (step_hint == 1) or (step_hint % int(live_log_every) == 0) or (step_hint >= int(steps)):
            print(
                f"[{stage_name}] live step~{step_hint:04d}/{int(steps)} "
                f"sum<ZZ>={float(current_sum_zz):+.6f} "
                f"E[cut]={expected_cut_from_sum_zz(float(current_sum_zz), int(n_edges)):.6f}"
            )
        return current_sum_zz, grad.tolist()

    print(
        f"[{stage_name}] CUDA-Q exact training for {int(steps)} steps "
        f"on target={str(exact_backend['target'])}"
    )
    _, final_thetas_list = optimizer.optimize(2 * int(p_layers), objective)
    final_thetas_np = np.asarray(final_thetas_list, dtype=np.float64)
    final_sum_zz = float(objective_eval(final_thetas_np))

    post_update_rows = call_rows[1:] + [{"thetas": final_thetas_np.tolist(), "sum_zz": float(final_sum_zz)}]
    history: List[Dict[str, Any]] = []
    theta_trajectory: List[Dict[str, Any]] = []
    for step, row in enumerate(post_update_rows[: int(steps)], start=1):
        history.append(_history_rows(step=step, sum_zz=float(row["sum_zz"]), n_edges=n_edges))
        theta_trajectory.append(
            {
                "step": int(step),
                "thetas": list(row["thetas"]),
            }
        )

    if history:
        print(
            f"[{stage_name}] completed optimizer_steps={len(history):04d}/{int(steps)} "
            f"sum<ZZ>={float(history[-1]['sum_zz']):+.6f} "
            f"E[cut]={float(history[-1]['expected_cut']):.6f}"
        )

    runtime = {
        "backend": "cudaq.observe",
        "target": str(active_target),
        "gradient_method": "central_difference_manual",
        "optimizer": "cudaq.optimizers.Adam",
        "lr": float(lr),
        "max_iterations": int(steps),
    }
    return history, final_thetas_np, theta_trajectory, runtime


def evaluate_exact_on_surrogate_trajectory(
    *,
    exact_sum_zz_batch_fn: Callable[[Sequence[Sequence[float]]], np.ndarray],
    theta_trajectory: Sequence[Dict[str, Any]],
    surrogate_history: Sequence[Dict[str, Any]],
    n_edges: int,
) -> List[Dict[str, Any]]:
    surrogate_by_step = {int(row["step"]): dict(row) for row in surrogate_history}
    theta_batch = np.asarray([row["thetas"] for row in theta_trajectory], dtype=np.float64)
    exact_sum_zz_vals = exact_sum_zz_batch_fn(theta_batch)
    rows: List[Dict[str, Any]] = []
    for theta_row, sum_zz in zip(theta_trajectory, exact_sum_zz_vals):
        step = int(theta_row["step"])
        base = surrogate_by_step[step]
        rows.append(
            {
                "step": int(step),
                "surrogate_sum_zz": float(base["sum_zz"]),
                "surrogate_expected_cut": float(base["expected_cut"]),
                "exact_sum_zz": float(sum_zz),
                "exact_expected_cut": float(expected_cut_from_sum_zz(float(sum_zz), int(n_edges))),
            }
        )
    return rows


def evaluate_surrogate_on_exact_trajectory(
    *,
    surrogate_program: Any,
    theta_trajectory: Sequence[Dict[str, Any]],
    exact_history: Sequence[Dict[str, Any]],
    device: str,
    n_edges: int,
) -> List[Dict[str, Any]]:
    exact_by_step = {int(row["step"]): dict(row) for row in exact_history}
    rows: List[Dict[str, Any]] = []
    for theta_row in theta_trajectory:
        step = int(theta_row["step"])
        thetas = torch.tensor(theta_row["thetas"], dtype=torch.float64, device=device)
        with torch.no_grad():
            sum_zz = float(surrogate_program.expval(thetas, obs_index=0).detach().cpu().item())
        base = exact_by_step[step]
        rows.append(
            {
                "step": int(step),
                "exact_sum_zz": float(base["sum_zz"]),
                "exact_expected_cut": float(base["expected_cut"]),
                "surrogate_sum_zz": float(sum_zz),
                "surrogate_expected_cut": float(expected_cut_from_sum_zz(float(sum_zz), int(n_edges))),
            }
        )
    return rows


def _plot_step_series(
    *,
    series: Sequence[Tuple[str, Sequence[float], str, str]],
    output_path: Path,
    title: str,
    ylabel: str = "Expected Cut",
    rebuild_steps: Optional[Sequence[int]] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    max_len = 0
    for label, values, color, linestyle in series:
        steps = np.arange(1, len(values) + 1)
        max_len = max(max_len, len(values))
        ax.plot(steps, values, label=label, color=color, linestyle=linestyle, linewidth=2.0)

    for step in rebuild_steps or []:
        if int(step) <= 0:
            continue
        ax.axvline(float(step), color="gray", linestyle=":", alpha=0.35)

    ax.set_xlim(1, max(1, max_len))
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_case1_exact_warmup(
    *,
    exact_history: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    _plot_step_series(
        series=[
            ("Exact value", [row["expected_cut"] for row in exact_history], "black", "-"),
        ],
        output_path=output_path,
        title="Case 1: Exact Warm-Up",
    )


def plot_case2_lwpp_warmup(
    *,
    surrogate_history: Sequence[Dict[str, Any]],
    exact_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    _plot_step_series(
        series=[
            ("MW surrogate value", [row["expected_cut"] for row in surrogate_history], "#4c78a8", "--"),
            ("Exact value on MW trajectory", [row["exact_expected_cut"] for row in exact_rows], "black", "-"),
        ],
        output_path=output_path,
        title="Case 2: LWPP Warm-Up",
    )


def plot_case3_lwpp_to_exact(
    *,
    exact_history: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    _plot_step_series(
        series=[
            ("Exact value", [row["expected_cut"] for row in exact_history], "#2ca02c", "-"),
        ],
        output_path=output_path,
        title="Case 3: LWPP -> Exact Fine-Tuning",
    )


def plot_case4_lwpp_to_coeff(
    *,
    surrogate_history: Sequence[Dict[str, Any]],
    exact_rows: Sequence[Dict[str, Any]],
    rebuild_log: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    _plot_step_series(
        series=[
            ("Coeff surrogate value", [row["expected_cut"] for row in surrogate_history], "#f58518", "--"),
            ("Exact value on coeff trajectory", [row["exact_expected_cut"] for row in exact_rows], "black", "-"),
        ],
        output_path=output_path,
        title="Case 4: LWPP -> Coefficient Truncation Fine-Tuning",
        rebuild_steps=[int(row["after_step"]) for row in rebuild_log if int(row["after_step"]) > 0],
    )


def plot_integrated_case_comparison(
    *,
    case1_history: Sequence[Dict[str, Any]],
    case2_history: Sequence[Dict[str, Any]],
    case2_exact_rows: Sequence[Dict[str, Any]],
    case3_history: Sequence[Dict[str, Any]],
    case4_history: Sequence[Dict[str, Any]],
    case4_exact_rows: Sequence[Dict[str, Any]],
    case4_rebuild_log: Sequence[Dict[str, Any]],
    warmup_steps_nominal: Optional[int] = None,
    finetune_steps_nominal: Optional[int] = None,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    warmup_steps = int(
        warmup_steps_nominal
        if warmup_steps_nominal is not None
        else max(len(case1_history), len(case2_exact_rows))
    )
    finetune_steps = int(
        finetune_steps_nominal
        if finetune_steps_nominal is not None
        else max(len(case3_history), len(case4_exact_rows))
    )
    total_steps = int(warmup_steps + finetune_steps)
    boundary_x = float(warmup_steps) + 0.5
    rebuild_global_lines = [
        int(warmup_steps) + int(row["after_step"])
        for row in case4_rebuild_log
        if int(row.get("after_step", 0)) > 0
    ]

    fig, ax = plt.subplots(figsize=(12, 5.6))

    if total_steps > warmup_steps:
        ax.axvspan(
            boundary_x,
            float(total_steps) + 0.5,
            color="#f58518",
            alpha=0.06,
            zorder=0,
        )

    ax.plot(
        [row["step"] for row in case1_history],
        [row["expected_cut"] for row in case1_history],
        color="black",
        linewidth=2.2,
        label="Case 1 exact warmup",
    )
    ax.plot(
        [row["step"] for row in case2_history],
        [row["expected_cut"] for row in case2_history],
        color="#4c78a8",
        linewidth=2.0,
        linestyle="--",
        label="Case 2 MW surrogate",
    )
    ax.plot(
        [row["step"] for row in case2_exact_rows],
        [row["exact_expected_cut"] for row in case2_exact_rows],
        color="#4c78a8",
        linewidth=2.2,
        label="Case 2 exact on MW warmup",
    )
    ax.plot(
        [int(warmup_steps) + int(row["step"]) for row in case3_history],
        [row["expected_cut"] for row in case3_history],
        color="#2ca02c",
        linewidth=2.2,
        label="Case 3 exact fine-tune",
    )
    ax.plot(
        [int(warmup_steps) + int(row["step"]) for row in case4_history],
        [row["expected_cut"] for row in case4_history],
        color="#f58518",
        linewidth=2.0,
        linestyle="--",
        label="Case 4 coeff surrogate",
    )
    ax.plot(
        [int(warmup_steps) + int(row["step"]) for row in case4_exact_rows],
        [row["exact_expected_cut"] for row in case4_exact_rows],
        color="#f58518",
        linewidth=2.2,
        label="Case 4 exact on coeff fine-tune",
    )

    ax.axvline(boundary_x, color="gray", linestyle=":", linewidth=1.1, alpha=0.9)
    for x in rebuild_global_lines:
        ax.axvline(float(x), color="gray", linestyle=":", linewidth=0.9, alpha=0.22, zorder=0)

    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.04 * (y_max - y_min) if y_max > y_min else y_max
    ax.text(max(1.5, 0.04 * max(1, warmup_steps)), y_text, "Warm-up", color="gray", va="top", fontsize=10)
    if total_steps > warmup_steps:
        ax.text(boundary_x + 2.0, y_text, "Fine-tuning", color="gray", va="top", fontsize=10)

    ax.set_xlim(1, max(1, total_steps))
    ax.set_xlabel(
        f"Global step (warm-up 1-{int(warmup_steps)}, fine-tuning {int(warmup_steps) + 1}-{int(total_steps)})"
    )
    ax.set_ylabel("Expected Cut")
    ax.set_title("Integrated Surrogate and Exact Comparison Across the Four Cases")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _final_exact_expected_cut_from_case(case_payload: Dict[str, Any], *, case_name: str) -> float:
    if case_name == "case1_exact_warmup":
        return float(case_payload["history"][-1]["expected_cut"])
    if case_name == "case2_lwpp_warmup":
        return float(case_payload["exact_curve"][-1]["exact_expected_cut"])
    if case_name == "case3_lwpp_to_exact":
        return float(case_payload["history"][-1]["expected_cut"])
    if case_name == "case4_lwpp_to_coeff":
        return float(case_payload["exact_curve"][-1]["exact_expected_cut"])
    raise KeyError(case_name)


def build_run_summary(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    cases = artifacts["cases"]
    return {
        "init_strategy": artifacts["config"]["init_strategy"],
        "run_seed": int(artifacts["config"]["run_seed"]),
        "n_qubits": int(artifacts["config"]["n_qubits"]),
        "p_layers": int(artifacts["config"]["p_layers"]),
        "graph_seed": int(artifacts["config"].get("graph_seed_used", artifacts["config"]["graph_seed"])),
        "final_exact_expected_cut": {
            name: _final_exact_expected_cut_from_case(payload, case_name=name)
            for name, payload in cases.items()
        },
    }


def _mean_std_curves(curves: Sequence[Sequence[float]]) -> Dict[str, List[float]]:
    if len(curves) == 0:
        return {"mean": [], "std": []}

    max_len = max(len(curve) for curve in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for idx, curve in enumerate(curves):
        curve_arr = np.asarray(curve, dtype=np.float64)
        arr[idx, : len(curve_arr)] = curve_arr

    return {
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std": np.nanstd(arr, axis=0).tolist(),
    }


def _finite_curve_xy(curve_payload: Dict[str, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(curve_payload["mean"], dtype=np.float64)
    mask = np.isfinite(y)
    x = np.arange(1, len(y) + 1, dtype=np.int64)[mask]
    return x, y[mask]


def _finite_band_xy(curve_payload: Dict[str, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(curve_payload["mean"], dtype=np.float64)
    s = np.asarray(curve_payload["std"], dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(s)
    x = np.arange(1, len(y) + 1, dtype=np.int64)[mask]
    return x, y[mask], s[mask]


def aggregate_group_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(records) == 0:
        raise ValueError("records must not be empty")

    exact_case1 = [[row["expected_cut"] for row in rec["cases"]["case1_exact_warmup"]["history"]] for rec in records]
    exact_case2 = [[row["exact_expected_cut"] for row in rec["cases"]["case2_lwpp_warmup"]["exact_curve"]] for rec in records]
    mw_case2 = [[row["expected_cut"] for row in rec["cases"]["case2_lwpp_warmup"]["history"]] for rec in records]
    exact_case3 = [[row["expected_cut"] for row in rec["cases"]["case3_lwpp_to_exact"]["history"]] for rec in records]
    exact_case4 = [[row["exact_expected_cut"] for row in rec["cases"]["case4_lwpp_to_coeff"]["exact_curve"]] for rec in records]
    coeff_case4 = [[row["expected_cut"] for row in rec["cases"]["case4_lwpp_to_coeff"]["history"]] for rec in records]

    finals_by_case: Dict[str, List[float]] = {
        "case1_exact_warmup": [curve[-1] for curve in exact_case1],
        "case2_lwpp_warmup": [curve[-1] for curve in exact_case2],
        "case3_lwpp_to_exact": [curve[-1] for curve in exact_case3],
        "case4_lwpp_to_coeff": [curve[-1] for curve in exact_case4],
    }

    return {
        "n_runs": int(len(records)),
        "case_curves": {
            "case1_exact_warmup": {
                "exact": _mean_std_curves(exact_case1),
            },
            "case2_lwpp_warmup": {
                "exact": _mean_std_curves(exact_case2),
                "mw_surrogate": _mean_std_curves(mw_case2),
            },
            "case3_lwpp_to_exact": {
                "exact": _mean_std_curves(exact_case3),
            },
            "case4_lwpp_to_coeff": {
                "exact": _mean_std_curves(exact_case4),
                "coeff_surrogate": _mean_std_curves(coeff_case4),
            },
        },
        "final_exact_expected_cut": {
            name: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": [float(v) for v in vals],
            }
            for name, vals in finals_by_case.items()
        },
    }


def plot_aggregate_mean_curves(aggregate_payload: Dict[str, Any], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    case_curves = aggregate_payload["case_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    ax = axes[0]
    x, y, s = _finite_band_xy(case_curves["case1_exact_warmup"]["exact"])
    ax.plot(x, y, color="black", linewidth=2.0, label="Case 1 exact")
    ax.fill_between(x, y - s, y + s, color="black", alpha=0.12)

    x, y, s = _finite_band_xy(case_curves["case2_lwpp_warmup"]["exact"])
    ax.plot(x, y, color="#4c78a8", linewidth=2.0, label="Case 2 exact")
    ax.fill_between(x, y - s, y + s, color="#4c78a8", alpha=0.15)
    ax.set_title("Warm-Up Stage")
    ax.set_xlabel("Step")
    ax.set_ylabel("Expected Cut")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    x, y, s = _finite_band_xy(case_curves["case3_lwpp_to_exact"]["exact"])
    ax.plot(x, y, color="#2ca02c", linewidth=2.0, label="Case 3 exact")
    ax.fill_between(x, y - s, y + s, color="#2ca02c", alpha=0.15)

    x, y, s = _finite_band_xy(case_curves["case4_lwpp_to_coeff"]["exact"])
    ax.plot(x, y, color="#f58518", linewidth=2.0, label="Case 4 exact")
    ax.fill_between(x, y - s, y + s, color="#f58518", alpha=0.15)
    ax.set_title("Fine-Tuning Stage")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_aggregate_figure1_like(aggregate_payload: Dict[str, Any], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    case_curves = aggregate_payload["case_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    ax = axes[0]
    x, y = _finite_curve_xy(case_curves["case1_exact_warmup"]["exact"])
    ax.plot(x, y, color="black", linewidth=2.0, label="Exact optimization")
    ax.set_title("Exact Optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel("Expected Cut")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    x, y = _finite_curve_xy(case_curves["case2_lwpp_warmup"]["mw_surrogate"])
    ax.plot(x, y, color="#4c78a8", linestyle="--", linewidth=2.0, label="MW optimization")
    x, y = _finite_curve_xy(case_curves["case2_lwpp_warmup"]["exact"])
    ax.plot(x, y, color="black", linewidth=2.0, label="Exact eval")
    ax.set_title("MW Optimization + Exact Evaluation")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_seed_overlay_figure2_like(records: Sequence[Dict[str, Any]], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, rec in enumerate(records):
        color = color_cycle[idx % len(color_cycle)]
        seed = int(rec["config"]["run_seed"])
        label = f"seed {seed}"
        case1 = rec["cases"]["case1_exact_warmup"]["history"]
        case2 = rec["cases"]["case2_lwpp_warmup"]
        case3 = rec["cases"]["case3_lwpp_to_exact"]["history"]

        axes[0, 0].plot(
            [row["step"] for row in case1],
            [row["expected_cut"] for row in case1],
            color=color,
            linewidth=1.8,
            label=label,
        )
        axes[0, 1].plot(
            [row["step"] for row in case3],
            [row["expected_cut"] for row in case3],
            color=color,
            linewidth=1.8,
            label=label,
        )
        axes[1, 0].plot(
            [row["step"] for row in case2["history"]],
            [row["expected_cut"] for row in case2["history"]],
            color=color,
            linewidth=1.8,
            label=label,
        )
        axes[1, 1].plot(
            [row["step"] for row in case2["exact_curve"]],
            [row["exact_expected_cut"] for row in case2["exact_curve"]],
            color=color,
            linewidth=1.8,
            label=label,
        )

    axes[0, 0].set_title("Direct Exact Optimization")
    axes[0, 1].set_title("LWPP-Initialized Exact Fine-Tuning")
    axes[1, 0].set_title("LWPP Warm-Up Surrogate Value")
    axes[1, 1].set_title("LWPP Warm-Up Exact Value")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Expected Cut")

    axes[0, 0].legend(ncol=2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
