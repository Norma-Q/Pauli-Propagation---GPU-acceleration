from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

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
        build_qaoa_theta_init_flattened_tqa,
        build_qaoa_theta_init_tqa,
        choose_device,
        expected_cut_from_sum_zz,
    )
except ImportError:
    from qaoa_surrogate_common import (
        build_maxcut_observable,
        build_qaoa_circuit,
        build_qaoa_theta_init_flattened_tqa,
        build_qaoa_theta_init_tqa,
        choose_device,
        expected_cut_from_sum_zz,
    )


DEFAULT_OUTPUT_ROOT = _THIS_DIR / "mw_training_coeff_finetuning_results"
MANAGED_OUTPUT_FILENAMES = {
    "artifacts.json",
    "integrated_training_exact_curve.png",
    "sampling_histogram.png",
    "sampling_stage_comparison.png",
    "notebook_config.json",
    "source_config.yaml",
    "mw_training_log.json",
    "mw_training_curve.png",
    "coeff_finetune_log.json",
    "coeff_finetune_curve.png",
    "coeff_rebuild_diagnostics.png",
    "combined_training_curve.json",
    "exact_curve_cache.json",
    "sampling_counts.json",
    "sampling_summary.json",
    "sampling_cut_arrays.npz",
    "run_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MW pretraining followed by coefficient-truncated finetuning for QAOA."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config for mw_training_coeff_finetuning.py",
    )
    parser.add_argument(
        "--init-strategy",
        type=str,
        default="",
        help="Optional override for QAOA initialization strategy (e.g. random, flattened_tqa).",
    )
    return parser.parse_args()


def cleanup_memory(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def float_tag(x: float) -> str:
    text = f"{float(x):.0e}" if abs(float(x)) < 1e-2 else str(float(x))
    return text.replace("-", "m").replace(".", "p")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def cleanup_managed_outputs(run_dir: Path, keep_filenames: Sequence[str]) -> None:
    keep = {str(name) for name in keep_filenames}
    for name in MANAGED_OUTPUT_FILENAMES:
        if name in keep:
            continue
        path = run_dir / name
        if path.exists():
            path.unlink()


def theta_trajectory_to_step_map(theta_trajectory: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for row in theta_trajectory:
        out[str(int(row["step"]))] = list(row["thetas"])
    return out


def checkpoint_thetas_to_step_map(
    checkpoint_thetas: Dict[str, Any],
    *,
    drop_step_zero: bool = True,
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for key, value in checkpoint_thetas.items():
        if not str(key).startswith("step_"):
            continue
        step = int(str(key).split("_", 1)[1])
        if drop_step_zero and step == 0:
            continue
        out[str(step)] = list(value)
    return out


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping.")
    return data


def load_existing_graph(
    *,
    graph_dir: Path,
    n_qubits: int,
) -> Tuple[List[Tuple[int, int]], Path, Optional[Path]]:
    json_path = graph_dir / f"Q{n_qubits}_edges.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected graph file not found for Q{n_qubits}: {json_path}"
        )

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Graph edge file must contain a list of [u, v] pairs: {json_path}")
    edges = [(int(u), int(v)) for u, v in raw]

    png_candidates = sorted(graph_dir.glob(f"Q{n_qubits}_*.png"))
    png_path = png_candidates[0] if png_candidates else None
    return edges, json_path, png_path


def edges_signature(edges: Sequence[Tuple[int, int]]) -> str:
    payload = json.dumps([[int(u), int(v)] for (u, v) in edges], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def brute_force_maxcut_summary(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    best_cut = -1
    n_optimal_assignments = 0
    representative_code: Optional[int] = None
    for code in range(1 << int(n_qubits)):
        cut = 0
        for u, v in edges:
            bu = (int(code) >> int(u)) & 1
            bv = (int(code) >> int(v)) & 1
            cut += int(bu != bv)
        if cut > best_cut:
            best_cut = int(cut)
            n_optimal_assignments = 1
            representative_code = int(code)
        elif cut == best_cut:
            n_optimal_assignments += 1

    return {
        "best_cut": int(best_cut),
        "n_optimal_assignments": int(n_optimal_assignments),
        "representative_code": None if representative_code is None else int(representative_code),
    }


def load_or_compute_bruteforce_optimum(
    *,
    graph_dir: Path,
    source_graph_json: Path,
    n_qubits: int,
    n_edges: int,
    edges: Sequence[Tuple[int, int]],
    max_qubits: int = 20,
) -> Optional[Dict[str, Any]]:
    if int(n_qubits) > int(max_qubits):
        return None

    cache_path = graph_dir / f"Q{int(n_qubits)}_bruteforce_maxcut.json"
    signature = edges_signature(edges)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("edges_signature") == signature
            and int(cached.get("n_qubits", -1)) == int(n_qubits)
        ):
            return cached

    summary = brute_force_maxcut_summary(
        n_qubits=int(n_qubits),
        edges=edges,
    )
    payload = {
        "source_graph_json": str(source_graph_json),
        "cache_path": str(cache_path),
        "n_qubits": int(n_qubits),
        "n_edges": int(n_edges),
        "edges_signature": str(signature),
        "best_cut": int(summary["best_cut"]),
        "best_expected_cut": float(summary["best_cut"]),
        "best_sum_zz": float(int(n_edges) - 2 * int(summary["best_cut"])),
        "n_optimal_assignments": int(summary["n_optimal_assignments"]),
        "representative_code": summary["representative_code"],
    }
    save_json(cache_path, payload)
    return payload


def normalize_init_strategy(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in {"flattened_tqa", "flattend_tqa"}:
        return "flattened_tqa"
    if key == "tqa":
        return "tqa"
    if key == "random":
        return "random"
    raise ValueError(f"Unknown init_strategy: {raw}")


def init_strategy_output_tag(raw: str) -> str:
    key = normalize_init_strategy(raw)
    if key == "flattened_tqa":
        return "flattend_tqa"
    return key


def build_initial_theta_np(
    *,
    init_strategy: str,
    n_layers: int,
    n_edges: int,
    n_qubits: int,
    delta_t: float,
    flatten_alpha: float,
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
    if key == "flattened_tqa":
        return build_qaoa_theta_init_flattened_tqa(
            p_layers=int(n_layers),
            n_edges=int(n_edges),
            n_qubits=int(n_qubits),
            delta_t=float(delta_t),
            flatten_alpha=float(flatten_alpha),
            dtype=np.float64,
        )
    if key == "random":
        rng = np.random.default_rng(int(seed))
        return rng.uniform(low=-np.pi, high=np.pi, size=(2 * int(n_layers),)).astype(np.float64)
    raise ValueError(f"Unknown init_strategy: {init_strategy}")


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
) -> Callable[..., Tuple[Any, Dict[str, Any]]]:
    preset = "hybrid" if str(device).startswith("cuda") else "cpu"
    effective_max_weight = int(n_qubits) if max_weight_override is None else int(max_weight_override)
    max_weight_mode = "full_support" if max_weight_override is None else "truncated"

    def _compile(*, build_thetas: Optional[torch.Tensor], build_min_abs: Optional[float]) -> Tuple[Any, Dict[str, Any]]:
        compile_start = time.time()
        preset_overrides: Dict[str, Any] = {
            "chunk_size": int(chunk_size),
            "compute_device": str(device) if str(device).startswith("cuda") else "cpu",
            "max_weight": int(effective_max_weight),
        }
        program = compile_expval_program(
            circuit=circuit,
            observables=[zz_obj],
            preset=preset,
            preset_overrides=preset_overrides,
            build_thetas=build_thetas,
            build_min_abs=build_min_abs,
            parallel_compile=bool(parallel_compile),
        )
        info = extract_compile_resources(program)
        info.update(
            {
                "stage_name": str(stage_name),
                "preset": str(preset),
                "preset_overrides": dict(preset_overrides),
                "compile_seconds": float(time.time() - compile_start),
                "build_min_abs": None if build_min_abs is None else float(build_min_abs),
                "effective_max_weight": int(effective_max_weight),
                "max_weight_mode": str(max_weight_mode),
            }
        )
        return program, info

    return _compile


def train_with_fixed_program(
    *,
    program: Any,
    thetas: torch.nn.Parameter,
    n_edges: int,
    steps: int,
    lr: float,
    checkpoint_steps: Sequence[int],
    stage_name: str,
    log_every: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    checkpoints = {"step_0": thetas.detach().cpu().numpy().tolist()}
    theta_trajectory: List[Dict[str, Any]] = []
    checkpoint_set = {int(x) for x in checkpoint_steps}
    optimizer = torch.optim.Adam([thetas], lr=float(lr))

    print(f"[{stage_name}] Starting training for {steps} steps...")
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = program.expval(thetas, obs_index=0)
        if not bool(loss.requires_grad):
            raise RuntimeError(
                "loss does not require gradients. Check compile settings for this stage."
            )
        loss.backward()
        optimizer.step()

        sum_zz = float(loss.detach().cpu().item())
        expected_cut = float(expected_cut_from_sum_zz(sum_zz, int(n_edges)))
        theta_np = thetas.detach().cpu().numpy()
        history.append(
            {
                "step": int(step + 1),
                "sum_zz": float(sum_zz),
                "expected_cut": float(expected_cut),
            }
        )
        theta_trajectory.append(
            {
                "step": int(step + 1),
                "thetas": theta_np.tolist(),
            }
        )

        if (step + 1) in checkpoint_set:
            checkpoints[f"step_{step + 1}"] = theta_np.tolist()

        if ((step + 1) % int(log_every) == 0) or (step == 0) or (step + 1 == int(steps)):
            print(
                f"[{stage_name}] step={step + 1:04d} "
                f"sum<ZZ>={sum_zz:+.6f} E[cut]={expected_cut:.6f}"
            )

    return history, checkpoints, thetas.detach().cpu().numpy(), theta_trajectory


def train_with_periodic_rebuild(
    *,
    compile_program_fn: Callable[..., Tuple[Any, Dict[str, Any]]],
    start_thetas_np: np.ndarray,
    device: str,
    n_edges: int,
    steps: int,
    lr: float,
    checkpoint_steps: Sequence[int],
    build_min_abs: float,
    rebuild_interval: int,
    stage_name: str,
    log_every: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    thetas = torch.nn.Parameter(
        torch.tensor(np.asarray(start_thetas_np), dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam([thetas], lr=float(lr))

    history: List[Dict[str, Any]] = []
    checkpoints = {"step_0": thetas.detach().cpu().numpy().tolist()}
    theta_trajectory: List[Dict[str, Any]] = []
    checkpoint_set = {int(x) for x in checkpoint_steps}
    rebuild_log: List[Dict[str, Any]] = []

    anchor = thetas.detach().clone()
    program, compile_info = compile_program_fn(
        build_thetas=anchor,
        build_min_abs=float(build_min_abs),
    )
    rebuild_log.append({"rebuild_index": 0, "after_step": 0, **compile_info})

    print(
        f"[{stage_name}] Starting finetuning for {steps} steps "
        f"with build_min_abs={build_min_abs:.1e} and rebuild_interval={rebuild_interval}."
    )
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = program.expval(thetas, obs_index=0)
        if not bool(loss.requires_grad):
            raise RuntimeError(
                "loss does not require gradients under the current coefficient truncation. "
                "Use a smaller build_min_abs or rebuild more often."
            )
        loss.backward()
        optimizer.step()

        sum_zz = float(loss.detach().cpu().item())
        expected_cut = float(expected_cut_from_sum_zz(sum_zz, int(n_edges)))
        theta_np = thetas.detach().cpu().numpy()
        history.append(
            {
                "step": int(step + 1),
                "sum_zz": float(sum_zz),
                "expected_cut": float(expected_cut),
            }
        )
        theta_trajectory.append(
            {
                "step": int(step + 1),
                "thetas": theta_np.tolist(),
            }
        )

        if (step + 1) in checkpoint_set:
            checkpoints[f"step_{step + 1}"] = theta_np.tolist()

        if ((step + 1) % int(log_every) == 0) or (step == 0) or (step + 1 == int(steps)):
            print(
                f"[{stage_name}] step={step + 1:04d} "
                f"sum<ZZ>={sum_zz:+.6f} E[cut]={expected_cut:.6f}"
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
    return history, checkpoints, thetas.detach().cpu().numpy(), rebuild_log, theta_trajectory


def train_with_exact_optimizer_pennylane(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    start_thetas_np: np.ndarray,
    n_edges: int,
    steps: int,
    lr: float,
    checkpoint_steps: Sequence[int],
    device_name: str,
    log_every: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except Exception as exc:
        raise RuntimeError(f"PennyLane exact optimizer backend is not available: {exc}") from exc

    dev = qml.device(str(device_name), wires=int(n_qubits))
    obs = _build_pennylane_sumzz_observable(qml, edges)

    @qml.qnode(dev, interface="autograd", diff_method="adjoint")
    def qnode(params):
        _apply_qaoa_pennylane(
            qml=qml,
            params=params,
            n_qubits=int(n_qubits),
            edges=edges,
            p_layers=int(p_layers),
        )
        return qml.expval(obs)

    def objective(params):
        return qnode(params)

    history: List[Dict[str, Any]] = []
    thetas = pnp.array(np.asarray(start_thetas_np, dtype=np.float64), requires_grad=True)
    checkpoints = {"step_0": np.asarray(thetas, dtype=np.float64).tolist()}
    theta_trajectory: List[Dict[str, Any]] = []
    checkpoint_set = {int(x) for x in checkpoint_steps}
    optimizer = qml.AdamOptimizer(stepsize=float(lr))

    print(
        f"[exact-opt] Starting exact finetuning for {steps} steps "
        f"on {device_name} with PennyLane Adam."
    )
    for step in range(int(steps)):
        thetas = optimizer.step(objective, thetas)
        sum_zz = float(objective(thetas))
        expected_cut = float(expected_cut_from_sum_zz(sum_zz, int(n_edges)))
        theta_np = np.asarray(thetas, dtype=np.float64)

        history.append(
            {
                "step": int(step + 1),
                "sum_zz": float(sum_zz),
                "expected_cut": float(expected_cut),
            }
        )
        theta_trajectory.append(
            {
                "step": int(step + 1),
                "thetas": theta_np.tolist(),
            }
        )

        if (step + 1) in checkpoint_set:
            checkpoints[f"step_{step + 1}"] = theta_np.tolist()

        if ((step + 1) % int(log_every) == 0) or (step == 0) or (step + 1 == int(steps)):
            print(
                f"[exact-opt] step={step + 1:04d} "
                f"sum<ZZ>={sum_zz:+.6f} E[cut]={expected_cut:.6f}"
            )

    backend_info = {
        "kind": "pennylane",
        "device_name": str(device_name),
        "optimizer": "AdamOptimizer",
        "stepsize": float(lr),
    }
    return history, checkpoints, np.asarray(thetas, dtype=np.float64), theta_trajectory, backend_info


def plot_history(history: List[Dict[str, Any]], title: str, output_path: Path) -> None:
    steps = [row["step"] for row in history]
    exp_cut = [row["expected_cut"] for row in history]
    sum_zz = [row["sum_zz"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, exp_cut, color="teal", linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Expected Cut")
    axes[0].set_title(f"{title}: Expected Cut")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, sum_zz, color="tab:orange", linewidth=2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Sum <ZZ>")
    axes[1].set_title(f"{title}: Sum <ZZ>")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rebuild_diagnostics(rebuild_log: List[Dict[str, Any]], output_path: Path) -> None:
    rebuild_steps = [row["after_step"] for row in rebuild_log]
    rebuild_nnz = [row["nnz_total"] for row in rebuild_log]
    rebuild_terms = [row["terms_after_zero_filter"] for row in rebuild_log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(rebuild_steps, rebuild_nnz, marker="o", color="tab:purple")
    axes[0].set_xlabel("After step")
    axes[0].set_ylabel("Total nnz")
    axes[0].set_title("Recompiled surrogate size")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rebuild_steps, rebuild_terms, marker="o", color="tab:green")
    axes[1].set_xlabel("After step")
    axes[1].set_ylabel("Terms after zero filter")
    axes[1].set_title("Surviving terms after rebuild")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_combined_surrogate_curve(
    *,
    mw_history: Sequence[Dict[str, Any]],
    coeff_history: Sequence[Dict[str, Any]],
    mw_steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in mw_history:
        rows.append(
            {
                "stage": "mw",
                "stage_step": int(row["step"]),
                "global_step": int(row["step"]),
                "sum_zz": float(row["sum_zz"]),
                "expected_cut": float(row["expected_cut"]),
            }
        )
    for row in coeff_history:
        rows.append(
            {
                "stage": "coeff",
                "stage_step": int(row["step"]),
                "global_step": int(mw_steps) + int(row["step"]),
                "sum_zz": float(row["sum_zz"]),
                "expected_cut": float(row["expected_cut"]),
            }
        )
    return rows


def build_combined_theta_trajectory(
    *,
    mw_theta_trajectory: Sequence[Dict[str, Any]],
    coeff_theta_trajectory: Sequence[Dict[str, Any]],
    mw_steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in mw_theta_trajectory:
        rows.append(
            {
                "stage": "mw",
                "stage_step": int(row["step"]),
                "global_step": int(row["step"]),
                "thetas": list(row["thetas"]),
            }
        )
    for row in coeff_theta_trajectory:
        rows.append(
            {
                "stage": "coeff",
                "stage_step": int(row["step"]),
                "global_step": int(mw_steps) + int(row["step"]),
                "thetas": list(row["thetas"]),
            }
        )
    return rows


def select_exact_backend_policy(
    *,
    n_qubits: int,
    large_step_stride: int,
) -> Dict[str, Any]:
    n = int(n_qubits)
    stride = max(1, int(large_step_stride))
    if n <= 10:
        return {
            "kind": "pennylane",
            "device_name": "lightning.qubit",
            "step_stride": int(stride),
            "batch_mode": "all",
        }
    if n <= 20:
        return {
            "kind": "pennylane",
            "device_name": "lightning.gpu",
            "step_stride": int(stride),
            "batch_mode": "all",
        }
    return {
        "kind": "none",
        "device_name": None,
        "step_stride": int(stride),
        "batch_mode": "disabled",
        "skipped_reason": "exact reevaluation is enabled only up to 20 qubits",
    }


def select_exact_optimizer_policy(n_qubits: int) -> Dict[str, Any]:
    n = int(n_qubits)
    if n <= 10:
        return {
            "enabled": True,
            "kind": "pennylane",
            "device_name": "lightning.qubit",
            "optimizer": "AdamOptimizer",
        }
    if n <= 20:
        return {
            "enabled": True,
            "kind": "pennylane",
            "device_name": "lightning.gpu",
            "optimizer": "AdamOptimizer",
        }
    return {
        "enabled": False,
        "kind": "none",
        "device_name": None,
        "optimizer": None,
        "skipped_reason": "exact optimizer branch is enabled only up to 20 qubits",
    }


def _apply_qaoa_pennylane(
    *,
    qml: Any,
    params: Any,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
) -> None:
    for q in range(int(n_qubits)):
        qml.Hadamard(wires=q)

    for layer in range(int(p_layers)):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        for (u, v) in edges:
            uu = int(u)
            vv = int(v)
            qml.CNOT(wires=[uu, vv])
            qml.RZ(gamma, wires=vv)
            qml.CNOT(wires=[uu, vv])
        for q in range(int(n_qubits)):
            qml.RX(beta, wires=q)


def _build_pennylane_sumzz_observable(qml: Any, edges: Sequence[Tuple[int, int]]) -> Any:
    obs = None
    for (u, v) in edges:
        term = qml.PauliZ(int(u)) @ qml.PauliZ(int(v))
        obs = term if obs is None else (obs + term)
    if obs is None:
        raise ValueError("MaxCut graph must contain at least one edge.")
    return obs


def evaluate_exact_sum_zz_batch_pennylane(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    theta_batch: np.ndarray,
    device_name: str,
) -> np.ndarray:
    try:
        import pennylane as qml
    except Exception as exc:
        raise RuntimeError(f"PennyLane exact backend is not available: {exc}") from exc

    theta_np = np.asarray(theta_batch, dtype=np.float64)
    if theta_np.ndim == 1:
        theta_np = theta_np.reshape(1, -1)
    if theta_np.ndim != 2:
        raise ValueError("theta_batch must have shape (batch, n_params)")

    dev = qml.device(str(device_name), wires=int(n_qubits))
    obs = _build_pennylane_sumzz_observable(qml, edges)

    @qml.qnode(dev)
    def qnode(params_by_op):
        _apply_qaoa_pennylane(
            qml=qml,
            params=params_by_op,
            n_qubits=int(n_qubits),
            edges=edges,
            p_layers=int(p_layers),
        )
        return qml.expval(obs)

    params_by_op = np.ascontiguousarray(theta_np.T)
    vals = np.asarray(qnode(params_by_op), dtype=np.float64).reshape(-1)
    return vals


def evaluate_exact_curve(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    combined_curve: Sequence[Dict[str, Any]],
    combined_theta_trajectory: Sequence[Dict[str, Any]],
    large_step_stride: int,
) -> Dict[str, Any]:
    policy = select_exact_backend_policy(
        n_qubits=int(n_qubits),
        large_step_stride=int(large_step_stride),
    )
    if str(policy["kind"]) == "none":
        return {
            "policy": dict(policy),
            "rows": [],
        }
    surrogate_by_global_step = {
        int(row["global_step"]): dict(row) for row in combined_curve
    }
    final_stage_step = {
        str(row["stage"]): max(
            int(r["stage_step"]) for r in combined_theta_trajectory if str(r["stage"]) == str(row["stage"])
        )
        for row in combined_theta_trajectory
    }

    selected_theta_rows = []
    for row in combined_theta_trajectory:
        stage_step = int(row["stage_step"])
        stage_name = str(row["stage"])
        is_final_stage_point = stage_step == int(final_stage_step[stage_name])
        if (
            int(policy["step_stride"]) > 1
            and (stage_step % int(policy["step_stride"]) != 0)
            and not is_final_stage_point
        ):
            continue
        selected_theta_rows.append(dict(row))

    exact_rows: List[Dict[str, Any]] = []
    if not selected_theta_rows:
        return {
            "policy": dict(policy),
            "rows": exact_rows,
        }

    theta_batch = np.asarray([row["thetas"] for row in selected_theta_rows], dtype=np.float64)
    sum_zz_vals = evaluate_exact_sum_zz_batch_pennylane(
        n_qubits=int(n_qubits),
        edges=edges,
        p_layers=int(p_layers),
        theta_batch=theta_batch,
        device_name=str(policy["device_name"]),
    )
    for row, sum_zz in zip(selected_theta_rows, sum_zz_vals):
        surrogate = surrogate_by_global_step[int(row["global_step"])]
        exact_rows.append(
            {
                "stage": str(row["stage"]),
                "stage_step": int(row["stage_step"]),
                "global_step": int(row["global_step"]),
                "surrogate_sum_zz": float(surrogate["sum_zz"]),
                "surrogate_expected_cut": float(surrogate["expected_cut"]),
                "exact_sum_zz": float(sum_zz),
                "exact_expected_cut": float(expected_cut_from_sum_zz(float(sum_zz), len(edges))),
            }
        )

    return {
        "policy": dict(policy),
        "rows": exact_rows,
    }


def build_exact_optimizer_curve(
    *,
    exact_history: Sequence[Dict[str, Any]],
    mw_steps: int,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in exact_history:
        rows.append(
            {
                "stage": "exact_optimizer",
                "stage_step": int(row["step"]),
                "global_step": int(mw_steps) + int(row["step"]),
                "exact_sum_zz": float(row["sum_zz"]),
                "exact_expected_cut": float(row["expected_cut"]),
            }
        )
    return {
        "policy": dict(policy),
        "rows": rows,
    }


def sample_counts_pennylane(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    thetas: np.ndarray,
    shots: int,
    seed: Optional[int],
    device_name: str,
) -> Dict[str, int]:
    try:
        import pennylane as qml
    except Exception as exc:
        raise RuntimeError(f"PennyLane sampling backend is not available: {exc}") from exc

    theta_np = np.asarray(thetas, dtype=np.float64).reshape(-1)
    dev = qml.device(str(device_name), wires=int(n_qubits), shots=int(shots), seed=seed)

    @qml.qnode(dev)
    def qnode(params):
        _apply_qaoa_pennylane(
            qml=qml,
            params=params,
            n_qubits=int(n_qubits),
            edges=edges,
            p_layers=int(p_layers),
        )
        return qml.sample(wires=list(range(int(n_qubits))))

    samples = np.asarray(qnode(theta_np), dtype=np.uint8)
    counts: Dict[str, int] = {}
    for row in samples:
        code = 0
        for q in range(int(n_qubits)):
            code |= (int(row[q]) & 1) << q
        key = str(int(code))
        counts[key] = counts.get(key, 0) + 1
    return counts


def select_sampling_backend_policy(n_qubits: int) -> Dict[str, str]:
    if int(n_qubits) <= 10:
        return {"kind": "pennylane", "device_name": "lightning.qubit"}
    if int(n_qubits) <= 20:
        return {"kind": "pennylane", "device_name": "lightning.gpu"}
    return {"kind": "cudaq", "device_name": "cudaq.sample"}


def sample_qaoa_counts(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    thetas: np.ndarray,
    shots: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    policy = select_sampling_backend_policy(int(n_qubits))
    if str(policy["kind"]) == "pennylane":
        counts = sample_counts_pennylane(
            n_qubits=int(n_qubits),
            edges=edges,
            p_layers=int(p_layers),
            thetas=thetas,
            shots=int(shots),
            seed=seed,
            device_name=str(policy["device_name"]),
        )
    else:
        counts = try_cudaq_sample(
            n_qubits=int(n_qubits),
            edges=edges,
            p_layers=int(p_layers),
            thetas=thetas,
            shots=int(shots),
            seed=seed,
        )
    return {
        "policy": dict(policy),
        "counts": counts,
    }


def build_checkpoint_sampling_thetas(
    *,
    mw_checkpoints: Dict[str, Any],
    coeff_checkpoints: Dict[str, Any],
    coeff_enabled: bool,
    coeff_steps: int,
) -> Dict[str, np.ndarray]:
    sample_theta_dict: Dict[str, np.ndarray] = {}
    for step in (1, 10, 25):
        key = f"step_{step}"
        if key in mw_checkpoints:
            sample_theta_dict[f"mw_step_{step}"] = np.asarray(mw_checkpoints[key], dtype=np.float64)

    if coeff_enabled:
        for step in (1, 50, int(coeff_steps)):
            key = f"step_{step}"
            if key in coeff_checkpoints:
                sample_theta_dict[f"coeff_step_{step}"] = np.asarray(coeff_checkpoints[key], dtype=np.float64)

    return sample_theta_dict


def build_stage_comparison_sampling_thetas(
    *,
    coeff_checkpoints: Dict[str, Any],
    exact_optimizer_checkpoints: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    sample_theta_dict: Dict[str, np.ndarray] = {}
    for step in (50, 150):
        coeff_key = f"step_{step}"
        if coeff_key in coeff_checkpoints:
            sample_theta_dict[f"coeff_step_{step}"] = np.asarray(
                coeff_checkpoints[coeff_key],
                dtype=np.float64,
            )

        exact_key = f"step_{step}"
        if exact_key in exact_optimizer_checkpoints:
            sample_theta_dict[f"exact_opt_step_{step}"] = np.asarray(
                exact_optimizer_checkpoints[exact_key],
                dtype=np.float64,
            )
    return sample_theta_dict


def plot_combined_training_overview(
    *,
    combined_curve: Sequence[Dict[str, Any]],
    surrogate_exact_curve_rows: Sequence[Dict[str, Any]],
    exact_optimizer_curve_rows: Sequence[Dict[str, Any]],
    rebuild_log: Sequence[Dict[str, Any]],
    mw_steps: int,
    coeff_steps: int,
    mw_label: str,
    output_path: Path,
    metric: str = "expected_cut",
    brute_force_optimum: Optional[Dict[str, Any]] = None,
) -> None:
    metric_key = "expected_cut" if str(metric) == "expected_cut" else "sum_zz"
    exact_metric_key = f"exact_{metric_key}"
    ylabel = "Expected Cut" if metric_key == "expected_cut" else "Sum <ZZ>"

    mw_rows = [row for row in combined_curve if str(row["stage"]) == "mw"]
    coeff_rows = [row for row in combined_curve if str(row["stage"]) == "coeff"]
    rebuild_global_lines = [
        int(mw_steps) + int(row["after_step"])
        for row in rebuild_log
        if int(row.get("after_step", 0)) > 0
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.8]},
    )
    ax_curve = axes[0]
    ax_rebuild = axes[1]

    if mw_rows:
        ax_curve.plot(
            [row["global_step"] for row in mw_rows],
            [row[metric_key] for row in mw_rows],
            color="#4c78a8",
            linestyle="--",
            linewidth=2.0,
            label=f"MW surrogate ({mw_label})",
        )
    if coeff_rows:
        ax_curve.plot(
            [row["global_step"] for row in coeff_rows],
            [row[metric_key] for row in coeff_rows],
            color="#f58518",
            linestyle="--",
            linewidth=2.0,
            label="Coeff surrogate",
        )
    if surrogate_exact_curve_rows:
        ax_curve.plot(
            [row["global_step"] for row in surrogate_exact_curve_rows],
            [row[exact_metric_key] for row in surrogate_exact_curve_rows],
            color="black",
            linewidth=2.4,
            marker="o",
            markersize=3.0,
            label="Exact on surrogate trajectory",
        )
    if exact_optimizer_curve_rows:
        ax_curve.plot(
            [row["global_step"] for row in exact_optimizer_curve_rows],
            [row[exact_metric_key] for row in exact_optimizer_curve_rows],
            color="#2ca02c",
            linewidth=2.4,
            marker="o",
            markersize=3.0,
            label="Exact optimizer trajectory",
        )
    if brute_force_optimum is not None:
        optimum_y = (
            float(brute_force_optimum["best_expected_cut"])
            if metric_key == "expected_cut"
            else float(brute_force_optimum["best_sum_zz"])
        )
        ax_curve.axhline(
            optimum_y,
            color="gray",
            linestyle="--",
            linewidth=1.8,
            label="Brute-force optimum",
        )

    boundary_x = float(mw_steps) + 0.5
    ax_curve.axvline(boundary_x, color="gray", linestyle=":", linewidth=1.5)
    for x in rebuild_global_lines:
        ax_curve.axvline(float(x), color="gray", linestyle=":", linewidth=1.0, alpha=0.25, zorder=0)
    y_min, y_max = ax_curve.get_ylim()
    y_text = y_max - 0.03 * (y_max - y_min) if y_max > y_min else y_max
    ax_curve.text(
        boundary_x + 1.5,
        y_text,
        "MW -> coeff",
        color="gray",
        va="top",
        fontsize=10,
    )
    ax_curve.set_ylabel(ylabel)
    ax_curve.set_title("Integrated Training Curve With Exact Comparison")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()

    if rebuild_log:
        rebuild_global = [int(mw_steps) + int(row["after_step"]) for row in rebuild_log]
        rebuild_nnz = [int(row["nnz_total"]) for row in rebuild_log]
        rebuild_terms = [int(row["terms_after_zero_filter"]) for row in rebuild_log]

        ax_rebuild.plot(
            rebuild_global,
            rebuild_nnz,
            color="tab:purple",
            marker="o",
            linewidth=2.0,
            label="nnz_total",
        )
        ax_rebuild.set_ylabel("Total nnz", color="tab:purple")
        ax_rebuild.tick_params(axis="y", labelcolor="tab:purple")
        ax_rebuild.grid(True, alpha=0.3)
        ax_rebuild.axvline(boundary_x, color="gray", linestyle=":", linewidth=1.5)
        for x in rebuild_global_lines:
            ax_rebuild.axvline(float(x), color="gray", linestyle=":", linewidth=1.0, alpha=0.25, zorder=0)

        ax_terms = ax_rebuild.twinx()
        ax_terms.plot(
            rebuild_global,
            rebuild_terms,
            color="tab:green",
            marker="s",
            linewidth=2.0,
            label="terms_after_zero_filter",
        )
        ax_terms.set_ylabel("Surviving terms", color="tab:green")
        ax_terms.tick_params(axis="y", labelcolor="tab:green")

        lines = ax_rebuild.get_lines() + ax_terms.get_lines()
        labels = [line.get_label() for line in lines]
        ax_rebuild.legend(lines, labels, loc="upper left")
        ax_rebuild.set_title("Coefficient rebuild diagnostics")
    else:
        ax_rebuild.axis("off")

    total_steps = int(mw_steps) + int(coeff_steps)
    ax_rebuild.set_xlabel(
        f"Global step (MW 1-{int(mw_steps)}, coeff {int(mw_steps) + 1}-{total_steps})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_checkpoint_sampling_histogram(
    *,
    cuts_by_label: Dict[str, np.ndarray],
    n_qubits: int,
    p_layers: int,
    output_path: Path,
) -> None:
    all_cuts = np.concatenate(list(cuts_by_label.values()))
    bins = np.arange(int(np.min(all_cuts)), int(np.max(all_cuts)) + 2) - 0.5
    color_map = {
        "mw_step_1": "#9ecae1",
        "mw_step_10": "#4c78a8",
        "mw_step_25": "#084594",
        "coeff_step_1": "#fdd0a2",
        "coeff_step_50": "#f58518",
        "coeff_step_150": "#7f0000",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, vals in cuts_by_label.items():
        color = color_map.get(label, "gray")
        ax.hist(
            vals,
            bins=bins,
            alpha=0.35,
            label=label.replace("_", " "),
            edgecolor="black",
            density=False,
            color=color,
        )
        ax.axvline(np.mean(vals), linestyle="--", linewidth=1.8, color=color)

    ax.set_xlabel("MaxCut value")
    ax.set_ylabel("Count (shots)")
    ax.set_title(f"Checkpoint Sampling Distribution (N={n_qubits}, p={p_layers})")
    ax.legend(ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stage_sampling_comparison(
    *,
    cuts_by_label: Dict[str, np.ndarray],
    n_qubits: int,
    p_layers: int,
    output_path: Path,
) -> None:
    ordered_labels = [
        "coeff_step_50",
        "coeff_step_150",
        "exact_opt_step_50",
        "exact_opt_step_150",
    ]
    present_labels = [label for label in ordered_labels if label in cuts_by_label]
    if len(present_labels) == 0:
        return

    all_cuts = np.concatenate([np.asarray(cuts_by_label[label]) for label in present_labels])
    bins = np.arange(int(np.min(all_cuts)), int(np.max(all_cuts)) + 2) - 0.5
    color_map = {
        "coeff_step_50": "#f58518",
        "coeff_step_150": "#7f0000",
        "exact_opt_step_50": "#74c476",
        "exact_opt_step_150": "#238b45",
    }
    title_map = {
        "coeff_step_50": "Coeff Surrogate Step 50",
        "coeff_step_150": "Coeff Surrogate Step 150",
        "exact_opt_step_50": "Exact Optimizer Step 50",
        "exact_opt_step_150": "Exact Optimizer Step 150",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    flat_axes = axes.flatten()
    for ax, label in zip(flat_axes, ordered_labels):
        if label not in cuts_by_label:
            ax.axis("off")
            continue
        vals = np.asarray(cuts_by_label[label], dtype=np.int64)
        color = color_map.get(label, "gray")
        ax.hist(
            vals,
            bins=bins,
            alpha=0.75,
            edgecolor="black",
            density=False,
            color=color,
        )
        ax.axvline(np.mean(vals), linestyle="--", linewidth=1.8, color=color)
        ax.set_title(title_map.get(label, label.replace("_", " ")))
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("MaxCut value")
    for ax in axes[:, 0]:
        ax.set_ylabel("Count (shots)")

    fig.suptitle(
        f"Sampling Comparison After MW Pretraining (N={n_qubits}, p={p_layers})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _bits_from_code(code: int, n_qubits: int) -> np.ndarray:
    bits = np.zeros((int(n_qubits),), dtype=np.uint8)
    for q in range(int(n_qubits)):
        bits[q] = (int(code) >> q) & 1
    return bits


def _parse_counts_keys(counts: Dict[str, int], n_qubits: int, bit_order: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k, v in counts.items():
        ks = str(k).strip()
        if all(ch in ("0", "1") for ch in ks) and len(ks) == int(n_qubits):
            code = 0
            if str(bit_order) == "le":
                for i, ch in enumerate(ks):
                    code |= (int(ch) & 1) << i
            else:
                for i, ch in enumerate(ks):
                    code |= (int(ch) & 1) << (int(n_qubits) - 1 - i)
        elif ks.isdigit():
            code = int(ks)
        else:
            raise ValueError(f"invalid bitstring key: {k}")
        out[int(code)] = int(v)
    return out


def cut_value_from_bits(bits01: np.ndarray, edges: Sequence[Tuple[int, int]]) -> int:
    val = 0
    for u, v in edges:
        val += int(int(bits01[int(u)]) != int(bits01[int(v)]))
    return int(val)


def get_cut_values(counts_dict: Dict[str, int], n_qubits: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    code_counts = _parse_counts_keys(counts_dict, n_qubits, "le")
    vals = []
    for code, cnt in code_counts.items():
        bits = _bits_from_code(code, n_qubits)
        cut = cut_value_from_bits(bits, edges)
        vals.extend([cut] * cnt)
    return np.asarray(vals, dtype=np.int64)


def try_cudaq_sample(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    thetas: np.ndarray,
    shots: int,
    seed: Optional[int],
) -> Dict[str, int]:
    try:
        import cudaq
    except Exception as exc:
        raise RuntimeError(f"CUDA-Q is not available: {exc}") from exc

    theta_np = np.asarray(thetas, dtype=np.float64).reshape(-1)
    params_list = [float(x) for x in theta_np.tolist()]

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

    kernel.mz(q)

    if seed is not None:
        cudaq.set_random_seed(int(seed))

    counts = cudaq.sample(kernel, params_list, shots_count=int(shots))
    return {str(k): int(v) for k, v in counts.items()}


def plot_sampling_histogram(
    *,
    cuts_by_label: Dict[str, np.ndarray],
    n_qubits: int,
    p_layers: int,
    output_path: Path,
) -> None:
    all_cuts = np.concatenate(list(cuts_by_label.values()))
    bins = np.arange(int(np.min(all_cuts)), int(np.max(all_cuts)) + 2) - 0.5
    colors = ["gray", "#5f9ea0", "#20b2aa", "#ff6347", "#9370db", "#ffb347"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for color, (label, vals) in zip(colors, cuts_by_label.items()):
        ax.hist(vals, bins=bins, alpha=0.55, label=label, edgecolor="black", density=False, color=color)
        ax.axvline(np.mean(vals), linestyle="--", linewidth=2, color=color)

    ax.set_xlabel("MaxCut value")
    ax.set_ylabel("Count (shots)")
    ax.set_title(f"Sampling Distribution Shift (N={n_qubits}, p={p_layers})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    raw_config = load_config(config_path)

    directory_cfg = dict(raw_config.get("DIRECTORY", {}))
    graph_cfg = dict(raw_config.get("GRAPH", {}))
    qaoa_cfg = dict(raw_config.get("QAOA", {}))
    mw_cfg = dict(raw_config.get("MW_TRAINING", {}))
    coeff_cfg = dict(raw_config.get("COEFF_FINETUNING", {}))
    analysis_cfg = dict(raw_config.get("ANALYSIS", {}))
    runtime_cfg = dict(raw_config.get("RUNTIME", {}))
    checkpoint_cfg = dict(raw_config.get("CHECKPOINTS", {}))
    sampling_cfg = dict(raw_config.get("SAMPLING", {}))

    n_qubits = int(graph_cfg["n_qubits"])
    p_layers = int(qaoa_cfg["n_layers"])
    edge_prob = float(graph_cfg.get("edge_prob", 0.7))
    seed = int(graph_cfg.get("seed", 42))

    delta_t = float(qaoa_cfg.get("delta_t", 0.8))
    init_strategy_config = str(qaoa_cfg.get("init_strategy", "flattened_tqa"))
    init_strategy_raw = str(args.init_strategy).strip() or init_strategy_config
    init_strategy = normalize_init_strategy(init_strategy_raw)
    init_strategy_tag = init_strategy_output_tag(init_strategy_raw)
    flatten_alpha = float(qaoa_cfg.get("flatten_alpha", 0.5))

    mw_max_weight = int(mw_cfg["max_weight"])
    mw_steps = int(mw_cfg.get("steps", 150))
    mw_lr = float(mw_cfg.get("lr", 0.05))

    coeff_enabled = bool(coeff_cfg.get("enabled", True))
    coeff_build_min_abs = float(coeff_cfg.get("build_min_abs", 1e-3))
    coeff_rebuild_interval = int(coeff_cfg.get("rebuild_interval", 15))
    coeff_steps = int(coeff_cfg.get("steps", 150))
    coeff_lr = float(coeff_cfg.get("lr", 0.02))
    coeff_max_weight_override = coeff_cfg.get("max_weight_override", None)
    coeff_max_weight_override = (
        None if coeff_max_weight_override is None else int(coeff_max_weight_override)
    )
    exact_enabled = bool(analysis_cfg.get("exact_enabled", True))
    exact_large_step_stride = int(analysis_cfg.get("exact_large_step_stride", 1))
    exact_optimizer_enabled = bool(analysis_cfg.get("exact_optimizer_enabled", True))
    exact_plot_metric = str(analysis_cfg.get("plot_metric", "expected_cut"))

    raw_device = str(runtime_cfg.get("device", "auto"))
    device = choose_device(raw_device)
    chunk_size = int(runtime_cfg.get("chunk_size", 20_000_000))
    parallel_compile = bool(runtime_cfg.get("parallel_compile", False))
    log_every = int(runtime_cfg.get("log_every", 10))

    mw_checkpoint_steps = [int(x) for x in checkpoint_cfg.get("mw_steps", [1, 10, 25])]
    coeff_checkpoint_steps = [int(x) for x in checkpoint_cfg.get("coeff_steps", [1, 50, coeff_steps])]

    sampling_enabled = bool(sampling_cfg.get("enabled", False))
    sampling_shots = int(sampling_cfg.get("shots", 4000))
    sampling_seed = int(sampling_cfg.get("seed", seed))

    graph_dir = Path(directory_cfg.get("graph_dir", _THIS_DIR / "graph")).expanduser().resolve()
    output_root_base = Path(directory_cfg.get("output_root", DEFAULT_OUTPUT_ROOT)).expanduser().resolve()
    output_root = output_root_base / init_strategy_tag
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = str(
        directory_cfg.get(
            "run_name",
            f"Q{n_qubits}_L{p_layers}_mw{mw_max_weight}_bma{float_tag(coeff_build_min_abs)}",
        )
    ).strip()
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] config: {config_path}")
    print(f"[info] output_dir: {run_dir}")
    print(f"[info] device: {device}")

    normalized_config = {
        "source_config_path": str(config_path),
        "run_name": str(run_name),
        "DIRECTORY": {
            "output_root_base": str(output_root_base),
            "output_root": str(output_root),
            "graph_dir": str(graph_dir),
        },
        "GRAPH": {
            "source_mode": "existing_graph_file",
            "n_qubits": int(n_qubits),
            "edge_prob": float(edge_prob),
            "seed": int(seed),
        },
        "QAOA": {
            "n_layers": int(p_layers),
            "delta_t": float(delta_t),
            "init_strategy_config": str(init_strategy_config),
            "init_strategy_runtime": str(init_strategy),
            "init_strategy_output_tag": str(init_strategy_tag),
            "flatten_alpha": float(flatten_alpha),
        },
        "MW_TRAINING": {
            "max_weight": int(mw_max_weight),
            "steps": int(mw_steps),
            "lr": float(mw_lr),
        },
        "COEFF_FINETUNING": {
            "enabled": bool(coeff_enabled),
            "build_min_abs": float(coeff_build_min_abs),
            "rebuild_interval": int(coeff_rebuild_interval),
            "steps": int(coeff_steps),
            "lr": float(coeff_lr),
            "max_weight_override": coeff_max_weight_override,
        },
        "ANALYSIS": {
            "exact_enabled": bool(exact_enabled),
            "exact_large_step_stride": int(exact_large_step_stride),
            "exact_optimizer_enabled": bool(exact_optimizer_enabled),
            "plot_metric": str(exact_plot_metric),
        },
        "RUNTIME": {
            "device_requested": str(raw_device),
            "device_runtime": str(device),
            "chunk_size": int(chunk_size),
            "parallel_compile": bool(parallel_compile),
            "log_every": int(log_every),
        },
        "CHECKPOINTS": {
            "mw_steps": list(mw_checkpoint_steps),
            "coeff_steps": list(coeff_checkpoint_steps),
        },
        "SAMPLING": {
            "enabled": bool(sampling_enabled),
            "shots": int(sampling_shots),
            "seed": int(sampling_seed),
        },
    }
    edges, source_graph_json, source_graph_png = load_existing_graph(
        graph_dir=graph_dir,
        n_qubits=n_qubits,
    )
    n_edges = len(edges)
    print(f"[info] graph_source_json: {source_graph_json}")
    brute_force_optimum = load_or_compute_bruteforce_optimum(
        graph_dir=graph_dir,
        source_graph_json=source_graph_json,
        n_qubits=n_qubits,
        n_edges=n_edges,
        edges=edges,
        max_qubits=20,
    )
    if brute_force_optimum is not None:
        print(
            "[info] brute-force optimum "
            f"E[cut]={float(brute_force_optimum['best_expected_cut']):.6f}"
        )

    circuit, _ = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=p_layers)
    zz_obj = build_maxcut_observable(n_qubits=n_qubits, edges=edges)
    init_theta_np = build_initial_theta_np(
        init_strategy=init_strategy,
        n_layers=p_layers,
        n_edges=n_edges,
        n_qubits=n_qubits,
        delta_t=delta_t,
        flatten_alpha=flatten_alpha,
        seed=seed,
    )

    mw_compile_fn = make_stage_compile_fn(
        stage_name="mw_train",
        circuit=circuit,
        zz_obj=zz_obj,
        device=device,
        chunk_size=chunk_size,
        parallel_compile=parallel_compile,
        max_weight_override=mw_max_weight,
        n_qubits=n_qubits,
    )
    coeff_compile_fn = make_stage_compile_fn(
        stage_name="coeff_finetune",
        circuit=circuit,
        zz_obj=zz_obj,
        device=device,
        chunk_size=chunk_size,
        parallel_compile=parallel_compile,
        max_weight_override=coeff_max_weight_override,
        n_qubits=n_qubits,
    )

    thetas_mw = torch.nn.Parameter(torch.tensor(init_theta_np, dtype=torch.float64, device=device))
    program_mw, mw_compile_info = mw_compile_fn(build_thetas=None, build_min_abs=None)
    print("[mw_train] compile info:")
    print(json.dumps(mw_compile_info, indent=2))

    mw_history, mw_checkpoints, trained_thetas_mw_np, mw_theta_trajectory = train_with_fixed_program(
        program=program_mw,
        thetas=thetas_mw,
        n_edges=n_edges,
        steps=mw_steps,
        lr=mw_lr,
        checkpoint_steps=mw_checkpoint_steps,
        stage_name="mw-train",
        log_every=log_every,
    )
    del program_mw
    cleanup_memory(device)

    coeff_history: List[Dict[str, Any]] = []
    coeff_checkpoints: Dict[str, Any] = {}
    coeff_final_thetas_np: Optional[np.ndarray] = None
    coeff_rebuild_log: List[Dict[str, Any]] = []
    coeff_theta_trajectory: List[Dict[str, Any]] = []
    if coeff_enabled:
        (
            coeff_history,
            coeff_checkpoints,
            coeff_final_thetas_np,
            coeff_rebuild_log,
            coeff_theta_trajectory,
        ) = train_with_periodic_rebuild(
            compile_program_fn=coeff_compile_fn,
            start_thetas_np=trained_thetas_mw_np,
            device=device,
            n_edges=n_edges,
            steps=coeff_steps,
            lr=coeff_lr,
            checkpoint_steps=coeff_checkpoint_steps,
            build_min_abs=coeff_build_min_abs,
            rebuild_interval=coeff_rebuild_interval,
            stage_name="coeff-finetune",
            log_every=log_every,
        )

    exact_optimizer_policy = select_exact_optimizer_policy(n_qubits=int(n_qubits))
    exact_optimizer_history: List[Dict[str, Any]] = []
    exact_optimizer_checkpoints: Dict[str, Any] = {}
    exact_optimizer_theta_trajectory: List[Dict[str, Any]] = []
    exact_optimizer_final_thetas_np: Optional[np.ndarray] = None
    exact_optimizer_runtime: Dict[str, Any] = dict(exact_optimizer_policy)
    if bool(exact_enabled) and bool(exact_optimizer_enabled) and bool(coeff_enabled) and bool(
        exact_optimizer_policy.get("enabled", False)
    ):
        (
            exact_optimizer_history,
            exact_optimizer_checkpoints,
            exact_optimizer_final_thetas_np,
            exact_optimizer_theta_trajectory,
            exact_optimizer_runtime,
        ) = train_with_exact_optimizer_pennylane(
            n_qubits=n_qubits,
            edges=edges,
            p_layers=p_layers,
            start_thetas_np=trained_thetas_mw_np,
            n_edges=n_edges,
            steps=coeff_steps,
            lr=coeff_lr,
            checkpoint_steps=coeff_checkpoint_steps,
            device_name=str(exact_optimizer_policy["device_name"]),
            log_every=log_every,
        )
    combined_curve = build_combined_surrogate_curve(
        mw_history=mw_history,
        coeff_history=coeff_history,
        mw_steps=mw_steps,
    )
    combined_theta_trajectory = build_combined_theta_trajectory(
        mw_theta_trajectory=mw_theta_trajectory,
        coeff_theta_trajectory=coeff_theta_trajectory,
        mw_steps=mw_steps,
    )
    surrogate_exact_curve_payload: Optional[Dict[str, Any]] = None
    exact_optimizer_curve_payload: Optional[Dict[str, Any]] = None
    if exact_enabled:
        print("[exact] Evaluating exact curve from saved theta trajectory...")
        surrogate_exact_curve_payload = evaluate_exact_curve(
            n_qubits=n_qubits,
            edges=edges,
            p_layers=p_layers,
            combined_curve=combined_curve,
            combined_theta_trajectory=combined_theta_trajectory,
            large_step_stride=exact_large_step_stride,
        )
        if exact_optimizer_history:
            exact_optimizer_curve_payload = build_exact_optimizer_curve(
                exact_history=exact_optimizer_history,
                mw_steps=mw_steps,
                policy=exact_optimizer_runtime,
            )
        plot_combined_training_overview(
            combined_curve=combined_curve,
            surrogate_exact_curve_rows=surrogate_exact_curve_payload["rows"],
            exact_optimizer_curve_rows=(
                [] if exact_optimizer_curve_payload is None else exact_optimizer_curve_payload["rows"]
            ),
            rebuild_log=coeff_rebuild_log,
            mw_steps=mw_steps,
            coeff_steps=coeff_steps,
            mw_label=f"max_weight={mw_max_weight}",
            output_path=run_dir / "integrated_training_exact_curve.png",
            metric=exact_plot_metric,
            brute_force_optimum=brute_force_optimum,
        )
    else:
        plot_combined_training_overview(
            combined_curve=combined_curve,
            surrogate_exact_curve_rows=[],
            exact_optimizer_curve_rows=[],
            rebuild_log=coeff_rebuild_log,
            mw_steps=mw_steps,
            coeff_steps=coeff_steps,
            mw_label=f"max_weight={mw_max_weight}",
            output_path=run_dir / "integrated_training_exact_curve.png",
            metric=exact_plot_metric,
            brute_force_optimum=brute_force_optimum,
        )

    sampling_summary: Optional[Dict[str, Any]] = None
    sampled_counts: Dict[str, Dict[str, int]] = {}
    sampling_backend_by_label: Dict[str, Dict[str, Any]] = {}
    sampling_cut_values_by_label: Dict[str, List[int]] = {}
    stage_comparison_summary: Optional[Dict[str, Any]] = None
    stage_comparison_counts: Dict[str, Dict[str, int]] = {}
    stage_comparison_backend_by_label: Dict[str, Dict[str, Any]] = {}
    stage_comparison_cut_values_by_label: Dict[str, List[int]] = {}
    if sampling_enabled:
        sample_theta_dict = build_checkpoint_sampling_thetas(
            mw_checkpoints=mw_checkpoints,
            coeff_checkpoints=coeff_checkpoints,
            coeff_enabled=bool(coeff_enabled),
            coeff_steps=int(coeff_steps),
        )

        for label, theta_np in sample_theta_dict.items():
            print(f"[sampling] {label}")
            sample_result = sample_qaoa_counts(
                n_qubits=n_qubits,
                edges=edges,
                p_layers=p_layers,
                thetas=theta_np,
                shots=sampling_shots,
                seed=sampling_seed,
            )
            sampled_counts[label] = dict(sample_result["counts"])
            sampling_backend_by_label[label] = dict(sample_result["policy"])

        cuts_by_label = {
            label: get_cut_values(counts, n_qubits, edges)
            for label, counts in sampled_counts.items()
        }
        sampling_cut_values_by_label = {
            label: vals.astype(int).tolist()
            for label, vals in cuts_by_label.items()
        }
        sampling_summary = {}
        for label, vals in cuts_by_label.items():
            sampling_summary[label] = {
                "backend": sampling_backend_by_label[label],
                "mean_cut": float(np.mean(vals)),
                "max_cut": int(np.max(vals)),
                "min_cut": int(np.min(vals)),
            }
        plot_checkpoint_sampling_histogram(
            cuts_by_label=cuts_by_label,
            n_qubits=n_qubits,
            p_layers=p_layers,
            output_path=run_dir / "sampling_histogram.png",
        )

        if exact_optimizer_checkpoints:
            comparison_theta_dict = build_stage_comparison_sampling_thetas(
                coeff_checkpoints=coeff_checkpoints,
                exact_optimizer_checkpoints=exact_optimizer_checkpoints,
            )
            for label, theta_np in comparison_theta_dict.items():
                if label.startswith("coeff_") and label in sampled_counts:
                    stage_comparison_counts[label] = dict(sampled_counts[label])
                    stage_comparison_backend_by_label[label] = dict(sampling_backend_by_label[label])
                    continue

                print(f"[sampling-compare] {label}")
                sample_result = sample_qaoa_counts(
                    n_qubits=n_qubits,
                    edges=edges,
                    p_layers=p_layers,
                    thetas=theta_np,
                    shots=sampling_shots,
                    seed=sampling_seed,
                )
                stage_comparison_counts[label] = dict(sample_result["counts"])
                stage_comparison_backend_by_label[label] = dict(sample_result["policy"])

            comparison_cuts_by_label = {
                label: get_cut_values(counts, n_qubits, edges)
                for label, counts in stage_comparison_counts.items()
            }
            stage_comparison_cut_values_by_label = {
                label: vals.astype(int).tolist()
                for label, vals in comparison_cuts_by_label.items()
            }
            stage_comparison_summary = {}
            for label, vals in comparison_cuts_by_label.items():
                stage_comparison_summary[label] = {
                    "backend": stage_comparison_backend_by_label[label],
                    "mean_cut": float(np.mean(vals)),
                    "max_cut": int(np.max(vals)),
                    "min_cut": int(np.min(vals)),
                }
            if len(comparison_cuts_by_label) > 0:
                plot_stage_sampling_comparison(
                    cuts_by_label=comparison_cuts_by_label,
                    n_qubits=n_qubits,
                    p_layers=p_layers,
                    output_path=run_dir / "sampling_stage_comparison.png",
                )

    artifacts = {
        "schema_version": 2,
        "config": normalized_config,
        "graph": {
            "source_json": str(source_graph_json),
            "source_png": None if source_graph_png is None else str(source_graph_png),
            "n_edges": int(n_edges),
            "edges": [[int(u), int(v)] for (u, v) in edges],
            "brute_force_optimum": brute_force_optimum,
        },
        "training": {
            "mw": {
                "compile_info": mw_compile_info,
                "start_thetas": init_theta_np.tolist(),
                "final_thetas": trained_thetas_mw_np.tolist(),
                "theta_by_step": theta_trajectory_to_step_map(mw_theta_trajectory),
                "checkpoint_thetas_by_step": checkpoint_thetas_to_step_map(
                    mw_checkpoints,
                    drop_step_zero=True,
                ),
                "history": mw_history,
            },
            "coeff": {
                "enabled": bool(coeff_enabled),
                "build_min_abs": float(coeff_build_min_abs),
                "rebuild_interval": int(coeff_rebuild_interval),
                "start_thetas": trained_thetas_mw_np.tolist(),
                "final_thetas": (
                    None if coeff_final_thetas_np is None else coeff_final_thetas_np.tolist()
                ),
                "theta_by_step": theta_trajectory_to_step_map(coeff_theta_trajectory),
                "checkpoint_thetas_by_step": checkpoint_thetas_to_step_map(
                    coeff_checkpoints,
                    drop_step_zero=True,
                ),
                "history": coeff_history,
                "rebuild_log": coeff_rebuild_log,
            },
            "exact_optimizer": {
                "enabled": bool(
                    bool(exact_enabled)
                    and bool(exact_optimizer_enabled)
                    and bool(coeff_enabled)
                    and bool(exact_optimizer_policy.get("enabled", False))
                ),
                "runtime": exact_optimizer_runtime,
                "start_thetas": trained_thetas_mw_np.tolist(),
                "final_thetas": (
                    None
                    if exact_optimizer_final_thetas_np is None
                    else exact_optimizer_final_thetas_np.tolist()
                ),
                "theta_by_step": theta_trajectory_to_step_map(exact_optimizer_theta_trajectory),
                "checkpoint_thetas_by_step": checkpoint_thetas_to_step_map(
                    exact_optimizer_checkpoints,
                    drop_step_zero=True,
                ),
                "history": exact_optimizer_history,
            },
        },
        "integrated_curve": {
            "plot_metric": str(exact_plot_metric),
            "surrogate_curve": combined_curve,
            "surrogate_exact_curve": surrogate_exact_curve_payload,
            "exact_optimizer_curve": exact_optimizer_curve_payload,
        },
        "sampling": {
            "enabled": bool(sampling_enabled),
            "shots": int(sampling_shots),
            "seed": int(sampling_seed),
            "backend_by_label": sampling_backend_by_label,
            "counts": sampled_counts,
            "cut_values_by_label": sampling_cut_values_by_label,
            "summary": sampling_summary,
            "stage_comparison": {
                "backend_by_label": stage_comparison_backend_by_label,
                "counts": stage_comparison_counts,
                "cut_values_by_label": stage_comparison_cut_values_by_label,
                "summary": stage_comparison_summary,
            },
        },
    }
    save_json(run_dir / "artifacts.json", artifacts)

    keep_outputs = ["artifacts.json", "integrated_training_exact_curve.png"]
    if sampling_enabled:
        keep_outputs.append("sampling_histogram.png")
        if len(stage_comparison_counts) > 0:
            keep_outputs.append("sampling_stage_comparison.png")
    cleanup_managed_outputs(run_dir, keep_outputs)

    print("[done] saved artifacts:")
    print(json.dumps({"kept_outputs": keep_outputs}, indent=2))


if __name__ == "__main__":
    main()
