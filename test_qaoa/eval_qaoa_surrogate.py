from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_tensor.api import compile_expval_program, pennylane_sample_small

from qaoa_surrogate_common import (
    brute_force_maxcut,
    build_maxcut_observable,
    build_qaoa_circuit,
    build_qaoa_theta_init_tqa,
    cut_value_from_bits,
    expected_cut_from_sum_zz,
    parse_min_abs_schedule,
)


def _choose_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(raw)


def _cpu_exact_overrides() -> Dict[str, object]:
    return {
        "build_device": "cpu",
        "step_device": "cpu",
        "stream_device": "cpu",
        "dtype": "float64",
        "max_weight": 1_000_000_000,
        "max_xy": 1_000_000_000,
        "offload_steps": False,
        "offload_back": False,
    }


def _compile_program_for_eval(
    *,
    n_qubits: int,
    edges: List[List[int]],
    p_layers: int,
    best_thetas: torch.Tensor,
    run_device: str,
    build_min_abs: float,
    build_min_mat_abs: Optional[float],
) -> Any:
    edge_pairs = [(int(e[0]), int(e[1])) for e in edges]
    circuit, _ = build_qaoa_circuit(n_qubits=n_qubits, edges=edge_pairs, p_layers=p_layers)
    zz_obj = build_maxcut_observable(n_qubits=n_qubits, edges=edge_pairs)

    preset = "gpu_full" if run_device.startswith("cuda") else "gpu_min"
    preset_overrides = None
    if run_device == "cpu" and preset == "gpu_min":
        preset_overrides = _cpu_exact_overrides()

    program = compile_expval_program(
        circuit=circuit,
        observables=[zz_obj],
        preset=preset,
        preset_overrides=preset_overrides,
        build_thetas=best_thetas.to(run_device),
        build_min_abs=float(build_min_abs),
        build_min_mat_abs=build_min_mat_abs,
    )
    return program, circuit, edge_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained MaxCut-QAOA surrogate checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--sampling-max-qubits", type=int, default=20)
    p.add_argument("--exact-max-qubits", type=int, default=20)
    p.add_argument("--min-abs-schedule", type=str, default="1e-2,1e-3,1e-4")
    p.add_argument("--build-min-mat-abs", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-anneal", action="store_true", help="Disable classical simulated-annealing baseline.")
    p.add_argument("--anneal-steps", type=int, default=50_000, help="SA proposals per restart.")
    p.add_argument("--anneal-restarts", type=int, default=8, help="SA restart count.")
    p.add_argument("--anneal-t0", type=float, default=2.5, help="SA initial temperature.")
    p.add_argument("--anneal-tf", type=float, default=1e-3, help="SA final temperature.")
    p.add_argument(
        "--anneal-seed",
        type=int,
        default=None,
        help="Optional SA RNG seed (defaults to --seed).",
    )
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--plot-dir", type=str, default="test/artifacts/plots")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _surrogate_expected_cut(program: Any, thetas_cpu: torch.Tensor, run_device: str, m_edges: int) -> float:
    sum_zz = float(
        program.expval(
            thetas_cpu.to(run_device),
            obs_index=0,
            stream_device=run_device,
            offload_back=bool(run_device.startswith("cuda")),
        )
        .detach()
        .cpu()
        .item()
    )
    return expected_cut_from_sum_zz(sum_zz, m_edges)


def _exact_expected_cut(program: Any, thetas_cpu: torch.Tensor, max_qubits: int, m_edges: int) -> float:
    sum_zz = float(program.expvals_pennylane(thetas_cpu, max_qubits=max_qubits)[0].item())
    return expected_cut_from_sum_zz(sum_zz, m_edges)


def _sample_cut_stats(
    *,
    circuit,
    thetas_cpu: torch.Tensor,
    n_qubits: int,
    edges,
    shots: int,
    max_qubits: int,
    seed: int,
    best_codes: set[int],
) -> Dict[str, Any]:
    samples = pennylane_sample_small(
        circuit=circuit,
        thetas=thetas_cpu,
        n_qubits=n_qubits,
        shots=int(shots),
        max_qubits=max_qubits,
        seed=int(seed),
    )
    samples_np = samples.detach().cpu().numpy().astype(np.uint8)
    cuts = np.asarray([cut_value_from_bits(row.tolist(), edges) for row in samples_np], dtype=np.int64)

    sampled_codes = []
    for row in samples_np:
        code = 0
        for q in range(n_qubits):
            code |= (int(row[q]) & 1) << q
        sampled_codes.append(code)
    sampled_codes_np = np.asarray(sampled_codes, dtype=np.int64)
    frac_optimal = float(np.mean(np.isin(sampled_codes_np, np.asarray(list(best_codes), dtype=np.int64))))
    return {
        "cuts": cuts,
        "sampled_mean_cut": float(np.mean(cuts)),
        "sampled_best_cut": int(np.max(cuts)),
        "sampled_fraction_optimal": frac_optimal,
    }


def _sweep_expected_cut(
    *,
    n_qubits: int,
    edges_json: List[List[int]],
    p_layers: int,
    thetas_cpu: torch.Tensor,
    m_edges: int,
    run_device: str,
    min_abs_values: List[float],
    build_min_mat_abs: Optional[float],
) -> List[Dict[str, float]]:
    rows = []
    for min_abs in min_abs_values:
        program, _, _edges = _compile_program_for_eval(
            n_qubits=n_qubits,
            edges=edges_json,
            p_layers=p_layers,
            best_thetas=thetas_cpu,
            run_device=run_device,
            build_min_abs=float(min_abs),
            build_min_mat_abs=build_min_mat_abs,
        )
        sum_zz = float(
            program.expval(
                thetas_cpu.to(run_device),
                obs_index=0,
                stream_device=run_device,
                offload_back=bool(run_device.startswith("cuda")),
            )
            .detach()
            .cpu()
            .item()
        )
        rows.append(
            {
                "min_abs": float(min_abs),
                "sum_zz": sum_zz,
                "expected_cut": expected_cut_from_sum_zz(sum_zz, m_edges),
            }
        )
    return rows


def _mask_to_leftmost_q0_bitstring(mask: int, n_qubits: int) -> str:
    return "".join("1" if ((int(mask) >> q) & 1) else "0" for q in range(int(n_qubits)))


def _cut_value_from_mask(mask: int, edges) -> float:
    val = 0.0
    for u, v in edges:
        if ((int(mask) >> int(u)) & 1) != ((int(mask) >> int(v)) & 1):
            val += 1.0
    return float(val)


def _simulated_annealing_maxcut(
    *,
    n_qubits: int,
    edges,
    steps: int,
    restarts: int,
    t0: float,
    tf: float,
    seed: int,
) -> Dict[str, Any]:
    n = int(n_qubits)
    if n <= 0:
        raise ValueError("n_qubits must be positive for simulated annealing")
    if int(steps) < 2:
        raise ValueError("anneal_steps must be >= 2")
    if int(restarts) < 1:
        raise ValueError("anneal_restarts must be >= 1")
    if float(t0) <= 0.0 or float(tf) <= 0.0:
        raise ValueError("anneal temperatures must be positive")

    # Build adjacency for O(degree) delta updates.
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        uu, vv = int(u), int(v)
        adj[uu].append(vv)
        adj[vv].append(uu)

    rng = np.random.default_rng(int(seed))
    alpha = (float(tf) / float(t0)) ** (1.0 / float(int(steps) - 1))

    best_mask = 0
    best_cut = -float("inf")

    for _ in range(int(restarts)):
        # Works for arbitrary n_qubits without relying on fixed-width integer RNG bounds.
        bits = rng.integers(0, 2, size=n, dtype=np.int8)
        mask = 0
        for q, b in enumerate(bits.tolist()):
            mask |= (int(b) & 1) << int(q)

        cut = _cut_value_from_mask(mask, edges)
        local_best_cut = cut
        local_best_mask = mask
        T = float(t0)

        for _ in range(int(steps)):
            u = int(rng.integers(0, n))
            bu = (mask >> u) & 1

            delta = 0.0
            for v in adj[u]:
                bv = (mask >> int(v)) & 1
                old_diff = bu ^ bv
                delta += 1.0 - 2.0 * float(old_diff)

            accept = False
            if delta >= 0.0:
                accept = True
            elif T > 0.0 and rng.random() < math.exp(delta / T):
                accept = True

            if accept:
                mask ^= 1 << u
                cut += delta
                if cut > local_best_cut:
                    local_best_cut = cut
                    local_best_mask = mask

            T *= alpha

        if local_best_cut > best_cut:
            best_cut = local_best_cut
            best_mask = local_best_mask

    return {
        "best_cut": float(best_cut),
        "best_mask": int(best_mask),
        "best_bitstring_leftmost_q0": _mask_to_leftmost_q0_bitstring(int(best_mask), n),
        "steps": int(steps),
        "restarts": int(restarts),
        "t0": float(t0),
        "tf": float(tf),
        "seed": int(seed),
    }


def _plot_min_abs_sweep(
    out_path: Path,
    trained_rows: List[Dict[str, float]],
    init_rows: List[Dict[str, float]],
    exact_trained: Optional[float],
    exact_init: Optional[float],
    anneal_best_cut: Optional[float],
) -> None:
    x = np.asarray([float(r["min_abs"]) for r in trained_rows], dtype=np.float64)
    y_tr = np.asarray([float(r["expected_cut"]) for r in trained_rows], dtype=np.float64)
    y_in = np.asarray([float(r["expected_cut"]) for r in init_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, y_tr, marker="o", label="trained (surrogate)")
    ax.plot(x, y_in, marker="s", label="initial (surrogate)")
    if exact_trained is not None:
        ax.axhline(float(exact_trained), linestyle="--", linewidth=1.5, label="trained exact (PennyLane)")
    if exact_init is not None:
        ax.axhline(float(exact_init), linestyle=":", linewidth=1.5, label="initial exact (PennyLane)")
    if anneal_best_cut is not None:
        ax.axhline(
            float(anneal_best_cut),
            linestyle="-.",
            linewidth=1.5,
            color="#d62728",
            label="classical annealing baseline",
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("build_min_abs (larger -> more truncation)")
    ax.set_ylabel("Expected cut")
    ax.set_title("Approximation Tightening vs Expected Cut")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_expected_cut_bars(
    out_path: Path,
    init_sur: float,
    trained_sur: float,
    init_exact: Optional[float],
    trained_exact: Optional[float],
    anneal_best_cut: Optional[float],
) -> None:
    labels = ["init_sur", "trained_sur"]
    vals = [float(init_sur), float(trained_sur)]
    colors = ["#7aa6c2", "#2f7e79"]
    if init_exact is not None and trained_exact is not None:
        labels += ["init_exact", "trained_exact"]
        vals += [float(init_exact), float(trained_exact)]
        colors += ["#9fa8ad", "#4f5d75"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, vals, color=colors)
    if anneal_best_cut is not None:
        ax.axhline(
            float(anneal_best_cut),
            linestyle="-.",
            linewidth=1.5,
            color="#d62728",
            label="classical annealing baseline",
        )
    ax.set_ylabel("Expected cut")
    ax.set_title("Initial vs Trained Expected Cut")
    ax.grid(True, axis="y", alpha=0.3)
    if anneal_best_cut is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sampling_hist_compare(out_path: Path, cuts_init: np.ndarray, cuts_trained: np.ndarray) -> None:
    cmin = int(min(np.min(cuts_init), np.min(cuts_trained)))
    cmax = int(max(np.max(cuts_init), np.max(cuts_trained)))
    bins = np.arange(cmin, cmax + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.hist(cuts_init, bins=bins, alpha=0.55, label="initial", color="#8da0cb")
    ax.hist(cuts_trained, bins=bins, alpha=0.55, label="trained", color="#66c2a5")
    ax.set_xlabel("Cut value")
    ax.set_ylabel("Count")
    ax.set_title("Sampling Distribution: Initial vs Trained")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_training_curve(
    out_path: Path,
    history_sum_zz: List[float],
    m_edges: int,
    anneal_best_cut: Optional[float],
) -> None:
    if len(history_sum_zz) == 0:
        return
    steps = np.arange(len(history_sum_zz), dtype=np.int64)
    exp_cut = np.asarray([expected_cut_from_sum_zz(v, m_edges) for v in history_sum_zz], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, exp_cut, linewidth=1.8)
    if anneal_best_cut is not None:
        ax.axhline(
            float(anneal_best_cut),
            linestyle="-.",
            linewidth=1.5,
            color="#d62728",
            label="classical annealing baseline",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Expected cut (from surrogate)")
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)
    if anneal_best_cut is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    data = torch.load(ckpt_path, map_location="cpu")

    cfg = dict(data["config"])
    n_qubits = int(cfg["n_qubits"])
    p_layers = int(cfg["p_layers"])
    edges_json = list(data["edges"])
    m_edges = int(data["m_edges"])
    best_thetas_cpu = data["best_thetas"].detach().cpu().to(torch.float64)
    history_sum_zz = [float(v) for v in data.get("history_sum_zz", [])]
    run_device = _choose_device(str(args.device))
    plot_dir = Path(args.plot_dir)

    if run_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    init_theta_np = build_qaoa_theta_init_tqa(
        p_layers=p_layers,
        n_edges=m_edges,
        n_qubits=n_qubits,
        delta_t=float(cfg["delta_t"]),
        dtype=np.float64,
    )
    init_thetas_cpu = torch.as_tensor(init_theta_np, dtype=torch.float64, device="cpu")
    if int(init_thetas_cpu.numel()) != int(data["n_params"]):
        raise RuntimeError("Initial theta size does not match stored n_params in checkpoint.")

    min_abs_values = parse_min_abs_schedule(args.min_abs_schedule)
    min_abs_values = sorted(min_abs_values, reverse=True)

    program, circuit, edges = _compile_program_for_eval(
        n_qubits=n_qubits,
        edges=edges_json,
        p_layers=p_layers,
        best_thetas=best_thetas_cpu,
        run_device=run_device,
        build_min_abs=float(min_abs_values[-1]),
        build_min_mat_abs=args.build_min_mat_abs,
    )

    result: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "n_qubits": n_qubits,
        "p_layers": p_layers,
        "m_edges": m_edges,
        "device": run_device,
    }

    anneal_best_cut: Optional[float] = None
    if not bool(args.no_anneal):
        anneal_seed = int(args.seed) if args.anneal_seed is None else int(args.anneal_seed)
        anneal_info = _simulated_annealing_maxcut(
            n_qubits=n_qubits,
            edges=edges,
            steps=int(args.anneal_steps),
            restarts=int(args.anneal_restarts),
            t0=float(args.anneal_t0),
            tf=float(args.anneal_tf),
            seed=anneal_seed,
        )
        anneal_best_cut = float(anneal_info["best_cut"])
        result["classical_annealing"] = anneal_info
        print(
            "[anneal] "
            f"best_cut={anneal_best_cut:.6f}, "
            f"steps={int(args.anneal_steps)}, restarts={int(args.anneal_restarts)}, seed={anneal_seed}"
        )

    trained_sweep = _sweep_expected_cut(
        n_qubits=n_qubits,
        edges_json=edges_json,
        p_layers=p_layers,
        thetas_cpu=best_thetas_cpu,
        m_edges=m_edges,
        run_device=run_device,
        min_abs_values=min_abs_values,
        build_min_mat_abs=args.build_min_mat_abs,
    )
    init_sweep = _sweep_expected_cut(
        n_qubits=n_qubits,
        edges_json=edges_json,
        p_layers=p_layers,
        thetas_cpu=init_thetas_cpu,
        m_edges=m_edges,
        run_device=run_device,
        min_abs_values=min_abs_values,
        build_min_mat_abs=args.build_min_mat_abs,
    )

    anchor_tr = float(trained_sweep[-1]["expected_cut"])
    anchor_in = float(init_sweep[-1]["expected_cut"])
    for row in trained_sweep:
        row["abs_diff_to_smallest_min_abs"] = abs(float(row["expected_cut"]) - anchor_tr)
    for row in init_sweep:
        row["abs_diff_to_smallest_min_abs"] = abs(float(row["expected_cut"]) - anchor_in)

    result["surrogate_min_abs_sweep"] = {
        "trained": trained_sweep,
        "initial": init_sweep,
        "anchor_min_abs": float(min_abs_values[-1]),
    }

    can_exact = n_qubits <= int(args.exact_max_qubits)
    can_sample = n_qubits <= int(args.sampling_max_qubits)
    exact_trained = None
    exact_init = None

    init_sur = _surrogate_expected_cut(program, init_thetas_cpu, run_device, m_edges)
    trained_sur = _surrogate_expected_cut(program, best_thetas_cpu, run_device, m_edges)

    result["expected_cut_compare"] = {
        "initial_surrogate": float(init_sur),
        "trained_surrogate": float(trained_sur),
        "delta_trained_minus_initial_surrogate": float(trained_sur - init_sur),
    }

    if can_exact:
        exact_trained = _exact_expected_cut(program, best_thetas_cpu, int(args.exact_max_qubits), m_edges)
        exact_init = _exact_expected_cut(program, init_thetas_cpu, int(args.exact_max_qubits), m_edges)
        result["expected_cut_compare"].update(
            {
                "initial_exact_pennylane": float(exact_init),
                "trained_exact_pennylane": float(exact_trained),
                "initial_abs_error_surrogate_vs_exact": abs(float(init_sur) - float(exact_init)),
                "trained_abs_error_surrogate_vs_exact": abs(float(trained_sur) - float(exact_trained)),
            }
        )
        if anneal_best_cut is not None:
            result["expected_cut_compare"]["anneal_gap_to_exact_trained"] = float(
                float(exact_trained) - anneal_best_cut
            )

    if can_exact and can_sample:
        bf = brute_force_maxcut(n_qubits=n_qubits, edges=edges)
        best_cut = int(bf["best_cut"])
        best_codes = set(int(c) for c in bf["best_codes"])

        stats_init = _sample_cut_stats(
            circuit=circuit,
            thetas_cpu=init_thetas_cpu,
            n_qubits=n_qubits,
            edges=edges,
            shots=int(args.shots),
            max_qubits=int(args.sampling_max_qubits),
            seed=int(args.seed),
            best_codes=best_codes,
        )
        stats_trained = _sample_cut_stats(
            circuit=circuit,
            thetas_cpu=best_thetas_cpu,
            n_qubits=n_qubits,
            edges=edges,
            shots=int(args.shots),
            max_qubits=int(args.sampling_max_qubits),
            seed=int(args.seed) + 1,
            best_codes=best_codes,
        )

        result.update(
            {
                "mode": "small_exact_sampling",
                "classical_max_cut": best_cut,
                "num_optimal_bitstrings": len(best_codes),
                "sampling_compare": {
                    "initial": {
                        "sampled_mean_cut": stats_init["sampled_mean_cut"],
                        "sampled_best_cut": stats_init["sampled_best_cut"],
                        "sampled_fraction_optimal": stats_init["sampled_fraction_optimal"],
                    },
                    "trained": {
                        "sampled_mean_cut": stats_trained["sampled_mean_cut"],
                        "sampled_best_cut": stats_trained["sampled_best_cut"],
                        "sampled_fraction_optimal": stats_trained["sampled_fraction_optimal"],
                    },
                },
            }
        )

        print(f"[small-mode] initial surrogate E[cut]: {init_sur:.6f}")
        print(f"[small-mode] trained surrogate E[cut]: {trained_sur:.6f}")
        print(f"[small-mode] initial exact E[cut]: {float(exact_init):.6f}")
        print(f"[small-mode] trained exact E[cut]: {float(exact_trained):.6f}")
        print(
            f"[small-mode] sampled mean cut init/trained: "
            f"{stats_init['sampled_mean_cut']:.6f} / {stats_trained['sampled_mean_cut']:.6f}"
        )

        if not args.no_plots:
            plot_dir.mkdir(parents=True, exist_ok=True)
            _plot_min_abs_sweep(
                plot_dir / "min_abs_sweep.png",
                trained_rows=trained_sweep,
                init_rows=init_sweep,
                exact_trained=exact_trained,
                exact_init=exact_init,
                anneal_best_cut=anneal_best_cut,
            )
            _plot_expected_cut_bars(
                plot_dir / "expected_cut_compare.png",
                init_sur=init_sur,
                trained_sur=trained_sur,
                init_exact=exact_init,
                trained_exact=exact_trained,
                anneal_best_cut=anneal_best_cut,
            )
            _plot_sampling_hist_compare(
                plot_dir / "sampling_cut_hist_init_vs_trained.png",
                cuts_init=stats_init["cuts"],
                cuts_trained=stats_trained["cuts"],
            )
            _plot_training_curve(
                plot_dir / "training_curve_expected_cut.png",
                history_sum_zz=history_sum_zz,
                m_edges=m_edges,
                anneal_best_cut=anneal_best_cut,
            )
            result["plot_dir"] = str(plot_dir)
    else:
        result.update({"mode": "large_min_abs_sweep"})
        print("[large-mode] min_abs sweep (larger -> smaller):")
        for row in trained_sweep:
            print(
                f"  trained min_abs={row['min_abs']:.2e} "
                f"E[cut]={row['expected_cut']:.6f} "
                f"|dE|={row['abs_diff_to_smallest_min_abs']:.6f}"
            )
        for row in init_sweep:
            print(
                f"  initial min_abs={row['min_abs']:.2e} "
                f"E[cut]={row['expected_cut']:.6f} "
                f"|dE|={row['abs_diff_to_smallest_min_abs']:.6f}"
            )
        if not args.no_plots:
            plot_dir.mkdir(parents=True, exist_ok=True)
            _plot_min_abs_sweep(
                plot_dir / "min_abs_sweep.png",
                trained_rows=trained_sweep,
                init_rows=init_sweep,
                exact_trained=None,
                exact_init=None,
                anneal_best_cut=anneal_best_cut,
            )
            _plot_expected_cut_bars(
                plot_dir / "expected_cut_compare.png",
                init_sur=init_sur,
                trained_sur=trained_sur,
                init_exact=None,
                trained_exact=None,
                anneal_best_cut=anneal_best_cut,
            )
            _plot_training_curve(
                plot_dir / "training_curve_expected_cut.png",
                history_sum_zz=history_sum_zz,
                m_edges=m_edges,
                anneal_best_cut=anneal_best_cut,
            )
            result["plot_dir"] = str(plot_dir)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved eval report: {out_path}")


if __name__ == "__main__":
    main()
