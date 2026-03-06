from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TEST_QAOA = _REPO_ROOT / "test_qaoa"
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    sys.path.insert(0, path)

from src_tensor.api import compile_expval_program
from qaoa_surrogate_common import (  # type: ignore
    build_maxcut_observable,
    build_qaoa_circuit,
    build_qaoa_theta_init_tqa,
    expected_cut_from_sum_zz,
    load_edges_json,
    make_ring_chord_graph,
)


def _choose_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(raw)


def _resolve_weight_tuple(mode: str) -> Dict[str, float]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "yz":
        return {"weight_x": 0.0, "weight_y": 1.0, "weight_z": 1.0}
    if mode_norm == "xyz":
        return {"weight_x": 1.0, "weight_y": 1.0, "weight_z": 1.0}
    raise ValueError("weight-mode must be one of: yz, xyz")


def _save_step_theta(step_dir: Path, step: int, thetas: torch.Tensor) -> Path:
    step_path = step_dir / f"step_{int(step):06d}.pt"
    torch.save({"thetas": thetas.detach().cpu()}, step_path)
    return step_path


def _load_theta(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if "thetas" in payload:
        return payload["thetas"].detach().cpu().to(torch.float64)
    if "best_thetas" in payload:
        return payload["best_thetas"].detach().cpu().to(torch.float64)
    raise KeyError(f"Checkpoint missing theta tensor: {path}")


def _plot_results(records: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    steps = [int(r["step"]) for r in records]
    approx = [float(r["approx_sum_zz"]) for r in records]
    exact = [float(r["exact_sum_zz"]) for r in records]
    abs_err = [float(r["abs_error_sum_zz"]) for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(steps, approx, marker="o", linewidth=1.6, label="surrogate")
    ax1.plot(steps, exact, marker="s", linewidth=1.6, label="exact")
    ax1.set_ylabel("<sum ZZ>")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(steps, abs_err, marker="o", linewidth=1.6, color="tab:red")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("|error| in <sum ZZ>")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Short QAOA training (<=10 steps), save per-step thetas, "
            "and compare surrogate expval with exact expval."
        )
    )
    p.add_argument("--edges-json", type=str, default="")
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--p-layers", type=int, default=3)
    p.add_argument("--delta-t", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chord-shift", type=int, default=2)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--preset", type=str, default="hybrid", choices=["hybrid", "cpu"])

    p.add_argument("--max-weight", type=int, default=2)
    p.add_argument("--weight-mode", type=str, default="yz", choices=["yz", "xyz"])
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--build-min-abs", type=float, default=1e-3)
    p.add_argument("--build-min-mat-abs", type=float, default=None)

    p.add_argument("--output-dir", type=str, default="QAOA/artifacts/weight_sweep_short_exact")
    p.add_argument("--run-name", type=str, default="short_exact_check")
    p.add_argument("--exact-max-qubits", type=int, default=20)
    p.add_argument("--match-tol", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_steps = int(args.steps)
    if n_steps < 1:
        raise ValueError("steps must be >= 1")
    if n_steps > 10:
        raise ValueError("For this experiment, steps must be <= 10")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    if str(args.edges_json).strip():
        edges = load_edges_json(args.edges_json)
    else:
        edges = make_ring_chord_graph(int(args.n_qubits), chord_shift=int(args.chord_shift))

    m_edges = len(edges)
    if m_edges < 1:
        raise ValueError("Graph must contain at least one edge")

    circuit, n_params = build_qaoa_circuit(
        n_qubits=int(args.n_qubits),
        edges=edges,
        p_layers=int(args.p_layers),
    )
    obs = build_maxcut_observable(n_qubits=int(args.n_qubits), edges=edges)

    init_theta_np = build_qaoa_theta_init_tqa(
        p_layers=int(args.p_layers),
        n_edges=int(m_edges),
        n_qubits=int(args.n_qubits),
        delta_t=float(args.delta_t),
        dtype=np.float64,
    )
    if int(init_theta_np.shape[0]) != int(n_params):
        raise RuntimeError("TQA init size mismatch")

    run_device = _choose_device(str(args.device))
    if run_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    weight_tuple = _resolve_weight_tuple(str(args.weight_mode))

    preset = str(args.preset)
    preset_overrides: Dict[str, Any] = {
        "max_weight": int(args.max_weight),
        "weight_x": float(weight_tuple["weight_x"]),
        "weight_y": float(weight_tuple["weight_y"]),
        "weight_z": float(weight_tuple["weight_z"]),
    }
    if int(args.chunk_size) > 0:
        preset_overrides["chunk_size"] = int(args.chunk_size)
    if preset == "cpu":
        preset_overrides["memory_device"] = "cpu"
        preset_overrides["compute_device"] = "cpu"
        preset_overrides["dtype"] = "float64"

    thetas = torch.nn.Parameter(torch.tensor(init_theta_np, dtype=torch.float64, device=run_device))

    compile_thetas = thetas.detach().cpu()
    program = compile_expval_program(
        circuit=circuit,
        observables=[obs],
        preset=preset,
        preset_overrides=preset_overrides,
        build_thetas=compile_thetas,
        build_min_abs=float(args.build_min_abs),
        build_min_mat_abs=args.build_min_mat_abs,
    )

    opt = torch.optim.Adam([thetas], lr=float(args.lr))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / str(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    step_dir = run_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)

    train_log: List[Dict[str, Any]] = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        zz_val = program.expval(thetas, obs_index=0)
        zz_val.backward()
        opt.step()

        train_sum_zz = float(zz_val.detach().cpu().item())
        exp_cut = float(expected_cut_from_sum_zz(train_sum_zz, m_edges))
        step_path = _save_step_theta(step_dir, step, thetas)
        train_log.append(
            {
                "step": int(step),
                "train_sum_zz": float(train_sum_zz),
                "train_expected_cut": float(exp_cut),
                "theta_path": str(step_path),
            }
        )
        print(f"[train] step={step:02d} sum<ZZ>={train_sum_zz:+.8f} E[cut]={exp_cut:.6f}")

    if int(args.n_qubits) > int(args.exact_max_qubits):
        raise ValueError(
            f"Exact PennyLane check requires n_qubits <= exact-max-qubits ({args.exact_max_qubits})"
        )

    eval_records: List[Dict[str, Any]] = []
    for item in train_log:
        step = int(item["step"])
        theta_path = Path(str(item["theta_path"]))
        theta_cpu = _load_theta(theta_path)

        with torch.no_grad():
            approx_sum_zz = float(program.expval(theta_cpu.to(run_device), obs_index=0).detach().cpu().item())
            exact_sum_zz = float(program.expvals_pennylane(theta_cpu, max_qubits=int(args.exact_max_qubits))[0].item())

        approx_cut = float(expected_cut_from_sum_zz(approx_sum_zz, m_edges))
        exact_cut = float(expected_cut_from_sum_zz(exact_sum_zz, m_edges))

        abs_err = float(abs(approx_sum_zz - exact_sum_zz))
        rel_err = float(abs_err / max(1e-12, abs(exact_sum_zz)))
        match = bool(abs_err <= float(args.match_tol))

        rec = {
            "step": int(step),
            "theta_path": str(theta_path),
            "approx_sum_zz": float(approx_sum_zz),
            "exact_sum_zz": float(exact_sum_zz),
            "abs_error_sum_zz": float(abs_err),
            "rel_error_sum_zz": float(rel_err),
            "approx_expected_cut": float(approx_cut),
            "exact_expected_cut": float(exact_cut),
            "match_within_tol": bool(match),
        }
        eval_records.append(rec)
        print(
            f"[eval] step={step:02d} approx={approx_sum_zz:+.8f} exact={exact_sum_zz:+.8f} "
            f"abs_err={abs_err:.3e} match={match}"
        )

    abs_errs = [float(r["abs_error_sum_zz"]) for r in eval_records]
    summary = {
        "max_abs_error_sum_zz": float(max(abs_errs) if abs_errs else 0.0),
        "mean_abs_error_sum_zz": float(np.mean(np.asarray(abs_errs, dtype=np.float64)) if abs_errs else 0.0),
        "n_match_within_tol": int(sum(1 for r in eval_records if bool(r["match_within_tol"]))),
        "n_total": int(len(eval_records)),
        "tol": float(args.match_tol),
    }

    result = {
        "config": {
            "n_qubits": int(args.n_qubits),
            "p_layers": int(args.p_layers),
            "steps": int(n_steps),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "preset": str(preset),
            "device": str(run_device),
            "max_weight": int(args.max_weight),
            "weight_mode": str(args.weight_mode),
            "weight_x": float(weight_tuple["weight_x"]),
            "weight_y": float(weight_tuple["weight_y"]),
            "weight_z": float(weight_tuple["weight_z"]),
            "chunk_size": int(args.chunk_size),
            "build_min_abs": float(args.build_min_abs),
            "build_min_mat_abs": args.build_min_mat_abs,
            "exact_max_qubits": int(args.exact_max_qubits),
            "match_tol": float(args.match_tol),
        },
        "graph": {
            "n_edges": int(m_edges),
            "edges": [[int(u), int(v)] for (u, v) in edges],
        },
        "training_steps": train_log,
        "expval_comparison": eval_records,
        "summary": summary,
    }

    result_json = run_dir / "expval_exact_comparison.json"
    result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_path = run_dir / "expval_exact_comparison.png"
    _plot_results(
        records=eval_records,
        out_path=plot_path,
        title=(
            f"QAOA expval surrogate vs exact | q={int(args.n_qubits)}, p={int(args.p_layers)}, "
            f"mode={args.weight_mode}, max_weight={int(args.max_weight)}"
        ),
    )

    print(f"saved steps: {step_dir}")
    print(f"saved result json: {result_json}")
    print(f"saved plot: {plot_path}")
    print(
        "summary: "
        f"max_abs_err={summary['max_abs_error_sum_zz']:.6e}, "
        f"mean_abs_err={summary['mean_abs_error_sum_zz']:.6e}, "
        f"match={summary['n_match_within_tol']}/{summary['n_total']} (tol={summary['tol']})"
    )


if __name__ == "__main__":
    main()
