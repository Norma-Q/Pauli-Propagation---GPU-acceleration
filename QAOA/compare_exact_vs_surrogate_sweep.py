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
from qaoa_surrogate_common import build_maxcut_observable, build_qaoa_circuit, expected_cut_from_sum_zz  # type: ignore


def _parse_steps(raw: str) -> Optional[List[int]]:
    text = str(raw).strip()
    if text == "":
        return None
    out: List[int] = []
    for part in text.split(","):
        t = part.strip()
        if t == "":
            continue
        out.append(int(t))
    if len(out) == 0:
        return None
    return sorted(set(out))


def _extract_step_from_name(name: str) -> Optional[int]:
    if not name.startswith("step_") or not name.endswith(".pt"):
        return None
    body = name[len("step_") : -len(".pt")]
    if not body.isdigit():
        return None
    return int(body)


def _collect_step_checkpoints(step_dir: Path, target_steps: Optional[List[int]]) -> List[Path]:
    if not step_dir.exists():
        return []
    files = sorted(step_dir.glob("step_*.pt"))
    if target_steps is None:
        return files
    target = set(int(x) for x in target_steps)
    out: List[Path] = []
    for f in files:
        s = _extract_step_from_name(f.name)
        if s is not None and s in target:
            out.append(f)
    return out


def _load_theta(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if "thetas" in payload:
        return payload["thetas"].detach().cpu().to(torch.float64)
    if "best_thetas" in payload:
        return payload["best_thetas"].detach().cpu().to(torch.float64)
    if "final_thetas" in payload:
        return payload["final_thetas"].detach().cpu().to(torch.float64)
    raise KeyError(f"No theta tensor found in checkpoint: {path}")


def _choose_runtime_device(preferred: str) -> str:
    pref = str(preferred).lower()
    if pref.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _plot_run(records: List[Dict[str, Any]], out_path: Path, title: str) -> None:
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
    ax2.set_xlabel("step")
    ax2.set_ylabel("|error| in <sum ZZ>")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_global(condition_rows: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    labels = [f"q{int(r['n_qubits'])}-{r['weight_mode']}-mw{int(r['max_weight'])}" for r in condition_rows]
    vals = [float(r["mean_abs_error_sum_zz"]) for r in condition_rows]

    order = np.argsort(np.asarray(vals, dtype=np.float64))
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(10, int(len(labels) * 0.5)), 5))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("mean |error| in <sum ZZ>")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "For each run in a sweep summary, compare stepwise surrogate expval against exact expval "
            "and save per-run/global JSON + plots."
        )
    )
    p.add_argument("--sweep-dir", type=str, default="QAOA/artifacts/weight_sweep_yz_xyz_cmp")
    p.add_argument("--steps", type=str, default="0,5,9")
    p.add_argument("--run-name-filter", type=str, default="")
    p.add_argument("--exact-max-qubits", type=int, default=20)
    p.add_argument("--match-tol", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = summary.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError("No run records in summary")

    target_steps = _parse_steps(args.steps)
    run_name_filter = str(args.run_name_filter).strip()

    out_root = sweep_dir / "analysis" / "exact_expval_compare"
    out_root.mkdir(parents=True, exist_ok=True)

    all_run_summaries: List[Dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status", "")) != "ok":
            continue

        run_name = str(rec.get("run_name", "")).strip()
        if run_name == "":
            continue
        if run_name_filter != "" and run_name_filter not in run_name:
            continue

        report_path = Path(str(rec.get("report", ""))).resolve()
        if not report_path.exists():
            print(f"[skip] missing run report: {run_name}")
            continue

        run_report = json.loads(report_path.read_text(encoding="utf-8"))
        cfg = run_report.get("config", {})
        graph = run_report.get("graph", {})

        n_qubits = int(cfg.get("n_qubits"))
        if int(n_qubits) > int(args.exact_max_qubits):
            print(f"[skip] n_qubits={n_qubits} exceeds exact-max-qubits for run={run_name}")
            continue

        p_layers = int(cfg.get("p_layers"))
        edges_raw = graph.get("edges", [])
        edges = [(int(u), int(v)) for (u, v) in edges_raw]

        circuit, _ = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=p_layers)
        obs = build_maxcut_observable(n_qubits=n_qubits, edges=edges)
        m_edges = len(edges)

        preset_name = str(cfg.get("preset", "hybrid"))
        preset_overrides = {
            "max_weight": int(cfg.get("max_weight", 1_000_000_000)),
            "weight_x": float(cfg.get("weight_x", 1.0)),
            "weight_y": float(cfg.get("weight_y", 1.0)),
            "weight_z": float(cfg.get("weight_z", 1.0)),
            "chunk_size": int(cfg.get("chunk_size", 1_000_000)),
        }
        if preset_name == "cpu":
            preset_overrides["memory_device"] = "cpu"
            preset_overrides["compute_device"] = "cpu"
            preset_overrides["dtype"] = "float64"

        checkpoint_path = Path(str(rec.get("checkpoint", ""))).resolve()
        if checkpoint_path.name == "":
            print(f"[skip] checkpoint path missing: {run_name}")
            continue
        step_dir = checkpoint_path.parent / f"{run_name}_steps"
        step_ckpts = _collect_step_checkpoints(step_dir, target_steps)
        if len(step_ckpts) == 0:
            print(f"[skip] no step checkpoints: {run_name}")
            continue

        first_theta = _load_theta(step_ckpts[0])
        build_min_abs = cfg.get("build_min_abs", None)
        build_min_mat_abs = cfg.get("build_min_mat_abs", None)

        program = compile_expval_program(
            circuit=circuit,
            observables=[obs],
            preset=preset_name,
            preset_overrides=preset_overrides,
            build_thetas=first_theta,
            build_min_abs=build_min_abs,
            build_min_mat_abs=build_min_mat_abs,
        )

        runtime_device = _choose_runtime_device(str(cfg.get("device", "cpu")))

        per_step: List[Dict[str, Any]] = []
        for ckpt in step_ckpts:
            step = _extract_step_from_name(ckpt.name)
            if step is None:
                continue
            theta_cpu = _load_theta(ckpt)
            theta_runtime = theta_cpu.to(runtime_device)

            with torch.no_grad():
                approx_sum_zz = float(program.expval(theta_runtime, obs_index=0).detach().cpu().item())
                exact_sum_zz = float(program.expvals_pennylane(theta_cpu, max_qubits=int(args.exact_max_qubits))[0].item())

            abs_err = float(abs(approx_sum_zz - exact_sum_zz))
            rel_err = float(abs_err / max(1e-12, abs(exact_sum_zz)))
            approx_cut = float(expected_cut_from_sum_zz(approx_sum_zz, m_edges))
            exact_cut = float(expected_cut_from_sum_zz(exact_sum_zz, m_edges))

            per_step.append(
                {
                    "step": int(step),
                    "checkpoint": str(ckpt),
                    "approx_sum_zz": float(approx_sum_zz),
                    "exact_sum_zz": float(exact_sum_zz),
                    "abs_error_sum_zz": float(abs_err),
                    "rel_error_sum_zz": float(rel_err),
                    "approx_expected_cut": float(approx_cut),
                    "exact_expected_cut": float(exact_cut),
                    "match_within_tol": bool(abs_err <= float(args.match_tol)),
                }
            )

        if len(per_step) == 0:
            print(f"[skip] no comparable checkpoints: {run_name}")
            continue

        per_step = sorted(per_step, key=lambda x: int(x["step"]))
        abs_errors = [float(x["abs_error_sum_zz"]) for x in per_step]

        run_payload = {
            "run_name": run_name,
            "n_qubits": int(n_qubits),
            "p_layers": int(p_layers),
            "weight_mode": str(cfg.get("weight_mode", rec.get("weight_mode", ""))),
            "max_weight": int(cfg.get("max_weight", rec.get("max_weight", 0))),
            "match_tol": float(args.match_tol),
            "steps": [int(x["step"]) for x in per_step],
            "expval_comparison": per_step,
            "summary": {
                "mean_abs_error_sum_zz": float(np.mean(np.asarray(abs_errors, dtype=np.float64))),
                "max_abs_error_sum_zz": float(np.max(np.asarray(abs_errors, dtype=np.float64))),
                "n_match_within_tol": int(sum(1 for x in per_step if bool(x["match_within_tol"]))),
                "n_total": int(len(per_step)),
            },
        }

        run_out_json = out_root / f"{run_name}_exact_vs_surrogate.json"
        run_out_json.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

        run_out_plot = out_root / f"{run_name}_exact_vs_surrogate.png"
        _plot_run(
            records=per_step,
            out_path=run_out_plot,
            title=(
                f"{run_name} | q={n_qubits}, p={p_layers}, "
                f"mode={run_payload['weight_mode']}, mw={run_payload['max_weight']}"
            ),
        )

        all_run_summaries.append(
            {
                "run_name": run_name,
                "n_qubits": int(n_qubits),
                "p_layers": int(p_layers),
                "weight_mode": str(run_payload["weight_mode"]),
                "max_weight": int(run_payload["max_weight"]),
                "mean_abs_error_sum_zz": float(run_payload["summary"]["mean_abs_error_sum_zz"]),
                "max_abs_error_sum_zz": float(run_payload["summary"]["max_abs_error_sum_zz"]),
                "n_match_within_tol": int(run_payload["summary"]["n_match_within_tol"]),
                "n_total": int(run_payload["summary"]["n_total"]),
                "json": str(run_out_json),
                "plot": str(run_out_plot),
            }
        )

        print(
            f"[ok] {run_name} mean_abs_err={run_payload['summary']['mean_abs_error_sum_zz']:.3e} "
            f"match={run_payload['summary']['n_match_within_tol']}/{run_payload['summary']['n_total']}"
        )

    if len(all_run_summaries) == 0:
        raise RuntimeError("No run-level exact-vs-surrogate comparisons were produced")

    grouped: Dict[tuple[int, str, int], List[Dict[str, Any]]] = {}
    for row in all_run_summaries:
        key = (int(row["n_qubits"]), str(row["weight_mode"]), int(row["max_weight"]))
        grouped.setdefault(key, []).append(row)

    cond_rows: List[Dict[str, Any]] = []
    for (n_qubits, weight_mode, max_weight), rows in grouped.items():
        mean_err = float(np.mean(np.asarray([float(r["mean_abs_error_sum_zz"]) for r in rows], dtype=np.float64)))
        max_err = float(np.max(np.asarray([float(r["max_abs_error_sum_zz"]) for r in rows], dtype=np.float64)))
        n_match = int(sum(int(r["n_match_within_tol"]) for r in rows))
        n_total = int(sum(int(r["n_total"]) for r in rows))
        cond_rows.append(
            {
                "n_qubits": int(n_qubits),
                "weight_mode": str(weight_mode),
                "max_weight": int(max_weight),
                "n_runs": int(len(rows)),
                "mean_abs_error_sum_zz": float(mean_err),
                "max_abs_error_sum_zz": float(max_err),
                "match_rate": float(n_match / max(1, n_total)),
            }
        )

    cond_rows = sorted(cond_rows, key=lambda x: float(x["mean_abs_error_sum_zz"]))

    global_payload = {
        "sweep_dir": str(sweep_dir),
        "steps": target_steps,
        "match_tol": float(args.match_tol),
        "per_run": all_run_summaries,
        "by_condition": cond_rows,
    }

    global_json = out_root / "exact_vs_surrogate_summary.json"
    global_json.write_text(json.dumps(global_payload, indent=2), encoding="utf-8")

    global_plot = out_root / "exact_vs_surrogate_condition_error.png"
    _plot_global(
        cond_rows,
        global_plot,
        title="Exact vs Surrogate error by condition (lower is better)",
    )

    qubit_values = sorted({int(row["n_qubits"]) for row in cond_rows})
    qubit_plots: List[str] = []
    for n_qubits in qubit_values:
        rows_q = [row for row in cond_rows if int(row["n_qubits"]) == int(n_qubits)]
        if len(rows_q) == 0:
            continue
        out_q = out_root / f"exact_vs_surrogate_condition_error_q{n_qubits}.png"
        _plot_global(
            rows_q,
            out_q,
            title=(
                "Exact vs Surrogate error by condition "
                f"(q={n_qubits}, lower is better)"
            ),
        )
        qubit_plots.append(str(out_q))

    global_payload["by_n_qubits_plots"] = qubit_plots
    global_json.write_text(json.dumps(global_payload, indent=2), encoding="utf-8")

    print(f"saved global summary: {global_json}")
    print(f"saved global plot: {global_plot}")
    for path in qubit_plots:
        print(f"saved qubit plot: {path}")


if __name__ == "__main__":
    main()
