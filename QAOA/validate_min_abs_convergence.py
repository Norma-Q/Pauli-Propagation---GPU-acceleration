from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TEST_QAOA = _REPO_ROOT / "test_qaoa"
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    sys.path.insert(0, path)

from src_tensor.api import compile_expval_program
from qaoa_surrogate_common import (
    build_maxcut_observable,
    build_qaoa_circuit,
    expected_cut_from_sum_zz,
    load_edges_json,
    parse_min_abs_schedule,
)


StepToken = Union[int, str]


class _TeeStream:
    def __init__(self, streams: List[TextIO]) -> None:
        self._streams = [s for s in streams if s is not None]

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def _setup_realtime_log(log_path: Path, append: bool, mirror_terminal: bool) -> TextIO:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if bool(append) else "w"
    log_fp = log_path.open(mode, encoding="utf-8", buffering=1)
    if bool(mirror_terminal):
        sys.stdout = _TeeStream([sys.stdout, log_fp])
        sys.stderr = _TeeStream([sys.stderr, log_fp])
    else:
        sys.stdout = _TeeStream([log_fp])
        sys.stderr = _TeeStream([log_fp])
    print(f"[log] writing realtime log to: {log_path}")
    return log_fp


def _resolve_sweep_dir(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p.resolve()

    candidates = [
        (_REPO_ROOT / p),
        (_THIS_DIR / p),
        (_THIS_DIR / "artifacts" / p.name),
        (_THIS_DIR / "artifacts" / "sweep_multi_graphs"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    tried = "\n  - ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(f"Sweep folder not found. Tried:\n  - {tried}")


def _choose_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(raw)


def _parse_steps(raw: str, include_final: bool) -> List[StepToken]:
    tokens: List[StepToken] = []
    text = str(raw).strip()
    if text == "":
        tokens = [0, 50, 100, 150, 200]
    else:
        for part in text.split(","):
            item = part.strip().lower()
            if item == "":
                continue
            if item in ("final", "best"):
                tokens.append("final")
            else:
                tokens.append(int(item))

    if include_final and "final" not in tokens:
        tokens.append("final")

    out: List[StepToken] = []
    seen: set[str] = set()
    for t in tokens:
        key = str(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _make_min_abs_values(args: argparse.Namespace) -> List[float]:
    if str(args.min_abs_values).strip() != "":
        return parse_min_abs_schedule(str(args.min_abs_values))

    hi = float(args.min_abs_max)
    lo = float(args.min_abs_min)
    n = int(args.min_abs_num_points)
    if hi <= 0.0 or lo <= 0.0:
        raise ValueError("min_abs_max and min_abs_min must be > 0")
    if hi < lo:
        hi, lo = lo, hi
    if n < 2:
        raise ValueError("min_abs_num_points must be >= 2")

    vals = np.logspace(np.log10(hi), np.log10(lo), num=n, dtype=np.float64)
    return [float(v) for v in vals.tolist()]


def _extract_step_from_name(name: str) -> Optional[int]:
    m = re.match(r"^step_(\d{6})\.pt$", str(name))
    if m is None:
        return None
    return int(m.group(1))


def _collect_step_checkpoints(step_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not step_dir.exists():
        return out
    for item in sorted(step_dir.glob("step_*.pt")):
        step = _extract_step_from_name(item.name)
        if step is None:
            continue
        out[int(step)] = item.resolve()
    return out


def _load_thetas_from_checkpoint(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu")
    if "best_thetas" in payload:
        return payload["best_thetas"].detach().cpu().numpy()
    if "thetas" in payload:
        return payload["thetas"].detach().cpu().numpy()
    if "final_thetas" in payload:
        return payload["final_thetas"].detach().cpu().numpy()
    raise KeyError(f"Checkpoint missing known theta keys: {path}")


def _extract_compile_resources(program: Any) -> Dict[str, Any]:
    psum = program.psum_union
    n_terms = int(psum.x_mask.shape[0])
    n_steps = int(len(psum.steps))

    nnz_const_total = 0
    nnz_cos_total = 0
    nnz_sin_total = 0
    for step in psum.steps:
        nnz_const_total += int(step.mat_const._nnz())
        nnz_cos_total += int(step.mat_cos._nnz())
        nnz_sin_total += int(step.mat_sin._nnz())

    nnz_total = int(nnz_const_total + nnz_cos_total + nnz_sin_total)
    return {
        "terms_after_zero_filter": int(n_terms),
        "n_steps": int(n_steps),
        "nnz_total": int(nnz_total),
        "nnz_const_total": int(nnz_const_total),
        "nnz_cos_total": int(nnz_cos_total),
        "nnz_sin_total": int(nnz_sin_total),
    }


def _discover_runs(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = summary.get("records", []) if isinstance(summary, dict) else []
    if not isinstance(records, list):
        return []

    out: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status", "")) != "ok":
            continue
        run_name = str(rec.get("run_name", "")).strip()
        if run_name == "":
            continue
        report = str(rec.get("report", "")).strip()
        checkpoint = str(rec.get("checkpoint", "")).strip()
        if report == "" or checkpoint == "":
            continue
        out.append(
            {
                "run_name": run_name,
                "n_qubits": rec.get("n_qubits"),
                "p_layers": rec.get("p_layers"),
                "graph_index": rec.get("graph_index"),
                "weight_mode": rec.get("weight_mode"),
                "max_weight": rec.get("max_weight"),
                "graph_edges_json": rec.get("graph_edges_json"),
                "report": str(Path(report).resolve()),
                "checkpoint": str(Path(checkpoint).resolve()),
            }
        )
    return out


def _infer_n_qubits(edges: Sequence[Tuple[int, int]]) -> int:
    if len(edges) == 0:
        raise ValueError("Edge list is empty")
    return max(max(int(u), int(v)) for (u, v) in edges) + 1


def _to_edge_pairs(raw_edges: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(e[0]), int(e[1])) for e in raw_edges]


def _resolve_graph_edges(rec: Dict[str, Any], report: Dict[str, Any]) -> List[Tuple[int, int]]:
    edges_json = str(rec.get("graph_edges_json", "")).strip()
    if edges_json != "" and Path(edges_json).exists():
        return load_edges_json(edges_json)

    graph_edges = report.get("graph", {}).get("edges", [])
    if isinstance(graph_edges, list) and len(graph_edges) > 0:
        return _to_edge_pairs(graph_edges)

    raise FileNotFoundError(f"Could not resolve graph edges for run={rec.get('run_name')}")


def _build_preset_and_overrides(
    report: Dict[str, Any],
    run_device: str,
    preset_choice: str,
    chunk_size: int,
) -> Tuple[str, Dict[str, Any]]:
    cfg = report.get("config", {}) if isinstance(report, dict) else {}
    cfg_preset = str(cfg.get("preset", "hybrid")).strip().lower()

    if preset_choice == "auto":
        preset = cfg_preset if cfg_preset in ("cpu", "hybrid") else ("hybrid" if run_device.startswith("cuda") else "cpu")
    elif preset_choice == "gpu":
        preset = "hybrid"
    elif preset_choice == "cpu":
        preset = "cpu"
    else:
        raise ValueError(f"Unsupported preset-choice: {preset_choice}")

    overrides: Dict[str, Any] = {}
    for k in ("max_weight", "weight_x", "weight_y", "weight_z"):
        if k in cfg and cfg[k] is not None:
            overrides[k] = cfg[k]
    if int(chunk_size) > 0:
        overrides["chunk_size"] = int(chunk_size)
    elif "chunk_size" in cfg and cfg["chunk_size"] is not None:
        overrides["chunk_size"] = int(cfg["chunk_size"])

    return preset, overrides


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        out_path.write_text("", encoding="utf-8")
        return

    keys: List[str] = []
    keyset: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in keyset:
                keyset.add(k)
                keys.append(k)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_convergence(
    *,
    min_abs_values: List[float],
    expected_values: List[float],
    abs_eps: float,
    tail_window: int,
) -> Dict[str, Any]:
    if len(min_abs_values) != len(expected_values):
        raise ValueError("min_abs_values and expected_values length mismatch")
    if len(min_abs_values) < 2:
        raise ValueError("need at least two min_abs points for convergence summary")

    deltas: List[float] = []
    for i in range(len(expected_values) - 1):
        deltas.append(float(abs(float(expected_values[i + 1]) - float(expected_values[i]))))

    tw = int(max(1, min(int(tail_window), len(deltas))))
    tail = deltas[-tw:]
    tail_max = float(np.max(np.asarray(tail, dtype=np.float64)))
    converged = bool(tail_max <= float(abs_eps))

    stable_idx: Optional[int] = None
    for i in range(len(deltas)):
        suffix = deltas[i:]
        if len(suffix) == 0:
            continue
        if float(np.max(np.asarray(suffix, dtype=np.float64))) <= float(abs_eps):
            stable_idx = i
            break

    if stable_idx is None:
        stable_from = None
        recommended = None
    else:
        stable_from = float(min_abs_values[stable_idx])
        recommended = float(min_abs_values[stable_idx])

    return {
        "converged": converged,
        "tail_window": int(tw),
        "tail_max_abs_delta": float(tail_max),
        "tail_mean_abs_delta": float(np.mean(np.asarray(tail, dtype=np.float64))),
        "stable_from_min_abs": stable_from,
        "recommended_min_abs": recommended,
        "pair_abs_deltas": [float(x) for x in deltas],
    }


def _plot_run_min_abs_convergence(
    *,
    run_name: str,
    out_path: Path,
    step_to_rows: Dict[str, List[Dict[str, Any]]],
    abs_eps: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)

    for step_label, rows in step_to_rows.items():
        xs = np.arange(len(rows), dtype=np.float64)
        xlabels = [f"{r['min_abs']:.1e}" for r in rows]
        ys = np.asarray([float(r["expected_cut"]) for r in rows], dtype=np.float64)
        axes[0].plot(xs, ys, marker="o", linewidth=1.6, label=f"step={step_label}")
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)

    axes[0].set_title("Expected cut vs min_abs")
    axes[0].set_xlabel("min_abs (decreasing)")
    axes[0].set_ylabel("surrogate expected cut")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    for step_label, rows in step_to_rows.items():
        if len(rows) < 2:
            continue
        dvals = []
        for i in range(len(rows) - 1):
            dvals.append(abs(float(rows[i + 1]["expected_cut"]) - float(rows[i]["expected_cut"])))
        xs = np.arange(len(dvals), dtype=np.float64)
        xlabels = [f"{rows[i]['min_abs']:.1e}->{rows[i + 1]['min_abs']:.1e}" for i in range(len(dvals))]
        axes[1].plot(xs, np.asarray(dvals, dtype=np.float64), marker="o", linewidth=1.6, label=f"step={step_label}")
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)

    axes[1].axhline(float(abs_eps), color="tab:red", linestyle="--", linewidth=1.2, label=f"abs_eps={abs_eps:g}")
    axes[1].set_title("Adjacent |Δ expected cut| vs min_abs")
    axes[1].set_xlabel("adjacent min_abs pair")
    axes[1].set_ylabel("|Δ expected cut|")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    fig.suptitle(f"min_abs convergence / {run_name}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate surrogate expected-cut convergence by decreasing min_abs over saved QAOA checkpoints."
    )
    p.add_argument("--sweep-dir", type=str, required=True)
    p.add_argument("--summary-json", type=str, default="")
    p.add_argument("--steps", type=str, default="0,50,100,150,200,final")
    p.add_argument(
        "--include-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always include final checkpoint in validation targets.",
    )
    p.add_argument("--run-name-filter", type=str, default="")
    p.add_argument("--n-qubits", type=int, default=0, help="0 means all qubits in summary")
    p.add_argument("--max-runs", type=int, default=0, help="0 means no limit")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--preset-choice", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--chunk-size", type=int, default=0)

    p.add_argument("--min-abs-values", type=str, default="", help="Explicit CSV list, e.g. 1e-2,3e-3,1e-3")
    p.add_argument("--min-abs-max", type=float, default=1e-2)
    p.add_argument("--min-abs-min", type=float, default=1e-5)
    p.add_argument("--min-abs-num-points", type=int, default=7)
    p.add_argument("--build-min-mat-abs", type=float, default=None)

    p.add_argument("--abs-eps", type=float, default=0.5)
    p.add_argument("--tail-window", type=int, default=2)

    p.add_argument("--analysis-subdir", type=str, default="min_abs_convergence")
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Realtime log file path. If empty, defaults to <analysis-root>/validate_min_abs_convergence.log",
    )
    p.add_argument(
        "--log-append",
        action="store_true",
        help="Append to log file instead of overwriting.",
    )
    p.add_argument(
        "--log-to-terminal",
        action="store_true",
        help="Also mirror logs to terminal. Default writes only to log file.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_device = _choose_device(str(args.device))
    if run_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    min_abs_values = _make_min_abs_values(args)
    if len(min_abs_values) < 2:
        raise ValueError("Need at least 2 min_abs values")

    step_tokens = _parse_steps(str(args.steps), bool(args.include_final))

    sweep_dir = _resolve_sweep_dir(str(args.sweep_dir))
    summary_path = Path(args.summary_json).resolve() if str(args.summary_json).strip() else (sweep_dir / "summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = _discover_runs(summary)

    run_filter = str(args.run_name_filter).strip()
    if run_filter != "":
        runs = [r for r in runs if run_filter in str(r["run_name"])]

    nq_filter = int(args.n_qubits)
    if nq_filter > 0:
        runs = [r for r in runs if int(r.get("n_qubits") or -1) == nq_filter]

    if int(args.max_runs) > 0:
        runs = runs[: int(args.max_runs)]

    if len(runs) == 0:
        raise RuntimeError("No runs discovered after filtering.")

    analysis_root = sweep_dir / "analysis" / str(args.analysis_subdir)
    analysis_root.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(str(args.log_file)).resolve()
        if str(args.log_file).strip()
        else (analysis_root / "validate_min_abs_convergence.log").resolve()
    )
    log_fp = _setup_realtime_log(
        log_path=log_path,
        append=bool(args.log_append),
        mirror_terminal=bool(args.log_to_terminal),
    )

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    missing_targets: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    print(f"sweep_dir: {sweep_dir}")
    print(f"summary_json: {summary_path}")
    print(f"target_steps: {step_tokens}")
    print(f"min_abs_values: {[float(x) for x in min_abs_values]}")
    print(f"n_runs: {len(runs)}")

    for idx, rec in enumerate(runs, start=1):
        run_name = str(rec["run_name"])
        report_path = Path(str(rec["report"])).resolve()
        checkpoint_path = Path(str(rec["checkpoint"])).resolve()
        step_dir = checkpoint_path.parent / f"{run_name}_steps"

        print("\n" + "=" * 100)
        print(f"[{idx}/{len(runs)}] run={run_name}")

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            edges = _resolve_graph_edges(rec, report)
            n_qubits = int(rec.get("n_qubits") or _infer_n_qubits(edges))
            p_layers = int(rec.get("p_layers") or report.get("config", {}).get("p_layers"))
            n_edges = int(len(edges))

            circuit, _ = build_qaoa_circuit(
                n_qubits=int(n_qubits),
                edges=edges,
                p_layers=int(p_layers),
            )
            zz_obj = build_maxcut_observable(
                n_qubits=int(n_qubits),
                edges=edges,
            )

            preset, preset_overrides = _build_preset_and_overrides(
                report=report,
                run_device=run_device,
                preset_choice=str(args.preset_choice),
                chunk_size=int(args.chunk_size),
            )

            steps_map = _collect_step_checkpoints(step_dir)
            targets: List[Tuple[str, Path]] = []
            for tok in step_tokens:
                if tok == "final":
                    if checkpoint_path.exists():
                        targets.append(("final", checkpoint_path))
                    else:
                        missing_targets.append({"run_name": run_name, "step": "final", "reason": "checkpoint_not_found"})
                    continue

                step_int = int(tok)
                ckpt = steps_map.get(step_int)
                if ckpt is None:
                    missing_targets.append({"run_name": run_name, "step": int(step_int), "reason": "step_checkpoint_not_found"})
                    continue
                targets.append((str(step_int), ckpt))

            if len(targets) == 0:
                print(f"[skip] no target checkpoints found for run={run_name}")
                continue

            run_out_dir = analysis_root / run_name
            run_out_dir.mkdir(parents=True, exist_ok=True)

            step_to_rows_for_plot: Dict[str, List[Dict[str, Any]]] = {}

            for step_label, ckpt_path in targets:
                print(f"  - validate step={step_label} using {ckpt_path.name}")

                thetas_np = _load_thetas_from_checkpoint(ckpt_path)
                theta_cpu = torch.tensor(np.asarray(thetas_np, dtype=np.float64), dtype=torch.float64, device="cpu")
                theta_dev = theta_cpu.to(run_device)

                rows_this_step: List[Dict[str, Any]] = []

                for min_abs in min_abs_values:
                    if bool(args.dry_run):
                        row = {
                            "run_name": run_name,
                            "n_qubits": int(n_qubits),
                            "p_layers": int(p_layers),
                            "graph_index": rec.get("graph_index"),
                            "weight_mode": rec.get("weight_mode"),
                            "max_weight": rec.get("max_weight"),
                            "target_step": step_label,
                            "checkpoint": str(ckpt_path),
                            "min_abs": float(min_abs),
                            "sum_zz": np.nan,
                            "expected_cut": np.nan,
                            "terms_after_zero_filter": np.nan,
                            "nnz_total": np.nan,
                            "status": "dry_run",
                        }
                        detail_rows.append(row)
                        rows_this_step.append(row)
                        continue

                    program = compile_expval_program(
                        circuit=circuit,
                        observables=[zz_obj],
                        preset=str(preset),
                        preset_overrides=dict(preset_overrides),
                        build_thetas=theta_cpu,
                        build_min_abs=float(min_abs),
                        build_min_mat_abs=args.build_min_mat_abs,
                    )
                    sum_zz = float(program.expval(theta_dev, obs_index=0).detach().cpu().item())
                    expected_cut = float(expected_cut_from_sum_zz(sum_zz, n_edges))
                    resources = _extract_compile_resources(program)

                    row = {
                        "run_name": run_name,
                        "n_qubits": int(n_qubits),
                        "p_layers": int(p_layers),
                        "graph_index": rec.get("graph_index"),
                        "weight_mode": rec.get("weight_mode"),
                        "max_weight": rec.get("max_weight"),
                        "target_step": step_label,
                        "checkpoint": str(ckpt_path),
                        "min_abs": float(min_abs),
                        "sum_zz": float(sum_zz),
                        "expected_cut": float(expected_cut),
                        "terms_after_zero_filter": int(resources["terms_after_zero_filter"]),
                        "nnz_total": int(resources["nnz_total"]),
                        "status": "ok",
                    }
                    detail_rows.append(row)
                    rows_this_step.append(row)

                step_to_rows_for_plot[str(step_label)] = rows_this_step

                if not bool(args.dry_run):
                    exp_vals = [float(r["expected_cut"]) for r in rows_this_step]
                    conv = _summarize_convergence(
                        min_abs_values=[float(r["min_abs"]) for r in rows_this_step],
                        expected_values=exp_vals,
                        abs_eps=float(args.abs_eps),
                        tail_window=int(args.tail_window),
                    )
                    summary_rows.append(
                        {
                            "run_name": run_name,
                            "n_qubits": int(n_qubits),
                            "p_layers": int(p_layers),
                            "graph_index": rec.get("graph_index"),
                            "target_step": step_label,
                            "abs_eps": float(args.abs_eps),
                            "converged": bool(conv["converged"]),
                            "tail_window": int(conv["tail_window"]),
                            "tail_max_abs_delta": float(conv["tail_max_abs_delta"]),
                            "tail_mean_abs_delta": float(conv["tail_mean_abs_delta"]),
                            "stable_from_min_abs": conv["stable_from_min_abs"],
                            "recommended_min_abs": conv["recommended_min_abs"],
                            "min_abs_values": [float(r["min_abs"]) for r in rows_this_step],
                            "expected_values": exp_vals,
                            "pair_abs_deltas": conv["pair_abs_deltas"],
                        }
                    )

            if not bool(args.dry_run):
                plot_path = run_out_dir / "min_abs_convergence.png"
                _plot_run_min_abs_convergence(
                    run_name=run_name,
                    out_path=plot_path,
                    step_to_rows=step_to_rows_for_plot,
                    abs_eps=float(args.abs_eps),
                )
                print(f"  - saved plot: {plot_path}")

        except Exception as e:
            failed_rows.append({"run_name": run_name, "error": str(e)})
            print(f"[failed] run={run_name}: {e}")
            if not bool(args.continue_on_error):
                break

    detail_json = analysis_root / "min_abs_convergence_detail.json"
    detail_csv = analysis_root / "min_abs_convergence_detail.csv"
    summary_json_out = analysis_root / "min_abs_convergence_summary.json"
    summary_csv_out = analysis_root / "min_abs_convergence_summary.csv"
    missing_json = analysis_root / "min_abs_convergence_missing_targets.json"
    failed_json = analysis_root / "min_abs_convergence_failed.json"

    detail_json.write_text(json.dumps(detail_rows, indent=2), encoding="utf-8")
    _write_csv(detail_rows, detail_csv)

    summary_json_out.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    _write_csv(summary_rows, summary_csv_out)

    missing_json.write_text(json.dumps(missing_targets, indent=2), encoding="utf-8")
    failed_json.write_text(json.dumps(failed_rows, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in detail_rows if r.get("status") == "ok")
    n_dry = sum(1 for r in detail_rows if r.get("status") == "dry_run")
    print("\n" + "-" * 100)
    print(f"saved detail json: {detail_json}")
    print(f"saved detail csv: {detail_csv}")
    print(f"saved summary json: {summary_json_out}")
    print(f"saved summary csv: {summary_csv_out}")
    print(f"saved missing-targets json: {missing_json}")
    print(f"saved failed json: {failed_json}")
    print(f"detail rows: ok={n_ok}, dry_run={n_dry}, total={len(detail_rows)}")
    print(f"summary rows: {len(summary_rows)}")
    print(f"missing targets: {len(missing_targets)}")
    print(f"failed runs: {len(failed_rows)}")
    log_fp.flush()

    if len(failed_rows) > 0 and not bool(args.continue_on_error):
        raise RuntimeError("Validation failed for one or more runs.")


if __name__ == "__main__":
    main()
