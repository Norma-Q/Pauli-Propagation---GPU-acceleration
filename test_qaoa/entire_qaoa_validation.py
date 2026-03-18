from __future__ import annotations

import argparse
import ctypes
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from qaoa_surrogate_common import (
    build_maxcut_observable,
    build_qaoa_circuit,
    expected_cut_from_sum_zz,
    load_edges_json,
    parse_min_abs_schedule,
)
from src_tensor.api import compile_expval_program, evaluate_expval_direct

try:
    from src_tensor.tensor_eval_only_impl import CPPEvalOnlyBackendUnavailableError
except Exception:  # pragma: no cover - optional backend
    class CPPEvalOnlyBackendUnavailableError(RuntimeError):
        pass


DEFAULT_THRESHOLDS = [
    1e-1,
    5e-2,
    1e-2,
    5e-3,
    1e-3,
    5e-4,
    1e-4,
    5e-5,
]
DEFAULT_CUDA_CHUNK_SIZE = 5_000_000
DEFAULT_CPU_CHUNK_SIZE = 1_000_000


def _cleanup_memory(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate trained QAOA parameters by decreasing min_abs thresholds."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the config yaml used by entire_qaoa_process.py",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="",
        help="Optional explicit result directory. Defaults to results/Q{n}_L{p}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(f"{x:.0e}" for x in DEFAULT_THRESHOLDS),
        help="Comma-separated min_abs schedule in the order to evaluate.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="auto",
        choices=["auto", "direct", "compile"],
        help="auto tries direct-eval first and falls back to compile only if direct backend is unavailable.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Optional chunk size override. 0 uses a device-specific default.",
    )
    parser.add_argument(
        "--stop-after-consecutive-failures",
        type=int,
        default=1,
        help="Stop early after this many consecutive failed thresholds. Smaller min_abs is typically harder.",
    )
    parser.add_argument(
        "--show-direct-progress",
        action="store_true",
        help="Show per-gate progress bar for direct evaluate-only mode.",
    )
    return parser.parse_args()


def _choose_device(raw: str) -> str:
    text = str(raw).strip().lower()
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to cpu.")
        return "cpu"
    return str(raw)


def _parse_thresholds(raw: str) -> List[float]:
    vals = parse_min_abs_schedule(str(raw))
    return [float(v) for v in vals]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("config yaml must be a mapping")
    return data


def _resolve_paths(
    *,
    config_path: Path,
    n_qubits: int,
    p_layers: int,
    result_dir_override: str,
) -> Tuple[Path, Path]:
    qaoa_root = config_path.resolve().parent.parent
    if str(result_dir_override).strip() != "":
        result_dir = Path(result_dir_override).expanduser().resolve()
    else:
        result_dir = (qaoa_root / "results" / f"Q{n_qubits}_L{p_layers}").resolve()
    graph_path = (qaoa_root / "graph" / f"Q{n_qubits}_edges.json").resolve()
    return result_dir, graph_path


def _load_training_payload(result_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates = [
        result_dir / "training_log.json",
        result_dir / "training.json",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        raise FileNotFoundError(
            f"No training file found in {result_dir}. Checked: {[str(x) for x in candidates]}"
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training payload must be a JSON object: {target}")
    return target, payload


def _extract_trained_thetas(payload: Dict[str, Any], target_path: Path) -> np.ndarray:
    if "trained_thetas" not in payload:
        raise KeyError(f"'trained_thetas' not found in {target_path}")
    return np.asarray(payload["trained_thetas"], dtype=np.float64)


def _resolve_max_weight(qaoa_cfg: Dict[str, Any], training_payload: Dict[str, Any]) -> int:
    if "max_weight" in training_payload:
        return int(training_payload["max_weight"])
    if "max_weight" in qaoa_cfg:
        return int(qaoa_cfg["max_weight"])
    return 3


def _build_preset_and_overrides(
    *,
    device: str,
    max_weight: int,
    chunk_size_override: int,
) -> Tuple[str, Dict[str, Any]]:
    if str(device).startswith("cuda"):
        preset = "hybrid"
        chunk_size = int(chunk_size_override) if int(chunk_size_override) > 0 else DEFAULT_CUDA_CHUNK_SIZE
        overrides = {
            "max_weight": int(max_weight),
            "compute_device": str(device),
            "chunk_size": int(chunk_size),
        }
    else:
        preset = "cpu"
        chunk_size = int(chunk_size_override) if int(chunk_size_override) > 0 else DEFAULT_CPU_CHUNK_SIZE
        overrides = {
            "max_weight": int(max_weight),
            "compute_device": "cpu",
            "chunk_size": int(chunk_size),
        }
    return preset, overrides


def _extract_compile_resources(program: Any) -> Dict[str, Any]:
    psum = program.psum_union
    nnz_total = 0
    for step in psum.steps:
        nnz_total += int(step.mat_const._nnz())
        nnz_total += int(step.mat_cos._nnz())
        nnz_total += int(step.mat_sin._nnz())
    return {
        "terms_after_zero_filter": int(psum.x_mask.shape[0]),
        "n_steps": int(len(psum.steps)),
        "nnz_total": int(nnz_total),
    }


def _is_direct_backend_unavailable(exc: Exception) -> bool:
    if isinstance(exc, CPPEvalOnlyBackendUnavailableError):
        return True
    text = str(exc)
    return "Direct evaluate-only backend is not available" in text


def _evaluate_threshold_direct(
    *,
    circuit: List[Any],
    zz_obj: Any,
    thetas_t: torch.Tensor,
    preset: str,
    preset_overrides: Dict[str, Any],
    min_abs: float,
    show_progress: bool,
) -> Dict[str, Any]:
    sum_zz_t = evaluate_expval_direct(
        circuit=circuit,
        observables=[zz_obj],
        thetas=thetas_t,
        preset=preset,
        preset_overrides=preset_overrides,
        min_abs=float(min_abs),
        show_progress=bool(show_progress),
    )
    sum_zz = float(sum_zz_t.detach().cpu().item())
    return {
        "method": "direct_eval",
        "sum_zz": float(sum_zz),
    }


def _evaluate_threshold_compile(
    *,
    circuit: List[Any],
    zz_obj: Any,
    thetas_t: torch.Tensor,
    preset: str,
    preset_overrides: Dict[str, Any],
    min_abs: float,
) -> Dict[str, Any]:
    program = compile_expval_program(
        circuit=circuit,
        observables=[zz_obj],
        preset=preset,
        preset_overrides=preset_overrides,
        build_thetas=thetas_t,
        build_min_abs=float(min_abs),
        parallel_compile=False,
    )
    with torch.no_grad():
        sum_zz = float(program.expval(thetas_t, obs_index=0).detach().cpu().item())
    resources = _extract_compile_resources(program)
    del program
    return {
        "method": "compile_program",
        "sum_zz": float(sum_zz),
        "compile_resources": resources,
    }


def _plot_validation(output_path: Path, rows: List[Dict[str, Any]], run_label: str) -> None:
    thresholds = np.asarray([float(row["min_abs"]) for row in rows], dtype=np.float64)
    values = np.asarray(
        [float(row["expected_cut"]) if bool(row["ok"]) else np.nan for row in rows],
        dtype=np.float64,
    )
    ok_mask = np.asarray([bool(row["ok"]) for row in rows], dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)

    ax = axes[0]
    if np.any(ok_mask):
        ax.plot(thresholds[ok_mask], values[ok_mask], marker="o", linewidth=1.8, color="tab:blue")
    if np.any(~ok_mask):
        fail_y = np.nanmin(values[ok_mask]) if np.any(ok_mask) else 0.0
        ax.scatter(
            thresholds[~ok_mask],
            np.full(np.count_nonzero(~ok_mask), fail_y),
            marker="x",
            s=60,
            color="tab:red",
            label="failed",
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylabel("Expected cut")
    ax.set_title("Validation sweep over build_min_abs")
    ax.grid(True, alpha=0.3)
    if np.any(~ok_mask):
        ax.legend(loc="best")

    ax2 = axes[1]
    ok_thresholds = thresholds[ok_mask]
    ok_values = values[ok_mask]
    if ok_thresholds.size >= 2:
        delta_x = ok_thresholds[1:]
        delta_y = np.abs(np.diff(ok_values))
        ax2.plot(delta_x, delta_y, marker="o", linewidth=1.6, color="tab:orange")
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("build_min_abs")
    ax2.set_ylabel(r"Adjacent $|\Delta E[\mathrm{cut}]|$")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(run_label, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_progress(
    *,
    result_dir: Path,
    rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    thresholds_done = [float(row["min_abs"]) for row in rows]
    expected_cuts = [float(row["expected_cut"]) if bool(row["ok"]) else float("nan") for row in rows]
    ok_mask = [bool(row["ok"]) for row in rows]

    payload: Dict[str, Any] = {
        **metadata,
        "thresholds_done": thresholds_done,
        "expected_cuts": expected_cuts,
        "ok_mask": ok_mask,
        "completed_count": int(len(rows)),
        "rows": rows,
    }
    state_path = result_dir / "validation_progress.json"
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run_label = f"Q{metadata['n_qubits']} / L={metadata['p_layers']} / mw={metadata['max_weight']}"
    _plot_validation(result_dir / "validation.png", rows, run_label)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)

    qaoa_cfg = dict(config.get("QAOA", {}))
    n_qubits = int(qaoa_cfg["n_qubits"])
    p_layers = int(qaoa_cfg["n_layers"])
    thresholds = _parse_thresholds(args.thresholds)
    device = _choose_device(args.device)

    result_dir, graph_path = _resolve_paths(
        config_path=config_path,
        n_qubits=n_qubits,
        p_layers=p_layers,
        result_dir_override=args.result_dir,
    )
    if not result_dir.exists():
        raise FileNotFoundError(f"result directory not found: {result_dir}")
    if not graph_path.exists():
        raise FileNotFoundError(f"graph file not found: {graph_path}")

    training_path, training_payload = _load_training_payload(result_dir)
    trained_thetas_np = _extract_trained_thetas(training_payload, training_path)
    max_weight = _resolve_max_weight(qaoa_cfg, training_payload)
    preset, preset_overrides = _build_preset_and_overrides(
        device=device,
        max_weight=max_weight,
        chunk_size_override=int(args.chunk_size),
    )

    edges = load_edges_json(graph_path)
    circuit, _ = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=p_layers)
    zz_obj = build_maxcut_observable(n_qubits=n_qubits, edges=edges)
    thetas_t = torch.as_tensor(trained_thetas_np, dtype=torch.float64)

    if str(device).startswith("cuda"):
        thetas_t = thetas_t.to(device)

    rows: List[Dict[str, Any]] = []
    eval_mode_runtime = str(args.eval_mode)
    consecutive_failures = 0
    stop_reason: Optional[str] = None

    metadata: Dict[str, Any] = {
        "config_path": str(config_path),
        "result_dir": str(result_dir),
        "graph_path": str(graph_path),
        "training_log_path": str(training_path),
        "n_qubits": int(n_qubits),
        "p_layers": int(p_layers),
        "n_edges": int(len(edges)),
        "max_weight": int(max_weight),
        "device": str(device),
        "preset": str(preset),
        "preset_overrides": dict(preset_overrides),
        "requested_thresholds": [float(x) for x in thresholds],
        "eval_mode_requested": str(args.eval_mode),
        "eval_mode_runtime": str(eval_mode_runtime),
        "stopped_early": False,
        "stop_reason": None,
    }

    print(f"[info] validating folder: {result_dir}")
    print(f"[info] training file: {training_path.name}")
    print(f"[info] device={device} preset={preset} max_weight={max_weight}")
    print(f"[info] thresholds={thresholds}")

    total = len(thresholds)
    for idx, min_abs in enumerate(thresholds, start=1):
        attempted_mode = eval_mode_runtime
        print(f"\n[START {idx}/{total}] min_abs={min_abs:.1e} mode={attempted_mode}")

        try:
            if eval_mode_runtime == "compile":
                out = _evaluate_threshold_compile(
                    circuit=circuit,
                    zz_obj=zz_obj,
                    thetas_t=thetas_t,
                    preset=preset,
                    preset_overrides=preset_overrides,
                    min_abs=float(min_abs),
                )
            else:
                try:
                    out = _evaluate_threshold_direct(
                        circuit=circuit,
                        zz_obj=zz_obj,
                        thetas_t=thetas_t,
                        preset=preset,
                        preset_overrides=preset_overrides,
                        min_abs=float(min_abs),
                        show_progress=bool(args.show_direct_progress),
                    )
                    eval_mode_runtime = "direct"
                except Exception as direct_exc:
                    if eval_mode_runtime == "auto" and _is_direct_backend_unavailable(direct_exc):
                        print("[warn] direct evaluate-only backend unavailable; falling back to compile mode.")
                        out = _evaluate_threshold_compile(
                            circuit=circuit,
                            zz_obj=zz_obj,
                            thetas_t=thetas_t,
                            preset=preset,
                            preset_overrides=preset_overrides,
                            min_abs=float(min_abs),
                        )
                        out["method"] = "compile_fallback"
                        eval_mode_runtime = "compile"
                    else:
                        raise

            expected_cut = expected_cut_from_sum_zz(out["sum_zz"], n_edges=len(edges))
            row: Dict[str, Any] = {
                "min_abs": float(min_abs),
                "sum_zz": float(out["sum_zz"]),
                "expected_cut": float(expected_cut),
                "method": str(out["method"]),
                "ok": True,
                "status": "ok",
            }
            if "compile_resources" in out:
                row["compile_resources"] = dict(out["compile_resources"])
            rows.append(row)
            consecutive_failures = 0
            print(
                f"  [OK] min_abs={min_abs:.1e} | method={row['method']} "
                f"| expected_cut={expected_cut:.6f}"
            )
        except Exception as exc:
            row = {
                "min_abs": float(min_abs),
                "sum_zz": float("nan"),
                "expected_cut": float("nan"),
                "method": str(attempted_mode),
                "ok": False,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            rows.append(row)
            consecutive_failures += 1
            print(f"  [FAIL] min_abs={min_abs:.1e} | {type(exc).__name__}: {exc}")
            if int(args.stop_after_consecutive_failures) > 0 and consecutive_failures >= int(args.stop_after_consecutive_failures):
                remaining = [float(x) for x in thresholds[idx:]]
                stop_reason = (
                    f"Stopped after {consecutive_failures} consecutive failure(s). "
                    f"Skipped smaller thresholds: {remaining}"
                )
                print(f"[stop] {stop_reason}")
                metadata["stopped_early"] = True
                metadata["stop_reason"] = stop_reason
                metadata["remaining_thresholds_skipped"] = remaining
                _cleanup_memory(device)
                _save_progress(result_dir=result_dir, rows=rows, metadata=metadata)
                break

        metadata["eval_mode_runtime"] = str(eval_mode_runtime)
        _cleanup_memory(device)
        _save_progress(result_dir=result_dir, rows=rows, metadata=metadata)

    if stop_reason is None:
        metadata["stopped_early"] = False
        metadata["stop_reason"] = None
        metadata["remaining_thresholds_skipped"] = []
        _save_progress(result_dir=result_dir, rows=rows, metadata=metadata)

    print(f"\n[done] saved: {result_dir / 'validation_progress.json'}")
    print(f"[done] saved: {result_dir / 'validation.png'}")


if __name__ == "__main__":
    main()
