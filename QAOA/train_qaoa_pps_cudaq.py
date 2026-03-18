from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os
from typing import Any, Dict, List, Optional, TextIO

# Ensure allocator handles fragmentation better (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
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

from src_tensor import api as _src_tensor_api
from qaoa_surrogate_common import (
    build_maxcut_observable,
    build_qaoa_circuit,
    build_qaoa_theta_init_tqa,
    expected_cut_from_sum_zz,
    load_edges_json,
    make_ring_chord_graph,
)

compile_expval_program = _src_tensor_api.compile_expval_program
_loaded_api = Path(_src_tensor_api.__file__).resolve()
if _REPO_ROOT not in _loaded_api.parents:
    raise ImportError(
        "Loaded src_tensor.api from unexpected location: "
        f"{_loaded_api} (expected under {_REPO_ROOT})"
    )


_EXACT_MAX_QUBITS = 20


def _choose_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(raw)


def _default_cpu_exact_overrides() -> Dict[str, object]:
    return {
        "memory_device": "cpu",
        "compute_device": "cpu",
        "dtype": "float64",
        "max_weight": 1_000_000_000,
        "weight_x": 1.0,
        "weight_y": 1.0,
        "weight_z": 1.0,
        "chunk_size": 1_000_000,
    }


def _resolve_weight_tuple(
    *,
    mode: str,
    weight_x: Optional[float],
    weight_y: Optional[float],
    weight_z: Optional[float],
) -> Dict[str, float]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "yz":
        base = {"weight_x": 0.0, "weight_y": 1.0, "weight_z": 1.0}
    elif mode_norm == "xyz":
        base = {"weight_x": 1.0, "weight_y": 1.0, "weight_z": 1.0}
    elif mode_norm == "custom":
        if weight_x is None or weight_y is None or weight_z is None:
            raise ValueError("weight-mode=custom requires --weight-x, --weight-y, --weight-z")
        base = {
            "weight_x": float(weight_x),
            "weight_y": float(weight_y),
            "weight_z": float(weight_z),
        }
    else:
        raise ValueError(f"Unknown weight-mode: {mode}")

    if weight_x is not None:
        base["weight_x"] = float(weight_x)
    if weight_y is not None:
        base["weight_y"] = float(weight_y)
    if weight_z is not None:
        base["weight_z"] = float(weight_z)
    return base


def _build_theta_init(
    *,
    p_layers: int,
    n_edges: int,
    n_qubits: int,
    delta_t: float,
    init_mode: str,
    mixer_odd_start: float,
    mixer_odd_end: float,
) -> np.ndarray:
    base = build_qaoa_theta_init_tqa(
        p_layers=int(p_layers),
        n_edges=int(n_edges),
        n_qubits=int(n_qubits),
        delta_t=float(delta_t),
        dtype=np.float64,
    )
    mode = str(init_mode).strip().lower()
    if mode == "tqa":
        return base
    if mode == "odd-linear-neg":
        if int(p_layers) < 1:
            raise ValueError("p_layers must be >= 1")
        start = float(mixer_odd_start)
        end = float(mixer_odd_end)
        if not (start < 0.0 and end < 0.0):
            raise ValueError("odd-linear-neg requires negative mixer_odd_start and mixer_odd_end")
        out = np.asarray(base, dtype=np.float64).copy()
        odd_vals = np.linspace(start, end, int(p_layers), dtype=np.float64)
        for layer in range(int(p_layers)):
            out[2 * layer + 1] = float(odd_vals[layer])
        return out
    raise ValueError(f"Unknown init-mode: {init_mode}")


def _extract_compile_resources(program: Any) -> Dict[str, Any]:
    psum = program.psum_union
    n_terms = int(psum.x_mask.shape[0])
    n_steps = int(len(psum.steps))

    nnz_const_total = 0
    nnz_cos_total = 0
    nnz_sin_total = 0
    implicit_same_total = 0
    implicit_anti_same_total = 0
    max_step_rows = 0
    max_step_cols = 0
    max_step_nnz_total = 0
    for step in psum.steps:
        n_const = int(step.mat_const._nnz())
        n_cos = int(step.mat_cos._nnz())
        n_sin = int(step.mat_sin._nnz())
        step_total = n_const + n_cos + n_sin

        nnz_const_total += n_const
        nnz_cos_total += n_cos
        nnz_sin_total += n_sin
        implicit_same_total += int(step.same_nnz())
        implicit_anti_same_total += int(step.anti_same_nnz())

        rows, cols = step.shape
        max_step_rows = max(max_step_rows, int(rows))
        max_step_cols = max(max_step_cols, int(cols))
        max_step_nnz_total = max(max_step_nnz_total, int(step_total))

    nnz_total = int(nnz_const_total + nnz_cos_total + nnz_sin_total)
    effective_work_total = int(nnz_total + implicit_same_total + implicit_anti_same_total)
    density_proxy = float(nnz_total) / float(max(1, n_steps * max_step_rows * max_step_cols))

    return {
        "terms_after_zero_filter": int(n_terms),
        "n_steps": int(n_steps),
        "nnz_const_total": int(nnz_const_total),
        "nnz_cos_total": int(nnz_cos_total),
        "nnz_sin_total": int(nnz_sin_total),
        "nnz_total": int(nnz_total),
        "implicit_same_total": int(implicit_same_total),
        "implicit_anti_same_total": int(implicit_anti_same_total),
        "gpu_work_proxy_nnz_total": int(nnz_total),
        "effective_work_proxy_total": int(effective_work_total),
        "max_step_rows": int(max_step_rows),
        "max_step_cols": int(max_step_cols),
        "max_step_nnz_total": int(max_step_nnz_total),
        "nnz_density_proxy": float(density_proxy),
    }


def _try_cudaq_sample(
    *,
    n_qubits: int,
    edges,
    p_layers: int,
    thetas: np.ndarray,
    shots: int,
    seed: Optional[int],
) -> Optional[Dict[str, Any]]:
    try:
        import cudaq  # type: ignore
    except Exception as e:
        print(f"cudaq is not available: {e}")
        return None

    thetas = np.asarray(thetas, dtype=np.float64).reshape(-1)

    @cudaq.kernel
    def qaoa_kernel(params: list[float]):
        q = cudaq.qvector(n_qubits)
        for i in range(n_qubits):
            cudaq.h(q[i])
        for layer in range(p_layers):
            gamma_idx = 2 * int(layer)
            beta_idx = 2 * int(layer) + 1
            for (u, v) in edges:
                cudaq.rzz(params[gamma_idx], q[u], q[v])
            for i in range(n_qubits):
                cudaq.rx(params[beta_idx], q[i])

    if seed is not None:
        cudaq.set_random_seed(int(seed))

    counts = cudaq.sample(qaoa_kernel, list(thetas.tolist()), shots=int(shots))
    data = {str(k): int(v) for k, v in counts.items()}
    return {
        "shots": int(shots),
        "n_qubits": int(n_qubits),
        "counts": data,
    }


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


def _compile_with_retry(
    circuit,
    obs,
    thetas,
    preset,
    preset_overrides,
    build_min_abs,
    build_min_mat_abs,
    max_retries=3,
    min_chunk_size=1_000_000,
):
    """Compile with adaptive chunk size to avoid CUDA OOM."""
    last_err = None
    overrides = dict(preset_overrides or {})
    cs = int(overrides.get("chunk_size", 0))
    cur_min_abs = build_min_abs
    for attempt in range(max_retries):
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            if cs > 0:
                overrides["chunk_size"] = int(cs)

            program = compile_expval_program(
                circuit=circuit,
                observables=[obs],
                preset=preset,
                preset_overrides=overrides,
                build_thetas=thetas,
                build_min_abs=cur_min_abs,
                build_min_mat_abs=build_min_mat_abs,
            )
            return program, cs
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if cs <= 0:
                base = int(overrides.get("chunk_size", 0))
                cs = max(min_chunk_size, base // 2) if base > 0 else int(min_chunk_size)
            else:
                cs = max(min_chunk_size, cs // 2)
            if cur_min_abs is not None:
                cur_min_abs = float(cur_min_abs) * 2.0
            print(
                f"[compile retry {attempt + 1}/{max_retries}] CUDA OOM -> "
                f"chunk_size={cs}, build_min_abs={cur_min_abs}"
            )
    if last_err is not None:
        raise last_err
    raise RuntimeError("compile_expval_program failed without a captured OOM exception")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaxCut-QAOA with PPS (tensor surrogate) and sample via cudaq.")
    p.add_argument("--edges-json", type=str, default="", help="Edge list JSON [[u,v], ...].")
    p.add_argument("--n-qubits", type=int, default=30)
    p.add_argument("--p-layers", type=int, default=6)
    p.add_argument("--delta-t", type=float, default=0.8)
    p.add_argument(
        "--init-mode",
        type=str,
        default="tqa",
        choices=["tqa", "odd-linear-neg"],
        help="Theta initialization mode.",
    )
    p.add_argument(
        "--mixer-odd-start",
        type=float,
        default=-1.0,
        help="When init-mode=odd-linear-neg, odd-index mixer start value.",
    )
    p.add_argument(
        "--mixer-odd-end",
        type=float,
        default=-0.05,
        help="When init-mode=odd-linear-neg, odd-index mixer end value (last layer).",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chord-shift", type=int, default=7)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--preset", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--log-every", type=int, default=1)

    p.add_argument("--build-min-abs", type=float, default=1e-3)
    p.add_argument("--build-min-mat-abs", type=float, default=None)
    p.add_argument(
        "--no-build-min-abs",
        action="store_true",
        help="Disable coefficient pruning by min_abs during compile/rebuild.",
    )
    p.add_argument("--chunk-size", type=int, default=0, help="Override PPS chunk_size (0 uses preset).")
    p.add_argument(
        "--rebuild-every",
        type=int,
        default=0,
        help="Recompile PPS program every N steps (0 disables).",
    )
    p.add_argument("--max-weight", type=int, default=1_000_000_000)
    p.add_argument("--weight-mode", type=str, default="yz", choices=["yz", "xyz", "custom"])
    p.add_argument("--weight-x", type=float, default=None)
    p.add_argument("--weight-y", type=float, default=None)
    p.add_argument("--weight-z", type=float, default=None)

    p.add_argument("--output-dir", type=str, default="QAOA/artifacts")
    p.add_argument("--run-name", type=str, default="qaoa_maxcut_30q")
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Realtime log file path. If empty, defaults to <output-dir>/<run-name>.log",
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
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint (.pt) to initialize thetas from (best_thetas or thetas).",
    )
    p.add_argument(
        "--resume-mode",
        type=str,
        default="init",
        choices=["init", "build", "both"],
        help=(
            "How to use --resume checkpoint: "
            "'init' initializes trainable theta, "
            "'build' uses resume theta only for initial PPS compile build_thetas, "
            "'both' applies both behaviors."
        ),
    )
    p.add_argument(
        "--init-noise-std",
        type=float,
        default=0.0,
        help="Add Gaussian noise N(0, std^2) to initial trainable theta after init/resume selection.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save parameters every N steps (0 disables).",
    )

    p.add_argument("--cudaq-sample", action="store_true", help="Run cudaq sampling after training.")
    p.add_argument("--shots", type=int, default=4000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    if str(args.edges_json).strip():
        edges = load_edges_json(args.edges_json)
        graph_source = "edges_json"
        graph_params = {"edges_json": str(args.edges_json)}
    else:
        edges = make_ring_chord_graph(int(args.n_qubits), chord_shift=int(args.chord_shift))
        graph_source = "ring_chord"
        graph_params = {"chord_shift": int(args.chord_shift)}

    m_edges = len(edges)
    if m_edges < 1:
        raise ValueError("Graph must contain at least one edge.")

    circuit, n_params = build_qaoa_circuit(
        n_qubits=int(args.n_qubits),
        edges=edges,
        p_layers=int(args.p_layers),
    )
    zz_obj = build_maxcut_observable(n_qubits=int(args.n_qubits), edges=edges)

    init_theta_np = _build_theta_init(
        p_layers=int(args.p_layers),
        n_edges=int(m_edges),
        n_qubits=int(args.n_qubits),
        delta_t=float(args.delta_t),
        init_mode=str(args.init_mode),
        mixer_odd_start=float(args.mixer_odd_start),
        mixer_odd_end=float(args.mixer_odd_end),
    )
    if int(init_theta_np.shape[0]) != int(n_params):
        raise RuntimeError("TQA init size mismatch.")

    run_device = _choose_device(str(args.device))
    if run_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    preset_raw = str(args.preset).strip().lower()
    if preset_raw in ("auto", "gpu"):
        preset = "hybrid"
    elif preset_raw == "cpu":
        preset = "cpu"
    else:
        raise ValueError(f"Unsupported preset: {args.preset}")

    weight_tuple = _resolve_weight_tuple(
        mode=str(args.weight_mode),
        weight_x=args.weight_x,
        weight_y=args.weight_y,
        weight_z=args.weight_z,
    )

    preset_overrides: Optional[Dict[str, object]] = None
    if run_device == "cpu" and preset == "cpu":
        preset_overrides = _default_cpu_exact_overrides()
    if preset_overrides is None:
        preset_overrides = {}
    if int(args.chunk_size) > 0:
        preset_overrides["chunk_size"] = int(args.chunk_size)
    preset_overrides["max_weight"] = int(args.max_weight)
    preset_overrides["weight_x"] = float(weight_tuple["weight_x"])
    preset_overrides["weight_y"] = float(weight_tuple["weight_y"])
    preset_overrides["weight_z"] = float(weight_tuple["weight_z"])

    resume_thetas_np: Optional[np.ndarray] = None
    resume_mode = str(args.resume_mode).strip().lower()
    if str(args.resume).strip():
        resume_path = Path(args.resume)
        payload = torch.load(resume_path, map_location="cpu")
        if "best_thetas" in payload:
            resume_thetas_np = payload["best_thetas"].detach().cpu().numpy()
        elif "thetas" in payload:
            resume_thetas_np = payload["thetas"].detach().cpu().numpy()
        else:
            raise KeyError("Resume checkpoint missing 'best_thetas' or 'thetas'.")
        if int(resume_thetas_np.shape[0]) != int(n_params):
            raise RuntimeError(
                f"Resume theta size mismatch: {int(resume_thetas_np.shape[0])} vs n_params={int(n_params)}"
            )
    elif resume_mode in ("build", "both"):
        print(
            f"[warn] resume-mode={resume_mode} requested but --resume is empty. "
            "Falling back to non-resume initialization/build."
        )

    if resume_thetas_np is not None and resume_mode in ("init", "both"):
        init_theta_np = np.asarray(resume_thetas_np, dtype=np.float64).copy()

    init_noise_std = float(args.init_noise_std)
    if init_noise_std > 0.0:
        init_theta_np = np.asarray(init_theta_np, dtype=np.float64) + np.random.normal(
            loc=0.0, scale=init_noise_std, size=init_theta_np.shape
        )

    thetas = torch.nn.Parameter(torch.tensor(init_theta_np, dtype=torch.float64, device=run_device))
    build_min_abs: Optional[float] = None if bool(args.no_build_min_abs) else float(args.build_min_abs)

    # Compile program (adaptive retry to avoid OOM)
    if resume_thetas_np is not None and resume_mode in ("build", "both"):
        compile_thetas = torch.as_tensor(resume_thetas_np, dtype=torch.float64, device="cpu")
    else:
        compile_thetas = thetas.detach().cpu()
    program, used_chunk = _compile_with_retry(
        circuit=circuit,
        obs=zz_obj,
        thetas=compile_thetas,
        preset=preset,
        preset_overrides=preset_overrides,
        build_min_abs=build_min_abs,
        build_min_mat_abs=args.build_min_mat_abs,
    )
    compile_events: List[Dict[str, Any]] = []
    first_resources = _extract_compile_resources(program)
    compile_events.append({"step": -1, "chunk_size": int(used_chunk), **first_resources})
    print(
        "[resource] compile step=-1 "
        f"terms={first_resources['terms_after_zero_filter']:,} "
        f"nnz_total={first_resources['nnz_total']:,}"
    )

    opt = torch.optim.Adam([thetas], lr=float(args.lr))
    best_val = float("inf")
    best_thetas = None
    history = []
    exact_enabled = int(args.n_qubits) <= int(_EXACT_MAX_QUBITS)
    history_exact: Optional[List[float]] = [] if exact_enabled else None
    best_exact_at_best_surrogate: Optional[float] = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(str(args.log_file)).resolve()
        if str(args.log_file).strip()
        else (out_dir / f"{args.run_name}.log").resolve()
    )
    log_fp = _setup_realtime_log(
        log_path=log_path,
        append=bool(args.log_append),
        mirror_terminal=bool(args.log_to_terminal),
    )
    step_dir = out_dir / f"{args.run_name}_steps"
    if int(args.save_every) > 0:
        step_dir.mkdir(parents=True, exist_ok=True)

    for step in range(int(args.steps)):
        if int(args.rebuild_every) > 0 and step > 0 and (step % int(args.rebuild_every) == 0):
            compile_thetas = thetas.detach().cpu()
            program, used_chunk = _compile_with_retry(
                circuit=circuit,
                obs=zz_obj,
                thetas=compile_thetas,
                preset=preset,
                preset_overrides=preset_overrides,
                build_min_abs=build_min_abs,
                build_min_mat_abs=args.build_min_mat_abs,
            )
            res = _extract_compile_resources(program)
            compile_events.append({"step": int(step), "chunk_size": int(used_chunk), **res})
            print(
                f"[resource] compile step={step} "
                f"terms={res['terms_after_zero_filter']:,} nnz_total={res['nnz_total']:,}"
            )
        opt.zero_grad(set_to_none=True)
        zz_val = program.expval(
            thetas,
            obs_index=0,
        )
        zz_val.backward()
        opt.step()

        val = float(zz_val.detach().cpu().item())
        history.append(val)
        exact_sum_zz: Optional[float] = None
        if exact_enabled:
            exact_sum_zz = float(
                program.expvals_pennylane(
                    thetas.detach().cpu(),
                    max_qubits=int(_EXACT_MAX_QUBITS),
                )[0].item()
            )
            assert history_exact is not None
            history_exact.append(exact_sum_zz)

        if val < best_val:
            best_val = val
            best_thetas = thetas.detach().cpu().clone()
            best_exact_at_best_surrogate = exact_sum_zz

        if int(args.save_every) > 0 and (step % int(args.save_every) == 0):
            step_path = step_dir / f"step_{step:06d}.pt"
            step_payload: Dict[str, Any] = {"thetas": thetas.detach().cpu()}
            if exact_sum_zz is not None:
                step_payload["exact_sum_zz"] = float(exact_sum_zz)
            torch.save(step_payload, step_path)

        if (step % int(args.log_every) == 0) or (step == int(args.steps) - 1):
            exp_cut = expected_cut_from_sum_zz(val, m_edges)
            if exact_sum_zz is None:
                print(f"step={step:04d} sum<ZZ>={val:+.8f} E[cut]={exp_cut:.6f}")
            else:
                exact_cut = expected_cut_from_sum_zz(exact_sum_zz, m_edges)
                abs_err = abs(val - exact_sum_zz)
                print(
                    f"step={step:04d} sum<ZZ>={val:+.8f} E[cut]={exp_cut:.6f} "
                    f"exact_sum<ZZ>={exact_sum_zz:+.8f} exact_E[cut]={exact_cut:.6f} "
                    f"|err|={abs_err:.6e}"
                )

    if best_thetas is None:
        raise RuntimeError("Training produced no best_thetas.")

    ckpt_path = out_dir / f"{args.run_name}.pt"
    torch.save({"best_thetas": best_thetas, "final_thetas": thetas.detach().cpu()}, ckpt_path)

    report = {
        "config": {
            "n_qubits": int(args.n_qubits),
            "p_layers": int(args.p_layers),
            "delta_t": float(args.delta_t),
            "init_mode": str(args.init_mode),
            "mixer_odd_start": float(args.mixer_odd_start),
            "mixer_odd_end": float(args.mixer_odd_end),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "device": run_device,
            "preset": preset,
            "build_min_abs": build_min_abs,
            "no_build_min_abs": bool(args.no_build_min_abs),
            "build_min_mat_abs": args.build_min_mat_abs,
            "rebuild_every": int(args.rebuild_every),
            "chunk_size": int(args.chunk_size),
            "max_weight": int(args.max_weight),
            "weight_mode": str(args.weight_mode),
            "weight_x": float(weight_tuple["weight_x"]),
            "weight_y": float(weight_tuple["weight_y"]),
            "weight_z": float(weight_tuple["weight_z"]),
            "log_file": str(log_path),
            "log_append": bool(args.log_append),
            "log_to_terminal": bool(args.log_to_terminal),
            "resume": str(args.resume),
            "resume_mode": str(args.resume_mode),
            "init_noise_std": float(args.init_noise_std),
            "resume_applied_to_init": bool(resume_thetas_np is not None and resume_mode in ("init", "both")),
            "resume_applied_to_build": bool(resume_thetas_np is not None and resume_mode in ("build", "both")),
        },
        "graph": {
            "source": graph_source,
            "params": graph_params,
            "edges": [[int(u), int(v)] for (u, v) in edges],
        },
        "training": {
            "history_sum_zz": [float(v) for v in history],
            "history_expected_cut": [expected_cut_from_sum_zz(v, m_edges) for v in history],
            "best_sum_zz": float(best_val),
            "best_expected_cut": float(expected_cut_from_sum_zz(best_val, m_edges)),
            "exact_enabled": bool(exact_enabled),
            "exact_max_qubits": int(_EXACT_MAX_QUBITS),
            "history_exact_sum_zz": ([float(v) for v in history_exact] if history_exact is not None else None),
            "history_exact_expected_cut": (
                [expected_cut_from_sum_zz(v, m_edges) for v in history_exact] if history_exact is not None else None
            ),
            "best_exact_sum_zz_at_best_surrogate": (
                float(best_exact_at_best_surrogate) if best_exact_at_best_surrogate is not None else None
            ),
            "best_exact_expected_cut_at_best_surrogate": (
                float(expected_cut_from_sum_zz(best_exact_at_best_surrogate, m_edges))
                if best_exact_at_best_surrogate is not None
                else None
            ),
        },
        "resources": {
            "compile_events": compile_events,
            "terms_after_zero_filter_last": int(compile_events[-1]["terms_after_zero_filter"]),
            "nnz_total_last": int(compile_events[-1]["nnz_total"]),
            "gpu_work_proxy_nnz_total_last": int(compile_events[-1]["gpu_work_proxy_nnz_total"]),
            "terms_after_zero_filter_mean": float(
                sum(int(x["terms_after_zero_filter"]) for x in compile_events) / max(1, len(compile_events))
            ),
            "nnz_total_mean": float(sum(int(x["nnz_total"]) for x in compile_events) / max(1, len(compile_events))),
            "nnz_total_std": float(
                np.std(np.asarray([int(x["nnz_total"]) for x in compile_events], dtype=np.float64))
            ),
            "rebuild_count": int(max(0, len(compile_events) - 1)),
        },
        "checkpoint": str(ckpt_path),
    }

    if args.cudaq_sample:
        sample = _try_cudaq_sample(
            n_qubits=int(args.n_qubits),
            edges=edges,
            p_layers=int(args.p_layers),
            thetas=best_thetas.detach().cpu().numpy(),
            shots=int(args.shots),
            seed=int(args.seed),
        )
        report["cudaq_sampling"] = sample

    report_path = out_dir / f"{args.run_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved report: {report_path}")
    log_fp.flush()


if __name__ == "__main__":
    main()
