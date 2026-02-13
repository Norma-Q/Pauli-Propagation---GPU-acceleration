from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import json

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_tensor.api import compile_expval_program

from qaoa_surrogate_common import (
    build_maxcut_observable,
    build_qaoa_circuit,
    build_qaoa_theta_init_tqa,
    expected_cut_from_sum_zz,
    load_edges_json,
    load_qaoa_problem_json,
    make_ring_chord_graph,
)


def _choose_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(raw)


def _default_cpu_exact_overrides() -> Dict[str, object]:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaxCut-QAOA with tensor surrogate backend.")
    p.add_argument("--problem-json", type=str, default="", help="QAOA problem definition JSON.")
    p.add_argument("--n-qubits", type=int, default=15)
    p.add_argument("--p-layers", type=int, default=4)
    p.add_argument("--delta-t", type=float, default=0.8, help="TQA initialization delta_t.")
    p.add_argument(
        "--delta-t-sweep",
        type=str,
        default="",
        help="Optional comma-separated sweep list for TQA delta_t (overrides --delta-t/problem init).",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--edges-json", type=str, default="", help="Optional path to JSON edge list [[u,v], ...].")
    p.add_argument("--chord-shift", type=int, default=7, help="Used only when --edges-json is not set.")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--preset", type=str, default="auto", choices=["auto", "gpu_min", "gpu_full"])
    p.add_argument("--build-min-abs", type=float, default=1e-4)
    p.add_argument("--build-min-mat-abs", type=float, default=None)
    p.add_argument("--log-every", type=int, default=25)

    p.add_argument("--output", type=str, default="test/artifacts/qaoa_train.pt")
    p.add_argument(
        "--output-json",
        type=str,
        default="auto",
        help="JSON report path. 'auto' writes next to --output with .json suffix.",
    )
    p.add_argument(
        "--save-per-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When sweeping delta_t, save per-delta checkpoints/reports in addition to the selected best run.",
    )
    return p.parse_args()


def _degree_stats(n_qubits: int, edges) -> Dict[str, object]:
    deg = [0 for _ in range(int(n_qubits))]
    for u, v in edges:
        deg[int(u)] += 1
        deg[int(v)] += 1
    arr = np.asarray(deg, dtype=np.int64)
    return {
        "min_degree": int(np.min(arr)),
        "max_degree": int(np.max(arr)),
        "mean_degree": float(np.mean(arr)),
        "degree_histogram": {str(k): int(np.sum(arr == k)) for k in sorted(set(arr.tolist()))},
    }


def _parse_delta_t_sweep(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        return []
    if any(v <= 0.0 for v in vals):
        raise ValueError("All delta_t values in --delta-t-sweep must be > 0.")
    return vals


def _delta_t_tag(delta_t: float) -> str:
    s = f"{float(delta_t):.6g}".lower()
    s = s.replace("-", "m").replace("+", "").replace(".", "p")
    return s


def _train_one_delta_t(
    *,
    delta_t: float,
    n_qubits: int,
    p_layers: int,
    m_edges: int,
    n_params: int,
    circuit,
    zz_obj,
    run_device: str,
    preset: str,
    preset_overrides: Optional[Dict[str, object]],
    build_min_abs: float,
    build_min_mat_abs: Optional[float],
    steps: int,
    lr: float,
    seed: int,
    log_every: int,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    init_theta_np = build_qaoa_theta_init_tqa(
        p_layers=int(p_layers),
        n_edges=int(m_edges),
        n_qubits=int(n_qubits),
        delta_t=float(delta_t),
        dtype=np.float64,
    )
    if int(init_theta_np.shape[0]) != int(n_params):
        raise RuntimeError(f"TQA init size mismatch: {int(init_theta_np.shape[0])} vs n_params={int(n_params)}")

    thetas = torch.nn.Parameter(torch.tensor(init_theta_np, dtype=torch.float64, device=run_device))
    program = compile_expval_program(
        circuit=circuit,
        observables=[zz_obj],
        preset=preset,
        preset_overrides=preset_overrides,
        build_min_abs=float(build_min_abs),
        build_min_mat_abs=build_min_mat_abs,
        build_thetas=thetas,
    )

    opt = torch.optim.Adam([thetas], lr=float(lr))
    best_val = float("inf")
    best_thetas = None
    history = []
    step_log = []

    stream_device = run_device
    offload_back = bool(run_device.startswith("cuda"))
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        zz_val = program.expval(
            thetas,
            obs_index=0,
            stream_device=stream_device,
            offload_back=offload_back,
        )
        zz_val.backward()
        opt.step()

        val = float(zz_val.detach().cpu().item())
        history.append(val)
        step_log.append(
            {
                "step": int(step),
                "sum_zz": float(val),
                "expected_cut": float(expected_cut_from_sum_zz(val, m_edges)),
            }
        )
        if val < best_val:
            best_val = val
            best_thetas = thetas.detach().cpu().clone()

        if (step % int(log_every) == 0) or (step == int(steps) - 1):
            exp_cut = expected_cut_from_sum_zz(val, m_edges)
            print(f"step={step:04d} sum<ZZ>={val:+.8f} E[cut]={exp_cut:.6f}")

    if best_thetas is None:
        raise RuntimeError("Training produced no best_thetas.")
    best_step = int(np.argmin(np.asarray(history, dtype=np.float64)))

    return {
        "delta_t": float(delta_t),
        "history_sum_zz": history,
        "history_expected_cut": [expected_cut_from_sum_zz(v, m_edges) for v in history],
        "step_log": step_log,
        "best_step": best_step,
        "best_sum_zz": float(best_val),
        "best_expected_cut": expected_cut_from_sum_zz(best_val, m_edges),
        "initial_sum_zz": float(history[0]) if history else None,
        "initial_expected_cut": expected_cut_from_sum_zz(history[0], m_edges) if history else None,
        "best_thetas": best_thetas,
        "final_thetas": thetas.detach().cpu(),
    }


def _build_payload(
    *,
    n_qubits: int,
    p_layers: int,
    delta_t: float,
    steps: int,
    lr: float,
    seed: int,
    run_device: str,
    preset: str,
    preset_overrides: Optional[Dict[str, object]],
    build_min_abs: float,
    build_min_mat_abs: Optional[float],
    problem_json: Optional[str],
    problem_definition,
    edges,
    graph_summary,
    qaoa_structure,
    surrogate_settings,
    training_config,
    m_edges: int,
    n_params: int,
    run_result: Dict[str, Any],
    delta_t_values: List[float],
    sweep_runs: List[Dict[str, Any]],
    selected_run_index: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "config": {
            "n_qubits": n_qubits,
            "p_layers": int(p_layers),
            "delta_t": float(delta_t),
            "steps": int(steps),
            "lr": float(lr),
            "seed": int(seed),
            "device": run_device,
            "preset": preset,
            "preset_overrides": preset_overrides,
            "build_min_abs": float(build_min_abs),
            "build_min_mat_abs": build_min_mat_abs,
            "problem_json": problem_json,
        },
        "problem_definition": problem_definition,
        "edges": [list(e) for e in edges],
        "graph_summary": graph_summary,
        "qaoa_structure": qaoa_structure,
        "surrogate_settings": surrogate_settings,
        "training_config": training_config,
        "m_edges": m_edges,
        "n_params": int(n_params),
        "history_sum_zz": run_result["history_sum_zz"],
        "history_expected_cut": run_result["history_expected_cut"],
        "step_log": run_result["step_log"],
        "best_step": int(run_result["best_step"]),
        "best_sum_zz": float(run_result["best_sum_zz"]),
        "best_expected_cut": float(run_result["best_expected_cut"]),
        "initial_sum_zz": run_result["initial_sum_zz"],
        "initial_expected_cut": run_result["initial_expected_cut"],
        "best_thetas": run_result["best_thetas"],
        "final_thetas": run_result["final_thetas"],
    }
    if len(delta_t_values) > 1:
        payload["delta_t_sweep"] = {
            "values": [float(v) for v in delta_t_values],
            "selected_run_index": int(selected_run_index),
            "selected_delta_t": float(delta_t),
            "runs": sweep_runs,
        }
    return payload


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    problem_def = None
    if str(args.problem_json).strip():
        problem_def = load_qaoa_problem_json(args.problem_json)
        n_qubits = int(problem_def["n_qubits"])
        edges = list(problem_def["edges"])
        p_layers = int(problem_def["qaoa"]["p_layers"])
        base_delta_t = float(problem_def["qaoa"]["init"]["delta_t"])
    else:
        n_qubits = int(args.n_qubits)
        p_layers = int(args.p_layers)
        base_delta_t = float(args.delta_t)
        if args.edges_json:
            edges = load_edges_json(args.edges_json)
        else:
            edges = make_ring_chord_graph(n_qubits=n_qubits, chord_shift=int(args.chord_shift))

    m_edges = len(edges)
    if m_edges < 1:
        raise ValueError("The graph must contain at least one edge.")

    circuit, n_params = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=int(p_layers))
    zz_obj = build_maxcut_observable(n_qubits=n_qubits, edges=edges)

    delta_t_values = _parse_delta_t_sweep(args.delta_t_sweep)
    if len(delta_t_values) == 0:
        delta_t_values = [float(base_delta_t)]
    else:
        print(f"[delta_t sweep] using explicit list: {delta_t_values}")

    run_device = _choose_device(str(args.device))
    if run_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    if str(args.preset) == "auto":
        preset = "gpu_full" if run_device.startswith("cuda") else "gpu_min"
    else:
        preset = str(args.preset)

    preset_overrides: Optional[Dict[str, object]] = None
    if run_device == "cpu" and preset == "gpu_min":
        # Keep parity with Tutorial/07 CPU fallback.
        preset_overrides = _default_cpu_exact_overrides()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(args.output_json).lower() == "auto":
        out_json_path = out_path.with_suffix(".json")
    elif str(args.output_json).strip() == "":
        out_json_path = None
    else:
        out_json_path = Path(args.output_json)

    qaoa_structure = {
        "n_qubits": n_qubits,
        "p_layers": int(p_layers),
        "n_edges": m_edges,
        "n_params": int(n_params),
        "init_state": "|+>^n (Hadamard on each qubit)",
        "layer_pattern": "for each layer: ZZ rotations on all edges, then X rotations on all qubits",
        "gate_counts": {
            "hadamard": int(n_qubits),
            "cost_pauli_rotation_zz": int(m_edges * int(p_layers)),
            "mixer_pauli_rotation_x": int(n_qubits * int(p_layers)),
            "total_gates": int(len(circuit)),
        },
    }
    surrogate_settings = {
        "preset": preset,
        "preset_overrides": preset_overrides,
        "compile": {
            "build_min_abs": float(args.build_min_abs),
            "build_min_mat_abs": args.build_min_mat_abs,
            "build_thetas_from": "TQA init theta",
        },
        "runtime": {
            "device": run_device,
            "stream_device": run_device,
            "offload_back": bool(run_device.startswith("cuda")),
            "dtype": "float64",
        },
    }

    run_results: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []
    for run_idx, delta_t in enumerate(delta_t_values):
        print("")
        print(f"[delta_t run {run_idx + 1}/{len(delta_t_values)}] delta_t={float(delta_t):.8f}")
        run_result = _train_one_delta_t(
            delta_t=float(delta_t),
            n_qubits=n_qubits,
            p_layers=int(p_layers),
            m_edges=m_edges,
            n_params=int(n_params),
            circuit=circuit,
            zz_obj=zz_obj,
            run_device=run_device,
            preset=preset,
            preset_overrides=preset_overrides,
            build_min_abs=float(args.build_min_abs),
            build_min_mat_abs=args.build_min_mat_abs,
            steps=int(args.steps),
            lr=float(args.lr),
            seed=int(args.seed),
            log_every=int(args.log_every),
        )
        run_results.append(run_result)
        run_summary = {
            "run_index": int(run_idx),
            "delta_t": float(delta_t),
            "best_step": int(run_result["best_step"]),
            "best_sum_zz": float(run_result["best_sum_zz"]),
            "best_expected_cut": float(run_result["best_expected_cut"]),
            "initial_sum_zz": run_result["initial_sum_zz"],
            "initial_expected_cut": run_result["initial_expected_cut"],
            "final_sum_zz": float(run_result["history_sum_zz"][-1]) if run_result["history_sum_zz"] else None,
            "final_expected_cut": (
                float(run_result["history_expected_cut"][-1]) if run_result["history_expected_cut"] else None
            ),
        }
        run_summaries.append(run_summary)
        print(
            f"[delta_t run {run_idx + 1}] best sum<ZZ>={float(run_result['best_sum_zz']):+.8f}, "
            f"best E[cut]={float(run_result['best_expected_cut']):.6f}"
        )

    if len(run_results) == 0:
        raise RuntimeError("No training runs executed.")

    selected_idx = int(np.argmin(np.asarray([r["best_sum_zz"] for r in run_results], dtype=np.float64)))
    selected_delta_t = float(delta_t_values[selected_idx])
    selected_run = run_results[selected_idx]

    # Optionally save all per-delta runs.
    if bool(args.save_per_delta) and len(delta_t_values) > 1:
        pt_suffix = out_path.suffix if out_path.suffix else ".pt"
        for run_idx, (delta_t, run_result) in enumerate(zip(delta_t_values, run_results)):
            tag = _delta_t_tag(float(delta_t))
            run_out_path = out_path.with_name(f"{out_path.stem}_dt{tag}{pt_suffix}")
            run_training_config = {
                "optimizer": "Adam",
                "lr": float(args.lr),
                "steps": int(args.steps),
                "seed": int(args.seed),
                "delta_t_tqa_init": float(delta_t),
                "log_every": int(args.log_every),
            }
            run_payload = _build_payload(
                n_qubits=n_qubits,
                p_layers=int(p_layers),
                delta_t=float(delta_t),
                steps=int(args.steps),
                lr=float(args.lr),
                seed=int(args.seed),
                run_device=run_device,
                preset=preset,
                preset_overrides=preset_overrides,
                build_min_abs=float(args.build_min_abs),
                build_min_mat_abs=args.build_min_mat_abs,
                problem_json=(str(args.problem_json) if str(args.problem_json).strip() else None),
                problem_definition=(problem_def["raw"] if problem_def is not None else None),
                edges=edges,
                graph_summary=_degree_stats(n_qubits, edges),
                qaoa_structure=qaoa_structure,
                surrogate_settings=surrogate_settings,
                training_config=run_training_config,
                m_edges=m_edges,
                n_params=int(n_params),
                run_result=run_result,
                delta_t_values=[float(delta_t)],
                sweep_runs=[],
                selected_run_index=0,
            )
            torch.save(run_payload, run_out_path)

            run_json_path = None
            if out_json_path is not None:
                run_json_path = out_json_path.with_name(f"{out_json_path.stem}_dt{tag}{out_json_path.suffix}")
                json_payload = {
                    "checkpoint_path": str(run_out_path),
                    "problem_json_path": (str(args.problem_json) if str(args.problem_json).strip() else None),
                    "problem_definition": (problem_def["raw"] if problem_def is not None else None),
                    "graph_summary": run_payload["graph_summary"],
                    "qaoa_structure": qaoa_structure,
                    "surrogate_settings": surrogate_settings,
                    "training_config": run_training_config,
                    "result_summary": {
                        "initial_sum_zz": run_payload["initial_sum_zz"],
                        "initial_expected_cut": run_payload["initial_expected_cut"],
                        "best_step": int(run_result["best_step"]),
                        "best_sum_zz": float(run_result["best_sum_zz"]),
                        "best_expected_cut": float(run_result["best_expected_cut"]),
                        "final_sum_zz": (
                            float(run_result["history_sum_zz"][-1]) if run_result["history_sum_zz"] else None
                        ),
                        "final_expected_cut": (
                            float(run_result["history_expected_cut"][-1])
                            if run_result["history_expected_cut"]
                            else None
                        ),
                    },
                    "step_log": run_result["step_log"],
                    "edges": [list(e) for e in edges],
                }
                run_json_path.parent.mkdir(parents=True, exist_ok=True)
                run_json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

            run_summaries[run_idx]["checkpoint_path"] = str(run_out_path)
            run_summaries[run_idx]["report_json_path"] = str(run_json_path) if run_json_path is not None else None

    training_config = {
        "optimizer": "Adam",
        "lr": float(args.lr),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "delta_t_tqa_init": float(selected_delta_t),
        "log_every": int(args.log_every),
    }
    payload = _build_payload(
        n_qubits=n_qubits,
        p_layers=int(p_layers),
        delta_t=float(selected_delta_t),
        steps=int(args.steps),
        lr=float(args.lr),
        seed=int(args.seed),
        run_device=run_device,
        preset=preset,
        preset_overrides=preset_overrides,
        build_min_abs=float(args.build_min_abs),
        build_min_mat_abs=args.build_min_mat_abs,
        problem_json=(str(args.problem_json) if str(args.problem_json).strip() else None),
        problem_definition=(problem_def["raw"] if problem_def is not None else None),
        edges=edges,
        graph_summary=_degree_stats(n_qubits, edges),
        qaoa_structure=qaoa_structure,
        surrogate_settings=surrogate_settings,
        training_config=training_config,
        m_edges=m_edges,
        n_params=int(n_params),
        run_result=selected_run,
        delta_t_values=[float(v) for v in delta_t_values],
        sweep_runs=run_summaries,
        selected_run_index=int(selected_idx),
    )
    torch.save(payload, out_path)

    if out_json_path is not None:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {
            "checkpoint_path": str(out_path),
            "problem_json_path": (str(args.problem_json) if str(args.problem_json).strip() else None),
            "problem_definition": (problem_def["raw"] if problem_def is not None else None),
            "graph_summary": payload["graph_summary"],
            "qaoa_structure": qaoa_structure,
            "surrogate_settings": surrogate_settings,
            "training_config": training_config,
            "result_summary": {
                "initial_sum_zz": payload["initial_sum_zz"],
                "initial_expected_cut": payload["initial_expected_cut"],
                "best_step": int(selected_run["best_step"]),
                "best_sum_zz": float(selected_run["best_sum_zz"]),
                "best_expected_cut": float(selected_run["best_expected_cut"]),
                "final_sum_zz": (
                    float(selected_run["history_sum_zz"][-1]) if selected_run["history_sum_zz"] else None
                ),
                "final_expected_cut": (
                    float(selected_run["history_expected_cut"][-1]) if selected_run["history_expected_cut"] else None
                ),
            },
            "step_log": selected_run["step_log"],
            "edges": [list(e) for e in edges],
        }
        if len(delta_t_values) > 1:
            json_payload["delta_t_sweep"] = {
                "values": [float(v) for v in delta_t_values],
                "selected_run_index": int(selected_idx),
                "selected_delta_t": float(selected_delta_t),
                "runs": run_summaries,
            }
        out_json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print("")
    print(f"saved checkpoint: {out_path}")
    if out_json_path is not None:
        print(f"saved train report json: {out_json_path}")
    if len(delta_t_values) > 1:
        print(f"delta_t sweep values: {delta_t_values}")
        print(f"selected delta_t: {float(selected_delta_t):.8f} (run {selected_idx + 1}/{len(delta_t_values)})")
    print(f"best sum<ZZ>: {float(selected_run['best_sum_zz']):+.8f}")
    print(f"best expected cut: {float(selected_run['best_expected_cut']):.6f}")


if __name__ == "__main__":
    main()
