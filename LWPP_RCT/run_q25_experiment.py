from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from qaoa_experiment_common import (
    DEFAULT_CONFIG_PATH,
    build_initial_theta_np,
    build_maxcut_observable,
    build_qaoa_circuit,
    build_run_summary,
    build_cudaq_exact_backend,
    choose_device,
    cleanup_memory,
    evaluate_exact_on_surrogate_trajectory,
    load_mapping_file,
    load_or_create_graph,
    make_stage_compile_fn,
    normalize_init_strategy,
    plot_case1_exact_warmup,
    plot_case2_lwpp_warmup,
    plot_case3_lwpp_to_exact,
    plot_case4_lwpp_to_coeff,
    plot_integrated_case_comparison,
    save_json,
    seed_tag,
    train_with_exact_optimizer_cudaq,
    train_with_fixed_program,
    train_with_periodic_rebuild,
    theta_trajectory_to_step_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Q25 LWPP/RCT experiment sweep under LWPP_RCT."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the experiment config (JSON or YAML).",
    )
    parser.add_argument(
        "--init-strategy",
        type=str,
        default="",
        help="Optional filter for a single init strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional filter for a single run seed.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Optional filter for a single QAOA depth.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run directory if artifacts.json already exists.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Do not run aggregate_q25_results.py after the sweep.",
    )
    return parser.parse_args()


def _normalize_config(raw: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    directory_cfg = dict(raw.get("DIRECTORY", {}))
    graph_cfg = dict(raw.get("GRAPH", {}))
    experiment_cfg = dict(raw.get("EXPERIMENT", {}))
    optimizer_cfg = dict(raw.get("OPTIMIZER", {}))
    warmup_cfg = dict(raw.get("WARMUP", {}))
    coeff_cfg = dict(raw.get("COEFF_FINETUNING", {}))
    runtime_cfg = dict(raw.get("RUNTIME", {}))

    output_root = Path(directory_cfg.get("output_root", config_path.parent / "results")).expanduser().resolve()
    graph_dir = Path(directory_cfg.get("graph_dir", config_path.parent / "graph")).expanduser().resolve()

    init_strategies = [normalize_init_strategy(x) for x in experiment_cfg.get("init_strategies", ["random", "near_zero", "tqa"])]
    run_seeds = [int(x) for x in experiment_cfg.get("run_seeds", [0, 1, 2, 3, 4])]
    p_layers_list = [int(x) for x in experiment_cfg.get("p_layers_list", [6, 9])]

    return {
        "config_path": str(config_path),
        "DIRECTORY": {
            "output_root": output_root,
            "graph_dir": graph_dir,
        },
        "GRAPH": {
            "n_qubits": int(graph_cfg.get("n_qubits", 25)),
            "edge_prob": float(graph_cfg.get("edge_prob", 0.7)),
            "graph_seed": int(graph_cfg.get("graph_seed", 42)),
            "graph_per_run_seed": bool(graph_cfg.get("graph_per_run_seed", True)),
            "max_tries": int(graph_cfg.get("max_tries", 100)),
            "create_if_missing": bool(graph_cfg.get("create_if_missing", True)),
        },
        "EXPERIMENT": {
            "init_strategies": init_strategies,
            "run_seeds": run_seeds,
            "p_layers_list": p_layers_list,
            "delta_t": float(experiment_cfg.get("delta_t", 1.2)),
        },
        "OPTIMIZER": {
            "name": str(optimizer_cfg.get("name", "Adam")),
            "lr": float(optimizer_cfg.get("lr", 0.01)),
        },
        "WARMUP": {
            "steps": int(warmup_cfg.get("steps", 400)),
            "mw_max_weight": int(warmup_cfg.get("mw_max_weight", 3)),
        },
        "COEFF_FINETUNING": {
            "steps": int(coeff_cfg.get("steps", 400)),
            "min_abs": float(coeff_cfg.get("min_abs", 1.0e-3)),
            "rebuild_interval": int(coeff_cfg.get("rebuild_interval", 10)),
            "max_weight_override": coeff_cfg.get("max_weight_override", None),
        },
        "RUNTIME": {
            "device_requested": str(runtime_cfg.get("device", "auto")),
            "chunk_size": int(runtime_cfg.get("chunk_size", 20_000_000)),
            "parallel_compile": bool(runtime_cfg.get("parallel_compile", False)),
            "log_every": int(runtime_cfg.get("log_every", 20)),
            "theta_save_stride": int(runtime_cfg.get("theta_save_stride", 1)),
        },
    }


def run_single_experiment(
    *,
    run_dir: Path,
    config: Dict[str, Any],
    init_strategy: str,
    run_seed: int,
    p_layers: int,
    edges: List[Any],
    graph_seed_used: int,
    graph_tag: str,
    source_graph_json: Path,
    source_graph_png: Optional[Path],
) -> Dict[str, Any]:
    graph_cfg = config["GRAPH"]
    runtime_cfg = config["RUNTIME"]
    optimizer_cfg = config["OPTIMIZER"]
    warmup_cfg = config["WARMUP"]
    coeff_cfg = config["COEFF_FINETUNING"]
    exp_cfg = config["EXPERIMENT"]

    n_qubits = int(graph_cfg["n_qubits"])
    n_edges = int(len(edges))
    lr = float(optimizer_cfg["lr"])
    device = choose_device(runtime_cfg["device_requested"])
    chunk_size = int(runtime_cfg["chunk_size"])
    parallel_compile = bool(runtime_cfg["parallel_compile"])
    log_every = int(runtime_cfg["log_every"])
    theta_save_stride = max(1, int(runtime_cfg["theta_save_stride"]))
    delta_t = float(exp_cfg["delta_t"])

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "init_strategy": init_strategy,
            "run_seed": int(run_seed),
            "n_qubits": int(n_qubits),
            "p_layers": int(p_layers),
            "device": str(device),
        },
        indent=2,
    ))

    def _run() -> Dict[str, Any]:
        init_theta_np = build_initial_theta_np(
            init_strategy=init_strategy,
            n_layers=int(p_layers),
            n_edges=int(n_edges),
            n_qubits=int(n_qubits),
            delta_t=float(delta_t),
            seed=int(run_seed),
        )

        circuit, _ = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=int(p_layers))
        zz_obj = build_maxcut_observable(n_qubits=n_qubits, edges=edges)

        mw_compile_fn = make_stage_compile_fn(
            stage_name="mw_warmup",
            circuit=circuit,
            zz_obj=zz_obj,
            device=device,
            chunk_size=chunk_size,
            parallel_compile=parallel_compile,
            max_weight_override=int(warmup_cfg["mw_max_weight"]),
            n_qubits=n_qubits,
            preset_name="hybrid",
        )
        coeff_compile_fn = make_stage_compile_fn(
            stage_name="coeff_finetune",
            circuit=circuit,
            zz_obj=zz_obj,
            device=device,
            chunk_size=chunk_size,
            parallel_compile=parallel_compile,
            max_weight_override=coeff_cfg["max_weight_override"],
            n_qubits=n_qubits,
            preset_name="hybrid",
        )

        exact_backend = build_cudaq_exact_backend(
            n_qubits=n_qubits,
            edges=edges,
            p_layers=int(p_layers),
        )
        mw_program, mw_compile_info = mw_compile_fn(build_thetas=None, build_min_abs=None)

        case1_history, case1_final_thetas, case1_theta_trajectory, case1_exact_runtime = train_with_exact_optimizer_cudaq(
            n_qubits=n_qubits,
            edges=edges,
            p_layers=int(p_layers),
            start_thetas_np=init_theta_np,
            n_edges=n_edges,
            steps=int(warmup_cfg["steps"]),
            lr=lr,
            stage_name="case1-exact-warmup",
            log_every=log_every,
        )

        case2_history, case2_final_thetas, case2_theta_trajectory = train_with_fixed_program(
            program=mw_program,
            start_thetas_np=init_theta_np,
            device=device,
            n_edges=n_edges,
            steps=int(warmup_cfg["steps"]),
            lr=lr,
            stage_name="case2-lwpp-warmup",
            log_every=log_every,
        )
        case2_exact_curve = evaluate_exact_on_surrogate_trajectory(
            exact_sum_zz_batch_fn=exact_backend["evaluate_batch"],
            theta_trajectory=case2_theta_trajectory,
            surrogate_history=case2_history,
            n_edges=n_edges,
        )

        case3_history, case3_final_thetas, case3_theta_trajectory, case3_exact_runtime = train_with_exact_optimizer_cudaq(
            n_qubits=n_qubits,
            edges=edges,
            p_layers=int(p_layers),
            start_thetas_np=case2_final_thetas,
            n_edges=n_edges,
            steps=int(coeff_cfg["steps"]),
            lr=lr,
            stage_name="case3-lwpp-to-exact",
            log_every=log_every,
        )

        (
            case4_history,
            case4_final_thetas,
            case4_theta_trajectory,
            case4_rebuild_log,
        ) = train_with_periodic_rebuild(
            compile_program_fn=coeff_compile_fn,
            start_thetas_np=case2_final_thetas,
            device=device,
            n_edges=n_edges,
            steps=int(coeff_cfg["steps"]),
            lr=lr,
            build_min_abs=float(coeff_cfg["min_abs"]),
            rebuild_interval=int(coeff_cfg["rebuild_interval"]),
            stage_name="case4-lwpp-to-coeff",
            log_every=log_every,
        )
        case4_exact_curve = evaluate_exact_on_surrogate_trajectory(
            exact_sum_zz_batch_fn=exact_backend["evaluate_batch"],
            theta_trajectory=case4_theta_trajectory,
            surrogate_history=case4_history,
            n_edges=n_edges,
        )

        plot_case1_exact_warmup(
            exact_history=case1_history,
            output_path=run_dir / "case1_exact_warmup.png",
        )
        plot_case2_lwpp_warmup(
            surrogate_history=case2_history,
            exact_rows=case2_exact_curve,
            output_path=run_dir / "case2_lwpp_warmup.png",
        )
        plot_case3_lwpp_to_exact(
            exact_history=case3_history,
            output_path=run_dir / "case3_lwpp_to_exact.png",
        )
        plot_case4_lwpp_to_coeff(
            surrogate_history=case4_history,
            exact_rows=case4_exact_curve,
            rebuild_log=case4_rebuild_log,
            output_path=run_dir / "case4_lwpp_to_coeff.png",
        )
        plot_integrated_case_comparison(
            case1_history=case1_history,
            case2_history=case2_history,
            case2_exact_rows=case2_exact_curve,
            case3_history=case3_history,
            case4_history=case4_history,
            case4_exact_rows=case4_exact_curve,
            case4_rebuild_log=case4_rebuild_log,
            warmup_steps_nominal=int(warmup_cfg["steps"]),
            finetune_steps_nominal=int(coeff_cfg["steps"]),
            output_path=run_dir / "integrated_comparison.png",
        )

        artifacts = {
            "schema_version": 1,
            "config": {
                "source_config_path": str(config["config_path"]),
                "run_dir": str(run_dir),
                "graph_source_json": str(source_graph_json),
                "graph_source_png": None if source_graph_png is None else str(source_graph_png),
                "init_strategy": str(init_strategy),
                "run_seed": int(run_seed),
                "graph_seed": int(graph_seed_used),
                "graph_seed_base": int(graph_cfg["graph_seed"]),
                "graph_seed_used": int(graph_seed_used),
                "graph_per_run_seed": bool(graph_cfg["graph_per_run_seed"]),
                "graph_tag": str(graph_tag),
                "n_qubits": int(n_qubits),
                "n_edges": int(n_edges),
                "p_layers": int(p_layers),
                "delta_t": float(delta_t),
                "optimizer": dict(optimizer_cfg),
                "warmup": dict(warmup_cfg),
                "coeff_finetuning": dict(coeff_cfg),
                "runtime": {
                    **runtime_cfg,
                    "device_runtime": str(device),
                },
            },
            "graph": {
                "edges": [[int(u), int(v)] for (u, v) in edges],
            },
            "compile": {
                "exact_backend": {
                    "backend": "cudaq.observe",
                    "optimizer_case1": case1_exact_runtime,
                    "optimizer_case3": case3_exact_runtime,
                },
                "mw": mw_compile_info,
                "coeff_initial": None if len(case4_rebuild_log) == 0 else dict(case4_rebuild_log[0]),
            },
            "cases": {
                "case1_exact_warmup": {
                    "start_thetas": init_theta_np.tolist(),
                    "final_thetas": case1_final_thetas.tolist(),
                    "history": case1_history,
                    "theta_by_step": theta_trajectory_to_step_map(
                        case1_theta_trajectory,
                        start_thetas=init_theta_np.tolist(),
                        final_thetas=case1_final_thetas.tolist(),
                        stride=theta_save_stride,
                    ),
                },
                "case2_lwpp_warmup": {
                    "start_thetas": init_theta_np.tolist(),
                    "final_thetas": case2_final_thetas.tolist(),
                    "history": case2_history,
                    "theta_by_step": theta_trajectory_to_step_map(
                        case2_theta_trajectory,
                        start_thetas=init_theta_np.tolist(),
                        final_thetas=case2_final_thetas.tolist(),
                        stride=theta_save_stride,
                    ),
                    "exact_curve": case2_exact_curve,
                },
                "case3_lwpp_to_exact": {
                    "start_thetas": case2_final_thetas.tolist(),
                    "final_thetas": case3_final_thetas.tolist(),
                    "history": case3_history,
                    "theta_by_step": theta_trajectory_to_step_map(
                        case3_theta_trajectory,
                        start_thetas=case2_final_thetas.tolist(),
                        final_thetas=case3_final_thetas.tolist(),
                        stride=theta_save_stride,
                    ),
                },
                "case4_lwpp_to_coeff": {
                    "start_thetas": case2_final_thetas.tolist(),
                    "final_thetas": case4_final_thetas.tolist(),
                    "history": case4_history,
                    "theta_by_step": theta_trajectory_to_step_map(
                        case4_theta_trajectory,
                        start_thetas=case2_final_thetas.tolist(),
                        final_thetas=case4_final_thetas.tolist(),
                        stride=theta_save_stride,
                    ),
                    "exact_curve": case4_exact_curve,
                    "rebuild_log": case4_rebuild_log,
                },
            },
        }
        artifacts["run_summary"] = build_run_summary(artifacts)
        save_json(run_dir / "artifacts.json", artifacts)
        return artifacts

    try:
        return _run()
    finally:
        cleanup_memory(device)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    raw = load_mapping_file(config_path)
    config = _normalize_config(raw, config_path)

    if args.init_strategy:
        config["EXPERIMENT"]["init_strategies"] = [normalize_init_strategy(args.init_strategy)]
    if args.seed is not None:
        config["EXPERIMENT"]["run_seeds"] = [int(args.seed)]
    if args.depth is not None:
        config["EXPERIMENT"]["p_layers_list"] = [int(args.depth)]

    output_root = Path(config["DIRECTORY"]["output_root"]).resolve()
    graph_dir = Path(config["DIRECTORY"]["graph_dir"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_device = choose_device(config["RUNTIME"]["device_requested"])
    finished_run_count = 0
    for init_strategy in config["EXPERIMENT"]["init_strategies"]:
        init_root = output_root / str(init_strategy)
        init_root.mkdir(parents=True, exist_ok=True)
        for run_seed in config["EXPERIMENT"]["run_seeds"]:
            resolved_graph_seed = int(config["GRAPH"]["graph_seed"]) + int(run_seed) if bool(config["GRAPH"]["graph_per_run_seed"]) else int(config["GRAPH"]["graph_seed"])
            graph_tag = f"_seed{int(run_seed):03d}" if bool(config["GRAPH"]["graph_per_run_seed"]) else ""
            edges, source_graph_json, source_graph_png, graph_created = load_or_create_graph(
                graph_dir=graph_dir,
                n_qubits=int(config["GRAPH"]["n_qubits"]),
                edge_prob=float(config["GRAPH"]["edge_prob"]),
                seed=int(resolved_graph_seed),
                graph_tag=str(graph_tag),
                create_if_missing=bool(config["GRAPH"]["create_if_missing"]),
                max_tries=int(config["GRAPH"]["max_tries"]),
            )
            print(
                json.dumps(
                    {
                        "run_seed": int(run_seed),
                        "graph_seed_used": int(resolved_graph_seed),
                        "graph_json": str(source_graph_json),
                        "graph_png": None if source_graph_png is None else str(source_graph_png),
                        "graph_created": bool(graph_created),
                        "n_edges": len(edges),
                    },
                    indent=2,
                )
            )
            seed_root = init_root / seed_tag(int(run_seed))
            seed_root.mkdir(parents=True, exist_ok=True)
            for p_layers in config["EXPERIMENT"]["p_layers_list"]:
                run_dir = seed_root / f"Q{int(config['GRAPH']['n_qubits'])}_L{int(p_layers)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                artifacts_path = run_dir / "artifacts.json"
                if args.skip_existing and artifacts_path.exists():
                    print(f"[skip] {artifacts_path}")
                    cleanup_memory(runtime_device)
                    continue
                result = None
                try:
                    result = run_single_experiment(
                        run_dir=run_dir,
                        config=config,
                        init_strategy=str(init_strategy),
                        run_seed=int(run_seed),
                        p_layers=int(p_layers),
                        edges=edges,
                        graph_seed_used=int(resolved_graph_seed),
                        graph_tag=str(graph_tag),
                        source_graph_json=source_graph_json,
                        source_graph_png=source_graph_png,
                    )
                    finished_run_count += 1
                finally:
                    if result is not None:
                        del result
                    cleanup_memory(runtime_device)

    print(json.dumps({"finished_runs": int(finished_run_count), "output_root": str(output_root)}, indent=2))

    if not args.skip_aggregate:
        from aggregate_q25_results import aggregate_results_from_config

        aggregate_results_from_config(config_path)


if __name__ == "__main__":
    main()
