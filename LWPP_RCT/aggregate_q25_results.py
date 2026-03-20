from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qaoa_experiment_common import (
    DEFAULT_CONFIG_PATH,
    aggregate_group_records,
    load_mapping_file,
    normalize_init_strategy,
    plot_aggregate_figure1_like,
    plot_aggregate_mean_curves,
    plot_seed_overlay_figure2_like,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Q25 LWPP/RCT experiment results."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the experiment config (JSON or YAML).",
    )
    return parser.parse_args()


def _normalize_for_aggregate(raw: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    directory_cfg = dict(raw.get("DIRECTORY", {}))
    experiment_cfg = dict(raw.get("EXPERIMENT", {}))
    graph_cfg = dict(raw.get("GRAPH", {}))

    output_root = Path(directory_cfg.get("output_root", config_path.parent / "results")).expanduser().resolve()
    init_strategies = [normalize_init_strategy(x) for x in experiment_cfg.get("init_strategies", ["random", "near_zero", "tqa"])]
    p_layers_list = [int(x) for x in experiment_cfg.get("p_layers_list", [6, 9])]

    return {
        "output_root": output_root,
        "init_strategies": init_strategies,
        "p_layers_list": p_layers_list,
        "n_qubits": int(graph_cfg.get("n_qubits", 25)),
    }


def _load_artifacts(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_results_from_config(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    raw = load_mapping_file(config_path)
    cfg = _normalize_for_aggregate(raw, config_path)

    output_root = Path(cfg["output_root"]).resolve()
    aggregate_root = output_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    overall_summary: Dict[str, Any] = {
        "config_path": str(config_path),
        "output_root": str(output_root),
        "groups": {},
    }

    for init_strategy in cfg["init_strategies"]:
        for p_layers in cfg["p_layers_list"]:
            pattern = output_root / str(init_strategy)
            artifact_paths = sorted(pattern.glob(f"seed_*/Q{int(cfg['n_qubits'])}_L{int(p_layers)}/artifacts.json"))
            if len(artifact_paths) == 0:
                continue

            records = [_load_artifacts(path) for path in artifact_paths]
            aggregate_payload = aggregate_group_records(records)

            group_key = f"{init_strategy}/Q{int(cfg['n_qubits'])}_L{int(p_layers)}"
            group_dir = aggregate_root / str(init_strategy) / f"Q{int(cfg['n_qubits'])}_L{int(p_layers)}"
            group_dir.mkdir(parents=True, exist_ok=True)

            summary_payload = {
                "group_key": group_key,
                "artifact_paths": [str(path) for path in artifact_paths],
                **aggregate_payload,
            }
            save_json(group_dir / "aggregate_summary.json", summary_payload)

            title_prefix = f"{init_strategy}, Q{int(cfg['n_qubits'])}, p={int(p_layers)}"
            plot_aggregate_mean_curves(
                aggregate_payload,
                group_dir / "mean_exact_curves.png",
                f"Mean Exact Curves ({title_prefix})",
            )
            plot_aggregate_figure1_like(
                aggregate_payload,
                group_dir / "figure1_like.png",
                f"Figure-1-Like Comparison ({title_prefix})",
            )
            plot_seed_overlay_figure2_like(
                records,
                group_dir / "figure2_like_seed_overlay.png",
                f"Seed Overlay ({title_prefix})",
            )

            overall_summary["groups"][group_key] = {
                "n_runs": int(summary_payload["n_runs"]),
                "final_exact_expected_cut": summary_payload["final_exact_expected_cut"],
                "aggregate_dir": str(group_dir),
            }

    save_json(aggregate_root / "overall_summary.json", overall_summary)
    print(json.dumps(overall_summary, indent=2))
    return overall_summary


def main() -> None:
    args = parse_args()
    aggregate_results_from_config(Path(args.config))


if __name__ == "__main__":
    main()
