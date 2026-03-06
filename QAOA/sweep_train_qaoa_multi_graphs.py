from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from make_maxcut_graph import _plot_graph, _simulated_annealing_maxcut


Edge = Tuple[int, int]


def _resolve_base_edges_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    search_order = [
        repo_root / candidate,
        script_dir / candidate,
        script_dir / "artifacts" / candidate.name,
        repo_root / "QAOA" / "artifacts" / candidate.name,
    ]
    for path in search_order:
        if path.exists():
            return path.resolve()

    tried = "\n  - ".join(str(p.resolve()) for p in search_order)
    raise FileNotFoundError(
        "Base edges file not found. Tried:\n"
        f"  - {tried}"
    )


def _normalize_edges(raw_edges: Sequence[Sequence[int]]) -> List[Edge]:
    edges: List[Edge] = []
    seen = set()
    for pair in raw_edges:
        if len(pair) != 2:
            raise ValueError(f"Invalid edge entry: {pair}")
        u = int(pair[0])
        v = int(pair[1])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in seen:
            seen.add((a, b))
            edges.append((a, b))
    edges.sort()
    return edges


def _infer_n_qubits(edges: Sequence[Edge]) -> int:
    if not edges:
        raise ValueError("Edge list is empty.")
    return max(max(u, v) for (u, v) in edges) + 1


def _is_connected(n_qubits: int, edges: Sequence[Edge]) -> bool:
    n = int(n_qubits)
    if n <= 1:
        return True
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        for nxt in adj[node]:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(nxt)
    return all(seen)


def _all_pairs(n_qubits: int) -> List[Edge]:
    pairs: List[Edge] = []
    for i in range(int(n_qubits)):
        for j in range(i + 1, int(n_qubits)):
            pairs.append((i, j))
    return pairs


def _sample_connected_graph_with_exact_edges(
    *,
    n_qubits: int,
    n_edges: int,
    seed: int,
    max_tries: int,
) -> List[Edge]:
    n = int(n_qubits)
    m = int(n_edges)
    if n <= 1:
        raise ValueError("n_qubits must be >= 2")

    all_pairs = _all_pairs(n)
    total_pairs = len(all_pairs)
    if m < n - 1:
        raise ValueError(f"n_edges={m} is too small to form a connected graph with n={n}")
    if m > total_pairs:
        raise ValueError(f"n_edges={m} exceeds complete graph size={total_pairs}")

    rng = random.Random(int(seed))
    for _ in range(int(max_tries)):
        selected = rng.sample(all_pairs, m)
        selected = sorted(selected)
        if _is_connected(n, selected):
            return selected

    raise RuntimeError(
        f"Failed to sample connected graph with n={n}, m={m} after max_tries={max_tries}"
    )


def _parse_p_layers(raw: str) -> List[int]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("p-layers list is empty.")
    return vals


def _parse_int_list(raw: str, *, field_name: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        t = part.strip()
        if t == "":
            continue
        out.append(int(t))
    if len(out) == 0:
        raise ValueError(f"{field_name} list is empty.")
    return out


def _parse_weight_modes(raw: str) -> List[str]:
    modes: List[str] = []
    allowed = {"yz", "xyz"}
    for part in str(raw).split(","):
        mode = part.strip().lower()
        if mode == "":
            continue
        if mode not in allowed:
            raise ValueError(f"Invalid weight mode '{mode}'. Allowed: {sorted(allowed)}")
        modes.append(mode)
    if len(modes) == 0:
        raise ValueError("weight-modes list is empty.")
    return modes


def _build_graph_viz_and_sa(
    *,
    n_qubits: int,
    edges: Sequence[Edge],
    graph_index: int,
    graph_source: str,
    plots_dir: Path,
    reports_dir: Path,
    title_prefix: str,
    anneal_steps: int,
    anneal_restarts: int,
    anneal_t0: float,
    anneal_tf: float,
    anneal_seed: int,
) -> Dict[str, object]:
    title = f"{title_prefix} (n={int(n_qubits)}, |E|={len(edges)}, {graph_source})"
    plot_path = plots_dir / f"graph_{int(graph_index):02d}.png"
    report_path = reports_dir / f"graph_{int(graph_index):02d}_anneal.json"

    anneal_result = _simulated_annealing_maxcut(
        n_qubits=int(n_qubits),
        edges=edges,
        steps=int(anneal_steps),
        restarts=int(anneal_restarts),
        t0=float(anneal_t0),
        tf=float(anneal_tf),
        seed=int(anneal_seed),
    )

    _plot_graph(
        out_path=plot_path,
        n_qubits=int(n_qubits),
        edges=edges,
        title=title,
        anneal_result=anneal_result,
    )

    report = {
        "graph_index": int(graph_index),
        "n_qubits": int(n_qubits),
        "n_edges": int(len(edges)),
        "graph_source": str(graph_source),
        "edges": [[int(u), int(v)] for (u, v) in edges],
        "annealing": anneal_result,
        "plot": str(plot_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "plot": str(plot_path),
        "anneal_report": str(report_path),
        "anneal_best_cut": float(anneal_result["best_cut"]),
        "anneal_best_bitstring": str(anneal_result["best_bitstring_leftmost_q0"]),
    }


def _build_train_cmd(
    *,
    python_exe: str,
    train_script: Path,
    edges_json: Path,
    n_qubits: int,
    p_layers: int,
    run_name: str,
    output_dir: Path,
    steps: int,
    lr: float,
    delta_t: float,
    seed: int,
    chunk_size: int,
    build_min_abs: float,
    rebuild_every: int,
    save_every: int,
    log_every: int,
    max_weight: int,
    weight_mode: str,
    no_build_min_abs: bool,
    init_mode: str,
    mixer_odd_start: float,
    mixer_odd_end: float,
) -> List[str]:
    cmd = [
        python_exe,
        str(train_script),
        "--edges-json",
        str(edges_json),
        "--n-qubits",
        str(int(n_qubits)),
        "--p-layers",
        str(int(p_layers)),
        "--steps",
        str(int(steps)),
        "--lr",
        str(float(lr)),
        "--delta-t",
        str(float(delta_t)),
        "--init-mode",
        str(init_mode),
        "--mixer-odd-start",
        str(float(mixer_odd_start)),
        "--mixer-odd-end",
        str(float(mixer_odd_end)),
        "--seed",
        str(int(seed)),
        "--output-dir",
        str(output_dir),
        "--run-name",
        str(run_name),
        "--build-min-abs",
        str(float(build_min_abs)),
        "--log-every",
        str(int(log_every)),
        "--max-weight",
        str(int(max_weight)),
        "--weight-mode",
        str(weight_mode),
    ]
    if int(chunk_size) > 0:
        cmd.extend(["--chunk-size", str(int(chunk_size))])
    if int(rebuild_every) > 0:
        cmd.extend(["--rebuild-every", str(int(rebuild_every))])
    if int(save_every) > 0:
        cmd.extend(["--save-every", str(int(save_every))])
    if bool(no_build_min_abs):
        cmd.append("--no-build-min-abs")
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate 3 additional graphs similar to a base MaxCut graph and "
            "train QAOA PPS for each p-layer setting."
        )
    )
    p.add_argument("--base-edges-json", type=str, default="QAOA/artifacts/maxcut_edges.json")
    p.add_argument("--n-qubits-list", type=str, default="", help="Optional comma-separated qubit sizes, e.g. 16,24,30")
    p.add_argument("--num-graphs", type=int, default=3, help="Number of new graphs to generate.")
    p.add_argument("--p-layers", type=str, default="3,5,7,9,11")
    p.add_argument("--max-weights", type=str, default="2,4,6,8")
    p.add_argument("--weight-modes", type=str, default="yz,xyz")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta-t", type=float, default=0.8)
    p.add_argument("--init-mode", type=str, default="tqa", choices=["tqa", "odd-linear-neg"])
    p.add_argument("--mixer-odd-start", type=float, default=-1.0)
    p.add_argument("--mixer-odd-end", type=float, default=-0.05)
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--build-min-abs", type=float, default=1e-3)
    p.add_argument("--no-build-min-abs", action="store_true")
    p.add_argument(
        "--min-abs-only",
        action="store_true",
        help="Disable weight-based truncation and keep only build_min_abs pruning.",
    )
    p.add_argument(
        "--max-weight-disabled-value",
        type=int,
        default=1_000_000_000,
        help="max_weight value to use when --min-abs-only is enabled.",
    )
    p.add_argument("--rebuild-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tries", type=int, default=3000)
    p.add_argument("--output-dir", type=str, default="QAOA/artifacts/weight_sweep")
    p.add_argument("--run-prefix", type=str, default="qaoa_pps")
    p.add_argument(
        "--python-exe",
        type=str,
        default="",
        help="Python executable used for spawned train jobs. Defaults to current interpreter.",
    )
    p.add_argument("--skip-training", action="store_true", help="Only generate graphs + plot + annealing report.")
    p.add_argument("--title-prefix", type=str, default="MaxCut Graph")
    p.add_argument("--anneal-steps", type=int, default=50_000)
    p.add_argument("--anneal-restarts", type=int, default=8)
    p.add_argument("--anneal-t0", type=float, default=2.5)
    p.add_argument("--anneal-tf", type=float, default=1e-3)
    p.add_argument("--anneal-seed-offset", type=int, default=20_000)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_edges_path = _resolve_base_edges_path(str(args.base_edges_json))

    raw_edges = json.loads(base_edges_path.read_text(encoding="utf-8"))
    base_edges = _normalize_edges(raw_edges)
    base_n_qubits = _infer_n_qubits(base_edges)
    base_n_edges = len(base_edges)
    base_density = (2.0 * float(base_n_edges)) / float(base_n_qubits * (base_n_qubits - 1))

    p_layers_list = _parse_p_layers(args.p_layers)
    max_weights = _parse_int_list(args.max_weights, field_name="max-weights")
    weight_modes = _parse_weight_modes(args.weight_modes)
    no_build_min_abs_flag = bool(args.no_build_min_abs)

    if bool(args.min_abs_only):
        max_weights = [int(args.max_weight_disabled_value)]
        no_build_min_abs_flag = False
        print(
            "[policy] min_abs_only enabled: "
            f"build_min_abs={float(args.build_min_abs)} and max_weight={int(args.max_weight_disabled_value)}"
        )

    if str(args.n_qubits_list).strip() != "":
        n_qubits_list = _parse_int_list(args.n_qubits_list, field_name="n-qubits-list")
    else:
        n_qubits_list = [int(base_n_qubits)]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_script = Path(__file__).resolve().parent / "train_qaoa_pps_cudaq.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    train_python_exe = str(args.python_exe).strip() or str(sys.executable)

    print(f"Base graph: n={base_n_qubits}, |E|={base_n_edges}, density={base_density:.6f}")
    print(f"Sweep n_qubits={n_qubits_list}")
    print(f"Generate graphs per n: {int(args.num_graphs)} | p-layers: {p_layers_list}")
    print(f"Weight modes: {weight_modes} | max_weights: {max_weights}")

    graph_meta_by_q: Dict[str, Dict[int, Dict[str, Any]]] = {}
    generated_graphs_by_q: Dict[str, List[Path]] = {}
    run_records: List[Dict[str, object]] = []

    for n_qubits in n_qubits_list:
        q_dir = out_dir / f"q{int(n_qubits):02d}"
        graphs_dir = q_dir / "graphs"
        plots_dir = q_dir / "plots"
        anneal_dir = q_dir / "anneal_reports"
        runs_dir = q_dir / "runs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        anneal_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        if int(n_qubits) == int(base_n_qubits):
            n_edges = int(base_n_edges)
            graph_source = "sampled_like_base_edges"
        else:
            complete_edges = int(n_qubits) * (int(n_qubits) - 1) // 2
            n_edges = int(round(base_density * complete_edges))
            n_edges = max(int(n_qubits) - 1, min(complete_edges, int(n_edges)))
            graph_source = "sampled_like_base_density"

        generated_graphs: List[Path] = []
        graph_meta: Dict[int, Dict[str, Any]] = {}
        for g_idx in range(1, int(args.num_graphs) + 1):
            g_seed = int(args.seed) + 10_000 + 1_000 * int(n_qubits) + g_idx
            edges = _sample_connected_graph_with_exact_edges(
                n_qubits=int(n_qubits),
                n_edges=int(n_edges),
                seed=g_seed,
                max_tries=int(args.max_tries),
            )
            g_path = graphs_dir / f"edges_graph_{g_idx:02d}.json"
            g_path.write_text(json.dumps([[u, v] for (u, v) in edges], indent=2), encoding="utf-8")
            generated_graphs.append(g_path)

            anneal_seed = int(args.seed) + int(args.anneal_seed_offset) + 1_000 * int(n_qubits) + g_idx
            viz_info = _build_graph_viz_and_sa(
                n_qubits=int(n_qubits),
                edges=edges,
                graph_index=g_idx,
                graph_source=graph_source,
                plots_dir=plots_dir,
                reports_dir=anneal_dir,
                title_prefix=str(args.title_prefix),
                anneal_steps=int(args.anneal_steps),
                anneal_restarts=int(args.anneal_restarts),
                anneal_t0=float(args.anneal_t0),
                anneal_tf=float(args.anneal_tf),
                anneal_seed=anneal_seed,
            )
            graph_meta[g_idx] = {
                "edges_json": str(g_path),
                "n_qubits": int(n_qubits),
                "n_edges": int(n_edges),
                "graph_source": graph_source,
                **viz_info,
            }
            print(
                f"Generated q={n_qubits} graph {g_idx}: {g_path} | "
                f"SA best cut={float(viz_info.get('anneal_best_cut', 0.0)):.1f} | plot={viz_info.get('plot')}"
            )

        graph_meta_by_q[str(int(n_qubits))] = graph_meta
        generated_graphs_by_q[str(int(n_qubits))] = generated_graphs

    failed = False

    if bool(args.skip_training):
        summary_path = out_dir / "summary.json"
        summary = {
            "base_graph": str(base_edges_path),
            "base_n_qubits": int(base_n_qubits),
            "base_n_edges": int(base_n_edges),
            "base_density": float(base_density),
            "n_qubits_list": [int(x) for x in n_qubits_list],
            "num_graphs": int(args.num_graphs),
            "p_layers": p_layers_list,
            "max_weights": max_weights,
            "weight_modes": weight_modes,
            "graph_meta_by_q": graph_meta_by_q,
            "records": run_records,
            "skip_training": True,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n" + "-" * 100)
        print(f"Saved summary (skip-training): {summary_path}")
        return

    for n_qubits in n_qubits_list:
        q_key = str(int(n_qubits))
        q_dir = out_dir / f"q{int(n_qubits):02d}"
        runs_dir = q_dir / "runs"
        generated_graphs = generated_graphs_by_q[q_key]
        graph_meta = graph_meta_by_q[q_key]

        for g_idx, g_path in enumerate(generated_graphs, start=1):
            for p_layers in p_layers_list:
                for weight_mode in weight_modes:
                    for max_weight in max_weights:
                        run_name = (
                            f"{args.run_prefix}_q{int(n_qubits):02d}_g{g_idx:02d}_p{int(p_layers):02d}"
                            f"_wm{weight_mode}_mw{int(max_weight):02d}"
                        )
                        run_seed = int(args.seed) + int(n_qubits) * 10_000 + g_idx * 1_000 + int(p_layers) * 10 + int(max_weight)

                        cmd = _build_train_cmd(
                            python_exe=train_python_exe,
                            train_script=train_script,
                            edges_json=g_path,
                            n_qubits=int(n_qubits),
                            p_layers=int(p_layers),
                            run_name=run_name,
                            output_dir=runs_dir,
                            steps=int(args.steps),
                            lr=float(args.lr),
                            delta_t=float(args.delta_t),
                            seed=int(run_seed),
                            chunk_size=int(args.chunk_size),
                            build_min_abs=float(args.build_min_abs),
                            rebuild_every=int(args.rebuild_every),
                            save_every=int(args.save_every),
                            log_every=int(args.log_every),
                            max_weight=int(max_weight),
                            weight_mode=str(weight_mode),
                            no_build_min_abs=bool(no_build_min_abs_flag),
                            init_mode=str(args.init_mode),
                            mixer_odd_start=float(args.mixer_odd_start),
                            mixer_odd_end=float(args.mixer_odd_end),
                        )

                        print("\n" + "=" * 100)
                        print(
                            f"Run q={n_qubits}, graph={g_idx}, p={p_layers}, mode={weight_mode}, "
                            f"max_weight={max_weight} -> {run_name}"
                        )
                        print(" ".join(cmd))

                        if args.dry_run:
                            status = "dry_run"
                            return_code = 0
                        else:
                            completed = subprocess.run(cmd, check=False)
                            return_code = int(completed.returncode)
                            status = "ok" if return_code == 0 else "failed"

                        ckpt_path = runs_dir / f"{run_name}.pt"
                        report_path = runs_dir / f"{run_name}.json"
                        run_records.append(
                            {
                                "n_qubits": int(n_qubits),
                                "graph_index": g_idx,
                                "graph_edges_json": str(g_path),
                                "graph_plot": str(graph_meta[g_idx]["plot"]),
                                "graph_anneal_report": str(graph_meta[g_idx]["anneal_report"]),
                                "graph_anneal_best_cut": float(graph_meta[g_idx]["anneal_best_cut"]),
                                "p_layers": int(p_layers),
                                "weight_mode": str(weight_mode),
                                "max_weight": int(max_weight),
                                "run_name": run_name,
                                "seed": int(run_seed),
                                "status": status,
                                "return_code": int(return_code),
                                "checkpoint": str(ckpt_path),
                                "report": str(report_path),
                            }
                        )

                        if status == "failed":
                            failed = True
                            print(
                                f"[FAILED] q={n_qubits}, graph={g_idx}, p={p_layers}, "
                                f"mode={weight_mode}, max_weight={max_weight}, return_code={return_code}"
                            )
                            if not args.continue_on_error:
                                summary_path = out_dir / "summary.json"
                                summary = {
                                    "base_graph": str(base_edges_path),
                                    "base_n_qubits": int(base_n_qubits),
                                    "base_n_edges": int(base_n_edges),
                                    "base_density": float(base_density),
                                    "n_qubits_list": [int(x) for x in n_qubits_list],
                                    "num_graphs": int(args.num_graphs),
                                    "p_layers": p_layers_list,
                                    "max_weights": max_weights,
                                    "weight_modes": weight_modes,
                                    "graph_meta_by_q": graph_meta_by_q,
                                    "records": run_records,
                                }
                                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                                raise RuntimeError(
                                    "Training failed for condition "
                                    f"q={n_qubits}, graph={g_idx}, p_layers={p_layers}, "
                                    f"weight_mode={weight_mode}, max_weight={max_weight}. "
                                    f"See summary: {summary_path}"
                                )

    summary_path = out_dir / "summary.json"
    summary = {
        "base_graph": str(base_edges_path),
        "base_n_qubits": int(base_n_qubits),
        "base_n_edges": int(base_n_edges),
        "base_density": float(base_density),
        "n_qubits_list": [int(x) for x in n_qubits_list],
        "num_graphs": int(args.num_graphs),
        "p_layers": p_layers_list,
        "max_weights": max_weights,
        "weight_modes": weight_modes,
        "graph_meta_by_q": graph_meta_by_q,
        "records": run_records,
        "skip_training": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "-" * 100)
    print(f"Saved summary: {summary_path}")
    n_ok = sum(1 for r in run_records if r["status"] in ("ok", "dry_run"))
    n_fail = sum(1 for r in run_records if r["status"] == "failed")
    print(f"Completed runs: {n_ok} ok, {n_fail} failed, total={len(run_records)}")
    if failed:
        raise RuntimeError("One or more runs failed. Check summary.json for details.")


if __name__ == "__main__":
    main()
