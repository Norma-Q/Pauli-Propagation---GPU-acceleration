from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_SWEEP_DIR = _THIS_DIR / "artifacts" / "sweep_multi_graphs_minabs_only_oddneg"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot best QAOA MaxCut results, top degenerate solutions, and per-graph "
            "sample distributions. Works in both CLI and notebooks."
        )
    )
    p.add_argument("--sweep-dir", type=str, default=str(_DEFAULT_SWEEP_DIR))
    p.add_argument("--analysis-dir", type=str, default="")
    p.add_argument("--summary-json", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--max-graphs-dist", type=int, default=12)
    p.add_argument("--max-qaoa-solutions", type=int, default=10)
    args, _unknown = p.parse_known_args(argv)
    return args


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_summary_json(sweep_dir: Path, explicit: str) -> Path:
    if str(explicit).strip() != "":
        p = Path(explicit)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"summary json not found: {p}")

    for cand in (sweep_dir / "summary.json", sweep_dir / "summary_partial.json"):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"Could not find summary json under {sweep_dir}")


def _resolve_analysis_dir(sweep_dir: Path, explicit: str) -> Path:
    if str(explicit).strip() != "":
        p = Path(explicit)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"analysis dir not found: {p}")
    p = sweep_dir / "analysis"
    if not p.exists():
        raise FileNotFoundError(f"analysis dir not found: {p}")
    return p.resolve()


def _resolve_out_dir(analysis_dir: Path, explicit: str) -> Path:
    return Path(explicit).resolve() if str(explicit).strip() != "" else (analysis_dir / "maxcut_viz").resolve()


def _extract_graph_root(summary: Dict[str, Any], sweep_dir: Path) -> Optional[Path]:
    records = summary.get("records", []) if isinstance(summary, dict) else []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        edges_json = str(rec.get("graph_edges_json", "")).strip()
        if edges_json != "":
            p = Path(edges_json)
            if p.exists():
                return p.parent.resolve()

    base_graph = str(summary.get("base_graph", "")).strip() if isinstance(summary, dict) else ""
    if base_graph != "":
        p = Path(base_graph)
        if p.exists():
            return p.parent.resolve()

    for cand in (sweep_dir / "q30" / "graphs", sweep_dir / "graphs"):
        if cand.exists():
            return cand.resolve()
    return None


def _extract_qroot_from_record(rec: Dict[str, Any], summary: Dict[str, Any], sweep_dir: Path) -> Optional[Path]:
    edges_json = str(rec.get("graph_edges_json", "")).strip()
    if edges_json != "":
        p = Path(edges_json)
        if p.exists() and p.parent.name == "graphs":
            return p.parent.parent.resolve()

    base_graph = str(summary.get("base_graph", "")).strip() if isinstance(summary, dict) else ""
    if base_graph != "":
        p = Path(base_graph)
        if p.exists() and p.parent.name == "graphs":
            return p.parent.parent.resolve()

    for cand in (sweep_dir / "q30", sweep_dir):
        if cand.exists():
            return cand.resolve()
    return None


def _find_anneal_report(qroot: Optional[Path], graph_index: int) -> Optional[Path]:
    if qroot is None:
        return None
    for cand in (
        qroot / "anneal_reports" / f"graph_{int(graph_index):02d}_anneal.json",
        qroot / "anneal_reports" / f"graph_{int(graph_index)}_anneal.json",
    ):
        if cand.exists():
            return cand.resolve()
    return None


def _load_anneal_result(qroot: Optional[Path], graph_index: int) -> Dict[str, Any]:
    report = _find_anneal_report(qroot, graph_index)
    if report is None:
        return {}
    payload = _read_json(report)
    annealing = payload.get("annealing", {}) if isinstance(payload, dict) else {}
    return annealing if isinstance(annealing, dict) else {}


def _latest_step_sampling(run_dir: Path) -> Optional[Path]:
    cands = list(run_dir.glob("step_*_sampling.json"))
    if not cands:
        return None

    def _step_num(path: Path) -> int:
        m = re.search(r"step_(\d+)_sampling\.json$", path.name)
        return int(m.group(1)) if m else -1

    return sorted(cands, key=_step_num)[-1]


def _bit_at(bitstring: str, q: int, bit_order: str) -> int:
    text = str(bitstring).strip()
    n = len(text)
    return int(text[int(q)]) if str(bit_order).lower() == "le" else int(text[n - 1 - int(q)])


def _infer_cut_from_bitstring(bitstring: str, edges: Sequence[Tuple[int, int]], bit_order: str) -> int:
    text = str(bitstring).strip()
    if text == "":
        raise ValueError("empty bitstring")
    total = 0
    for u, v in edges:
        if _bit_at(text, int(u), bit_order) != _bit_at(text, int(v), bit_order):
            total += 1
    return int(total)


def _extract_hist_from_counts(counts: Dict[str, Any], edges: Sequence[Tuple[int, int]], bit_order: str) -> Dict[int, float]:
    hist: Dict[int, float] = defaultdict(float)
    for bitstring, weight in counts.items():
        try:
            cut = _infer_cut_from_bitstring(str(bitstring), edges, bit_order)
            hist[int(cut)] += float(weight)
        except Exception:
            continue
    return dict(hist)


def _extract_hist(payload: Dict[str, Any], edges: Sequence[Tuple[int, int]]) -> Dict[int, float]:
    counts = payload.get("counts")
    if isinstance(counts, dict) and counts:
        return _extract_hist_from_counts(counts, edges, str(payload.get("bit_order", "le")))

    for key in ("cut_hist", "hist", "histogram"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            out: Dict[int, float] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = float(v)
                except Exception:
                    continue
            if out:
                return out
    return {}


def _normalize_hist(hist: Dict[int, float]) -> Dict[int, float]:
    if not hist:
        return {}
    total = float(sum(hist.values()))
    if total <= 0.0:
        return {}
    return {int(k): float(v) / total for k, v in sorted(hist.items())}


def _mean_from_prob(hist_prob: Dict[int, float]) -> float:
    return float(sum(float(k) * float(v) for k, v in hist_prob.items())) if hist_prob else float("nan")


def _best_row_key(row: Dict[str, Any]) -> Tuple[float, float, int]:
    p_layers = row.get("p_layers")
    return (
        float(row.get("qaoa_best_cut", float("-inf"))),
        float(row.get("qaoa_mean_cut", float("-inf"))),
        -int(p_layers) if p_layers is not None else 0,
    )


def _top_best_bitstrings_from_counts(
    counts: Dict[str, Any],
    edges: Sequence[Tuple[int, int]],
    bit_order: str,
    limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_weight = 0.0
    for bitstring, weight in counts.items():
        try:
            w = float(weight)
            cut = int(_infer_cut_from_bitstring(str(bitstring), edges, bit_order))
        except Exception:
            continue
        rows.append({"bitstring": str(bitstring), "cut": cut, "weight": w})
        total_weight += w

    if not rows:
        return []

    best_cut = max(int(r["cut"]) for r in rows)
    best_rows = [r for r in rows if int(r["cut"]) == best_cut]
    best_rows.sort(key=lambda r: (-float(r["weight"]), str(r["bitstring"])))

    out: List[Dict[str, Any]] = []
    for rank, row in enumerate(best_rows[: max(1, int(limit))], start=1):
        prob = float(row["weight"]) / total_weight if total_weight > 0.0 else float("nan")
        out.append(
            {
                "rank": int(rank),
                "bitstring": str(row["bitstring"]),
                "cut": int(row["cut"]),
                "weight": float(row["weight"]),
                "prob": float(prob),
            }
        )
    return out


def _circular_layout(n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_qubits), endpoint=False)
    return np.cos(theta), np.sin(theta)


def _plot_partition(
    ax: Axes,
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    bitstring: str,
    bit_order: str,
    title: str,
    cut_value: float,
) -> None:
    x, y = _circular_layout(int(n_qubits))
    partition = np.asarray([_bit_at(bitstring, q, bit_order) for q in range(int(n_qubits))], dtype=np.int64)

    for u, v in edges:
        uu = int(u)
        vv = int(v)
        is_cut = partition[uu] != partition[vv]
        ax.plot(
            [x[uu], x[vv]],
            [y[uu], y[vv]],
            color="#2ca02c" if is_cut else "#b0b7c3",
            alpha=0.95 if is_cut else 0.75,
            linewidth=2.2 if is_cut else 1.0,
            linestyle="-" if is_cut else "--",
            zorder=1,
        )

    node_colors = np.where(partition > 0, "#ff7f0e", "#1f77b4")
    ax.scatter(x, y, c=node_colors, s=105, edgecolors="black", linewidths=0.6, zorder=3)
    for q in range(int(n_qubits)):
        ax.text(x[q] * 1.08, y[q] * 1.08, str(q), fontsize=8, ha="center", va="center")

    ax.set_title(f"{title}\ncut={float(cut_value):.1f}", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "partition 0: blue\npartition 1: orange\ncut edges: green",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def _save_partition_gallery(
    out_path: Path,
    *,
    graph_index: int,
    run_name: str,
    p_layers: int,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    qaoa_solutions: Sequence[Dict[str, Any]],
    anneal_bitstring: Optional[str],
    anneal_cut: float,
    bit_order: str,
) -> None:
    panels = list(qaoa_solutions)
    if anneal_bitstring:
        panels.append(
            {
                "rank": None,
                "bitstring": str(anneal_bitstring),
                "cut": float(anneal_cut),
                "weight": None,
                "prob": None,
                "anneal": True,
            }
        )

    n_panels = len(panels)
    ncols = min(3, max(1, n_panels))
    nrows = int(math.ceil(float(n_panels) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.2 * nrows), squeeze=False)
    flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, sol in enumerate(panels):
        if sol.get("anneal", False):
            title = f"Graph {int(graph_index)} | Annealing best"
        else:
            prob = sol.get("prob")
            prob_text = "n/a" if prob is None or not np.isfinite(float(prob)) else f"{100.0 * float(prob):.2f}%"
            title = (
                f"Graph {int(graph_index)} | QAOA #{int(sol['rank'])} ({run_name}, p={int(p_layers)})\n"
                f"count={int(float(sol['weight']))}, prob={prob_text}"
            )
        _plot_partition(
            flat_axes[idx],
            n_qubits=int(n_qubits),
            edges=edges,
            bitstring=str(sol["bitstring"]),
            bit_order=bit_order,
            title=title,
            cut_value=float(sol["cut"]),
        )

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_distribution_plot(
    out_path: Path,
    *,
    graph_index: int,
    run_name: str,
    p_layers: int,
    hist_prob: Dict[int, float],
    qaoa_best_cut: float,
    anneal_best_cut: float,
) -> None:
    xs = np.asarray(sorted(hist_prob.keys()), dtype=np.int64)
    ys = np.asarray([hist_prob[int(k)] for k in xs], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(xs, ys, width=0.8, alpha=0.88, color="#5f9ea0", edgecolor="black", linewidth=0.6, label="QAOA samples")
    ax.axvline(float(qaoa_best_cut), color="tab:green", linestyle="--", linewidth=2.0, label="QAOA best")
    if np.isfinite(float(anneal_best_cut)):
        ax.axvline(float(anneal_best_cut), color="tab:red", linestyle="-.", linewidth=2.0, label="Anneal best")
    ax.set_title(f"Graph {int(graph_index)} | {run_name} (p={int(p_layers)})")
    ax.set_xlabel("Cut value")
    ax.set_ylabel("Probability")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    *,
    sweep_dir: str | Path,
    analysis_dir: str | Path | None = None,
    summary_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    max_graphs_dist: int = 12,
    max_qaoa_solutions: int = 10,
) -> Dict[str, Any]:
    sweep_dir_path = Path(sweep_dir).resolve()
    if not sweep_dir_path.exists():
        raise FileNotFoundError(f"sweep dir not found: {sweep_dir_path}")

    summary_path = _resolve_summary_json(sweep_dir_path, "" if summary_json is None else str(summary_json))
    analysis_dir_path = _resolve_analysis_dir(sweep_dir_path, "" if analysis_dir is None else str(analysis_dir))
    out_dir_path = _resolve_out_dir(analysis_dir_path, "" if out_dir is None else str(out_dir))
    out_dir_path.mkdir(parents=True, exist_ok=True)

    summary = _read_json(summary_path)
    records = summary.get("records", []) if isinstance(summary, dict) else []
    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError(f"No records found in summary: {summary_path}")

    graph_root = _extract_graph_root(summary, sweep_dir_path)
    run_rows: List[Dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status", "")).strip() != "ok":
            continue

        run_name = str(rec.get("run_name", "")).strip()
        if run_name == "":
            continue

        graph_index_raw = rec.get("graph_index")
        p_layers_raw = rec.get("p_layers")
        if graph_index_raw is None or p_layers_raw is None:
            continue

        graph_index = int(graph_index_raw)
        p_layers = int(p_layers_raw)

        edges_json = str(rec.get("graph_edges_json", "")).strip()
        if edges_json == "" and graph_root is not None:
            edges_json = str(graph_root / f"edges_graph_{graph_index:02d}.json")
        edges_path = Path(edges_json)
        if not edges_path.exists():
            continue
        edges = [(int(u), int(v)) for (u, v) in _read_json(edges_path)]

        run_dir = analysis_dir_path / run_name
        if not run_dir.exists():
            continue
        sampling_path = run_dir / "final_sampling.json"
        if not sampling_path.exists():
            latest = _latest_step_sampling(run_dir)
            if latest is None:
                continue
            sampling_path = latest

        payload = _read_json(sampling_path)
        counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
        if not isinstance(counts, dict):
            counts = {}

        hist_prob = _normalize_hist(_extract_hist(payload, edges))
        if not hist_prob:
            continue

        bit_order = str(payload.get("bit_order", "le"))
        qroot = _extract_qroot_from_record(rec, summary, sweep_dir_path)
        anneal_result = _load_anneal_result(qroot, graph_index)
        anneal_best_cut_raw = anneal_result.get("best_cut", rec.get("graph_anneal_best_cut", payload.get("graph_anneal_best_cut")))
        anneal_best_cut = float(anneal_best_cut_raw) if anneal_best_cut_raw is not None else float("nan")
        anneal_best_bitstring = anneal_result.get("best_bitstring_leftmost_q0")
        anneal_best_bitstring = None if anneal_best_bitstring is None else str(anneal_best_bitstring)

        qaoa_solutions = _top_best_bitstrings_from_counts(counts, edges, bit_order, limit=int(max_qaoa_solutions))

        best_info = payload.get("best", {}) if isinstance(payload, dict) else {}
        qaoa_best_bitstring = None
        if isinstance(best_info, dict):
            raw_bitstring = best_info.get("bitstring")
            if raw_bitstring is not None and str(raw_bitstring).strip() != "":
                qaoa_best_bitstring = str(raw_bitstring)
        if qaoa_best_bitstring is None and qaoa_solutions:
            qaoa_best_bitstring = str(qaoa_solutions[0]["bitstring"])
        if qaoa_best_bitstring is None:
            continue

        qaoa_best_cut = float(_infer_cut_from_bitstring(qaoa_best_bitstring, edges, bit_order))
        if not qaoa_solutions:
            qaoa_solutions = [
                {
                    "rank": 1,
                    "bitstring": qaoa_best_bitstring,
                    "cut": int(qaoa_best_cut),
                    "weight": 0.0,
                    "prob": float("nan"),
                }
            ]

        run_rows.append(
            {
                "run_name": run_name,
                "graph_index": int(graph_index),
                "n_qubits": int(rec.get("n_qubits", len({q for e in edges for q in e}) + 1)),
                "p_layers": int(p_layers),
                "edges_json": str(edges_path),
                "sampling_json": str(sampling_path),
                "bit_order": bit_order,
                "edges": edges,
                "qaoa_best_bitstring": qaoa_best_bitstring,
                "qaoa_best_cut": qaoa_best_cut,
                "qaoa_solutions": qaoa_solutions,
                "qaoa_mean_cut": float(_mean_from_prob(hist_prob)),
                "anneal_best_cut": float(anneal_best_cut),
                "anneal_best_bitstring": anneal_best_bitstring,
                "hist_prob": hist_prob,
            }
        )

    if not run_rows:
        raise RuntimeError("No usable run rows were found in the analysis artifacts.")

    best_by_graph: Dict[int, Dict[str, Any]] = {}
    for row in run_rows:
        g = int(row["graph_index"])
        if g not in best_by_graph or _best_row_key(row) > _best_row_key(best_by_graph[g]):
            best_by_graph[g] = row

    selected = [best_by_graph[g] for g in sorted(best_by_graph)]
    csv_path = out_dir_path / "best_qaoa_vs_anneal_by_graph.csv"
    partition_dir = out_dir_path / "maxcut_partition_viz"
    dist_dir = out_dir_path / "sample_distributions"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_index",
                "run_name",
                "p_layers",
                "qaoa_num_best_solutions",
                "qaoa_best_bitstring",
                "qaoa_best_cut",
                "qaoa_mean_cut",
                "anneal_best_bitstring",
                "anneal_best_cut",
                "gap_qaoa_minus_anneal",
                "sampling_json",
                "partition_plot",
                "distribution_plot",
            ],
        )
        writer.writeheader()

        for row in selected:
            partition_plot_path = partition_dir / f"graph_{int(row['graph_index']):02d}_partition.png"
            dist_plot_path = dist_dir / f"graph_{int(row['graph_index']):02d}_distribution.png"
            anneal_best = float(row["anneal_best_cut"])

            _save_partition_gallery(
                partition_plot_path,
                graph_index=int(row["graph_index"]),
                run_name=str(row["run_name"]),
                p_layers=int(row["p_layers"]),
                n_qubits=int(row["n_qubits"]),
                edges=row["edges"],
                qaoa_solutions=row["qaoa_solutions"],
                anneal_bitstring=row.get("anneal_best_bitstring"),
                anneal_cut=anneal_best,
                bit_order=str(row["bit_order"]),
            )
            _save_distribution_plot(
                dist_plot_path,
                graph_index=int(row["graph_index"]),
                run_name=str(row["run_name"]),
                p_layers=int(row["p_layers"]),
                hist_prob=row["hist_prob"],
                qaoa_best_cut=float(row["qaoa_best_cut"]),
                anneal_best_cut=anneal_best,
            )

            writer.writerow(
                {
                    "graph_index": int(row["graph_index"]),
                    "run_name": str(row["run_name"]),
                    "p_layers": int(row["p_layers"]),
                    "qaoa_num_best_solutions": int(len(row["qaoa_solutions"])),
                    "qaoa_best_bitstring": str(row["qaoa_best_bitstring"]),
                    "qaoa_best_cut": float(row["qaoa_best_cut"]),
                    "qaoa_mean_cut": float(row["qaoa_mean_cut"]),
                    "anneal_best_bitstring": row.get("anneal_best_bitstring"),
                    "anneal_best_cut": anneal_best,
                    "gap_qaoa_minus_anneal": float(row["qaoa_best_cut"]) - anneal_best if np.isfinite(anneal_best) else np.nan,
                    "sampling_json": str(row["sampling_json"]),
                    "partition_plot": str(partition_plot_path),
                    "distribution_plot": str(dist_plot_path),
                }
            )

    x = np.arange(len(selected), dtype=float)
    qaoa_best_arr = np.asarray([float(r["qaoa_best_cut"]) for r in selected], dtype=np.float64)
    anneal_arr = np.asarray([float(r["anneal_best_cut"]) for r in selected], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(10.0, 0.9 * len(selected) + 4.0), 5.0))
    width = 0.42
    ax.bar(x - width / 2.0, qaoa_best_arr, width=width, label="QAOA best cut")
    ax.bar(x + width / 2.0, anneal_arr, width=width, label="Classical annealing best cut")
    ax.set_xticks(x)
    ax.set_xticklabels([f"g{int(r['graph_index'])}" for r in selected], rotation=45)
    ax.set_ylabel("Cut value")
    ax.set_title("Best MaxCut by graph: QAOA vs Classical Annealing")
    ax.legend()
    fig.tight_layout()
    best_plot_path = out_dir_path / "best_qaoa_vs_anneal_by_graph.png"
    fig.savefig(best_plot_path, dpi=160)
    plt.close(fig)

    dist_rows = selected[: max(1, int(max_graphs_dist))]
    return {
        "summary_json": str(summary_path),
        "analysis_dir": str(analysis_dir_path),
        "out_dir": str(out_dir_path),
        "csv": str(csv_path),
        "best_plot": str(best_plot_path),
        "partition_plots": [str(partition_dir / f"graph_{int(row['graph_index']):02d}_partition.png") for row in selected],
        "distribution_plots": [str(dist_dir / f"graph_{int(row['graph_index']):02d}_distribution.png") for row in dist_rows],
        "n_graphs": int(len(selected)),
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    result = run_analysis(
        sweep_dir=args.sweep_dir,
        analysis_dir=args.analysis_dir if str(args.analysis_dir).strip() != "" else None,
        summary_json=args.summary_json if str(args.summary_json).strip() != "" else None,
        out_dir=args.out_dir if str(args.out_dir).strip() != "" else None,
        max_graphs_dist=int(args.max_graphs_dist),
        max_qaoa_solutions=int(args.max_qaoa_solutions),
    )
    print(f"Saved: {result['csv']}")
    print(f"Saved: {result['best_plot']}")
    for path in result["partition_plots"]:
        print(f"Saved: {path}")
    for path in result["distribution_plots"]:
        print(f"Saved: {path}")
    return result


if __name__ == "__main__":
    main()
