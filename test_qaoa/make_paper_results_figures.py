from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/ubuntu/PPS-lab/test_qaoa")
RESULTS_DIR = ROOT / "results"
GRAPH_DIR = ROOT / "graph"
OUT_DIR = ROOT / "paper_figures"
NOTEBOOK_EXPORT_DIR = OUT_DIR / "notebook_exports"


def load_edges_count(n_qubits: int) -> int:
    edges = json.loads((GRAPH_DIR / f"Q{n_qubits}_edges.json").read_text())
    return int(len(edges))


def parse_run_name(name: str) -> tuple[int, int, int]:
    match = re.match(r"Q(\d+)_L(\d+)(?:_mw(\d+))?$", name)
    if match is None:
        raise ValueError(f"Unexpected run name: {name}")
    n_qubits = int(match.group(1))
    depth = int(match.group(2))
    max_weight = int(match.group(3)) if match.group(3) else 3
    return n_qubits, depth, max_weight


def load_training_history(run_name: str) -> dict[str, list[float]]:
    path = RESULTS_DIR / run_name / "training_log.json"
    payload = json.loads(path.read_text())
    return payload["training_history"]


def load_validation_progress(run_name: str) -> tuple[list[float], list[float]]:
    path = RESULTS_DIR / run_name / "validation_progress.json"
    payload = json.loads(path.read_text())
    thresholds = [float(x) for x in payload["thresholds_done"]]
    values = [float(x) for x in payload["expected_cuts"]]
    clean_thresholds: list[float] = []
    clean_values: list[float] = []
    for threshold, value in zip(thresholds, values):
        if math.isnan(value):
            continue
        clean_thresholds.append(threshold)
        clean_values.append(value)
    return clean_thresholds, clean_values


def compute_relative_span_percent(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(100.0 * (arr.max() - arr.min()) / arr.mean())


def compute_tail_span_percent(
    thresholds: list[float],
    values: list[float],
    *,
    cutoff: float = 1e-2,
) -> float:
    tail_values = [value for threshold, value in zip(thresholds, values) if threshold <= cutoff]
    if len(tail_values) < 2:
        return math.nan
    return compute_relative_span_percent(tail_values)


def filter_threshold_window(
    thresholds: list[float],
    values: list[float],
    *,
    max_threshold: float,
) -> tuple[list[float], list[float]]:
    selected = [(threshold, value) for threshold, value in zip(thresholds, values) if threshold <= max_threshold]
    return [item[0] for item in selected], [item[1] for item in selected]


def collect_depth_sweep_points(n_qubits: int) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    n_edges = load_edges_count(n_qubits)
    for run_dir in sorted(RESULTS_DIR.glob(f"Q{n_qubits}_L*")):
        log_path = run_dir / "training_log.json"
        if not log_path.exists():
            continue
        run_n, depth, max_weight = parse_run_name(run_dir.name)
        if max_weight != 3:
            continue
        history = load_training_history(run_dir.name)
        initial_cut = float(history["expected_cut"][0])
        final_cut = float(history["expected_cut"][-1])
        out.append((depth, final_cut / n_edges, (final_cut - initial_cut) / n_edges))
    return sorted(out)


def collect_validated_spans() -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        validation_path = run_dir / "validation_progress.json"
        if not validation_path.exists():
            continue
        thresholds, values = load_validation_progress(run_dir.name)
        if len(values) < 7:
            continue
        _, values_from_5e2 = filter_threshold_window(thresholds, values, max_threshold=5e-2)
        rel_span_from_5e2 = compute_relative_span_percent(values_from_5e2)
        rel_span_from_1e2 = compute_tail_span_percent(thresholds, values)
        out.append((run_dir.name, rel_span_from_5e2, rel_span_from_1e2))
    return sorted(out, key=lambda item: item[1], reverse=True)


def collect_fraction_curve(depth: int) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    pattern = re.compile(rf"Q(\d+)_L{depth}(?:_mw(\d+))?$")
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        log_path = run_dir / "training_log.json"
        if not log_path.exists():
            continue
        match = pattern.match(run_dir.name)
        if match is None:
            continue
        n_qubits = int(match.group(1))
        if n_qubits < 40:
            continue
        n_edges = load_edges_count(n_qubits)
        history = load_training_history(run_dir.name)
        final_cut = float(history["expected_cut"][-1])
        out.append((n_qubits, final_cut / n_edges))
    return sorted(out)


def plot_depth_and_validation() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    depth_systems = {
        40: ("Q40", "#1f77b4"),
        45: ("Q45", "#ff7f0e"),
        50: ("Q50", "#9467bd"),
        55: ("Q55", "#2ca02c"),
    }

    ax = axes[0, 0]
    for n_qubits, (label, color) in depth_systems.items():
        points = collect_depth_sweep_points(n_qubits)
        depths = [x[0] for x in points]
        finals = [x[1] for x in points]
        ax.plot(depths, finals, marker="o", linewidth=2.2, markersize=6, color=color, label=label)
    ax.axhline(0.5, color="dimgray", linestyle="--", linewidth=1.2, label="Random-cut baseline")
    ax.set_xlabel("QAOA depth L")
    ax.set_ylabel(
        "Edge-normalized final expected cut\n"
        r"$\mathbb{E}[\mathrm{cut}]_{\mathrm{final}}/N_{\mathrm{edges}}$"
    )
    ax.set_title("Final cut fraction rises with depth")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for n_qubits, (label, color) in depth_systems.items():
        points = collect_depth_sweep_points(n_qubits)
        depths = [x[0] for x in points]
        gains = [x[2] for x in points]
        ax.plot(depths, gains, marker="o", linewidth=2.2, markersize=6, color=color, label=label)
    ax.set_xlabel("QAOA depth L")
    ax.set_ylabel(
        "Edge-normalized improvement\n"
        r"$\Delta \mathbb{E}[\mathrm{cut}]/N_{\mathrm{edges}}$"
    )
    ax.set_title("Relative gain narrows as depth grows")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    representative_runs = {
        "Q50_L9": ("Q50, L=9", "#9467bd"),
        "Q70_L5": ("Q70, L=5", "#d62728"),
        "Q100_L3": ("Q100, L=3", "#1f77b4"),
    }
    for run_name, (label, color) in representative_runs.items():
        thresholds, values = load_validation_progress(run_name)
        thresholds, values = filter_threshold_window(thresholds, values, max_threshold=5e-2)
        ax.plot(thresholds, values, marker="o", linewidth=2.0, markersize=5, color=color, label=label)
    ax.axvline(1e-2, color="dimgray", linestyle="--", linewidth=1.2, label=r"$10^{-2}$ cutoff")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"build\_min\_abs threshold")
    ax.set_ylabel("Validated expected cut")
    ax.set_title(r"Validation curves after excluding the harsh $10^{-1}$ point")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    ax = axes[1, 1]
    validated_spans = collect_validated_spans()
    labels = [name.replace("_mw3", "").replace("_mw4", "").replace("_", "-") for name, _, _ in validated_spans]
    spans_from_5e2 = [value for _, value, _ in validated_spans]
    spans_from_1e2 = [value for _, _, value in validated_spans]
    x = np.arange(len(labels))
    ax.bar(x, spans_from_5e2, color="#6baed6", edgecolor="black", linewidth=0.4, label=r"Thresholds $\leq 5\times 10^{-2}$")
    ax.plot(x, spans_from_1e2, color="#d95f0e", marker="o", linewidth=1.6, markersize=4, label=r"Thresholds $\leq 10^{-2}$")
    ax.axhline(1.0, color="dimgray", linestyle=":", linewidth=1.2, label="1%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Relative validation span [%]")
    ax.set_title(r"Span shrinks sharply once the $10^{-1}$ point is removed")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False)

    fig.suptitle("Large-scale PPS optimization: depth trend and threshold-refinement analysis", fontsize=14, y=0.99)
    fig.tight_layout()
    out_path = OUT_DIR / "paper_results_depth_validation.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scaling_extension() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    ax = axes[0]
    representative_runs = {
        "Q55_L9": ("Q55, L=9, mw=3", "#2ca02c"),
        "Q70_L5": ("Q70, L=5, mw=3", "#d62728"),
        "Q100_L5": ("Q100, L=5, mw=3", "#1f77b4"),
    }
    for run_name, (label, color) in representative_runs.items():
        n_qubits, _, _ = parse_run_name(run_name)
        n_edges = load_edges_count(n_qubits)
        history = load_training_history(run_name)
        steps = history["step"]
        fractions = np.asarray(history["expected_cut"], dtype=np.float64) / float(n_edges)
        ax.plot(steps, fractions, linewidth=2.2, color=color, label=label)
    ax.axhline(0.5, color="dimgray", linestyle="--", linewidth=1.2, label="Random-cut baseline")
    ax.set_xlabel("Training step")
    ax.set_ylabel(
        "Edge-normalized expected cut\n"
        r"$\mathbb{E}[\mathrm{cut}]/N_{\mathrm{edges}}$"
    )
    ax.set_title("Representative normalized training traces")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for depth, color in [(3, "#6a3d9a"), (5, "#1f78b4")]:
        curve = collect_fraction_curve(depth)
        qubits = [item[0] for item in curve]
        values = [item[1] for item in curve]
        ax.plot(qubits, values, marker="o", linewidth=2.2, markersize=6, color=color, label=f"L={depth}")
    ax.axhline(0.5, color="dimgray", linestyle="--", linewidth=1.2, label="Random-cut baseline")
    ax.set_xlabel("Qubit count")
    ax.set_ylabel(
        "Edge-normalized final expected cut\n"
        r"$\mathbb{E}[\mathrm{cut}]_{\mathrm{final}}/N_{\mathrm{edges}}$"
    )
    ax.set_title("Archived large-scale runs from 40Q to 100Q")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle("PPS optimization remains effective up to 100 qubits", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = OUT_DIR / "paper_results_scaling_extension.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_validation_representatives() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.2, 4.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.0], hspace=0.05, wspace=0.22)

    representative_runs = {
        "Q70_L5": ("Q70, L=5", "#d62728"),
        "Q100_L3": ("Q100, L=3", "#1f77b4"),
    }

    top_axes: list[plt.Axes] = []
    bottom_axes: list[plt.Axes] = []

    for col, (run_name, (title, color)) in enumerate(representative_runs.items()):
        ax_top = fig.add_subplot(grid[0, col])
        ax_bottom = fig.add_subplot(grid[1, col], sharex=ax_top)
        top_axes.append(ax_top)
        bottom_axes.append(ax_bottom)

        thresholds, values = load_validation_progress(run_name)
        thresholds, values = filter_threshold_window(thresholds, values, max_threshold=5e-2)
        train_history = load_training_history(run_name)
        train_final = float(train_history["expected_cut"][-1])
        span_from_5e2 = compute_relative_span_percent(values)
        span_from_1e2 = compute_tail_span_percent(thresholds, values, cutoff=1e-2)
        strictest_diff = float(values[-1] - train_final)

        upper_values = values + [train_final]
        upper_span = max(upper_values) - min(upper_values)
        upper_pad = max(0.8, 0.35 * upper_span)
        upper_min = math.floor((min(upper_values) - upper_pad) * 2.0) / 2.0
        upper_max = math.ceil((max(upper_values) + upper_pad) * 2.0) / 2.0
        lower_max = max(10.0, min(30.0, 0.06 * upper_min))

        for ax in (ax_top, ax_bottom):
            ax.plot(thresholds, values, marker="o", linewidth=2.2, markersize=5, color=color, label="validated value")
            ax.axhline(train_final, color="black", linestyle="--", linewidth=1.2, label="training endpoint")
            ax.axvline(1e-2, color="dimgray", linestyle=":", linewidth=1.1)
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.grid(alpha=0.25)

        ax_top.set_ylim(upper_min, upper_max)
        ax_bottom.set_ylim(0.0, lower_max)
        ax_bottom.set_yticks([0.0])
        ax_bottom.set_yticklabels(["0"])
        ax_top.set_title(title)
        ax_bottom.set_xlabel(r"build\_min\_abs threshold")

        if col == 0:
            ax_top.set_ylabel("Expected cut")
        else:
            ax_top.set_ylabel("")
            ax_bottom.set_yticklabels([])

        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False, labelbottom=False)
        ax_bottom.xaxis.tick_bottom()

        d = 0.012
        kwargs_top = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)
        kwargs_bottom = dict(transform=ax_bottom.transAxes, color="black", clip_on=False, linewidth=1.0)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom)
        ax_bottom.text(
            -0.08,
            1.02,
            r"$\sim$",
            transform=ax_bottom.transAxes,
            fontsize=16,
            ha="center",
            va="bottom",
        )

        ax_top.text(
            0.03,
            0.05,
            (
                rf"span $(5\times 10^{{-2}}\!\to\!10^{{-4}})$: {span_from_5e2:.2f}%"
                "\n"
                rf"span $(10^{{-2}}\!\to\!10^{{-4}})$: {span_from_1e2:.2f}%"
                "\n"
                rf"$10^{{-4}}$ - train: {strictest_diff:+.3f}"
            ),
            transform=ax_top.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
        )

    top_axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        r"Representative threshold-refinement curves with a zero-anchored broken y-axis",
        fontsize=14,
        y=1.02,
    )
    fig.subplots_adjust(top=0.83, bottom=0.16, left=0.08, right=0.98)
    out_path = OUT_DIR / "paper_results_validation_representatives.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_notebook_cudaq_transfer() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    panels = [
        ("qaoa_notebook_training_curve.png", "PPS training trace"),
        ("qaoa_notebook_cudaq_hist.png", "CUDA-Q sampled cut distribution"),
    ]

    for ax, (filename, title) in zip(axes, panels):
        image_path = NOTEBOOK_EXPORT_DIR / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Missing notebook export: {image_path}")
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("off")

    fig.suptitle("30-qubit PPS optimization transfers to CUDA-Q sampling", fontsize=14, y=0.98)
    fig.tight_layout()
    out_path = OUT_DIR / "paper_results_30q_cudaq_check.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    depth_path = plot_depth_and_validation()
    scaling_path = plot_scaling_extension()
    representative_validation_path = plot_validation_representatives()
    print(depth_path)
    print(scaling_path)
    print(representative_validation_path)
    try:
        notebook_path = plot_notebook_cudaq_transfer()
    except FileNotFoundError as exc:
        print(f"[warn] {exc}")
    else:
        print(notebook_path)


if __name__ == "__main__":
    main()
