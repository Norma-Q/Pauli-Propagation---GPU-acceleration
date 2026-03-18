from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TEST_QAOA = _REPO_ROOT / "test_qaoa"
for path in (str(_REPO_ROOT), str(_TEST_QAOA)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qaoa_surrogate_common import make_ring_chord_graph


def _is_connected(n_qubits: int, edges) -> bool:
    n = int(n_qubits)
    if n <= 1:
        return True
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                stack.append(w)
    return all(seen)


def _make_erdos_renyi_graph(
    *, n_qubits: int, edge_prob: float, seed: int, ensure_connected: bool, max_tries: int
):
    n = int(n_qubits)
    p = float(edge_prob)
    if not (0.0 < p <= 1.0):
        raise ValueError("edge_prob must be in (0,1].")
    rng = np.random.default_rng(int(seed))
    for _ in range(int(max_tries)):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if float(rng.random()) < p:
                    edges.append((i, j))
        if len(edges) == 0:
            continue
        if (not ensure_connected) or _is_connected(n, edges):
            return sorted(edges)
    raise RuntimeError("Failed to generate Erdos-Renyi graph with requested constraints.")


def _make_random_regular_graph(
    *, n_qubits: int, degree: int, seed: int, ensure_connected: bool, max_tries: int
):
    n = int(n_qubits)
    d = int(degree)
    if d < 1 or d >= n:
        raise ValueError("regular_degree must satisfy 1 <= d < n_qubits.")
    if (n * d) % 2 != 0:
        raise ValueError("n_qubits * regular_degree must be even.")

    rng = np.random.default_rng(int(seed))
    for _ in range(int(max_tries)):
        stubs = []
        for v in range(n):
            stubs.extend([v] * d)
        stubs = np.asarray(stubs, dtype=np.int64)
        rng.shuffle(stubs)

        ok = True
        edge_set = set()
        for k in range(0, stubs.shape[0], 2):
            u = int(stubs[k])
            v = int(stubs[k + 1])
            if u == v:
                ok = False
                break
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                ok = False
                break
            edge_set.add((a, b))
        if not ok:
            continue

        edges = sorted(edge_set)
        if len(edges) != (n * d) // 2:
            continue
        if (not ensure_connected) or _is_connected(n, edges):
            return edges

    raise RuntimeError("Failed to generate random-regular graph with requested constraints.")


def _mask_to_leftmost_q0_bitstring(mask: int, n_qubits: int) -> str:
    return "".join("1" if ((int(mask) >> q) & 1) else "0" for q in range(int(n_qubits)))


def _cut_value_from_mask(mask: int, edges) -> float:
    val = 0.0
    for u, v in edges:
        if ((int(mask) >> int(u)) & 1) != ((int(mask) >> int(v)) & 1):
            val += 1.0
    return float(val)


def _simulated_annealing_maxcut(
    *,
    n_qubits: int,
    edges,
    steps: int,
    restarts: int,
    t0: float,
    tf: float,
    seed: int,
):
    n = int(n_qubits)
    if n <= 0:
        raise ValueError("n_qubits must be positive for simulated annealing")
    if int(steps) < 2:
        raise ValueError("anneal_steps must be >= 2")
    if int(restarts) < 1:
        raise ValueError("anneal_restarts must be >= 1")
    if float(t0) <= 0.0 or float(tf) <= 0.0:
        raise ValueError("anneal temperatures must be positive")

    adj = [[] for _ in range(n)]
    for u, v in edges:
        uu, vv = int(u), int(v)
        adj[uu].append(vv)
        adj[vv].append(uu)

    rng = np.random.default_rng(int(seed))
    alpha = (float(tf) / float(t0)) ** (1.0 / float(int(steps) - 1))

    best_mask = 0
    best_cut = -float("inf")

    for _ in range(int(restarts)):
        bits = rng.integers(0, 2, size=n, dtype=np.int8)
        mask = 0
        for q, b in enumerate(bits.tolist()):
            mask |= (int(b) & 1) << int(q)

        cut = _cut_value_from_mask(mask, edges)
        local_best_cut = cut
        local_best_mask = mask
        T = float(t0)

        for _ in range(int(steps)):
            u = int(rng.integers(0, n))
            bu = (mask >> u) & 1

            delta = 0.0
            for v in adj[u]:
                bv = (mask >> int(v)) & 1
                old_diff = bu ^ bv
                delta += 1.0 - 2.0 * float(old_diff)

            accept = False
            if delta >= 0.0:
                accept = True
            elif T > 0.0 and rng.random() < np.exp(delta / T):
                accept = True

            if accept:
                mask ^= 1 << u
                cut += delta
                if cut > local_best_cut:
                    local_best_cut = cut
                    local_best_mask = mask

            T *= alpha

        if local_best_cut > best_cut:
            best_cut = local_best_cut
            best_mask = local_best_mask

    return {
        "best_cut": float(best_cut),
        "best_mask": int(best_mask),
        "best_bitstring_leftmost_q0": _mask_to_leftmost_q0_bitstring(int(best_mask), n),
        "steps": int(steps),
        "restarts": int(restarts),
        "t0": float(t0),
        "tf": float(tf),
        "seed": int(seed),
    }


def _plot_graph(out_path: Path, n_qubits: int, edges, title: str, anneal_result) -> None:
    n = int(n_qubits)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    deg = np.zeros((n,), dtype=np.int64)
    for u, v in edges:
        deg[int(u)] += 1
        deg[int(v)] += 1

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axs[0]
    for u, v in edges:
        ax.plot([x[u], x[v]], [y[u], y[v]], color="#8aa2b2", alpha=0.8, linewidth=1.0)
    sc = ax.scatter(x, y, c=deg, cmap="viridis", s=85, edgecolors="black", linewidths=0.5)
    for i in range(n):
        ax.text(x[i] * 1.08, y[i] * 1.08, str(i), fontsize=8, ha="center", va="center")
    ax.set_title(f"{title}\nGraph view (color=degree)")
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Degree")

    if anneal_result is not None:
        ax.text(
            0.02,
            0.98,
            f"SA best cut: {anneal_result['best_cut']:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    ax2 = axs[1]
    bins = np.arange(int(np.min(deg)), int(np.max(deg)) + 2) - 0.5
    ax2.hist(deg, bins=bins, color="#5f9ea0", edgecolor="black", alpha=0.85, rwidth=0.9)
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Count")
    ax2.set_title("Degree histogram")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a MaxCut graph and plot it.")
    p.add_argument("--n-qubits", type=int, default=30)
    p.add_argument(
        "--graph-type",
        type=str,
        default="erdos_renyi",
        choices=["ring_chord", "erdos_renyi", "random_regular"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chord-shift", type=int, default=7)
    p.add_argument("--edge-prob", type=float, default=0.25)
    p.add_argument("--regular-degree", type=int, default=3)
    p.add_argument("--ensure-connected", action="store_true")
    p.add_argument("--max-tries", type=int, default=300)
    p.add_argument("--output-edges", type=str, default="QAOA/artifacts/maxcut_edges.json")
    p.add_argument("--output-plot", type=str, default="QAOA/artifacts/maxcut_graph.png")
    p.add_argument("--output-report", type=str, default="QAOA/artifacts/maxcut_report.json")
    p.add_argument("--title", type=str, default="MaxCut Graph")
    p.add_argument("--anneal-steps", type=int, default=50_000)
    p.add_argument("--anneal-restarts", type=int, default=8)
    p.add_argument("--anneal-t0", type=float, default=2.5)
    p.add_argument("--anneal-tf", type=float, default=1e-3)
    p.add_argument("--anneal-seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if str(args.graph_type) == "ring_chord":
        edges = make_ring_chord_graph(int(args.n_qubits), chord_shift=int(args.chord_shift))
        graph_source = "ring_chord"
        graph_params = {"chord_shift": int(args.chord_shift)}
    elif str(args.graph_type) == "erdos_renyi":
        edges = _make_erdos_renyi_graph(
            n_qubits=int(args.n_qubits),
            edge_prob=float(args.edge_prob),
            seed=int(args.seed),
            ensure_connected=bool(args.ensure_connected),
            max_tries=int(args.max_tries),
        )
        graph_source = "erdos_renyi"
        graph_params = {
            "edge_prob": float(args.edge_prob),
            "seed": int(args.seed),
            "ensure_connected": bool(args.ensure_connected),
            "max_tries": int(args.max_tries),
        }
    else:
        edges = _make_random_regular_graph(
            n_qubits=int(args.n_qubits),
            degree=int(args.regular_degree),
            seed=int(args.seed),
            ensure_connected=bool(args.ensure_connected),
            max_tries=int(args.max_tries),
        )
        graph_source = "random_regular"
        graph_params = {
            "regular_degree": int(args.regular_degree),
            "seed": int(args.seed),
            "ensure_connected": bool(args.ensure_connected),
            "max_tries": int(args.max_tries),
        }

    out_edges = Path(args.output_edges)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    out_edges.write_text(json.dumps([[int(u), int(v)] for (u, v) in edges], indent=2), encoding="utf-8")

    anneal_seed = int(args.seed) if args.anneal_seed is None else int(args.anneal_seed)
    anneal_result = _simulated_annealing_maxcut(
        n_qubits=int(args.n_qubits),
        edges=edges,
        steps=int(args.anneal_steps),
        restarts=int(args.anneal_restarts),
        t0=float(args.anneal_t0),
        tf=float(args.anneal_tf),
        seed=anneal_seed,
    )

    out_plot = Path(args.output_plot)
    _plot_graph(
        out_path=out_plot,
        n_qubits=int(args.n_qubits),
        edges=edges,
        title=f"{args.title} (n={int(args.n_qubits)}, |E|={len(edges)}, {graph_source})",
        anneal_result=anneal_result,
    )

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "n_qubits": int(args.n_qubits),
        "edges": [[int(u), int(v)] for (u, v) in edges],
        "graph": {
            "source": graph_source,
            "params": graph_params,
        },
        "annealing": anneal_result,
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"saved edges: {out_edges}")
    print(f"saved plot: {out_plot}")
    print(f"saved report: {out_report}")
    print(
        f"summary: n_qubits={int(args.n_qubits)} edges={len(edges)} graph_type={graph_source} params={graph_params}"
    )


if __name__ == "__main__":
    main()
