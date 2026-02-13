from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qaoa_surrogate_common import (
    load_edges_json,
    make_qaoa_problem_dict,
    make_ring_chord_graph,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a QAOA problem JSON + visualization plot.")
    p.add_argument("--name", type=str, default="qaoa_problem")
    p.add_argument("--description", type=str, default="")
    p.add_argument("--n-qubits", type=int, default=15)
    p.add_argument("--p-layers", type=int, default=4)
    p.add_argument("--delta-t", type=float, default=0.8)

    p.add_argument(
        "--graph-type",
        type=str,
        default="ring_chord",
        choices=["ring_chord", "erdos_renyi", "random_regular"],
        help="Graph generator type when --edges-json is not given.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for generated graphs.")

    p.add_argument("--edges-json", type=str, default="", help="Optional edge list [[u,v], ...].")
    p.add_argument("--chord-shift", type=int, default=7, help="Used for ring+chord graph.")
    p.add_argument("--edge-prob", type=float, default=0.25, help="Erdos-Renyi edge probability.")
    p.add_argument("--regular-degree", type=int, default=3, help="Random-regular degree.")
    p.add_argument("--ensure-connected", action="store_true", help="Regenerate until connected.")
    p.add_argument("--max-tries", type=int, default=300, help="Max retries for random graph generation.")

    p.add_argument("--output-json", type=str, default="test/artifacts/qaoa_problem.json")
    p.add_argument("--output-plot", type=str, default="test/artifacts/qaoa_problem.png")
    return p.parse_args()


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


def _plot_problem(out_path: Path, n_qubits: int, edges, title: str) -> None:
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


def main() -> None:
    args = parse_args()

    if args.edges_json:
        edges = load_edges_json(args.edges_json)
        graph_source = "edges_json"
        graph_params = {"edges_json": str(args.edges_json)}
    else:
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

    problem = make_qaoa_problem_dict(
        n_qubits=int(args.n_qubits),
        edges=edges,
        p_layers=int(args.p_layers),
        delta_t=float(args.delta_t),
        name=str(args.name),
        description=str(args.description),
        graph_source=graph_source,
        graph_params=graph_params,
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(problem, indent=2), encoding="utf-8")

    out_plot = Path(args.output_plot)
    _plot_problem(
        out_path=out_plot,
        n_qubits=int(problem["n_qubits"]),
        edges=[(int(e[0]), int(e[1])) for e in problem["edges"]],
        title=f"{problem['name']} (n={problem['n_qubits']}, |E|={len(problem['edges'])})",
    )

    print(f"saved problem json: {out_json}")
    print(f"saved problem plot: {out_plot}")
    print(
        f"problem summary: n_qubits={int(problem['n_qubits'])}, "
        f"edges={len(problem['edges'])}, "
        f"p_layers={int(problem['qaoa']['p_layers'])}, "
        f"delta_t={float(problem['qaoa']['init']['delta_t'])}, "
        f"graph_type={graph_source}"
    )


if __name__ == "__main__":
    main()
