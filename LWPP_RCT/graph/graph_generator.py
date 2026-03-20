from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from qaoa_experiment_common import generate_erdos_renyi_connected, save_graph_circle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate connected Erdős-Rényi graphs for LWPP_RCT experiments."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[25],
        help="List of qubit counts to generate.",
    )
    parser.add_argument(
        "--edge-prob",
        type=float,
        default=0.7,
        help="Erdős-Rényi edge probability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base graph seed.",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=100,
        help="Maximum number of connectivity retries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _THIS_DIR.mkdir(parents=True, exist_ok=True)

    for n_qubits in args.qubits:
        edges = generate_erdos_renyi_connected(
            n_qubits=int(n_qubits),
            edge_prob=float(args.edge_prob),
            seed=int(args.seed),
            max_tries=int(args.max_tries),
        )
        json_path = _THIS_DIR / f"Q{int(n_qubits)}_edges.json"
        png_path = _THIS_DIR / f"Q{int(n_qubits)}_renyi{str(float(args.edge_prob))[2:]}.png"

        json_path.write_text(
            json.dumps([[int(u), int(v)] for (u, v) in edges], indent=2),
            encoding="utf-8",
        )
        try:
            save_graph_circle(int(n_qubits), edges, png_path)
        except Exception:
            pass

        print(
            json.dumps(
                {
                    "n_qubits": int(n_qubits),
                    "n_edges": int(len(edges)),
                    "json_path": str(json_path),
                    "png_path": str(png_path),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
