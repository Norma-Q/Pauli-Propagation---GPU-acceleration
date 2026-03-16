from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TEST_QAOA = _REPO_ROOT / "test_qaoa"
for path in (str(_REPO_ROOT), str(_TEST_QAOA)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qaoa_surrogate_common import load_edges_json, cut_value_from_bits


def _bits_from_code(code: int, n_qubits: int) -> np.ndarray:
    bits = np.zeros((int(n_qubits),), dtype=np.uint8)
    for q in range(int(n_qubits)):
        bits[q] = (int(code) >> q) & 1
    return bits


def _load_counts(path: Path, n_qubits: int, bit_order: str) -> Dict[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("counts json must be an object {bitstring_or_int: count}")
    out: Dict[int, int] = {}
    for k, v in raw.items():
        if isinstance(k, str):
            ks = k.strip()
            if all(ch in ("0", "1") for ch in ks) and len(ks) == int(n_qubits):
                code = 0
                if str(bit_order) == "le":
                    for i, ch in enumerate(ks):
                        code |= (int(ch) & 1) << i
                else:
                    for i, ch in enumerate(ks):
                        code |= (int(ch) & 1) << (int(n_qubits) - 1 - i)
            elif ks.isdigit():
                code = int(ks)
            else:
                raise ValueError(f"invalid bitstring key: {k}")
        else:
            code = int(k)
        out[int(code)] = int(v)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MaxCut value distribution from sampling counts.")
    p.add_argument("--counts-json", type=str, required=True, help="Counts JSON from cudaq sampling.")
    p.add_argument("--edges-json", type=str, required=True, help="Edge list JSON [[u,v], ...].")
    p.add_argument("--n-qubits", type=int, default=30)
    p.add_argument("--output-plot", type=str, default="QAOA/artifacts/maxcut_sampling_hist.png")
    p.add_argument(
        "--bit-order",
        type=str,
        default="le",
        choices=["le", "be"],
        help="Bit order for codes: le uses qubit0 as LSB; be reverses bits.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    edges = load_edges_json(args.edges_json)
    counts = _load_counts(Path(args.counts_json), int(args.n_qubits), str(args.bit_order))

    cut_vals = []
    for code, cnt in counts.items():
        bits = _bits_from_code(code, int(args.n_qubits))
        cut = cut_value_from_bits(bits.tolist(), edges)
        cut_vals.extend([cut] * int(cnt))

    if len(cut_vals) == 0:
        raise RuntimeError("No samples in counts JSON.")

    arr = np.asarray(cut_vals, dtype=np.int64)
    bins = np.arange(int(arr.min()), int(arr.max()) + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.hist(arr, bins=bins, color="#5f9ea0", edgecolor="black", alpha=0.85)
    ax.set_xlabel("MaxCut value")
    ax.set_ylabel("Count")
    ax.set_title("Sampling Distribution (MaxCut value)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.output_plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"saved plot: {out_path}")


if __name__ == "__main__":
    main()
