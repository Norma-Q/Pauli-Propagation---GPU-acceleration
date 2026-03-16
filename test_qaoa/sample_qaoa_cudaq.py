from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
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


def _parse_counts_keys(counts: Dict[str, int], n_qubits: int, bit_order: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k, v in counts.items():
        ks = str(k).strip()
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
        out[int(code)] = int(v)
    return out


def _try_cudaq_sample(
    *,
    n_qubits: int,
    edges,
    p_layers: int,
    thetas: np.ndarray,
    shots: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    try:
        import cudaq  # type: ignore
    except Exception as e:
        raise RuntimeError(f"cudaq is not available: {e}") from e

    thetas = np.asarray(thetas, dtype=np.float64).reshape(-1)
    expected_params = 2 * int(p_layers)
    if int(thetas.shape[0]) != expected_params:
        raise ValueError(
            "Theta length mismatch: expected "
            f"{expected_params} = p_layers* (|E| + n_qubits) but got {int(thetas.shape[0])}."
        )

    if len(edges) == 0:
        raise ValueError("Edge list is empty.")
    max_node = max(max(int(u), int(v)) for (u, v) in edges)
    min_node = min(min(int(u), int(v)) for (u, v) in edges)
    if min_node < 0 or max_node >= int(n_qubits):
        raise ValueError(
            f"Edge indices out of range for n_qubits={int(n_qubits)}: "
            f"min={min_node}, max={max_node}."
        )

    n_edges = int(len(edges))
    n_params = 2 * int(p_layers)
    params_list = [float(x) for x in thetas.tolist()]
    if len(params_list) != n_params:
        raise ValueError(
            f"Theta length mismatch: expected {n_params}, got {len(params_list)}."
        )

    kernel, params = cudaq.make_kernel(list[float])
    q = kernel.qalloc(int(n_qubits))
    for i in range(int(n_qubits)):
        kernel.h(q[i])

    for layer in range(int(p_layers)):
        gamma_idx = 2 * layer
        beta_idx = 2 * layer + 1
        for (u, v) in edges:
            uu = int(u)
            vv = int(v)
            # ZZ rotation via CNOT-RZ-CNOT. Use rz(2 * theta) to match exp(-i theta/2 ZZ).
            kernel.cx(q[uu], q[vv])
            kernel.rz(params[gamma_idx], q[vv])
            kernel.cx(q[uu], q[vv])
        for i in range(int(n_qubits)):
            kernel.rx(params[beta_idx], q[i])

    kernel.mz(q)

    if seed is not None:
        cudaq.set_random_seed(int(seed))

    counts = cudaq.sample(kernel, params_list, shots_count=int(shots))
    return {str(k): int(v) for k, v in counts.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample a trained QAOA circuit using cudaq.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint with best_thetas.")
    p.add_argument("--edges-json", type=str, required=True, help="Edge list JSON [[u,v], ...].")
    p.add_argument("--n-qubits", type=int, default=30)
    p.add_argument("--p-layers", type=int, default=4)
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default="QAOA/artifacts/qaoa_sampling_counts.json")
    p.add_argument("--output-plot", type=str, default="QAOA/artifacts/maxcut_sampling_hist.png")
    p.add_argument(
        "--bit-order",
        type=str,
        default="le",
        choices=["le", "be"],
        help="Bit order for bitstring keys: le uses leftmost as qubit0; be reverses.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    payload = torch.load(ckpt_path, map_location="cpu")
    if "best_thetas" in payload:
        thetas = payload["best_thetas"].detach().cpu().numpy()
    elif "thetas" in payload:
        thetas = payload["thetas"].detach().cpu().numpy()
    else:
        raise KeyError("Checkpoint missing 'best_thetas' or 'thetas'.")

    edges = load_edges_json(args.edges_json)

    counts = _try_cudaq_sample(
        n_qubits=int(args.n_qubits),
        edges=edges,
        p_layers=int(args.p_layers),
        thetas=thetas,
        shots=int(args.shots),
        seed=int(args.seed),
    )

    code_counts = _parse_counts_keys(counts, int(args.n_qubits), str(args.bit_order))
    cut_vals = []
    best_cut = -1
    best_code = None
    for code, cnt in code_counts.items():
        bits = _bits_from_code(code, int(args.n_qubits))
        cut = cut_value_from_bits(bits.tolist(), edges)
        cut_vals.extend([cut] * int(cnt))
        if cut > best_cut:
            best_cut = int(cut)
            best_code = int(code)

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

    plot_path = Path(args.output_plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    best_bits = _bits_from_code(int(best_code), int(args.n_qubits)) if best_code is not None else None
    best_bits_str = "".join(str(int(b)) for b in best_bits.tolist()) if best_bits is not None else None

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_qubits": int(args.n_qubits),
        "p_layers": int(args.p_layers),
        "shots": int(args.shots),
        "bit_order": str(args.bit_order),
        "best": {
            "code": int(best_code) if best_code is not None else None,
            "bitstring": best_bits_str,
            "cut": int(best_cut) if best_code is not None else None,
        },
        "counts": counts,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved counts+best: {out_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
