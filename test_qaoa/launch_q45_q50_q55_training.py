from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


_THIS_DIR = Path(__file__).resolve().parent
_SWEEP_SCRIPT = _THIS_DIR / "sweep_train_qaoa_multi_graphs.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Launch large-scale MaxCut-QAOA training for n_qubits in {45,50,55} "
            "using the existing sweep pipeline."
        )
    )
    p.add_argument("--base-edges-json", type=str, default="QAOA/artifacts/maxcut_edges.json")
    p.add_argument("--output-dir", type=str, default="QAOA/artifacts/sweep_q45_q50_q55_minabs_oddneg")
    p.add_argument("--run-prefix", type=str, default="qaoa_pps")

    p.add_argument("--n-qubits-list", type=str, default="45,50,55")
    p.add_argument("--num-graphs", type=int, default=3)

    p.add_argument("--p-layers", type=str, default="3,5,7")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta-t", type=float, default=0.8)

    p.add_argument("--init-mode", type=str, default="odd-linear-neg", choices=["tqa", "odd-linear-neg"])
    p.add_argument("--mixer-odd-start", type=float, default=-1.0)
    p.add_argument("--mixer-odd-end", type=float, default=-0.05)

    p.add_argument("--build-min-abs", type=float, default=1e-3)
    p.add_argument("--max-weight-disabled-value", type=int, default=1_000_000_000)

    p.add_argument("--weight-modes", type=str, default="yz")
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--rebuild-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_cmd(args: argparse.Namespace) -> List[str]:
    if not _SWEEP_SCRIPT.exists():
        raise FileNotFoundError(f"Sweep script not found: {_SWEEP_SCRIPT}")

    cmd = [
        sys.executable,
        str(_SWEEP_SCRIPT),
        "--base-edges-json",
        str(args.base_edges_json),
        "--output-dir",
        str(args.output_dir),
        "--run-prefix",
        str(args.run_prefix),
        "--n-qubits-list",
        str(args.n_qubits_list),
        "--num-graphs",
        str(int(args.num_graphs)),
        "--p-layers",
        str(args.p_layers),
        "--steps",
        str(int(args.steps)),
        "--lr",
        str(float(args.lr)),
        "--delta-t",
        str(float(args.delta_t)),
        "--init-mode",
        str(args.init_mode),
        "--mixer-odd-start",
        str(float(args.mixer_odd_start)),
        "--mixer-odd-end",
        str(float(args.mixer_odd_end)),
        "--build-min-abs",
        str(float(args.build_min_abs)),
        "--min-abs-only",
        "--max-weight-disabled-value",
        str(int(args.max_weight_disabled_value)),
        "--weight-modes",
        str(args.weight_modes),
        "--chunk-size",
        str(int(args.chunk_size)),
        "--rebuild-every",
        str(int(args.rebuild_every)),
        "--save-every",
        str(int(args.save_every)),
        "--log-every",
        str(int(args.log_every)),
        "--seed",
        str(int(args.seed)),
    ]

    if bool(args.continue_on_error):
        cmd.append("--continue-on-error")
    if bool(args.dry_run):
        cmd.append("--dry-run")

    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_cmd(args)
    print("Running:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
