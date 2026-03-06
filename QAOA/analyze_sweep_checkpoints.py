from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TEST_QAOA = _REPO_ROOT / "test_qaoa"
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    sys.path.insert(0, path)

from qaoa_surrogate_common import cut_value_from_bits, load_edges_json


def _bits_from_code(code: int, n_qubits: int) -> np.ndarray:
    bits = np.zeros((int(n_qubits),), dtype=np.uint8)
    for q in range(int(n_qubits)):
        bits[q] = (int(code) >> q) & 1
    return bits


def _parse_counts_keys(counts: Dict[str, int], n_qubits: int, bit_order: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for key, value in counts.items():
        ks = str(key).strip()
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
            raise ValueError(f"invalid bitstring key: {key}")
        out[int(code)] = int(value)
    return out


def _try_cudaq_sample(
    *,
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p_layers: int,
    thetas: np.ndarray,
    shots: int,
    seed: Optional[int],
) -> Dict[str, int]:
    try:
        import cudaq  # type: ignore
    except Exception as e:
        raise RuntimeError(f"cudaq is not available: {e}") from e

    thetas = np.asarray(thetas, dtype=np.float64).reshape(-1)
    expected_params = 2 * int(p_layers)
    if int(thetas.shape[0]) != expected_params:
        raise ValueError(
            f"Theta length mismatch: expected {expected_params}, got {int(thetas.shape[0])}."
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
            kernel.cx(q[uu], q[vv])
            kernel.rz(params[gamma_idx], q[vv])
            kernel.cx(q[uu], q[vv])
        for i in range(int(n_qubits)):
            kernel.rx(params[beta_idx], q[i])

    kernel.mz(q)

    if seed is not None:
        cudaq.set_random_seed(int(seed))

    counts = cudaq.sample(kernel, [float(x) for x in thetas.tolist()], shots_count=int(shots))
    return {str(k): int(v) for k, v in counts.items()}


def _resolve_sweep_dir(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p.resolve()

    candidates = [
        (_REPO_ROOT / p),
        (_THIS_DIR / p),
        (_THIS_DIR / "artifacts" / p.name),
        (_THIS_DIR / "artifacts" / "sweep_multi_graphs"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    tried = "\n  - ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(f"Sweep folder not found. Tried:\n  - {tried}")


def _parse_steps(raw: str) -> Optional[List[int]]:
    text = str(raw).strip()
    if text == "":
        return None
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    if not out:
        return None
    return sorted(set(out))


def _infer_n_qubits(edges: Sequence[Tuple[int, int]]) -> int:
    if len(edges) == 0:
        raise ValueError("Edge list is empty.")
    return max(max(int(u), int(v)) for (u, v) in edges) + 1


def _extract_step_from_name(name: str) -> Optional[int]:
    m = re.match(r"^step_(\d{6})\.pt$", str(name))
    if m is None:
        return None
    return int(m.group(1))


def _collect_step_checkpoints(step_dir: Path, target_steps: Optional[List[int]]) -> List[Tuple[int, Path]]:
    if not step_dir.exists():
        return []
    all_items = sorted(step_dir.glob("step_*.pt"))
    out: List[Tuple[int, Path]] = []
    target = set(target_steps) if target_steps is not None else None
    for item in all_items:
        step = _extract_step_from_name(item.name)
        if step is None:
            continue
        if target is not None and step not in target:
            continue
        out.append((int(step), item))
    return out


def _load_thetas_from_checkpoint(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu")
    if "best_thetas" in payload:
        return payload["best_thetas"].detach().cpu().numpy()
    if "thetas" in payload:
        return payload["thetas"].detach().cpu().numpy()
    if "final_thetas" in payload:
        return payload["final_thetas"].detach().cpu().numpy()
    raise KeyError(f"Checkpoint missing known theta keys: {path}")


def _analyze_counts(
    *,
    counts: Dict[str, int],
    n_qubits: int,
    edges: Sequence[Tuple[int, int]],
    bit_order: str,
) -> Dict[str, Any]:
    code_counts = _parse_counts_keys(counts, int(n_qubits), str(bit_order))
    cut_vals: List[int] = []
    weighted_sum = 0.0
    total = 0
    best_cut = -1
    best_code: Optional[int] = None

    for code, cnt in code_counts.items():
        bits = _bits_from_code(int(code), int(n_qubits))
        cut = int(cut_value_from_bits(bits.tolist(), edges))
        cut_vals.extend([cut] * int(cnt))
        weighted_sum += float(cut) * float(cnt)
        total += int(cnt)
        if cut > best_cut:
            best_cut = int(cut)
            best_code = int(code)

    if total <= 0:
        raise RuntimeError("No samples produced.")

    arr = np.asarray(cut_vals, dtype=np.int64)
    mean_cut = weighted_sum / float(total)

    best_bits = _bits_from_code(int(best_code), int(n_qubits)) if best_code is not None else None
    best_bits_str = "".join(str(int(b)) for b in best_bits.tolist()) if best_bits is not None else None

    return {
        "total_samples": int(total),
        "cut_values": arr,
        "mean_cut": float(mean_cut),
        "best": {
            "code": int(best_code) if best_code is not None else None,
            "bitstring": best_bits_str,
            "cut": int(best_cut) if best_code is not None else None,
        },
    }


def _save_histogram(values: np.ndarray, out_path: Path, title: str) -> None:
    bins = np.arange(int(values.min()), int(values.max()) + 2) - 0.5
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.hist(values, bins=bins, color="#5f9ea0", edgecolor="black", alpha=0.85)
    ax.set_xlabel("MaxCut value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _discover_runs(
    *,
    sweep_dir: Path,
    summary: Dict[str, Any],
    default_edges_json: Optional[str],
) -> List[Dict[str, Any]]:
    runs_dir = sweep_dir / "runs"
    out: List[Dict[str, Any]] = []

    graph_meta = summary.get("graph_meta", {}) if isinstance(summary, dict) else {}

    records = summary.get("records", []) if isinstance(summary, dict) else []
    if isinstance(records, list) and len(records) > 0:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            run_name = str(rec.get("run_name", "")).strip()
            if run_name == "":
                continue
            p_layers_raw = rec.get("p_layers")
            if p_layers_raw is None:
                continue
            p_layers = int(p_layers_raw)
            edges_json = str(rec.get("graph_edges_json", "")).strip()
            if edges_json == "":
                g_idx_raw = rec.get("graph_index")
                if g_idx_raw is None:
                    continue
                g_idx = int(g_idx_raw)
                gm = graph_meta.get(str(g_idx), {}) if isinstance(graph_meta, dict) else {}
                edges_json = str(gm.get("edges_json", "")).strip()
            if edges_json == "":
                if default_edges_json is None:
                    continue
                edges_json = str(Path(default_edges_json).resolve())

            n_qubits_raw = rec.get("n_qubits")
            max_weight_raw = rec.get("max_weight")
            graph_index_raw = rec.get("graph_index")
            anneal_best_raw = rec.get("graph_anneal_best_cut")
            step_dir_raw = rec.get("step_dir")
            checkpoint_raw = rec.get("checkpoint")
            if step_dir_raw is not None and str(step_dir_raw).strip() != "":
                step_dir = Path(str(step_dir_raw)).resolve()
            elif checkpoint_raw is not None and str(checkpoint_raw).strip() != "":
                ckpt_path = Path(str(checkpoint_raw)).resolve()
                step_dir = ckpt_path.parent / f"{run_name}_steps"
            else:
                step_dir = runs_dir / f"{run_name}_steps"
            out.append(
                {
                    "run_name": run_name,
                    "p_layers": int(p_layers),
                    "n_qubits": int(n_qubits_raw) if n_qubits_raw is not None else None,
                    "weight_mode": rec.get("weight_mode"),
                    "max_weight": int(max_weight_raw) if max_weight_raw is not None else None,
                    "graph_index": int(graph_index_raw) if graph_index_raw is not None else None,
                    "graph_anneal_best_cut": float(anneal_best_raw) if anneal_best_raw is not None else None,
                    "edges_json": edges_json,
                    "step_dir": step_dir,
                    "report": rec.get("report"),
                }
            )
        return out

    pattern = re.compile(r"^(?P<run>.+)_steps$")
    for child in sorted(runs_dir.glob("*_steps")):
        m = pattern.match(child.name)
        if m is None:
            continue
        run_name = str(m.group("run"))

        p_match = re.search(r"_p(?P<p>\d+)", run_name)
        if p_match is None:
            continue
        p_layers = int(p_match.group("p"))

        g_match = re.search(r"_g(?P<g>\d+)", run_name)
        g_idx = int(g_match.group("g")) if g_match is not None else -1

        gm = graph_meta.get(str(g_idx), {}) if isinstance(graph_meta, dict) else {}
        edges_json = str(gm.get("edges_json", "")).strip()
        if edges_json == "":
            if default_edges_json is None:
                continue
            edges_json = str(Path(default_edges_json).resolve())
        out.append(
            {
                "run_name": run_name,
                "p_layers": int(p_layers),
                "n_qubits": None,
                "weight_mode": None,
                "max_weight": None,
                "graph_index": g_idx,
                "graph_anneal_best_cut": gm.get("anneal_best_cut") if isinstance(gm, dict) else None,
                "edges_json": edges_json,
                "step_dir": child,
                "report": None,
            }
        )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved step checkpoints inside sweep folder using cudaq sampling and save outputs."
    )
    parser.add_argument("--sweep-dir", type=str, default="QAOA/artifacts/weight_sweep")
    parser.add_argument("--summary-json", type=str, default="", help="Optional explicit summary.json path.")
    parser.add_argument("--default-edges-json", type=str, default="", help="Fallback edges file if run-level edges are unavailable.")
    parser.add_argument("--steps", type=str, default="", help="Comma-separated step ids, e.g. 0,50,100. Empty analyzes all found step checkpoints.")
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bit-order", type=str, default="le", choices=["le", "be"])
    parser.add_argument("--run-name-filter", type=str, default="", help="Optional substring filter for run_name.")
    parser.add_argument("--dry-run", action="store_true", help="Only print targets; do not run cudaq sampling.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sweep_dir = _resolve_sweep_dir(str(args.sweep_dir))
    summary_path = Path(args.summary_json).resolve() if str(args.summary_json).strip() else (sweep_dir / "summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_steps = _parse_steps(str(args.steps))

    default_edges_json = str(args.default_edges_json).strip() or None
    runs = _discover_runs(
        sweep_dir=sweep_dir,
        summary=summary,
        default_edges_json=default_edges_json,
    )

    run_filter = str(args.run_name_filter).strip()
    if run_filter != "":
        runs = [r for r in runs if run_filter in str(r["run_name"])]

    if len(runs) == 0:
        raise RuntimeError("No runs discovered for analysis.")

    analysis_root = sweep_dir / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    global_records: List[Dict[str, Any]] = []
    failed = False

    for run in runs:
        run_name = str(run["run_name"])
        p_layers = int(run["p_layers"])
        run_n_qubits = run.get("n_qubits")
        run_weight_mode = run.get("weight_mode")
        run_max_weight = run.get("max_weight")
        run_graph_index = run.get("graph_index")
        run_anneal_best = run.get("graph_anneal_best_cut")
        run_report = run.get("report")
        edges_json = Path(str(run["edges_json"])).resolve()
        step_dir = Path(run["step_dir"]).resolve()

        edges = load_edges_json(edges_json)
        n_qubits = _infer_n_qubits(edges)

        ckpts = _collect_step_checkpoints(step_dir, target_steps)
        if len(ckpts) == 0:
            print(f"[skip] no checkpoints: run={run_name}, step_dir={step_dir}")
            continue

        run_out_dir = analysis_root / run_name
        run_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== analyze run={run_name} p={p_layers} n_ckpts={len(ckpts)} ===")
        for step_id, ckpt_path in ckpts:
            base = f"step_{int(step_id):06d}"
            out_json = run_out_dir / f"{base}_sampling.json"
            out_plot = run_out_dir / f"{base}_hist.png"

            if bool(args.dry_run):
                rec = {
                    "run_name": run_name,
                    "p_layers": int(p_layers),
                    "n_qubits": int(n_qubits),
                    "step": int(step_id),
                    "checkpoint": str(ckpt_path),
                    "output_json": str(out_json),
                    "output_plot": str(out_plot),
                    "status": "dry_run",
                }
                global_records.append(rec)
                print(f"[dry-run] {run_name} step={step_id} -> {out_json.name}")
                continue

            try:
                thetas = _load_thetas_from_checkpoint(ckpt_path)
                counts = _try_cudaq_sample(
                    n_qubits=int(n_qubits),
                    edges=edges,
                    p_layers=int(p_layers),
                    thetas=thetas,
                    shots=int(args.shots),
                    seed=int(args.seed) + int(step_id),
                )
                analysis = _analyze_counts(
                    counts=counts,
                    n_qubits=int(n_qubits),
                    edges=edges,
                    bit_order=str(args.bit_order),
                )
                _save_histogram(
                    values=analysis["cut_values"],
                    out_path=out_plot,
                    title=f"{run_name} / {base} (shots={int(args.shots)})",
                )

                payload = {
                    "run_name": run_name,
                    "step": int(step_id),
                    "checkpoint": str(ckpt_path),
                    "edges_json": str(edges_json),
                    "n_qubits": int(n_qubits),
                    "p_layers": int(p_layers),
                    "run_n_qubits": (None if run_n_qubits is None else int(run_n_qubits)),
                    "weight_mode": (None if run_weight_mode is None else str(run_weight_mode)),
                    "max_weight": (None if run_max_weight is None else int(run_max_weight)),
                    "graph_index": (None if run_graph_index is None else int(run_graph_index)),
                    "graph_anneal_best_cut": (
                        None if run_anneal_best is None else float(run_anneal_best)
                    ),
                    "run_report": (None if run_report is None else str(run_report)),
                    "shots": int(args.shots),
                    "bit_order": str(args.bit_order),
                    "mean_cut": float(analysis["mean_cut"]),
                    "best": analysis["best"],
                    "counts": counts,
                    "hist_plot": str(out_plot),
                }
                out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                rec = {
                    "run_name": run_name,
                    "step": int(step_id),
                    "n_qubits": int(n_qubits),
                    "p_layers": int(p_layers),
                    "weight_mode": (None if run_weight_mode is None else str(run_weight_mode)),
                    "max_weight": (None if run_max_weight is None else int(run_max_weight)),
                    "graph_index": (None if run_graph_index is None else int(run_graph_index)),
                    "graph_anneal_best_cut": (
                        None if run_anneal_best is None else float(run_anneal_best)
                    ),
                    "run_report": (None if run_report is None else str(run_report)),
                    "checkpoint": str(ckpt_path),
                    "output_json": str(out_json),
                    "output_plot": str(out_plot),
                    "mean_cut": float(analysis["mean_cut"]),
                    "best_cut": int(analysis["best"]["cut"]),
                    "status": "ok",
                }
                global_records.append(rec)
                print(
                    f"[ok] {run_name} step={step_id} mean_cut={float(analysis['mean_cut']):.4f} "
                    f"best_cut={int(analysis['best']['cut'])}"
                )

            except Exception as e:
                failed = True
                rec = {
                    "run_name": run_name,
                    "step": int(step_id),
                    "checkpoint": str(ckpt_path),
                    "status": "failed",
                    "error": str(e),
                }
                global_records.append(rec)
                print(f"[failed] {run_name} step={step_id}: {e}")
                if not bool(args.continue_on_error):
                    summary_out = analysis_root / "analysis_summary.json"
                    summary_out.write_text(
                        json.dumps(
                            {
                                "sweep_dir": str(sweep_dir),
                                "summary_json": str(summary_path),
                                "steps": target_steps,
                                "records": global_records,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    raise

    summary_out = analysis_root / "analysis_summary.json"
    summary_out.write_text(
        json.dumps(
            {
                "sweep_dir": str(sweep_dir),
                "summary_json": str(summary_path),
                "steps": target_steps,
                "records": global_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_ok = sum(1 for x in global_records if x.get("status") in ("ok", "dry_run"))
    n_fail = sum(1 for x in global_records if x.get("status") == "failed")
    print("\n" + "-" * 100)
    print(f"saved analysis summary: {summary_out}")
    print(f"completed analyses: ok={n_ok}, failed={n_fail}, total={len(global_records)}")

    if failed and not bool(args.continue_on_error):
        raise RuntimeError("Analysis failed for one or more checkpoints.")


if __name__ == "__main__":
    main()
