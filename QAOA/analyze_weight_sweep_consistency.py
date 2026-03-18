from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _resolve_path(path_str: str, *, default_candidates: List[Path]) -> Path:
    if str(path_str).strip() != "":
        p = Path(path_str)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"File not found: {p}")

    for c in default_candidates:
        if c.exists():
            return c.resolve()

    tried = "\n  - ".join(str(x.resolve()) for x in default_candidates)
    raise FileNotFoundError(f"Could not resolve required file. Tried:\n  - {tried}")


def _iqr(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _norm_minmax(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float((v - lo) / (hi - lo))


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    return int(v)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    return float(v)


def _condition_key(rec: Dict[str, Any]) -> Tuple[int, str, int]:
    n_qubits = int(rec["n_qubits"])
    weight_mode = str(rec["weight_mode"])
    max_weight = int(rec["max_weight"])
    return n_qubits, weight_mode, max_weight


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate weight sweep artifacts and rank (max_weight, weight_mode) conditions "
            "by consistency-quality-resource tradeoff."
        )
    )
    p.add_argument("--sweep-dir", type=str, default="QAOA/artifacts/weight_sweep")
    p.add_argument("--summary-json", type=str, default="")
    p.add_argument("--analysis-summary", type=str, default="")
    p.add_argument("--use-step", type=str, default="latest", help="latest or explicit integer step")
    p.add_argument("--lambda-iqr", type=float, default=0.25)
    p.add_argument("--mu-terms", type=float, default=0.20)
    p.add_argument("--nu-gpu", type=float, default=0.20)
    p.add_argument("--xi-fail", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep dir not found: {sweep_dir}")

    summary_path = _resolve_path(
        args.summary_json,
        default_candidates=[sweep_dir / "summary.json"],
    )
    analysis_summary_path = _resolve_path(
        args.analysis_summary,
        default_candidates=[sweep_dir / "analysis" / "analysis_summary.json"],
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis_summary = json.loads(analysis_summary_path.read_text(encoding="utf-8"))

    records = summary.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError("summary.json has no run records")

    by_run_meta: Dict[str, Dict[str, Any]] = {}
    expected_by_cond: Dict[Tuple[int, str, int], int] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        run_name = str(rec.get("run_name", "")).strip()
        if run_name == "":
            continue

        n_qubits = _safe_int(rec.get("n_qubits"))
        weight_mode = rec.get("weight_mode")
        max_weight = _safe_int(rec.get("max_weight"))
        if n_qubits is None or weight_mode is None or max_weight is None:
            continue

        meta = {
            "run_name": run_name,
            "status": str(rec.get("status", "")),
            "n_qubits": int(n_qubits),
            "p_layers": _safe_int(rec.get("p_layers")),
            "graph_index": _safe_int(rec.get("graph_index")),
            "weight_mode": str(weight_mode),
            "max_weight": int(max_weight),
            "anneal_best_cut": _safe_float(rec.get("graph_anneal_best_cut")),
            "report": str(rec.get("report", "")) if rec.get("report") is not None else "",
        }
        by_run_meta[run_name] = meta

        cond = (int(n_qubits), str(weight_mode), int(max_weight))
        expected_by_cond[cond] = expected_by_cond.get(cond, 0) + 1

    analysis_records = analysis_summary.get("records", [])
    if not isinstance(analysis_records, list) or len(analysis_records) == 0:
        raise RuntimeError("analysis_summary.json has no analysis records")

    run_step_records: Dict[str, List[Dict[str, Any]]] = {}
    for rec in analysis_records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status", "")) != "ok":
            continue
        run_name = str(rec.get("run_name", "")).strip()
        if run_name == "":
            continue
        if run_name not in by_run_meta:
            continue
        run_step_records.setdefault(run_name, []).append(rec)

    selected_run_records: List[Dict[str, Any]] = []
    use_latest = str(args.use_step).strip().lower() == "latest"
    explicit_step: Optional[int] = None if use_latest else int(args.use_step)

    for run_name, recs in run_step_records.items():
        if len(recs) == 0:
            continue
        if use_latest:
            chosen = max(recs, key=lambda x: int(x.get("step", -1)))
        else:
            step_target = -1 if explicit_step is None else int(explicit_step)
            matches = [x for x in recs if int(x.get("step", -1)) == step_target]
            if len(matches) == 0:
                continue
            chosen = matches[0]

        mean_cut_raw = chosen.get("mean_cut")
        best_cut_raw = chosen.get("best_cut")
        if mean_cut_raw is None or best_cut_raw is None:
            continue

        merged = dict(by_run_meta[run_name])
        merged.update(
            {
                "step": int(chosen.get("step", -1)),
                "mean_cut": float(mean_cut_raw),
                "best_cut": float(best_cut_raw),
            }
        )

        anneal_best = merged.get("anneal_best_cut")
        if anneal_best is not None and float(anneal_best) > 0:
            merged["quality_ratio"] = float(merged["mean_cut"]) / float(anneal_best)
        else:
            merged["quality_ratio"] = float("nan")

        report_path_text = str(merged.get("report", "")).strip()
        resources = {}
        if report_path_text != "":
            report_path = Path(report_path_text)
            if report_path.exists():
                run_report = json.loads(report_path.read_text(encoding="utf-8"))
                resources = run_report.get("resources", {}) if isinstance(run_report, dict) else {}

        merged["terms_after_zero_filter_last"] = _safe_float(resources.get("terms_after_zero_filter_last"))
        merged["gpu_work_proxy_nnz_total_last"] = _safe_float(resources.get("gpu_work_proxy_nnz_total_last"))
        merged["nnz_total_last"] = _safe_float(resources.get("nnz_total_last"))

        selected_run_records.append(merged)

    if len(selected_run_records) == 0:
        raise RuntimeError("No usable analyzed runs after filtering/join")

    grouped: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = {}
    for rec in selected_run_records:
        cond = _condition_key(rec)
        grouped.setdefault(cond, []).append(rec)

    raw_rows: List[Dict[str, Any]] = []
    for cond, recs in grouped.items():
        n_qubits, weight_mode, max_weight = cond

        quality_vals = [float(r["quality_ratio"]) for r in recs if np.isfinite(float(r["quality_ratio"]))]
        terms_vals = [float(r["terms_after_zero_filter_last"]) for r in recs if r.get("terms_after_zero_filter_last") is not None]
        gpu_vals = [float(r["gpu_work_proxy_nnz_total_last"]) for r in recs if r.get("gpu_work_proxy_nnz_total_last") is not None]

        q_median = float(median(quality_vals)) if len(quality_vals) > 0 else float("nan")
        q_iqr = _iqr(quality_vals)
        q_mean = float(np.mean(quality_vals)) if len(quality_vals) > 0 else float("nan")
        q_std = float(np.std(quality_vals)) if len(quality_vals) > 0 else float("nan")
        q_cv = float(q_std / abs(q_mean)) if len(quality_vals) > 0 and abs(q_mean) > 1e-12 else float("nan")

        terms_med = float(median(terms_vals)) if len(terms_vals) > 0 else float("nan")
        gpu_med = float(median(gpu_vals)) if len(gpu_vals) > 0 else float("nan")

        expected = int(expected_by_cond.get(cond, len(recs)))
        success = int(len(recs))
        fail_rate = float(max(0, expected - success)) / float(max(1, expected))

        raw_rows.append(
            {
                "n_qubits": int(n_qubits),
                "weight_mode": str(weight_mode),
                "max_weight": int(max_weight),
                "n_runs_selected": int(success),
                "n_runs_expected": int(expected),
                "fail_rate": float(fail_rate),
                "quality_ratio_median": float(q_median),
                "quality_ratio_iqr": float(q_iqr),
                "quality_ratio_cv": float(q_cv),
                "terms_after_zero_filter_median": float(terms_med),
                "gpu_work_proxy_nnz_total_median": float(gpu_med),
            }
        )

    valid_rows = [r for r in raw_rows if np.isfinite(r["quality_ratio_median"])]
    if len(valid_rows) == 0:
        raise RuntimeError("No valid condition rows with finite quality_ratio_median")

    terms_list = [r["terms_after_zero_filter_median"] for r in valid_rows if np.isfinite(r["terms_after_zero_filter_median"])]
    gpu_list = [r["gpu_work_proxy_nnz_total_median"] for r in valid_rows if np.isfinite(r["gpu_work_proxy_nnz_total_median"])]

    terms_lo = float(min(terms_list)) if len(terms_list) > 0 else 0.0
    terms_hi = float(max(terms_list)) if len(terms_list) > 0 else 1.0
    gpu_lo = float(min(gpu_list)) if len(gpu_list) > 0 else 0.0
    gpu_hi = float(max(gpu_list)) if len(gpu_list) > 0 else 1.0

    for row in valid_rows:
        terms_norm = _norm_minmax(
            float(row["terms_after_zero_filter_median"]), terms_lo, terms_hi
        ) if np.isfinite(float(row["terms_after_zero_filter_median"])) else 1.0

        gpu_norm = _norm_minmax(
            float(row["gpu_work_proxy_nnz_total_median"]), gpu_lo, gpu_hi
        ) if np.isfinite(float(row["gpu_work_proxy_nnz_total_median"])) else 1.0

        score = (
            float(row["quality_ratio_median"])
            - float(args.lambda_iqr) * float(row["quality_ratio_iqr"])
            - float(args.mu_terms) * float(terms_norm)
            - float(args.nu_gpu) * float(gpu_norm)
            - float(args.xi_fail) * float(row["fail_rate"])
        )

        row["terms_norm"] = float(terms_norm)
        row["gpu_norm"] = float(gpu_norm)
        row["score"] = float(score)

    ranked = sorted(valid_rows, key=lambda x: float(x["score"]), reverse=True)

    out_dir = sweep_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weight_sweep_consistency.json"
    payload = {
        "sweep_dir": str(sweep_dir),
        "summary_json": str(summary_path),
        "analysis_summary_json": str(analysis_summary_path),
        "use_step": str(args.use_step),
        "score_weights": {
            "lambda_iqr": float(args.lambda_iqr),
            "mu_terms": float(args.mu_terms),
            "nu_gpu": float(args.nu_gpu),
            "xi_fail": float(args.xi_fail),
        },
        "ranking": ranked,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved: {out_path}")
    print("top-10 conditions:")
    for i, row in enumerate(ranked[:10], start=1):
        print(
            f"{i:02d}. q={row['n_qubits']} mode={row['weight_mode']} mw={row['max_weight']} "
            f"score={row['score']:.6f} q_med={row['quality_ratio_median']:.6f} "
            f"iqr={row['quality_ratio_iqr']:.6f} terms_med={row['terms_after_zero_filter_median']:.1f} "
            f"gpu_nnz_med={row['gpu_work_proxy_nnz_total_median']:.1f} fail={row['fail_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
