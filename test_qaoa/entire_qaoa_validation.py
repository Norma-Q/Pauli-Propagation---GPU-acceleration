# entire_qaoa_process.py의 결과가 results 폴더에 저장된다고 가정하고, 그 결과를 검증하는 테스트 코드

# results 폴더에서 Q40*, Q45* 폴더의 training.json 파일을 읽으면, qaoa회로에서 얻은 최적화된 파라미터가 있음.
# 하지만 40Q, 45Q에서 실제 exact한 값을 구할 수 없기 때문에, 이를 검증하기 위해 PPS의 min_abs validation을 사용할 것임
# min_abs validation은 QAOA회로를 컴파일할 때, 작은 가중치를 제거하는 방식인데, 
# coeff threshold를 [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]로 바꿔가면서 전파한 프로그램의 기댓값이 비슷한 수준으로 수렴하는지를 확인함.
# 시스템 규모가 크기 때문에 *e-5에서 터질 확률이 크기 때문에, 클 경우에 대비한 예외처리를 해주자.

# 그리고 validation 결과는 results 폴더 내 각 폴더 속에서 validation.png로 저장하며. 
# png는 가로축이 coeff threshold, 세로축이 기댓값이 되도록 그려주자.
# 이 그림에서 기댓값이 수렴한다면, Max Weight Truncation으로 최적화된 파라미터가 제대로 최적화된 파라미터라고 볼 수 있을 것임.

# 이 파일은 단 하나의 테스트를 돌리게 되어 있으며, 터미널에서 config 파일을 인자로 받아서, 해당 config에 맞는 결과 폴더에서 validation을 수행하도록 하자.

import argparse
import gc
import json
from pathlib import Path
import ctypes

import matplotlib.pyplot as plt
import numpy as np  
import torch
import yaml

from qaoa_surrogate_common import (
	build_maxcut_observable,
	build_qaoa_circuit,
	default_cpu_exact_overrides,
	expected_cut_from_sum_zz,
)

from src_tensor.api import compile_expval_program


DEFAULT_THRESHOLDS = [1e-2, 5e-3, 1e-3]


def _cleanup_memory(device: str) -> None:
	gc.collect()
	if device.startswith("cuda") and torch.cuda.is_available():
		pass
		# torch.cuda.empty_cache()  # Disabled due to CUDA indexing assertion errors
	try:
		libc = ctypes.CDLL("libc.so.6")
		if hasattr(libc, "malloc_trim"):
			libc.malloc_trim(0)
	except Exception:
		pass


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Validate trained QAOA params by min_abs threshold sweep."
	)
	parser.add_argument(
		"-c",
		"--config",
		type=str,
		required=True,
		help="Path to config yaml used by entire_qaoa_process.py",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Run device for surrogate evaluation.",
	)
	parser.add_argument(
		"--thresholds",
		type=str,
		default=",".join(f"{x:.0e}" for x in DEFAULT_THRESHOLDS),
		help="Comma-separated min_abs thresholds (e.g. 1e-2,5e-3,1e-3)",
	)
	return parser.parse_args()


def _choose_device(raw: str) -> str:
	if raw == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	if raw == "cuda" and not torch.cuda.is_available():
		print("[warn] CUDA requested but unavailable; falling back to CPU.")
		return "cpu"
	return raw


def _parse_thresholds(raw: str) -> list[float]:
	vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
	if len(vals) == 0:
		raise ValueError("threshold schedule cannot be empty")
	if any(v <= 0.0 for v in vals):
		raise ValueError("all thresholds must be > 0")
	return vals


def _load_config(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	if not isinstance(data, dict):
		raise ValueError("config yaml must be a mapping")
	return data


def _resolve_paths(config_path: Path, n_qubits: int, p_layers: int) -> tuple[Path, Path]:
	config_dir = config_path.resolve().parent
	qaoa_root = config_dir.parent
	result_dir = qaoa_root / "results" / f"Q{n_qubits}_L{p_layers}"
	graph_path = qaoa_root / "graph" / f"Q{n_qubits}_edges.json"
	return result_dir, graph_path


def _load_training_thetas(result_dir: Path) -> np.ndarray:
	candidates = [result_dir / "training.json", result_dir / "training_log.json"]
	target = None
	for c in candidates:
		if c.exists():
			target = c
			break
	if target is None:
		raise FileNotFoundError(
			f"No training file found. checked: {[str(x) for x in candidates]}"
		)

	with open(target, "r", encoding="utf-8") as f:
		data = json.load(f)
	if "trained_thetas" not in data:
		raise KeyError(f"'trained_thetas' not found in {target}")
	return np.asarray(data["trained_thetas"], dtype=np.float64)


def _load_edges(graph_path: Path) -> list[tuple[int, int]]:
	with open(graph_path, "r", encoding="utf-8") as f:
		raw = json.load(f)
	if not isinstance(raw, list):
		raise ValueError(f"invalid graph json: {graph_path}")
	return [(int(e[0]), int(e[1])) for e in raw]


def _evaluate_expected_cut(
	*,
	n_qubits: int,
	p_layers: int,
	edges: list[tuple[int, int]],
	thetas: torch.Tensor,
	device: str,
	min_abs: float,
) -> float:
	circuit, _ = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=p_layers)
	zz_obj = build_maxcut_observable(n_qubits=n_qubits, edges=edges)

	preset = "hybrid" if device.startswith("cuda") else "cpu"
	preset_overrides = default_cpu_exact_overrides() if preset == "cpu" else {'chunk_size': 20_000_000, 'compute_device': 'cuda:3'}

	thetas_dev = thetas.detach().to(device)
	program = compile_expval_program(
		circuit=circuit,
		observables=[zz_obj],
		preset=preset,
		preset_overrides= preset_overrides,		
		build_thetas=thetas_dev,
		build_min_abs=float(min_abs)
	)

	with torch.no_grad():
		sum_zz = float(program.expval(thetas_dev, obs_index=0).detach().cpu().item())

	del program, thetas_dev, circuit, zz_obj
	return expected_cut_from_sum_zz(sum_zz, n_edges=len(edges))


def _plot_validation(
	output_path: Path,
	thresholds: list[float],
	expected_cuts: list[float],
	ok_mask: list[bool],
) -> None:
	x = np.arange(len(thresholds), dtype=np.int64)
	y = np.asarray(expected_cuts, dtype=np.float64)
	ok = np.asarray(ok_mask, dtype=bool)

	fig, ax = plt.subplots(figsize=(7.2, 4.6))
	if np.any(ok):
		ax.plot(x[ok], y[ok], marker="o", linewidth=1.8)
	if np.any(~ok):
		ax.scatter(x[~ok], np.zeros_like(x[~ok]), marker="x", s=60, label="failed")

	ax.set_xlabel("coeff threshold (build_min_abs)")
	ax.set_xticks(x)
	ax.set_xticklabels([f"{th:.0e}" for th in thresholds], rotation=30)
	ax.set_ylabel("Expected cut")
	ax.set_title("Min-abs validation for trained QAOA parameters")
	if np.any(ok):
		ymax = float(np.nanmax(y[ok]))
		if ymax <= 0.0:
			ymax = 1.0
		ax.set_ylim(0.0, ymax * 1.02)
	else:
		ax.set_ylim(0.0, 1.0)
	ax.grid(True, alpha=0.3)
	if np.any(~ok):
		ax.legend()

	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def _save_partial_state(
	result_dir: Path,
	thresholds_done: list[float],
	expected_cuts: list[float],
	ok_mask: list[bool],
) -> None:
	plot_path = result_dir / "validation.png"
	_plot_validation(plot_path, thresholds_done, expected_cuts, ok_mask)

	state_path = result_dir / "validation_progress.json"
	with open(state_path, "w", encoding="utf-8") as f:
		json.dump(
			{
				"thresholds_done": thresholds_done,
				"expected_cuts": expected_cuts,
				"ok_mask": ok_mask,
				"completed_count": len(thresholds_done),
			},
			f,
			indent=2,
		)


def main() -> None:
	args = parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = _load_config(config_path)

	qaoa_cfg = config.get("QAOA", {})
	n_qubits = int(qaoa_cfg["n_qubits"])
	p_layers = int(qaoa_cfg["n_layers"])
	thresholds = _parse_thresholds(args.thresholds)
	device = _choose_device(args.device)

	result_dir, graph_path = _resolve_paths(config_path, n_qubits, p_layers)
	if not result_dir.exists():
		raise FileNotFoundError(f"result directory not found: {result_dir}")
	if not graph_path.exists():
		raise FileNotFoundError(f"graph file not found: {graph_path}")

	trained_thetas_np = _load_training_thetas(result_dir)
	trained_thetas = torch.tensor(trained_thetas_np, dtype=torch.float64)
	edges = _load_edges(graph_path)

	expected_cuts: list[float] = []
	ok_mask: list[bool] = []

	print(f"[info] validating folder: {result_dir}")
	print(f"[info] device: {device}")

	total = len(thresholds)
	for idx, th in enumerate(thresholds, start=1):
		print(f"\n[START {idx}/{total}] threshold={th:.1e}")
		try:
			exp_cut = _evaluate_expected_cut(
				n_qubits=n_qubits,
				p_layers=p_layers,
				edges=edges,
				thetas=trained_thetas,
				device=device,
				min_abs=float(th),
			)
			expected_cuts.append(float(exp_cut))
			ok_mask.append(True)
			print(f"  [OK] threshold={th:.1e} | expected_cut={exp_cut:.6f}")
		except Exception as e:
			expected_cuts.append(np.nan)
			ok_mask.append(False)
			print(f"  [FAIL] threshold={th:.1e} | failed: {e}")

		_cleanup_memory(device)

		# Save partial progress at every threshold so completed results survive later failures.
		thresholds_done = thresholds[:len(expected_cuts)]
		_save_partial_state(
			result_dir=result_dir,
			thresholds_done=thresholds_done,
			expected_cuts=expected_cuts,
			ok_mask=ok_mask,
		)

	plot_path = result_dir / "validation.png"
	thresholds_done = thresholds[:len(expected_cuts)]
	_plot_validation(plot_path, thresholds_done, expected_cuts, ok_mask)
	print(f"[done] validation plot saved to: {plot_path}")


if __name__ == "__main__":
	main()