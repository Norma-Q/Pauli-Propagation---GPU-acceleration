from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from src_tensor import api as _src_tensor_api
from test_qaoa.qaoa_surrogate_common import (
    build_maxcut_observable,
    build_qaoa_circuit,
    build_qaoa_theta_init_tqa,
)

compile_expval_program = _src_tensor_api.compile_expval_program
_loaded_api = Path(_src_tensor_api.__file__).resolve()
if _REPO_ROOT not in _loaded_api.parents:
    raise ImportError(
        "Loaded src_tensor.api from unexpected location: "
        f"{_loaded_api} (expected under {_REPO_ROOT})"
    )


def monitor_peak_memory(pid, stop_event, result_container, sample_interval=0.02):
    process = psutil.Process(pid)
    cpu_peak = 0
    gpu_peak = 0
    cpu_samples = []
    gpu_samples = []
    t0 = time.perf_counter()
    while not stop_event.is_set():
        try:
            cpu = process.memory_info().rss
            if cpu > cpu_peak:
                cpu_peak = cpu
            cpu_samples.append(cpu)

            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated()
                if gpu > gpu_peak:
                    gpu_peak = gpu
                gpu_samples.append(gpu)
            else:
                gpu_samples.append(0)

            time.sleep(sample_interval)
        except Exception:
            break
    result_container["cpu_peak"] = cpu_peak / (1024 * 1024)
    result_container["gpu_peak"] = gpu_peak / (1024 * 1024)
    result_container["cpu_mean"] = (sum(cpu_samples) / len(cpu_samples)) / (1024 * 1024) if cpu_samples else 0
    result_container["gpu_mean"] = (sum(gpu_samples) / len(gpu_samples)) / (1024 * 1024) if gpu_samples else 0


def main() -> None:
    n_qubits = 30
    p_layers = 4
    chord_shift = 7
    edges_path = Path("./QAOA/artifacts/maxcut_edges.json")

    edges = json.loads(edges_path.read_text(encoding="utf-8"))

    print(f"Creating QAOA circuit with {n_qubits} qubits, p={p_layers}...")
    circuit, n_params = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=p_layers)
    obs = build_maxcut_observable(n_qubits=n_qubits, edges=edges)
    print(f"Edges: {len(edges)} | Params: {n_params}")

    configs = [
        {
            "name": "Hybrid Safe (Recommended)",
            "preset": "hybrid",
            "desc": "Storage: CPU, Compute: GPU (Chunked)",
        },
    ]

    print(
        f"\nStarting Benchmark on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}..."
    )
    print("-" * 160)
    print(
        f"{'Configuration':<30} | {'Time (s)':<10} | {'Terms':<12} | "
        f"{'Peak GPU (MB)':<14} | {'Peak CPU (MB)':<14} | "
        f"{'Mean GPU (MB)':<14} | {'Mean CPU (MB)':<14}"
    )
    print("-" * 160)

    process = psutil.Process(os.getpid())

    for cfg in configs:
        program = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        stop_event = threading.Event()
        peak_container = {
            "cpu_peak": process.memory_info().rss / (1024 * 1024),
            "gpu_peak": 0,
            "cpu_mean": 0,
            "gpu_mean": 0,
        }
        monitor_thread = threading.Thread(
            target=monitor_peak_memory,
            args=(os.getpid(), stop_event, peak_container),
            daemon=True,
        )
        monitor_thread.start()

        try:
            t0 = time.perf_counter()
            init_theta_np = build_qaoa_theta_init_tqa(
                p_layers=p_layers,
                n_edges=len(edges),
                n_qubits=n_qubits,
                delta_t=0.8,
            )
            thetas = torch.load("/home/quantum/ys_lee/Pauli-Propagation---GPU-acceleration/QAOA/artifacts/qaoa_maxcut_30q_steps/step_000100.pt")['thetas']

            print(cfg["preset"])
            program = compile_expval_program(
                circuit=circuit,
                observables=[obs],
                preset=cfg["preset"],
                preset_overrides={
                    "chunk_size": 30_000_000,
                },
                build_thetas=thetas,
                build_min_abs=1e-4,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            stop_event.set()
            monitor_thread.join()

            gpu_peak = peak_container["gpu_peak"]
            cpu_peak = peak_container["cpu_peak"]
            gpu_mean = peak_container["gpu_mean"]
            cpu_mean = peak_container["cpu_mean"]

            n_terms = program.psum_union.x_mask.shape[0]

            print(
                f"{cfg['name']:<30} | {t1-t0:<10.4f} | {n_terms:<12,} | "
                f"{gpu_peak:<14.0f} | {cpu_peak:<14.0f} | "
                f"{gpu_mean:<14.0f} | {cpu_mean:<14.0f}"
            )

        except RuntimeError as e:
            stop_event.set()
            monitor_thread.join()

            gpu_peak = peak_container["gpu_peak"]
            err_msg = "OOM / Fail" if "out of memory" in str(e).lower() else "Error"
            print(
                f"{cfg['name']:<30} | {err_msg:<10} | {'-':<12} | "
                f"{gpu_peak:<14.0f} | {'-':<14} | {'-':<14} | {'-':<14}"
            )

        del program
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("-" * 160)


if __name__ == "__main__":
    main()
