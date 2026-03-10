import gc
import os
import psutil
import torch
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TEST_QAOA = _REPO_ROOT / "test_qaoa"
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
for path in (str(_TEST_QAOA), str(_REPO_ROOT)):
    sys.path.insert(0, path)

from test_qaoa.qaoa_surrogate_common import build_maxcut_observable, build_qaoa_circuit, load_edges_json
from src_tensor import tensor_propagate as tp

propagate_surrogate_tensor = tp.propagate_surrogate_tensor
zero_filter_tensor_backprop_with_keep_mask = tp.zero_filter_tensor_backprop_with_keep_mask


def sparse_bytes(sp):
    if sp._nnz() == 0:
        return 0
    try:
        idx = sp.indices()
        val = sp.values()
    except Exception:
        sp = sp.coalesce()
        idx = sp.indices()
        val = sp.values()
    return idx.numel() * idx.element_size() + val.numel() * val.element_size()


def psum_bytes(psum):
    total = psum.x_mask.numel() * psum.x_mask.element_size()
    total += psum.z_mask.numel() * psum.z_mask.element_size()
    total += psum.coeff_init.numel() * psum.coeff_init.element_size()
    for st in psum.steps:
        total += sparse_bytes(st.mat_const)
        total += sparse_bytes(st.mat_cos)
        total += sparse_bytes(st.mat_sin)
        if st.same_cols is not None:
            total += st.same_cols.numel() * st.same_cols.element_size()
        if st.anti_same_pos is not None:
            total += st.anti_same_pos.numel() * st.anti_same_pos.element_size()
    return int(total)


process = psutil.Process(os.getpid())


def rss_mb():
    return process.memory_info().rss / (1024 * 1024)


edges = load_edges_json("QAOA/artifacts/sweep_multi_graphs/graphs/edges_graph_01.json")
n_qubits = 30
p_layers = 2

print("[module]", tp.__file__, flush=True)

circuit, _ = build_qaoa_circuit(n_qubits=n_qubits, edges=edges, p_layers=p_layers)
obs = build_maxcut_observable(n_qubits=n_qubits, edges=edges)
thetas = torch.zeros((2 * p_layers,), dtype=torch.float64)

print("[build] rss_mb", round(rss_mb(), 1), flush=True)
psum = propagate_surrogate_tensor(
    circuit=circuit,
    observable=obs,
    max_weight=1_000_000_000,
    memory_device="cpu",
    compute_device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float64",
    thetas=thetas,
    min_abs=1e-3,
    min_mat_abs=None,
    chunk_size=1_000_000,
)

before = psum_bytes(psum)
print(
    "[before] rss_mb",
    round(rss_mb(), 1),
    "psum_mb",
    round(before / 1024 / 1024, 3),
    "x_rows",
    int(psum.x_mask.shape[0]),
    "coeff",
    int(psum.coeff_init.numel()),
    "steps",
    len(psum.steps),
    flush=True,
)

filtered, keep = zero_filter_tensor_backprop_with_keep_mask(
    psum,
    compute_device="cuda" if torch.cuda.is_available() else "cpu",
    chunk_size=1_000_000,
)

after_in = psum_bytes(psum)
after_filtered = psum_bytes(filtered)
print(
    "[after/input] rss_mb",
    round(rss_mb(), 1),
    "input_psum_mb",
    round(after_in / 1024 / 1024, 6),
    "x_rows",
    int(psum.x_mask.shape[0]),
    "coeff",
    int(psum.coeff_init.numel()),
    "steps",
    len(psum.steps),
    flush=True,
)
print(
    "[after/output] filtered_mb",
    round(after_filtered / 1024 / 1024, 3),
    "x_rows",
    int(filtered.x_mask.shape[0]),
    "coeff",
    int(filtered.coeff_init.numel()),
    "steps",
    len(filtered.steps),
    "keep_true",
    int(keep.sum().item()),
    "keep_total",
    int(keep.numel()),
    flush=True,
)

del filtered, keep, psum
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("[post_gc] rss_mb", round(rss_mb(), 1), flush=True)

print("\n===== second case (small graph) =====", flush=True)
edges2 = [(i, (i + 1) % 8) for i in range(8)]
n_qubits2 = 8
p_layers2 = 3
circuit2, _ = build_qaoa_circuit(n_qubits=n_qubits2, edges=edges2, p_layers=p_layers2)
obs2 = build_maxcut_observable(n_qubits=n_qubits2, edges=edges2)
thetas2 = torch.zeros((2 * p_layers2,), dtype=torch.float64)

psum2 = propagate_surrogate_tensor(
    circuit=circuit2,
    observable=obs2,
    max_weight=1_000_000_000,
    memory_device="cpu",
    compute_device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float64",
    thetas=thetas2,
    min_abs=1e-3,
    min_mat_abs=None,
    chunk_size=1_000_000,
)

b2 = psum_bytes(psum2)
print(
    "[second/before] rss_mb",
    round(rss_mb(), 1),
    "psum_mb",
    round(b2 / 1024 / 1024, 3),
    "x_rows",
    int(psum2.x_mask.shape[0]),
    "coeff",
    int(psum2.coeff_init.numel()),
    "steps",
    len(psum2.steps),
    flush=True,
)

filtered2, keep2 = zero_filter_tensor_backprop_with_keep_mask(
    psum2,
    compute_device="cuda" if torch.cuda.is_available() else "cpu",
    chunk_size=1_000_000,
)

ain2 = psum_bytes(psum2)
fout2 = psum_bytes(filtered2)
print(
    "[second/after-input] rss_mb",
    round(rss_mb(), 1),
    "input_psum_mb",
    round(ain2 / 1024 / 1024, 6),
    "x_rows",
    int(psum2.x_mask.shape[0]),
    "coeff",
    int(psum2.coeff_init.numel()),
    "steps",
    len(psum2.steps),
    flush=True,
)
print(
    "[second/after-output] filtered_mb",
    round(fout2 / 1024 / 1024, 3),
    "x_rows",
    int(filtered2.x_mask.shape[0]),
    "coeff",
    int(filtered2.coeff_init.numel()),
    "steps",
    len(filtered2.steps),
    "keep_true",
    int(keep2.sum().item()),
    "keep_total",
    int(keep2.numel()),
    flush=True,
)
