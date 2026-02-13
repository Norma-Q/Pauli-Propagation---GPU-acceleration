from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import json
import sys

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pauli_surrogate_python import CliffordGate, PauliRotation, PauliSum


Edge = Tuple[int, int]


@dataclass(frozen=True)
class QAOAParams:
    gammas: np.ndarray
    betas: np.ndarray


def canonical_edge(u: int, v: int) -> Edge:
    if int(u) == int(v):
        raise ValueError("Self-loop edges are not allowed for MaxCut.")
    a, b = int(u), int(v)
    return (a, b) if a < b else (b, a)


def load_edges_json(path: str | Path) -> List[Edge]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Edge JSON must be a list of [u, v] pairs.")
    edges = sorted({canonical_edge(int(e[0]), int(e[1])) for e in raw})
    return edges


def make_qaoa_problem_dict(
    *,
    n_qubits: int,
    edges: Sequence[Edge],
    p_layers: int,
    delta_t: float,
    name: str = "qaoa_problem",
    description: str = "",
    graph_source: str = "custom",
    graph_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    n = int(n_qubits)
    p = int(p_layers)
    if n < 2:
        raise ValueError("n_qubits must be >= 2")
    if p < 1:
        raise ValueError("p_layers must be >= 1")
    if not (float(delta_t) > 0.0):
        raise ValueError("delta_t must be > 0")

    edge_list = sorted({canonical_edge(int(u), int(v)) for (u, v) in edges})
    if len(edge_list) < 1:
        raise ValueError("Problem must contain at least one edge.")
    if any(max(u, v) >= n or min(u, v) < 0 for (u, v) in edge_list):
        raise ValueError("Edge index out of range for n_qubits.")

    return {
        "schema_version": 1,
        "name": str(name),
        "description": str(description),
        "n_qubits": n,
        "edges": [[int(u), int(v)] for (u, v) in edge_list],
        "qaoa": {
            "p_layers": p,
            "init": {
                "type": "tqa",
                "delta_t": float(delta_t),
            },
        },
        "graph": {
            "source": str(graph_source),
            "params": dict(graph_params or {}),
        },
    }


def load_qaoa_problem_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Problem JSON must be an object.")

    n_qubits = int(raw["n_qubits"])
    edges_raw = raw["edges"]
    if not isinstance(edges_raw, list):
        raise ValueError("Problem JSON 'edges' must be a list.")
    edges = sorted({canonical_edge(int(e[0]), int(e[1])) for e in edges_raw})
    if any(max(u, v) >= n_qubits or min(u, v) < 0 for (u, v) in edges):
        raise ValueError("Problem JSON has edges outside qubit range.")

    qaoa_raw = raw["qaoa"]
    p_layers = int(qaoa_raw["p_layers"])
    init_raw = qaoa_raw["init"]
    init_type = str(init_raw.get("type", "tqa")).lower()
    if init_type != "tqa":
        raise ValueError("Currently only qaoa.init.type='tqa' is supported.")
    delta_t = float(init_raw["delta_t"])
    if delta_t <= 0.0:
        raise ValueError("Problem JSON qaoa.init.delta_t must be > 0.")

    return {
        "schema_version": int(raw.get("schema_version", 1)),
        "name": str(raw.get("name", "qaoa_problem")),
        "description": str(raw.get("description", "")),
        "n_qubits": n_qubits,
        "edges": edges,
        "qaoa": {
            "p_layers": p_layers,
            "init": {"type": "tqa", "delta_t": delta_t},
        },
        "graph": dict(raw.get("graph", {})),
        "raw": raw,
    }


def make_ring_chord_graph(n_qubits: int, chord_shift: int = 7) -> List[Edge]:
    if int(n_qubits) < 2:
        raise ValueError("n_qubits must be >= 2")
    n = int(n_qubits)
    shift = int(chord_shift) % n
    if shift == 0:
        raise ValueError("chord_shift must not be 0 modulo n_qubits")

    edge_set = set()
    for i in range(n):
        edge_set.add(canonical_edge(i, (i + 1) % n))
        edge_set.add(canonical_edge(i, (i + shift) % n))
    return sorted(edge_set)


def build_maxcut_observable(n_qubits: int, edges: Sequence[Edge]) -> PauliSum:
    obs = PauliSum(int(n_qubits))
    for u, v in edges:
        obs.add_from_str("ZZ", 1.0, qubits=[int(u), int(v)])
    return obs


def build_qaoa_circuit(n_qubits: int, edges: Sequence[Edge], p_layers: int):
    circuit = []
    n = int(n_qubits)
    p = int(p_layers)

    for q in range(n):
        circuit.append(CliffordGate("H", [q]))

    param_idx = 0
    for _ in range(p):
        for (u, v) in edges:
            circuit.append(PauliRotation("ZZ", [int(u), int(v)], param_idx=param_idx))
            param_idx += 1
        for q in range(n):
            circuit.append(PauliRotation("X", [q], param_idx=param_idx))
            param_idx += 1
    return circuit, param_idx


def tqa_init_qaoa_params(p: int, delta_t: float, dtype=np.float64) -> QAOAParams:
    if int(p) < 1:
        raise ValueError("p must be >= 1")
    if not (float(delta_t) > 0.0):
        raise ValueError("delta_t must be > 0")
    i = np.arange(1, int(p) + 1, dtype=dtype)
    pp = dtype(p)
    gammas = (i / pp) * dtype(delta_t)
    betas = (dtype(1.0) - (i / pp)) * dtype(delta_t)
    return QAOAParams(gammas=gammas, betas=betas)


def build_qaoa_theta_init_tqa(
    p_layers: int,
    n_edges: int,
    n_qubits: int,
    delta_t: float,
    dtype=np.float64,
) -> np.ndarray:
    params = tqa_init_qaoa_params(p=int(p_layers), delta_t=float(delta_t), dtype=dtype)
    thetas: List[float] = []
    for l in range(int(p_layers)):
        thetas.extend([float(params.gammas[l])] * int(n_edges))
        thetas.extend([float(params.betas[l])] * int(n_qubits))
    return np.asarray(thetas, dtype=dtype)


def cut_value_from_bits(bits01: Sequence[int], edges: Sequence[Edge]) -> int:
    val = 0
    for u, v in edges:
        val += int(int(bits01[int(u)]) != int(bits01[int(v)]))
    return int(val)


def cut_value_from_code(code: int, n_qubits: int, edges: Sequence[Edge]) -> int:
    val = 0
    for u, v in edges:
        bu = (int(code) >> int(u)) & 1
        bv = (int(code) >> int(v)) & 1
        val += int(bu != bv)
    return int(val)


def brute_force_maxcut(n_qubits: int, edges: Sequence[Edge]) -> Dict[str, object]:
    n = int(n_qubits)
    if n < 1:
        raise ValueError("n_qubits must be >= 1")
    best_cut = -1
    best_codes: List[int] = []
    for code in range(1 << n):
        c = cut_value_from_code(code, n, edges)
        if c > best_cut:
            best_cut = c
            best_codes = [int(code)]
        elif c == best_cut:
            best_codes.append(int(code))
    return {"best_cut": int(best_cut), "best_codes": best_codes}


def expected_cut_from_sum_zz(sum_zz: float, n_edges: int) -> float:
    return 0.5 * (float(n_edges) - float(sum_zz))


def parse_min_abs_schedule(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("min_abs schedule cannot be empty")
    if any(v <= 0.0 for v in vals):
        raise ValueError("all min_abs values must be > 0")
    # Keep user-given order.
    return vals
