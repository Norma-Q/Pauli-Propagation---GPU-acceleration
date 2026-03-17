# 기존 모듈 임포트
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test_qaoa.make_graph import _make_erdos_renyi_graph as make_erdos_renyi_graph

def save_graph_circle(n, edges, dir = None):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    
    plt.figure(figsize=(6, 6))
    # Draw edges
    for u, v in edges:
        plt.plot([x[u], x[v]], [y[u], y[v]], color="gray", alpha=0.5)
    # Draw nodes
    plt.scatter(x, y, s=200, c='skyblue', edgecolors='black')
    for i in range(n):
        plt.text(x[i]*1.1, y[i]*1.1, str(i), ha='center', va='center')
    plt.axis('off')
    plt.title(f"Ring-Chord Graph (n={n})")
    plt.savefig(dir)

e_list = []
EDGE_PROB = 0.2 
SEED = 42
q_list = [100]
for q in q_list:
    N_QUBITS = q
    edges = make_erdos_renyi_graph(n_qubits = N_QUBITS, edge_prob=0.15, seed=SEED,
                                   ensure_connected=True, max_tries=100)
    e_list.append(edges)
    save_graph_circle(N_QUBITS, edges, dir = f"/home/ubuntu/PPS-lab/test_qaoa/graph/Q{N_QUBITS}_renyi{str(EDGE_PROB)[2:]}.png")

e_dict = {q: e for q, e in zip(q_list, e_list)}

import json
for key in e_dict:
    with open(f"/home/ubuntu/PPS-lab/test_qaoa/graph/Q{key}_edges.json", "w") as f:
        json.dump(e_dict[key], f,)
