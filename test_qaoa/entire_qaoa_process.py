import sys
from pathlib import Path
import os
import json
import argparse
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict

# [Path Setup] Ensure src_tensor and other modules are importable
if "/home/ubuntu/PPS-lab" not in sys.path:
    sys.path.insert(0, "/home/ubuntu/PPS-lab")

from src_tensor.api import compile_expval_program
from make_qaoa_problem import _make_erdos_renyi_graph as make_erdos_renyi_graph
from qaoa_surrogate_common import (
    build_qaoa_circuit,
    build_maxcut_observable,
    build_qaoa_theta_init_tqa,
    expected_cut_from_sum_zz,
)

def graph_visualization_and_save(n, edges, graph_path):
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
    plt.savefig(f"/home/ubuntu/PPS-lab/test_qaoa/graph/Q{n}_renyi15.png")
    plt.close()

    with open(graph_path, "w") as f:
        json.dump(edges, f)

def visualize_and_save(log_dict, output_dir):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1) # Loss Plot
    plt.plot(log_dict["step"], log_dict["loss"], label="Loss (Sum<ZZ>)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2) # Expected Cut Value Plot
    plt.plot(log_dict["step"], log_dict["expected_cut"], label="Expected Cut", color='orange')
    plt.xlabel("Step")
    plt.ylabel("Expected Cut Value")
    plt.title("Expected Cut Value Over Time")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3) # Bar Chart
    plt.bar(["Initial", "Trained"], [log_dict["expected_cut"][0], log_dict["expected_cut"][-1]], color=["gray", "teal"])
    plt.ylabel("Expected Cut")
    plt.title("Performance Comparison")
    # plt.show() # Blocking execution in non-interactive environments

    plt.tight_layout() # save
    plt.savefig(os.path.join(output_dir, "loss_and_value_history.png"))
    plt.close()

def train(program, thetas, n_edges, STEPS, LR, output_dir):   
    log_dict = {dict_key: [] for dict_key in ["step", "loss", "expected_cut"]}
    
    optimizer = torch.optim.Adam([thetas], lr=LR)

    print(f"Starting training for {STEPS} steps...")

    for step in range(STEPS):
        optimizer.zero_grad()
        
        # Calculate Expectation Value <H>
        loss = program.expval(thetas, obs_index=0)
        
        loss.backward()
        optimizer.step()
        
        # Logging
        val = loss.item()
        exp_cut = expected_cut_from_sum_zz(val, n_edges)
        
        log_dict["step"].append(step)
        log_dict["loss"].append(val)
        log_dict["expected_cut"].append(exp_cut)

        if step % 10 == 0 or step == STEPS - 1:
            print(f"Step {step:03d} | Loss(Sum<ZZ>): {val:+.4f} | Expected Cut: {exp_cut:.4f}")

    print("Training complete.")
    return log_dict, thetas


def main():    
    #################### Configuration Loading ####################
    parser = argparse.ArgumentParser(description = "QAOA Training Script")
    parser.add_argument('-c', '--config', type = str, required = True, help = "Path to the configuration file")

    args, _ = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) # config는 yaml의 dict 형태로 저장됨
        config = EasyDict(config)
    ###############################################################

    ###################### Parameters Setting ######################
    N_QUBITS = config.QAOA.n_qubits
    P_LAYERS = config.QAOA.n_layers   
    DELTA_T = 0.8  # TQA initialization parameter
    STEPS = 150
    LR = 0.05
    SEED = 42
    EDGE_PROB = 0.15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_WEIGHT = 4

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ######################################################################



    ############################ Output Directory Setup ##########################
    output_dir = f"/home/ubuntu/PPS-lab/test_qaoa/results/Q{N_QUBITS}_L{P_LAYERS}"
    os.makedirs(output_dir, exist_ok=True)
    ################################################################################



    ########################## Load Graph Information #########################
    graph_path = f"/home/ubuntu/PPS-lab/test_qaoa/graph/Q{N_QUBITS}_edges.json"
    try:
        with open(graph_path, "r") as f:
            edges = json.load(f) # list of list where each inner list is [i, j] representing an edge between qubits i and j
    except:
        print(f"Graph file not found at {graph_path}. Generating a new graph...")
        edges = make_erdos_renyi_graph(n_qubits = N_QUBITS, edge_prob=EDGE_PROB, seed=SEED,
                                       ensure_connected=True, max_tries=3)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        graph_visualization_and_save(N_QUBITS, edges, graph_path)  

    n_edges = len(edges)
    print(f"Generated graph with {n_edges} edges.")
    #######################################################################



    ########################### QAOA Circuit Construction #########################
    # Build Circuit Structure
    circuit, n_params = build_qaoa_circuit(N_QUBITS, edges, P_LAYERS)
    zz_obj = build_maxcut_observable(N_QUBITS, edges)

    # Initialize Parameters (TQA)
    init_theta_np = build_qaoa_theta_init_tqa(p_layers=P_LAYERS, n_edges=n_edges, 
                                              n_qubits=N_QUBITS, delta_t=DELTA_T, dtype=np.float64)
    
    # Convert to PyTorch Parameter
    initial_thetas = torch.nn.Parameter(torch.tensor(init_theta_np, dtype=torch.float64, device=DEVICE))

    # Compile Surrogate Program
    program = compile_expval_program(
        circuit=circuit, observables=[zz_obj], preset="hybrid",
        preset_overrides={'max_weight': MAX_WEIGHT,                    
                          'chunk_size' : 10_000_000})
    print("Program compiled successfully.")


    ########################## Training & Saving #############################
    log_dict, trained_thetas = train(program, initial_thetas, n_edges, STEPS, LR, output_dir)
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump({'initial_thetas': init_theta_np.tolist(),
                    'trained_thetas': trained_thetas.detach().cpu().numpy().tolist(),
                    'training_history': log_dict}, f, indent=2)
    visualize_and_save(log_dict, output_dir) 
    ############################################################################   
    
if __name__ == "__main__":
    main()