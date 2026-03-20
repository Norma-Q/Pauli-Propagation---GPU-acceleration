# LWPP_RCT

Q25 MaxCut-QAOA experiment workspace for the LWPP -> RCT paper direction.

## Files

- `run_q25_experiment.py`
  - Main sweep runner for `3 init x 5 seeds x 2 depths x 4 cases`.
  - Case 1 and case 3 use CUDA-Q exact optimization with `cudaq.optimizers.Adam`.
- `aggregate_q25_results.py`
  - Aggregates saved `artifacts.json` files and creates seed-mean plots.
- `qaoa_experiment_common.py`
  - Shared graph, compile, CUDA-Q exact evaluation, training, and plotting helpers.
- `q25_experiment_config.json`
  - Default experiment config.
- `q10_example_config.json`
  - Single-run 10-qubit example config.
- `run_all.sh`
  - Small wrapper that runs the default sweep.
- `run_q10_example.sh`
  - Small wrapper for a single 10-qubit example run.
- `graph/graph_generator.py`
  - Deterministic connected Erdős-Rényi graph generator.
- `graph/Q25_edges_seed000.json`, `graph/Q25_edges_seed001.json`, ...
  - Per-seed graphs used by the Q25 study when `graph_per_run_seed=true`.

## Output layout

Results are saved under:

- `results/random/seed_000/Q25_L6/`
- `results/random/seed_000/Q25_L9/`
- `results/near_zero/seed_000/Q25_L6/`
- `results/tqa/seed_004/Q25_L9/`

By default, each `run_seed` gets its own graph file, so `seed_000` and `seed_001` will typically point to different `Q25_edges_seedXXX.json` inputs.

Each run directory contains:

- `artifacts.json`
- `case1_exact_warmup.png`
- `case2_lwpp_warmup.png`
- `case3_lwpp_to_exact.png`
- `case4_lwpp_to_coeff.png`
- `integrated_comparison.png`

Aggregate outputs are saved under:

- `results/aggregate/<init>/Q25_L6/`
- `results/aggregate/<init>/Q25_L9/`

## Usage

```bash
python /home/ubuntu/PPS-lab/LWPP_RCT/run_q25_experiment.py
```

Single subset:

```bash
python /home/ubuntu/PPS-lab/LWPP_RCT/run_q25_experiment.py \
  --init-strategy random \
  --seed 0 \
  --depth 6 \
  --skip-aggregate
```

Aggregate only:

```bash
python /home/ubuntu/PPS-lab/LWPP_RCT/aggregate_q25_results.py
```

10-qubit single example:

```bash
bash /home/ubuntu/PPS-lab/LWPP_RCT/run_q10_example.sh
```

Notes:

- The exact branch is evaluated with CUDA-Q, not PPS full-support compilation.
- For case 2 and case 4, surrogate training stores every step's `theta`, and exact values are reevaluated in batch at the end with `cudaq.observe(...)`.
