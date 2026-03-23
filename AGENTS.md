# QGAN Drug Discovery

## Defaults

- Default to Korean unless the user asks otherwise.
- Default environment is "pps-drug-discovery"
- Prefer maintained code in `src/`, `src_tensor/`, and `Model_1/` over older notebooks.
- Prefer the repo-native stack: PyTorch, PennyLane, NumPy, pandas, RDKit, and `src_tensor`.

## Source of truth

- `src/GAN_models.py`: classical GAN components
- `src/QGAN_models.py`: PennyLane quantum generator
- `src_tensor/`: scalable tensor-surrogate backend
- `Model_1/train_model1_qgan_ddp.py`: maintained DDP training pipeline
- `Model_1/model1_qgan_ddp_config.toml`: main training knobs
- `src/MoleculeAnalyzer.py`: molecule-property analysis

## Repo rules

- Do not introduce `cudaq` or another quantum SDK unless the user explicitly asks.
- When discussing a quantum model, state whether it is exact PennyLane/QNode or tensor-surrogate based.
- When suggesting training changes, tie them to concrete config keys when possible.
- Ground result claims in actual artifacts such as `training_log.csv`, `resolved_config.json`, `dataset_info.json`, and `circuit_info.json`.
- Do not edit checkpoints, run outputs, or logged artifacts unless the user asks.
- Be careful with claims about quantum advantage, novelty, or performance; say when evidence is weak.

## Skills

- `$quantum-data-analyst`
- `$professor-reviewer`
- `$professor-writer`
- `$quantum-ml-engineer`
