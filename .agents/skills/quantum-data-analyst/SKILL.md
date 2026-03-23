---
name: quantum-data-analyst
description: Use for data analysis in this repository, especially when the data comes from quantum or hybrid quantum-classical experiments. This includes training logs, shot data, expectation values, latent tensors, generated samples, and molecular-property results. 양자 데이터를 읽을 줄 아는 데이터 분석가 역할로, training_log.csv 분석, expval 해석, histogram 비교, mode collapse 진단, 분자 물성 분포 분석 요청일 때 사용한다. Do not trigger when the main task is circuit design, architecture implementation, or manuscript writing.
---

# Quantum Data Analyst

이 스킬은 양자 데이터를 읽을 줄 아는 데이터 분석가 역할에 맞춰져 있다.

## Priorities

- 먼저 artifact의 출처와 타입을 확인한다. 예: `training_log.csv`, `resolved_config.json`, `dataset_info.json`, `circuit_info.json`.
- 결과가 shot data인지, probability distribution인지, expectation value인지, latent tensor인지, 생성 샘플인지, 분자 property table인지 구분한다.
- `src/MoleculeAnalyzer.py`에 이미 있는 분자 지표는 우선 재활용한다.

## Workflow

1. 입력 데이터의 shape, split, baseline, 비교 대상을 정리한다.
2. 양자 결과라면 expval range, symmetry, heavy states, shot noise 가능성을 점검한다.
3. QGAN/GAN 로그라면 generator-critic 균형, mode collapse 징후, gradient penalty 영향, 학습 불안정성을 본다.
4. 분자 결과라면 validity, uniqueness, novelty와 함께 QED, SA, logP, MW, HBD, HBA, TPSA, PAINS, Fsp3 분포를 요약한다.
5. 필요한 경우 pandas/matplotlib 기반의 재현 가능한 시각화 코드를 제공한다.
6. 마지막에는 진단 요약, 근거, 다음 분석 또는 다음 실험 제안 순으로 정리한다.

## Output expectations

- 숫자만 나열하지 말고 가능한 원인을 같이 설명한다.
- fidelity, significance, tomography 같은 해석은 참조 정보가 있을 때만 사용한다.
- 시각화 코드는 가능한 한 바로 실행 가능한 형태로 제공한다.
