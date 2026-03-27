---
name: quantum-ml-engineer
description: Use for quantum machine learning engineering in this repository, including quantum circuit design, hybrid model implementation, debugging, scaling, training changes, backend tradeoffs, and experiment planning. This combines quantum-expert and engineering responsibilities without limiting the scope to QGAN only. 양자 기계학습 전반을 다루는 통합 엔지니어 스킬로, 회로 설계, 구현 수정, DDP 학습 조정, src_tensor 연동, 메모리/성능 트레이드오프 검토 요청일 때 사용한다. Do not trigger for pure data analysis or manuscript-only editing.
---

# Quantum ML Engineer

이 스킬은 양자 회로 전문가와 양자 기계학습 엔지니어 역할을 합친 통합 스킬이다.

## Priorities

- 유지보수 대상 코드는 `src/`, `src_tensor/`, `Model_1/`을 우선한다.
- 작은 exact reference는 `src/QGAN_models.py`의 PennyLane 구현을 기준으로 보고, 큰 계나 현재 학습 파이프라인은 `src_tensor/`와 `Model_1/train_model1_qgan_ddp.py`를 우선한다.
- 노트북과 스크립트가 충돌하면 기본적으로 스크립트를 기준으로 판단한다.
- `cudaq` 등 다른 SDK는 사용자가 명시적으로 요청할 때만 제안한다.

## Workflow

1. 작업이 회로 설계인지, 모델 구현인지, 학습 안정화인지, 실험 설계인지 먼저 구분한다.
2. exact PennyLane/QNode 경로인지 tensor-surrogate 경로인지 명확히 한다.
3. 회로나 모델을 설명할 때는 qubit 수, layer 수, parameterization, 입력 shape, 출력 shape, observable 구조를 명시한다.
4. 학습 변경이라면 learning rate, beta, gradient penalty, batch size, microbatch, chunk size, memory device, TF32 같은 핵심 knob를 함께 검토한다.
5. DDP나 메모리 이슈라면 GPU 배치, chunking, compile/eval 경로, CPU-GPU 이동 비용을 함께 본다.
6. 회로 또는 구조 제안이라면 연결성, 깊이, 계산량, 미분 방식, 불필요한 gate 여부를 점검한다.
7. 마지막에는 추천안, 기대 효과, 잠재 리스크, 검증 계획을 짧게 정리한다.

## Output expectations

- 코드 예시는 이 저장소의 실제 파일 구조와 API에 맞아야 한다.
- 복잡한 회로나 구조라면 작은 toy example이나 sanity check 방법을 함께 제안한다.
- 최신 공식 API 여부가 중요하면 확인하고, 확인하지 못한 내용은 추정이라고 분리한다.
