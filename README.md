# TSP
TSP Test

# PPO-based TSP Solver for TSPLIB lin318 (CPU Only, No Pre-training)

이 저장소는 **단일 TSPLIB 인스턴스(lin318)** 에 대해,  
**사전 학습 없이 랜덤 초기화된 PPO 정책**을 사용해  
**단일 CPU 환경에서 약 20분 내에 최적해 대비 +0.08%까지 도달한 실험 코드와 로그**를 포함한다.

> - 인스턴스: TSPLIB `lin318`  
> - 알려진 최적해(Optimal): 42029 (EUC_2D)  
> - 이 실험의 최고 기록(Best Shot): 42064 (오차 약 +0.08%)

---

## 핵심 아이디어

이 코드는 “평균 오차를 조금씩 줄이는 안정적인 솔버”라기보다,

> **고에너지 탐색을 반복하면서, 간헐적으로 최적해 근처까지 깊게 파고드는 고위험·고보상형 정책**

을 목표로 설계되었다.

- PPO 정책이 계속해서 **위상 구조(geometry/topology)에 민감한 행동 분포**를 학습하면서,
- 매 에피소드마다 **로컬 서치(3-opt)** 를 붙여 후처리를 수행하고,
- 그 과정에서 **매우 낮은 오차의 투어가 “튀어나오는” 샷**을 노린다.

즉, 평균 에러는 3~4% 수준에서 진동하지만,  
학습 중 특정 시점에 **0.1% 미만 오차가 발생하는 순간**을 만들어내는 데 초점을 맞춘 구조다.

---

## 특징 (Key Features)

### 1. No Pre-training (사전 학습 없음)

- **TSPLIB lin318** 단일 인스턴스만 사용한다.
- 정책 네트워크와 가치 네트워크는 **완전 랜덤 초기화** 상태에서 시작한다.
- 별도의 대규모 데이터셋, 사전 학습, imitation learning 없이  
  **PPO + 로컬 서치만으로 최적해 근처까지 도달**한 로그를 제공한다.

### 2. CPU-Only Efficiency (CPU 전용 효율성)

- 전체 학습 및 평가 파이프라인은 **CPU 전용**으로 동작하도록 작성되어 있다.
- 3-opt는 `numba`로 최적화된 버전을 사용하는 것을 가정한다.
- 고성능 GPU 없이도, 연구/실험 환경에서 **가볍게 재실행 가능한 구조**를 목표로 한다.

### 3. Converged in ~20 mins (20분 내 수렴)

- lin318 기준으로, 약 **20분 이내에 +0.08% 오차 샷**이 등장한다.  
  (로그 상에서는 약 `elapsed=0:19:49` 위치에서 42064, err=+0.08% 기록)
- 이후 에피소드가 더 진행되면서 평균 에러는 3~4% 선에서 진동하지만,  
  **이미 “한 번 찍고 간” 저오차 샷이 존재한다**는 것이 이 실험의 포인트다.

---

## 구성 파일

이 저장소는 최소한의 형태로 다음 세 파일만 포함한다고 가정한다.

1. `README.md` — 이 설명 파일 + MIT License  
2. `ppo_tsp_lin318_stage1.py` — lin318 단일 인스턴스용 PPO 학습 스크립트  
3. `log_lin318_stage1.txt` — 위 스크립트를 한 번 실행한 실제 로그 (0.08% 기록 포함)

> 실제 실행을 위해서는, 프로젝트 내부에 맞는  
> `env.py`, `model.py`, `ls.py` (2-opt / 3-opt) 구현이 별도 필요하다.  
> 여기서는 **학습 루프 구조와 로그 형식**을 공유하는 데 집중한다.

---

## 코드 개요 (`ppo_tsp_lin318_stage1.py`)

### 1. 환경 (`TSPEnv`)

- TSPLIB `.tsp` 파일 하나를 받아서 **단일 인스턴스 환경**을 구성한다.
- 주요 인터페이스 (예시):
  - `reset()`: 새 episode 시작
  - `step(action)`: action 실행, `(state, reward, done, info)` 반환
  - `to_tensor(state)`: 현재 상태를 PyTorch 텐서로 변환
  - `get_tour()`: 현재 episode에서 만들어진 투어 반환
  - `get_coords()`: 2D 좌표 배열 반환 (로컬 서치용)
  - `tour_cost(tour)`: TSPLIB 정의에 맞는 EUC_2D integer 코스트 계산

### 2. 모델 (`PolicyNet`, `ValueNet`)

- `PolicyNet`: 현재 상태에 대해 **다음 행동의 로짓(logits)** 을 출력하는 정책 네트워크
- `ValueNet`: 현재 상태의 가치(value)를 스칼라로 출력하는 네트워크
- 두 네트워크 모두 **사전 학습 없이 랜덤 초기화**로 시작한다.

### 3. 로컬 서치 (`two_opt`, `three_opt`)

- 각 episode 롤아웃에서 얻은 투어에 대해,  
  - 2-opt(20회 제한)를 이용해 **에피소드별 best(2opt20)** 를 추적
  - 평가 시에는 numba 기반 3-opt를 사용해 **최종 코스트를 측정**

### 4. PPO 학습 루프

- `collect_rollout`:
  - 여러 에피소드(예: 8 episodes)를 롤아웃하여
  - `(obs, actions, logprobs, rewards, dones, values)` 시퀀스를 모은다.
- `compute_gae`:
  - GAE(Generalized Advantage Estimation)를 사용해 advantage를 계산한다.
- `ppo_update`:
  - 클리핑된 PPO objective + value loss + entropy bonus로  
    policy / value 네트워크를 업데이트한다.
- 각 에피소드마다:
  - 2-opt 기준으로 새로운 best 코스트가 나오면  
    `"[ep XXX] new best(2opt20): ..."` 로그를 출력한다.
  - 일정 간격마다 3-opt 평가를 수행하고  
    `"[eval ep] 3-opt float=... EUC_2D=... err=... elapsed=..."` 같은 로그를 남긴다.
  - 주기적으로 체크포인트를 저장한다.  
    (`model_state`, `value_state`, `optimizer_state` 포함)

---

## 실행 예시

```bash
python ppo_tsp_lin318_stage1.py \
  --tsp_path data/lin318.tsp \
  --opt_cost 42029 \
  --episodes 400 \
  --rollout_episodes 8 \
  --ckpt_dir ckpt_lin318
