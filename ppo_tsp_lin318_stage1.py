import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ============================================================
#  lin318 단일 인스턴스용 PPO 학습 스크립트 (Stage-1)
#
#  특징:
#    - No Pre-training: 완전 랜덤 초기화에서 시작
#    - CPU-Only: device="cpu" 고정
#    - Converged in ~20 mins: lin318 기준 0.08% 샷 등장
#
#  주의:
#    - 아래 env/model/ls 모듈은 프로젝트 환경에 맞게
#      별도 구현되어 있어야 한다.
#      * TSPEnv
#      * PolicyNet
#      * ValueNet
#      * two_opt, three_opt
# ============================================================

from env import TSPEnv          # 단일 TSP 인스턴스 환경
from model import PolicyNet, ValueNet
from ls import two_opt, three_opt  # numba 기반 2-opt / 3-opt 구현


# ---------- 인자 파서 ----------

def parse_args():
    p = argparse.ArgumentParser()

    # 데이터 / 체크포인트
    p.add_argument("--tsp_path", type=str, required=True,
                   help="TSPLIB .tsp 경로 (예: data/lin318.tsp)")
    p.add_argument("--opt_cost", type=float, default=None,
                   help="알려진 최적해 (예: lin318 = 42029). 없으면 None.")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="체크포인트 저장 디렉토리")
    p.add_argument("--ckpt_interval", type=int, default=20,
                   help="N 에피소드마다 체크포인트 저장")

    # 학습 스텝
    p.add_argument("--episodes", type=int, default=400,
                   help="PPO 업데이트(에피소드) 수")
    p.add_argument("--rollout_episodes", type=int, default=8,
                   help="한 번의 PPO 업데이트에서 rollout할 episode 개수")

    # PPO 하이퍼파라미터
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)

    # 평가 관련
    p.add_argument("--eval_interval", type=int, default=1,
                   help="N 에피소드마다 3-opt 평가")
    p.add_argument("--eval_temperature", type=float, default=0.7,
                   help="평가 시 policy sampling temperature")
    p.add_argument("--eval_three_opt_iters", type=int, default=200,
                   help="평가용 3-opt iteration 수")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


# ---------- 유틸 ----------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_hms(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:d}:{s:02d}"


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """
    rewards, values, dones : 1D torch tensor
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t]
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    return advantages


# ---------- 평가 ----------

@torch.no_grad()
def eval_once(env_eval: TSPEnv,
              policy: nn.Module,
              opt_cost: float | None,
              three_opt_iters: int,
              temperature: float,
              device: str = "cpu"):
    """
    학습 중간에 1회 평가:
      - policy로 투어 샘플링
      - 3-opt(local search) 적용
      - float / EUC_2D / err 출력용 값 반환
    """
    state = env_eval.reset()
    done = False

    while not done:
        obs_tensor = env_eval.to_tensor(state)
        obs_tensor = obs_tensor.to(device)

        logits = policy(obs_tensor)

        if temperature <= 0.0:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = Categorical(logits=logits / temperature)
            action = dist.sample()

        action_item = action.item()
        state, reward, done, info = env_eval.step(action_item)

    tour = env_eval.get_tour()
    coords = env_eval.get_coords()

    tour_ls, cost_float = three_opt(
        tour,
        coords,
        max_iters=three_opt_iters,
        use_numba=True,
    )

    cost_int = env_eval.tour_cost(tour_ls)

    if opt_cost is not None:
        err = (cost_int - opt_cost) / opt_cost * 100.0
    else:
        err = None

    return cost_float, cost_int, err


# ---------- 체크포인트 ----------

def save_checkpoint(ckpt_dir: Path,
                    instance_name: str,
                    ep: int,
                    best_cost_2opt: float,
                    policy: nn.Module,
                    value_net: nn.Module,
                    optimizer: torch.optim.Optimizer):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{instance_name}_ep{ep:04d}.pt"

    torch.save(
        {
            "model_state": policy.state_dict(),
            "value_state": value_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ep": ep,
            "best_cost_2opt": best_cost_2opt,
        },
        ckpt_path,
    )
    print(f"[ckpt] saved: {ckpt_path}")


# ---------- rollout ----------

def collect_rollout(env: TSPEnv,
                    policy: nn.Module,
                    value_net: nn.Module,
                    rollout_episodes: int,
                    device: str = "cpu"):
    """
    단일 인스턴스 환경에서 rollout_episodes 만큼 rollout을 모아서
    PPO 업데이트에 필요한 데이터와 에피소드별 투어를 함께 반환.
    """
    obs_list = []
    act_list = []
    logp_list = []
    rew_list = []
    done_list = []
    val_list = []

    episode_tours = []
    episode_raw_costs = []

    for ep_idx in range(rollout_episodes):
        state = env.reset()
        done = False

        while not done:
            obs_tensor = env.to_tensor(state)
            obs_tensor = obs_tensor.to(device)

            logits = policy(obs_tensor)
            value = value_net(obs_tensor).squeeze(-1)

            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            action_item = action.item()
            next_state, reward, done, info = env.step(action_item)

            obs_list.append(obs_tensor.detach().cpu())
            act_list.append(action.detach().cpu())
            logp_list.append(logp.detach().cpu())
            rew_list.append(float(reward))
            done_list.append(float(done))
            val_list.append(value.detach().cpu())

            state = next_state

        tour = env.get_tour()
        raw_cost = env.tour_cost(tour)
        episode_tours.append(tour)
        episode_raw_costs.append(raw_cost)

    obs = torch.cat(obs_list, dim=0)
    actions = torch.stack(act_list).view(-1).long()
    logprobs = torch.stack(logp_list).view(-1)
    rewards = torch.tensor(rew_list, dtype=torch.float32)
    dones = torch.tensor(done_list, dtype=torch.float32)
    values = torch.stack(val_list).view(-1)

    return (
        obs,
        actions,
        logprobs,
        rewards,
        dones,
        values,
        episode_tours,
        episode_raw_costs,
    )


# ---------- PPO 업데이트 ----------

def ppo_update(policy: nn.Module,
               value_net: nn.Module,
               optimizer: torch.optim.Optimizer,
               obs: torch.Tensor,
               actions: torch.Tensor,
               old_logprobs: torch.Tensor,
               returns: torch.Tensor,
               advantages: torch.Tensor,
               clip_range: float,
               entropy_coef: float,
               value_coef: float,
               max_grad_norm: float,
               ppo_epochs: int,
               batch_size: int,
               device: str = "cpu"):
    N = obs.shape[0]
    indices = np.arange(N)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs = obs.to(device)
    actions = actions.to(device)
    old_logprobs = old_logprobs.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, N, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]
            mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=device)

            mb_obs = obs[mb_idx_t]
            mb_actions = actions[mb_idx_t]
            mb_old_logp = old_logprobs[mb_idx_t]
            mb_returns = returns[mb_idx_t]
            mb_adv = advantages[mb_idx_t]

            logits = policy(mb_obs)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            values = value_net(mb_obs).squeeze(-1)

            ratios = torch.exp(logp - mb_old_logp)
            surr1 = ratios * mb_adv
            surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_returns - values).pow(2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            optimizer.step()


# ---------- main ----------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cpu"

    tsp_path = Path(args.tsp_path)
    instance_name = tsp_path.stem

    print(f"[data] {tsp_path.name}")
    env = TSPEnv(tsp_path)
    eval_env = TSPEnv(tsp_path)

    policy = PolicyNet()
    value_net = ValueNet()
    policy.to(device)
    value_net.to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.lr,
    )

    ckpt_dir = Path(args.ckpt_dir)
    best_cost_2opt = float("inf")

    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        (
            obs,
            actions,
            old_logprobs,
            rewards,
            dones,
            values,
            episode_tours,
            episode_raw_costs,
        ) = collect_rollout(
            env,
            policy,
            value_net,
            rollout_episodes=args.rollout_episodes,
            device=device,
        )

        advantages = compute_gae(
            rewards,
            values,
            dones,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        returns = advantages + values

        ppo_update(
            policy=policy,
            value_net=value_net,
            optimizer=optimizer,
            obs=obs,
            actions=actions,
            old_logprobs=old_logprobs,
            returns=returns,
            advantages=advantages,
            clip_range=args.clip_range,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            device=device,
        )

        # 2-opt20 기준 best 업데이트
        coords = env.get_coords()
        updated_this_ep = False
        for tour in episode_tours:
            _, cost_2opt = two_opt(
                tour,
                coords,
                max_iters=20,
                use_numba=True,
            )
            if cost_2opt < best_cost_2opt:
                best_cost_2opt = cost_2opt
                updated_this_ep = True

        if updated_this_ep:
            print(f"[ep {ep:03d}] new best(2opt20): {best_cost_2opt:.2f}")

        # 3-opt 평가
        if (ep % args.eval_interval) == 0:
            eval_float, eval_int, eval_err = eval_once(
                eval_env,
                policy,
                opt_cost=args.opt_cost,
                three_opt_iters=args.eval_three_opt_iters,
                temperature=args.eval_temperature,
                device=device,
            )
            elapsed = time.time() - t0
            elapsed_str = format_hms(elapsed)

            if eval_err is not None:
                print(
                    f"  [eval ep {ep}] 3-opt float={eval_float:.2f}  "
                    f"EUC_2D={int(eval_int):d}  err={eval_err:+.2f}%  "
                    f"elapsed={elapsed_str}"
                )
            else:
                print(
                    f"  [eval ep {ep}] 3-opt cost={eval_int:.2f}  "
                    f"elapsed={elapsed_str}"
                )

        # 체크포인트
        if (ep % args.ckpt_interval) == 0:
            save_checkpoint(
                ckpt_dir,
                instance_name,
                ep,
                best_cost_2opt,
                policy,
                value_net,
                optimizer,
            )

    # 마지막 에피소드 체크포인트
    save_checkpoint(
        ckpt_dir,
        instance_name,
        args.episodes,
        best_cost_2opt,
        policy,
        value_net,
        optimizer,
    )


if __name__ == "__main__":
    main()
