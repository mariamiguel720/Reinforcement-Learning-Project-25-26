# FUNCTION CONFIG_B 

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Project modules
from envs.env_setup import (
    ENV_ID, N_STATES, N_ACTIONS, STATE_SURVIVED, STATE_DIED,
    GAMMA, INTENSITY, SOFA_BIAS, LAM,
    make_sepsis_env,
)
from envs.continuous_sepsis_env import ContinuousICUSepsisEnv, FEATURE_NAMES
from envs.wrappers import (
    EpisodicNoisyObsEnv, EpisodicMissingObsEnv,
    AcuteEventEnv, make_clinical_env
)

# Config B constants
OBS_DIM = 47   # 47 physiological features


SEED = 42


# Run random policy rollouts to collect episode statistics and trajectories for offline analysis and DQN pretraining.
def rollout(env, n_episodes: int, seed: int = SEED):
    """Run n_episodes with a random policy. Return a list of episode dicts."""
    rng = np.random.default_rng(seed)
    records = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(1_000_000)))
        ep_return, ep_len, done = 0.0, 0, False
        obs_traj = [obs]
        while not done:
            action = env.action_space.sample()
            obs, r, te, tr, info = env.step(action)
            ep_return += r
            ep_len    += 1
            done = te or tr
            obs_traj.append(obs)
        records.append({
            'episode'   : ep,
            'return'    : ep_return,
            'length'    : ep_len,
            'survived'  : ep_return > 0,
            'obs_traj'  : np.array(obs_traj[:-1]),   # exclude terminal
            **{k: v for k, v in info.items()},
        })
    return records


# DQN - Network and Replay Buffer
class QNetwork(nn.Module):
    """
    Q-value approximator: Q(s, a) for all 25 actions simultaneously.

    LayerNorm after each hidden layer handles the different scales of the
    47 physiological features without requiring explicit observation normalization.
    """
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (s, a, r, s', done) transitions.
    Samples uniform random mini-batches to break temporal correlations.
    """
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
    

# EVALUATION FUNCTION 

N_EPISODES = 1000
clinical_env = make_clinical_env()
records_clinical = rollout(clinical_env, N_EPISODES)
clinical_env.close()

df_clinical = pd.DataFrame([{k: v for k, v in r.items() if k != 'obs_traj'}
                            for r in records_clinical])

clinical_rand_mean = df_clinical['return'].mean()

def evaluate(model, env_fn, n_episodes=1000, seed=SEED, is_ppo=False, label='Agent'):
    """Greedy evaluation on the clinical environment. Works for DQN and PPO."""
    model.eval()
    np.random.seed(seed)
    returns, lengths = [], []
    noisy_r, clean_r, missing_r, nomiss_r = [], [], [], []

    env_eval = env_fn()

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env_eval.reset(seed=np.random.randint(100_000))
            total_r, steps, done = 0.0, 0, False
            ep_noisy   = info.get('noisy_episode', False)
            ep_missing = info.get('missing_features') is not None

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                if is_ppo:
                    logits, _ = model(obs_t)
                    action = int(logits.argmax(1).item())
                else:
                    action = int(model(obs_t).argmax(1).item())
                obs, r, te, tr, _ = env_eval.step(action)
                total_r += r; steps += 1; done = te or tr

            returns.append(total_r); lengths.append(steps)
            (noisy_r if ep_noisy else clean_r).append(total_r)
            (missing_r if ep_missing else nomiss_r).append(total_r)

    env_eval.close()
    model.train()

    returns  = np.array(returns)
    survival = float(np.mean(returns > 0)) * 100

    print(f'═' * 52)
    print(f'  {label} — Evaluation ({n_episodes} episodes)')
    print(f'═' * 52)
    print(f'  Overall survival rate  : {survival:.1f}%')
    print(f'  Mean return            : {np.mean(returns):.4f}')
    print(f'  Mean episode length    : {np.mean(lengths):.1f} steps')
    print(f'  vs Random baseline     : {survival - clinical_rand_mean*100:+.1f}pp')
    print(f'─' * 52)
    print(f'  Noisy episodes   ({len(noisy_r):3d}): {np.mean(np.array(noisy_r)>0)*100:.1f}% survival')
    print(f'  Clean episodes   ({len(clean_r):3d}): {np.mean(np.array(clean_r)>0)*100:.1f}% survival')
    print(f'  Missing feat. ep ({len(missing_r):3d}): {np.mean(np.array(missing_r)>0)*100:.1f}% survival')
    print(f'═' * 52)

    return returns, np.array(lengths), np.array(noisy_r), np.array(clean_r), \
           np.array(missing_r), np.array(nomiss_r)


# PLOTTING FUNCTION
def plot_learning_curve(returns, label, color, filename):
    """Learning curve with smoothed returns and random baseline reference."""
    fig, ax = plt.subplots(figsize=(12, 4))

    window = 200
    smoothed = pd.Series(returns).rolling(window).mean()

    ax.plot(returns, alpha=0.15, color=color, linewidth=0.5)
    ax.plot(smoothed, color=color, linewidth=2, label=f'Smoothed (window={window})')
    ax.axhline(clinical_rand_mean, color='red', linestyle='--',
               linewidth=1.5, label=f'Random baseline (~{clinical_rand_mean:.3f})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(f'{label} — Learning Curve', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()