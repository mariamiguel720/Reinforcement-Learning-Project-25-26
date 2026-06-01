import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats

from src.ppo_functions import *

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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
OBS_DIM = 47




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



# EVALUATION FUNCTION 
def evaluate(model, env_fn, n_episodes=1000, is_ppo=False, is_sac=False, label='Agent', baseline=None):
    """Greedy evaluation on the clinical environment. Works for DQN and PPO."""
    model.eval()
    np.random.seed(42)
    returns, lengths = [], []
    noisy_r, clean_r, missing_r, nomiss_r = [], [], [], []
    acute_r, noacute_r = [], []

    env_eval = env_fn()

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env_eval.reset(seed=np.random.randint(100_000))
            total_r, steps, done = 0.0, 0, False
            ep_noisy   = info.get('noisy_episode', False)
            ep_missing = info.get('missing_features') is not None
            ep_acute   = False

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                if is_ppo:
                    logits, _ = model(obs_t)
                    action = int(logits.argmax(1).item())
                elif is_sac:
                    probs, _ = model(obs_t)
                    action   = int(probs.argmax(1).item())
                else:
                    action = int(model(obs_t).argmax(1).item())
                obs, r, te, tr, info = env_eval.step(action)

                if info.get('acute_event', False):
                    ep_acute = True
                total_r += r; steps += 1; done = te or tr

            returns.append(total_r); lengths.append(steps)
            (noisy_r if ep_noisy else clean_r).append(total_r)
            (missing_r if ep_missing else nomiss_r).append(total_r)
            (acute_r if ep_acute else noacute_r).append(total_r)

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
    if baseline is not None:
        print(f'  vs Random baseline     : {survival - baseline*100:+.1f}pp')
    print(f'─' * 52)
    print(f'  Noisy episodes   ({len(noisy_r):3d}): {np.mean(np.array(noisy_r)>0)*100:.1f}% survival')
    print(f'  Clean episodes   ({len(clean_r):3d}): {np.mean(np.array(clean_r)>0)*100:.1f}% survival')
    print(f'  Missing feat. ep ({len(missing_r):3d}): {np.mean(np.array(missing_r)>0)*100:.1f}% survival')
    print(f'  Acute event ep   ({len(acute_r):3d}): {np.mean(np.array(acute_r)>0)*100 if len(acute_r) else 0:.1f}% survival')
    print(f'  No acute event   ({len(noacute_r):3d}): {np.mean(np.array(noacute_r)>0)*100 if len(noacute_r) else 0:.1f}% survival')
    print(f'═' * 52)

    return {
        'returns':  returns,
        'lengths':  np.array(lengths),
        'noisy':    np.array(noisy_r),
        'clean':    np.array(clean_r),
        'missing':  np.array(missing_r),
        'nomiss':   np.array(nomiss_r),
        'acute':    np.array(acute_r),
        'noacute':  np.array(noacute_r),
    }


# PLOTTING FUNCTIONS

def plot_learning_curve(returns, label, window=200, trend=True):
    """Learning curve with smoothed returns and optional linear trend line."""
    fig, ax = plt.subplots(figsize=(12, 4))

    returns = np.asarray(returns)
    smoothed = pd.Series(returns).rolling(window).mean()

    ax.plot(smoothed, color="#2f94d7", linewidth=2, label=f'Smoothed (window={window})')

    # Trend line
    if trend:
        x = np.arange(len(returns))
        slope, intercept = np.polyfit(x, returns, 1)
        ax.plot(x, slope * x + intercept, color='#0a1f35', linewidth=1.5,
                linestyle='--', label=f'Trend (slope={slope:.2e})')

    # Eixo y apertado a volta da curva suavizada (ignora NaNs iniciais da rolling)
    valid = smoothed.dropna()
    margin = (valid.max() - valid.min()) * 0.1
    ax.set_ylim(valid.min() - margin, valid.max() + margin)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(f'{label} — Learning Curve', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()


def survival_ci(returns, confidence=0.95):
    """Confidence interval for survival rate."""
    n = len(returns)
    p = np.mean(np.array(returns) > 0)
    ci = stats.binom.interval(confidence, n, p)
    lower = (ci[0] / n) * 100
    upper = (ci[1] / n) * 100
    return p * 100, lower, upper


def plot_episode_type_comparison(noisy, clean, missing, label, baseline=None):
    """Bar chart comparing survival rates across clinical episode types."""
    groups = ['Clean\nepisodes', 'Noisy\nepisodes', 'Missing\nfeatures']
    data   = [clean, noisy, missing]

    rates, lowers, uppers = [], [], []
    for d in data:
        rate, lower, upper = survival_ci(d)
        rates.append(rate)
        lowers.append(rate - lower)
        uppers.append(upper - rate)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(groups, rates, color="#2f94d7", width=0.5,
                  yerr=[lowers, uppers], capsize=5,
                  error_kw={'ecolor': 'black', 'linewidth': 1.2})

    for bar, rate, n in zip(bars, rates, [len(d) for d in data]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(uppers) + 3,
                f'{rate:.1f}%\n(n={n})',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Survival Rate (%)')
    ax.set_title(f'{label} — Survival Rate by Episode Type', fontweight='bold')
    ax.set_ylim(0, 100)

    if baseline is not None:
        ax.axhline(y=baseline * 100, color='#0a1f35',
                   linestyle='--', linewidth=1.2, label='Random baseline')
        ax.legend()

    plt.tight_layout()
    plt.show()
