"""
shap_analysis.py — SHAP Interpretability for PPO (Config B)

Designed to be used step-by-step in the notebook, one function per cell:

  Step 1 — collect_observations()       : roll out PPO and collect states
  Step 2 — compute_shap_policy()        : compute SHAP for policy head
  Step 3 — plot_shap_bar()              : bar chart of top features
  Step 4 — plot_shap_beeswarm()         : beeswarm (direction of impact)
  Step 5 — compute_shap_value()         : compute SHAP for value head
  Step 6 — plot_shap_bar()              : reuse for value
  Step 7 — plot_shap_beeswarm()         : reuse for value
  Step 8 — compare_shap_policy_vs_value(): side-by-side ranking comparison
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap


from src.ppo_functions import *

from envs.wrappers import make_clinical_env
from envs.continuous_sepsis_env import FEATURE_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ------------------------------------------  ABLATION STUDY  ------------------------------------------------------

# All 8 wrapper combinations (False = wrapper OFF, True = wrapper ON)
ABLATION_CONDITIONS = [
    dict(noisy=False, missing=False, acute=False),  # clean baseline
    dict(noisy=True,  missing=False, acute=False),  # noisy only
    dict(noisy=False, missing=True,  acute=False),  # missing only
    dict(noisy=False, missing=False, acute=True),   # acute only
    dict(noisy=True,  missing=True,  acute=False),  # noisy + missing
    dict(noisy=True,  missing=False, acute=True),   # noisy + acute
    dict(noisy=False, missing=True,  acute=True),   # missing + acute
    dict(noisy=True,  missing=True,  acute=True),   # full clinical
]

ABLATION_LABELS = [
    'Baseline (clean)',
    'Noisy only',
    'Missing only',
    'Acute only',
    'Noisy + Missing',
    'Noisy + Acute',
    'Missing + Acute',
    'Full clinical',
]

#  Factory: compose all three wrappers 

def make_ablation_env(
    noisy:  bool = True,
    missing: bool = True,
    acute:  bool = True,
    sofa_bias=SOFA_BIAS, lam=LAM,
    malfunction_prob=0.15, noise_std=0.10,
    missing_prob=0.15,    n_missing=4,
    event_prob=0.01,
):
    """
    Configurable environment for ablation studies.

    Each wrapper can be toggled independently:
        noisy   — EpisodicNoisyObsEnv
        missing — EpisodicMissingObsEnv
        acute   — AcuteEventEnv

    Use make_ablation_env(noisy=False, missing=False, acute=False) for
    the clean baseline and progressively add wrappers to isolate effects.
    """
    from envs.continuous_sepsis_env import ContinuousICUSepsisEnv

    env = ContinuousICUSepsisEnv(params=make_sepsis_env(sofa_bias=sofa_bias, lam=lam))
    if noisy:
        env = EpisodicNoisyObsEnv(env, malfunction_prob=malfunction_prob, noise_std=noise_std)
    if missing:
        env = EpisodicMissingObsEnv(env, missing_prob=missing_prob, n_missing=n_missing)
    if acute:
        env = AcuteEventEnv(env, event_prob=event_prob)
    return env

def train_ppo_ablation(
    condition: dict,
    n_updates=150,
    n_steps=4096,
    n_epochs=10,
    batch_size=256,
    lr=1e-4,
    clip=0.2,
    gae_lam=0.95,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad=0.5,
    eval_every=25,
    eval_episodes=500,
    seed=42,
):
    """
    Train PPO on a single ablation condition and return survival rate.

    `condition` is a dict with keys noisy/missing/acute (bool) passed
    directly to make_ablation_env().
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model     = ActorCritic().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer    = RolloutBuffer()

    env      = make_ablation_env(**condition)
    env_eval = make_ablation_env(**condition)

    obs, _    = env.reset(seed=int(rng.integers(1_000_000)))
    ep_return = 0.0
    all_returns  = []
    eval_history = []
    best_eval    = -float('inf')
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for update in range(n_updates):
        buffer.clear()

        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs_t)

            next_obs, reward, te, tr, _ = env.step(action.item())
            done = te or tr
            buffer.push(obs, action.item(), reward, float(done), log_prob.item(), value)
            ep_return += reward
            obs = next_obs
            if done:
                all_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset(seed=int(rng.integers(1_000_000)))

        advantages_raw = compute_gae(buffer.rewards, buffer.values, buffer.dones, lam=gae_lam)
        advantages_raw = torch.FloatTensor(advantages_raw).to(device)
        returns_gae    = advantages_raw + torch.FloatTensor([v.item() for v in buffer.values]).to(device)
        advantages     = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        states_t  = torch.FloatTensor(np.array(buffer.states)).to(device)
        actions_t = torch.LongTensor(buffer.actions).to(device)
        old_lp_t  = torch.FloatTensor(buffer.log_probs).to(device)

        indices = np.arange(n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps - batch_size + 1, batch_size):
                idx = indices[start:start + batch_size]
                log_prob, value, entropy = model.evaluate(states_t[idx], actions_t[idx])
                ratio      = torch.exp(log_prob - old_lp_t[idx])
                surr1      = ratio * advantages[idx]
                surr2      = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value, returns_gae[idx])
                loss = actor_loss + vf_coef * critic_loss + (-ent_coef * entropy.mean())
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimiser.step()

        if (update + 1) % eval_every == 0 or update == n_updates - 1:
            model.eval()
            eval_returns = []
            with torch.no_grad():
                for _ in range(eval_episodes):
                    e_obs, _ = env_eval.reset(seed=int(rng.integers(1_000_000)))
                    e_ret, e_done = 0.0, False
                    while not e_done:
                        e_obs_t = torch.FloatTensor(e_obs).unsqueeze(0).to(device)
                        e_action, _, _ = model.get_action(e_obs_t)
                        e_obs, e_r, e_te, e_tr, _ = env_eval.step(e_action.item())
                        e_ret += e_r
                        e_done = e_te or e_tr
                    eval_returns.append(e_ret)
            model.train()
            mean_eval = float(np.mean(eval_returns))
            eval_history.append((update + 1, mean_eval))
            if mean_eval > best_eval:
                best_eval  = mean_eval
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    env.close()
    env_eval.close()

    model.load_state_dict(best_state)
    model.eval()
    return {
        'model':        model,
        'best_eval':    best_eval,
        'eval_history': eval_history,
        'returns':      all_returns,
    }


def run_ablation_study(
    best_hp: dict,
    seeds: list = [42, 123, 7],
    n_updates: int = 150,
    eval_episodes: int = 500,
    save_dir: str = 'models/ablation',
):
    """
    Train PPO on all 8 ablation conditions and return a summary DataFrame.

    `best_hp` should be the dict of best hyperparameters from Optuna tuning,
    e.g. {'lr': 1e-4, 'ent_coef': 0.02, 'n_steps': 4096, ...}.
    Results are averaged across `seeds` for robustness.

    Returns a pandas DataFrame with columns:
        condition, noisy, missing, acute, mean_survival, std_survival
    """

    os.makedirs(save_dir, exist_ok=True)

    hp = {k: v for k, v in best_hp.items()
          if k in ('n_steps', 'n_epochs', 'batch_size', 'lr', 'clip',
                   'gae_lam', 'ent_coef', 'vf_coef', 'max_grad')}
    hp['n_updates']     = n_updates
    hp['eval_episodes'] = eval_episodes

    records = []
    for label, condition in zip(ABLATION_LABELS, ABLATION_CONDITIONS):
        print(f'\n{"="*60}')
        print(f'  Condition: {label}')
        print(f'{"="*60}')
        seed_survivals = []
        for seed in seeds:
            print(f'  seed={seed} ... ', end='', flush=True)
            result = train_ppo_ablation(condition=condition, seed=seed, **hp)
            survival = result['best_eval']
            seed_survivals.append(survival)
            print(f'survival={survival:.4f}')

        mean_s = float(np.mean(seed_survivals))
        std_s  = float(np.std(seed_survivals))
        print(f'  → mean={mean_s:.4f}  std={std_s:.4f}')

        records.append({
            'condition':       label,
            'noisy':           condition['noisy'],
            'missing':         condition['missing'],
            'acute':           condition['acute'],
            'mean_survival':   mean_s,
            'std_survival':    std_s,
            'seed_survivals':  seed_survivals,
        })

        # Save intermediate results so progress is not lost
        pd.DataFrame(records).to_csv(f'{save_dir}/ablation_results.csv', index=False)

    df = pd.DataFrame(records)
    print('\n' + '='*60)
    print(df[['condition', 'mean_survival', 'std_survival']].to_string(index=False))
    print('='*60)
    print(f'\nResults saved to {save_dir}/ablation_results.csv')
    return df


def plot_ablation_results(df_ablation, baseline_survival=None):
    """
    Bar chart of survival rate for each ablation condition.
    `df_ablation` is the DataFrame returned by run_ablation_study().
    `baseline_survival` is the clean baseline survival rate (%) — drawn as a dashed line.
    """
    labels   = df_ablation['condition'].tolist()
    means    = df_ablation['mean_survival'].tolist()
    stds     = df_ablation['std_survival'].tolist()

    colors = []
    for _, row in df_ablation.iterrows():
        n_wrappers = int(row['noisy']) + int(row['missing']) + int(row['acute'])
        palette = ['#2f94d7', '#f0a500', '#e06c2f', '#c0392b']
        colors.append(palette[n_wrappers])

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, [m * 100 for m in means],
                  yerr=[s * 100 for s in stds],
                  color=colors, width=0.6, capsize=5,
                  error_kw={'ecolor': 'black', 'linewidth': 1.2})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s * 100 + 1.5,
                f'{m*100:.1f}%',
                ha='center', va='bottom', fontsize=9)

    if baseline_survival is None and len(means) > 0:
        baseline_survival = means[0] * 100  # first row is clean baseline
    if baseline_survival is not None:
        ax.axhline(baseline_survival * (1 if baseline_survival > 1 else 100),
                   color='#0a1f35', linestyle='--', linewidth=1.4,
                   label='Clean baseline')

    ax.axhline(clinical_rand_mean * 100, color='grey',
               linestyle=':', linewidth=1.2, label='Random baseline')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('PPO Ablation Study — Impact of Clinical Reality Wrappers', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    plt.show()

def print_ablation_summary(df_ablation):
    """
    Print a summary table and auto-interpretation of the ablation results.
    `df_ablation` is the DataFrame returned by run_ablation_study().
    """
    baseline = df_ablation.loc[df_ablation['condition'] == 'Baseline (clean)', 'mean_survival'].values[0]

    print('═' * 65)
    print('  PPO Ablation Study — Summary')
    print('═' * 65)
    print(f'  {"Condition":<22} {"Survival":>10} {"Std":>8} {"vs Baseline":>12}')
    print('─' * 65)
    for _, row in df_ablation.iterrows():
        delta = (row['mean_survival'] - baseline) * 100
        print(f'  {row["condition"]:<22} {row["mean_survival"]*100:>9.1f}%'
              f' {row["std_survival"]*100:>7.1f}%'
              f' {delta:>+11.1f}pp')
    print('═' * 65)

    # Identify most damaging single wrapper
    singles = df_ablation[
        df_ablation['condition'].isin(['Noisy only', 'Missing only', 'Acute only'])
    ].copy()
    singles['delta'] = (singles['mean_survival'] - baseline) * 100
    worst = singles.loc[singles['delta'].idxmin()]

    # Check additivity: noisy+missing vs noisy_only + missing_only
    def get(cond): return df_ablation.loc[df_ablation['condition'] == cond, 'mean_survival'].values[0]
    additive_nm  = (get('Noisy only') + get('Missing only') - baseline) * 100
    actual_nm    = (get('Noisy + Missing') - baseline) * 100
    interaction  = actual_nm - additive_nm

    print(f'\n  Most damaging single wrapper : {worst["condition"]} ({worst["delta"]:+.1f}pp)')
    print(f'  Full clinical vs baseline   : {(get("Full clinical") - baseline)*100:+.1f}pp')
    print(f'  Noisy+Missing additivity gap: {interaction:+.1f}pp '
          f'({"synergistic" if interaction < -1 else "antagonistic" if interaction > 1 else "approximately additive"})')
    print('═' * 65)

# ------------------------------------------ SHAP ANALYSIS  ------------------------------------------------------

# Collect observations 
def collect_observations(model, n_episodes: int = 500, seed: int = 42):
    """
    Roll out the trained PPO policy and collect all visited observations.
    Returns a float32 numpy array of shape (N_steps, 47).
    """
    model.eval()
    rng = np.random.default_rng(seed)
    env = make_clinical_env()
    all_obs = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
            done = False
            while not done:
                all_obs.append(obs.copy())
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits, _ = model(obs_t)
                action = int(logits.argmax(1).item())
                obs, _, te, tr, _ = env.step(action)
                done = te or tr

    env.close()
    data = np.array(all_obs, dtype=np.float32)
    print(f'Collected {len(data):,} observations from {n_episodes} episodes.')
    return data


# SHAP wrappers
class _PolicyWrapper(torch.nn.Module):
    """Returns the probability of the most-likely action."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        return probs.max(dim=-1, keepdim=True).values


class _ValueWrapper(torch.nn.Module):
    """Returns V(s) for each observation."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, value = self.model(x)
        return value.unsqueeze(-1) if value.dim() == 1 else value


# Compute SHAP values 
def compute_shap_policy(model, obs_data, n_background=100, n_explain=200, seed=42):
    """
    Compute SHAP values for the policy head (action confidence).
    Returns (shap_values, explain_data, feature_order_by_importance).
    """
    model.eval()
    rng     = np.random.default_rng(seed)
    idx_bg  = rng.choice(len(obs_data), n_background, replace=False)
    idx_exp = rng.choice(len(obs_data), n_explain,    replace=False)

    background = torch.FloatTensor(obs_data[idx_bg]).to(device)
    explain_np = obs_data[idx_exp]
    explain_t  = torch.FloatTensor(explain_np).to(device)

    wrapper   = _PolicyWrapper(model).to(device)
    explainer = shap.DeepExplainer(wrapper, background)
    sv        = explainer.shap_values(explain_t, check_additivity=False)

    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv).squeeze()          # ensure shape (n_explain, 47)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    order = np.argsort(np.abs(sv).mean(axis=0).flatten())[::-1]

    print(f'Policy SHAP computed — {n_explain} observations explained.')
    return sv, explain_np, order


def compute_shap_value(model, obs_data, n_background=100, n_explain=200, seed=42):
    """
    Compute SHAP values for the critic head (state value V(s)).
    Returns (shap_values, explain_data, feature_order_by_importance).
    """
    model.eval()
    rng     = np.random.default_rng(seed)
    idx_bg  = rng.choice(len(obs_data), n_background, replace=False)
    idx_exp = rng.choice(len(obs_data), n_explain,    replace=False)

    background = torch.FloatTensor(obs_data[idx_bg]).to(device)
    explain_np = obs_data[idx_exp]
    explain_t  = torch.FloatTensor(explain_np).to(device)

    wrapper   = _ValueWrapper(model).to(device)
    explainer = shap.DeepExplainer(wrapper, background)
    sv        = explainer.shap_values(explain_t, check_additivity=False)

    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv).squeeze()          # ensure shape (n_explain, 47)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    order = np.argsort(np.abs(sv).mean(axis=0).flatten())[::-1]

    print(f'Value SHAP computed — {n_explain} observations explained.')
    return sv, explain_np, order


# Bar chart
def plot_shap_bar(shap_values, order, title, color='#2471a3', top_n=15):
    """
    Horizontal bar chart of mean |SHAP| for the top_n features.
    Works for both policy and value SHAP — pass the respective shap_values and order.
    """
    mean_abs = np.abs(shap_values).mean(axis=0).flatten()
    top      = order[:top_n]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([FEATURE_NAMES[i] for i in top[::-1]],
            mean_abs[top[::-1]], color=color)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.show()

    print(f'\nTop 10 features:')
    for rank, i in enumerate(top[:10], 1):
        print(f'  {rank:2d}. {FEATURE_NAMES[i]:<30}  mean|SHAP| = {mean_abs[i]:.4f}')


# Beeswarm
def plot_shap_beeswarm(shap_values, explain_data, order, title, top_n=15):
    """
    Beeswarm (summary) plot showing direction and magnitude of each feature's impact.
    Red = high feature value pushes output up; blue = pushes output down.
    Works for both policy and value SHAP.
    """
    top = order[:top_n]
    print(title)
    shap.summary_plot(
        shap_values[:, top],
        explain_data[:, top],
        feature_names=[FEATURE_NAMES[i] for i in top],
        show=True,
        plot_size=(10, 6),
    )


# Policy vs Value comparison
def compare_shap_policy_vs_value(policy_order, value_order, top_n=15):
    """
    Side-by-side rank comparison: which features matter for action choice
    vs which features matter for state value estimation.
    """
    policy_names = [FEATURE_NAMES[i] for i in policy_order[:top_n]]
    value_names  = [FEATURE_NAMES[i] for i in value_order[:top_n]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].barh(policy_names[::-1], range(top_n, 0, -1), color='#2471a3')
    axes[0].set_title('Policy SHAP\n(drives action choice)', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('Importance rank', fontsize=11)
    axes[0].tick_params(labelsize=11)

    axes[1].barh(value_names[::-1], range(top_n, 0, -1), color='#1a5276')
    axes[1].set_title('Value SHAP\n(drives V(s) estimate)', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Importance rank', fontsize=11)
    axes[1].tick_params(labelsize=11)

    plt.suptitle('Policy vs Value — Feature Importance Comparison',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    overlap = set(policy_names) & set(value_names)
    only_policy = set(policy_names) - set(value_names)
    only_value  = set(value_names)  - set(policy_names)

    print(f'\nFeatures in BOTH top-{top_n}  ({len(overlap)}): {sorted(overlap)}')
    print(f'Only in Policy top-{top_n}    ({len(only_policy)}): {sorted(only_policy)}')
    print(f'Only in Value top-{top_n}     ({len(only_value)}): {sorted(only_value)}')
