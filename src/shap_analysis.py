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
import torch
import matplotlib.pyplot as plt
import shap

from envs.wrappers import make_clinical_env
from envs.continuous_sepsis_env import FEATURE_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Step 1: Collect observations ─────────────────────────────────────────────

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


# ── SHAP wrappers ─────────────────────────────────────────────────────────────

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


# ── Step 2 & 5: Compute SHAP values ──────────────────────────────────────────

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
    sv    = np.array(sv)
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
    sv    = np.array(sv)
    order = np.argsort(np.abs(sv).mean(axis=0).flatten())[::-1]

    print(f'Value SHAP computed — {n_explain} observations explained.')
    return sv, explain_np, order


# ── Step 3 & 6: Bar chart ─────────────────────────────────────────────────────

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


# ── Step 4 & 7: Beeswarm ─────────────────────────────────────────────────────

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


# ── Step 8: Policy vs Value comparison ───────────────────────────────────────

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
