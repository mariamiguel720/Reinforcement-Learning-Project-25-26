# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import time
import os
import torch.nn.functional as F
from torch.distributions import Categorical
import itertools
import copy
import optuna

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




#-------------------------------------- PPO — ActorCritic Network & RolloutBuffer --------------------------------------

class ActorCritic(nn.Module):
    """
    Shared backbone with two heads:
    - Actor: outputs logits over 25 actions
    - Critic: outputs a scalar state value V(s)

    Tanh activations and orthogonal initialisation are standard
    for PPO — they stabilise the policy gradient updates.

    Parameters
    ----------
    obs_dim   : int — number of input features (default: 47)
    n_actions : int — number of discrete actions (default: 25)
    hidden    : int — number of units in each hidden layer (default: 256)
    """
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor head — policy
        self.actor_head  = nn.Linear(hidden, n_actions)

        # Critic head — value
        self.critic_head = nn.Linear(hidden, 1)

        # Orthogonal initialisation
        self._init_weights()

    def _init_weights(self):
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits   = self.actor_head(features)   # (batch, 25)
        value    = self.critic_head(features)  # (batch, 1)
        return logits, value

    def get_action(self, obs: torch.Tensor):
        """
        Sample an action from the current policy.

        Returns
        -------
        action   : Tensor (batch,) — sampled action index
        log_prob : Tensor (batch,) — log-probability of the sampled action
        value    : Tensor (batch,) — critic estimate V(s)
        """
        logits, value = self.forward(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Re-evaluate stored actions under the current policy (used in the PPO update).

        Returns
        -------
        log_prob : Tensor (batch,) — log-probability of actions under current policy
        value    : Tensor (batch,) — critic estimate V(s)
        entropy  : Tensor (batch,) — policy entropy (used in the entropy bonus)
        """
        logits, value = self.forward(obs)
        dist    = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """
    Stores one batch of on-policy experience.

    Unlike ReplayBuffer (DQN), this buffer is cleared after every PPO update —
    on-policy data collected under the old policy cannot be reused.
    """
    def __init__(self):
        self.states, self.actions   = [], []
        self.rewards, self.dones    = [], []
        self.log_probs, self.values = [], []

    def push(self, state, action, reward, done, log_prob, value):
        """Append a single transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)



# GAE COMPUTATION FUNCTION
def compute_gae(rewards, values, dones, lam=0.95):
    """
    Generalised Advantage Estimation (GAE).

    Combines TD(0) and Monte Carlo returns to balance bias and variance.
    The terminal state is bootstrapped with value 0.

    Parameters
    ----------
    rewards : list[float] — per-step rewards from the rollout
    values  : list[Tensor] — critic estimates V(s) for each step
    dones   : list[float] — episode termination flags (1.0 = terminal)
    lam     : float — GAE lambda; 0 = TD(0), 1 = Monte Carlo

    Returns
    -------
    list[float] — advantage estimates for each step in the rollout
    """
    advantages = []
    gae = 0.0
    values_np = [v.item() for v in values] + [0.0]  # bootstrap with 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values_np[t+1] * (1 - dones[t]) - values_np[t]
        gae   = delta + GAMMA * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return advantages



def train_ppo_optuna(
    n_updates         = 150,
    n_steps           = 4096,
    n_epochs          = 10,
    batch_size        = 256,
    lr                = 1e-4,
    clip              = 0.2,
    gae_lam           = 0.95,
    ent_coef          = 0.02,
    vf_coef           = 0.5,
    max_grad          = 0.5,
    seed              = 42,
    eval_every        = 25,       # evaluate every N updates
    eval_episodes     = 100,      # 100 episodes: lower noise than 30
    trial             = None,     # Optuna trial object; None = normal run
    use_inner_pruning = False,    # only active for the first two seeds
    save_dir          = None,     # None during tuning, path for final retrain
    save_tag          = 'ppo',
):
    """
    PPO training function compatible with Optuna.

    When `trial` is provided and `use_inner_pruning=True`, the function
    reports the mean return at each eval checkpoint to Optuna and raises
    `TrialPruned` if the pruner decides to cut the trial early.

    Parameters
    ----------
    n_updates         : int   — number of PPO update iterations
    n_steps           : int   — environment steps collected per update (rollout size)
    n_epochs          : int   — number of epochs to reuse each rollout
    batch_size        : int   — mini-batch size for the PPO update
    lr                : float — Adam learning rate
    clip              : float — PPO clipping epsilon
    gae_lam           : float — GAE lambda for advantage estimation
    ent_coef          : float — entropy bonus coefficient
    vf_coef           : float — value function loss coefficient
    max_grad          : float — maximum L2 norm for gradient clipping
    seed              : int   — random seed for reproducibility
    eval_every        : int   — run a greedy evaluation every N updates
    eval_episodes     : int   — number of greedy rollouts per evaluation checkpoint
    trial             : optuna.Trial | None — if None, runs without Optuna integration
    use_inner_pruning : bool  — whether to report to the pruner at each eval checkpoint
    save_dir          : str | None — if provided, best checkpoint and returns are saved here
    save_tag          : str   — prefix for saved file names

    Returns
    -------
    dict with keys:
        'model'        : ActorCritic — trained model (eval mode)
        'best_eval'    : float       — best mean return across eval checkpoints
        'best_state'   : dict        — state_dict of the best checkpoint
        'eval_history' : list[tuple] — (update_idx, mean_return) at each checkpoint
        'returns'      : list[float] — per-episode returns during training
        'duration_s'   : float       — wall-clock training time in seconds
    """
    # Reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model     = ActorCritic().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer    = RolloutBuffer()
    # Initialise best_state to current weights so it is never None,
    # even if the trial is pruned before the first eval checkpoint.
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    all_returns  = []
    eval_history = []
    best_eval    = -float('inf')
    t0           = time.time()

    env      = make_clinical_env()
    env_eval = make_clinical_env()

    obs, _    = env.reset(seed=int(rng.integers(1_000_000)))
    ep_return = 0.0

    for update in range(n_updates):

        # Collect rollout
        buffer.clear()
        current_ep_returns = []

        for step in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs_t)

            next_obs, reward, te, tr, _ = env.step(action.item())
            done = te or tr

            buffer.push(obs, action.item(), reward, float(done),
                        log_prob.item(), value)

            ep_return += reward
            obs        = next_obs

            if done:
                all_returns.append(ep_return)
                current_ep_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset(seed=int(rng.integers(1_000_000)))

        # Compute advantages (GAE)
        advantages_raw = compute_gae(buffer.rewards, buffer.values, buffer.dones,
                                     lam=gae_lam)
        advantages_raw = torch.FloatTensor(advantages_raw).to(device)
        # Critic targets: raw GAE + V(s)  — must be computed BEFORE normalisation
        returns_gae    = advantages_raw + torch.FloatTensor(
            [v.item() for v in buffer.values]).to(device)
        # Normalise advantages for the actor loss only (reduces gradient variance)
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        states_t  = torch.FloatTensor(np.array(buffer.states)).to(device)
        actions_t = torch.LongTensor(buffer.actions).to(device)
        old_lp_t  = torch.FloatTensor(buffer.log_probs).to(device)

        # PPO update (n_epochs over same rollout)
        indices = np.arange(n_steps)
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            # Stop before last incomplete mini-batch to avoid empty idx tensors
            for start in range(0, n_steps - batch_size + 1, batch_size):
                idx = indices[start:start + batch_size]

                log_prob, value, entropy = model.evaluate(states_t[idx],
                                                          actions_t[idx])
                ratio      = torch.exp(log_prob - old_lp_t[idx])
                surr1      = ratio * advantages[idx]
                surr2      = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value, returns_gae[idx])
                entropy_loss = -entropy.mean()
                loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimiser.step()

        # Periodic evaluation checkpoint
        if (update + 1) % eval_every == 0 or update == n_updates - 1:
            model.eval()
            eval_returns = []
            with torch.no_grad():
                for _ in range(eval_episodes):
                    e_obs, _ = env_eval.reset(
                        seed=int(rng.integers(1_000_000)))
                    e_ret, e_done = 0.0, False
                    while not e_done:
                        e_obs_t = torch.FloatTensor(e_obs).unsqueeze(0).to(device)
                        e_action, _, _ = model.get_action(e_obs_t)
                        e_obs, e_r, e_te, e_tr, _ = env_eval.step(e_action.item())
                        e_ret  += e_r
                        e_done  = e_te or e_tr
                    eval_returns.append(e_ret)
            model.train()

            mean_eval = float(np.mean(eval_returns))
            eval_history.append((update + 1, mean_eval))

            # Track best checkpoint
            if mean_eval > best_eval:
                best_eval  = mean_eval
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}

            # Report to Optuna and (optionally) prune
            if trial is not None and use_inner_pruning:
                trial.report(mean_eval, step=update + 1)
                if trial.should_prune():
                    env.close()
                    env_eval.close()
                    raise optuna.TrialPruned()

    env.close()
    env_eval.close()
    duration = time.time() - t0

    # Persist to disk only for the final retrain
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, f'{save_dir}/{save_tag}.pth')
        np.save(f'{save_dir}/{save_tag}_returns.npy', np.array(all_returns))
        np.save(f'{save_dir}/{save_tag}_eval_history.npy',
                np.array(eval_history, dtype=object), allow_pickle=True)

    model.eval()
    return {
        'model':        model,
        'best_eval':    best_eval,
        'best_state':   best_state,
        'eval_history': eval_history,
        'returns':      all_returns,
        'duration_s':   duration,
    }

def objective_ppo(trial, seeds, hp):
    """
    Optuna objective for PPO hyperparameter search.

    Delegates to run_multi_seed_trial_ppo and returns the mean metric across
    seeds. Intended to be wrapped in a lambda when passed to study.optimize().

    Parameters
    ----------
    trial : optuna.Trial  — current Optuna trial
    seeds : list[int]     — random seeds to average over for robustness
    hp    : dict          — hyperparameters built from trial.suggest_* calls;
                            must include 'n_updates'

    Returns
    -------
    float — mean eval metric across all seeds
    """
    seed_metrics = run_multi_seed_trial_ppo(
        trial     = trial,
        hp        = hp,
        seeds     = seeds,
        n_updates = hp['n_updates'],
    )
    return float(np.mean(seed_metrics))

def run_multi_seed_trial_ppo(trial, hp, seeds=[42, 123, 7], n_updates=150):
    """
    Run multiple seeds for a single Optuna PPO trial.

    The first `PRUNE_SEEDS` seeds use inner pruning (reporting intermediate metrics
    at each eval checkpoint). The remaining seeds run without inner pruning to
    provide a more stable estimate before the final pruning decision between seeds.

    Outer pruning reports at step n_updates + i + 1, strictly beyond the inner
    range, so the MedianPruner always compares trials at consistent checkpoints.

    Parameters
    ----------
    trial     : optuna.Trial — current Optuna trial for reporting and pruning
    hp        : dict — hyperparameters to pass to train_ppo_optuna (must NOT contain 'n_updates')
    seeds     : list[int] — random seeds to run; each seed is one independent training run
    n_updates : int — number of PPO updates per seed

    Returns
    -------
    list[float] — per-seed metric (mean of last 3 eval checkpoints for each seed)
    """
    seed_metrics = []
    seed_times   = []
    PRUNE_SEEDS  = 2   # inner pruning active for the first 2 seeds only

    for i, seed in enumerate(seeds):
        hp_clean = {k: v for k, v in hp.items() if k != "n_updates"}
        result = train_ppo_optuna(
            trial             = trial,
            seed              = seed,
            n_updates         = n_updates,
            use_inner_pruning = (i < PRUNE_SEEDS),
            save_dir          = None,        # no disk I/O during tuning
            **hp_clean,
        )

        # Use mean of the last 3 eval checkpoints as the seed metric
        last3 = [r for _, r in result['eval_history'][-3:]]
        m     = float(np.mean(last3))
        seed_metrics.append(m)
        seed_times.append(result['duration_s'])

        # Report running mean after each seed (step beyond inner pruning range)
        running_mean = float(np.mean(seed_metrics))
        trial.report(running_mean, step=n_updates + i + 1)

        # Prune between seeds if running mean is already bad after seed 2+
        if i >= 1 and trial.should_prune():
            raise optuna.TrialPruned()

    # Store per-seed diagnostics for analysis
    trial.set_user_attr('seed_metrics', seed_metrics)
    trial.set_user_attr('seed_times_s', seed_times)
    trial.set_user_attr('total_time_s', float(sum(seed_times)))
    trial.set_user_attr('mean_metric',  float(np.mean(seed_metrics)))
    trial.set_user_attr('std_metric',   float(np.std(seed_metrics)))

    return seed_metrics


