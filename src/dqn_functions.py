# FUNCTION CONFIG_B 

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





#----------------------------------------------- DQN - Network and Replay Buffer -----------------------------------------------
class QNetwork(nn.Module):
    """
    Q-value approximator: Q(s, a) for all 25 actions simultaneously.

    LayerNorm after each hidden layer handles the different scales of the
    47 physiological features without requiring explicit observation normalisation.

    Parameters
    ----------
    obs_dim   : int — number of input features (default: 47)
    n_actions : int — number of discrete actions (default: 25)
    hidden    : int — number of units in each hidden layer (default: 256)
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
        """Return Q-values for all actions given observation x. Shape: (batch, n_actions)."""
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (s, a, r, s', done) transitions.
    Samples uniform random mini-batches to break temporal correlations.

    Parameters
    ----------
    capacity : int — maximum number of transitions stored; oldest are dropped when full.
    """
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a single transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch of transitions.

        Returns
        -------
        Tuple of five np.ndarrays: (states, actions, rewards, next_states, dones).
        """
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
    

# DQN TRAINING FUNCTION
def train_dqn(
    run_id:          int   = 1,
    n_episodes:      int   = 25_000,
    buffer_capacity: int   = 100_000,
    batch_size:      int   = 64,
    min_buffer:      int   = 500,
    lr:              float = 1e-3,
    grad_clip:       float = 10.0,
    eps_start:       float = 1.0,
    eps_end:         float = 0.05,
    eps_decay:       float = 0.9995,
    target_update:   int   = 50,
    save_dir:        str   = 'models',
    verbose:         int   = 1000,
):
    """
    Train a DQN agent on the Config B clinical environment.

    Saves model weights, episode returns, and TD losses to `save_dir` at the
    end of training so results survive session restarts.

    Parameters
    ----------
    run_id          : int   — label used in saved file names (dqn_run{run_id}.*)
    n_episodes      : int   — total number of training episodes
    buffer_capacity : int   — maximum transitions stored in the replay buffer
    batch_size      : int   — number of transitions sampled per gradient step
    min_buffer      : int   — minimum buffer size before gradient updates begin
    lr              : float — Adam learning rate
    grad_clip       : float — maximum L2 norm for gradient clipping
    eps_start       : float — starting epsilon for epsilon-greedy exploration
    eps_end         : float — minimum epsilon (exploration floor)
    eps_decay       : float — multiplicative decay applied to epsilon each episode
    target_update   : int   — frequency (in episodes) of hard target-network updates
    save_dir        : str   — directory where weights and arrays are saved
    verbose         : int   — print a progress line every N episodes; 0 = silent

    Returns
    -------
    dict with keys:
        'online'      : QNetwork  — trained online network (eval mode)
        'returns'     : list[float] — per-episode returns (len = n_episodes)
        'losses'      : list[float] — per-step TD losses (len = gradient steps)
        'total_steps' : int         — total environment steps taken
        'train_time'  : float       — wall-clock training time in seconds
        'run_id'      : int         — the run_id passed in
    """

    # networks & optimiser 
    online = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimiser = torch.optim.Adam(online.parameters(), lr=lr)
    buffer    = ReplayBuffer(buffer_capacity)

    # training state
    all_returns = []
    all_losses  = []
    eps         = eps_start
    total_steps = 0
    t0          = time.time()

    env = make_clinical_env()

    # main loop
    for ep in range(n_episodes):

        obs, _    = env.reset(seed=np.random.randint(100_000))
        ep_return = 0.0
        done      = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < eps:
                action = env.action_space.sample()                    # explore
            else:
                with torch.no_grad():
                    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = int(online(obs_t).argmax(1).item())      # exploit

            next_obs, reward, te, tr, _ = env.step(action)
            done = te or tr
            buffer.push(obs, action, reward, next_obs, float(done))
            obs        = next_obs
            ep_return += reward
            total_steps += 1

            # Gradient update (only once replay buffer is large enough)
            if len(buffer) >= min_buffer:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states_t      = torch.FloatTensor(states).to(device)
                actions_t     = torch.LongTensor(actions).to(device)
                rewards_t     = torch.FloatTensor(rewards).to(device)
                next_states_t = torch.FloatTensor(next_states).to(device)
                dones_t       = torch.FloatTensor(dones).to(device)

                # Current Q-values
                q_values = online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # Target Q-values (Bellman equation)
                with torch.no_grad():
                    next_q   = target(next_states_t).max(1)[0]
                    q_target = rewards_t + GAMMA * next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_values, q_target)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
                optimiser.step()
                all_losses.append(loss.item())

        all_returns.append(ep_return)
        eps = max(eps_end, eps * eps_decay)

        # Hard update target network
        if ep % target_update == 0:
            target.load_state_dict(online.state_dict())

        # Progress logging
        if verbose and ep % verbose == 0:
            mean_ret = np.mean(all_returns[-100:]) if all_returns else 0.0
            print(
                f'[Run {run_id}] Episode {ep:>6}/{n_episodes} | '
                f'Eps: {eps:.3f} | '
                f'Mean return (last 100): {mean_ret:.4f}'
            )

    env.close()
    train_time = time.time() - t0

    # save artefacts
    os.makedirs(save_dir, exist_ok=True)
    torch.save(online.state_dict(),                    f'{save_dir}/dqn_run{run_id}.pth')
    np.save(f'{save_dir}/dqn_returns_run{run_id}.npy', np.array(all_returns))
    np.save(f'{save_dir}/dqn_losses_run{run_id}.npy',  np.array(all_losses))

    # final summary
    online.eval()
    print(f'\nTraining complete in {train_time:.1f}s')
    print(f'Total steps:                {total_steps:,}')
    print(f'Final epsilon:              {eps:.4f}')
    print(f'Mean return (last 1000 ep): {np.mean(all_returns[-1000:]):.4f}')
    print(f'Saved → {save_dir}/dqn_run{run_id}.*')

    return {
        'online':      online,
        'returns':     all_returns,
        'losses':      all_losses,
        'total_steps': total_steps,
        'train_time':  train_time,
        'run_id':      run_id,
    }

# -------------------------------------- DQN — Optuna-compatible training --------------------------------------

def train_dqn_optuna(
    n_episodes:      int   = 10_000,
    buffer_capacity: int   = 100_000,
    batch_size:      int   = 64,
    min_buffer:      int   = 500,
    lr:              float = 1e-3,
    grad_clip:       float = 10.0,
    eps_start:       float = 1.0,
    eps_end:         float = 0.05,
    eps_decay:       float = 0.9995,
    target_update:   int   = 50,
    seed:            int   = 42,
    eval_every:      int   = 1000,   # evaluate every N episodes
    eval_episodes:   int   = 100,    # greedy rollouts per checkpoint
    trial            = None,         # Optuna trial object; None = normal run
    use_inner_pruning: bool = False,  # only active for the first two seeds
    save_dir:        str  = None,    # None during tuning, path for final retrain
    save_tag:        str  = 'dqn',
):
    """
    DQN training function compatible with Optuna.

    When `trial` is provided and `use_inner_pruning=True`, the function
    reports the mean return at each eval checkpoint (step = eval_count, starting
    at 1) and raises `TrialPruned` if the pruner decides to cut early.

    Inner pruning uses steps in [1, n_evals].
    Outer pruning (in run_multi_seed_trial_dqn) uses steps beyond n_evals,
    so the two ranges never overlap and the MedianPruner compares consistently.

    Parameters
    ----------
    n_episodes        : int   — total number of training episodes
    buffer_capacity   : int   — maximum transitions stored in the replay buffer
    batch_size        : int   — number of transitions sampled per gradient step
    min_buffer        : int   — minimum buffer size before gradient updates begin
    lr                : float — Adam learning rate
    grad_clip         : float — maximum L2 norm for gradient clipping
    eps_start         : float — starting epsilon for epsilon-greedy exploration
    eps_end           : float — minimum epsilon (exploration floor)
    eps_decay         : float — multiplicative decay applied to epsilon each episode
    target_update     : int   — frequency (in episodes) of hard target-network updates
    seed              : int   — random seed for reproducibility
    eval_every        : int   — run a greedy evaluation every N episodes
    eval_episodes     : int   — number of greedy rollouts per evaluation checkpoint
    trial             : optuna.Trial | None — if None, runs without Optuna integration
    use_inner_pruning : bool  — whether to report to the pruner at each eval checkpoint
    save_dir          : str | None — if provided, best checkpoint and returns are saved here
    save_tag          : str   — prefix for saved file names

    Returns
    -------
    dict with keys:
        'online'       : QNetwork    — trained online network (eval mode)
        'best_eval'    : float       — best mean return across eval checkpoints
        'best_state'   : dict        — state_dict of the best checkpoint
        'eval_history' : list[tuple] — (episode, mean_return) at each checkpoint
        'returns'      : list[float] — per-episode returns during training
        'duration_s'   : float       — wall-clock training time in seconds
        'n_evals'      : int         — total number of eval checkpoints completed
    """
    # Reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    online = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimiser = torch.optim.Adam(online.parameters(), lr=lr)
    buffer    = ReplayBuffer(buffer_capacity)
    best_state = {k: v.clone() for k, v in online.state_dict().items()}

    all_returns  = []
    eval_history = []
    best_eval    = -float('inf')
    eval_count   = 0
    eps          = eps_start
    t0           = time.time()

    env      = make_clinical_env()
    env_eval = make_clinical_env()

    for ep in range(n_episodes):
        obs, _    = env.reset(seed=int(rng.integers(1_000_000)))
        ep_return = 0.0
        done      = False

        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = int(online(obs_t).argmax(1).item())

            next_obs, reward, te, tr, _ = env.step(action)
            done = te or tr
            buffer.push(obs, action, reward, next_obs, float(done))
            obs        = next_obs
            ep_return += reward

            if len(buffer) >= min_buffer:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_t      = torch.FloatTensor(states).to(device)
                actions_t     = torch.LongTensor(actions).to(device)
                rewards_t     = torch.FloatTensor(rewards).to(device)
                next_states_t = torch.FloatTensor(next_states).to(device)
                dones_t       = torch.FloatTensor(dones).to(device)

                q_values = online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q   = target(next_states_t).max(1)[0]
                    q_target = rewards_t + GAMMA * next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_values, q_target)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
                optimiser.step()

        all_returns.append(ep_return)
        eps = max(eps_end, eps * eps_decay)

        if ep % target_update == 0:
            target.load_state_dict(online.state_dict())

        # Periodic evaluation checkpoint
        if (ep + 1) % eval_every == 0 or ep == n_episodes - 1:
            online.eval()
            eval_returns = []
            with torch.no_grad():
                for _ in range(eval_episodes):
                    e_obs, _ = env_eval.reset(seed=int(rng.integers(1_000_000)))
                    e_ret, e_done = 0.0, False
                    while not e_done:
                        e_obs_t = torch.FloatTensor(e_obs).unsqueeze(0).to(device)
                        e_action = int(online(e_obs_t).argmax(1).item())
                        e_obs, e_r, e_te, e_tr, _ = env_eval.step(e_action)
                        e_ret  += e_r
                        e_done  = e_te or e_tr
                    eval_returns.append(e_ret)
            online.train()

            mean_eval = float(np.mean(eval_returns))
            eval_count += 1
            eval_history.append((ep + 1, mean_eval))

            if mean_eval > best_eval:
                best_eval  = mean_eval
                best_state = {k: v.clone() for k, v in online.state_dict().items()}

            # Report to Optuna using eval_count as step — stays in [1, n_evals]
            # so it never overlaps with the outer pruning steps used in
            # run_multi_seed_trial_dqn (which start at n_evals + 1).
            if trial is not None and use_inner_pruning:
                trial.report(mean_eval, step=eval_count)
                if trial.should_prune():
                    env.close()
                    env_eval.close()
                    raise optuna.TrialPruned()

    env.close()
    env_eval.close()
    duration = time.time() - t0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, f'{save_dir}/{save_tag}.pth')
        np.save(f'{save_dir}/{save_tag}_returns.npy', np.array(all_returns))
        np.save(f'{save_dir}/{save_tag}_eval_history.npy',
                np.array(eval_history, dtype=object), allow_pickle=True)

    online.eval()
    return {
        'online':       online,
        'best_eval':    best_eval,
        'best_state':   best_state,
        'eval_history': eval_history,
        'returns':      all_returns,
        'duration_s':   duration,
        'n_evals':      eval_count,
    }


def run_multi_seed_trial_dqn(trial, hp, seeds=[42, 123, 7], n_episodes=10_000):
    """
    Run multiple seeds for a single Optuna DQN trial.

    The first `PRUNE_SEEDS` seeds use inner pruning (steps 1..n_evals).
    After each seed the running mean is reported at step n_evals + i + 1,
    which is strictly beyond the inner range — no step overlap, so the
    MedianPruner always compares trials at consistent checkpoints.

    Parameters
    ----------
    trial      : optuna.Trial — current Optuna trial for reporting and pruning
    hp         : dict — hyperparameters to pass to train_dqn_optuna (must NOT contain 'n_episodes')
    seeds      : list[int] — random seeds to run; each seed is one independent training run
    n_episodes : int — number of training episodes per seed

    Returns
    -------
    list[float] — per-seed metric (mean of last 3 eval checkpoints for each seed)
    """
    seed_metrics = []
    seed_times   = []
    n_evals      = None   # filled after first seed completes
    PRUNE_SEEDS  = 2

    for i, seed in enumerate(seeds):
        hp_clean = {k: v for k, v in hp.items() if k != 'n_episodes'}
        result = train_dqn_optuna(
            trial             = trial,
            seed              = seed,
            n_episodes        = n_episodes,
            use_inner_pruning = (i < PRUNE_SEEDS),
            save_dir          = None,
            **hp_clean,
        )

        # Capture n_evals from the first seed (consistent across seeds with same hp)
        if n_evals is None:
            n_evals = result['n_evals']

        last3 = [r for _, r in result['eval_history'][-3:]]
        m     = float(np.mean(last3))
        seed_metrics.append(m)
        seed_times.append(result['duration_s'])

        # Outer pruning step is strictly beyond the inner range
        running_mean = float(np.mean(seed_metrics))
        trial.report(running_mean, step=n_evals + i + 1)

        if i >= 1 and trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr('seed_metrics', seed_metrics)
    trial.set_user_attr('seed_times_s', seed_times)
    trial.set_user_attr('total_time_s', float(sum(seed_times)))
    trial.set_user_attr('mean_metric',  float(np.mean(seed_metrics)))
    trial.set_user_attr('std_metric',   float(np.std(seed_metrics)))

    return seed_metrics


def objective_dqn(trial, seeds, hp):
    """
    Optuna objective for DQN hyperparameter search.

    Delegates to run_multi_seed_trial_dqn and returns the mean metric across
    seeds. Intended to be wrapped in a lambda when passed to study.optimize().

    Parameters
    ----------
    trial : optuna.Trial  — current Optuna trial
    seeds : list[int]     — random seeds to average over for robustness
    hp    : dict          — hyperparameters built from trial.suggest_* calls;
                            must include 'n_episodes'

    Returns
    -------
    float — mean eval metric across all seeds
    """
    seed_metrics = run_multi_seed_trial_dqn(
        trial      = trial,
        hp         = hp,
        seeds      = seeds,
        n_episodes = hp['n_episodes'],
    )
    return float(np.mean(seed_metrics))



# -------------------------------------- Double DQN — Network and Training Function --------------------------------------

class QNetworkV2(nn.Module):
    """
    Enhanced Q-value approximator for Double DQN.

    Extends the original QNetwork with an additional hidden layer (256-256-128)
    to increase representational capacity. LayerNorm after each hidden layer
    stabilises training across the heterogeneous scales of the 47 physiological
    features, without requiring explicit observation normalisation.

    Compared to QNetwork (two hidden layers of 256 units), the extra 128-unit
    layer adds a compression stage before the output, encouraging the network
    to form more abstract representations of the clinical state before
    mapping to action values.

    Parameters
    ----------
    obs_dim   : int — number of input features (default: 47)
    n_actions : int — number of discrete actions (default: 25)
    """
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),     nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),     nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions given observation x. Shape: (batch, n_actions)."""
        return self.net(x)


def train_double_dqn(
    n_episodes      = 100_000,
    buffer_capacity = 50_000,
    batch_size      = 32,
    min_buffer      = 500,
    lr              = 2.19e-05,
    grad_clip       = 10.0,
    eps_start       = 1.0,
    eps_end         = 0.05,
    eps_decay       = 0.999285,
    target_update   = 25,
    seed            = 42,
    eval_every      = 1_000,
    eval_episodes   = 100,
    save_dir        = None,
    save_tag        = 'ddqn',
):
    """
    Train a Double DQN agent on the Config B clinical environment.

    Extends the standard DQN training loop with the Double DQN target update,
    which decouples action selection from action evaluation to reduce
    overestimation bias. In vanilla DQN, the target network is used for both
    selecting and evaluating the best next action, systematically overestimating
    Q-values in stochastic environments. Double DQN instead uses the online
    network to select the greedy next action and the target network only to
    evaluate its value, producing less biased Bellman targets and more stable
    learning in noisy clinical settings.

    All default hyperparameters match the best configuration identified by the
    Optuna search on the standard DQN, so that any performance difference can
    be attributed to the algorithmic improvement rather than re-tuning.

    The best checkpoint (highest mean eval return) is restored at the end of
    training rather than using the final weights, guarding against late-stage
    instability or regression.

    Parameters
    ----------
    n_episodes      : int   — total number of training episodes
    buffer_capacity : int   — maximum transitions stored in the replay buffer;
                              oldest transitions are dropped when full
    batch_size      : int   — number of transitions sampled per gradient step
    min_buffer      : int   — minimum buffer size before gradient updates begin
    lr              : float — Adam learning rate (default matches Optuna best)
    grad_clip       : float — maximum L2 norm for gradient clipping
    eps_start       : float — starting epsilon for epsilon-greedy exploration
    eps_end         : float — minimum epsilon (exploration floor)
    eps_decay       : float — multiplicative decay applied to epsilon each episode
                              (default matches Optuna best: slow decay, prolonged exploration)
    target_update   : int   — frequency (in episodes) of hard target-network updates
    seed            : int   — random seed for reproducibility
    eval_every      : int   — run a greedy evaluation every N episodes
    eval_episodes   : int   — number of greedy rollouts per evaluation checkpoint
    save_dir        : str | None — if provided, best checkpoint and returns are saved here
    save_tag        : str   — prefix for saved file names

    Returns
    -------
    dict with keys:
        'online'       : QNetworkV2  — trained online network (eval mode, best checkpoint)
        'best_eval'    : float       — best mean return across all eval checkpoints
        'best_state'   : dict        — state_dict corresponding to best_eval
        'eval_history' : list[tuple] — (episode, mean_return) at each checkpoint
        'returns'      : list[float] — per-episode returns during training
        'duration_s'   : float       — wall-clock training time in seconds
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    online = QNetworkV2().to(device)
    target = QNetworkV2().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimiser  = torch.optim.Adam(online.parameters(), lr=lr)
    buffer     = ReplayBuffer(buffer_capacity)
    best_state = {k: v.clone() for k, v in online.state_dict().items()}

    all_returns  = []
    eval_history = []
    best_eval    = -float('inf')
    eps          = eps_start
    t0           = time.time()

    env      = make_clinical_env()
    env_eval = make_clinical_env()

    for ep in range(n_episodes):
        obs, _    = env.reset(seed=int(rng.integers(1_000_000)))
        ep_return = 0.0
        done      = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < eps:
                action = env.action_space.sample()                       # explore
            else:
                with torch.no_grad():
                    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = int(online(obs_t).argmax(1).item())         # exploit

            next_obs, reward, te, tr, _ = env.step(action)
            done = te or tr
            buffer.push(obs, action, reward, next_obs, float(done))
            obs        = next_obs
            ep_return += reward

            # Gradient update (only once replay buffer is large enough)
            if len(buffer) >= min_buffer:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_t      = torch.FloatTensor(states).to(device)
                actions_t     = torch.LongTensor(actions).to(device)
                rewards_t     = torch.FloatTensor(rewards).to(device)
                next_states_t = torch.FloatTensor(next_states).to(device)
                dones_t       = torch.FloatTensor(dones).to(device)

                # Current Q-values for the actions taken
                q_values = online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # Double DQN target: online selects the next action, target evaluates it.
                # This breaks the positive feedback loop in vanilla DQN where the same
                # network both picks and scores the action, leading to overestimation.
                with torch.no_grad():
                    next_actions = online(next_states_t).argmax(1)
                    next_q       = target(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    q_target     = rewards_t + GAMMA * next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_values, q_target)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
                optimiser.step()

        all_returns.append(ep_return)
        eps = max(eps_end, eps * eps_decay)

        # Hard update: copy online weights into target network
        if ep % target_update == 0:
            target.load_state_dict(online.state_dict())

        # Periodic evaluation checkpoint
        if (ep + 1) % eval_every == 0 or ep == n_episodes - 1:
            online.eval()
            eval_returns = []
            with torch.no_grad():
                for _ in range(eval_episodes):
                    e_obs, _ = env_eval.reset(seed=int(rng.integers(1_000_000)))
                    e_ret, e_done = 0.0, False
                    while not e_done:
                        e_obs_t  = torch.FloatTensor(e_obs).unsqueeze(0).to(device)
                        e_action = int(online(e_obs_t).argmax(1).item())
                        e_obs, e_r, e_te, e_tr, _ = env_eval.step(e_action)
                        e_ret  += e_r
                        e_done  = e_te or e_tr
                    eval_returns.append(e_ret)
            online.train()

            mean_eval = float(np.mean(eval_returns))
            eval_history.append((ep + 1, mean_eval))

            # Track best checkpoint across the full training run
            if mean_eval > best_eval:
                best_eval  = mean_eval
                best_state = {k: v.clone() for k, v in online.state_dict().items()}

            if (ep + 1) % 10_000 == 0:
                elapsed = (time.time() - t0) / 60
                print(f'  ep {ep+1:>6d} | eval {mean_eval:.4f} | best {best_eval:.4f} | eps {eps:.4f} | {elapsed:.1f} min')

    env.close()
    env_eval.close()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, f'{save_dir}/{save_tag}.pth')
        np.save(f'{save_dir}/{save_tag}_returns.npy', np.array(all_returns))
        np.save(f'{save_dir}/{save_tag}_eval_history.npy',          
                np.array(eval_history, dtype=object), allow_pickle=True)

    online.eval()
    return {
        'online':       online,
        'best_eval':    best_eval,
        'best_state':   best_state,
        'eval_history': eval_history,
        'returns':      all_returns,
        'duration_s':   time.time() - t0,
    }