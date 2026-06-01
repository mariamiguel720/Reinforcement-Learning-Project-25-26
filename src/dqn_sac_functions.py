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




# SAC

# SAC NETWORKS

class SACActor(nn.Module):
    """
    Policy network for discrete SAC.

    Maps observations to a distribution over actions via softmax.
    Both probs and log_probs are returned to avoid recomputation in the
    critic and entropy updates.

    Parameters
    ----------
    obs_dim   : int — number of input features
    n_actions : int — number of discrete actions
    hidden    : int — number of units in each hidden layer
    """
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        """
        Return action probabilities and log-probabilities for all actions.

        Returns
        -------
        probs     : Tensor (batch, n_actions) — softmax probabilities
        log_probs : Tensor (batch, n_actions) — log-softmax (numerically stable)
        """
        logits = self.net(obs)
        probs  = F.softmax(logits, dim=-1)
        # Numerical stability: log(probs) directly via log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample(self, obs):
        """
        Sample a single action from the current policy.

        Returns
        -------
        action    : Tensor (batch,) — sampled action indices
        probs     : Tensor (batch, n_actions)
        log_probs : Tensor (batch, n_actions)
        """
        probs, log_probs = self.forward(obs)
        dist   = Categorical(probs=probs)
        action = dist.sample()
        return action, probs, log_probs


class SACCritic(nn.Module):
    """
    Q-network for discrete SAC. Outputs Q-values for all actions simultaneously.

    Two independent instances are used (double-Q) to reduce overestimation bias.

    Parameters
    ----------
    obs_dim   : int — number of input features
    n_actions : int — number of discrete actions
    hidden    : int — number of units in each hidden layer
    """
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        """Return Q-values for all actions. Shape: (batch, n_actions)."""
        return self.net(obs)


# SAC UPDATE FUNCTION
def update_sac(actor, critic1, critic2, critic1_target, critic2_target,
               actor_optim, critic1_optim, critic2_optim,
               buffer, batch_size, alpha, tau,
               log_dict=None):
    """
    Perform a single gradient update step for all SAC networks.

    Updates both critics via MSE against the Bellman target, then updates the
    actor to maximise expected Q minus entropy, and finally soft-updates both
    target critics with Polyak averaging.

    Parameters
    ----------
    actor, critic1, critic2                   : live networks
    critic1_target, critic2_target            : target networks (no grad)
    actor_optim, critic1_optim, critic2_optim : Adam optimisers
    buffer     : ReplayBuffer — source of training transitions
    batch_size : int   — number of transitions to sample
    alpha      : float — entropy temperature coefficient
    tau        : float — Polyak averaging rate for soft target updates
    log_dict   : dict | None — if provided, losses and diagnostics are appended
    """
    if len(buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states_t      = torch.FloatTensor(states).to(device)
    actions_t     = torch.LongTensor(actions).to(device)
    rewards_t     = torch.FloatTensor(rewards).to(device)
    next_states_t = torch.FloatTensor(next_states).to(device)
    dones_t       = torch.FloatTensor(dones).to(device)

    # CRITIC UPDATE (double-Q: use min of two target critics)
    with torch.no_grad():
        next_probs, next_log_probs = actor(next_states_t)
        next_q1 = critic1_target(next_states_t)
        next_q2 = critic2_target(next_states_t)
        next_q  = torch.min(next_q1, next_q2)
        next_v  = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=1)
        target_q = rewards_t + GAMMA * (1 - dones_t) * next_v

    current_q1 = critic1(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    current_q2 = critic2(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    critic1_loss = F.mse_loss(current_q1, target_q)
    critic2_loss = F.mse_loss(current_q2, target_q)

    critic1_optim.zero_grad(); critic1_loss.backward(); critic1_optim.step()
    critic2_optim.zero_grad(); critic2_loss.backward(); critic2_optim.step()

    # ACTOR UPDATE (use min of live critics)
    probs, log_probs = actor(states_t)
    with torch.no_grad():
        q1_live = critic1(states_t)
        q2_live = critic2(states_t)
        q_values = torch.min(q1_live, q2_live)
    actor_loss = (probs * (alpha * log_probs - q_values)).sum(dim=1).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # SOFT TARGET UPDATE (both critics)
    with torch.no_grad():
        for p, pt in zip(critic1.parameters(), critic1_target.parameters()):
            pt.data.mul_(1 - tau); pt.data.add_(tau * p.data)
        for p, pt in zip(critic2.parameters(), critic2_target.parameters()):
            pt.data.mul_(1 - tau); pt.data.add_(tau * p.data)

    if log_dict is not None:
        log_dict['critic_loss'].append((critic1_loss.item() + critic2_loss.item()) / 2)
        log_dict['actor_loss'].append(actor_loss.item())
        log_dict['q_mean'].append(q_values.mean().item())
        log_dict['q_max'].append(q_values.max().item())
        log_dict['policy_entropy'].append(-(probs * log_probs).sum(dim=1).mean().item())

# SAC TRAINING FUNCTION 
def train_sac(
    run_id:       int   = 1,
    n_episodes:   int   = 5_000,
    buffer_size:  int   = 100_000,
    batch_size:   int   = 256,
    min_buffer:   int   = 1_000,
    lr_actor:     float = 3e-4,
    lr_critic:    float = 3e-4,
    alpha:        float = 0.05,          # entropy coefficient (lowered: less forced exploration)
    tau:          float = 0.005,         # soft target update rate
    hidden:       int   = 256,
    save_dir:     str   = 'models',
    save_weights: bool  = True,
    verbose:      int   = 500,
):
    """
    Train a discrete Soft Actor-Critic agent on the Config B clinical environment.

    Uses double-Q critics to reduce overestimation bias and a fixed entropy
    temperature alpha (no automatic tuning).

    Parameters
    ----------
    run_id       : int   — label used in saved file names (sac_*_run{run_id}.*)
    n_episodes   : int   — total number of training episodes
    buffer_size  : int   — maximum transitions stored in the replay buffer
    batch_size   : int   — number of transitions sampled per gradient step
    min_buffer   : int   — minimum buffer size before gradient updates begin
    lr_actor     : float — Adam learning rate for the actor
    lr_critic    : float — Adam learning rate for both critics
    alpha        : float — entropy temperature (higher = more exploration)
    tau          : float — Polyak averaging rate for soft target updates
    hidden       : int   — number of units in each hidden layer
    save_dir     : str   — directory where weights and returns are saved
    save_weights : bool  — whether to persist network weights to disk
    verbose      : int   — print a progress line every N episodes; 0 = silent

    Returns
    -------
    dict with keys:
        'actor'      : SACActor  — trained actor (eval mode)
        'critic1'    : SACCritic — first trained critic
        'critic2'    : SACCritic — second trained critic
        'returns'    : list[float] — per-episode returns (len = n_episodes)
        'log_dict'   : dict        — per-step losses and diagnostics
        'train_time' : float       — wall-clock training time in seconds
        'run_id'     : int         — the run_id passed in
    """
    env = make_clinical_env()

    # Networks (double-Q: two independent critics)
    actor          = SACActor(OBS_DIM, N_ACTIONS, hidden).to(device)
    critic1        = SACCritic(OBS_DIM, N_ACTIONS, hidden).to(device)
    critic2        = SACCritic(OBS_DIM, N_ACTIONS, hidden).to(device)
    critic1_target = copy.deepcopy(critic1).to(device)
    critic2_target = copy.deepcopy(critic2).to(device)
    for p in list(critic1_target.parameters()) + list(critic2_target.parameters()):
        p.requires_grad = False

    # Optimizers
    actor_optim   = torch.optim.Adam(actor.parameters(),   lr=lr_actor)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr_critic)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr_critic)

    # Buffer (reuses the existing ReplayBuffer)
    buffer = ReplayBuffer(buffer_size)

    # Training state
    all_returns = []
    t0          = time.time()

    log_dict = {
        'critic_loss':    [],
        'actor_loss':     [],
        'q_mean':         [],
        'q_max':          [],
        'policy_entropy': [],
    }
    for ep in range(n_episodes):
        obs, _    = env.reset(seed=np.random.randint(100_000))
        ep_return = 0.0
        done      = False

        while not done:
            # Action selection (sample from current policy)
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _ = actor.sample(obs_t)
                action = int(action.item())

            next_obs, reward, te, tr, _ = env.step(action)
            done = te or tr

            buffer.push(obs, action, reward, next_obs, float(done))
            obs        = next_obs
            ep_return += reward

            # Update networks
            if len(buffer) >= min_buffer:
                update_sac(actor, critic1, critic2, critic1_target, critic2_target,
                           actor_optim, critic1_optim, critic2_optim,
                           buffer, batch_size, alpha, tau, log_dict=log_dict)

        all_returns.append(ep_return)

        if verbose and (ep + 1) % verbose == 0:
            mean_ret = np.mean(all_returns[-verbose:])
            recent_q       = np.mean(log_dict['q_mean'][-verbose:]) if log_dict['q_mean'] else 0
            recent_entropy = np.mean(log_dict['policy_entropy'][-verbose:]) if log_dict['policy_entropy'] else 0
            recent_c_loss  = np.mean(log_dict['critic_loss'][-verbose:]) if log_dict['critic_loss'] else 0
            print(f'[SAC Run {run_id}] Ep {ep+1:5d}/{n_episodes} | '
                f'return: {mean_ret:.4f} | '
                f'Q_mean: {recent_q:+.3f} | '
                f'entropy: {recent_entropy:.3f} | '
                f'c_loss: {recent_c_loss:.4f}')

    env.close()
    train_time = time.time() - t0

    # Save
    if save_weights:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(actor.state_dict(),   f'{save_dir}/sac_actor_run{run_id}.pth')
        torch.save(critic1.state_dict(), f'{save_dir}/sac_critic1_run{run_id}.pth')
        torch.save(critic2.state_dict(), f'{save_dir}/sac_critic2_run{run_id}.pth')
        np.save(f'{save_dir}/sac_returns_run{run_id}.npy', np.array(all_returns))

    actor.eval()
    print(f'\nTraining complete in {train_time:.1f}s')
    print(f'Mean return (last 1000 ep): {np.mean(all_returns[-1000:]):.4f}')

    return {
        'actor':      actor,
        'critic1':    critic1,
        'critic2':    critic2,
        'returns':    all_returns,
        'log_dict':   log_dict,
        'train_time': train_time,
        'run_id':     run_id,
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