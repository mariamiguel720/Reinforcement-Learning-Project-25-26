# FUNCTION CONFIG_B 

# Import necessary libraries
import pandas as pd
import numpy as np
from sympy import gamma
import torch
import torch.nn as nn
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as F
from torch.distributions import Categorical
import itertools
import random
import copy

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


#----------------------------------------------- DQN - Network and Replay Buffer -----------------------------------------------
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
    Train a DQN agent on the Config B clinical environment. Saves model weights, episode returns and TD losses to 
    `save_dir` at the end of training so results survive session restarts.

    Parameters:
        - run_id          : integer label used in saved file names
                      (dqn_run{run_id}.pth, dqn_returns_run{run_id}.npy, …)
        - n_episodes      : total number of training episodes
        - buffer_capacity : maximum transitions stored in the replay buffer
        - batch_size      : number of transitions sampled per gradient step
        - min_buffer      : minimum transitions in buffer before training starts
        - lr              : Adam learning rate
        - grad_clip       : maximum L2 norm for gradient clipping
        - eps_start       : starting epsilon for epsilon-greedy exploration
        - eps_end         : minimum epsilon (exploration floor)
        - eps_decay       : epsilon is multiplied by this value after each episode
        - target_update   : frequency (in episodes) of hard target-network updates
        - save_dir        : directory where weights and arrays are saved
        - verbose         : print a progress line every N episodes; 0 = silent

    Returns:
        - dict with keys:
            'online'       : trained QNetwork (online network, in eval mode)
            'returns'      : list[float]  per-episode returns  (len = n_episodes)
            'losses'       : list[float]  per-step TD losses   (len = gradient steps)
            'total_steps'  : int          total environment steps taken
            'train_time'   : float        wall-clock training time in seconds
            'run_id'       : int          the run_id passed in
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



#-------------------------------------- PPO — ActorCritic Network & RolloutBuffer --------------------------------------

class ActorCritic(nn.Module):
    """
    Shared backbone with two heads:
    - Actor: outputs logits over 25 actions
    - Critic: outputs a scalar state value V(s)

    Tanh activations and orthogonal initialisation are standard
    for PPO — they stabilise the policy gradient updates.
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
        """Sample action and return log_prob and value."""
        logits, value = self.forward(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        logits, value = self.forward(obs)
        dist    = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """
    Stores one batch of on-policy experience.
    Unlike ReplayBuffer, this is cleared after every update.
    """
    def __init__(self):
        self.states, self.actions   = [], []
        self.rewards, self.dones    = [], []
        self.log_probs, self.values = [], []

    def push(self, state, action, reward, done, log_prob, value):
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


# Architecture summary
ppo_model = ActorCritic().to(device)
n_params  = sum(p.numel() for p in ppo_model.parameters())
print('ActorCritic architecture:')
print(ppo_model)
print(f'\nTotal parameters: {n_params:,}')


# GAE COMPUTATION FUNCTION
def compute_gae(rewards, values, dones, lam=0.95):
    """
    Generalised Advantage Estimation (GAE).
    Combines TD(0) and Monte Carlo to balance bias/variance.
    """
    advantages = []
    gae = 0.0
    values_np = [v.item() for v in values] + [0.0]  # bootstrap with 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values_np[t+1] * (1 - dones[t]) - values_np[t]
        gae   = delta + GAMMA * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return advantages



def train_ppo(
    run_id:      int   = 1,
    n_updates:   int   = 100,
    n_steps:     int   = 4096,
    n_epochs:    int   = 10,
    batch_size:  int   = 256,
    lr:          float = 3e-4,
    clip:        float = 0.2,
    gae_lam:     float = 0.95,
    ent_coef:    float = 0.01,
    vf_coef:     float = 0.5,
    max_grad:    float = 0.5,
    save_dir:    str   = 'models',
):
    """
    Train a PPO agent on the Config B clinical environment.
 
    Saves model weights and episode returns to `save_dir` at the end of
    training so results survive session restarts.
 
    Parameters:
        - run_id     : integer label used in saved file names
                    (ppo_run{run_id}.pth, ppo_returns_run{run_id}.npy)
        - n_updates  : total number of rollout/update cycles
        - n_steps    : environment steps collected per rollout
        - n_epochs   : gradient epochs per update (reuse of each rollout)
        - batch_size : mini-batch size within each epoch
        - lr         : Adam learning rate (eps fixed at 1e-5 as standard for PPO)
        - clip       : PPO clipping epsilon
        - gae_lam    : GAE lambda — controls bias/variance trade-off
        - ent_coef   : entropy bonus coefficient (encourages exploration)
        - vf_coef    : value function loss coefficient
        - max_grad   : maximum L2 norm for gradient clipping
        - save_dir   : directory where weights and arrays are saved
 
    Returns:
        - dict with keys:
            'model'       : trained ActorCritic (in eval mode)
            'returns'     : list[float]  per-episode returns
            'train_time'  : float        wall-clock training time in seconds
            'run_id'      : int          the run_id passed in
    """
 
    # model & optimiser 
    model     = ActorCritic().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    buffer    = RolloutBuffer()
 
    # training state
    all_returns = []
    t0          = time.time()
 
    env = make_clinical_env()
    obs, _ = env.reset(seed=SEED)
 
    ep_return          = 0.0
    current_ep_returns = []
 
    # main loop
    for update in range(n_updates):
 
        # Collect rollout
        buffer.clear()
 
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
                obs, _ = env.reset(seed=np.random.randint(100_000))
 
        # Compute advantages (GAE) and normalise
        advantages  = compute_gae(buffer.rewards, buffer.values, buffer.dones, lam=gae_lam)
        advantages  = torch.FloatTensor(advantages).to(device)
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        returns_gae = advantages + torch.FloatTensor(
            [v.item() for v in buffer.values]).to(device)
 
        # Convert buffer to tensors
        states_t  = torch.FloatTensor(np.array(buffer.states)).to(device)
        actions_t = torch.LongTensor(buffer.actions).to(device)
        old_lp_t  = torch.FloatTensor(buffer.log_probs).to(device)
 
        # PPO update — N epochs over the same rollout
        indices = np.arange(n_steps)
 
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
 
            for start in range(0, n_steps, batch_size):
                idx = indices[start:start + batch_size]
 
                log_prob, value, entropy = model.evaluate(states_t[idx], actions_t[idx])
 
                # Clipped surrogate objective
                ratio        = torch.exp(log_prob - old_lp_t[idx])
                surr1        = ratio * advantages[idx]
                surr2        = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[idx]
                actor_loss   = -torch.min(surr1, surr2).mean()
                critic_loss  =  F.mse_loss(value, returns_gae[idx])
                entropy_loss = -entropy.mean()
 
                loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss
 
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimiser.step()
 
        # Progress
        mean_ret           = np.mean(current_ep_returns) if current_ep_returns else 0.0
        current_ep_returns = []
        print(
            f'[Run {run_id}] Update {update+1:3d}/{n_updates} | '
            f'Episodes: {len(all_returns):5d} | '
            f'Mean return: {mean_ret:.4f}'
        )
 
    env.close()
    train_time = time.time() - t0
 
    # save artefacts 
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(),                      f'{save_dir}/ppo_run{run_id}.pth')
    np.save(f'{save_dir}/ppo_returns_run{run_id}.npy',  np.array(all_returns))
 
    # final summary
    model.eval()
    print(f'\nTraining complete in {train_time:.1f}s')
    print(f'Total episodes:             {len(all_returns):,}')
    print(f'Mean return (last 1000 ep): {np.mean(all_returns[-1000:]):.4f}')
    print(f'Saved → {save_dir}/ppo_run{run_id}.*')
 
    return {
        'model':      model,
        'returns':    all_returns,
        'train_time': train_time,
        'run_id':     run_id,
    }
 





# EVALUATION FUNCTION 
N_EPISODES = 1000
clinical_env = make_clinical_env()
records_clinical = rollout(clinical_env, N_EPISODES)
clinical_env.close()

df_clinical = pd.DataFrame([{k: v for k, v in r.items() if k != 'obs_traj'}
                            for r in records_clinical])

clinical_rand_mean = df_clinical['return'].mean()

def evaluate(model, env_fn, n_episodes=1000, seed=SEED, is_ppo=False, is_sac=False, label='Agent'):
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
                elif is_sac:
                    probs, _ = model(obs_t)
                    action   = int(probs.argmax(1).item())
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

def ppo_grid_search(
    param_grid:  dict,
    seeds:       list  = [42, 123, 7],
    n_updates:   int   = 150,
    n_samples:   int   = None,
    save_dir:    str   = 'models/ppo_tuning',
    csv_name:    str   = 'ppo_tuning_results.csv',
):
    """
    Grid/random search for PPO hyperparameters with multiple seeds.

    Parameters:
        - param_grid : dict {param_name: value or [values_to_try]}.
                       Single values are fixed; lists are tuned.
        - seeds      : seeds to replicate each configuration
        - n_updates  : updates per training run
        - n_samples  : if set, randomly samples this many combos
        - save_dir   : where to save partial CSV and model weights
        - csv_name   : results CSV filename

    Returns:
        - pd.DataFrame sorted by survival_mean descending
    """

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, csv_name)

    # Separate tuned (list) from fixed (scalar) params
    tuned = {k: v for k, v in param_grid.items() if isinstance(v, list)}
    fixed = {k: v for k, v in param_grid.items() if not isinstance(v, list)}

    # Generate combinations of tuned params only
    names  = list(tuned.keys())
    values = list(tuned.values())
    combos = list(itertools.product(*values))

    if n_samples is not None and n_samples < len(combos):
        random.seed(42)
        combos = random.sample(combos, n_samples)

    print(f'Tuned params: {names}')
    print(f'Fixed params: {fixed}')
    print(f'Total configurations: {len(combos)} | Seeds: {len(seeds)} | '
          f'Total runs: {len(combos) * len(seeds)}\n')

    results = []

    for i, combo in enumerate(combos):
        config = dict(zip(names, combo))
        print(f'\n=== Config {i+1}/{len(combos)}: {config} ===')

        seed_survivals = []
        seed_returns   = []

        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            r = train_ppo(
                run_id    = f'tune_{i}_s{seed}',
                n_updates = n_updates,
                save_dir  = save_dir,
                **config,
                **fixed,
            )

            eval_returns, *_ = evaluate(
                r['model'], make_clinical_env, is_ppo=True, label=f'tune_{i}'
            )
            survival    = float(np.mean(eval_returns > 0)) * 100
            mean_return = float(np.mean(eval_returns))

            seed_survivals.append(survival)
            seed_returns.append(mean_return)

            print(f'  Seed {seed}: survival={survival:.1f}%, return={mean_return:.4f}')

        results.append({
            **config,
            'survival_mean': np.mean(seed_survivals),
            'survival_std':  np.std(seed_survivals),
            'return_mean':   np.mean(seed_returns),
            'return_std':    np.std(seed_returns),
        })

        pd.DataFrame(results).sort_values('survival_mean', ascending=False) \
            .to_csv(csv_path, index=False)

        print(f'  -> survival = {results[-1]["survival_mean"]:.1f}% '
              f'± {results[-1]["survival_std"]:.1f}%')

    print(f'\nDone. Results saved to {csv_path}')
    return pd.DataFrame(results).sort_values('survival_mean', ascending=False)


# SAC

# SAC NETWORKS

class SACActor(nn.Module):
    """Policy network for discrete SAC. Outputs action probabilities."""
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        logits = self.net(obs)
        probs  = F.softmax(logits, dim=-1)
        # Numerical stability: log(probs) directly via log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample(self, obs):
        """Sample an action from the policy."""
        probs, log_probs = self.forward(obs)
        dist   = Categorical(probs=probs)
        action = dist.sample()
        return action, probs, log_probs


class SACCritic(nn.Module):
    """Q-network. Outputs Q-values for all discrete actions."""
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


# SAC UPDATE FUNCTION 
def update_sac(actor, critic, critic_target,
               actor_optim, critic_optim,
               buffer, batch_size, alpha, gamma, tau,
               log_dict=None):
    """Single gradient step for SAC."""
    if len(buffer) < batch_size:
        return
    
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states_t      = torch.FloatTensor(states).to(device)
    actions_t     = torch.LongTensor(actions).to(device)
    rewards_t     = torch.FloatTensor(rewards).to(device)
    next_states_t = torch.FloatTensor(next_states).to(device)
    dones_t       = torch.FloatTensor(dones).to(device)

    # CRITIC UPDATE
    with torch.no_grad():
        next_probs, next_log_probs = actor(next_states_t)
        next_q   = critic_target(next_states_t)
        next_v   = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=1)
        target_q = rewards_t + gamma * (1 - dones_t) * next_v

    current_q   = critic(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    critic_loss = F.mse_loss(current_q, target_q)

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # ACTOR UPDATE
    probs, log_probs = actor(states_t)
    with torch.no_grad():
        q_values = critic(states_t)
    actor_loss = (probs * (alpha * log_probs - q_values)).sum(dim=1).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # SOFT TARGET UPDATE
    with torch.no_grad():
        for p, p_target in zip(critic.parameters(), critic_target.parameters()):
            p_target.data.mul_(1 - tau)
            p_target.data.add_(tau * p.data)
    
    # LOGGING (novo)
    if log_dict is not None:
        log_dict['critic_loss'].append(critic_loss.item())
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
    alpha:        float = 0.2,           # entropy coefficient (fixed)
    tau:          float = 0.005,         # soft target update rate
    gamma:        float = GAMMA,
    hidden:       int   = 256,
    save_dir:     str   = 'models',
    save_weights: bool  = True,
    verbose:      int   = 500,
):
    """
    Discrete Soft Actor-Critic (Phase 1: single Q-network, fixed alpha).

    Returns dict with keys: 'actor', 'critic', 'returns', 'train_time', 'run_id'.
    """
    env = make_clinical_env()

    # Networks
    actor          = SACActor(OBS_DIM, N_ACTIONS, hidden).to(device)
    critic         = SACCritic(OBS_DIM, N_ACTIONS, hidden).to(device)
    critic_target  = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad = False

    # Optimizers
    actor_optim  = torch.optim.Adam(actor.parameters(),  lr=lr_actor)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr_critic)

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
                update_sac(actor, critic, critic_target,
                           actor_optim, critic_optim,
                           buffer, batch_size, alpha, gamma, tau, log_dict=log_dict)

        all_returns.append(ep_return)

        if verbose and (ep + 1) % verbose == 0:
            mean_ret = np.mean(all_returns[-verbose:])
            recent_q       = np.mean(log_dict['q_mean'][-1000:]) if log_dict['q_mean'] else 0
            recent_entropy = np.mean(log_dict['policy_entropy'][-1000:]) if log_dict['policy_entropy'] else 0
            recent_c_loss  = np.mean(log_dict['critic_loss'][-1000:]) if log_dict['critic_loss'] else 0
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
        torch.save(actor.state_dict(),  f'{save_dir}/sac_actor_run{run_id}.pth')
        torch.save(critic.state_dict(), f'{save_dir}/sac_critic_run{run_id}.pth')
        np.save(f'{save_dir}/sac_returns_run{run_id}.npy', np.array(all_returns))

    actor.eval()
    print(f'\nTraining complete in {train_time:.1f}s')
    print(f'Mean return (last 1000 ep): {np.mean(all_returns[-1000:]):.4f}')

    return {
        'actor':      actor,
        'critic':     critic,
        'returns':    all_returns,
        'log_dict':   log_dict,
        'train_time': train_time,
        'run_id':     run_id,
    }