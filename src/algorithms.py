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
import itertools
import random
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
    reports the mean survival rate at each eval checkpoint to Optuna and
    raises `TrialPruned` if the pruner decides to cut the trial early.

    Returns a dict with:
        'model'       : trained ActorCritic (eval mode)
        'best_eval'   : best mean return seen during eval checkpoints
        'best_state'  : state_dict of the best checkpoint (for final retrain)
        'eval_history': list of (update_idx, mean_return) tuples
        'returns'     : all per-episode returns during training
        'duration_s'  : wall-clock seconds
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

        # --- Collect rollout ---
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

        # --- Compute advantages (GAE) ---
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

        # --- PPO update (n_epochs over same rollout) ---
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

        # --- Periodic evaluation checkpoint ---
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



# ------------------------------------- A2C - Shared Actor-Critic Network -----------------------------------------------------

# A2C CONFIGURATION:
class A2CNetwork(nn.Module):
    """Shared-backbone Actor-Critic network for discrete actions.

    Architecture (mirrors the PPO network already in the notebook):
        Input  : (batch, 47)
        Shared : Linear(47→256, Tanh) → Linear(256→256, Tanh)
        Actor  : Linear(256→25)  → log-softmax
        Critic : Linear(256→1)
    """

    def __init__(self, obs_dim: int = 47, n_actions: int = 25,
                 hidden: int = 256):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor head — outputs log-probabilities
        self.actor_head  = nn.Linear(hidden, n_actions)

        # Critic head — outputs scalar state-value V(s)
        self.critic_head = nn.Linear(hidden, 1)

        # Orthogonal initialisation (standard for policy-gradient methods)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor):
        """Returns (log_probs, value).

        Args:
            obs: float tensor of shape (batch, 47)
        Returns:
            log_probs : (batch, 25)  — log π(a|s) for all actions
            value     : (batch,)     — V(s)
        """
        features  = self.backbone(obs)
        log_probs = torch.log_softmax(self.actor_head(features), dim=-1)
        value     = self.critic_head(features).squeeze(-1)
        return log_probs, value

    def act(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, value).

        Used during rollout collection (single observation, no grad).
        """
        log_probs, value = self.forward(obs)
        dist   = Categorical(logits=log_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Re-evaluate (log_prob, entropy, value) for a batch of transitions.

        Used during the gradient update step.
        """
        log_probs, value = self.forward(obs)
        dist      = Categorical(logits=log_probs)
        log_prob  = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_prob, entropy, value


# A2C TRAINING FUNCTION
def train_a2c(
    run_id:      int   = 1,
    n_updates:   int   = 150,      # total number of gradient updates
    n_steps:     int   = 2048,     # rollout length (steps collected per update)
    lr:          float = 3e-4,     # learning rate (actor + critic, shared optimiser)
    ent_coef:    float = 0.01,     # entropy bonus coefficient β
    vf_coef:     float = 0.5,      # critic loss weight
    max_grad:    float = 0.5,      # gradient clipping norm
    log_every:   int   = 10,       # print progress every N updates
    save_dir:    str   = 'models', # directory where weights and arrays are saved
) -> dict:
    """Train an A2C agent on the clinical ICU-Sepsis environment.

    Returns a dict with keys:
        'model'   : trained A2CNetwork
        'returns' : list of per-episode returns collected during training
    """

    print('=' * 52)
    print(f'  A2C — Run {run_id}')
    print('=' * 52)
    print(f'  n_updates : {n_updates}')
    print(f'  n_steps   : {n_steps}  (steps per rollout)')
    print(f'  lr        : {lr}')
    print(f'  ent_coef  : {ent_coef}')
    print(f'  vf_coef   : {vf_coef}')
    print(f'  max_grad  : {max_grad}')
    print('=' * 52)

    torch.manual_seed(SEED + run_id)
    np.random.seed(SEED + run_id)

    # Model, optimiser & environment
    model = A2CNetwork().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    env   = make_clinical_env()

    all_returns   = []       # one entry per completed episode
    recent        = deque(maxlen=200)
    total_steps   = 0

    obs, _ = env.reset(seed=SEED + run_id)

    for update in range(1, n_updates + 1):

        # Collect a rollout of n_steps transitions
        obs_buf      = []
        act_buf      = []
        logp_buf     = []
        val_buf      = []
        rew_buf      = []
        done_buf     = []

        ep_return    = 0.0
        ep_steps     = 0

        for _ in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = model.act(obs_t)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob.item())
            val_buf.append(value.item())
            rew_buf.append(reward)
            done_buf.append(done)

            obs          = next_obs
            ep_return   += reward
            ep_steps    += 1
            total_steps += 1

            if done:
                all_returns.append(ep_return)
                recent.append(ep_return)
                ep_return = 0.0
                ep_steps  = 0
                obs, _    = env.reset()

        # Compute bootstrapped returns & advantages
        # Bootstrap from V(s_last) if the rollout ended mid-episode
        with torch.no_grad():
            last_obs_t = torch.tensor(obs, dtype=torch.float32,
                                      device=device).unsqueeze(0)
            _, last_val = model.forward(last_obs_t)
            last_val    = last_val.item()

        returns = []
        R = last_val if not done_buf[-1] else 0.0
        for r, d in zip(reversed(rew_buf), reversed(done_buf)):
            R = r + GAMMA * R * (1.0 - d)
            returns.insert(0, R)

        # Gradient update
        obs_t  = torch.tensor(np.array(obs_buf),  dtype=torch.float32, device=device)
        act_t  = torch.tensor(act_buf,            dtype=torch.long,    device=device)
        ret_t  = torch.tensor(returns,            dtype=torch.float32, device=device)

        log_prob_t, entropy_t, val_t = model.evaluate_actions(obs_t, act_t)

        advantage = ret_t - val_t.detach()           # one-step TD advantage

        # Normalise advantage (reduces variance, stabilises training)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss   = -(log_prob_t * advantage).mean()
        critic_loss  =  vf_coef  * (ret_t - val_t).pow(2).mean()
        entropy_loss = -ent_coef * entropy_t.mean()

        loss = actor_loss + critic_loss + entropy_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        opt.step()

        # Logging
        if update % log_every == 0 and recent:
            mean_ret  = float(np.mean(recent))
            surv_rate = float(np.mean([r > 0 for r in recent])) * 100
            ep_done   = len(all_returns)
            print(f'  update {update:4d}/{n_updates} | '
                  f'ep {ep_done:5d} | '
                  f'mean return {mean_ret:.4f} | '
                  f'survival {surv_rate:.1f}% | '
                  f'loss {loss.item():.4f}')

    env.close()
    print('─' * 52)
    print(f'  Training complete. Total episodes: {len(all_returns)}')
    print('─' * 52)

    # Save weights and returns
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/a2c_run{run_id}.pth')
    np.save(f'{save_dir}/a2c_returns_run{run_id}.npy', np.array(all_returns))
    print(f'  Saved → {save_dir}/a2c_run{run_id}.*')

    return {'model': model, 'returns': all_returns}




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
def update_sac(actor, critic1, critic2, critic1_target, critic2_target,
               actor_optim, critic1_optim, critic2_optim,
               buffer, batch_size, alpha, tau,
               log_dict=None):
    """Single gradient step for SAC with double-Q (reduces overestimation bias)."""
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
    Discrete Soft Actor-Critic with double-Q and lower alpha.

    Returns dict with keys: 'actor', 'critic1', 'critic2', 'returns', 'train_time', 'run_id'.
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