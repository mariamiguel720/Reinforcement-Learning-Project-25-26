"""
functions_ConfigA.py
====================
All shared functions for Configuration A: tabular RL on ICU-Sepsis-v2.

Usage in notebook
-----------------
    import functions_ConfigA as fa

    # After Policy Iteration:
    fa.configure(S, A, SEED, GAMMA, pi_V, P_pi, R_shaped)

    # Training:
    Q, returns = fa.train_dynaq_ps(**params, n_episodes=60_000)

Contents
--------
  Colour palette
  make_env()              - silent environment factory
  configure()             - inject MDP globals after PI
  smooth(), silent_eval()
  evaluate_policy()
  make_optuna_callback()
  plot_results()          - model-free algorithms
  plot_pi_results()       - PI-specific (no Optuna panel)
  train_dynaq_ps()
  train_expected_sarsa()
  train_qlearning()
  train_sarsa()
  train_double_ql()
  train_mc()
  train_warmstart()
"""

import numpy as np
import matplotlib.pyplot as plt
import contextlib, io, os, time, heapq

# ── Colour palette (blue tones only, aligned with Config B) ──────────────────
H_COL = '#2f94d7'
B0 = '#0a1f35'; B1 = '#1a3a5c'; B2 = '#1f6fb2'; B3 = '#4a9fd4'
B4 = '#7bbee0'; B5 = '#a8c4e0'; B6 = '#d4e8f5'
REF_RAND = '#6baed6'; REF_PI = '#08306b'

C_PI  = B0; C_DQ = B2; C_ES = B1; C_QL = B3
C_SAR = B4; C_DQL = H_COL; C_MC = B5; C_WS = B6

SEED      = 42
PLOTS_DIR = 'plots'

# ── MDP context (set via configure() after Policy Iteration) ──────────────────
_ctx: dict = {}


def configure(S, A, seed, gamma, pi_V, P_pi, R_shaped):
    """
    Inject MDP globals. Call immediately after Policy Iteration.

    Parameters
    ----------
    S, A       : int
    seed       : int  (typically 42)
    gamma      : float  (1.0 for ICU-Sepsis)
    pi_V       : np.ndarray shape (S,)
    P_pi       : np.ndarray shape (S,A,S)
    R_shaped   : np.ndarray shape (S,A)
    """
    global _ctx
    _ctx = dict(S=int(S), A=int(A), seed=int(seed),
                gamma=float(gamma), pi_V=np.asarray(pi_V),
                P_pi=np.asarray(P_pi), R_shaped=np.asarray(R_shaped))
    print(f'  fa.configure: S={S}, A={A}, seed={seed}, gamma={gamma}')
    print(f'  pi_V range: [{pi_V.min():.4f}, {pi_V.max():.4f}]')


def _require_ctx():
    if not _ctx:
        raise RuntimeError(
            "fa.configure() has not been called. "
            "Call it after Policy Iteration with pi_V, P_pi, R_shaped."
        )


# ── Environment factory ───────────────────────────────────────────────────────

def make_env():
    """Create ICU-Sepsis environment without any print output."""
    from envs.env_setup import make_sepsis_env
    with contextlib.redirect_stdout(io.StringIO()):
        return make_sepsis_env()


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth(arr, w):
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def silent_eval(policy, n_eval=300):
    """Evaluate policy silently. Returns survival rate (%). Used in Optuna."""
    _require_ctx()
    np.random.seed(_ctx['seed'])
    env = make_env(); survived = 0
    for _ in range(n_eval):
        obs, _ = env.reset(seed=np.random.randint(100_000)); done = False
        while not done:
            obs, r, te, tr, _ = env.step(int(policy[int(obs)])); done = te or tr
        if r > 0: survived += 1
    env.close()
    return survived / n_eval * 100


def evaluate_policy(policy, n_episodes=1000, label='Policy'):
    """Evaluate policy. Prints progress every 200 episodes. No animation."""
    np.random.seed(_ctx.get('seed', SEED))
    env = make_env(); returns, lengths = [], []; survived = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        total_r, steps, done = 0.0, 0, False
        while not done:
            obs, r, te, tr, _ = env.step(int(policy[int(obs)]))
            total_r += r; steps += 1; done = te or tr
        returns.append(total_r); lengths.append(steps)
        if total_r > 0: survived += 1
        if (ep + 1) % 200 == 0:
            print(f'  [{label}] Ep {ep+1}/{n_episodes} | '
                  f'Survival: {survived/(ep+1)*100:.1f}%')
    env.close()
    print(f'  [{label}] Result: {survived/n_episodes*100:.1f}% ({n_episodes} episodes)')
    return np.array(returns), np.array(lengths)


def make_optuna_callback(algo_name, n_trials=30):
    """One-line Optuna callback per trial. No animation."""
    def cb(study, trial):
        print(f'  [{algo_name}] Trial {trial.number+1:>2}/{n_trials} | '
              f'Value: {trial.value:.1f}% | Best: {study.best_value:.1f}%')
    return cb


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(study, train_returns, label, color,
                 final_surv, pi_surv, rand_surv,
                 pi_mean_ret, rand_mean_ret, plot_name):
    """
    3-panel figure for model-free algorithms.
      Panel 1: Learning curve  (y = return, rolling avg)
      Panel 2: Optuna optimisation history
      Panel 3: Final survival rate bar chart
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sw       = min(1000, max(1, len(train_returns) // 10))
    smoothed = smooth(np.array(train_returns), sw)
    eps_ax   = np.arange(sw, len(train_returns) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(label, fontsize=13, fontweight='bold', color=H_COL)

    axes[0].plot(eps_ax, smoothed, color=color, lw=2.0, label=label)
    axes[0].axhline(pi_mean_ret, color=REF_PI, lw=1.5, ls=':', label=f'PI {pi_mean_ret:.3f}')
    axes[0].axhline(rand_mean_ret, color=REF_RAND, lw=1.5, ls='--', label=f'Random {rand_mean_ret:.3f}')
    axes[0].fill_between(eps_ax, smoothed, rand_mean_ret,
                          where=(smoothed >= rand_mean_ret), alpha=0.12, color=color)
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Return (rolling avg)')
    axes[0].set_title('Learning Curve', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].spines[['top', 'right']].set_visible(False)

    if study is not None:
        vals = [t.value for t in study.trials if t.value is not None]
        best = np.maximum.accumulate(vals)
        axes[1].scatter(range(1, len(vals)+1), vals, color=color, alpha=0.35, s=20)
        axes[1].plot(range(1, len(best)+1), best, color=color, lw=2, label=f'Best: {max(best):.1f}%')
        axes[1].set_xlabel('Optuna Trial'); axes[1].set_ylabel('Survival Rate (%)')
        axes[1].set_title('Optuna Optimisation History', fontweight='bold')
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=12, color='grey')
    axes[1].spines[['top', 'right']].set_visible(False)

    bars = axes[2].bar([label, 'PI', 'Random'], [final_surv, pi_surv, rand_surv],
                       color=[color, REF_PI, REF_RAND], edgecolor='white', alpha=0.9)
    axes[2].set_ylabel('Survival Rate (%)'); axes[2].set_ylim(0, 100)
    axes[2].set_title('Final Survival Rate', fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)
    for bar, v in zip(bars, [final_surv, pi_surv, rand_surv]):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{plot_name}.png', dpi=150, bbox_inches='tight'); plt.show()

    ci = 1.96 * np.sqrt((final_surv/100)*(1-final_surv/100)/1000)*100
    print(f'{"="*55}')
    print(f'  {label}')
    print(f'{"="*55}')
    print(f'  Survival Rate : {final_surv:.1f}% +/- {ci:.1f}%')
    print(f'  vs PI         : {final_surv - pi_surv:+.1f} pp')
    print(f'  vs Random     : {final_surv - rand_surv:+.1f} pp')
    if study is not None:
        print(f'  Best Optuna   : {study.best_value:.1f}%')
        print(f'  Best params   : {study.best_params}')
    print(f'{"="*55}')


def plot_pi_results(pi_V, eval_returns, pi_surv, rand_surv, S_n, plot_name):
    """
    PI-specific 3-panel figure (replaces N/A Optuna panel with V_PI distribution).
      Panel 1: V_PI landscape (sorted survival probabilities per state)
      Panel 2: V_PI distribution histogram
      Panel 3: Final survival rate vs random baseline
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sv = np.sort(pi_V[:S_n])
    v_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)
    colors = [plt.cm.Blues(0.25 + 0.75 * v) for v in v_norm]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle('Policy Iteration', fontsize=13, fontweight='bold', color=H_COL)

    axes[0].bar(range(len(sv)), sv, color=colors, width=1.0)
    axes[0].axhline(0.5, color=B0, lw=1.0, ls=':', alpha=0.6, label='V = 0.5')
    axes[0].set_xlabel('States (sorted by V_PI)')
    axes[0].set_ylabel('V_PI(s)  =  P(survive | optimal policy)')
    axes[0].set_title('Value Function Landscape', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].hist(pi_V[:S_n], bins=30, color=B2, edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(pi_V[:S_n]), color=B0, lw=1.8, ls='--',
                    label=f'Mean = {np.mean(pi_V[:S_n]):.3f}')
    axes[1].set_xlabel('V_PI(s)'); axes[1].set_ylabel('Number of States')
    axes[1].set_title('V_PI Distribution', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].spines[['top', 'right']].set_visible(False)

    bars = axes[2].bar(['Policy Iteration', 'Random'], [pi_surv, rand_surv],
                       color=[REF_PI, REF_RAND], edgecolor='white', alpha=0.9)
    axes[2].set_ylabel('Survival Rate (%)'); axes[2].set_ylim(0, 100)
    axes[2].set_title('Final Survival Rate', fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)
    for bar, v in zip(bars, [pi_surv, rand_surv]):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{plot_name}.png', dpi=150, bbox_inches='tight'); plt.show()

    ci = 1.96 * np.sqrt((pi_surv/100)*(1-pi_surv/100)/1000)*100
    print(f'{"="*55}')
    print(f'  Policy Iteration')
    print(f'{"="*55}')
    print(f'  Survival Rate : {pi_surv:.1f}% +/- {ci:.1f}%')
    print(f'  Mean Return   : {np.mean(eval_returns):.4f}')
    print(f'  Mean V_PI     : {np.mean(pi_V[:S_n]):.4f}')
    print(f'  vs Random     : {pi_surv - rand_surv:+.1f} pp')
    print(f'{"="*55}')


# ── Training functions ────────────────────────────────────────────────────────

def _progress(episode, n_episodes, surv_win, a_t, epsilon, label, t0):
    print(f'  [{label}] Ep {episode+1:>6}/{n_episodes} | '
          f'alpha: {a_t:.5f} | eps: {epsilon:.5f} | '
          f'Surv(500): {np.mean(surv_win)*100:.1f}% | {time.time()-t0:.0f}s')


def train_dynaq_ps(alpha_start, alpha_end, alpha_decay,
                   k, ps_theta, eps_decay, n_episodes, verbose=True):
    """Dyna-Q with Prioritized Sweeping and Reward Shaping."""
    _require_ctx()
    S = _ctx['S']; A = _ctx['A']; seed = _ctx['seed']
    gamma = _ctx['gamma']; pi_V = _ctx['pi_V']
    P_pi = _ctx['P_pi']; R_shaped = _ctx['R_shaped']
    np.random.seed(seed)
    Q = np.zeros((S, A), dtype=np.float64); pq = []; epsilon = 1.0
    returns = []; surv_win = []; clinical = np.arange(S - 2)
    env = make_env(); t0 = time.time()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        state = int(obs); a_t = max(alpha_end, alpha_start*(alpha_decay**episode))
        total_r, done = 0.0, False
        while not done:
            action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[state])))
            next_obs, reward, te, tr, _ = env.step(action)
            next_state = int(next_obs); done = te or tr
            sr = reward + gamma*pi_V[next_state]*(not done) - pi_V[state]
            td = sr + gamma*np.max(Q[next_state])*(not done) - Q[state, action]
            Q[state, action] += a_t*td; total_r += reward
            if abs(td) > ps_theta and len(pq) < 100_000:
                heapq.heappush(pq, (-abs(td), int(state), int(action)))
            n_p = 0
            while n_p < k and pq:
                _, sp, ap = heapq.heappop(pq)
                if sp >= S-2: continue
                sn = int(np.random.choice(S, p=P_pi[sp, ap]))
                Q[sp, ap] += a_t*(R_shaped[sp, ap]+gamma*np.max(Q[sn])-Q[sp, ap]); n_p += 1
            while n_p < k:
                ss = int(np.random.choice(clinical)); aa = int(np.random.randint(A))
                sn = int(np.random.choice(S, p=P_pi[ss, aa]))
                Q[ss, aa] += a_t*(R_shaped[ss, aa]+gamma*np.max(Q[sn])-Q[ss, aa]); n_p += 1
            state = next_state
        epsilon = max(0.005, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0: _progress(episode, n_episodes, surv_win, a_t, epsilon, 'Dyna-Q+PS', t0)
    env.close()
    if verbose: print(f'  [Dyna-Q+PS] Done: {time.time()-t0:.0f}s | PQ: {len(pq)}')
    return Q, returns


def train_expected_sarsa(alpha_start, alpha_end, alpha_decay,
                          lam, eps_decay, n_episodes, verbose=True):
    """Expected SARSA with Eligibility Traces and Reward Shaping."""
    _require_ctx()
    S = _ctx['S']; A = _ctx['A']; seed = _ctx['seed']
    gamma = _ctx['gamma']; pi_V = _ctx['pi_V']
    np.random.seed(seed)
    Q = np.zeros((S, A), dtype=np.float64); e = np.zeros((S, A), dtype=np.float64)
    epsilon = 1.0; inv_A = 1.0/A; returns = []; surv_win = []
    env = make_env(); t0 = time.time()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        state = int(obs); a_t = max(alpha_end, alpha_start*(alpha_decay**episode))
        e.fill(0.0); total_r, done = 0.0, False
        action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[state])))
        while not done:
            next_obs, reward, te, tr, _ = env.step(action)
            next_state = int(next_obs); done = te or tr
            exp_q = (0.0 if done else (1-epsilon)*np.max(Q[next_state])+epsilon*inv_A*np.sum(Q[next_state]))
            sr = reward+gamma*pi_V[next_state]*(not done)-pi_V[state]
            delta = sr+gamma*exp_q-Q[state, action]
            e[state, action] = 1.0; Q += a_t*delta*e; e *= gamma*lam
            total_r += reward; state = next_state
            if not done:
                action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[state])))
        epsilon = max(0.005, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0: _progress(episode, n_episodes, surv_win, a_t, epsilon, 'ExpSARSA', t0)
    env.close()
    if verbose: print(f'  [ExpSARSA] Done: {time.time()-t0:.0f}s')
    return Q, returns


def train_qlearning(alpha_start, alpha_end, alpha_decay,
                     eps_decay, n_episodes, verbose=True):
    """Q-Learning with Learning Rate Decay and Reward Shaping."""
    _require_ctx()
    S = _ctx['S']; A = _ctx['A']; seed = _ctx['seed']
    gamma = _ctx['gamma']; pi_V = _ctx['pi_V']
    np.random.seed(seed)
    Q = np.zeros((S, A), dtype=np.float64); epsilon = 1.0; returns = []; surv_win = []
    env = make_env(); t0 = time.time()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        state = int(obs); a_t = max(alpha_end, alpha_start*(alpha_decay**episode))
        total_r, done = 0.0, False
        while not done:
            action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[state])))
            next_obs, reward, te, tr, _ = env.step(action)
            next_state = int(next_obs); done = te or tr
            sr = reward+gamma*pi_V[next_state]*(not done)-pi_V[state]
            Q[state, action] += a_t*(sr+gamma*np.max(Q[next_state])*(not done)-Q[state, action])
            total_r += reward; state = next_state
        epsilon = max(0.005, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0: _progress(episode, n_episodes, surv_win, a_t, epsilon, 'Q-Learning', t0)
    env.close()
    if verbose: print(f'  [Q-Learning] Done: {time.time()-t0:.0f}s')
    return Q, returns


def train_sarsa(alpha_start, alpha_end, alpha_decay,
                 lam, eps_decay, n_episodes, verbose=True):
    """SARSA with Eligibility Traces and Reward Shaping."""
    _require_ctx()
    S = _ctx['S']; A = _ctx['A']; seed = _ctx['seed']
    gamma = _ctx['gamma']; pi_V = _ctx['pi_V']
    np.random.seed(seed)
    Q = np.zeros((S, A), dtype=np.float64); e = np.zeros((S, A), dtype=np.float64)
    epsilon = 1.0; returns = []; surv_win = []
    env = make_env(); t0 = time.time()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        state = int(obs); a_t = max(alpha_end, alpha_start*(alpha_decay**episode))
        e.fill(0.0); total_r, done = 0.0, False
        action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[state])))
        while not done:
            next_obs, reward, te, tr, _ = env.step(action)
            next_state = int(next_obs); done = te or tr
            na = (0 if done else (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[next_state]))))
            sr = reward+gamma*pi_V[next_state]*(not done)-pi_V[state]
            delta = sr+gamma*Q[next_state, na]*(not done)-Q[state, action]
            e[state, action] = 1.0; Q += a_t*delta*e; e *= gamma*lam
            total_r += reward; state = next_state; action = na
        epsilon = max(0.005, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0: _progress(episode, n_episodes, surv_win, a_t, epsilon, 'SARSA', t0)
    env.close()
    if verbose: print(f'  [SARSA] Done: {time.time()-t0:.0f}s')
    return Q, returns


def train_double_ql(alpha_start, alpha_end, alpha_decay,
                     eps_decay, n_episodes, verbose=True):
    """Double Q-Learning with Reward Shaping."""
    _require_ctx()
    S = _ctx['S']; A = _ctx['A']; seed = _ctx['seed']
    gamma = _ctx['gamma']; pi_V = _ctx['pi_V']
    np.random.seed(seed)
    QA = np.zeros((S, A), dtype=np.float64); QB = np.zeros((S, A), dtype=np.float64)
    epsilon = 1.0; returns = []; surv_win = []
    env = make_env(); t0 = time.time()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        state = int(obs); a_t = max(alpha_end, alpha_start*(alpha_decay**episode))
        total_r, done = 0.0, False
        while not done:
            action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(QA[state]+QB[state])))
            next_obs, reward, te, tr, _ = env.step(action)
            next_state = int(next_obs); done = te or tr
            sr = reward+gamma*pi_V[next_state]*(not done)-pi_V[state]
            if np.random.rand()<0.5:
                a_s = int(np.argmax(QA[next_state]))
                QA[state, action] += a_t*(sr+gamma*QB[next_state, a_s]*(not done)-QA[state, action])
            else:
                b_s = int(np.argmax(QB[next_state]))
                QB[state, action] += a_t*(sr+gamma*QA[next_state, b_s]*(not done)-QB[state, action])
            total_r += reward; state = next_state
        epsilon = max(0.005, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0: _progress(episode, n_episodes, surv_win, a_t, epsilon, 'Double-QL', t0)
    env.close()
    if verbose: print(f'  [Double Q-Learning] Done: {time.time()-t0:.0f}s')
    return (QA, QB), returns


def train_mc(eps_decay, eps_end, n_episodes, verbose=True):
    """Monte Carlo Control with Reward Shaping (first-visit)."""
    _require_ctx()
    S = _ctx['S']; A = _ctx['A']; seed = _ctx['seed']
    gamma = _ctx['gamma']; pi_V = _ctx['pi_V']
    np.random.seed(seed)
    Q = np.zeros((S, A), dtype=np.float64)
    rs = np.zeros((S, A), dtype=np.float64); rc = np.zeros((S, A), dtype=np.float64)
    epsilon = 1.0; returns = []; surv_win = []
    env = make_env(); t0 = time.time()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        state = int(obs); ts, ta, tr = [], [], []; total_r, done = 0.0, False
        while not done:
            action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[state])))
            next_obs, reward, te, tr_, _ = env.step(action)
            next_state = int(next_obs); done = te or tr_
            sr = reward+gamma*pi_V[next_state]*(not done)-pi_V[state]
            ts.append(state); ta.append(action); tr.append(sr)
            total_r += reward; state = next_state
        G = 0.0; visited = set()
        for t in range(len(ts)-1, -1, -1):
            G = gamma*G+tr[t]; sa = (ts[t], ta[t])
            if sa not in visited:
                visited.add(sa); rs[sa] += G; rc[sa] += 1; Q[sa] = rs[sa]/rc[sa]
        epsilon = max(eps_end, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0:
                print(f'  [MC] Ep {episode+1:>6}/{n_episodes} | eps: {epsilon:.5f} | '
                      f'Surv(500): {np.mean(surv_win)*100:.1f}% | {time.time()-t0:.0f}s')
    env.close()
    if verbose: print(f'  [MC] Done: {time.time()-t0:.0f}s')
    return Q, returns, rc


def train_warmstart(alpha_start, alpha_end, alpha_decay,
                     eps_start, eps_decay, n_episodes, verbose=True):
    """V_PI Warm-Start Q-Learning with Reward Shaping."""
    _require_ctx()
    S = _ctx['S']; A = _ctx['A']; seed = _ctx['seed']
    gamma = _ctx['gamma']; pi_V = _ctx['pi_V']
    np.random.seed(seed)
    Q = np.tile(pi_V[:, np.newaxis], (1, A)).astype(np.float64)
    epsilon = eps_start; returns = []; surv_win = []
    env = make_env(); t0 = time.time()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100_000))
        state = int(obs); a_t = max(alpha_end, alpha_start*(alpha_decay**episode))
        total_r, done = 0.0, False
        while not done:
            action = (np.random.randint(A) if np.random.rand()<epsilon else int(np.argmax(Q[state])))
            next_obs, reward, te, tr, _ = env.step(action)
            next_state = int(next_obs); done = te or tr
            sr = reward+gamma*pi_V[next_state]*(not done)-pi_V[state]
            Q[state, action] += a_t*(sr+gamma*np.max(Q[next_state])*(not done)-Q[state, action])
            total_r += reward; state = next_state
        epsilon = max(0.005, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0: _progress(episode, n_episodes, surv_win, a_t, epsilon, 'WarmStart', t0)
    env.close()
    if verbose: print(f'  [WarmStart] Done: {time.time()-t0:.0f}s')
    return Q, returns
