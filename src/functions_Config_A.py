"""
functions_ConfigA.py
All shared functions for Configuration A: tabular RL on ICU-Sepsis-v2.
"""

import numpy as np
import matplotlib.pyplot as plt
import contextlib, io, os, time, heapq


# ── Constants and colour palette ──────────────────────────────────────────────
SEED      = 42
PLOTS_DIR = 'plots'

RETURN_YLIM = (0.55, 0.92)
SURV_YLIM   = (60, 95)

H_COL = '#2f94d7'
B0 = '#0a1f35'; B1 = '#1a3a5c'; B2 = '#1f6fb2'; B3 = '#4a9fd4'
B4 = '#7bbee0'; B5 = '#a8c4e0'; B6 = '#d4e8f5'

REF_RAND = '#6baed6'
REF_PI   = '#08306b'
C_DQ     = B2
C_QL     = B3


# ── MDP context (set via configure() after Policy Iteration) ──────────────────
_ctx: dict = {}


def configure(S, A, seed, gamma, pi_V, P_pi, R_shaped):
    """Inject MDP globals after Policy Iteration."""
    global _ctx
    _ctx = dict(S=int(S), A=int(A), seed=int(seed),
                gamma=float(gamma), pi_V=np.asarray(pi_V),
                P_pi=np.asarray(P_pi), R_shaped=np.asarray(R_shaped))
    print(f'  fa.configure: S={S}, A={A}, seed={seed}, gamma={gamma}')
    print(f'  pi_V range: [{pi_V.min():.4f}, {pi_V.max():.4f}]')


def _require_ctx():
    if not _ctx:
        raise RuntimeError("fa.configure() has not been called.")


# ── Environment factory ───────────────────────────────────────────────────────

def make_env():
    """Create ICU-Sepsis environment without print output."""
    from envs.env_setup import make_sepsis_env
    with contextlib.redirect_stdout(io.StringIO()):
        return make_sepsis_env()


# ── Utilities ─────────────────────────────────────────────────────────────────

def smooth(arr, w):
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def silent_eval(policy, n_eval=300):
    """ Returns survival rate (%)."""
    _require_ctx()
    np.random.seed(_ctx['seed'])
    env = make_env(); survived = 0
    for _ in range(n_eval):
        obs, _ = env.reset(seed=np.random.randint(100_000)); done = False
        while not done:
            obs, r, te, tr, _ = env.step(int(policy[int(obs)]))
            done = te or tr
        if r > 0: survived += 1
    env.close()
    return survived / n_eval * 100


def evaluate_policy(policy, n_episodes=1000, label='Policy'):
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
    """One-line Optuna callback per trial."""
    def cb(study, trial):
        print(f'  [{algo_name}] Trial {trial.number+1:>2}/{n_trials} | '
              f'Value: {trial.value:.1f}% | Best: {study.best_value:.1f}%')
    return cb


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curve(train_returns, label, color,
                        pi_mean_ret=None, rand_mean_ret=None, plot_name=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sw       = min(1000, max(1, len(train_returns) // 10))
    smoothed = smooth(np.array(train_returns), sw)
    eps_ax   = np.arange(sw, len(train_returns) + 1)

    z          = np.polyfit(eps_ax, smoothed, 1)
    trend_line = np.poly1d(z)(eps_ax)
    slope_10k  = z[0] * 10_000

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eps_ax, smoothed,   color=color, lw=2.0, label=label)
    ax.plot(eps_ax, trend_line, color=color, lw=1.5, ls='--', alpha=0.55, label='Trend')
    ax.set_ylim(RETURN_YLIM)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (rolling avg)')
    ax.set_title(f'{label}  -  Learning Curve', fontweight='bold', color=H_COL)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    if plot_name:
        plt.savefig(f'{PLOTS_DIR}/{plot_name}_lc.png', dpi=150, bbox_inches='tight')
    plt.show()

    status = ('Converged (flat)' if abs(slope_10k) < 0.001
              else ('Still improving' if slope_10k > 0 else 'Declining'))
    print(f'  [{label}] Trend: {slope_10k:+.5f} return / 10k episodes -> {status}')
    print(f'  [{label}] Final smoothed return: {smoothed[-1]:.4f}')


def plot_survival_rate(final_surv, pi_surv, rand_surv,
                        label, color, agreement_pi=None, plot_name=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([label, 'PI', 'Random'],
                  [final_surv, pi_surv, rand_surv],
                  color=[color, REF_PI, REF_RAND],
                  edgecolor='white', alpha=0.9)
    ax.set_ylim(SURV_YLIM)
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title(f'{label}  -  Survival Rate', fontweight='bold', color=H_COL)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, v in zip(bars, [final_surv, pi_surv, rand_surv]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    if plot_name:
        plt.savefig(f'{PLOTS_DIR}/{plot_name}_sr.png', dpi=150, bbox_inches='tight')
    plt.show()

    n  = 2000
    ci = 1.96 * np.sqrt((final_surv/100) * (1 - final_surv/100) / n) * 100
    print(f'  [{label}] Survival Rate : {final_surv:.1f}% +/- {ci:.1f}%')
    print(f'  [{label}] vs PI         : {final_surv - pi_surv:+.1f} pp')
    print(f'  [{label}] vs Random     : {final_surv - rand_surv:+.1f} pp')
    if agreement_pi is not None:
        print(f'  [{label}] Agreement PI  : {agreement_pi:.1f}%')


def plot_optuna_history(study, label, color, plot_name=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    vals = [t.value for t in study.trials if t.value is not None]
    best = np.maximum.accumulate(vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(range(1, len(vals)+1), vals,
               color=color, alpha=0.4, s=30, label='Trial value')
    ax.plot(range(1, len(best)+1), best,
            color=color, lw=2.5, label=f'Best: {max(best):.1f}%')
    ax.axhline(max(best), color=color, lw=1.0, ls=':', alpha=0.4)
    ax.set_ylim(SURV_YLIM)
    ax.set_xlabel('Optuna Trial')
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title(f'{label}  -  Optuna Optimisation History',
                  fontweight='bold', color=H_COL)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    if plot_name:
        plt.savefig(f'{PLOTS_DIR}/{plot_name}_opt.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'  [{label}] Best Optuna : {study.best_value:.1f}%')
    print(f'  [{label}] Best params : {study.best_params}')


# ── Optuna runners ────────────────────────────────────────────────────────────

def run_optuna(objective, algo_name, n_trials, color=None,
                plot_name=None, seed=42):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials,
                    callbacks=[make_optuna_callback(algo_name, n_trials)])

    c  = color    or H_COL
    pn = (plot_name
           or 'fig_' + algo_name.replace('+','').replace(' ','_').lower() + '_opt')
    plot_optuna_history(study, algo_name, c, plot_name=pn)
    print(f'  Best survival : {study.best_value:.1f}%')
    print(f'  Best params   : {study.best_params}')
    return study


def _parse_param(trial, name, low, high, log, ptype):
    if ptype == 'int':
        return trial.suggest_int(name, int(low), int(high))
    return trial.suggest_float(name, float(low), float(high), log=bool(log))


def run_optuna_dynaq(param_ranges, n_trials, trial_eps, trial_eval, seed=42):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _require_ctx()

    def objective(trial):
        params = {
            name: _parse_param(trial, name, low, high, log, ptype)
            for name, (low, high, log, ptype) in param_ranges.items()
        }
        Q, _ = train_dynaq_ps(**params, n_episodes=trial_eps, verbose=False)
        return silent_eval(np.argmax(Q, axis=1), trial_eval)

    return run_optuna(objective, 'Dyna-Q+PS', n_trials, color=C_DQ,
                       plot_name='fig_5_dq_opt', seed=seed)


def run_optuna_qlearning(param_ranges, n_trials, trial_eps, trial_eval, seed=42):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _require_ctx()

    def objective(trial):
        params = {
            name: _parse_param(trial, name, low, high, log, ptype)
            for name, (low, high, log, ptype) in param_ranges.items()
        }
        Q, _ = train_qlearning(**params, n_episodes=trial_eps, verbose=False)
        return silent_eval(np.argmax(Q, axis=1), trial_eval)

    return run_optuna(objective, 'Q-Learning', n_trials, color=C_QL,
                       plot_name='fig_6_ql_opt', seed=seed)


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
            action = (np.random.randint(A) if np.random.rand()<epsilon
                       else int(np.argmax(Q[state])))
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
            if (episode+1)%5000==0:
                _progress(episode, n_episodes, surv_win, a_t, epsilon, 'Dyna-Q+PS', t0)
    env.close()
    if verbose: print(f'  [Dyna-Q+PS] Done: {time.time()-t0:.0f}s | PQ: {len(pq)}')
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
            action = (np.random.randint(A) if np.random.rand()<epsilon
                       else int(np.argmax(Q[state])))
            next_obs, reward, te, tr, _ = env.step(action)
            next_state = int(next_obs); done = te or tr
            sr = reward + gamma*pi_V[next_state]*(not done) - pi_V[state]
            Q[state, action] += a_t*(sr+gamma*np.max(Q[next_state])*(not done) - Q[state, action])
            total_r += reward; state = next_state
        epsilon = max(0.005, epsilon*eps_decay); returns.append(total_r)
        if verbose:
            surv_win.append(int(total_r>0))
            if len(surv_win)>500: surv_win.pop(0)
            if (episode+1)%5000==0:
                _progress(episode, n_episodes, surv_win, a_t, epsilon, 'Q-Learning', t0)
    env.close()
    if verbose: print(f'  [Q-Learning] Done: {time.time()-t0:.0f}s')
    return Q, returns


# ── Creative Extension: Clinical Decision Support Analysis ────────────────────

def clinical_acuity_analysis(pi_V, cluster_centers, sofa_scores, S_n,
                              feature_names, seed=42):
    """ V_PI as a Clinical Acuity Score (SHAP)."""
    from sklearn.ensemble import GradientBoostingRegressor
    import shap

    X = cluster_centers[:S_n].astype(np.float64)
    y = pi_V[:S_n]

    surr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                      random_state=seed, subsample=0.8)
    surr.fit(X, y)
    r2 = surr.score(X, y)
    print(f'  Surrogate R2 (V_PI): {r2:.4f}  (>0.85 = good fit for SHAP)')

    exp       = shap.TreeExplainer(surr)
    shap_vals = exp.shap_values(X)
    shap_imp  = np.abs(shap_vals).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('9.1 V_PI as a Clinical Acuity Score',
                  fontsize=13, fontweight='bold', color=H_COL)

    sc = axes[0].scatter(sofa_scores, y, c=y, cmap='Blues_r',
                          s=18, alpha=0.7, vmin=0, vmax=0.9)
    plt.colorbar(sc, ax=axes[0], label='V_PI(s)')
    axes[0].set_xlabel('SOFA Score (clinical severity)')
    axes[0].set_ylabel('V_PI(s)  =  P(survive | optimal treatment)')
    axes[0].set_title('AI Acuity vs Clinical SOFA Score', fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)
    r = np.corrcoef(sofa_scores, y)[0, 1]
    axes[0].text(0.05, 0.05, f'r = {r:.3f}', transform=axes[0].transAxes,
                  fontsize=11, fontweight='bold', color=B0)

    sorted_idx = np.argsort(shap_imp)[::-1][:15]
    bar_cols   = [B2 if shap_vals[:, i].mean() > 0 else 'tomato'
                   for i in sorted_idx]
    axes[1].barh([feature_names[i] for i in sorted_idx],
                  shap_imp[sorted_idx],
                  color=bar_cols, edgecolor='white', alpha=0.9)
    axes[1].set_xlabel('Mean |SHAP| value')
    axes[1].set_title('Top 15 Features Driving V_PI Predictions',
                       fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fig_9_1_acuity.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'  SOFA vs V_PI correlation: r = {r:.3f}')
    print('  Top 5 features:')
    for i in sorted_idx[:5]:
        direction = 'positive' if shap_vals[:, i].mean() > 0 else 'negative'
        print(f'    {feature_names[i]:<20}: |SHAP| = {shap_imp[i]:.4f}  ({direction})')

    return shap_vals, shap_imp


def treatment_strategy_analysis(pi_V, pi_policy, dq_policy, ql_policy, S_n):
    """ Treatment Strategy by Patient Prognosis Quartile."""
    y        = pi_V[:S_n]
    q_bounds = np.percentile(y, [25, 50, 75])
    q_idx    = np.digitize(y, q_bounds)
    q_labels = ['Q1\nCritical', 'Q2\nSevere', 'Q3\nModerate', 'Q4\nStable']

    algorithms = [
        ('PI',          pi_policy[:S_n], REF_PI),
        ('Dyna-Q+PS',   dq_policy[:S_n], C_DQ),
        ('Q-Learning',  ql_policy[:S_n], C_QL),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('9.2 Treatment Strategy by Patient Prognosis Quartile',
                  fontsize=13, fontweight='bold', color=H_COL)

    results = {}
    for ax_idx, (algo_name, policy, color) in enumerate(algorithms):
        intensity = ((policy // 5) + (policy % 5)) / 8.0
        q_means   = [intensity[q_idx == q].mean() for q in range(4)]
        q_stds    = [intensity[q_idx == q].std()  for q in range(4)]
        results[algo_name] = q_means

        bars = axes[ax_idx].bar(q_labels, q_means, color=color,
                                 edgecolor='white', alpha=0.9, width=0.6)
        axes[ax_idx].errorbar(q_labels, q_means, yerr=q_stds,
                                fmt='none', color=B0, capsize=4, lw=1.5)
        axes[ax_idx].set_ylim(0, 0.8)
        axes[ax_idx].set_ylabel('Mean Treatment Intensity (0=none, 1=max)')
        axes[ax_idx].set_title(algo_name, fontweight='bold')
        axes[ax_idx].spines[['top', 'right']].set_visible(False)
        for bar, v in zip(bars, q_means):
            axes[ax_idx].text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.01,
                                f'{v:.3f}', ha='center',
                                fontsize=9, fontweight='bold')

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fig_9_2_treatment.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('  Mean treatment intensity by V_PI quartile:')
    header = f'  {"Quartile":<15}'
    for name, _, _ in algorithms:
        header += f'{name:>13}'
    print(header)
    print('  ' + '-' * (15 + 13 * 3))
    for q in range(4):
        row = f'  {q_labels[q].replace(chr(10), " "):<15}'
        for name, _, _ in algorithms:
            row += f'{results[name][q]:>13.3f}'
        print(row)

    return results


def shap_comparison(Q_dq, Q_ql, shap_vpi_imp,
                     cluster_centers, S_n, feature_names, seed=42):
    """ Comparison of Learned Feature Importance (SHAP)."""
    from sklearn.ensemble import GradientBoostingRegressor
    import shap

    X = cluster_centers[:S_n].astype(np.float64)

    def _fit_shap(Q_table, label):
        y    = np.max(Q_table[:S_n], axis=1)
        surr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                          random_state=seed, subsample=0.8)
        surr.fit(X, y)
        r2  = surr.score(X, y)
        exp = shap.TreeExplainer(surr)
        sv  = exp.shap_values(X)
        imp = np.abs(sv).mean(axis=0)
        print(f'  Surrogate R2 ({label}): {r2:.4f}')
        return imp

    print('  Computing SHAP for each algorithm value function...')
    imp_dq = _fit_shap(Q_dq, 'Dyna-Q+PS')
    imp_ql = _fit_shap(Q_ql, 'Q-Learning')

    def _norm(x): return x / (x.max() + 1e-9)
    n_vpi = _norm(shap_vpi_imp)
    n_dq  = _norm(imp_dq)
    n_ql  = _norm(imp_ql)

    r_dq = float(np.corrcoef(n_vpi, n_dq)[0, 1])
    r_ql = float(np.corrcoef(n_vpi, n_ql)[0, 1])

    top_idx = np.argsort(n_vpi)[::-1][:15]
    x_pos   = np.arange(15)
    w       = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('9.3 Feature Importance: V_PI vs Dyna-Q+PS vs Q-Learning',
                  fontsize=13, fontweight='bold', color=H_COL)
    ax.bar(x_pos - w, n_vpi[top_idx], w, label='V_PI (optimal)',
           color=REF_PI, edgecolor='white', alpha=0.9)
    ax.bar(x_pos,     n_dq[top_idx],  w,
           label=f'Dyna-Q+PS (r={r_dq:.3f})',
           color=C_DQ, edgecolor='white', alpha=0.9)
    ax.bar(x_pos + w, n_ql[top_idx],  w,
           label=f'Q-Learning (r={r_ql:.3f})',
           color=C_QL, edgecolor='white', alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([feature_names[i] for i in top_idx],
                        rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Normalised Mean |SHAP| Value')
    ax.set_title('Top 15 features by V_PI importance', fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fig_9_3_shap.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\n  Feature importance correlation with V_PI:')
    print(f'    Dyna-Q+PS  : r = {r_dq:.3f}')
    print(f'    Q-Learning : r = {r_ql:.3f}')

    return imp_dq, imp_ql, r_dq, r_ql
