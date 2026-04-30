"""
FGAWO Simulation: Fuzzy-Genetic Adaptive Window Optimization
for Balanced Sliding Window Protocol

Run:  python fgawo_simulation.py
Output:
  - results_data.json          (upload this to Claude)
  - graph_01_window_vs_time.png
  - graph_02_congestion_vs_time.png
  - graph_03_loss_vs_time.png
  - graph_04_delay_vs_time.png
  - graph_05_throughput_vs_time.png
  - graph_06_comparison_bar.png
  - graph_07_3d_surface.png
  - graph_08_window_comparison_line.png
  - graph_09_qos_comparison.png
  - graph_10_fitness_convergence.png
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(42)

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
W_MIN, W_MAX = 1, 64
D_MAX        = 300.0       # ms
MSS          = 1500 * 8    # bits
B_LINK       = 100e6       # bps
DELTA_PROC   = 5.0         # ms processing overhead
RHO          = 0.3         # adaptation rate
T_STEPS      = 500         # simulation steps

# GA
N_POP  = 50
N_GEN  = 100
KAPPA  = 0.5

# Window output singletons defined inline in build_rules() below

# ─────────────────────────────────────────────
# MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────

def mf_congestion(C):
    low  = max(0.0, (0.30 - C) / 0.30)
    med  = max(0.0, 1.0 - abs(C - 0.50) / 0.25)
    high = max(0.0, (C - 0.70) / 0.30)
    return low, med, high

def mf_loss(p):
    neg = max(0.0, (0.05 - p) / 0.05)
    mod = max(0.0, 1.0 - abs(p - 0.10) / 0.10)
    sev = max(0.0, (p - 0.15) / 0.85)
    return neg, mod, sev

def mf_delay(d):
    sht = max(0.0, 1.0 - d / 60.0)
    mod = max(0.0, 1.0 - abs(d - 120.0) / 60.0)
    lng = max(0.0, (d - 180.0) / 120.0)
    return sht, mod, lng

# ─────────────────────────────────────────────
# RULE BASE  (27 rules: C x p x d → W)
# index order: (C: low/med/high) x (p: neg/mod/sev) x (d: sht/mod/lng)
# ─────────────────────────────────────────────
# Flat rule list: (c_idx, p_idx, d_idx, window_singleton_value)
# c: 0=Low, 1=Med, 2=High  |  p: 0=Neg, 1=Mod, 2=Sev  |  d: 0=Sht, 1=Mod, 2=Lng
_VL = W_MAX
_L  = W_MIN + 0.75 * (W_MAX - W_MIN)
_M  = W_MIN + 0.50 * (W_MAX - W_MIN)
_S  = W_MIN + 0.25 * (W_MAX - W_MIN)
_VS = W_MIN

def build_rules():
    """Returns list of 27 (c_idx, p_idx, d_idx, output_singleton) tuples."""
    rules = [
        # C=Low (ci=0)
        (0, 0, 0, _VL), (0, 0, 1, _L),  (0, 0, 2, _M),   # p=Neg
        (0, 1, 0, _L),  (0, 1, 1, _M),  (0, 1, 2, _S),   # p=Mod
        (0, 2, 0, _M),  (0, 2, 1, _S),  (0, 2, 2, _VS),  # p=Sev
        # C=Med (ci=1)
        (1, 0, 0, _L),  (1, 0, 1, _M),  (1, 0, 2, _S),   # p=Neg
        (1, 1, 0, _M),  (1, 1, 1, _S),  (1, 1, 2, _VS),  # p=Mod
        (1, 2, 0, _S),  (1, 2, 1, _VS), (1, 2, 2, _VS),  # p=Sev
        # C=High (ci=2)
        (2, 0, 0, _M),  (2, 0, 1, _S),  (2, 0, 2, _VS),  # p=Neg
        (2, 1, 0, _S),  (2, 1, 1, _VS), (2, 1, 2, _VS),  # p=Mod
        (2, 2, 0, _VS), (2, 2, 1, _VS), (2, 2, 2, _VS),  # p=Sev
    ]
    return rules

RULES = build_rules()
N_RULES = len(RULES)  # 27

# ─────────────────────────────────────────────
# FUZZY INFERENCE  (defuzzification)
# ─────────────────────────────────────────────

def fuzzy_infer(C, p, d, weights):
    mu_c = mf_congestion(C)
    mu_p = mf_loss(p)
    mu_d = mf_delay(d)

    num = 0.0
    den = 0.0
    for k, (ci, pi, di, w_hat) in enumerate(RULES):
        phi = min(mu_c[ci], mu_p[pi], mu_d[di])
        wk  = weights[k]
        num += wk * phi * w_hat
        den += wk * phi

    if den < 1e-9:
        return (W_MIN + W_MAX) / 2.0
    return np.clip(num / den, W_MIN, W_MAX)

# ─────────────────────────────────────────────
# THROUGHPUT  (Mbps)
# ─────────────────────────────────────────────

def throughput(W, p, d):
    rtt = (2 * d + DELTA_PROC) / 1000.0   # seconds
    if rtt <= 0:
        return 0.0
    return (W * MSS / rtt) * (1 - p) / 1e6   # Mbps

# ─────────────────────────────────────────────
# COST FUNCTION
# ─────────────────────────────────────────────

def cost(C, p, d, W, coeff):
    alpha, beta, gamma, eta = coeff
    th    = throughput(W, p, d)
    th_max = throughput(W_MAX, 0.0, 10.0)
    return alpha*C + beta*p + gamma*(d/D_MAX) - eta*(th/th_max)

# ─────────────────────────────────────────────
# SIMULATE ONE TRAJECTORY
# ─────────────────────────────────────────────

def simulate(weights, coeff, scenario='medium', T=T_STEPS):
    rho = 0.3
    W_cur = float((W_MIN + W_MAX) // 2)

    C_arr = np.zeros(T)
    p_arr = np.zeros(T)
    d_arr = np.zeros(T)
    W_arr = np.zeros(T)
    th_arr = np.zeros(T)

    # Initial conditions
    d_cur = 100.0
    for t in range(T):
        # --- Generate network state ---
        if scenario == 'low':
            C = np.random.uniform(0.0, 0.3)
            p = np.random.uniform(0.0, 0.05)
        elif scenario == 'medium':
            C = np.random.uniform(0.3, 0.6)
            p = np.random.uniform(0.05, 0.15)
        elif scenario == 'high':
            base_C = np.random.uniform(0.5, 0.8)
            spike  = 0.9 if (t % 50 < 15) else 0.0
            C = min(1.0, base_C + spike * 0.2)
            p = np.random.uniform(0.1, 0.3)
        elif scenario == 'mixed':
            # Gradually changing load
            phase = t / T
            C = 0.1 + 0.8 * abs(np.sin(2 * np.pi * phase))
            C += np.random.normal(0, 0.05)
            C = np.clip(C, 0.0, 1.0)
            p = 0.02 + 0.2 * C + np.random.normal(0, 0.01)
            p = np.clip(p, 0.0, 0.5)

        # Delay random walk
        d_cur = np.clip(d_cur + np.random.normal(0, 10), 20, D_MAX)
        d = d_cur

        # --- FGAWO control ---
        W_target = fuzzy_infer(C, p, d, weights)
        W_cur    = (1 - rho) * W_cur + rho * W_target
        W_int    = int(round(np.clip(W_cur, W_MIN, W_MAX)))

        C_arr[t]  = C
        p_arr[t]  = p
        d_arr[t]  = d
        W_arr[t]  = W_int
        th_arr[t] = throughput(W_int, p, d)

    return C_arr, p_arr, d_arr, W_arr, th_arr

# ─────────────────────────────────────────────
# STATIC WINDOW BASELINE
# ─────────────────────────────────────────────

def simulate_static(scenario='medium', T=T_STEPS):
    """BDP-based static window — uses full W_MAX regardless of conditions."""
    d_cur = 100.0
    C_arr = np.zeros(T); p_arr = np.zeros(T)
    d_arr = np.zeros(T); W_arr = np.zeros(T); th_arr = np.zeros(T)

    for t in range(T):
        if scenario == 'low':
            C = np.random.uniform(0.0, 0.3)
            p = np.random.uniform(0.0, 0.05)
        elif scenario == 'medium':
            C = np.random.uniform(0.3, 0.6)
            p = np.random.uniform(0.05, 0.15)
        elif scenario == 'high':
            base_C = np.random.uniform(0.5, 0.8)
            spike  = 0.9 if (t % 50 < 15) else 0.0
            C = min(1.0, base_C + spike * 0.2)
            p = np.random.uniform(0.1, 0.3)
        elif scenario == 'mixed':
            phase = t / T
            C = np.clip(0.1 + 0.8 * abs(np.sin(2 * np.pi * phase)) + np.random.normal(0, 0.05), 0, 1)
            p = np.clip(0.02 + 0.2 * C + np.random.normal(0, 0.01), 0, 0.5)

        d_cur = np.clip(d_cur + np.random.normal(0, 10), 20, D_MAX)
        W_static = int(round(np.clip((B_LINK * (2*d_cur/1000)) / MSS, W_MIN, W_MAX)))
        C_arr[t]  = C; p_arr[t] = p; d_arr[t] = d_cur
        W_arr[t]  = W_static
        th_arr[t] = throughput(W_static, p, d_cur)

    return C_arr, p_arr, d_arr, W_arr, th_arr

# ─────────────────────────────────────────────
# TCP-AIMD BASELINE
# ─────────────────────────────────────────────

def simulate_aimd(scenario='medium', T=T_STEPS):
    """Simplified TCP AIMD: +1 per RTT on success, /2 on loss."""
    d_cur = 100.0
    W_cur = 1.0
    C_arr = np.zeros(T); p_arr = np.zeros(T)
    d_arr = np.zeros(T); W_arr = np.zeros(T); th_arr = np.zeros(T)

    for t in range(T):
        if scenario == 'low':
            C = np.random.uniform(0.0, 0.3)
            p = np.random.uniform(0.0, 0.05)
        elif scenario == 'medium':
            C = np.random.uniform(0.3, 0.6)
            p = np.random.uniform(0.05, 0.15)
        elif scenario == 'high':
            base_C = np.random.uniform(0.5, 0.8)
            spike  = 0.9 if (t % 50 < 15) else 0.0
            C = min(1.0, base_C + spike * 0.2)
            p = np.random.uniform(0.1, 0.3)
        elif scenario == 'mixed':
            phase = t / T
            C = np.clip(0.1 + 0.8 * abs(np.sin(2 * np.pi * phase)) + np.random.normal(0, 0.05), 0, 1)
            p = np.clip(0.02 + 0.2 * C + np.random.normal(0, 0.01), 0, 0.5)

        d_cur = np.clip(d_cur + np.random.normal(0, 10), 20, D_MAX)

        if np.random.random() < p:
            W_cur = max(W_MIN, W_cur / 2.0)   # multiplicative decrease
        else:
            W_cur = min(W_MAX, W_cur + 1.0)   # additive increase

        W_int = int(round(W_cur))
        C_arr[t]  = C; p_arr[t] = p; d_arr[t] = d_cur
        W_arr[t]  = W_int
        th_arr[t] = throughput(W_int, p, d_cur)

    return C_arr, p_arr, d_arr, W_arr, th_arr

# ─────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────

def fitness(chrom, T=200):
    weights = chrom[:N_RULES]
    coeff   = chrom[N_RULES:]
    _, p_arr, d_arr, W_arr, th_arr = simulate(weights, coeff, scenario='mixed', T=T)
    C_arr, _, _, _, _ = simulate(weights, coeff, scenario='mixed', T=T)
    scores = []
    for t in range(T):
        c_val = cost(C_arr[t], p_arr[t], d_arr[t], W_arr[t], coeff)
        scores.append(th_arr[t] - KAPPA * c_val)
    return np.mean(scores)

def ga_optimize(n_pop=N_POP, n_gen=N_GEN):
    dim = N_RULES + 4   # 27 rule weights + alpha,beta,gamma,eta
    pop = np.random.dirichlet(np.ones(N_RULES), size=n_pop)
    coeff_init = np.random.dirichlet(np.ones(4), size=n_pop)
    pop = np.hstack([pop, coeff_init])

    best_fitness_history = []
    best_chrom = pop[0].copy()
    best_fit   = -np.inf

    print("Running GA optimization...")
    for g in range(n_gen):
        fits = np.array([fitness(pop[i]) for i in range(n_pop)])

        gen_best_idx = np.argmax(fits)
        if fits[gen_best_idx] > best_fit:
            best_fit   = fits[gen_best_idx]
            best_chrom = pop[gen_best_idx].copy()

        best_fitness_history.append(best_fit)
        if (g + 1) % 10 == 0:
            print(f"  Gen {g+1}/{n_gen}  best_fitness={best_fit:.4f}")

        # Tournament selection
        new_pop = [best_chrom.copy()]   # elitism
        while len(new_pop) < n_pop:
            idx = np.random.choice(n_pop, 3, replace=False)
            winner = idx[np.argmax(fits[idx])]
            new_pop.append(pop[winner].copy())
        pop = np.array(new_pop)

        # SBX Crossover
        for i in range(1, n_pop - 1, 2):
            if np.random.random() < 0.9:
                eta_c = 20
                u = np.random.random(dim)
                beta = np.where(u < 0.5,
                                (2*u)**(1/(eta_c+1)),
                                (1/(2*(1-u)))**(1/(eta_c+1)))
                child1 = 0.5 * ((1+beta)*pop[i] + (1-beta)*pop[i+1])
                child2 = 0.5 * ((1-beta)*pop[i] + (1+beta)*pop[i+1])
                pop[i]   = np.clip(child1, 0, 1)
                pop[i+1] = np.clip(child2, 0, 1)

        # Polynomial mutation
        for i in range(1, n_pop):
            for j in range(dim):
                if np.random.random() < 1.0/dim:
                    u = np.random.random()
                    eta_m = 20
                    if u < 0.5:
                        delta = (2*u)**(1/(eta_m+1)) - 1
                    else:
                        delta = 1 - (2*(1-u))**(1/(eta_m+1))
                    pop[i, j] = np.clip(pop[i, j] + delta, 0, 1)

        # Normalise rule weights
        for i in range(n_pop):
            s = pop[i, :N_RULES].sum()
            if s > 0:
                pop[i, :N_RULES] /= s
            s2 = pop[i, N_RULES:].sum()
            if s2 > 0:
                pop[i, N_RULES:] /= s2

    return best_chrom, best_fitness_history

# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────

COLORS = {
    'fgawo' : '#1f77b4',
    'static': '#d62728',
    'aimd'  : '#2ca02c',
}

def moving_avg(arr, w=15):
    return np.convolve(arr, np.ones(w)/w, mode='same')

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── 1. GA Offline Training ──────────────────
    best_chrom, fitness_history = ga_optimize(N_POP, N_GEN)
    weights = best_chrom[:N_RULES]
    coeff   = best_chrom[N_RULES:]
    print(f"\nBest weights (first 5): {weights[:5].round(4)}")
    print(f"Cost coefficients (α,β,γ,η): {coeff.round(4)}")

    # ── 2. Run All Scenarios ────────────────────
    scenarios = ['low', 'medium', 'high', 'mixed']
    results   = {}

    for sc in scenarios:
        np.random.seed(42)
        C_f, p_f, d_f, W_f, th_f = simulate(weights, coeff, scenario=sc)
        np.random.seed(42)
        C_s, p_s, d_s, W_s, th_s = simulate_static(scenario=sc)
        np.random.seed(42)
        C_a, p_a, d_a, W_a, th_a = simulate_aimd(scenario=sc)

        results[sc] = {
            'fgawo' : {'C': C_f.tolist(), 'p': p_f.tolist(), 'd': d_f.tolist(),
                       'W': W_f.tolist(), 'th': th_f.tolist()},
            'static': {'C': C_s.tolist(), 'p': p_s.tolist(), 'd': d_s.tolist(),
                       'W': W_s.tolist(), 'th': th_s.tolist()},
            'aimd'  : {'C': C_a.tolist(), 'p': p_a.tolist(), 'd': d_a.tolist(),
                       'W': W_a.tolist(), 'th': th_a.tolist()},
        }

    # Summary statistics
    summary = {}
    for sc in scenarios:
        summary[sc] = {}
        for method in ['fgawo', 'static', 'aimd']:
            d = results[sc][method]
            W_arr  = np.array(d['W'])
            th_arr = np.array(d['th'])
            p_arr  = np.array(d['p'])
            summary[sc][method] = {
                'mean_throughput_mbps' : round(float(np.mean(th_arr)), 4),
                'mean_loss_pct'        : round(float(np.mean(p_arr) * 100), 4),
                'mean_window'          : round(float(np.mean(W_arr)), 2),
                'window_variance'      : round(float(np.var(W_arr)), 2),
                'qos_score'            : round(float(np.mean(th_arr) * (1 - np.mean(p_arr))), 4),
                'mean_delay_ms'        : round(float(np.mean(d['d'])), 2),
            }

    # Save JSON
    output = {
        'ga_fitness_history': fitness_history,
        'best_weights'       : weights.tolist(),
        'best_coeff'         : coeff.tolist(),
        'summary'            : summary,
        'simulation_params'  : {
            'T_STEPS': T_STEPS, 'W_MIN': W_MIN, 'W_MAX': W_MAX,
            'RHO': RHO, 'N_POP': N_POP, 'N_GEN': N_GEN, 'KAPPA': KAPPA,
            'MSS_bytes': 1500, 'B_LINK_Mbps': 100,
        },
        'results': results,
    }
    with open('results_data.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved: results_data.json")

    # ── PLOTS ───────────────────────────────────

    t_axis = np.arange(T_STEPS)
    sc_main = 'mixed'   # primary scenario for time-series graphs
    r = results[sc_main]

    # ── Graph 1: Window Size vs Time ────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, moving_avg(r['fgawo']['W']),  color=COLORS['fgawo'],  lw=1.8, label='FGAWO')
    ax.plot(t_axis, moving_avg(r['static']['W']), color=COLORS['static'], lw=1.4, ls='--', label='Static (BDP)')
    ax.plot(t_axis, moving_avg(r['aimd']['W']),   color=COLORS['aimd'],   lw=1.4, ls=':',  label='TCP-AIMD')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Window Size (packets)')
    ax.set_title('Window Size vs Time (Mixed Scenario)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_01_window_vs_time.png', dpi=150)
    plt.close()
    print("Saved: graph_01_window_vs_time.png")

    # ── Graph 2: Congestion Level vs Time ───────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, moving_avg(r['fgawo']['C']), color='#9467bd', lw=1.8)
    ax.fill_between(t_axis, 0, moving_avg(r['fgawo']['C']), alpha=0.2, color='#9467bd')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Congestion Level C(t)')
    ax.set_title('Congestion Level vs Time (Mixed Scenario)')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_02_congestion_vs_time.png', dpi=150)
    plt.close()
    print("Saved: graph_02_congestion_vs_time.png")

    # ── Graph 3: Packet Loss vs Time ────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, moving_avg(np.array(r['fgawo']['p'])*100),  color=COLORS['fgawo'],  lw=1.8, label='FGAWO')
    ax.plot(t_axis, moving_avg(np.array(r['static']['p'])*100), color=COLORS['static'], lw=1.4, ls='--', label='Static')
    ax.plot(t_axis, moving_avg(np.array(r['aimd']['p'])*100),   color=COLORS['aimd'],   lw=1.4, ls=':',  label='TCP-AIMD')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Packet Loss Rate (%)')
    ax.set_title('Packet Loss Rate vs Time (Mixed Scenario)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_03_loss_vs_time.png', dpi=150)
    plt.close()
    print("Saved: graph_03_loss_vs_time.png")

    # ── Graph 4: Delay vs Time ───────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, moving_avg(r['fgawo']['d']), color='#8c564b', lw=1.8)
    ax.set_xlabel('Time Step'); ax.set_ylabel('One-Way Delay (ms)')
    ax.set_title('Propagation Delay vs Time (Mixed Scenario)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_04_delay_vs_time.png', dpi=150)
    plt.close()
    print("Saved: graph_04_delay_vs_time.png")

    # ── Graph 5: Throughput vs Time ──────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, moving_avg(r['fgawo']['th']),  color=COLORS['fgawo'],  lw=1.8, label='FGAWO')
    ax.plot(t_axis, moving_avg(r['static']['th']), color=COLORS['static'], lw=1.4, ls='--', label='Static')
    ax.plot(t_axis, moving_avg(r['aimd']['th']),   color=COLORS['aimd'],   lw=1.4, ls=':',  label='TCP-AIMD')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Throughput (Mbps)')
    ax.set_title('Throughput vs Time (Mixed Scenario)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_05_throughput_vs_time.png', dpi=150)
    plt.close()
    print("Saved: graph_05_throughput_vs_time.png")

    # ── Graph 6: Comparison Bar Chart ───────────
    metrics = ['mean_throughput_mbps', 'mean_loss_pct', 'qos_score', 'window_variance']
    m_labels = ['Avg Throughput\n(Mbps)', 'Avg Loss\n(%)', 'QoS Score', 'Window\nVariance']
    methods  = ['fgawo', 'static', 'aimd']
    m_names  = ['FGAWO', 'Static (BDP)', 'TCP-AIMD']
    m_colors = [COLORS['fgawo'], COLORS['static'], COLORS['aimd']]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for ai, (metric, mlabel) in enumerate(zip(metrics, m_labels)):
        vals = [summary['mixed'][m][metric] for m in methods]
        bars = axes[ai].bar(m_names, vals, color=m_colors, edgecolor='black', linewidth=0.7)
        for bar, v in zip(bars, vals):
            axes[ai].text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                          f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        axes[ai].set_title(mlabel); axes[ai].set_ylabel(mlabel)
        axes[ai].tick_params(axis='x', rotation=15)
        axes[ai].grid(axis='y', alpha=0.3)
    fig.suptitle('Method Comparison — Mixed Scenario', fontweight='bold')
    plt.tight_layout()
    plt.savefig('graph_06_comparison_bar.png', dpi=150)
    plt.close()
    print("Saved: graph_06_comparison_bar.png")

    # ── Graph 7: 3D Surface W = f(C, p) ─────────
    C_range = np.linspace(0, 1, 50)
    p_range = np.linspace(0, 0.4, 50)
    CG, PG  = np.meshgrid(C_range, p_range)
    WG = np.vectorize(lambda c, p: fuzzy_infer(c, p, 100.0, weights))(CG, PG)

    fig = plt.figure(figsize=(10, 7))
    ax3 = fig.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(CG, PG, WG, cmap=cm.viridis, edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax3, shrink=0.5, label='Window Size W')
    ax3.set_xlabel('Congestion C'); ax3.set_ylabel('Packet Loss p')
    ax3.set_zlabel('Window Size W (pkts)')
    ax3.set_title('3D Control Surface: W = f(C, p) at d=100ms')
    plt.tight_layout()
    plt.savefig('graph_07_3d_surface.png', dpi=150)
    plt.close()
    print("Saved: graph_07_3d_surface.png")

    # ── Graph 8: Window Comparison Across Scenarios
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    sc_titles = {'low': 'Low Load', 'medium': 'Medium Load',
                 'high': 'High Load (Burst)', 'mixed': 'Mixed (Dynamic)'}
    for i, sc in enumerate(scenarios):
        ax = axes[i]
        ax.plot(t_axis, moving_avg(results[sc]['fgawo']['W']),  color=COLORS['fgawo'],  lw=1.8, label='FGAWO')
        ax.plot(t_axis, moving_avg(results[sc]['static']['W']), color=COLORS['static'], lw=1.4, ls='--', label='Static')
        ax.plot(t_axis, moving_avg(results[sc]['aimd']['W']),   color=COLORS['aimd'],   lw=1.4, ls=':',  label='TCP-AIMD')
        ax.set_title(sc_titles[sc]); ax.set_ylabel('W (pkts)')
        ax.set_xlabel('Time Step'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Window Size Comparison Across All Scenarios', fontweight='bold')
    plt.tight_layout()
    plt.savefig('graph_08_window_comparison_line.png', dpi=150)
    plt.close()
    print("Saved: graph_08_window_comparison_line.png")

    # ── Graph 9: QoS Score Across Scenarios ─────
    sc_labels = [sc_titles[s] for s in scenarios]
    fgawo_qos  = [summary[s]['fgawo']['qos_score']  for s in scenarios]
    static_qos = [summary[s]['static']['qos_score'] for s in scenarios]
    aimd_qos   = [summary[s]['aimd']['qos_score']   for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, fgawo_qos,  width, label='FGAWO',        color=COLORS['fgawo'],  edgecolor='black', lw=0.7)
    ax.bar(x,         static_qos, width, label='Static (BDP)', color=COLORS['static'], edgecolor='black', lw=0.7)
    ax.bar(x + width, aimd_qos,   width, label='TCP-AIMD',     color=COLORS['aimd'],   edgecolor='black', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(sc_labels, rotation=15)
    ax.set_ylabel('QoS Score = Throughput × (1 − Loss)'); ax.set_title('QoS Score Across All Scenarios')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_09_qos_comparison.png', dpi=150)
    plt.close()
    print("Saved: graph_09_qos_comparison.png")

    # ── Graph 10: GA Fitness Convergence ────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fitness_history, color='#e377c2', lw=2)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best Fitness')
    ax.set_title('GA Fitness Convergence over Generations')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_10_fitness_convergence.png', dpi=150)
    plt.close()
    print("Saved: graph_10_fitness_convergence.png")

    # ── Print Summary Table ──────────────────────
    print("\n" + "="*70)
    print(f"{'Scenario':<10} {'Method':<12} {'Throughput':>12} {'Loss%':>8} {'QoS':>8} {'Var(W)':>10}")
    print("="*70)
    for sc in scenarios:
        for method in methods:
            s = summary[sc][method]
            print(f"{sc:<10} {method:<12} "
                  f"{s['mean_throughput_mbps']:>12.4f} "
                  f"{s['mean_loss_pct']:>8.3f} "
                  f"{s['qos_score']:>8.4f} "
                  f"{s['window_variance']:>10.2f}")
        print("-"*70)

    print("\nAll done! Upload 'results_data.json' to Claude for the paper sections.")

if __name__ == '__main__':
    main()
