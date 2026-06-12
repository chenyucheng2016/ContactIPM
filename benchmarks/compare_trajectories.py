#!/usr/bin/env python3
"""
Compare ContactIPM vs acados trajectories for benchmark problems.

Usage:
    python compare_trajectories.py pendulum
    python compare_trajectories.py quadrotor
    python compare_trajectories.py chain_mass
    python compare_trajectories.py all

Reads JSON dumps from benchmarks/data/. If JSON not found, attempts to run
the C++ binary and acados script via subprocess to generate them.
Plots saved to benchmarks/figures/{problem}_comparison.png.
"""
import sys
import os
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join('benchmarks', 'data')
FIG_DIR = os.path.join('benchmarks', 'figures')

PROBLEMS = {
    'pendulum': {
        'cpp_binary': os.path.join('build', 'Release', 'pendulum_nmpc_paper.exe'),
        'py_script': os.path.join('benchmarks', 'acados_pendulum.py'),
        'state_labels': ['x (cart)', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$'],
        'ctrl_labels': ['F'],
        'cons_labels': [r'$x \geq -2$', r'$x \leq 2$', r'$F \geq -30$', r'$F \leq 30$'],
    },
    'quadrotor': {
        'cpp_binary': os.path.join('build', 'Release', 'quadrotor_2d_nmpc.exe'),
        'py_script': os.path.join('benchmarks', 'acados_quadrotor.py'),
        'state_labels': ['y', 'z', r'$\phi$', r'$v_y$', r'$v_z$', r'$\dot{\phi}$'],
        'ctrl_labels': [r'$u_1$ (thrust)', r'$u_2$ (torque)'],
        'cons_labels': [r'$u_1 \geq 0$', r'$u_1 \leq 2mg$',
                         r'$u_2 \geq -\tau$', r'$u_2 \leq \tau$',
                         r'$z \geq 0$', 'inactive'],
    },
    'chain_mass': {
        'cpp_binary': os.path.join('build', 'Release', 'chain_mass_nmpc.exe'),
        'py_script': os.path.join('benchmarks', 'acados_chain_mass.py'),
        'state_labels': [f'x{i}' for i in range(5)] + [f'v{i}' for i in range(5)],
        'ctrl_labels': [f'F{i}' for i in range(3)],
        'cons_labels': ([f'F{i}>-Fmax' for i in range(3)] +
                        [f'F{i}<Fmax' for i in range(3)] +
                        [f'x{i}>-Xmax' for i in range(5)] +
                        [f'x{i}<Xmax' for i in range(5)]),
    },
}


def load_json(problem_name):
    """Load JSON data for both solvers. Returns (cipm, acados) dicts or None."""
    cipm_path = os.path.join(DATA_DIR, f'contactipm_{problem_name}.json')
    acad_path = os.path.join(DATA_DIR, f'acados_{problem_name}.json')

    cipm = None
    acad = None

    if os.path.exists(cipm_path):
        with open(cipm_path) as f:
            cipm = json.load(f)
    if os.path.exists(acad_path):
        with open(acad_path) as f:
            acad = json.load(f)

    return cipm, acad


def generate_json(problem_name):
    """Run C++ binary and/or acados script via subprocess to generate JSON."""
    cfg = PROBLEMS[problem_name]
    cipm_path = os.path.join(DATA_DIR, f'contactipm_{problem_name}.json')
    acad_path = os.path.join(DATA_DIR, f'acados_{problem_name}.json')

    if not os.path.exists(cipm_path):
        binary = cfg['cpp_binary']
        if os.path.exists(binary):
            print(f"  Running C++ binary: {binary}")
            result = subprocess.run([binary], capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"  C++ binary failed (code {result.returncode})")
                print(result.stderr[:500])
        else:
            print(f"  C++ binary not found: {binary}")

    if not os.path.exists(acad_path):
        script = cfg['py_script']
        if os.path.exists(script):
            print(f"  Running acados script: {script}")
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"  acados script failed (code {result.returncode})")
                print(result.stderr[:500])
        else:
            print(f"  acados script not found: {script}")


def extract_arrays(data):
    """Extract numpy arrays from JSON data dict."""
    stages = data['stages']
    N = data['N']
    NX = data['NX']
    NU = data['NU']
    NC = data['NC']

    X = np.array([s['x'] for s in stages])                         # (N+1, NX)
    U = np.array([s['u'] if s['u'] else [np.nan]*NU for s in stages])  # (N+1, NU)
    costs = np.array([s['cost'] for s in stages])                  # (N+1,)

    G = None
    if all(s.get('g') for s in stages):
        G = np.array([s['g'] for s in stages])                     # (N+1, NC)

    S = None
    if all(s.get('s') and len(s['s']) > 0 for s in stages):
        S = np.array([s['s'] for s in stages])                     # (N+1, NC)

    Lam = None
    if all(s.get('lambda') and len(s['lambda']) > 0 for s in stages):
        Lam = np.array([s['lambda'] for s in stages])              # (N+1, NC)

    return X, U, costs, G, S, Lam


def plot_comparison(cipm, acad, problem_name, cfg):
    """Create multi-panel comparison plot."""
    N = cipm['N'] if cipm else acad['N']
    NX = (cipm or acad)['NX']
    NU = (cipm or acad)['NU']
    NC = (cipm or acad)['NC']
    k_arr = np.arange(N + 1)

    cX, cU, cCost, cG, cS, cLam = extract_arrays(cipm) if cipm else (None,)*6
    aX, aU, aCost, aG, aS, aLam = extract_arrays(acad) if acad else (None,)*6

    state_labels = cfg.get('state_labels', [f'x[{i}]' for i in range(NX)])
    ctrl_labels = cfg.get('ctrl_labels', [f'u[{i}]' for i in range(NU)])
    cons_labels = cfg.get('cons_labels', [f'g[{i}]' for i in range(NC)])

    # Count panels
    n_panels = NX + NU + 4  # states + controls + cost_bar + cumcost + constr
    has_slack = cipm and cS is not None
    if has_slack:
        n_panels += 2

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels), sharex=False)
    fig.suptitle(f'{problem_name}: ContactIPM vs acados', fontsize=14, fontweight='bold')

    pidx = 0

    # ── 1. State trajectories ──
    for i in range(NX):
        ax = axes[pidx]; pidx += 1
        if cX is not None:
            ax.plot(k_arr, cX[:, i], 'b-o', markersize=3, label='ContactIPM')
        if aX is not None:
            ax.plot(k_arr, aX[:, i], 'r--s', markersize=3, label='acados')
        ax.set_ylabel(state_labels[i] if i < len(state_labels) else f'x[{i}]')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    # ── 2. Control trajectories ──
    for i in range(NU):
        ax = axes[pidx]; pidx += 1
        if cU is not None:
            ax.plot(k_arr, cU[:, i], 'b-o', markersize=3, label='ContactIPM')
        if aU is not None:
            ax.plot(k_arr, aU[:, i], 'r--s', markersize=3, label='acados')
        ax.set_ylabel(ctrl_labels[i] if i < len(ctrl_labels) else f'u[{i}]')
        ax.grid(True, alpha=0.3)

    # ── 3. Per-stage cost (bar chart) ──
    ax = axes[pidx]; pidx += 1
    bar_w = 0.35
    ks = np.arange(N + 1)
    if cCost is not None:
        ax.bar(ks - bar_w/2, cCost, bar_w, color='steelblue', alpha=0.8, label='ContactIPM')
    if aCost is not None:
        ax.bar(ks + bar_w/2, aCost, bar_w, color='salmon', alpha=0.8, label='acados')
    ax.set_ylabel('Stage cost')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 4. Cumulative cost ──
    ax = axes[pidx]; pidx += 1
    if cCost is not None:
        ax.plot(k_arr, np.cumsum(cCost), 'b-o', markersize=3, label='ContactIPM')
    if aCost is not None:
        ax.plot(k_arr, np.cumsum(aCost), 'r--s', markersize=3, label='acados')
    ax.set_ylabel('Cumulative cost')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 5. Constraint values ──
    ax = axes[pidx]; pidx += 1
    if cG is not None:
        for i in range(NC):
            lbl = cons_labels[i] if i < len(cons_labels) else f'g[{i}]'
            ax.plot(k_arr, cG[:, i], '-', alpha=0.7, label=f'CIPM {lbl}')
    if aG is not None:
        for i in range(NC):
            lbl = cons_labels[i] if i < len(cons_labels) else f'g[{i}]'
            ax.plot(k_arr, aG[:, i], '--', alpha=0.7, label=f'acados {lbl}')
    ax.axhline(0, color='k', lw=0.5, ls='-')
    ax.set_ylabel('g(x,u)')
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── 6. ContactIPM slacks & multipliers ──
    if has_slack:
        ax = axes[pidx]; pidx += 1
        for i in range(NC):
            lbl = cons_labels[i] if i < len(cons_labels) else f's[{i}]'
            ax.plot(k_arr, cS[:, i], '-o', markersize=2, alpha=0.7, label=lbl)
        ax.set_ylabel('Slacks s (CIPM)')
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[pidx]; pidx += 1
        for i in range(NC):
            lbl = cons_labels[i] if i < len(cons_labels) else f'λ[{i}]'
            ax.plot(k_arr, cLam[:, i], '-o', markersize=2, alpha=0.7, label=lbl)
        ax.set_ylabel(r'$\lambda$ (CIPM)')
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Stage k')

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    figpath = os.path.join(FIG_DIR, f'{problem_name}_comparison.png')
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {figpath}")
    plt.close(fig)


def print_analysis(cipm, acad, problem_name):
    """Print per-stage cost table, constraint violation summary."""
    N = (cipm or acad)['N']

    print(f"\n{'='*70}")
    print(f"  ANALYSIS: {problem_name}")
    print(f"{'='*70}")

    # Per-stage cost table
    c_costs = [s['cost'] for s in cipm['stages']] if cipm else None
    a_costs = [s['cost'] for s in acad['stages']] if acad else None

    print(f"\n{'k':>3} | {'CIPM cost':>12} | {'acados cost':>12} | {'diff':>12}")
    print("-" * 50)
    for k in range(N + 1):
        cc = c_costs[k] if c_costs else float('nan')
        ac = a_costs[k] if a_costs else float('nan')
        diff = cc - ac if (c_costs and a_costs) else float('nan')
        print(f"{k:3d} | {cc:12.6f} | {ac:12.6f} | {diff:12.6f}")

    if c_costs and a_costs:
        c_run = sum(c_costs[:N]); c_term = c_costs[N]
        a_run = sum(a_costs[:N]); a_term = a_costs[N]
        print("-" * 50)
        print(f"{'Run':>3} | {c_run:12.6f} | {a_run:12.6f} | {c_run - a_run:12.6f}")
        print(f"{'Term':>3} | {c_term:12.6f} | {a_term:12.6f} | {c_term - a_term:12.6f}")
        c_tot = c_run + c_term; a_tot = a_run + a_term
        print(f"{'Tot':>3} | {c_tot:12.6f} | {a_tot:12.6f} | {c_tot - a_tot:12.6f}")

    # Constraint violation summary
    for solver_name, data in [('ContactIPM', cipm), ('acados', acad)]:
        if not data:
            continue
        print(f"\n── {solver_name} constraint violations ──")
        NC = data['NC']
        has_slack = all(s.get('s') and len(s['s']) > 0 for s in data['stages'])

        for ci in range(NC):
            max_g = -1e20
            max_stage = -1
            n_active = 0
            for k in range(N + 1):
                g = data['stages'][k]['g']
                if ci >= len(g):
                    continue
                gk = g[ci]
                if gk < -1e8:
                    continue
                if has_slack:
                    sk = data['stages'][k]['s'][ci]
                    # feasibility: g+s should be >= 0, violation = -(g+s)
                    eff = gk + sk
                else:
                    eff = gk
                if eff > max_g:
                    max_g = eff
                    max_stage = k
                if gk > -0.01:
                    n_active += 1

            if max_g > -1e8:
                status = 'ACTIVE' if max_g > -0.01 else 'satisfied'
                print(f"  g[{ci:2d}]: max={max_g:+.4e} (stage {max_stage:2d}), "
                      f"active_count={n_active:2d}  [{status}]")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <problem|all>")
        print(f"  Problems: {', '.join(PROBLEMS.keys())}")
        sys.exit(1)

    target = sys.argv[1]
    problems = list(PROBLEMS.keys()) if target == 'all' else [target]

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    for prob in problems:
        if prob not in PROBLEMS:
            print(f"Unknown problem: {prob}")
            continue

        print(f"\n{'#'*60}")
        print(f"# {prob}")
        print(f"{'#'*60}")

        cipm, acad = load_json(prob)
        if cipm is None or acad is None:
            print("  JSON not found, attempting to generate...")
            generate_json(prob)
            cipm, acad = load_json(prob)

        if cipm is None and acad is None:
            print(f"  No data for {prob}, skipping.")
            continue

        if cipm and acad:
            plot_comparison(cipm, acad, prob, PROBLEMS[prob])
        print_analysis(cipm, acad, prob)


if __name__ == '__main__':
    main()
