"""
Master benchmark runner: ContactIPM vs acados SQP+HPIPM

Runs all 3 acados benchmarks, parses output, and prints comparison table.
"""
import subprocess
import sys
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = [
    ('Pendulum',   'acados_pendulum.py'),
    ('Quadrotor',  'acados_quadrotor.py'),
    ('Chain Mass', 'acados_chain_mass.py'),
]

# ContactIPM results (from running the C++ examples — update after timing)
CONTACTIPM_RESULTS = {
    'Pendulum':   {'iters': '4',  'cost': '7.36', 'prim_inf': '6.76e-09', 'status': 'Success', 'time_ms': '0.57', 'u0': '15.112'},
    'Quadrotor':  {'iters': '81', 'cost': '23.56', 'prim_inf': '1.96e-01', 'status': 'Success', 'time_ms': '0.59', 'u0': 'see C++'},
    'Chain Mass': {'iters': '24', 'cost': '388.84', 'prim_inf': '9.99e-04', 'status': 'Success', 'time_ms': '4.28', 'u0': 'see C++'},
}


def parse_output(output):
    """Parse acados benchmark output into a dict."""
    result = {}
    for line in output.split('\n'):
        line = line.strip()
        if 'SQP iterations:' in line:
            m = re.search(r':\s*(\d+)', line)
            if m: result['iters'] = m.group(1)
        elif line.startswith('Solve time:'):
            m = re.search(r'([\d.]+)', line.split(':')[1])
            if m: result['time_ms'] = m.group(1)
        elif line.startswith('Cost:'):
            m = re.search(r'([\d.eE+\-]+)', line.split(':')[1])
            if m: result['cost'] = m.group(1)
        elif 'Max cons viol:' in line:
            m = re.search(r'([\d.eE+\-]+)', line.split(':')[1])
            if m: result['max_viol'] = m.group(1)
        elif 'First u*' in line:
            m = re.search(r'\[(.+)\]', line)
            if m: result['u0'] = m.group(1)
        elif line.startswith('Status:'):
            result['status'] = line.split(':')[1].strip()
    return result


def run_benchmark(name, script):
    """Run one benchmark script and return parsed results."""
    script_path = os.path.join(SCRIPT_DIR, script)
    print(f"\n{'─' * 60}")
    print(f"  Running: {name}")
    print(f"{'─' * 60}")

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=600
        )
        # Combine stdout + stderr (acados SQP log goes to stderr)
        combined = (proc.stdout or '') + '\n' + (proc.stderr or '')
        print(proc.stdout or '')
        if proc.returncode != 0 and 'SOLVE COMPLETE' not in combined:
            print(f"STDERR:\n{(proc.stderr or '')[:1000]}")
            return None
        return parse_output(combined)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def print_table(acados_results):
    """Print final comparison table."""
    print("\n\n")
    print("=" * 82)
    print("  BENCHMARK COMPARISON: ContactIPM (Mehrotra IPM) vs acados (SQP+HPIPM)")
    print("=" * 82)
    print()

    for name in ['Pendulum', 'Quadrotor', 'Chain Mass']:
        cipm = CONTACTIPM_RESULTS.get(name, {})
        acad = acados_results.get(name, {})

        print(f"  ┌─ {name} {'─' * (60 - len(name))}")
        print(f"  │ {'Metric':<16} {'ContactIPM':<18} {'acados':<20}")
        print(f"  │ {'─'*16} {'─'*18} {'─'*20}")

        metrics = [
            ('Status',       'status',   'status'),
            ('Iterations',   'iters',    'iters'),
            ('Cost',         'cost',     'cost'),
            ('Solve (ms)',   'time_ms',  'time_ms'),
            ('Cons viol',    'prim_inf', 'max_viol'),
            ('First u*',     'u0',       'u0'),
        ]

        for label, cipm_key, acad_key in metrics:
            c = str(cipm.get(cipm_key, '—'))
            a = str(acad.get(acad_key, '—'))
            print(f"  │ {label:<16} {c:<18} {a:<20}")
        print(f"  └{'─'*66}")
        print()

    print("=" * 82)
    print("  Notes:")
    print("  - ContactIPM: Mehrotra predictor-corrector IPM + Riccati KKT (single-pass)")
    print("  - acados: SQP outer loop + HPIPM QP solver + ERK4 integrator")
    print("  - Both solvers use identical x0, cost weights, constraints, and tolerances")
    print("  - Initial guess: x0 propagated (pendulum/chain), RK4 forward sim (quadrotor)")
    print("  - Single-epoch solve (no averaging)")
    print("=" * 82)


def main():
    print("=" * 60)
    print("  ContactIPM vs acados — Full Benchmark Suite")
    print("=" * 60)

    acados_results = {}
    for name, script in BENCHMARKS:
        result = run_benchmark(name, script)
        acados_results[name] = result if result else {}

    print_table(acados_results)


if __name__ == '__main__':
    main()
