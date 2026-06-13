"""
Master benchmark runner: ContactIPM vs acados SQP+HPIPM

Runs all 3 benchmarks for BOTH solvers (C/C++ executables), parses output,
and prints comparison table.  All results are fresh — no hardcoded numbers.
"""
import subprocess
import sys
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, 'bin')
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
CONTACTIPM_BIN = os.path.join(PROJECT_ROOT, 'build', 'Release')

# acados C benchmarks (compiled main_*.exe in bin/)
ACADOS_BENCHMARKS = [
    ('Pendulum',   'main_pendulum.exe'),
    ('Quadrotor',  'main_quadrotor_2d.exe'),
    ('Chain Mass', 'main_chain_mass.exe'),
]

# ContactIPM C++ examples (build/Release/)
CONTACTIPM_BENCHMARKS = [
    ('Pendulum',   'pendulum_nmpc_paper.exe'),
    ('Quadrotor',  'quadrotor_2d_nmpc.exe'),
    ('Chain Mass', 'chain_mass_nmpc.exe'),
]

N_RUNS = 5  # min-of-N for timing stability


def parse_acados_output(output):
    """Parse acados C benchmark output (=== BENCHMARK SUMMARY === block)."""
    result = {}
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('Status:'):
            result['status'] = line.split(':',1)[1].strip()
        elif line.startswith('SQP iterations:'):
            m = re.search(r':\s*(\d+)', line)
            if m: result['iters'] = m.group(1)
        elif line.startswith('Solve time:'):
            m = re.search(r'([\d.]+)', line.split(':',1)[1])
            if m: result['time_ms'] = m.group(1)
        elif line.startswith('Total:'):
            m = re.search(r'([\d.eE+\-]+)', line.split(':',1)[1])
            if m: result['cost'] = m.group(1)
        elif line.startswith('Max cons viol:'):
            m = re.search(r'([\d.eE+\-]+)', line.split(':',1)[1])
            if m: result['max_viol'] = m.group(1)
        elif line.startswith('First u*'):
            m = re.search(r'\[(.+)\]', line)
            if m: result['u0'] = m.group(1)
    return result


def parse_contactipm_output(output):
    """Parse ContactIPM C++ example output (first === SOLVE COMPLETE === block)."""
    result = {}
    in_block = False
    for line in output.split('\n'):
        line = line.strip()
        if '=== SOLVE COMPLETE ===' in line:
            if in_block:
                break  # stop at second block
            in_block = True
            continue
        if not in_block:
            continue
        if line.startswith('Status:'):
            result['status'] = line.split(':',1)[1].strip()
        elif line.startswith('Iterations:'):
            m = re.search(r'(\d+)', line.split(':',1)[1])
            if m: result['iters'] = m.group(1)
        elif line.startswith('Primal inf:'):
            m = re.search(r'([\d.eE+\-]+)', line.split(':',1)[1])
            if m: result['prim_inf'] = m.group(1)
        elif line.startswith('Cost:'):
            m = re.search(r'([\d.eE+\-]+)', line.split(':',1)[1])
            if m:
                val = float(m.group(1))
                if val > 0:  # skip Cost: 0.0000 from second block
                    result['cost'] = m.group(1)
        elif line.startswith('First u*'):
            m = re.search(r'\[(.+)\]', line)
            if m: result['u0'] = m.group(1)
    # Also grab Total (solver) for per-node cost
    m = re.search(r'Total \(solver\):\s+([\d.]+)', output)
    if m:
        result['cost'] = m.group(1)
    return result


def run_exe(exe_path, extra_paths=None):
    """Run an executable and return (stdout+stderr, returncode)."""
    env = os.environ.copy()
    for p in ([BIN_DIR] + (extra_paths or [])):
        if os.path.isdir(p):
            env['PATH'] = p + os.pathsep + env.get('PATH', '')
    proc = subprocess.run(
        [exe_path], timeout=600, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out = proc.stdout.decode('utf-8', errors='replace')
    err = proc.stderr.decode('utf-8', errors='replace')
    return out + '\n' + err, proc.returncode


def run_acados_benchmark(name, exe_name):
    """Run one acados C benchmark (min of N_RUNS for timing)."""
    exe_path = os.path.join(BIN_DIR, exe_name)
    print(f"\n{'─' * 60}")
    print(f"  [acados] {name}  ({exe_name})")
    print(f"{'─' * 60}")

    if not os.path.isfile(exe_path):
        print(f"  ERROR: executable not found: {exe_path}")
        return None

    acados_lib = r'C:\Users\cheny\Documents\GitHub\acados\lib'
    result = None
    for i in range(N_RUNS):
        combined, rc = run_exe(exe_path, [acados_lib])
        parsed = parse_acados_output(combined)
        if parsed:
            if result is None:
                result = parsed
            elif 'time_ms' in parsed:
                # keep min time
                if float(parsed['time_ms']) < float(result.get('time_ms', '9999')):
                    result['time_ms'] = parsed['time_ms']

    if result and '=== BENCHMARK SUMMARY ===' in combined:
        idx = combined.index('=== BENCHMARK SUMMARY ===')
        print(combined[idx:])
    elif result:
        for k, v in result.items():
            print(f"  {k}: {v}")
    return result


def run_contactipm_benchmark(name, exe_name):
    """Run one ContactIPM C++ benchmark (min of N_RUNS for timing)."""
    exe_path = os.path.join(CONTACTIPM_BIN, exe_name)
    print(f"\n{'─' * 60}")
    print(f"  [ContactIPM] {name}  ({exe_name})")
    print(f"{'─' * 60}")

    if not os.path.isfile(exe_path):
        print(f"  ERROR: executable not found: {exe_path}")
        return None

    result = None
    for i in range(N_RUNS):
        combined, rc = run_exe(exe_path)
        parsed = parse_contactipm_output(combined)
        # Extract Solve time from second SOLVE COMPLETE block
        m = re.search(r'Solve time:\s+([\d.]+)\s+ms', combined)
        if m:
            parsed['time_ms'] = m.group(1)
        if parsed:
            if result is None:
                result = parsed
            elif 'time_ms' in parsed:
                if float(parsed['time_ms']) < float(result.get('time_ms', '9999')):
                    result['time_ms'] = parsed['time_ms']

    if result:
        for k, v in result.items():
            print(f"  {k}: {v}")
    return result


def print_table(cipm_results, acados_results):
    """Print final comparison table."""
    print("\n\n")
    print("=" * 82)
    print("  BENCHMARK COMPARISON: ContactIPM (Mehrotra IPM) vs acados (SQP+HPIPM)")
    print("=" * 82)
    print()

    for name in ['Pendulum', 'Quadrotor', 'Chain Mass']:
        cipm = cipm_results.get(name, {})
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
    print(f"  Notes:")
    print(f"  - ContactIPM: Mehrotra predictor-corrector IPM + Riccati KKT (C++, min of {N_RUNS} runs)")
    print(f"  - acados: SQP + HPIPM via C API (main_*.exe, min of {N_RUNS} runs)")
    print("  - Both solvers use identical x0, cost weights, constraints, and tolerances")
    print("  - Single-epoch solve (no closed-loop simulation)")
    print("  - All results fresh from executables — no hardcoded numbers")
    print("=" * 82)


def main():
    print("=" * 60)
    print("  ContactIPM vs acados — Full Benchmark Suite (all fresh)")
    print("=" * 60)

    # ── Run ContactIPM benchmarks ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  ContactIPM benchmarks")
    print("=" * 60)
    cipm_results = {}
    for name, exe in CONTACTIPM_BENCHMARKS:
        result = run_contactipm_benchmark(name, exe)
        cipm_results[name] = result if result else {}

    # ── Run acados benchmarks ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  acados benchmarks")
    print("=" * 60)
    acados_results = {}
    for name, exe in ACADOS_BENCHMARKS:
        result = run_acados_benchmark(name, exe)
        acados_results[name] = result if result else {}

    print_table(cipm_results, acados_results)


if __name__ == '__main__':
    main()
