#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /absolute/path/to/acados" >&2
    exit 2
fi

acados_source="$(realpath "$1")"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(realpath "$script_dir/..")"
mkdir -p "$repo_root/tmp"
output_root="$(mktemp -d "$repo_root/tmp/benchmark_verification.XXXXXX")"
python_acados="${PYTHON_ACADOS:-python3.11}"

required=(
    "$repo_root/build-contact/contact_cartpole_soft_walls"
    "$repo_root/build-contact/contact_push_box"
    "$repo_root/build-contact/contact_transport"
    "$repo_root/build-contact/contact_push_t"
    "$repo_root/build-crisp/examples/pushbot_example"
    "$repo_root/build-crisp/examples/pushbox_example"
    "$repo_root/build-crisp/examples/cartTransp_example"
    "$repo_root/build-crisp/examples/pushT_example"
    "$repo_root/build-impact-contact/contact_impact_push_box"
    "$repo_root/build-impact-contact/contact_impact_push_t"
    "$repo_root/build-impact-contact/contact_impact_cart_transport"
)
for path in "${required[@]}"; do
    if [[ ! -x "$path" ]]; then
        echo "missing executable: $path" >&2
        echo "build the benchmark suites from README.md first" >&2
        exit 1
    fi
done

mkdir -p "$output_root"
export ACADOS_SOURCE_DIR="$acados_source"
export CONTACTIPM_IMPACT_CASADI_LIBRARY="$repo_root/.deps/casadi-3.7.2/casadi"
export PYTHONPATH="$acados_source/interfaces/acados_template:$repo_root/.deps/casadi-3.7.2:$repo_root/.deps/acados-py311${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$acados_source/lib:$CONTACTIPM_IMPACT_CASADI_LIBRARY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$repo_root"

python3 benchmarks/contact_ipm/verify_source_parity.py
python3 -m unittest discover -s benchmarks/contact_ipm -p 'test_*.py'
python3 -m unittest discover -s benchmarks/impact_comparison -p 'test_*.py'
cmake --build build-contact -j2 --target \
    test_riccati test_filter_ls test_hessian_approx test_barrier_manager \
    test_kkt_diag test_globalization test_precond_invariance \
    test_mpcc_relaxation contact_push_box_closed_loop
(cd build-contact && ctest --output-on-failure)

python3 benchmarks/contact_ipm/run_benchmarks.py \
    --contactipm-build build-contact \
    --crisp-build build-crisp \
    --problems cartpole_soft_walls push_box transport push_t \
    --push-t-segment 8 --warmups 0 --repetitions 1 \
    --threads 1 --cpu-affinity auto --seed 2027 \
    --output "$output_root/crisp_smoke.json"

"$python_acados" benchmarks/impact_comparison/run_impact_comparison.py \
    --contact-build build-impact-contact \
    --impact-build build-impact \
    --impact-repository benchmarks/IMPACT \
    --case-limit 1 --warmups 0 --repetitions 1 \
    --threads 1 --cpu 0 --seed 2027 \
    --output "$output_root/impact_smoke.json"

"$python_acados" benchmarks/acados_comparison/run_acados_benchmarks.py \
    --suite crisp --mode robustness --case-limit 1 --cpu 0 \
    --output "$output_root/acados_crisp_smoke.json"
"$python_acados" benchmarks/acados_comparison/run_acados_benchmarks.py \
    --suite impact --mode robustness --case-limit 1 --cpu 0 \
    --output "$output_root/acados_impact_smoke.json"

python3 benchmarks/impact_comparison/verify_comparison.py \
    --impact-repository benchmarks/IMPACT \
    --robustness benchmarks/impact_comparison/results/2026-07-29_robustness_50.json \
    --timing benchmarks/impact_comparison/results/2026-07-29_timing_20x.json

"$python_acados" -c \
    "import json, pathlib; [json.load(p.open()) for p in pathlib.Path('benchmarks/acados_comparison/results').glob('*.json')]"

echo "All ContactIPM, CRISP, IMPACT, and acados smoke checks passed."
