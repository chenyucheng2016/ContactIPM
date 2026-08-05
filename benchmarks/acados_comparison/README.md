# acados contact benchmark

This directory runs acados 0.5.1 on the frozen CRISP-parameter and
IMPACT-parameter contact tasks. The two suites remain separate because their
physical parameters are different.

The exact source is `https://github.com/acados/acados.git` at commit
`dc6668f8538e49831f03a36ddf755de6e92d38ea`; see `acados_manifest.json`.

The baseline uses full SQP with `PARTIAL_CONDENSING_HPIPM`. Complementarity is
encoded exactly as `a >= 0`, `b >= 0`, and `a*b <= 0`; no relaxation is used.
A trial is eligible only when acados returns status 0 and the independent
physical audit passes.

## Build

Clone and build the pinned acados revision in WSL:

```bash
git clone https://github.com/acados/acados.git ../acados
git -C ../acados checkout dc6668f8538e49831f03a36ddf755de6e92d38ea
git -C ../acados submodule update --init --recursive
cmake -S ../acados -B ../acados/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../acados/build -j2
cmake --install ../acados/build
python3.11 -m pip install --target "$PWD/.deps/acados-py311" Deprecated==1.2.18
python3.11 -m pip install --target "$PWD/.deps/casadi-3.7.2" \
  --no-deps casadi==3.7.2
```

Generate the seven solvers with Windows Python, whose native
`t_renderer.exe` is included in the checkout:

```powershell
$env:ACADOS_SOURCE_DIR = (Resolve-Path ..\acados).Path
python -m pip install casadi==3.7.2 Deprecated==1.2.18
$env:PYTHONPATH = "$env:ACADOS_SOURCE_DIR\interfaces\acados_template"
$cases = @(
  "crisp cartpole_soft_walls", "crisp push_box", "crisp transport",
  "crisp push_t", "impact push_box", "impact push_t",
  "impact cart_transport"
)
foreach ($case in $cases) {
  $suite, $problem = $case.Split(" ")
  python benchmarks/acados_comparison/acados_contact_benchmarks.py generate `
    --suite $suite --problem $problem
}
```

Compile those generated sources against the WSL acados libraries:

```bash
bash benchmarks/acados_comparison/build_generated_wsl.sh \
  "$(realpath ../acados)"
```

## Run

In WSL, set the runtime paths and run each suite:

```bash
export ACADOS_SOURCE_DIR="$(realpath ../acados)"
export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:$PWD/.deps/casadi-3.7.2:$PWD/.deps/acados-py311"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib"

python3.11 benchmarks/acados_comparison/run_acados_benchmarks.py \
  --suite crisp --mode robustness --cpu 0 \
  --output benchmarks/acados_comparison/results/crisp_robustness.json
python3.11 benchmarks/acados_comparison/run_acados_benchmarks.py \
  --suite impact --mode robustness --cpu 0 \
  --output benchmarks/acados_comparison/results/impact_robustness.json

python3.11 benchmarks/acados_comparison/run_acados_benchmarks.py \
  --suite crisp --mode timing --warmups 1 --repetitions 20 --cpu 0 \
  --output benchmarks/acados_comparison/results/crisp_timing_20x.json
python3.11 benchmarks/acados_comparison/run_acados_benchmarks.py \
  --suite impact --mode timing --warmups 1 --repetitions 20 --cpu 0 \
  --output benchmarks/acados_comparison/results/impact_timing_20x.json
```

Reported solve time is acados `time_tot`; source generation, compilation,
solver construction, process startup, and trajectory auditing are excluded.

Once the ContactIPM, CRISP, and IMPACT builds from the top-level README also
exist, run every comparison smoke test and audit with:

```bash
bash benchmarks/verify_all_benchmark_smoke_wsl.sh \
  "$(realpath ../acados)"
```
