# Reproducing the competitor comparisons

This guide reproduces the ContactIPM comparisons against the pinned CRISP and
IMPACT repositories. Run the commands from the ContactIPM repository root in
Ubuntu or WSL. The experiments reported in this repository used Ubuntu 20.04,
GCC 9.4, CMake 3.16, Python 3.8, one solver thread, and locally measured wall
time.

No execution time quoted by either competitor paper is imported into our
tables. Exact wall times are machine-dependent; the frozen instances, solver
outputs, independent audits, and paired timing protocol are reproducible.

## 1. Obtain the pinned sources

Download and extract the anonymized repository archive from the URL provided
in the paper, then enter its top-level directory. Because the anonymous archive
does not include Git submodule contents, obtain the released competitor sources
at the pinned revisions:

```bash
git clone https://github.com/ComputationalRobotics/CRISP.git benchmarks/CRISP
git -C benchmarks/CRISP checkout d429c06e02b77bba33fbdc2da91d980d3088257b
git clone https://github.com/JonasPflaume/IMPACT.git benchmarks/IMPACT
git -C benchmarks/IMPACT checkout f56f1403659122bd4017ed36b21d83a490bc6687
```

For a Git clone with submodules configured, the equivalent command is:

```bash
git submodule update --init --recursive
```

The expected competitor revisions are:

- CRISP: `d429c06e02b77bba33fbdc2da91d980d3088257b`
- IMPACT: `f56f1403659122bd4017ed36b21d83a490bc6687`

## 2. Common prerequisites

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config python3 python3-pip \
  libboost-all-dev libyaml-cpp-dev pybind11-dev
```

The successful comparison environment used Eigen 5.0.1, which supplies the
BLAS and LAPACK compatibility libraries needed by IMPACT:

```bash
git clone --branch 5.0.1 --depth 1 \
  https://gitlab.com/libeigen/eigen.git /tmp/eigen-5.0.1
cmake -S /tmp/eigen-5.0.1 -B /tmp/eigen-5.0.1-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN_BUILD_BLAS=ON \
  -DEIGEN_BUILD_LAPACK=ON
cmake --build /tmp/eigen-5.0.1-build -j2
sudo cmake --install /tmp/eigen-5.0.1-build
sudo ldconfig
```

After installation, verify:

```bash
test -f /usr/local/lib/libeigen_blas.so
test -f /usr/local/lib/libeigen_lapack.so
pkg-config --modversion eigen3
```

## 3. Build and test ContactIPM

```bash
cmake -S . -B build-contact \
  -DCMAKE_BUILD_TYPE=Release \
  -DNMPC_BUILD_EXAMPLES=OFF \
  -DNMPC_BUILD_TESTS=ON \
  -DNMPC_BUILD_CONTACT_BENCHMARKS=ON
cmake --build build-contact -j2
(cd build-contact && ctest --output-on-failure)
python3 -m unittest discover -s benchmarks/contact_ipm -p 'test_*.py'
python3 -m unittest discover -s benchmarks/impact_comparison -p 'test_*.py'
```

The subshell form for CTest is intentional: CTest 3.16 does not support the
newer `ctest --test-dir` option.

## 4. Reproduce ContactIPM versus CRISP

CRISP additionally requires CppAD, CppADCodeGen, and PIQP 0.5.0. Follow the
installation section in `benchmarks/CRISP/ReadMe.md`; the environment used for
the checked-in results had CppAD `20240000.7` and PIQP `v0.5.0`.

Apply the tracked, idempotent benchmark instrumentation before configuring
CRISP:

```bash
python3 benchmarks/contact_ipm/instrument_crisp.py \
  --crisp-root benchmarks/CRISP
python3 benchmarks/contact_ipm/verify_source_parity.py

cmake -S benchmarks/CRISP/src -B build-crisp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-crisp -j2 --target \
  pushbot_example pushbox_example cartTransp_example pushT_example
```

The CRISP submodule appearing modified after instrumentation is expected. The
script adds deterministic trajectory export and controlled benchmark inputs;
it does not change CRISP solver settings.

Run a quick exact-instance smoke comparison:

```bash
python3 benchmarks/contact_ipm/run_benchmarks.py \
  --contactipm-build build-contact \
  --crisp-build build-crisp \
  --problems cartpole_soft_walls push_box transport \
  --warmups 0 --repetitions 1 --threads 1 \
  --cpu-affinity auto --seed 2027 \
  --output crisp_smoke.json
python3 benchmarks/contact_ipm/summarize_benchmarks.py crisp_smoke.json
```

Run the publication timing protocol for the three single-instance problems:

```bash
python3 benchmarks/contact_ipm/run_benchmarks.py \
  --contactipm-build build-contact \
  --crisp-build build-crisp \
  --problems cartpole_soft_walls push_box transport \
  --warmups 1 --repetitions 20 --threads 1 \
  --cpu-affinity auto --seed 2027 \
  --output crisp_timing_fast_20x.json
python3 benchmarks/contact_ipm/summarize_benchmarks.py \
  crisp_timing_fast_20x.json
```

Run the predeclared Push T timing instance separately:

```bash
python3 benchmarks/contact_ipm/run_benchmarks.py \
  --contactipm-build build-contact \
  --crisp-build build-crisp \
  --problems push_t --push-t-segment 8 \
  --warmups 1 --repetitions 20 --threads 1 \
  --cpu-affinity auto --seed 2027 \
  --output crisp_timing_push_t_seg08_20x.json
python3 benchmarks/contact_ipm/summarize_benchmarks.py \
  crisp_timing_push_t_seg08_20x.json
```

The 55-case robustness sweep, Push Box ablation, matched tracking-effort
study, and duration-fixed scaling commands are documented in
`benchmarks/contact_ipm/README.md`.

## 5. Reproduce ContactIPM versus IMPACT

IMPACT is built from the pinned submodule without source changes. Install the
same CasADi wheel used by the checked-in comparison into a repository-local
directory:

```bash
python3 -m pip install --target "$PWD/.deps/casadi-3.7.2" \
  --no-deps casadi==3.7.2
export CONTACTIPM_IMPACT_CASADI_LIBRARY="$PWD/.deps/casadi-3.7.2/casadi"
```

Build the ContactIPM adapters:

```bash
cmake -S benchmarks/impact_comparison/contactipm_build \
  -B build-impact-contact \
  -DCMAKE_BUILD_TYPE=Release \
  -DCONTACTIPM_SOURCE_DIR="$PWD" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -ffast-math"
cmake --build build-impact-contact -j2
```

Build IMPACT through the compatibility overlay:

```bash
cmake -S benchmarks/impact_comparison/impact_build \
  -B build-impact \
  -DCMAKE_BUILD_TYPE=Release \
  -DIMPACT_SOURCE_DIR="$PWD/benchmarks/IMPACT" \
  -DEIGEN_BLAS_LIBRARY=/usr/local/lib/libeigen_blas.so \
  -DEIGEN_LAPACK_LIBRARY=/usr/local/lib/libeigen_lapack.so \
  -Dcasadi_DIR="$CONTACTIPM_IMPACT_CASADI_LIBRARY/cmake"
cmake --build build-impact -j2
```

Run a one-case-per-problem smoke comparison:

```bash
python3 benchmarks/impact_comparison/run_impact_comparison.py \
  --contact-build build-impact-contact \
  --impact-build build-impact \
  --impact-repository benchmarks/IMPACT \
  --case-limit 1 --warmups 0 --repetitions 1 \
  --threads 1 --seed 2027 \
  --output impact_smoke.json
```

Run the deterministic 50-case robustness suite:

```bash
python3 benchmarks/impact_comparison/run_impact_comparison.py \
  --contact-build build-impact-contact \
  --impact-build build-impact \
  --impact-repository benchmarks/IMPACT \
  --warmups 0 --repetitions 1 \
  --threads 1 --seed 2027 \
  --output impact_robustness_50.json
```

Run controlled timing on the official case-zero instance of all three common
problems. Replace CPU `0` if that CPU is unavailable in your affinity mask:

```bash
python3 benchmarks/impact_comparison/run_impact_comparison.py \
  --contact-build build-impact-contact \
  --impact-build build-impact \
  --impact-repository benchmarks/IMPACT \
  --case-limit 1 --warmups 1 --repetitions 20 \
  --threads 1 --cpu 0 --seed 2027 \
  --output impact_timing_20x.json
```

Generate the side-by-side tables and verify the provenance and fairness gates:

```bash
python3 benchmarks/impact_comparison/summarize_impact_comparison.py \
  --robustness impact_robustness_50.json \
  --timing impact_timing_20x.json \
  --output-prefix contactipm_vs_impact
python3 benchmarks/impact_comparison/verify_comparison.py \
  --impact-repository benchmarks/IMPACT \
  --robustness impact_robustness_50.json \
  --timing impact_timing_20x.json
```

## 6. Native CasADi benchmark timing

The Python ContactIPM adapter is useful for development, but its solve timer
includes Python/CasADi callback dispatch. Build generated C callbacks to measure
the solver with the same CasADi problem definitions and derivatives:

```bash
cmake -S . -B build-casadi-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DNMPC_BUILD_EXAMPLES=OFF \
  -DNMPC_BUILD_TESTS=OFF \
  -DNMPC_BUILD_CASADI_NATIVE_BENCHMARKS=ON
cmake --build build-casadi-native -j2
```

Each executable defaults to one warmup and 20 measured solves. For example:

```bash
build-casadi-native/casadi_native_cart_pendulum
build-casadi-native/casadi_native_quadcopter
build-casadi-native/casadi_native_quadcopter_tracking
build-casadi-native/casadi_native_hanging_chain_2d
build-casadi-native/casadi_native_hanging_chain_3d
```

The reported interval covers `ContactIPM::solve`, including all generated
derivative and globalization evaluations. CasADi code generation, compilation,
problem construction, warmups, and result reporting are outside that interval.

## 7. Checked-in results and raw data

Publication-facing artifacts are tracked under:

- `benchmarks/contact_ipm/results/`
- `benchmarks/impact_comparison/results/`

The JSON artifacts contain commands, revisions, executable/source hashes,
solver output, independent audit metrics, and paired timing observations.
Per-trial raw trajectory directories use the `_raw` suffix and are deliberately
not committed because of their size. Every new runner invocation creates its
own raw directory, allowing the independent audit to be repeated from the
newly generated trajectories.

For exact interpretation of success, read feasibility and task metrics before
the scalar objective. The audits report dynamics defects, side feasibility,
physical complementarity, terminal tracking, effort, force, contact impulse,
and contact-mode changes. A larger objective alone is not treated as evidence
of a worse contact solution.

## 8. Closed-loop Push Box validation

Build the receding-horizon executable with the same Release contact build, then
run the fixed 50-rollout suite on one available CPU:

```bash
cmake --build build-contact -j2 --target contact_push_box_closed_loop
taskset -c 0 ./build-contact/contact_push_box_closed_loop \
  --mode suite --seed 2027 \
  --output closed_loop_push_box.json \
  --trajectory-dir closed_loop_push_box_trajectories
```

Generate the validated summary, the four-panel figure, and two representative
GIFs:

```bash
python3 benchmarks/contact_ipm/summarize_closed_loop.py \
  closed_loop_push_box.json --require-suite
python3 benchmarks/contact_ipm/plot_closed_loop.py \
  closed_loop_push_box.json closed_loop_push_box_trajectories \
  --output-prefix closed_loop_push_box_validation
python3 benchmarks/contact_ipm/animate_closed_loop.py \
  closed_loop_push_box.json closed_loop_push_box_trajectories \
  --output-dir closed_loop_push_box_media
```

The suite contains 5 nominal, 15 initial-pose, 10 mass/friction-mismatch,
10 isolated state-reset diagnostics, and 10 disturbed-motion rollouts. Disturbed
motion uses independent uniform initial x/y offsets in `[-0.075, 0.075]` m and
yaw offset in `[-0.075, 0.075]` rad; independent uniform mass and friction
scale factors in
`[0.85, 1.15]`; and zero-mean Gaussian measurement noise with standard
deviations `0.001` m in x/y and `0.001` rad in yaw. After the 1.5 s control
update, one direct state reset adds independent uniform x/y offsets in
`[-0.1125, 0.1125]` m and a
yaw offset in `[-0.09, 0.09]` rad; it first appears in the recorded state at
1.6 s. All draws use deterministic seed 2027. The suite uses zero
initialization only for the free state/control guesses of the first MPC solve
(stage zero is fixed to
the measurement) and shifted solver state thereafter. Task success requires
terminal tracking for ten consecutive steps, no unrecovered solver failure, and
independently audited dynamics, side-feasibility, and physical-complementarity
tolerances. The reference run is
[`2026-07-29_closed_loop_push_box_50_summary.md`](benchmarks/contact_ipm/results/2026-07-29_closed_loop_push_box_50_summary.md).

The source Push Box model is quasi-static and has no velocity, momentum,
penetration, or friction-cone state. The scheduled reset is therefore a direct
pose/yaw state change rather than an external force or inertial impulse. Deadline
statistics are reported
honestly and do not establish a hard real-time guarantee.

## 9. Reproduce the acados comparison

The acados source, Python interface, and CasADi versions are frozen in
`benchmarks/acados_comparison/acados_manifest.json`. Follow the copy-paste build
and run commands in `README.md#reproduce-the-acados-comparison`. Those commands
generate all seven CRISP- and IMPACT-parameter solvers, compile them natively in
WSL, and rerun the robustness and controlled 20-repeat timing protocols.

After the ContactIPM, CRISP, IMPACT, and acados builds exist, the complete smoke
regression is:

```bash
bash benchmarks/verify_all_benchmark_smoke_wsl.sh \
  "$(realpath ../acados)"
```

It checks ContactIPM CTest, both Python audit suites, CRISP source parity, all
four ContactIPM/CRISP pairs, all three ContactIPM/IMPACT pairs, all seven acados
formulations, IMPACT artifact provenance, and JSON readability of the checked-in
acados results.
