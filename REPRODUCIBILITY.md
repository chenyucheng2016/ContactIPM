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

Historical ContactIPM artifacts use the neutral labels `contactipm-src-01`
through `contactipm-src-17`. Their exact source-only snapshots are included in
`provenance/contactipm-source-snapshots.bundle`; the bundle digest and per-file
SHA-256 manifests are listed in `provenance/source-snapshots.json`. For example:

```bash
git clone provenance/contactipm-source-snapshots.bundle /tmp/contactipm-sources
git -C /tmp/contactipm-sources checkout contactipm-src-02
(cd /tmp/contactipm-sources && sha256sum -c \
  "$OLDPWD/provenance/manifests/contactipm-src-02.sha256")
```

The neutral snapshots contain source and build inputs but no generated results,
media, repository history, or author metadata. Current SRBD source is present
directly in this archive.

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

The JSON artifacts contain commands, source snapshots, external dependency
revisions, executable/source hashes,
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

## 8. Continuous SRBD closed-loop validation

Configure a Release build without the optional MuJoCo dependency, then build
the continuous runner and its focused acceptance tests:

```bash
cmake -S . -B build-srbd-repro \
  -DCMAKE_BUILD_TYPE=Release \
  -DNMPC_BUILD_EXAMPLES=ON \
  -DNMPC_BUILD_TESTS=ON \
  -DNMPC_BUILD_MUJOCO=OFF

cmake --build build-srbd-repro -j2 --target \
  quadruped_cito_srbd_closed_loop \
  test_quadruped_cito_realtime \
  test_quadruped_cito_realtime_execution \
  test_quadruped_cito_rolling_audit \
  test_quadruped_cito_sustained_topology

(cd build-srbd-repro && ctest --output-on-failure \
  -R 'QuadrupedCITO(RealtimeInvariants|RealtimeExecutionTiming|RollingAudit|SustainedTopology)$')
```

Run the 40 s correctness experiment on the registered 2 cm sinusoidal terrain:

```bash
./build-srbd-repro/quadruped_cito_srbd_closed_loop \
  --output-dir /tmp/contactipm_srbd_02cm \
  --updates 200 \
  --terrain-amplitude 0.02 \
  --wave-number-x 4 \
  --wave-number-y 3 \
  --deadline-ms 0
```

The initial cold solve is followed by 200 warm updates at a 5 Hz simulated
cadence. Between updates, an independent SRBD plant integrates four successive
0.05 s control stages. The state is fed into the next warm solve without an
episode reset. The task sequence contains 24 eight-centimeter transitions in
RL--RR--FL--FR order, followed by a final four-foot-support hold.

A successful run exits with status zero and reports `published=200`,
`fallbacks=0`, `tasks=24/24`, `audit=PASS`, and
`simulated_time=40.000 s`. It writes:

- `metadata.csv` and `summary.csv`;
- `planner_updates.csv`;
- `executed_trajectory.csv`;
- `accepted_plan_stages.csv`;
- `task_ledger.csv`.

The correctness run disables the solver watchdog so convergence and physical
validity can be tested independently of operating-system scheduling. Its
recorded wall times remain useful diagnostics, but are not a hard real-time
guarantee. The separate `RealtimeExecutionTiming` test checks the bounded
execution-path timing invariants. ContactIPM is positioned here as the 5 Hz
high-level planner; articulated MuJoCo tracking through a high-rate convex
whole-body controller remains future work.

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
