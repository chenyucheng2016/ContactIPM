# ContactIPM versus IMPACT

This directory contains a local, parameter-faithful comparison on the three
examples available in both solvers: Push Box, Push T, and Cart Transport.
CRISP code and result artifacts are intentionally outside this workflow.

## Fairness rules

- IMPACT is executed from commit
  `f56f1403659122bd4017ed36b21d83a490bc6687` without substantive source edits.
- ContactIPM uses the same state and control initialization as the checked-in
  IMPACT multiple-shooting front end. Push Box and Push T linearly interpolate
  states from start to goal; Cart repeats the start state; all control and
  contact variables start at zero.
- ContactIPM exactly eliminates only algebraic split variables (`w`) in Push T
  and Cart. They are reconstructed before objective and feasibility auditing.
  No competitor solution is used as a warm start.
- Robustness and solution quality use the predeclared 50-case local suite in
  `impact_cases.json`. The official repository example is case zero. The paper
  does not publish its original random instances or sampling distribution.
- Timing uses local execution only: warmups, one thread, fixed CPU affinity,
  and randomized adjacent solver pairs. No timing value from the paper is used.

## Build and run

Run from the ContactIPM repository root in Ubuntu or WSL. The complete fresh
machine setup, including Eigen BLAS/LAPACK installation, is in
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).

Install the pinned CasADi wheel locally and expose its shared libraries:

```bash
python3 -m pip install --target "$PWD/.deps/casadi-3.7.2" \
  --no-deps casadi==3.7.2
export CONTACTIPM_IMPACT_CASADI_LIBRARY="$PWD/.deps/casadi-3.7.2/casadi"
```

Build the ContactIPM adapters and unmodified pinned IMPACT checkout:

```bash
cmake -S benchmarks/impact_comparison/contactipm_build \
  -B build-impact-contact -DCMAKE_BUILD_TYPE=Release \
  -DCONTACTIPM_SOURCE_DIR="$PWD" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -ffast-math"
cmake --build build-impact-contact -j2

cmake -S benchmarks/impact_comparison/impact_build \
  -B build-impact -DCMAKE_BUILD_TYPE=Release \
  -DIMPACT_SOURCE_DIR="$PWD/benchmarks/IMPACT" \
  -DEIGEN_BLAS_LIBRARY=/usr/local/lib/libeigen_blas.so \
  -DEIGEN_LAPACK_LIBRARY=/usr/local/lib/libeigen_lapack.so \
  -Dcasadi_DIR="$CONTACTIPM_IMPACT_CASADI_LIBRARY/cmake"
cmake --build build-impact -j2
```

The two overlays supply only include paths and compatibility CMake targets;
they do not patch IMPACT. Both sets of timing executables use Release mode,
`-O3 -DNDEBUG -march=native -mtune=native -ffast-math`.

Run one frozen case per common problem as a smoke test:

```bash
python3 benchmarks/impact_comparison/run_impact_comparison.py \
  --contact-build build-impact-contact --impact-build build-impact \
  --impact-repository benchmarks/IMPACT \
  --case-limit 1 --warmups 0 --repetitions 1 \
  --threads 1 --seed 2027 --output impact_smoke.json
```

For the complete 50-case robustness and 20-pair timing commands, table
generation, and provenance verification, follow
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md#5-reproduce-contactipm-versus-impact).

## Artifacts

- `impact_source_cases.json`: frozen IMPACT definitions and initialization.
- `impact_cases.json`: deterministic local 50-case suite (seed 2027).
- `run_impact_comparison.py`: paired runner and raw-artifact recorder.
- `run_impact_audit.py`: canonical independent physics/task audit entry point.
- `summarize_impact_comparison.py`: JSON and Markdown table generator.
- `verify_comparison.py`: source, initialization, objective, and timing-protocol
  checks.
- `results/2026-07-29_robustness_50.json`: 150 robustness pairs.
- `results/2026-07-29_timing_20x.json`: 60 controlled timing pairs.
- `results/2026-07-29_contactipm_vs_impact.md`: side-by-side tables.
