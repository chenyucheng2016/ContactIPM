# WSL interleaved development benchmark - 2026-07-27

This is a development milestone, not a publication-quality timing claim. Source
snapshot `contactipm-src-12` captures the measured solver and benchmark source
plus its regression registrations. The raw runner recorded the earlier base
snapshot `contactipm-src-06` because the measured changes were not yet frozen.

## Protocol

- Ubuntu 20.04 under WSL2, Linux 6.6.87.2, x86-64, GCC 9.4 Release builds.
- One process and one thread (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`).
- One warmup for every solver/problem pair.
- Three measured repetitions, randomized and interleaved with seed 2027.
- Process wall clock measured locally by the same Python runner.
- No execution times from the CRISP paper were used.
- All 24 measured processes returned zero.

## Locally measured wall time

Values are median seconds with the observed three-run range in parentheses. `CRISP / ContactIPM` greater than one favors ContactIPM.

| Problem | ContactIPM | CRISP | CRISP / ContactIPM |
|---|---:|---:|---:|
| Cartpole with Soft Walls | 0.3945 (0.3788-0.3988) | 0.1396 (0.1299-0.1624) | 0.35x |
| Push Box | 0.2235 (0.2192-0.2275) | 0.3790 (0.3655-0.4038) | 1.70x |
| Transport | 0.0638 (0.0601-0.0661) | 0.3136 (0.3102-0.3183) | 4.91x |
| Push T, 50 initial conditions | 59.8147 (58.7273-60.4985) | 347.8974 (339.7686-350.6365) | 5.82x |

ContactIPM is faster on three of four problems in this development run and slower on Cartpole. These ratios include complete executable wall time and must not be presented as final paper statistics from only three repetitions.

## Correctness evidence

All four ContactIPM adapters passed their raw task gates. Their measured physical complementarity maxima were `9.997e-7` (Cartpole), `9.805e-7` (Push Box), `1.908e-6` (Transport), and `4.564e-6` (worst Push-T segment), all below the configured `1e-5` gate. Push T passed 50/50 source initial conditions in every measured repetition; its worst raw dynamics defect was `6.580e-6` and worst side violation was `8.499e-9`.

CRISP Push T emits all trajectories, so the runner independently re-evaluated both solvers with the common raw gates: equality/dynamics `<= 1e-5`, side violation `<= 1e-8`, physical product `<= 1e-5`, terminal position `< 0.1`, and terminal angle `< pi/6`. CRISP passed 27/50 in every repetition, with worst equality `1.9998` and worst physical product `4.484e-4`; ContactIPM passed 50/50.

The other three unmodified CRISP examples do not emit trajectories. Their processes returned zero and self-reported small constraint violations, but they have not yet received the same independent raw audit. Therefore the current evidence supports a strong Push-T robustness comparison and timing measurements on all four problems, not a fully audited cross-solver success comparison on Cartpole, Push Box, and Transport.

## Reproduction and next publication checks

The complete machine-readable record, including commands, environment, raw stdout/stderr, timings, and all three Push-T trajectory audits, is in [2026-07-27_wsl_interleaved_3x.json](2026-07-27_wsl_interleaved_3x.json).

Before making paper claims, export raw CRISP trajectories for the other three problems, audit both solvers with identical gates, pin hardware and CPU affinity, increase repetitions, report confidence intervals, and investigate Cartpole where ContactIPM is currently slower.
