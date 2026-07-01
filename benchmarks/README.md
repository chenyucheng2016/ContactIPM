# Benchmarks: ContactIPM vs acados SQP+HPIPM

Comparison of the ContactIPM Mehrotra IPM solver against acados using
SQP outer loop + HPIPM (Riccati-based interior point QP solver).

## Problems

| Example    | NX | NU | N  | dt    | Dynamics           | Constraints        |
|------------|----|----|----|-------|--------------------|--------------------|
| Pendulum   | 4  | 1  | 20 | 0.05  | Cart-pole RK4      | \|x\|<=2, \|F\|<=30 |
| Quadrotor  | 6  | 2  | 20 | 0.04  | 2D quad RK4        | 0<=u1<=2mg, \|u2\|<=tau, z>=0 |
| Chain Mass | 10 | 3  | 20 | 0.05  | Spring-mass Euler  | \|Fi\|<=5, \|xi\|<=2 |

## Setup

### 1. Build acados (in WSL)
```bash
cd /mnt/c/Users/cheny/Documents/GitHub
git clone https://github.com/acados/acados.git
cd acados && git submodule update --recursive --init
mkdir build && cd build
cmake -DACADOS_INSTALL_DIR=.. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
make install -j$(nproc)
```

### 2. Run setup script
```bash
cd /mnt/c/Users/cheny/Documents/GitHub/nmpc_solver/benchmarks
bash setup.sh
```

### 3. Run benchmarks
```bash
source .venv/bin/activate
export ACADOS_SOURCE_DIR=/mnt/c/Users/cheny/Documents/GitHub/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH

# Run all
python3 run_all.py

# Or individually
python3 acados_pendulum.py
python3 acados_quadrotor.py
python3 acados_chain_mass.py
```

### 4. Run ContactIPM (in PowerShell, for timing)
```powershell
cd c:\Users\cheny\Documents\GitHub\ipm_claude\nmpc_solver\build
cmake --build . --config Release
.\Release\pendulum_nmpc_paper.exe
.\Release\quadrotor_2d_nmpc.exe
.\Release\chain_mass_nmpc.exe
```

## Metrics

- **SQP/IPM iterations**: outer loop convergence speed
- **Solve time**: wall-clock per solve (ms)
- **Cost**: optimal objective value
- **Constraint violation**: max |g(x,u)| over horizon
- **First u***: first optimal control (sanity check solutions match)

---

## Fatrop Benchmark Suite (9 problems, 3 solvers)

Same CasADi-generated C code, same problem definitions, different solver backends:
- **ContactIPM**: Riccati-based Mehrotra IPM (C++)
- **acados**: SQP + PARTIAL_CONDENSING_HPIPM + EXACT Hessians + ERK (C)
- **IPOPT**: Interior point via CasADi nlpsol (Python)

### Unified tolerances
`tol_stat=0.02, tol_primal/eq=1e-4, tol_compl/comp=1e-4, tol_ineq=1e-4`

### Results

| Problem              | ContactIPM         | acados             | IPOPT              |
|----------------------|--------------------|--------------------|--------------------|
| cart_pendulum_mpc    | FAIL 445it 28.0ms  | OK 2it 0.4ms       | OK 68it 116.9ms    |
| cart_pendulum_time   | FAIL 103it 25.0ms  | FAIL 1it 3.5ms     | FAIL 159it 713.9ms |
| quadcopter_mpc       | FAIL 135it 16.3ms  | FAIL 1it 0.3ms     | OK 22it 52.6ms     |
| quadcopter_p2p       | CRASH              | OK 7it 1.9ms       | OK 11it 19.1ms     |
| quadcopter_one_obs   | CRASH              | FAIL 1it 4.5ms     | OK 114it 589.2ms   |
| quadcopter_three_obs | CRASH              | FAIL 1it 4.2ms     | OK 133it 1620.5ms  |
| truck_trailer_time   | FAIL 0it 0.7ms     | FAIL 1it 2.3ms     | FAIL 208it 946.9ms |
| hanging_chain_2d     | FAIL 178it 387.4ms | OK 3it 9.4ms       | OK 11it 76.6ms     |
| hanging_chain_3d     | FAIL 152it 2796.3ms| OK 3it 25.6ms      | OK 10it 141.6ms    |
| **Solved**           | **0/9**            | **4/9**            | **7/9**            |

### How to run

```powershell
# Run all 9 problems x 3 solvers
cd benchmarks\fatrop_benchmarks
python run_all_benchmarks.py --json

# Run specific problem(s)
python run_all_benchmarks.py cart_pendulum_mpc quadcopter_p2p

# Run specific solver only
python run_all_benchmarks.py --solver ipopt

# IPOPT only
python benchmark_ipopt.py [problem_name]
```
