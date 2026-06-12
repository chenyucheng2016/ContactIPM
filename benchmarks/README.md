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
