# ContactIPM

**Interior Point Method solver for contact-rich NMPC in quadruped robotics**

## Objective

ContactIPM is a high-performance Nonlinear Model Predictive Control (NMPC) solver designed for contact-rich locomotion and manipulation tasks in quadruped robots. The solver uses a primal-dual Interior Point Method (IPM) with Riccati recursion to efficiently solve the constrained optimization problems arising in legged robot motion planning.

### Key Features

- **Mehrotra Predictor-Corrector IPM** — Fast convergence with adaptive barrier parameter updates
- **Riccati Recursion** — Exploits the banded structure of the KKT system for O(N) complexity
- **Filter Line Search** — Robust globalization strategy balancing objective decrease and feasibility
- **Second-Order Correction (SOC)** — Mitigates the Maratos effect for fast local convergence
- **Flexible Constraint Handling** — Supports box constraints, nonlinear inequalities, and complementarity conditions
- **Header-Only Design** — Easy integration into existing robotics frameworks

## Current Status

### Working

| Feature | Status | Notes |
|---------|--------|-------|
| Core IPM solver | ✅ Stable | Mehrotra predictor-corrector with adaptive μ |
| Riccati KKT solver | ✅ Stable | Banded Cholesky factorization |
| Filter line search | ✅ Stable | IPOPT-style filter with SOC |
| Barrier management | ✅ Stable | Condition-based μ reduction |
| Example: Pendulum | ✅ Converges | 4 iterations |
| Example: Quadrotor | ✅ Converges | 81 iterations |
| Example: Chain Mass | ✅ Converges | 24 iterations |

### In Development

- Contact dynamics modeling for quadruped robots
- Hybrid system handling (stance/flight phase transitions)
- Friction cone constraints
- Multi-contact scheduling

## Getting Started

### Prerequisites

- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)
- CMake 3.16+
- Eigen (included as header-only)

### Build

```bash
# Clone the repository
git clone https://github.com/chenyucheng2016/ContactIPM.git
cd ContactIPM

# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Run Examples

```bash
# From the build directory
./pendulum_nmpc_paper    # Cart-pole swing-up
./quadrotor_2d_nmpc      # 2D quadrotor trajectory tracking
./chain_mass_nmpc        # Chain mass with force constraints
```

### Run Tests

```bash
cd build
ctest --output-on-failure
```

## Usage

### Basic Example

```cpp
#include <nmpc/nmpc_ipm_paper.hpp>

// Define problem dimensions
constexpr int NX = 4;   // State dimension
constexpr int NU = 1;   // Control dimension
constexpr int NC = 4;   // Constraint dimension
constexpr int N  = 20;  // Horizon length

// Create problem definition
auto problem = std::make_shared<nmpc::NMPCProblem<NX, NU, NC, N>>();

// Set dynamics, cost, constraints...
problem->dynamics = my_dynamics;
problem->cost = my_cost;
problem->constraints = my_constraints;

// Create solver
nmpc::NMPCSolverPaper<NX, NU, NC, N> solver(problem);

// Solve
auto result = solver.solve(initial_state, reference_trajectory);

if (result.converged) {
    // Extract optimal controls
    for (int k = 0; k < N; ++k) {
        auto u_opt = result.stages[k].u;
        // Apply u_opt[0] to system...
    }
}
```

### Solver Parameters

Key parameters in `nmpc::NMPCSolverParams`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tol_primal` | 1e-2 | Primal feasibility tolerance |
| `tol_dual` | 1e-2 | Dual feasibility tolerance |
| `tol_compl` | 1e-2 | Complementarity tolerance |
| `mu_min` | 1e-3 | Minimum barrier parameter |
| `max_iter` | 200 | Maximum iterations |
| `verbosity` | 1 | Log verbosity (0=silent, 1=summary, 2=detailed) |

## Architecture

```
ContactIPM/
├── include/nmpc/
│   ├── nmpc_core.hpp           # Core types (Vec, Mat, Stage)
│   ├── nmpc_problem.hpp        # Problem definition interface
│   ├── nmpc_ipm_paper.hpp      # Main IPM solver
│   ├── nmpc_riccati.hpp        # Riccati KKT solver
│   ├── nmpc_filter_ls.hpp      # Filter line search
│   ├── nmpc_barrier_manager.hpp# Barrier parameter control
│   ├── nmpc_hessian_approx.hpp # Gauss-Newton Hessian
│   ├── nmpc_kkt_diag.hpp       # KKT residual diagnostics
│   └── nmpc_preconditioner.hpp # Jacobi preconditioning
├── examples/                   # Example problems
└── tests/                      # Unit tests
```

## References

- Mehrotra, S. (1992). "On the implementation of a primal-dual interior point method"
- Wächter, A., & Biegler, L. T. (2006). "On the implementation of an interior-point filter line-search algorithm"
- Nocedal, J., & Wright, S. J. (2006). "Numerical Optimization"

## License

To be determined.

## Contact

For questions or contributions, please open an issue or pull request.
