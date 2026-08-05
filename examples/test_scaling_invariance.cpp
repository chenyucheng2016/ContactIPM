// Quick invariance test: solve a single-stage QP in physical and scaled coords
// Build: same as quadrotor_2d_nmpc
#include "nmpc/nmpc_preconditioner.hpp"
#include "nmpc/nmpc_riccati.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace nmpc;

static constexpr int NX = 3, NU = 2, NC = 2, H = 0;
using Stage = StageData<NX, NU, NC>;

int main() {
    // Physical QP data
    Stage phys[1];
    
    // Random symmetric positive definite Hessian
    phys[0].Qxx = {}; phys[0].Quu = {}; phys[0].Qux = {};
    // Qxx = [[10, 1, 0], [1, 5, 2], [0, 2, 8]]
    phys[0].Qxx(0,0) = 10; phys[0].Qxx(0,1) = 1;  phys[0].Qxx(0,2) = 0;
    phys[0].Qxx(1,0) = 1;  phys[0].Qxx(1,1) = 5;  phys[0].Qxx(1,2) = 2;
    phys[0].Qxx(2,0) = 0;  phys[0].Qxx(2,1) = 2;  phys[0].Qxx(2,2) = 8;
    
    // Quu = [[4, 0.5], [0.5, 3]]
    phys[0].Quu(0,0) = 4;   phys[0].Quu(0,1) = 0.5;
    phys[0].Quu(1,0) = 0.5; phys[0].Quu(1,1) = 3;
    
    // Qux (NU x NX)
    phys[0].Qux(0,0) = 0.5; phys[0].Qux(0,1) = 0.1; phys[0].Qux(0,2) = 0.2;
    phys[0].Qux(1,0) = 0.3; phys[0].Qux(1,1) = 0.4; phys[0].Qux(1,2) = 0.1;
    
    // Gradient
    phys[0].qx[0] = 1.0; phys[0].qx[1] = -0.5; phys[0].qx[2] = 0.3;
    phys[0].qu[0] = -0.2; phys[0].qu[1] = 0.7;
    
    // No dynamics (H=0, single stage)
    // A, B, c unused for H=0
    
    // Constraints: Cx (NC x NX), Cu (NC x NU)
    phys[0].Cx(0,0) = 1; phys[0].Cx(0,1) = 0; phys[0].Cx(0,2) = -1;
    phys[0].Cx(1,0) = 0; phys[0].Cx(1,1) = 1; phys[0].Cx(1,2) = 0;
    phys[0].Cu(0,0) = 0; phys[0].Cu(0,1) = 0;
    phys[0].Cu(1,0) = 1; phys[0].Cu(1,1) = -1;
    phys[0].d[0] = -0.5; phys[0].d[1] = 0.3;
    
    // Slack and duals
    phys[0].s[0] = 1.0; phys[0].s[1] = 0.5;
    phys[0].lambda[0] = 0.2; phys[0].lambda[1] = 0.3;
    
    // ===== Solve PHYSICAL QP via Riccati (H=0, so just one stage) =====
    // For H=0: S = Quu, K = S^{-1}*Qux, dx from initial condition
    // Actually for a static QP with H=0, the "Riccati" is just:
    // P = Qxx, p = qx, S = Quu, d = S^{-1}*(qu + BtP*c)
    // Since there's no dynamics, dx = dx0 (initial condition)
    
    // Let's just solve the KKT system directly:
    // [H  C^T] [dz]   [-g]
    // [C  0  ] [w ] = [-d-s*w/s] ... actually let me just solve H*dz = -g with constraint
    
    // For the invariance test, let's solve the unconstrained Newton system:
    // H * dz = -g  (ignoring constraints for simplicity)
    // where H = [[Qxx, Qux^T], [Qux, Quu]], g = [qx; qu]
    
    // Actually, let's use the Riccati for H=0 (terminal stage only):
    // P[0] = Qxx + reg*I, p[0] = qx
    // The step dx[0] = dx0 (initial condition), du[0] = -K*dx0 - d
    // For a meaningful test, set dx0 = some initial state error
    
    // Let me do a simpler test: just check that transform_qp preserves
    // the KKT matrix eigenvalues (condition number)
    
    // Compute H0 diagonal for scaling
    Stage stages_for_scaling = phys[0];
    
    // Compute Lx from Qxx diagonal
    Mat<NX,NX> Lx, inv_Lx;
    Lx.zero(); inv_Lx.zero();
    for (int i = 0; i < NX; ++i) {
        double h = phys[0].Qxx(i,i);
        // Add barrier contribution
        for (int j = 0; j < NC; ++j) {
            double sj = phys[0].s[j];
            if (sj > 1e-20) {
                double wj = phys[0].lambda[j] / sj;
                double cji = phys[0].Cx(j,i);
                h += wj * cji * cji;
            }
        }
        double s = std::sqrt(std::max(h, 1e-14));
        Lx(i,i) = s;
        inv_Lx(i,i) = 1.0/s;
        printf("Lx[%d] = %.6f  (H0_diag = %.6f)\n", i, s, h);
    }
    
    Mat<NU,NU> Lu, inv_Lu;
    Lu.zero(); inv_Lu.zero();
    for (int i = 0; i < NU; ++i) {
        double d = phys[0].Quu(i,i);
        double s = std::sqrt(std::max(d, 1e-14));
        Lu(i,i) = s;
        inv_Lu(i,i) = 1.0/s;
        printf("Lu[%d] = %.6f  (Quu_diag = %.6f)\n", i, s, d);
    }
    
    // Now test: solve H*dz = -g in physical space
    // H = [[Qxx, Qux^T], [Qux, Quu]] (5x5)
    // g = [qx; qu] (5x1)
    
    // Simple Gaussian elimination for 5x5 system
    const int N = NX + NU; // 5
    double H_phys[N][N], g_phys[N];
    
    // Fill H
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NX; ++j) H_phys[i][j] = phys[0].Qxx(i,j);
        for (int j = 0; j < NU; ++j) H_phys[i][NX+j] = phys[0].Qux(j,i); // Qux^T
    }
    for (int i = 0; i < NU; ++i) {
        for (int j = 0; j < NX; ++j) H_phys[NX+i][j] = phys[0].Qux(i,j);
        for (int j = 0; j < NU; ++j) H_phys[NX+i][NX+j] = phys[0].Quu(i,j);
    }
    // Fill g
    for (int i = 0; i < NX; ++i) g_phys[i] = phys[0].qx[i];
    for (int i = 0; i < NU; ++i) g_phys[NX+i] = phys[0].qu[i];
    
    // Solve H*dz = -g using Gaussian elimination with partial pivoting
    double A[N][N+1], dz_phys[N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) A[i][j] = H_phys[i][j];
        A[i][N] = -g_phys[i];
    }
    for (int k = 0; k < N; ++k) {
        // Partial pivoting
        int max_row = k;
        for (int i = k+1; i < N; ++i)
            if (std::fabs(A[i][k]) > std::fabs(A[max_row][k])) max_row = i;
        if (max_row != k) for (int j = k; j <= N; ++j) { double t = A[k][j]; A[k][j] = A[max_row][j]; A[max_row][j] = t; }
        for (int i = k+1; i < N; ++i) {
            double f = A[i][k] / A[k][k];
            for (int j = k; j <= N; ++j) A[i][j] -= f * A[k][j];
        }
    }
    for (int i = N-1; i >= 0; --i) {
        dz_phys[i] = A[i][N];
        for (int j = i+1; j < N; ++j) dz_phys[i] -= A[i][j] * dz_phys[j];
        dz_phys[i] /= A[i][i];
    }
    
    printf("\nPhysical Newton step:\n");
    for (int i = 0; i < NX; ++i) printf("  dx[%d] = %.10f\n", i, dz_phys[i]);
    for (int i = 0; i < NU; ++i) printf("  du[%d] = %.10f\n", i, dz_phys[NX+i]);
    
    // ===== Now solve in SCALED space =====
    // Transform: Q' = inv_L*Q*inv_L^T, q' = inv_L*q, etc.
    // For the Hessian, the full H in scaled space is:
    // H_scaled = diag(inv_Lx, inv_Lu) * H * diag(inv_Lx, inv_Lu)^T
    
    double H_scaled[N][N], g_scaled[N];
    double scale[N];
    for (int i = 0; i < NX; ++i) scale[i] = inv_Lx(i,i);
    for (int i = 0; i < NU; ++i) scale[NX+i] = inv_Lu(i,i);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            H_scaled[i][j] = scale[i] * H_phys[i][j] * scale[j];
        g_scaled[i] = scale[i] * g_phys[i];
    }
    
    // Solve H_scaled * dz_hat = -g_scaled
    double B[N][N+1], dz_hat[N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) B[i][j] = H_scaled[i][j];
        B[i][N] = -g_scaled[i];
    }
    for (int k = 0; k < N; ++k) {
        int max_row = k;
        for (int i = k+1; i < N; ++i)
            if (std::fabs(B[i][k]) > std::fabs(B[max_row][k])) max_row = i;
        if (max_row != k) for (int j = k; j <= N; ++j) { double t = B[k][j]; B[k][j] = B[max_row][j]; B[max_row][j] = t; }
        for (int i = k+1; i < N; ++i) {
            double f = B[i][k] / B[k][k];
            for (int j = k; j <= N; ++j) B[i][j] -= f * B[k][j];
        }
    }
    for (int i = N-1; i >= 0; --i) {
        dz_hat[i] = B[i][N];
        for (int j = i+1; j < N; ++j) dz_hat[i] -= B[i][j] * dz_hat[j];
        dz_hat[i] /= B[i][i];
    }
    
    printf("\nScaled Newton step (dz_hat):\n");
    for (int i = 0; i < NX; ++i) printf("  dx_hat[%d] = %.10f\n", i, dz_hat[i]);
    for (int i = 0; i < NU; ++i) printf("  du_hat[%d] = %.10f\n", i, dz_hat[NX+i]);
    
    // Recover physical: dz = inv_L^T * dz_hat = inv_L * dz_hat (diagonal)
    double dz_recovered[N];
    for (int i = 0; i < NX; ++i) dz_recovered[i] = inv_Lx(i,i) * dz_hat[i];
    for (int i = 0; i < NU; ++i) dz_recovered[NX+i] = inv_Lu(i,i) * dz_hat[NX+i];
    
    printf("\nRecovered physical step (inv_L * dz_hat):\n");
    for (int i = 0; i < NX; ++i) printf("  dx_rec[%d] = %.10f\n", i, dz_recovered[i]);
    for (int i = 0; i < NU; ++i) printf("  du_rec[%d] = %.10f\n", i, dz_recovered[NX+i]);
    
    // Compare
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::fabs(dz_phys[i] - dz_recovered[i]);
        if (err > max_err) max_err = err;
    }
    printf("\n||dz_phys - dz_recovered||_inf = %.2e\n", max_err);
    printf("Test %s\n", max_err < 1e-10 ? "PASSED" : "FAILED");
    
    return max_err < 1e-10 ? 0 : 1;
}
