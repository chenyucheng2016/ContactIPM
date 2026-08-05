// Multi-stage Riccati invariance test under diagonal scaling
// Tests that the Newton step from the Riccati recursion is invariant
// when using the correct Convention A transforms.
#include "nmpc/nmpc_core.hpp"
#include "nmpc/nmpc_riccati.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace nmpc;

#define RAND_DOUBLE (double)RAND_MAX

static constexpr int NX = 3, NU = 2, H = 4, NC = 1;
using Stage = StageData<NX, NU, NC>;
using WS = RiccatiWorkspace<NX, NU, H>;
using Ric = RiccatiSolver<NX, NU, NC, H>;

int main() {
    const int N = H;
    
    // ===== Build a random multi-stage LQR =====
    Stage phys[N + 1];
    
    // Seed for reproducibility
    srand(42);
    auto rnd = []() { return 0.1 + 2.0 * rand() / RAND_DOUBLE; };
    
    for (int k = 0; k <= N; ++k) {
        // Random SPD Qxx
        phys[k].Qxx.zero();
        for (int i = 0; i < NX; ++i) {
            phys[k].Qxx(i,i) = 1.0 + 5.0 * rand() / RAND_DOUBLE;
            for (int j = 0; j < i; ++j) {
                double v = 0.3 * rand() / RAND_DOUBLE;
                phys[k].Qxx(i,j) = v;
                phys[k].Qxx(j,i) = v;
            }
        }
        
        if (k < N) {
            // Random SPD Quu
            phys[k].Quu.zero();
            for (int i = 0; i < NU; ++i) {
                phys[k].Quu(i,i) = 1.0 + 3.0 * rand() / RAND_DOUBLE;
                for (int j = 0; j < i; ++j) {
                    double v = 0.2 * rand() / RAND_DOUBLE;
                    phys[k].Quu(i,j) = v;
                    phys[k].Quu(j,i) = v;
                }
            }
            
            // Random Qux
            for (int i = 0; i < NU; ++i)
                for (int j = 0; j < NX; ++j)
                    phys[k].Qux(i,j) = 0.5 * rand() / RAND_DOUBLE;
            
            // Random A (stable-ish)
            phys[k].A.zero();
            for (int i = 0; i < NX; ++i) {
                for (int j = 0; j < NX; ++j)
                    phys[k].A(i,j) = 0.3 * rand() / RAND_DOUBLE;
                phys[k].A(i,i) = 0.5 + 0.3 * rand() / RAND_DOUBLE;
            }
            
            // Random B
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NU; ++j)
                    phys[k].B(i,j) = 0.5 * rand() / RAND_DOUBLE;
            
            // Random c
            for (int i = 0; i < NX; ++i)
                phys[k].c[i] = 0.2 * rand() / RAND_DOUBLE;
            
            // Random qu
            for (int i = 0; i < NU; ++i)
                phys[k].qu[i] = rand() / RAND_DOUBLE;
        }
        
        // Random qx
        for (int i = 0; i < NX; ++i)
            phys[k].qx[i] = rand() / RAND_DOUBLE;
    }
    
    // Initial state error
    Vec<NX> dx0;
    for (int i = 0; i < NX; ++i)
        dx0[i] = 0.5 * rand() / RAND_DOUBLE;
    
    // ===== Solve PHYSICAL Riccati =====
    WS ws_phys;
    // Copy physical stages to a working array (Riccati modifies stages for LHS)
    Stage phys_work[N + 1];
    for (int k = 0; k <= N; ++k) phys_work[k] = phys[k];
    
    double reg_phys = 0.0;
    Status st = Ric::backward_lhs(phys_work, ws_phys, 0.0, reg_phys);
    if (st != Status::SUCCESS) { printf("Physical LHS failed\n"); return 1; }
    st = Ric::backward_rhs(phys_work, ws_phys);
    if (st != Status::SUCCESS) { printf("Physical RHS failed\n"); return 1; }
    st = Ric::forward(phys_work, ws_phys, dx0);
    if (st != Status::SUCCESS) { printf("Physical forward failed\n"); return 1; }
    
    printf("Physical Riccati: reg=%.2e\n", reg_phys);
    printf("Physical backward:\n");
    for (int k = 0; k <= N; ++k) {
        printf("  k=%d P_diag=", k);
        for (int i = 0; i < NX; ++i) printf("%.8e ", ws_phys.P[k](i,i));
        printf(" p=", k);
        for (int i = 0; i < NX; ++i) printf("%.8e ", ws_phys.p[k][i]);
        if (k < N) {
            printf(" K[0,0]=%.8e d=", ws_phys.K[0](0,0));
            for (int i = 0; i < NU; ++i) printf("%.8e ", ws_phys.d[k][i]);
        }
        printf("\n");
    }
    printf("Physical Newton step:\n");
    for (int k = 0; k <= N; ++k) {
        printf("  k=%d dx=[", k);
        for (int i = 0; i < NX; ++i) printf("%.8f ", ws_phys.dx[k][i]);
        printf("]");
        if (k < N) {
            printf(" du=[");
            for (int i = 0; i < NU; ++i) printf("%.8f ", ws_phys.du[k][i]);
            printf("]");
        }
        printf("\n");
    }
    
    // ===== Compute scaling =====
    Mat<NX,NX> Lx[N + 1], inv_Lx[N + 1];
    Mat<NU,NU> Lu[N], inv_Lu[N];
    
    for (int k = 0; k <= N; ++k) {
        Lx[k].zero(); inv_Lx[k].zero();
        for (int i = 0; i < NX; ++i) {
            double h = phys[k].Qxx(i,i);
            double s = std::sqrt(std::max(h, 1e-14));
            Lx[k](i,i) = s;
            inv_Lx[k](i,i) = 1.0 / s;
        }
        if (k < N) {
            Lu[k].zero(); inv_Lu[k].zero();
            for (int i = 0; i < NU; ++i) {
                double d = phys[k].Quu(i,i);
                double s = std::sqrt(std::max(d, 1e-14));
                Lu[k](i,i) = s;
                inv_Lu[k](i,i) = 1.0 / s;
            }
        }
    }
    
    // ===== Transform to scaled space (Convention A) =====
    Stage scaled[N + 1];
    for (int k = 0; k <= N; ++k) scaled[k] = phys[k];  // copy
    
    for (int k = 0; k <= N; ++k) {
        // Qxx' = inv_Lx * Qxx * inv_Lx^T
        // For diagonal: Qxx'(i,j) = inv_Lx(i) * Qxx(i,j) * inv_Lx(j)
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NX; ++j)
                scaled[k].Qxx(i,j) = inv_Lx[k](i,i) * phys[k].Qxx(i,j) * inv_Lx[k](j,j);
        
        // qx' = inv_Lx * qx
        for (int i = 0; i < NX; ++i)
            scaled[k].qx[i] = inv_Lx[k](i,i) * phys[k].qx[i];
        
        if (k < N) {
            // Quu' = inv_Lu * Quu * inv_Lu^T
            for (int i = 0; i < NU; ++i)
                for (int j = 0; j < NU; ++j)
                    scaled[k].Quu(i,j) = inv_Lu[k](i,i) * phys[k].Quu(i,j) * inv_Lu[k](j,j);
            
            // Qux' = inv_Lu * Qux * inv_Lx^T
            for (int i = 0; i < NU; ++i)
                for (int j = 0; j < NX; ++j)
                    scaled[k].Qux(i,j) = inv_Lu[k](i,i) * phys[k].Qux(i,j) * inv_Lx[k](j,j);
            
            // qu' = inv_Lu * qu
            for (int i = 0; i < NU; ++i)
                scaled[k].qu[i] = inv_Lu[k](i,i) * phys[k].qu[i];
            
            // A' = inv_Lx_{k+1} * A * inv_Lx_k  (Convention A)
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NX; ++j)
                    scaled[k].A(i,j) = inv_Lx[k+1](i,i) * phys[k].A(i,j) * inv_Lx[k](j,j);
            
            // B' = inv_Lx_{k+1} * B * inv_Lu_k
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NU; ++j)
                    scaled[k].B(i,j) = inv_Lx[k+1](i,i) * phys[k].B(i,j) * inv_Lu[k](j,j);
            
            // c' = inv_Lx_{k+1} * c
            for (int i = 0; i < NX; ++i)
                scaled[k].c[i] = inv_Lx[k+1](i,i) * phys[k].c[i];
        }
    }
    
    // ===== Solve SCALED Riccati =====
    WS ws_scaled;
    Stage scaled_work[N + 1];
    for (int k = 0; k <= N; ++k) scaled_work[k] = scaled[k];
    
    double reg_scaled = 0.0;
    st = Ric::backward_lhs(scaled_work, ws_scaled, 0.0, reg_scaled);
    if (st != Status::SUCCESS) { printf("Scaled LHS failed\n"); return 1; }
    st = Ric::backward_rhs(scaled_work, ws_scaled);
    if (st != Status::SUCCESS) { printf("Scaled RHS failed\n"); return 1; }
    
    // Scale initial condition: dx̂0 = inv_Lx * dx0  (Convention A)
    Vec<NX> dx0_hat;
    for (int i = 0; i < NX; ++i)
        dx0_hat[i] = inv_Lx[0](i,i) * dx0[i];
    
    st = Ric::forward(scaled_work, ws_scaled, dx0_hat);
    if (st != Status::SUCCESS) { printf("Scaled forward failed\n"); return 1; }
    
    printf("\nScaled Riccati: reg=%.2e\n", reg_scaled);
    
    // Terminal stage check
    printf("\nTerminal stage check:\n");
    printf("  phys qx_4=["); for (int i=0;i<NX;++i) printf("%.8e ",phys[N].qx[i]); printf("]\n");
    printf("  scaled qhat_4=["); for (int i=0;i<NX;++i) printf("%.8e ",scaled[N].qx[i]); printf("]\n");
    printf("  inv_Lx4*qx4=["); for (int i=0;i<NX;++i) printf("%.8e ",inv_Lx[N](i,i)*phys[N].qx[i]); printf("]\n");
    printf("  ws_phys.p4=["); for (int i=0;i<NX;++i) printf("%.8e ",ws_phys.p[N][i]); printf("]\n");
    printf("  ws_scaled.p4=["); for (int i=0;i<NX;++i) printf("%.8e ",ws_scaled.p[N][i]); printf("]\n");
    
    // Check c transform at k=3
    printf("\nc-transform check (k=3):\n");
    printf("  phys c=["); for(int i=0;i<NX;++i) printf("%.8e ",phys[3].c[i]); printf("]\n");
    printf("  scaled c=["); for(int i=0;i<NX;++i) printf("%.8e ",scaled[3].c[i]); printf("]\n");
    printf("  Lx4*c=["); for(int i=0;i<NX;++i) printf("%.8e ",Lx[4](i,i)*phys[3].c[i]); printf("]\n");
    printf("  inv_Lx4*c=["); for(int i=0;i<NX;++i) printf("%.8e ",inv_Lx[4](i,i)*phys[3].c[i]); printf("]\n");
    
    // Check A transform at k=3
    printf("\nA-transform check (k=3, [0,0]):\n");
    printf("  phys A=%.8e  scaled A=%.8e  Lx4*A*inv_Lx3=%.8e  inv_Lx4*A*inv_Lx3=%.8e\n",
        phys[3].A(0,0), scaled[3].A(0,0),
        Lx[4](0,0)*phys[3].A(0,0)*inv_Lx[3](0,0),
        inv_Lx[4](0,0)*phys[3].A(0,0)*inv_Lx[3](0,0));
    
    printf("Scaled backward:\n");
    for (int k = 0; k <= N; ++k) {
        printf("  k=%d P_diag=", k);
        for (int i = 0; i < NX; ++i) printf("%.8e ", ws_scaled.P[k](i,i));
        printf(" p=", k);
        for (int i = 0; i < NX; ++i) printf("%.8e ", ws_scaled.p[k][i]);
        if (k < N) {
            printf(" K[0,0]=%.8e d=", ws_scaled.K[k](0,0));
            for (int i = 0; i < NU; ++i) printf("%.8e ", ws_scaled.d[k][i]);
        }
        printf("\n");
    }
    // Direct P recursion check at k=3
    {
        int k = 3;
        printf("\nP recursion check at k=%d:\n", k);
        // Term 1: Qhat_xx + Ahat^T * Phat_{k+1} * Ahat
        double term1[NX][NX];
        for (int r = 0; r < NX; ++r) for (int c = 0; c < NX; ++c) {
            double ata = 0;
            for (int i = 0; i < NX; ++i) for (int j = 0; j < NX; ++j)
                ata += scaled[k].A(i,r) * ws_scaled.P[k+1](i,j) * scaled[k].A(j,c);
            term1[r][c] = scaled[k].Qxx(r,c) + ata;
        }
        // Term 2: Qhat_ux^T * Khat
        double term2[NX][NX];
        for (int r = 0; r < NX; ++r) for (int c = 0; c < NX; ++c) {
            double s = 0;
            for (int m = 0; m < NU; ++m) s += scaled[k].Qux(m,r) * ws_scaled.K[k](m,c);
            term2[r][c] = s;
        }
        printf("  Qhat+Ahat^T*Phat*Ahat diag=[");
        for (int i = 0; i < NX; ++i) printf("%.8e ", term1[i][i]); printf("]\n");
        printf("  Qhat_ux^T*Khat diag=[");
        for (int i = 0; i < NX; ++i) printf("%.8e ", term2[i][i]); printf("]\n");
        printf("  P_hat = term1-term2 diag=[");
        for (int i = 0; i < NX; ++i) printf("%.8e ", term1[i][i]-term2[i][i]); printf("]\n");
        printf("  ws_scaled.P[3] diag=[");
        for (int i = 0; i < NX; ++i) printf("%.8e ", ws_scaled.P[k](i,i)); printf("]\n");
        // Expected: inv_Lx * (Qxx + A^T*P*A - Qux^T*K) * inv_Lx
        printf("  inv_Lx*(phys_P_rec)*inv_Lx diag=[");
        for (int i = 0; i < NX; ++i) {
            double ata_phys = 0;
            for (int ii = 0; ii < NX; ++ii) for (int jj = 0; jj < NX; ++jj)
                ata_phys += phys[k].A(ii,i) * ws_phys.P[k+1](ii,jj) * phys[k].A(jj,i);
            double prec = phys[k].Qxx(i,i) + ata_phys;
            for (int m = 0; m < NU; ++m) prec -= phys[k].Qux(m,i) * ws_phys.K[k](m,i);
            printf("%.8e ", inv_Lx[k](i,i) * prec * inv_Lx[k](i,i));
        }
        printf("]\n");
        // Check individual transforms
        printf("\n  Individual transform checks at k=%d:\n", k);
        // Qxx transform: Qhat = inv_Lx*Qxx*inv_Lx?
        printf("    Qhat[0,0]=%.8e  inv_Lx*Qxx*inv_Lx[0,0]=%.8e\n",
            scaled[k].Qxx(0,0), inv_Lx[k](0,0)*phys[k].Qxx(0,0)*inv_Lx[k](0,0));
        // A transform: Ahat = inv_Lx_{k+1}*A*inv_Lx_k?
        printf("    Ahat[0,0]=%.8e  inv_Lx4*A*inv_Lx3[0,0]=%.8e\n",
            scaled[k].A(0,0), inv_Lx[k+1](0,0)*phys[k].A(0,0)*inv_Lx[k](0,0));
        // P_{k+1} transform: Phat = inv_Lx*P*inv_Lx?
        printf("    Phat_4[0,0]=%.8e  inv_Lx4*P4*inv_Lx4[0,0]=%.8e\n",
            ws_scaled.P[k+1](0,0), inv_Lx[k+1](0,0)*ws_phys.P[k+1](0,0)*inv_Lx[k+1](0,0));
        // Qux transform: Qhat_ux = inv_Lu*Qux*inv_Lx?
        printf("    Qhat_ux[0,0]=%.8e  inv_Lu*Qux*inv_Lx[0,0]=%.8e\n",
            scaled[k].Qux(0,0), inv_Lu[k](0,0)*phys[k].Qux(0,0)*inv_Lx[k](0,0));
        // K transform: Khat = inv_Lu*K*Lx?
        printf("    Khat[0,0]=%.8e  Lu*K*inv_Lx[0,0]=%.8e  inv_Lu*K*Lx[0,0]=%.8e\n",
            ws_scaled.K[k](0,0),
            Lu[k](0,0)*ws_phys.K[k](0,0)*inv_Lx[k](0,0),
            inv_Lu[k](0,0)*ws_phys.K[k](0,0)*Lx[k](0,0));
        // A^T*P*A term check
        double ata_scaled = 0, ata_expected = 0;
        for (int ii = 0; ii < NX; ++ii) for (int jj = 0; jj < NX; ++jj) {
            ata_scaled += scaled[k].A(ii,0) * ws_scaled.P[k+1](ii,jj) * scaled[k].A(jj,0);
            ata_expected += phys[k].A(ii,0) * ws_phys.P[k+1](ii,jj) * phys[k].A(jj,0);
        }
        printf("    (Ahat^T*Phat*Ahat)[0,0]=%.8e  inv_Lx*(A^T*P*A)*inv_Lx[0,0]=%.8e\n",
            ata_scaled, inv_Lx[k](0,0)*ata_expected*inv_Lx[k](0,0));
        // Manual check: compute A^T*P*A using phys A and phys P, then scale
        {
            // First verify SymMat access is correct
            printf("    SymMat P_4 access check: P(0,1)=%.8e P(1,0)=%.8e\n",
                ws_phys.P[k+1](0,1), ws_phys.P[k+1](1,0));
            printf("    SymMat P_4 access check: P(0,2)=%.8e P(2,0)=%.8e\n",
                ws_phys.P[k+1](0,2), ws_phys.P[k+1](2,0));
            // Also check scaled
            printf("    Scaled P_4 access: P(0,1)=%.8e P(1,0)=%.8e\n",
                ws_scaled.P[k+1](0,1), ws_scaled.P[k+1](1,0));
            // Compute P*A and Phat*Ahat element-by-element
            printf("\n    P*A element [0,0]: %.8e\n",
                ws_phys.P[k+1](0,0)*phys[k].A(0,0) + ws_phys.P[k+1](0,1)*phys[k].A(1,0) + ws_phys.P[k+1](0,2)*phys[k].A(2,0));
            printf("    Phat*Ahat element [0,0]: %.8e\n",
                ws_scaled.P[k+1](0,0)*scaled[k].A(0,0) + ws_scaled.P[k+1](0,1)*scaled[k].A(1,0) + ws_scaled.P[k+1](0,2)*scaled[k].A(2,0));
            printf("    inv_Lx4*(PA)[0,0]*inv_Lx3: %.8e\n",
                inv_Lx[k+1](0,0) * (ws_phys.P[k+1](0,0)*phys[k].A(0,0) + ws_phys.P[k+1](0,1)*phys[k].A(1,0) + ws_phys.P[k+1](0,2)*phys[k].A(2,0)) * inv_Lx[k](0,0));
            // Check: Phat*Ahat should equal inv_Lx4 * (P*A) * inv_Lx3
            // i.e., Phat(i,m)*Ahat(m,j) = inv_Lx4(i) * P(i,m) * A(m,j) * inv_Lx3(j)
            // For i=0, j=0: sum_m Phat(0,m)*Ahat(m,0) = inv_Lx4(0) * sum_m P(0,m)*A(m,0) * inv_Lx3(0)
            // Wait, that's not right. Let me re-derive:
            // Phat = inv_Lx4 * P * inv_Lx4, Ahat = inv_Lx4 * A * inv_Lx3
            // Phat*Ahat = inv_Lx4 * P * inv_Lx4 * inv_Lx4 * A * inv_Lx3
            //           = inv_Lx4 * P * (inv_Lx4^2) * A * inv_Lx3
            // This is NOT inv_Lx4 * P * A * inv_Lx3 unless inv_Lx4^2 = I!
            printf("\n    *** KEY INSIGHT: Phat*Ahat = inv_Lx4*P*inv_Lx4^2*A*inv_Lx3 ***\n");
            printf("    *** NOT inv_Lx4*(P*A)*inv_Lx3! ***\n");
            printf("    *** inv_Lx4^2(0,0) = %.8e (not 1!) ***\n", inv_Lx[k+1](0,0)*inv_Lx[k+1](0,0));
            double AtPA[NX][NX];
            for (int r = 0; r < NX; ++r) for (int c = 0; c < NX; ++c) {
                double s = 0;
                for (int i = 0; i < NX; ++i) for (int j = 0; j < NX; ++j)
                    s += phys[k].A(i,r) * ws_phys.P[k+1](i,j) * phys[k].A(j,c);
                AtPA[r][c] = s;
            }
            printf("    (A^T*P*A)[0,0]=%.8e  inv_Lx3*(AtPA)*inv_Lx3[0,0]=%.8e\n",
                AtPA[0][0], inv_Lx[k](0,0)*AtPA[0][0]*inv_Lx[k](0,0));
            // Also check: does scaled A^T * scaled P * scaled A give same as AtPA scaled?
            double AtPA_scaled[NX][NX];
            for (int r = 0; r < NX; ++r) for (int c = 0; c < NX; ++c) {
                double s = 0;
                for (int i = 0; i < NX; ++i) for (int j = 0; j < NX; ++j)
                    s += scaled[k].A(i,r) * ws_scaled.P[k+1](i,j) * scaled[k].A(j,c);
                AtPA_scaled[r][c] = s;
            }
            printf("    (Ahat^T*Phat*Ahat)[0,0] direct=%.8e\n", AtPA_scaled[0][0]);
            // Compare element-by-element: AtPA_scaled vs inv_Lx*AtPA*inv_Lx
            printf("    Element comparison (AtPA_scaled vs inv_Lx*AtPA*inv_Lx):\n");
            for (int r = 0; r < NX; ++r) {
                printf("      [%d,0]: scaled=%.8e  expected=%.8e\n", r,
                    AtPA_scaled[r][0], inv_Lx[k](r,r)*AtPA[r][0]*inv_Lx[k](0,0));
            }
        }
        // Print FULL P_4 matrices
        printf("\n  Full P_4 matrices:\n");
        printf("    phys P_4:\n");
        for (int r = 0; r < NX; ++r) {
            printf("      "); for (int c = 0; c < NX; ++c) printf("%.8e ", ws_phys.P[k+1](r,c)); printf("\n");
        }
        printf("    scaled P_4:\n");
        for (int r = 0; r < NX; ++r) {
            printf("      "); for (int c = 0; c < NX; ++c) printf("%.8e ", ws_scaled.P[k+1](r,c)); printf("\n");
        }
        printf("    inv_Lx4*P4*inv_Lx4:\n");
        for (int r = 0; r < NX; ++r) {
            printf("      "); for (int c = 0; c < NX; ++c) printf("%.8e ", inv_Lx[k+1](r,r)*ws_phys.P[k+1](r,c)*inv_Lx[k+1](c,c)); printf("\n");
        }
        // Print phys Qxx_4 for reference
        printf("    phys Qxx_4:\n");
        for (int r = 0; r < NX; ++r) {
            printf("      "); for (int c = 0; c < NX; ++c) printf("%.8e ", phys[k+1].Qxx(r,c)); printf("\n");
        }
    }
    // Print expected scaled values: P_hat = inv_Lx*P*inv_Lx, p_hat = inv_Lx*p
    // K_hat = inv_Lu*K*Lx, d_hat = inv_Lu*d  (Convention A)
    printf("Expected scaled (inv_Lx*P*inv_Lx, inv_Lu*K*Lx, inv_Lu*d):\n");
    for (int k = 0; k <= N; ++k) {
        printf("  k=%d P_diag=", k);
        for (int i = 0; i < NX; ++i) printf("%.8e ", inv_Lx[k](i,i)*ws_phys.P[k](i,i)*inv_Lx[k](i,i));
        printf(" p=", k);
        for (int i = 0; i < NX; ++i) printf("%.8e ", inv_Lx[k](i,i)*ws_phys.p[k][i]);
        if (k < N) {
            printf(" K[0,0]=%.8e d=", inv_Lu[k](0,0)*ws_phys.K[k](0,0)*Lx[k](0,0));
            for (int i = 0; i < NU; ++i) printf("%.8e ", inv_Lu[k](i,i)*ws_phys.d[k][i]);
        }
        printf("\n");
    }
    
    // Recover physical step: dx = inv_Lx * dx̂, du = inv_Lu * dû
    printf("Recovered physical step (from scaled Riccati):\n");
    double max_err_dx = 0, max_err_du = 0;
    for (int k = 0; k <= N; ++k) {
        Vec<NX> dx_rec, du_rec;
        for (int i = 0; i < NX; ++i)
            dx_rec[i] = inv_Lx[k](i,i) * ws_scaled.dx[k][i];
        if (k < N)
            for (int i = 0; i < NU; ++i)
                du_rec[i] = inv_Lu[k](i,i) * ws_scaled.du[k][i];
        
        printf("  k=%d dx=[", k);
        for (int i = 0; i < NX; ++i) printf("%.8f ", dx_rec[i]);
        printf("]");
        if (k < N) {
            printf(" du=[");
            for (int i = 0; i < NU; ++i) printf("%.8f ", du_rec[i]);
            printf("]");
        }
        printf("\n");
        
        for (int i = 0; i < NX; ++i) {
            double err = std::fabs(dx_rec[i] - ws_phys.dx[k][i]);
            if (err > max_err_dx) max_err_dx = err;
        }
        if (k < N)
            for (int i = 0; i < NU; ++i) {
                double err = std::fabs(du_rec[i] - ws_phys.du[k][i]);
                if (err > max_err_du) max_err_du = err;
            }
    }
    
    printf("\n||dx_phys - dx_rec||_inf = %.2e\n", max_err_dx);
    printf("||du_phys - du_rec||_inf = %.2e\n", max_err_du);
    
    double max_err = std::max(max_err_dx, max_err_du);
    printf("\nTest %s (threshold 1e-8)\n", max_err < 1e-8 ? "PASSED" : "FAILED");
    
    if (max_err >= 1e-8) {
        printf("\nPer-stage comparison:\n");
        for (int k = 0; k <= N; ++k) {
            printf("  k=%d: ", k);
            for (int i = 0; i < NX; ++i) {
                double rec = inv_Lx[k](i,i) * ws_scaled.dx[k][i];
                double ref = ws_phys.dx[k][i];
                double err = std::fabs(rec - ref);
                if (err > 1e-8)
                    printf(" dx[%d] err=%.2e(ref=%.6f rec=%.6f)", i, err, ref, rec);
            }
            if (k < N)
                for (int i = 0; i < NU; ++i) {
                    double rec = inv_Lu[k](i,i) * ws_scaled.du[k][i];
                    double ref = ws_phys.du[k][i];
                    double err = std::fabs(rec - ref);
                    if (err > 1e-8)
                        printf(" du[%d] err=%.2e(ref=%.6f rec=%.6f)", i, err, ref, rec);
                }
            printf("\n");
        }
    }
    
    // ===== Full KKT solve (no Riccati) for physical =====
    // dx0 is FIXED to dx0_given, so we substitute it out.
    // Variables: [du_0, dx_1, du_1, ..., dx_N]  (dx0 is NOT a variable)
    const int ns = N + 1;
    const int Nz = N * NU + ns * NX;  // N*NU controls + (N+1)*NX states, but dx0 fixed
    // Actually: variables are du_0(Nu), dx_1(Nx), du_1(Nu), ..., dx_N(Nx)
    // Total = N*NU + N*NX (excluding dx0)
    const int Nz_free = N * NU + N * NX;
    const int Nc = N * NX;  // dynamics constraints
    const int Nk = Nz_free + Nc;
    
    std::vector<double> Kmat(Nk * Nk, 0.0), rhs(Nk, 0.0);
    auto IX = [&](int r, int c) { return r * Nk + c; };
    
    // Variable layout: [du_0(NU), dx_1(NX), du_1(NU), dx_2(NX), ..., dx_N(NX)]
    // then multipliers: [nu_0(NX), nu_1(NX), ..., nu_{N-1}(NX)]
    auto du_offset = [&](int k) { return k * (NU + NX); };        // du_k
    auto dx_offset = [&](int k) { return k * (NU + NX) + NU; };   // dx_k (k>=1)
    auto nu_offset = [&](int k) { return Nz_free + k * NX; };      // nu_k
    
    // Cost: sum_k [0.5*dx_k^T Qxx_k dx_k + du_k^T Qux_k dx_k + 0.5*du_k^T Quu_k du_k + qx_k^T dx_k + qu_k^T du_k]
    // with dx_0 = dx0_given substituted
    
    for (int k = 0; k <= N; ++k) {
        if (k == 0) {
            // dx0 is fixed. Its cost contribution:
            // 0.5*dx0^T Qxx_0 dx0 + dx0^T Qux_0^T du0 + qx_0^T dx0  (constant + linear in du0)
            // Linear in du0: Qux_0 * dx0  → goes to rhs of du0 equation
            // Cross with dx1: none directly
            int duo = du_offset(0);
            for (int i = 0; i < NU; ++i) {
                // Quu block
                for (int j = 0; j < NU; ++j)
                    Kmat[IX(duo+i, duo+j)] += phys[0].Quu(i,j);
                // rhs: -qu_0 - Qux_0 * dx0_given
                double r = -phys[0].qu[i];
                for (int j = 0; j < NX; ++j)
                    r -= phys[0].Qux(i,j) * dx0[j];
                rhs[duo+i] = r;
            }
        } else {
            // Qxx for dx_k
            int dxk = dx_offset(k);
            for (int i = 0; i < NX; ++i) {
                for (int j = 0; j < NX; ++j)
                    Kmat[IX(dxk+i, dxk+j)] += phys[k].Qxx(i,j);
                rhs[dxk+i] = -phys[k].qx[i];
            }
        }
        if (k < N) {
            int duk = du_offset(k);
            // Qux cross: du_k^T Qux_k dx_k
            if (k > 0) {
                int dxk = dx_offset(k);
                for (int i = 0; i < NU; ++i)
                    for (int j = 0; j < NX; ++j) {
                        Kmat[IX(duk+i, dxk+j)] = phys[k].Qux(i,j);
                        Kmat[IX(dxk+j, duk+i)] = phys[k].Qux(i,j);
                    }
            }
            // Qxx for dx_{k+1} (added above when k+1 > 0)
            // Dynamics constraint: dx_{k+1} = A_k*dx_k + B_k*du_k + c_k
            // (for k=0, dx_0 is fixed, so: dx_1 = B_0*du_0 + (A_0*dx0 + c_0))
            int nuk = nu_offset(k);
            int dxk1 = dx_offset(k+1);
            for (int i = 0; i < NX; ++i) {
                // B_k * du_k term
                for (int j = 0; j < NU; ++j) {
                    Kmat[IX(nuk+i, duk+j)] = phys[k].B(i,j);
                    Kmat[IX(duk+j, nuk+i)] = phys[k].B(i,j);
                }
                // A_k * dx_k term (only if k > 0, since dx_0 is fixed)
                if (k > 0) {
                    int dxk = dx_offset(k);
                    for (int j = 0; j < NX; ++j) {
                        Kmat[IX(nuk+i, dxk+j)] = phys[k].A(i,j);
                        Kmat[IX(dxk+j, nuk+i)] = phys[k].A(i,j);
                    }
                }
                // -dx_{k+1} term
                Kmat[IX(nuk+i, dxk1+i)] = -1.0;
                Kmat[IX(dxk1+i, nuk+i)] = -1.0;
                // rhs: -(A_k*dx0_given + c_k) for k=0, or -c_k for k>0
                double r = -phys[k].c[i];
                if (k == 0)
                    for (int j = 0; j < NX; ++j)
                        r -= phys[0].A(i,j) * dx0[j];
                rhs[nuk+i] = r;
            }
        }
    }
    // For k=0 dynamics, dx1 also gets Qxx_1 cross with the constraint
    // Actually need to add Qxx_1 to dx1 block (already done above when k=1)
    // But we also need the dynamics constraint k=0 to properly account for dx1
    // The constraint is: nu_0^T (B_0*du_0 + c_0 + A_0*dx0 - dx_1) = 0
    // This gives: -I on dx1 (done), B_0 on du_0 (done), rhs = -(A_0*dx0+c_0) (done)
    
    // Gaussian elimination
    std::vector<double> aug(Nk * (Nk+1));
    for (int i = 0; i < Nk; ++i) {
        for (int j = 0; j < Nk; ++j) aug[i*(Nk+1)+j] = Kmat[IX(i,j)];
        aug[i*(Nk+1)+Nk] = rhs[i];
    }
    for (int k = 0; k < Nk; ++k) {
        int mx = k;
        for (int i = k+1; i < Nk; ++i)
            if (std::fabs(aug[i*(Nk+1)+k]) > std::fabs(aug[mx*(Nk+1)+k])) mx = i;
        if (mx != k) for (int j = k; j <= Nk; ++j) { double t = aug[k*(Nk+1)+j]; aug[k*(Nk+1)+j] = aug[mx*(Nk+1)+j]; aug[mx*(Nk+1)+j] = t; }
        for (int i = k+1; i < Nk; ++i) {
            double f = aug[i*(Nk+1)+k] / aug[k*(Nk+1)+k];
            for (int j = k; j <= Nk; ++j) aug[i*(Nk+1)+j] -= f * aug[k*(Nk+1)+j];
        }
    }
    std::vector<double> sol(Nk);
    for (int i = Nk-1; i >= 0; --i) {
        sol[i] = aug[i*(Nk+1)+Nk];
        for (int j = i+1; j < Nk; ++j) sol[i] -= aug[i*(Nk+1)+j] * sol[j];
        sol[i] /= aug[i*(Nk+1)+i];
    }
    
    // Extract solution: map back to dx/du format
    // dx0 = dx0_given, du_k = sol[du_offset(k)], dx_k = sol[dx_offset(k)]
    std::vector<double> dx_sol((N+1)*NX, 0.0), du_sol(N*NU, 0.0);
    for (int i = 0; i < NX; ++i) dx_sol[i] = dx0[i];  // dx0
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < NU; ++i)
            du_sol[k*NU+i] = sol[du_offset(k)+i];
    for (int k = 1; k <= N; ++k)
        for (int i = 0; i < NX; ++i)
            dx_sol[k*NX+i] = sol[dx_offset(k)+i];
    
    printf("\nFull KKT physical Newton step:\n");
    for (int k = 0; k <= N; ++k) {
        printf("  k=%d dx=[", k);
        for (int i = 0; i < NX; ++i) printf("%.8f ", dx_sol[k*NX+i]);
        printf("]");
        if (k < N) {
            printf(" du=[");
            for (int i = 0; i < NU; ++i) printf("%.8f ", du_sol[k*NU+i]);
            printf("]");
        }
        printf("\n");
    }
    
    double kkt_err = 0;
    for (int k = 0; k <= N; ++k)
        for (int i = 0; i < NX; ++i) {
            double e = std::fabs(dx_sol[k*NX+i] - ws_phys.dx[k][i]);
            if (e > kkt_err) kkt_err = e;
        }
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < NU; ++i) {
            double e = std::fabs(du_sol[k*NU+i] - ws_phys.du[k][i]);
            if (e > kkt_err) kkt_err = e;
        }
    printf("\n||KKT_full - Riccati_phys||_inf = %.2e\n", kkt_err);
    printf("Riccati %s full KKT\n", kkt_err < 1e-8 ? "MATCHES" : "DOES NOT MATCH");
    
    return max_err < 1e-8 ? 0 : 1;
}
