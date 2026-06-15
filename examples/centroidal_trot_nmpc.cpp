/**
 * @file    centroidal_trot_nmpc.cpp
 * @brief   3D centroidal quadruped trotting NMPC benchmark with explicit contact schedule.
 *
 * Based on: Zanelli et al. (CDC 2017) — designed as a head-to-head solver benchmark
 * against acados.  Features nonlinear rigid-body dynamics, changing contact sets,
 * friction-cone inequalities, force saturation, and aggressive reference tracking.
 *
 * State:   x = [p_B(3), v_B(3), q_WB(4), omega_B(3)]     (13D)
 * Control: u = [f_FL(3), f_FR(3), f_RL(3), f_RR(3)]      (12D)
 *
 * Continuous dynamics (centroidal rigid body, 4 point contacts):
 *   p_B_dot = v_B
 *   m * v_B_dot = sum_i f_i + m*g
 *   q_dot = 0.5 * q (x) [0; omega_B]   (quaternion kinematics)
 *   I * omega_B_dot = sum_i (r_i - p_B) x f_i  -  omega_B x (I * omega_B)
 *
 * Contact schedule: explicit trot (diagonal pairs), T=1.0s, N=50, h=0.02s.
 * Constraints: friction cones (box approx), force saturation, state bounds.
 */

#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "nmpc/nmpc_solver_paper.hpp"
#include "trajectory_dump.hpp"

using namespace nmpc;

// == Dimensions ================================================================
constexpr int NX = 13;   // [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
constexpr int NU = 12;   // [fFL_x,y,z, fFR_x,y,z, fRL_x,y,z, fRR_x,y,z]
constexpr int NC = 34;   // 20 active-foot + 6 inactive-foot + 8 state bounds
constexpr int N  = 50;   // horizon length

// == Physical parameters =======================================================
constexpr double BODY_MASS  = 12.0;                          // kg
constexpr double IXX = 0.08, IYY = 0.32, IZZ = 0.35;       // principal inertia (kg*m^2)
constexpr double GRAV_Z     = -9.81;                         // m/s^2
constexpr double MU_FRIC    = 0.7;                           // friction coefficient
constexpr double FZ_MAX     = 120.0;                         // max normal force (N)

// == Horizon / discretization ==================================================
constexpr double T_HORIZON  = 1.0;   // seconds
constexpr double DT = T_HORIZON / N; // 0.02 s

// == State / control indices ===================================================
// x:  0-2  p_B     3-5  v_B     6-9  q_WB [w,x,y,z]    10-12  omega_B
// u:  0-2  f_FL    3-5  f_FR    6-8  f_RL               9-11   f_RR

// == Contact schedule: trot (diagonal pairs) ===================================
//   Phase 0  [0, 12)    FL + RR active
//   Phase 1  [12, 25)   FR + RL active
//   Phase 2  [25, 37)   FL + RR active
//   Phase 3  [37, 50]   FR + RL active
static constexpr int PH_BOUNDS[4][2] = {{0,12},{12,25},{25,37},{37,51}};
static constexpr int PH_FEET  [4][2] = {{0,3}, {1,2},  {0,3},  {1,2} };

// Fixed footholds r_i in world frame (per phase, per active foot)
static constexpr double RF[4][2][3] = {
    {{0.02, 0.19, 0.0}, {0.02,-0.19, 0.0}},   // Phase 0: FL, RR
    {{0.22,-0.19, 0.0}, {0.22, 0.19, 0.0}},   // Phase 1: FR, RL
    {{0.42, 0.19, 0.0}, {0.42,-0.19, 0.0}},   // Phase 2: FL, RR
    {{0.62,-0.19, 0.0}, {0.62, 0.19, 0.0}},   // Phase 3: FR, RL
};

// == Constraint index map ======================================================
// Layout (always):
//   [0..9]   = first active foot  (10 constraints)
//   [10..19] = second active foot (10 constraints)
//   [20..22] = first inactive foot  (3 constraints: -fx, fx, -fz)
//   [23..25] = second inactive foot (3 constraints: -fx, fx, -fz)
//   [26..33] = state bounds (8 constraints)
//
// Which foot maps to which slot depends on the phase.
// ==============================================================================

// == Type aliases ===============================================================
using VecX = Vec<NX>; using VecU = Vec<NU>; using VecC = Vec<NC>;
using MatXX = Mat<NX,NX>; using MatXU = Mat<NX,NU>;
using MatCX = Mat<NC,NX>; using MatCU = Mat<NC,NU>;

// == Helpers ====================================================================
static int get_phase(int k) {
    for (int p = 0; p < 4; ++p)
        if (k >= PH_BOUNDS[p][0] && k < PH_BOUNDS[p][1]) return p;
    return 3;
}
// ==============================================================================
//  Centroidal dynamics — RK4 integration + quaternion projection
//
//  Time-varying dynamics via k-aware overloads: the contact footholds
//  change between phases, requiring the stage index k.  The k-free
//  fallback uses Phase-0 footholds for backward compatibility.
// ==============================================================================
struct CentroidalDyn : DynamicsModel<NX, NU> {

    // Core continuous dynamics with phase-dependent footholds
    void continuous_phase(int k, const VecX& s, const VecU& c, VecX& dx) const {
        const double qw=s[6], qx=s[7], qy=s[8], qz=s[9];
        const double wx=s[10], wy=s[11], wz=s[12];

        // p_dot = v
        dx[0]=s[3]; dx[1]=s[4]; dx[2]=s[5];

        // Sum ground-reaction forces
        double fx=0, fy=0, fz=0;
        for (int i=0; i<4; ++i) { fx+=c[i*3]; fy+=c[i*3+1]; fz+=c[i*3+2]; }

        // v_dot = (sum_f)/m + g
        dx[3] = fx / BODY_MASS;
        dx[4] = fy / BODY_MASS;
        dx[5] = fz / BODY_MASS + GRAV_Z;

        // q_dot = 0.5 * q (x) [0; omega]   (Hamilton product)
        dx[6] = 0.5*(-qx*wx - qy*wy - qz*wz);
        dx[7] = 0.5*( qw*wx + qy*wz - qz*wy);
        dx[8] = 0.5*( qw*wy - qx*wz + qz*wx);
        dx[9] = 0.5*( qw*wz + qx*wy - qy*wx);

        // Sum contact torques:  tau = sum_i (r_i(k) - p_B) x f_i
        // Only active feet have non-zero force, but we compute for all
        // (inactive forces are zero by constraint, so no torque contribution)
        int ph = get_phase(k);
        double taux=0, tauy=0, tauz=0;
        for (int i=0; i<4; ++i) {
            double cfx=c[i*3], cfy=c[i*3+1], cfz=c[i*3+2];
            if (std::fabs(cfx)+std::fabs(cfy)+std::fabs(cfz) < 1e-12) continue;

            // Find foothold for this foot in the current phase
            int sl = -1;
            if      (i == PH_FEET[ph][0]) sl = 0;
            else if (i == PH_FEET[ph][1]) sl = 1;
            // If foot is inactive, use the other active foot's foothold x-coord
            // at the same z (approximate — but force is zero anyway)
            double rfx, rfy, rfz;
            if (sl >= 0) {
                rfx = RF[ph][sl][0]; rfy = RF[ph][sl][1]; rfz = RF[ph][sl][2];
            } else {
                // Inactive foot: use a nominal foothold (force ~ 0, so this
                // doesn't affect the torque — just avoids reading garbage)
                rfx = s[0]; rfy = (i%2==0) ? 0.19 : -0.19; rfz = 0.0;
            }
            double rx = rfx - s[0], ry = rfy - s[1], rz = rfz - s[2];
            taux += ry*cfz - rz*cfy;
            tauy += rz*cfx - rx*cfz;
            tauz += rx*cfy - ry*cfx;
        }

        // Euler equation:  I * omega_dot = tau - omega x (I * omega)
        double Iwx = IXX*wx, Iwy = IYY*wy, Iwz = IZZ*wz;
        dx[10] = (taux - (wy*Iwz - wz*Iwy)) / IXX;
        dx[11] = (tauy - (wz*Iwx - wx*Iwz)) / IYY;
        dx[12] = (tauz - (wx*Iwy - wy*Iwx)) / IZZ;
    }

    // RK4 helper: integrate using a specific continuous dynamics function
    template <typename F>
    static void rk4(F& f, const VecX& x, const VecU& u, double dt, VecX& nx) {
        VecX k1,k2,k3,k4,tmp;
        f(x,u,k1);
        for(int i=0;i<NX;++i) tmp[i]=x[i]+0.5*dt*k1[i]; f(tmp,u,k2);
        for(int i=0;i<NX;++i) tmp[i]=x[i]+0.5*dt*k2[i]; f(tmp,u,k3);
        for(int i=0;i<NX;++i) tmp[i]=x[i]+dt*k3[i];     f(tmp,u,k4);
        for(int i=0;i<NX;++i)
            nx[i]=x[i]+(dt/6.0)*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
        // Quaternion normalisation
        double qn = std::sqrt(nx[6]*nx[6]+nx[7]*nx[7]+nx[8]*nx[8]+nx[9]*nx[9]);
        if (qn > 1e-10) {
            double inv = 1.0/qn;
            nx[6]*=inv; nx[7]*=inv; nx[8]*=inv; nx[9]*=inv;
        }
    }

    // k-free fallback (Phase 0 footholds)
    Status discrete_step(const VecX& x, const VecU& u, double dt,
                         VecX& nx) override {
        auto f = [this](const VecX& s, const VecU& c, VecX& d) {
            continuous_phase(0, s, c, d);
        };
        rk4(f, x, u, dt, nx);
        return Status::SUCCESS;
    }

    Status linearize(const VecX& x, const VecU& u, double dt,
                     MatXX& A, MatXU& B) override {
        return linearize(x, u, dt, A, B, 0);  // delegate to k-aware
    }

    // ── k-aware overloads (phase-dependent footholds) ──────────────

    Status discrete_step(const VecX& x, const VecU& u, double dt,
                         VecX& nx, int k) override {
        auto f = [this, k](const VecX& s, const VecU& c, VecX& d) {
            continuous_phase(k, s, c, d);
        };
        rk4(f, x, u, dt, nx);
        return Status::SUCCESS;
    }

    Status linearize(const VecX& x, const VecU& u, double dt,
                     MatXX& A, MatXU& B, int k) override {
        const double eps = 1e-6;
        VecX xp,xm,fp,fm;
        for (int j=0; j<NX; ++j) {
            xp=x; xp[j]+=eps; xm=x; xm[j]-=eps;
            discrete_step(xp,u,dt,fp,k); discrete_step(xm,u,dt,fm,k);
            for (int i=0; i<NX; ++i) A(i,j)=(fp[i]-fm[i])/(2*eps);
        }
        VecU up,um;
        for (int j=0; j<NU; ++j) {
            up=u; up[j]+=eps; um=u; um[j]-=eps;
            discrete_step(x,up,dt,fp,k); discrete_step(x,um,dt,fm,k);
            for (int i=0; i<NX; ++i) B(i,j)=(fp[i]-fm[i])/(2*eps);
        }
        return Status::SUCCESS;
    }
};

// ==============================================================================
//  Cost model — reference tracking with time-varying position reference
// ==============================================================================
struct TrotCost : CostModel<NX, NU> {
    // Weights: Qp=diag(40,40,80), Qv=diag(8,8,12), Qq=diag(20,20,10,10),
    //          Qw=diag(2,2,1),    R=1e-3*I_12
    static constexpr double Qp[3]  = {40.0, 40.0, 80.0};
    static constexpr double Qv[3]  = { 8.0,  8.0, 12.0};
    static constexpr double Qq[4]  = {20.0, 20.0, 10.0, 10.0};
    static constexpr double Qw[3]  = { 2.0,  2.0,  1.0};
    static constexpr double Rf     = 1e-3;
    static constexpr double TSCL   = 7.0;   // terminal ≈ 7x stage

    // Reference: p_ref(t) = [0.8t, 0, 0.28],  v_ref = [0.8, 0, 0]
    static void ref_at(int k, double pref[3]) {
        double t = k * DT;
        pref[0] = 0.8 * t;  pref[1] = 0.0;  pref[2] = 0.28;
    }

    double stage_cost(const VecX& x, const VecU& u, int k) override {
        double pr[3]; ref_at(k, pr);
        double c = 0;
        for (int i=0;i<3;++i) { double d=x[i]-pr[i]; c+=Qp[i]*d*d; }
        double dvx=x[3]-0.8;
        c += Qv[0]*dvx*dvx + Qv[1]*x[4]*x[4] + Qv[2]*x[5]*x[5];
        double qw_e=x[6]-1;
        c += Qq[0]*qw_e*qw_e + Qq[1]*x[7]*x[7] + Qq[2]*x[8]*x[8] + Qq[3]*x[9]*x[9];
        for (int i=0;i<3;++i) c += Qw[i]*x[10+i]*x[10+i];
        for (int i=0;i<NU;++i) c += Rf*u[i]*u[i];
        return 0.5*c;
    }

    double terminal_cost(const VecX& x) override {
        double pr[3] = {0.8, 0.0, 0.28};
        double c = 0;
        for (int i=0;i<3;++i) { double d=x[i]-pr[i]; c+=Qp[i]*d*d; }
        c += Qv[0]*x[3]*x[3] + Qv[1]*x[4]*x[4] + Qv[2]*x[5]*x[5];
        double qw_e=x[6]-1;
        c += Qq[0]*qw_e*qw_e + Qq[1]*x[7]*x[7] + Qq[2]*x[8]*x[8] + Qq[3]*x[9]*x[9];
        for (int i=0;i<3;++i) c += Qw[i]*x[10+i]*x[10+i];
        return 0.5*TSCL*c;
    }

    Status stage_gradient(const VecX& x, const VecU& u, int k,
                          VecX& qx, VecU& qu) override {
        double pr[3]; ref_at(k, pr);
        for (int i=0;i<3;++i) qx[i]   = Qp[i]*(x[i]-pr[i]);
        qx[3] = Qv[0]*(x[3]-0.8);
        qx[4] = Qv[1]*x[4];
        qx[5] = Qv[2]*x[5];
        qx[6] = Qq[0]*(x[6]-1);
        qx[7] = Qq[1]*x[7];  qx[8] = Qq[2]*x[8];  qx[9] = Qq[3]*x[9];
        for (int i=0;i<3;++i) qx[10+i] = Qw[i]*x[10+i];
        for (int i=0;i<NU;++i) qu[i]   = Rf*u[i];
        return Status::SUCCESS;
    }

    Status stage_hessian(const VecX&, const VecU&, int,
                         MatXX& Qxx, Mat<NU,NU>& Quu,
                         Mat<NU,NX>& Qux) override {
        Qxx.zero();
        for (int i=0;i<3;++i) Qxx(i,i)=Qp[i];
        for (int i=0;i<3;++i) Qxx(3+i,3+i)=Qv[i];
        for (int i=0;i<4;++i) Qxx(6+i,6+i)=Qq[i];
        for (int i=0;i<3;++i) Qxx(10+i,10+i)=Qw[i];
        Quu.zero();
        for (int i=0;i<NU;++i) Quu(i,i)=Rf;
        Qux.zero();
        return Status::SUCCESS;
    }

    Status terminal_gradient(const VecX& x, VecX& qx) override {
        double pr[3] = {0.8, 0.0, 0.28};
        for (int i=0;i<3;++i) qx[i]   = TSCL*Qp[i]*(x[i]-pr[i]);
        qx[3] = TSCL*Qv[0]*x[3];  qx[4] = TSCL*Qv[1]*x[4];  qx[5] = TSCL*Qv[2]*x[5];
        qx[6] = TSCL*Qq[0]*(x[6]-1);
        qx[7] = TSCL*Qq[1]*x[7];
        qx[8] = TSCL*Qq[2]*x[8];
        qx[9] = TSCL*Qq[3]*x[9];
        for (int i=0;i<3;++i) qx[10+i] = TSCL*Qw[i]*x[10+i];
        return Status::SUCCESS;
    }

    Status terminal_hessian(const VecX&, MatXX& Qxx) override {
        Qxx.zero();
        for (int i=0;i<3;++i) Qxx(i,i)      = TSCL*Qp[i];
        for (int i=0;i<3;++i) Qxx(3+i,3+i)   = TSCL*Qv[i];
        for (int i=0;i<4;++i) Qxx(6+i,6+i)   = TSCL*Qq[i];
        for (int i=0;i<3;++i) Qxx(10+i,10+i) = TSCL*Qw[i];
        return Status::SUCCESS;
    }
};

// ==============================================================================
//  Constraint model — contact schedule + friction cones + state bounds
//
//  Layout (g <= 0):
//    For each foot i = 0..3:
//      if active (10 constraints):
//        -f_z              (normal force >= 0)
//        f_z - FZ_MAX      (normal force <= 120)
//        f_x - mu*f_z      (friction x+)
//        -f_x - mu*f_z     (friction x-)
//        f_y - mu*f_z      (friction y+)
//        -f_y - mu*f_z     (friction y-)
//        -f_x - FXY        (component bound)
//        f_x - FXY         (component bound)
//        -f_y - FXY        (component bound)
//        f_y - FXY         (component bound)
//      if inactive (3 constraints):
//        -f_x, f_x, -f_z   (zero force)
//    State bounds (8 constraints):
//      PZ_MIN - p_z        (height >= 0.22)
//      p_z - PZ_MAX        (height <= 0.35)
//      -phi - PHI_MAX      (|roll|  <= 0.35)
//      phi - PHI_MAX
//      -theta - PHI_MAX    (|pitch| <= 0.35)
//      theta - PHI_MAX
//      omega_z - OMEGA_MAX   (yaw rate <= 8 rad/s)
//      -omega_z - OMEGA_MAX
//
//  Total: 2 active*10 + 2 inactive*3 + 8 state = 20 + 6 + 8 = 34
// ==============================================================================

struct TrotCons : ConstraintModel<NX, NU, NC> {
    static constexpr double PZ_MIN = 0.22, PZ_MAX = 0.35;
    static constexpr double PHI_MAX = 0.7;    // rad  (roll and pitch)
    static constexpr double OMEGA_MAX = 8.0;   // rad/s (yaw rate bound)
    static constexpr double FXY_MAX = MU_FRIC * FZ_MAX;  // 84 N
    static constexpr int CB_S = 26;  // state bounds start

    // Get constraint base index for a foot given phase k
    // Active feet -> slot 0 or 10 (in order PH_FEET[ph][0]=0, PH_FEET[ph][1]=10)
    // Inactive feet -> slot 20 or 23 (in encounter order)
    static void foot_slots(int ph, int foot_base[4]) {
        int inactive_idx = 0;
        for (int fi = 0; fi < 4; ++fi) {
            if (fi == PH_FEET[ph][0]) {
                foot_base[fi] = 0;
            } else if (fi == PH_FEET[ph][1]) {
                foot_base[fi] = 10;
            } else {
                foot_base[fi] = 20 + inactive_idx * 3;  // 20 or 23
                ++inactive_idx;
            }
        }
    }

    Status evaluate(const VecX& x, const VecU& u, int k, VecC& g) override {
        int ph = get_phase(k);
        int fb[4];
        foot_slots(ph, fb);

        for (int fi = 0; fi < 4; ++fi) {
            int off = fi * 3;
            double fx = u[off], fy = u[off+1], fz = u[off+2];
            int b = fb[fi];
            if (fi == PH_FEET[ph][0] || fi == PH_FEET[ph][1]) {
                // Active foot (10 constraints)
                g[b+0] = -fz;
                g[b+1] = fz - FZ_MAX;
                g[b+2] = fx - MU_FRIC*fz;
                g[b+3] = -fx - MU_FRIC*fz;
                g[b+4] = fy - MU_FRIC*fz;
                g[b+5] = -fy - MU_FRIC*fz;
                g[b+6] = -fx - FXY_MAX;
                g[b+7] = fx - FXY_MAX;
                g[b+8] = -fy - FXY_MAX;
                g[b+9] = fy - FXY_MAX;
            } else {
                // Inactive foot (3 constraints): zero force
                g[b+0] = -fx;
                g[b+1] = fx;
                g[b+2] = -fz;
            }
        }
        // -- State bounds --
        int s = CB_S;
        g[s+0] = PZ_MIN - x[2];
        g[s+1] = x[2] - PZ_MAX;
        double roll  = 2.0*(x[6]*x[7] + x[8]*x[9]);
        double pitch = 2.0*(x[6]*x[8] - x[7]*x[9]);
        g[s+2] = -roll  - PHI_MAX;
        g[s+3] = roll   - PHI_MAX;
        g[s+4] = -pitch - PHI_MAX;
        g[s+5] = pitch  - PHI_MAX;
        g[s+6] = -x[12] - OMEGA_MAX;
        g[s+7] = x[12] - OMEGA_MAX;
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const VecX& x, VecC& g) override {
        for (int i = 0; i < CB_S; ++i) g[i] = -1e10;   // no forces at terminal
        int s = CB_S;
        g[s+0] = PZ_MIN - x[2];
        g[s+1] = x[2] - PZ_MAX;
        double roll  = 2.0*(x[6]*x[7] + x[8]*x[9]);
        double pitch = 2.0*(x[6]*x[8] - x[7]*x[9]);
        g[s+2] = -roll  - PHI_MAX;
        g[s+3] = roll   - PHI_MAX;
        g[s+4] = -pitch - PHI_MAX;
        g[s+5] = pitch  - PHI_MAX;
        g[s+6] = -x[12] - OMEGA_MAX;
        g[s+7] = x[12] - OMEGA_MAX;
        return Status::SUCCESS;
    }

    Status jacobian(const VecX& x, const VecU&, int k,
                    MatCX& Cx, MatCU& Cu) override {
        Cx.zero(); Cu.zero();
        int ph = get_phase(k);
        int fb[4];
        foot_slots(ph, fb);

        for (int fi = 0; fi < 4; ++fi) {
            int off = fi * 3;
            int b = fb[fi];
            if (fi == PH_FEET[ph][0] || fi == PH_FEET[ph][1]) {
                // Active foot Jacobians
                Cu(b+0, off+2) = -1.0;
                Cu(b+1, off+2) = 1.0;
                Cu(b+2, off+0) = 1.0;  Cu(b+2, off+2) = -MU_FRIC;
                Cu(b+3, off+0) = -1.0; Cu(b+3, off+2) = -MU_FRIC;
                Cu(b+4, off+1) = 1.0;  Cu(b+4, off+2) = -MU_FRIC;
                Cu(b+5, off+1) = -1.0; Cu(b+5, off+2) = -MU_FRIC;
                Cu(b+6, off+0) = -1.0;
                Cu(b+7, off+0) = 1.0;
                Cu(b+8, off+1) = -1.0;
                Cu(b+9, off+1) = 1.0;
            } else {
                // Inactive foot Jacobians
                Cu(b+0, off+0) = -1.0;
                Cu(b+1, off+0) =  1.0;
                Cu(b+2, off+2) = -1.0;
            }
        }

        // -- State bound Jacobians --
        int s = CB_S;
        Cx(s+0, 2) = -1.0;
        Cx(s+1, 2) =  1.0;
        Cx(s+2, 6) = -2.0*x[7];  Cx(s+2, 7) = -2.0*x[6];
        Cx(s+2, 8) = -2.0*x[9];  Cx(s+2, 9) = -2.0*x[8];
        Cx(s+3, 6) =  2.0*x[7];  Cx(s+3, 7) =  2.0*x[6];
        Cx(s+3, 8) =  2.0*x[9];  Cx(s+3, 9) =  2.0*x[8];
        Cx(s+4, 6) = -2.0*x[8];  Cx(s+4, 7) =  2.0*x[9];
        Cx(s+4, 8) = -2.0*x[6];  Cx(s+4, 9) =  2.0*x[7];
        Cx(s+5, 6) =  2.0*x[8];  Cx(s+5, 7) = -2.0*x[9];
        Cx(s+5, 8) =  2.0*x[6];  Cx(s+5, 9) = -2.0*x[7];
        Cx(s+6, 12) = -1.0;
        Cx(s+7, 12) =  1.0;
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const VecX& x, MatCX& Cx) override {
        Cx.zero();
        int s = CB_S;
        Cx(s+0, 2) = -1.0;
        Cx(s+1, 2) =  1.0;
        Cx(s+2, 6) = -2.0*x[7];  Cx(s+2, 7) = -2.0*x[6];
        Cx(s+2, 8) = -2.0*x[9];  Cx(s+2, 9) = -2.0*x[8];
        Cx(s+3, 6) =  2.0*x[7];  Cx(s+3, 7) =  2.0*x[6];
        Cx(s+3, 8) =  2.0*x[9];  Cx(s+3, 9) =  2.0*x[8];
        Cx(s+4, 6) = -2.0*x[8];  Cx(s+4, 7) =  2.0*x[9];
        Cx(s+4, 8) = -2.0*x[6];  Cx(s+4, 9) =  2.0*x[7];
        Cx(s+5, 6) =  2.0*x[8];  Cx(s+5, 7) = -2.0*x[9];
        Cx(s+5, 8) =  2.0*x[6];  Cx(s+5, 9) = -2.0*x[7];
        Cx(s+6, 12) = -1.0;
        Cx(s+7, 12) =  1.0;
        return Status::SUCCESS;
    }
};

// ==============================================================================
//  Main: setup, warm-start, solve, output
// ==============================================================================
int main() {
    printf("=======================================================\n");
    printf("  3D Centroidal Trot NMPC Benchmark\n");
    printf("  NX=%d  NU=%d  NC=%d  N=%d  dt=%.3f\n", NX, NU, NC, N, DT);
    printf("=======================================================\n\n");

    CentroidalDyn dyn;
    TrotCost  cost;
    TrotCons  cons;

    using Problem = NMPCProblem<NX, NU, NC, N>;
    auto prob_ptr = new Problem();
    Problem& prob = *prob_ptr;
    prob.dynamics    = &dyn;
    prob.cost        = &cost;
    prob.constraints = &cons;
    prob.dt          = DT;

    // --- Initial condition (perturbed) ---
    VecX x0;
    x0[0] = -0.05;       // px offset
    x0[1] =  0.0;
    x0[2] =  0.28;       // pz nominal
    x0[3] =  0.2;        // vx perturbation
    x0[4] =  0.0;  x0[5] = 0.0;
    // q(0) ≈ identity with small pitch θ=0.05 rad
    // q = [cos(θ/2), 0, sin(θ/2), 0]
    double ht = 0.025;
    x0[6] = std::cos(ht);  // qw
    x0[7] = 0.0;           // qx
    x0[8] = std::sin(ht);  // qy  (pitch)
    x0[9] = 0.0;           // qz
    x0[10] = 0.0; x0[11] = 0.0; x0[12] = 0.0;   // omega = 0

    prob.x0 = x0;

    // --- Warm-start: simulate forward with gravity compensation stance ---
    constexpr double HZ = 0.5 * BODY_MASS * (-GRAV_Z);   // ~58.86 N per active foot
    prob.stages[0].x = x0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u.zero();
        int ph = get_phase(k);
        for (int s = 0; s < 2; ++s) {
            int fi = PH_FEET[ph][s];
            prob.stages[k].u[fi*3 + 2] = HZ;   // fz = mg/2
        }
        // Simulate forward for dynamically feasible guess
        VecX nx;
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, nx, k);
        prob.stages[k + 1].x = nx;
    }
    prob.stages[N].u.zero();

    // --- Solver configuration (heap-allocated: ~35 MB workspace) ---
    auto solver = new NMPCSolverPaper<NX, NU, NC, N>();
    PaperIPMParams pp;
    pp.mu_init     = 10.0;
    pp.max_iters   = 300;
    pp.tol_primal  = 1e-2;
    pp.tol_compl   = 5e-2;
    pp.tol_ineq    = 1e-2;
    pp.tol_stat    = 1.0;  // relative stationarity (large cost/costate terms cancel)
    pp.kappa_eps   = 10.0;
    pp.max_same_mu = 15;
    pp.tau         = 0.99;
    pp.soc_max     = 4;
    pp.verbosity   = 1;
    solver->configure(pp);

    // --- Solve ---
    auto t0 = std::chrono::high_resolution_clock::now();
    Status st = solver->solve(prob);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    const auto& r = solver->last_stats();
    printf("\n===== SOLVE COMPLETE =====\n");
    printf("Status:          %s\n",    status_string(st));
    printf("Iterations:      %d\n",    r.inner_iterations);
    printf("Final mu:        %.3e\n",  r.barrier_param);
    printf("Primal infeas:   %.3e\n",  r.primal_infeas);
    printf("Dual infeas:     %.3e\n",  r.dual_infeas);
    printf("Complementarity: %.3e\n",  r.complementarity);
    printf("Ineq viol est:   %.3e\n",  r.condition_estimate);
    printf("SOC steps:       %d\n",    r.soc_steps);
    printf("Penalty weight:  %.1f\n",  r.penalty_weight);
    printf("Regularization:  %.3e\n",  r.regularization);
    printf("Cost:            %.4f\n",  r.cost);
    printf("Solve time:      %.3f ms\n", ms);

    printf("\nFirst u* (N):\n");
    const char* fn[4] = {"FL","FR","RL","RR"};
    for (int i=0; i<4; ++i)
        printf("  %s: [%.2f, %.2f, %.2f]\n", fn[i],
               prob.stages[0].u[i*3], prob.stages[0].u[i*3+1],
               prob.stages[0].u[i*3+2]);
    printf("COM pos: [%.4f, %.4f, %.4f]\n",
           prob.stages[0].x[0], prob.stages[0].x[1], prob.stages[0].x[2]);
    printf("COM vel: [%.4f, %.4f, %.4f]\n",
           prob.stages[0].x[3], prob.stages[0].x[4], prob.stages[0].x[5]);

    // --- Dump trajectory ---
    print_trajectory_table(prob);
    dump_trajectory_json(prob,
                         "benchmarks/data/contactipm_centroidal_trot.json",
                         "centroidal_trot", r.inner_iterations, r.cost, DT);

    // --- Constraint violation summary ---
    VecC gv;
    double max_viol = 0.0;
    int viol_k = -1, viol_i = -1;
    for (int k = 0; k <= N; ++k) {
        if (k < N)
            cons.evaluate(prob.stages[k].x, prob.stages[k].u, k, gv);
        else
            cons.evaluate_terminal(prob.stages[k].x, gv);
        for (int i = 0; i < NC; ++i) {
            if (gv[i] > -1e8 && gv[i] > max_viol) {
                max_viol = gv[i]; viol_k = k; viol_i = i;
            }
        }
    }
    printf("\nMax constraint violation: %.4e (k=%d, i=%d)\n",
           max_viol, viol_k, viol_i);

    delete solver;
    delete prob_ptr;
    return 0;
}
