#pragma once
/**
 * @file    nmpc_barrier_manager.hpp
 * @brief   Barrier update strategy — decides when to reduce μ.
 *
 * Core logic:
 *   1. Compute barrier error:  E_μ = max(primal_inf, compl_inf)
 *   2. If E_μ ≤ κ·μ  →  barrier subproblem solved  →  reduce μ
 *   3. Otherwise       →  not solved yet             →  hold μ fixed
 *
 * μ reduction uses Mehrotra/LOQO adaptive centering:
 *     μ_new = max(σ·μ, κ_μ·μ, μ_min)
 *
 * Dependencies: nmpc_core.hpp only.
 */

#include "nmpc_core.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Barrier update strategy parameters
// ─────────────────────────────────────────────────────────────────────────────

struct BarrierUpdateParams {
    double mu_init     = 1.0;
    double mu_min      = 1e-5;     // barrier parameter floor

    double kappa_eps   = 10.0;     // E_μ ≤ κ·μ → barrier solved
    double kappa_mu    = 0.2;      // monotone reduction cap: μ_new ≥ κ_μ·μ
    int    max_same_mu = 30;       // force reduction after N iters at same μ

    double m_safe      = 0.0;      // complementarity safeguard: s·λ ≥ m_safe·μ
    int    verbosity   = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  BarrierUpdateStrategy
// ─────────────────────────────────────────────────────────────────────────────

class BarrierUpdateStrategy {
public:
    BarrierUpdateStrategy() = default;

    void reset(const BarrierUpdateParams& p) {
        p_ = p;
        iters_at_mu_ = 0;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Convergence gate: is the overall problem solved?
    //  (barrier at minimum AND KKT converged — caller checks KKT)
    // ═══════════════════════════════════════════════════════════════════

    bool at_minimum(double mu) const {
        return mu <= p_.mu_min;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Core update — call once after each accepted step.
    //
    //  Judges if barrier subproblem is solved:
    //    E_μ = max(primal, compl) ≤ κ·μ   AND   stat_inf ≤ stat_gate
    //  If solved (or forced after max_same_mu iters): reduce μ.
    //  If not: keep μ fixed.
    //
    //  Returns true if μ was changed (caller should reset filter).
    // ═══════════════════════════════════════════════════════════════════

    bool update(double& mu, double primal_inf, double compl_inf, double sigma,
                double stat_inf = -1.0) {
        double old_mu = mu;
        iters_at_mu_++;

        double E_mu = std::max(primal_inf, compl_inf);
        bool primal_compl_ok = (E_mu <= p_.kappa_eps * mu);

        // Stationarity gate: only reduce μ when stationarity is reasonable
        // Use a generous gate (100 * mu) to avoid blocking progress entirely,
        // but prevent μ reduction when stationarity is clearly diverging.
        // Skip gate if stat_inf < 0 (caller didn't provide stationarity).
        double stat_gate = 100.0 * mu + 2.0;  // allows stat ≤ 2.0 + 100μ
        bool stat_ok = (stat_inf < 0.0) || (stat_inf <= stat_gate);

        bool solved = (primal_compl_ok && stat_ok);
        bool force  = (iters_at_mu_ >= p_.max_same_mu);

        if (!solved && !force) {
            // Barrier subproblem not yet solved — hold μ fixed
            if (p_.verbosity >= 3)
                printf("  [barrier: HOLD E_mu=%.2e k*mu=%.2e stat=%.2e gate=%.2e iters=%d/%d]\n",
                       E_mu, p_.kappa_eps * mu, stat_inf, stat_gate,
                       iters_at_mu_, p_.max_same_mu);
            return false;
        }

        // Barrier solved (or forced) — reduce μ
        double mu_adaptive = std::max(sigma * mu, p_.mu_min);
        double mu_monotone = std::max(p_.kappa_mu * mu, p_.mu_min);
        mu = std::max(mu_adaptive, mu_monotone);
        iters_at_mu_ = 0;

        if (p_.verbosity >= 2)
            printf("  [barrier: %s E_mu=%.2e, k*mu=%.2e, stat=%.2e, sig*mu=%.2e, kap*mu=%.2e -> mu=%.3e]\n",
                   solved ? "SOLVED" : "FORCE", E_mu, p_.kappa_eps * mu,
                   stat_inf, mu_adaptive, mu_monotone, mu);
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Accessors
    // ═══════════════════════════════════════════════════════════════════

    double m_safe()      const { return p_.m_safe; }
    double mu_min()      const { return p_.mu_min; }
    double kappa_eps()   const { return p_.kappa_eps; }
    int    iters_at_mu() const { return iters_at_mu_; }

private:
    BarrierUpdateParams p_;
    int iters_at_mu_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Legacy: BarrierManager (kept for backward compatibility)
// ─────────────────────────────────────────────────────────────────────────────

struct BarrierParams {
    double mu_init       = 1.0;
    double mu_min        = 1e-5;
    double kappa_eps     = 10.0;
    double kappa_mu      = 0.2;
    int    max_same_mu   = 30;
};

class BarrierManager {
public:
    BarrierManager() = default;

    void init(const BarrierParams& params = BarrierParams{}) {
        params_ = params;
        mu_     = params_.mu_init;
        iters_at_mu_ = 0;
    }

    double update(double primal_inf, double compl_inf, double sigma) {
        iters_at_mu_++;
        double E_mu = std::max(primal_inf, compl_inf);
        bool barrier_solved = (E_mu <= params_.kappa_eps * mu_);
        bool force_reduce = (iters_at_mu_ >= params_.max_same_mu);

        if (barrier_solved || force_reduce) {
            double mu_adaptive = std::max(sigma * mu_, params_.mu_min);
            double mu_monotone = std::max(params_.kappa_mu * mu_, params_.mu_min);
            mu_ = std::max(mu_adaptive, mu_monotone);
            iters_at_mu_ = 0;
        }
        return mu_;
    }

    bool   is_at_min()     const { return mu_ <= params_.mu_min; }
    double mu()            const { return mu_; }
    int    iters_at_mu()   const { return iters_at_mu_; }
    double kappa_eps()     const { return params_.kappa_eps; }
    double mu_min()        const { return params_.mu_min; }

    void reset() {
        mu_ = params_.mu_init;
        iters_at_mu_ = 0;
    }

private:
    BarrierParams params_;
    double mu_ = 1.0;
    int    iters_at_mu_ = 0;
};

inline bool is_barrier_solved(double primal_inf, double compl_inf,
                               double mu, double kappa_eps = 10.0) {
    double E_mu = std::max(primal_inf, compl_inf);
    return E_mu <= kappa_eps * mu;
}

} // namespace nmpc
