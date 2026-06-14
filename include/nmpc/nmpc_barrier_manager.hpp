#pragma once
/**
 * @file    nmpc_barrier_manager.hpp
 * @brief   σ-modulation adaptive barrier update strategy.
 *
 * Instead of fixed κ parameters, μ reduction uses the Mehrotra centering
 * parameter σ with an exponent modulated by subproblem difficulty:
 *
 *   Easy  (≤ fast_threshold iters):  σ_eff = σ^1.5   — more aggressive
 *   Normal (between thresholds):     σ_eff = σ        — standard Mehrotra
 *   Hard  (≥ slow_threshold iters):  σ_eff = σ^0.5   — more conservative
 *
 * This is self-tuning: σ already encodes complementarity quality, so
 * modulating its exponent preserves robustness while adapting speed.
 *
 * μ reduction:  μ_new = max(σ_eff · μ, μ_min)
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
    int    max_same_mu = 30;       // force reduction after N iters at same μ

    // σ-modulation exponents (difficulty-aware)
    double sigma_exp_easy   = 1.5;  // easy subproblems: σ^1.5 (more aggressive)
    double sigma_exp_normal = 1.0;  // standard Mehrotra
    double sigma_exp_hard   = 0.5;  // hard subproblems: σ^0.5 (conservative)

    // Classification thresholds (iterations at current μ)
    int fast_threshold = 2;   // ≤ this → easy
    int slow_threshold = 4;   // ≥ this → hard

    double m_safe      = 0.0;  // complementarity safeguard: s·λ ≥ m_safe·μ
    int    verbosity   = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  BarrierUpdateStrategy  (σ-modulation)
// ─────────────────────────────────────────────────────────────────────────────

class BarrierUpdateStrategy {
public:
    BarrierUpdateStrategy() = default;

    void reset(const BarrierUpdateParams& p) {
        p_ = p;
        iters_at_mu_ = 0;
        last_sigma_exp_ = p.sigma_exp_normal;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Convergence gate: is the overall problem solved?
    // ═══════════════════════════════════════════════════════════════════

    bool at_minimum(double mu) const {
        return mu <= p_.mu_min;
    }

    double last_sigma_exp() const { return last_sigma_exp_; }

    // ═══════════════════════════════════════════════════════════════════
    //  Core update — call once after each accepted Newton step.
    //
    //  1. Check if barrier subproblem is solved (E_μ ≤ κ·μ, stat OK)
    //  2. If not solved: hold μ fixed (return false)
    //  3. If solved or forced:
    //     a. Choose σ exponent based on iters_at_mu_
    //     b. Compute σ_eff = σ^exponent
    //     c. μ_new = max(σ_eff · μ, μ_min)
    //     d. Return true (caller should reset filter)
    // ═══════════════════════════════════════════════════════════════════

    bool update(double& mu, double primal_inf, double compl_inf, double sigma,
                double stat_inf = -1.0, bool mehrotra_healthy = true) {
        iters_at_mu_++;

        double E_mu = std::max(primal_inf, compl_inf);
        bool primal_compl_ok = (E_mu <= p_.kappa_eps * mu);

        // Stationarity gate: only reduce μ when stationarity is reasonable
        double stat_gate = 100.0 * mu + 2.0;
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

        // ── Choose sigma exponent based on difficulty ─────────────────
        // The Mehrotra-corrector health signal refines the classification:
        //   - rejected cross-term -> subproblem is hard -> sigma^0.5
        //     (conservative mu reduction; the second-order correction
        //      couldn't help, so reduce mu slowly)
        //   - healthy corrector + solved + fast -> sigma^1.5
        //     (aggressive mu reduction; the corrector confirms the step
        //      is good, so reduce mu faster)
        // The solved/force gate is NOT changed -- only the exponent.
        double sigma_exp;
        const char* difficulty;

        if (!mehrotra_healthy) {
            sigma_exp = p_.sigma_exp_hard;
            difficulty = "hard(mehrotra)";
        } else if (solved && iters_at_mu_ <= p_.fast_threshold) {
            sigma_exp = p_.sigma_exp_easy;
            difficulty = "easy";
        } else if (!solved || iters_at_mu_ >= p_.slow_threshold) {
            sigma_exp = p_.sigma_exp_hard;
            difficulty = "hard";
        } else {
            sigma_exp = p_.sigma_exp_normal;
            difficulty = "norm";
        }

        last_sigma_exp_ = sigma_exp;

        // ── σ-modulated μ reduction ───────────────────────────────────
        // σ_eff = σ^exponent, bounded below to prevent extreme reduction
        double sigma_clamped = std::max(sigma, 1e-4);
        double sigma_eff = std::pow(sigma_clamped, sigma_exp);
        double mu_new = std::max(sigma_eff * mu, p_.mu_min);

        // ── Logging ───────────────────────────────────────────────────
        if (p_.verbosity >= 2) {
            printf("  [barrier: %s E_mu=%.2e k*mu=%.2e stat=%.2e "
                   "iters=%d %s σ=%.3f→σ_eff=%.3f -> mu=%.3e]\n",
                   solved ? "SOLVED" : "FORCE", E_mu, p_.kappa_eps * mu,
                   stat_inf, iters_at_mu_, difficulty,
                   sigma, sigma_eff, mu_new);
        }

        mu = mu_new;
        iters_at_mu_ = 0;
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
    int    iters_at_mu_ = 0;
    double last_sigma_exp_ = 1.0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Legacy: BarrierManager (kept for backward compatibility)
// ─────────────────────────────────────────────────────────────────────────────

struct BarrierParams {
    double mu_init       = 1.0;
    double mu_min        = 1e-5;
    double kappa_eps     = 10.0;
    double sigma_exp_easy   = 1.5;
    double sigma_exp_normal = 1.0;
    double sigma_exp_hard   = 0.5;
    int    fast_threshold   = 2;
    int    slow_threshold   = 4;
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
            double sigma_exp;
            if (barrier_solved && iters_at_mu_ <= params_.fast_threshold)
                sigma_exp = params_.sigma_exp_easy;
            else if (!barrier_solved || iters_at_mu_ >= params_.slow_threshold)
                sigma_exp = params_.sigma_exp_hard;
            else
                sigma_exp = params_.sigma_exp_normal;

            double sigma_clamped = std::max(sigma, 1e-4);
            double sigma_eff = std::pow(sigma_clamped, sigma_exp);
            mu_ = std::max(sigma_eff * mu_, params_.mu_min);
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
