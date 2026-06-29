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

    // === Two-phase barrier schedule ===
    double epsilon_g     = 0.01;   // feasibility threshold: max(g+) < eps_g → Phase B
    double phase_a_hold_min_g = 0.1;  // Phase A only holds mu when max(g+) > this
                                       // (prevents hold for small violations where
                                       //  stationarity is the real bottleneck)

    // === Alpha-feedback sensor ===
    double alpha_tiny_thresh  = 0.05;  // alpha < this → "tiny" step
    int    alpha_tiny_limit   = 3;     // consecutive tiny alphas → freeze mu

    // === Graduated FTB reduction ===
    double ftb_slow_thresh    = 0.2;   // alpha > this → normal schedule
    double ftb_hold_thresh   = 0.05;  // alpha < this → hold mu

    // === Feasibility restoration ===
    double restoration_alpha_thresh = 0.02;  // alpha < this counts as restoration trigger
    int    restoration_count_limit  = 5;     // consecutive tiny iters → restoration
    int    restoration_max_iters    = 20;    // max iterations in restoration before forced exit

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
        last_E_mu_ = 1.0;
        stall_count_ = 0;
        tiny_alpha_count_ = 0;
        restoration_alpha_count_ = 0;
        restoration_active_ = false;
        phase_ = Phase::INFEASIBILITY;
        last_max_g_pos_ = 0.0;
        restoration_iters_ = 0;
        restoration_slacks_enlarged_ = false;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Convergence gate: is the overall problem solved?
    // ═══════════════════════════════════════════════════════════════════

    bool at_minimum(double mu) const {
        return mu <= p_.mu_min;
    }

    double last_sigma_exp() const { return last_sigma_exp_; }

    // ═══════════════════════════════════════════════════════════════════
    //  Phase enum: infeasibility reduction vs central-path tracking
    // ═══════════════════════════════════════════════════════════════════

    enum class Phase { INFEASIBILITY, CENTRAL_PATH };

    Phase phase() const { return phase_; }
    bool  restoration_active() const { return restoration_active_; }
    bool  should_enlarge_slacks() const {
        return restoration_active_ && !restoration_slacks_enlarged_;
    }
    void  mark_slacks_enlarged() { restoration_slacks_enlarged_ = true; }
    int   tiny_alpha_count() const { return tiny_alpha_count_; }

    // ═══════════════════════════════════════════════════════════════════
    //  Core update — call once after each accepted Newton step.
    //
    //  Phase A (infeasibility reduction):
    //    max(g+) > epsilon_g → hold μ or reduce very slowly.
    //    Primary objective: reduce constraint violation.
    //
    //  Phase B (central-path tracking):
    //    max(g+) < epsilon_g → resume normal σ-modulation schedule.
    //
    //  Alpha-feedback:
    //    - Graduated μ reduction based on alpha_p
    //    - Freeze μ when alpha_p is persistently tiny
    //    - Restoration trigger: enlarge slacks after prolonged stall
    // ═══════════════════════════════════════════════════════════════════

    bool update(double& mu, double primal_inf, double compl_inf, double sigma,
                double stat_inf = -1.0, bool mehrotra_healthy = true,
                double alpha_p = 1.0, double max_g_pos = 0.0) {
        iters_at_mu_++;

        // ── Phase detection ───────────────────────────────────────
        // Phase A: infeasibility reduction (constraints still violated)
        // Phase B: central-path tracking (constraints approximately feasible)
        if (max_g_pos > p_.epsilon_g)
            phase_ = Phase::INFEASIBILITY;
        else
            phase_ = Phase::CENTRAL_PATH;

        // ── Restoration exit condition ────────────────────────────
        // ── Restoration exit: feasibility restored OR timeout ──
        if (restoration_active_) {
            restoration_iters_++;
            if (max_g_pos < p_.epsilon_g) {
                restoration_active_ = false;
                restoration_iters_ = 0;
                restoration_slacks_enlarged_ = false;
                if (p_.verbosity >= 1)
                    printf("  [barrier: RESTORATION EXIT — feasibility restored"
                           " max_g+=%.3e]\n", max_g_pos);
            } else if (restoration_iters_ >= p_.restoration_max_iters) {
                restoration_active_ = false;
                restoration_iters_ = 0;
                restoration_slacks_enlarged_ = false;
                if (p_.verbosity >= 1)
                    printf("  [barrier: RESTORATION TIMEOUT — %d iters,"
                           " resuming normal schedule max_g+=%.3e]\n",
                           p_.restoration_max_iters, max_g_pos);
            }
        }

        // ── Alpha-feedback tracking ───────────────────────────────
        // Count consecutive tiny alpha steps (globalization sensor).
        if (alpha_p < p_.alpha_tiny_thresh)
            tiny_alpha_count_++;
        else
            tiny_alpha_count_ = 0;

        // Count consecutive very-tiny alpha steps (restoration trigger).
        if (alpha_p < p_.restoration_alpha_thresh)
            restoration_alpha_count_++;
        else
            restoration_alpha_count_ = 0;

        // ── Restoration trigger ───────────────────────────────────
        // Prolonged tiny steps → freeze μ and signal caller to enlarge slacks.
        if (restoration_alpha_count_ >= p_.restoration_count_limit
            && !restoration_active_) {
            restoration_active_ = true;
            restoration_iters_ = 0;
            if (p_.verbosity >= 1)
                printf("  [barrier: RESTORATION TRIGGERED — %d consecutive"
                       " alpha<%.2f, mu frozen]\n",
                       restoration_alpha_count_, p_.restoration_alpha_thresh);
            // Don't reduce μ during restoration.
            return false;
        }

        // During restoration: hold μ fixed.
        if (restoration_active_) {
            if (p_.verbosity >= 2)
                printf("  [barrier: RESTORATION HOLD mu=%.3e max_g+=%.3e"
                       " alpha_p=%.4f]\n", mu, max_g_pos, alpha_p);
            return false;
        }

        // ── Alpha-sensor: freeze μ if persistently tiny ───────────
        bool alpha_sensor_frozen =
            (tiny_alpha_count_ >= p_.alpha_tiny_limit);

        double E_mu = std::max(primal_inf, compl_inf);
        bool primal_compl_ok = (E_mu <= p_.kappa_eps * mu);

        // Stationarity gate: only reduce mu when stationarity is reasonable.
        double stat_gate = 100.0 * mu + 2.0;
        bool stat_ok = (stat_inf < 0.0) || (stat_inf <= stat_gate);

        bool solved = (primal_compl_ok && stat_ok);

        // ── Force-reduction via stagnation detection ─────────────
        // Phase A: monitor max(g+) improvement (constraint reduction).
        // Phase B: monitor E_μ improvement (barrier subproblem convergence).

        double improv = 0.0;
        if (last_E_mu_ > 1e-14)
            improv = (last_E_mu_ - E_mu) / last_E_mu_;
        last_E_mu_ = E_mu;
        double prev_max_g_pos = last_max_g_pos_;
        last_max_g_pos_ = max_g_pos;

        constexpr double kStallThreshold = 0.02;
        if (improv < kStallThreshold && iters_at_mu_ >= 4)
            stall_count_++;
        else
            stall_count_ = 0;

        constexpr int kStallForceLimit = 5;
        // In Phase A, allow more stall iterations before forcing reduction
        int stall_limit = (phase_ == Phase::INFEASIBILITY) ? 15 : kStallForceLimit;
        bool force_by_stall = (stall_count_ >= stall_limit);

        // Phase-A stall guard: when max_g+ is growing (infeasibility worsening),
        // suppress stall-based force reduction.  Reducing μ when constraints are
        // getting worse weakens the barrier and makes recovery harder.
        bool g_growing = (phase_ == Phase::INFEASIBILITY
                          && max_g_pos >= prev_max_g_pos * 0.99
                          && prev_max_g_pos > 0.01);
        if (g_growing) {
            force_by_stall = false;
        }

        int effective_max = (phase_ == Phase::INFEASIBILITY)
                          ? std::max(p_.max_same_mu, 30)
                          : p_.max_same_mu;
        // In Phase A with growing max_g+: don't force by iteration count either
        if (g_growing && iters_at_mu_ < effective_max * 3) {
            // Allow up to 3x the normal max_same_mu before forcing
        }
        bool force = ((iters_at_mu_ >= effective_max) || force_by_stall);
        if (g_growing) {
            // Override: only force after much longer patience in Phase A.
            // But allow normal reduction once stationarity has improved
            // (meaning the constraint force has caught up with the costate).
            if (stat_inf >= 0.0 && stat_inf < 0.5) {
                // Stationarity is good enough — allow normal μ schedule
                force = ((iters_at_mu_ >= effective_max) || force_by_stall);
            } else {
                int phase_a_force_limit = effective_max * 3;
                force = (iters_at_mu_ >= phase_a_force_limit);
            }
        }

        // Alpha-sensor overrides force: don't reduce μ if alpha is tiny.
        if (alpha_sensor_frozen && !force_by_stall) {
            force = false;
            if (p_.verbosity >= 2)
                printf("  [barrier: ALPHA-SENSOR freeze — %d consecutive"
                       " alpha<%.2f, holding mu=%.3e]\n",
                       tiny_alpha_count_, p_.alpha_tiny_thresh, mu);
        }

        if (!solved && !force) {
            if (p_.verbosity >= 3)
                printf("  [barrier: HOLD E_mu=%.2e k*mu=%.2e stat=%.2e"
                       " gate=%.2e iters=%d/%d improv=%.1f%% stall=%d"
                       " phase=%s]\n",
                       E_mu, p_.kappa_eps * mu, stat_inf, stat_gate,
                       iters_at_mu_, effective_max, 100.0*improv,
                       stall_count_,
                       (phase_ == Phase::INFEASIBILITY) ? "A" : "B");
            return false;
        }

        // ── Phase A: infeasibility reduction ──────────────────────
        // Hold μ only when constraint violations are SIGNIFICANT
        // (max_g+ > phase_a_hold_min_g).  For small violations near
        // epsilon_g, let mu reduce normally so stationarity can improve.
        if (phase_ == Phase::INFEASIBILITY && !force_by_stall && !force
            && max_g_pos > p_.phase_a_hold_min_g) {
            // Don't reduce μ during infeasibility reduction unless forced.
            if (p_.verbosity >= 2)
                printf("  [barrier: PHASE-A HOLD max_g+=%.3e>%.3e"
                       " mu=%.3e alpha_p=%.4f]\n",
                       max_g_pos, p_.epsilon_g, mu, alpha_p);
            return false;
        }

        // ── Choose sigma exponent with graduated FTB feedback ──────
        double sigma_exp;
        const char* difficulty;

        // Graduated FTB reduction based on alpha_p:
        //   alpha_p > ftb_slow_thresh:  normal schedule
        //   ftb_hold < alpha < ftb_slow: sigma^0.8 (slow down)
        //   alpha_p < ftb_hold_thresh:  hold mu (no reduction)
        if (alpha_p < p_.ftb_hold_thresh && !force_by_stall) {
            // Alpha too small — hold μ to prevent worsening
            if (p_.verbosity >= 2)
                printf("  [barrier: FTB-HOLD alpha_p=%.4f<%.2f"
                       " mu=%.3e]\n",
                       alpha_p, p_.ftb_hold_thresh, mu);
            return false;
        }

        if (alpha_p < p_.ftb_slow_thresh) {
            sigma_exp = 0.8;
            difficulty = "ftb-slow";
        } else if (!mehrotra_healthy) {
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

        // ── σ-modulated μ reduction ───────────────────────────────
        double sigma_clamped = std::max(sigma, 1e-4);
        double sigma_eff = std::pow(sigma_clamped, sigma_exp);
        double mu_new = std::max(sigma_eff * mu, p_.mu_min);

        // Phase-A conservatism: when max_g+ is still significant and the
        // solver just exited a long hold, limit reduction to 10% per step
        // to avoid destabilizing the step direction.
        if (g_growing && mu_new < mu * 0.90) {
            mu_new = mu * 0.90;
        }

        // ── Logging ───────────────────────────────────────────────
        if (p_.verbosity >= 2) {
            printf("  [barrier: %s E_mu=%.2e k*mu=%.2e stat=%.2e "
                   "iters=%d %s σ=%.3f→σ_eff=%.3f -> mu=%.3e"
                   " phase=%s max_g+=%.3e]\n",
                   solved ? "SOLVED" : "FORCE", E_mu, p_.kappa_eps * mu,
                   stat_inf, iters_at_mu_, difficulty,
                   sigma, sigma_eff, mu_new,
                   (phase_ == Phase::INFEASIBILITY) ? "A" : "B",
                   max_g_pos);
        }

        mu = mu_new;
        iters_at_mu_ = 0;
        last_max_g_pos_ = 0.0;
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
    double last_E_mu_ = 1.0;     // E_μ from previous iteration (for stall detection)
    int    stall_count_ = 0;     // consecutive stalled iterations at same μ
    int    tiny_alpha_count_ = 0;            // consecutive alpha_p < alpha_tiny_thresh
    int    restoration_alpha_count_ = 0;     // consecutive alpha_p < restoration_alpha_thresh
    bool   restoration_active_ = false;      // restoration mode active
    Phase  phase_ = Phase::INFEASIBILITY;   // current globalization phase
    double last_max_g_pos_ = 0.0;            // previous max(g+) for Phase A stall detection
    int    restoration_iters_ = 0;           // iterations spent in current restoration
    bool   restoration_slacks_enlarged_ = false;  // slacks enlarged (once per restoration)
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
