#pragma once
/**
 * @file    nmpc_filter_ls.hpp
 * @brief   Standalone backtracking filter line-search module.
 *
 * Implements Algorithm A (Step A-5) from:
 *   Wächter & Biegler, "On the Implementation of an Interior-Point
 *   Filter Line-Search Algorithm for Large-Scale Nonlinear Programming"
 *
 * This is a self-contained module: it takes a trial-point evaluator,
 * a filter, and parameters, and runs the backtracking + SOC loop.
 *
 * Usage:
 *   1. Instantiate FilterLineSearch<N>.
 *   2. Call filter_ls.init() to set up the filter.
 *   3. At each SQP/IPM iteration, call filter_ls.search().
 */

#include "nmpc_core.hpp"
#include <cmath>
#include <cstdio>

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Filter parameters
// ─────────────────────────────────────────────────────────────────────────────

struct FilterLSParams {
    // ── Tolerances ────────────────────────────────────────────────
    double theta_min     = 1e-4;    // "sufficiently feasible" threshold
    double gamma_theta   = 1e-5;    // constraint violation margin
    double gamma_phi     = 1e-5;    // objective margin
    double kappa_theta   = 1e-1;    // SOC constraint improvement factor

    // ── Line-search ───────────────────────────────────────────────
    double alpha_min     = 1e-14;   // minimum step before declaring failure
    double eta_phi       = 1e-4;    // Armijo constant for objective
    double s_theta       = 1.1;     // switching condition power
    double s_phi         = 2.3;     // switching condition power

    // ── Second-order correction ───────────────────────────────────
    int    soc_max       = 4;       // max SOC attempts per iteration
    double kappa_soc     = 0.99;    // SOC constraint improvement required

    // ── Filter size ───────────────────────────────────────────────
    int    filter_max_size = 100;   // maximum entries in filter

    // ── Logging ───────────────────────────────────────────────────
    int    verbosity     = 0;       // 0=silent, 1=per-iter LS, 2=+SOC detail
};

// ─────────────────────────────────────────────────────────────────────────────
//  Filter entry: (θ, φ) pair for constraint violation & objective
// ─────────────────────────────────────────────────────────────────────────────

struct FilterEntry {
    double theta;  // constraint violation  ‖c(x)‖₁  (or ‖c(x)‖∞)
    double phi;    // objective value  f(x)
};

// ─────────────────────────────────────────────────────────────────────────────
//  Filter: maintains a set of (θ, φ) pairs that dominate rejected points
// ─────────────────────────────────────────────────────────────────────────────

template <int MAX_SIZE = 100>
struct Filter {
    FilterEntry entries[MAX_SIZE];
    int         count = 0;

    void reset() { count = 0; }

    // Check if (theta, phi) is dominated by any filter entry.
    // Returns true if the point IS inside the filter (REJECT).
    bool contains(double theta, double phi, double gamma_theta, double gamma_phi) const {
        for (int i = 0; i < count; ++i) {
            if (entries[i].theta <= theta + gamma_theta &&
                entries[i].phi   <= phi   + gamma_phi) {
                return true;  // dominated → inside filter → reject
            }
        }
        return false;  // not dominated → acceptable to filter
    }

    // Add a new entry and remove dominated ones.
    // The new entry dominates any existing entry with BOTH larger theta AND larger phi.
    void add(double theta, double phi, double gamma_theta, double gamma_phi) {
        // Remove entries that are dominated by the new one
        int j = 0;
        for (int i = 0; i < count; ++i) {
            if (!(theta <= entries[i].theta + gamma_theta &&
                  phi   <= entries[i].phi   + gamma_phi)) {
                entries[j++] = entries[i];
            }
        }
        count = j;

        // Add new entry (if room)
        if (count < MAX_SIZE) {
            entries[count].theta = theta;
            entries[count].phi   = phi;
            count++;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Line-search status
// ─────────────────────────────────────────────────────────────────────────────

enum class LSStatus {
    ACCEPTED,          // step accepted
    FAILED,            // line-search failed (α < α_min)
    CONTINUE           // should not happen in normal flow
};

// ─────────────────────────────────────────────────────────────────────────────
//  Line-search result
// ─────────────────────────────────────────────────────────────────────────────

struct LSResult {
    LSStatus status;
    double   alpha;      // accepted step size
    int      ls_iters;   // number of trial steps
    bool     soc_used;   // was SOC attempted?
    int      soc_iters;  // number of SOC steps used
};

// ─────────────────────────────────────────────────────────────────────────────
//  Trial-point evaluator interface (user provides via lambda or callback)
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU, int HORIZON>
class TrialPointEvaluator {
public:
    virtual ~TrialPointEvaluator() = default;

    // Evaluate the trial point at step size alpha (READ-ONLY).
    // Computes trial values WITHOUT modifying solver state.
    // Sets:
    //   out_theta = constraint violation (e.g., ‖c(z)‖₁)
    //   out_phi   = objective value f(z)
    // Returns whether evaluation succeeded (no NaN, etc.)
    virtual bool evaluate(double alpha, double& out_theta, double& out_phi) = 0;

    // Current iterate data (for SOC computation)
    virtual double current_theta() const = 0;
    virtual double current_phi()   const = 0;

    // Compute the directional derivative Dφ for the barrier subproblem.
    // Used by the switching condition and Armijo test.
    virtual double compute_Dphi() const = 0;

    // Compute a second-order correction step and evaluate it.
    // Returns true if SOC improved enough to try again.
    virtual bool compute_soc(double alpha, double& out_theta, double& out_phi) { return false; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Switching condition: the gradient-based test for fast local convergence
// ─────────────────────────────────────────────────────────────────────────────

inline bool switching_condition(double Dphi, double alpha,
                                 double theta, double s_theta, double s_phi) {
    // Condition: Dphi * alpha < 0  AND  alpha * (-Dphi)^{s_phi} > delta * theta^{s_theta}
    // Dphi is the directional derivative (should be < 0 for descent)
    if (Dphi >= 0.0) return false;
    double lhs = alpha * std::pow(-Dphi, s_phi);
    double rhs = 1e-8 * std::pow(theta + 1e-14, s_theta);  // δ = 1e-8
    return lhs > rhs;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Armijo condition: sufficient decrease in the objective
// ─────────────────────────────────────────────────────────────────────────────

inline bool armijo_condition(double phi_trial, double phi0,
                              double Dphi, double alpha,
                              double eta_phi) {
    if (Dphi >= 0.0) return false;  // not a descent direction
    return phi_trial <= phi0 + eta_phi * alpha * Dphi;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bi-objective improvement: either θ or φ improves sufficiently
// ─────────────────────────────────────────────────────────────────────────────

inline bool bi_objective_improvement(double theta_trial, double theta0,
                                      double phi_trial,   double phi0,
                                      double gamma_theta, double gamma_phi,
                                      double theta_min) {
    // Case 1: constraint violation improves enough
    if (theta_trial <= (1.0 - gamma_theta) * theta0) return true;
    // Case 2: constraint violation is small AND objective improves
    if (theta0 <= theta_min && phi_trial <= phi0 + gamma_phi) return true;
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Filter line-search: implements Algorithm A, Step A-5
// ─────────────────════════════════════════════════════════════════════════════

template <int NX, int NU, int HORIZON>
class FilterLineSearch {
public:
    using Eval = TrialPointEvaluator<NX, NU, HORIZON>;

    FilterLineSearch() = default;

    void init(const FilterLSParams& params = FilterLSParams{}) {
        params_ = params;
        filter_.reset();
        // Initialize filter with a safe upper bound
        filter_.add(1e10, 1e10, params_.gamma_theta, params_.gamma_phi);
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Main search routine
    // ═══════════════════════════════════════════════════════════════════

    LSResult search(Eval& eval, double alpha_max) {
        LSResult result;
        result.alpha     = alpha_max;
        result.ls_iters  = 1;
        result.soc_used  = false;
        result.soc_iters = 0;
        result.status    = LSStatus::FAILED;

        const int v = params_.verbosity;

        double theta0 = eval.current_theta();
        double phi0   = eval.current_phi();
        double Dphi   = eval.compute_Dphi();

        if (v >= 1) {
            printf("    [ls:0] theta0=%.4e phi0=%.4e Dphi=%.3e\n",
                   theta0, phi0, Dphi);
        }

        // ── Phase 1 + 2: backtracking loop ─────────────────────────
        for (int l = 0; l < 50; ++l) {  // safety cap on backtrack count
            double alpha = result.alpha;

            // Evaluate trial point
            double theta_trial, phi_trial;
            bool ok = eval.evaluate(alpha, theta_trial, phi_trial);
            if (!ok) {
                if (v >= 1)
                    printf("    [ls:%d] a=%.3e NaN/inf -> shrink\n", l, alpha);
                result.alpha *= 0.5;
                result.ls_iters++;
                if (result.alpha < params_.alpha_min) break;
                continue;
            }

            // ── Compute acceptance tests ──────────────────────────
            bool in_filter = filter_.contains(theta_trial, phi_trial,
                                               params_.gamma_theta, params_.gamma_phi);
            bool sw_cond   = switching_condition(Dphi, alpha, theta0,
                                                 params_.s_theta, params_.s_phi);
            bool armijo    = armijo_condition(phi_trial, phi0, Dphi, alpha,
                                               params_.eta_phi);
            bool bi_obj    = bi_objective_improvement(theta_trial, theta0,
                                                       phi_trial,   phi0,
                                                       params_.gamma_theta,
                                                       params_.gamma_phi,
                                                       params_.theta_min);
            bool feas_case = (theta0 <= params_.theta_min && sw_cond);

            // DIAGNOSTIC: show why bi_obj passes/fails
            if (v >= 1) {
                double theta_req = (1.0 - params_.gamma_theta) * theta0;
                printf("    [filter-diag] theta_trial=%.4e theta0=%.4e req=%.4e bi_obj=%d feas=%d armijo=%d\n",
                       theta_trial, theta0, theta_req, bi_obj, feas_case, armijo);
            }

            if (v >= 1) {
                printf("    [ls:%d] a=%.3e theta=%.4e phi=%.4e"
                       " | filt=%d sw=%d arm=%d bi=%d case=%s\n",
                       l, alpha, theta_trial, phi_trial,
                       in_filter, sw_cond, armijo, bi_obj,
                       feas_case ? "OPT" : "FEAS");
            }

            bool trigger_soc = false;
            bool accepted    = false;

            if (!in_filter) {
                if (feas_case) {
                    // Case I: prioritizing optimality
                    if (armijo) {
                        if (theta0 > params_.theta_min || theta_trial > 1e-8)
                            filter_.add(theta_trial, phi_trial,
                                        params_.gamma_theta, params_.gamma_phi);
                        accepted = true;
                    } else {
                        trigger_soc = true;
                    }
                } else {
                    // Case II: prioritizing feasibility
                    if (bi_obj) {
                        filter_.add(theta_trial, phi_trial,
                                    params_.gamma_theta, params_.gamma_phi);
                        accepted = true;
                    } else {
                        trigger_soc = true;
                    }
                }
            } else {
                trigger_soc = true;
            }

            if (accepted) {
                result.status   = LSStatus::ACCEPTED;
                result.alpha    = alpha;
                result.ls_iters = l + 1;
                if (v >= 1)
                    printf("    [ls:%d] ACCEPT a=%.4f case=%s\n",
                           l, alpha, feas_case ? "OPT(armijo)" : "FEAS(bi-obj)");
                return result;
            }

            // ── SOC decision logging ─────────────────────────────
            if (v >= 2) {
                // Explain WHY SOC was or was not triggered
                if (trigger_soc) {
                    if (l != 0) {
                        printf("    [soc] SKIP: l=%d != 0 (SOC only at l=0)\n", l);
                    } else if (theta_trial < theta0) {
                        printf("    [soc] SKIP: theta_trial=%.3e < theta0=%.3e"
                               " (constraint improved)\n", theta_trial, theta0);
                    } else {
                        const char* reason = "unknown";
                        if (in_filter) {
                            reason = "in_filter";
                        } else if (feas_case && !armijo) {
                            reason = "sw_cond met, armijo failed";
                        } else if (!feas_case && !bi_obj) {
                            reason = "bi_obj failed";
                        }
                        printf("    [soc] TRIGGER: %s"
                               " (theta_trial=%.3e >= theta0=%.3e)\n",
                               reason, theta_trial, theta0);
                    }
                }
            }

            // ── Phase 3: Second-Order Correction ─────────────────
            if (trigger_soc && l == 0 && theta_trial >= theta0) {
                result.soc_used = true;
                double theta_soc_prev = theta_trial;
                for (int p = 0; p < params_.soc_max; ++p) {
                    double theta_soc, phi_soc;
                    bool soc_ok = eval.compute_soc(alpha, theta_soc, phi_soc);
                    if (!soc_ok) {
                        if (v >= 2)
                            printf("    [soc:%d] compute_soc failed\n", p);
                        break;
                    }
                    result.soc_iters++;

                    bool soc_in_filter = filter_.contains(theta_soc, phi_soc,
                                                          params_.gamma_theta, params_.gamma_phi);
                    bool soc_armijo    = armijo_condition(phi_soc, phi0, Dphi, alpha,
                                                          params_.eta_phi);
                    bool soc_bi_obj    = bi_objective_improvement(theta_soc, theta0,
                                                                   phi_soc,   phi0,
                                                                   params_.gamma_theta,
                                                                   params_.gamma_phi,
                                                                   params_.theta_min);

                    if (v >= 2) {
                        printf("    [soc:%d] theta=%.4e phi=%.4e"
                               " filt=%d arm=%d bi=%d\n",
                               p, theta_soc, phi_soc,
                               soc_in_filter, soc_armijo, soc_bi_obj);
                    }

                    if (!soc_in_filter) {
                        bool soc_accepted = false;
                        if (feas_case) {
                            if (soc_armijo) {
                                if (theta0 > params_.theta_min || theta_soc > 1e-8)
                                    filter_.add(theta_soc, phi_soc,
                                                params_.gamma_theta, params_.gamma_phi);
                                soc_accepted = true;
                            }
                        } else {
                            if (soc_bi_obj) {
                                filter_.add(theta_soc, phi_soc,
                                            params_.gamma_theta, params_.gamma_phi);
                                soc_accepted = true;
                            }
                        }
                        if (soc_accepted) {
                            if (v >= 2)
                                printf("    [soc:%d] ACCEPT\n", p);
                            result.status = LSStatus::ACCEPTED;
                            result.alpha  = alpha;
                            result.ls_iters = l + 1;
                            return result;
                        }
                    }

                    // Check if SOC is still improving constraints
                    if (theta_soc > params_.kappa_soc * theta_soc_prev) {
                        if (v >= 2)
                            printf("    [soc:%d] stall: theta_soc=%.3e > k*prev=%.3e\n",
                                   p, theta_soc, params_.kappa_soc * theta_soc_prev);
                        break;
                    }
                    theta_soc_prev = theta_soc;
                }
            }

            // ── Phase 4: Backtrack ────────────────────────────────
            result.alpha *= 0.5;
            result.ls_iters++;

            if (result.alpha < params_.alpha_min) {
                if (v >= 1)
                    printf("    [ls:%d] a=%.3e < alpha_min -> FAILED\n",
                           l + 1, result.alpha);
                break;
            }
        }

        // Line-search failed (state was never modified — evaluate is read-only)
        result.status = LSStatus::FAILED;
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Reset filter (call when μ is reduced)
    // ═══════════════════════════════════════════════════════════════════

    void reset_filter() {
        filter_.reset();
        filter_.add(1e10, 1e10, params_.gamma_theta, params_.gamma_phi);
    }

    // Expose filter state for debugging
    int filter_size() const { return filter_.count; }

private:
    FilterLSParams     params_;
    Filter<100>        filter_;
};

} // namespace nmpc
