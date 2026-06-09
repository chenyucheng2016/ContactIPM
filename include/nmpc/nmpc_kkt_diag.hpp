#pragma once
/**
 * @file    nmpc_kkt_diag.hpp
 * @brief   KKT system diagnostics: condition number, inertia, rank deficiency.
 *
 * Lightweight diagnostics for monitoring KKT solve health during IPM iterations.
 * All functions use only the LDL^T factorization stored in SymMat<N> — no
 * additional factorization is needed.
 *
 * Diagnostics:
 *   1. CondEstimate  — max/min diagonal ratio (crude but zero-cost)
 *   2. InertiaCount  — count pos/neg pivots from D diagonal (after LDL^T)
 *   3. RankEstimate  — count near-zero pivots
 *
 * Dependencies: nmpc_core.hpp only.
 */

#include "nmpc_core.hpp"
#include <cmath>
#include <algorithm>

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  KKT diagnostic results
// ─────────────────────────────────────────────────────────────────────────────

struct KKTDiag {
    double cond_est;       // max/min diagonal ratio
    int    pos_pivots;     // number of positive D entries
    int    neg_pivots;     // number of negative D entries
    int    zero_pivots;    // number of near-zero D entries (< 1e-12)
    double max_diag;       // largest |D| entry
    double min_diag;       // smallest non-zero |D| entry
    double reg_applied;    // regularization actually used

    bool is_psd()       const { return neg_pivots == 0; }
    bool is_full_rank() const { return zero_pivots == 0; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Diagonal ratio condition estimate (cost: O(N), no extra factorization)
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
double cond_estimate_diag(const SymMat<N>& H) {
    double dmax = 0.0, dmin = 1e100;
    for (int i = 0; i < N; ++i) {
        double d = std::fabs(H(i, i));
        if (d > dmax) dmax = d;
        if (d > 1e-14 && d < dmin) dmin = d;
    }
    if (dmin > 1e99) dmin = 1e-14;
    return dmax / dmin;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Inertia from LDL^T diagonal D (cost: O(N))
//
//  After ldlt_factorize(), the diagonal entries of SymMat<N> are the D matrix.
//  Count positive / negative / near-zero pivots.
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
KKTDiag compute_inertia(const SymMat<N>& L_D, double reg = 1e-12) {
    KKTDiag d;
    d.pos_pivots = 0;
    d.neg_pivots = 0;
    d.zero_pivots = 0;
    d.max_diag = 0.0;
    d.min_diag = 1e100;

    for (int i = 0; i < N; ++i) {
        double piv = L_D(i, i);
        double apiv = std::fabs(piv);
        if (apiv > d.max_diag) d.max_diag = apiv;
        if (apiv > 1e-14 && apiv < d.min_diag) d.min_diag = apiv;

        if (piv > reg)
            d.pos_pivots++;
        else if (piv < -reg)
            d.neg_pivots++;
        else
            d.zero_pivots++;
    }

    if (d.min_diag > 1e99) d.min_diag = 1e-14;
    d.cond_est = d.max_diag / d.min_diag;
    d.reg_applied = 0.0;

    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Combined: condition + inertia from a factorized SymMat
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
KKTDiag diagnose_kkt(const SymMat<N>& L_D_factorized, double reg_used = 0.0) {
    KKTDiag d = compute_inertia<N>(L_D_factorized);
    d.reg_applied = reg_used;
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rank estimate for a Jacobian matrix (uses diagonal dominance proxy)
// ─────────────────────────────────────────────────────────────────────────────

template <int Rows, int Cols>
int estimate_rank(const Mat<Rows, Cols>& J, double tol = 1e-8) {
    // Column norm proxy: count columns with significant norm
    int rank = 0;
    for (int c = 0; c < Cols; ++c) {
        double nrm = 0.0;
        for (int r = 0; r < Rows; ++r)
            nrm += J(r, c) * J(r, c);
        if (std::sqrt(nrm) > tol) rank++;
    }
    return rank;
}

} // namespace nmpc
