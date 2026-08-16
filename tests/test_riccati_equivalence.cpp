/**
 * @file    test_riccati_equivalence.cpp
 * @brief   Differential test for the optimized Riccati recursion.
 *
 * The reference below retains the pre-optimization nested-loop evaluation of
 * A^T P A.  Randomized problems exercise the complete backward and forward
 * passes and verify that optional diagnostics do not change the Newton step.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "nmpc/nmpc_riccati.hpp"

namespace {

using namespace nmpc;

constexpr int kNx = 7;
constexpr int kNu = 4;
constexpr int kNc = 1;
constexpr int kHorizon = 12;
constexpr int kCases = 128;
constexpr double kTolerance = 5e-12;

using Stage = StageData<kNx, kNu, kNc>;
using Workspace = RiccatiWorkspace<kNx, kNu, kHorizon>;
using Solver = RiccatiSolver<kNx, kNu, kNc, kHorizon>;

bool close(double actual, double expected) {
    const double scale = 1.0 + std::max(std::fabs(actual), std::fabs(expected));
    return std::fabs(actual - expected) <= kTolerance * scale;
}

bool report_mismatch(const char* quantity, int case_index, int stage,
                     int row, int column, double actual, double expected) {
    if (close(actual, expected)) return false;
    std::printf(
        "FAIL: case=%d quantity=%s stage=%d row=%d column=%d "
        "actual=%.17g expected=%.17g error=%.3e\n",
        case_index, quantity, stage, row, column, actual, expected,
        std::fabs(actual - expected));
    return true;
}

void reference_compute_pk(Workspace& workspace, const Stage& stage,
                          const SymMat<kNx>& p_next, int stage_index,
                          double regularization) {
    SymMat<kNx>& pk = workspace.P[stage_index];
    pk.copy_lower_from(stage.Qxx);

    // Deliberately retain the pre-optimization algebra: P*A is recomputed for
    // every output row instead of being cached once for the stage.
    for (int row = 0; row < kNx; ++row) {
        for (int column = 0; column <= row; ++column) {
            double value = 0.0;
            for (int inner = 0; inner < kNx; ++inner) {
                double p_row = 0.0;
                for (int j = 0; j < kNx; ++j)
                    p_row += p_next(inner, j) * stage.A(j, column);
                value += stage.A(inner, row) * p_row;
            }
            pk(row, column) += value;
        }
    }

    for (int row = 0; row < kNx; ++row) {
        for (int column = 0; column <= row; ++column) {
            double value = 0.0;
            for (int control = 0; control < kNu; ++control) {
                value += workspace.Qux_plus_BtPA(control, row)
                       * workspace.K[stage_index](control, column);
            }
            pk(row, column) -= value;
        }
    }

    for (int i = 0; i < kNx; ++i) {
        pk(i, i) += regularization
                  * std::max(std::fabs(pk(i, i)), 1e-14);
    }
}

Status reference_backward_lhs(Stage stages[], Workspace& workspace,
                              double regularization_base,
                              double& regularization_used) {
    workspace.P[kHorizon].copy_lower_from(stages[kHorizon].Qxx);
    for (int i = 0; i < kNx; ++i) {
        workspace.P[kHorizon](i, i) +=
            regularization_base
            * std::max(std::fabs(workspace.P[kHorizon](i, i)), 1e-14);
    }
    regularization_used = regularization_base;

    for (int stage_index = kHorizon - 1; stage_index >= 0; --stage_index) {
        Stage& stage = stages[stage_index];
        const SymMat<kNx>& p_next = workspace.P[stage_index + 1];

        workspace.S.zero();
        for (int row = 0; row < kNu; ++row)
            for (int column = 0; column <= row; ++column)
                workspace.S(row, column) = stage.Quu(row, column);

        for (int row = 0; row < kNu; ++row) {
            for (int column = 0; column < kNx; ++column) {
                double value = 0.0;
                for (int inner = 0; inner < kNx; ++inner)
                    value += stage.B(inner, row) * p_next(inner, column);
                workspace.BtP(row, column) = value;
            }
        }
        for (int row = 0; row < kNu; ++row) {
            for (int column = 0; column <= row; ++column) {
                double value = 0.0;
                for (int inner = 0; inner < kNx; ++inner)
                    value += workspace.BtP(row, inner)
                           * stage.B(inner, column);
                workspace.S(row, column) += value;
            }
        }

        for (int row = 0; row < kNu; ++row) {
            for (int column = 0; column < kNx; ++column) {
                double value = 0.0;
                for (int inner = 0; inner < kNx; ++inner)
                    value += workspace.BtP(row, inner)
                           * stage.A(inner, column);
                workspace.Qux_plus_BtPA(row, column) =
                    stage.Qux(row, column) + value;
            }
        }
        workspace.BtP_stages[stage_index] = workspace.BtP;
        workspace.Qux_plus_BtPA_stages[stage_index] =
            workspace.Qux_plus_BtPA;

        double minimum_diagonal = 1e100;
        for (int i = 0; i < kNu; ++i)
            minimum_diagonal =
                std::min(minimum_diagonal, workspace.S(i, i));
        double regularization = regularization_base;
        if (minimum_diagonal < 0.0) {
            regularization =
                std::max(regularization, -minimum_diagonal * 1.1);
        }

        const SymMat<kNu> saved_s = workspace.S;
        bool factored = false;
        for (int attempt = 0; attempt < 6; ++attempt) {
            if (regularization > 1e12) break;
            workspace.S = saved_s;
            for (int i = 0; i < kNu; ++i) {
                workspace.S(i, i) +=
                    regularization
                    * std::max(std::fabs(saved_s(i, i)), 1e-14);
            }
            double minimum_pivot = 0.0;
            if (workspace.S.ldlt_factorize(1e-14, &minimum_pivot)) {
                factored = true;
                break;
            }
            regularization =
                std::max(regularization * 10.0,
                         2.0 * std::fabs(minimum_pivot) + 1e-12);
        }
        if (!factored) return Status::KKT_SINGULAR;
        regularization_used =
            std::max(regularization_used, regularization);
        workspace.S_fact[stage_index] = workspace.S;

        for (int column = 0; column < kNx; ++column) {
            Vec<kNu> rhs;
            for (int row = 0; row < kNu; ++row)
                rhs[row] = workspace.Qux_plus_BtPA(row, column);
            workspace.S_fact[stage_index].ldlt_solve(rhs);
            for (int row = 0; row < kNu; ++row)
                workspace.K[stage_index](row, column) = rhs[row];
        }

        reference_compute_pk(workspace, stage, p_next, stage_index,
                             regularization_base);
    }
    return Status::SUCCESS;
}

Status reference_backward_rhs(Stage stages[], Workspace& workspace) {
    workspace.p[kHorizon] = stages[kHorizon].qx;
    for (int stage_index = kHorizon - 1; stage_index >= 0; --stage_index) {
        Stage& stage = stages[stage_index];
        const Vec<kNx>& p_next = workspace.p[stage_index + 1];
        workspace.BtP = workspace.BtP_stages[stage_index];

        Vec<kNu> rhs;
        for (int row = 0; row < kNu; ++row) {
            double btp = 0.0;
            for (int inner = 0; inner < kNx; ++inner)
                btp += stage.B(inner, row) * p_next[inner];
            double btpc = 0.0;
            for (int inner = 0; inner < kNx; ++inner)
                btpc += workspace.BtP(row, inner) * stage.c[inner];
            rhs[row] = stage.qu[row] + btp + btpc;
        }
        workspace.S_fact[stage_index].ldlt_solve(rhs);
        workspace.d[stage_index] = rhs;

        Vec<kNx>& p = workspace.p[stage_index];
        p = stage.qx;
        Vec<kNx> affine_next;
        for (int row = 0; row < kNx; ++row) {
            double pc = 0.0;
            for (int column = 0; column < kNx; ++column)
                pc += workspace.P[stage_index + 1](row, column)
                    * stage.c[column];
            affine_next[row] = p_next[row] + pc;
        }
        for (int column = 0; column < kNx; ++column) {
            double value = 0.0;
            for (int row = 0; row < kNx; ++row)
                value += stage.A(row, column) * affine_next[row];
            p[column] += value;
        }
        const Mat<kNu, kNx>& qup =
            workspace.Qux_plus_BtPA_stages[stage_index];
        for (int column = 0; column < kNx; ++column) {
            double value = 0.0;
            for (int row = 0; row < kNu; ++row)
                value += qup(row, column) * workspace.d[stage_index][row];
            p[column] -= value;
        }
    }
    return Status::SUCCESS;
}

Status reference_forward(Stage stages[], Workspace& workspace,
                         const Vec<kNx>& initial_step) {
    workspace.dx[0] = initial_step;
    for (int stage_index = 0; stage_index < kHorizon; ++stage_index) {
        for (int row = 0; row < kNu; ++row) {
            double feedback = 0.0;
            for (int column = 0; column < kNx; ++column) {
                feedback += workspace.K[stage_index](row, column)
                          * workspace.dx[stage_index][column];
            }
            workspace.du[stage_index][row] =
                -feedback - workspace.d[stage_index][row];
        }
        for (int row = 0; row < kNx; ++row) {
            double state_value = 0.0;
            double control_value = 0.0;
            for (int column = 0; column < kNx; ++column) {
                state_value += stages[stage_index].A(row, column)
                             * workspace.dx[stage_index][column];
            }
            for (int column = 0; column < kNu; ++column) {
                control_value += stages[stage_index].B(row, column)
                               * workspace.du[stage_index][column];
            }
            workspace.dx[stage_index + 1][row] =
                state_value + control_value + stages[stage_index].c[row];
        }
    }
    return Status::SUCCESS;
}

double random_value(std::mt19937_64& generator, double scale) {
    std::uniform_real_distribution<double> distribution(-scale, scale);
    return distribution(generator);
}

template <int Dimension>
void random_spd(std::mt19937_64& generator, Mat<Dimension, Dimension>& matrix,
                double diagonal, double scale) {
    Mat<Dimension, Dimension> factor;
    for (int row = 0; row < Dimension; ++row)
        for (int column = 0; column < Dimension; ++column)
            factor(row, column) = random_value(generator, scale);

    for (int row = 0; row < Dimension; ++row) {
        for (int column = 0; column < Dimension; ++column) {
            double value = 0.0;
            for (int inner = 0; inner < Dimension; ++inner)
                value += factor(inner, row) * factor(inner, column);
            matrix(row, column) = value + (row == column ? diagonal : 0.0);
        }
    }
}

void generate_problem(std::mt19937_64& generator, Stage stages[],
                      Vec<kNx>& initial_step) {
    for (int stage_index = 0; stage_index <= kHorizon; ++stage_index) {
        Stage& stage = stages[stage_index];
        random_spd(generator, stage.Qxx, 0.5, 0.18);
        random_spd(generator, stage.Quu, 0.8, 0.12);
        stage.Qxu.zero();
        stage.Cx.zero();
        stage.Cu.zero();

        for (int row = 0; row < kNx; ++row) {
            stage.qx[row] = random_value(generator, 0.3);
            stage.c[row] = random_value(generator, 0.04);
            for (int column = 0; column < kNx; ++column) {
                stage.A(row, column) = random_value(generator, 0.04)
                    + (row == column ? 0.94 : 0.0);
            }
            for (int column = 0; column < kNu; ++column)
                stage.B(row, column) = random_value(generator, 0.12);
        }
        for (int row = 0; row < kNu; ++row) {
            stage.qu[row] = random_value(generator, 0.3);
            for (int column = 0; column < kNx; ++column) {
                stage.Qux(row, column) = random_value(generator, 0.025);
                stage.Qxu(column, row) = stage.Qux(row, column);
            }
        }
    }
    for (int row = 0; row < kNx; ++row)
        initial_step[row] = random_value(generator, 0.25);
}

bool compare_workspaces(const Workspace& actual, const Workspace& expected,
                        int case_index) {
    for (int stage = 0; stage <= kHorizon; ++stage) {
        for (int row = 0; row < kNx; ++row) {
            if (report_mismatch("p", case_index, stage, row, -1,
                                actual.p[stage][row], expected.p[stage][row]))
                return false;
            if (report_mismatch("dx", case_index, stage, row, -1,
                                actual.dx[stage][row], expected.dx[stage][row]))
                return false;
            for (int column = 0; column <= row; ++column) {
                if (report_mismatch("P", case_index, stage, row, column,
                                    actual.P[stage](row, column),
                                    expected.P[stage](row, column)))
                    return false;
            }
        }
        if (stage == kHorizon) continue;
        for (int row = 0; row < kNu; ++row) {
            if (report_mismatch("d", case_index, stage, row, -1,
                                actual.d[stage][row], expected.d[stage][row]))
                return false;
            if (report_mismatch("du", case_index, stage, row, -1,
                                actual.du[stage][row], expected.du[stage][row]))
                return false;
            for (int column = 0; column < kNx; ++column) {
                if (report_mismatch("K", case_index, stage, row, column,
                                    actual.K[stage](row, column),
                                    expected.K[stage](row, column)))
                    return false;
                if (report_mismatch(
                        "BtP", case_index, stage, row, column,
                        actual.BtP_stages[stage](row, column),
                        expected.BtP_stages[stage](row, column)))
                    return false;
                if (report_mismatch(
                        "Qux_plus_BtPA", case_index, stage, row, column,
                        actual.Qux_plus_BtPA_stages[stage](row, column),
                        expected.Qux_plus_BtPA_stages[stage](row, column)))
                    return false;
            }
            for (int column = 0; column <= row; ++column) {
                if (report_mismatch("S_fact", case_index, stage, row, column,
                                    actual.S_fact[stage](row, column),
                                    expected.S_fact[stage](row, column)))
                    return false;
            }
        }
    }
    return true;
}

bool run_randomized_equivalence() {
    std::mt19937_64 generator(0x5249434341545449ULL);
    for (int case_index = 0; case_index < kCases; ++case_index) {
        Stage reference_stages[kHorizon + 1];
        Vec<kNx> initial_step;
        generate_problem(generator, reference_stages, initial_step);

        Stage diagnostics_on_stages[kHorizon + 1];
        Stage diagnostics_off_stages[kHorizon + 1];
        std::copy(reference_stages, reference_stages + kHorizon + 1,
                  diagnostics_on_stages);
        std::copy(reference_stages, reference_stages + kHorizon + 1,
                  diagnostics_off_stages);

        Workspace reference{};
        Workspace diagnostics_on{};
        Workspace diagnostics_off{};
        const double regularization_base =
            1e-10 * static_cast<double>(1 + case_index % 5);
        double reference_regularization = 0.0;
        double diagnostics_on_regularization = 0.0;
        double diagnostics_off_regularization = 0.0;

        if (reference_backward_lhs(reference_stages, reference,
                                   regularization_base,
                                   reference_regularization)
                != Status::SUCCESS
            || reference_backward_rhs(reference_stages, reference)
                != Status::SUCCESS
            || reference_forward(reference_stages, reference, initial_step)
                != Status::SUCCESS) {
            std::printf("FAIL: reference recursion failed in case %d\n",
                        case_index);
            return false;
        }

        if (Solver::backward(diagnostics_on_stages, diagnostics_on,
                             regularization_base,
                             diagnostics_on_regularization, true)
                != Status::SUCCESS
            || Solver::forward(diagnostics_on_stages, diagnostics_on,
                               initial_step, true) != Status::SUCCESS) {
            std::printf("FAIL: diagnostics-on recursion failed in case %d\n",
                        case_index);
            return false;
        }
        const double schur_residual = Solver::schur_residual;
        const double stationarity = Solver::riccati_direct_stationarity;
        const double corrected_stationarity =
            Solver::riccati_direct_stationarity_corr;
        if (!std::isfinite(schur_residual) || schur_residual < 0.0
            || !std::isfinite(stationarity) || stationarity < 0.0
            || !std::isfinite(corrected_stationarity)
            || corrected_stationarity < 0.0) {
            std::printf("FAIL: invalid diagnostics in case %d\n", case_index);
            return false;
        }

        if (Solver::backward(diagnostics_off_stages, diagnostics_off,
                             regularization_base,
                             diagnostics_off_regularization, false)
                != Status::SUCCESS
            || Solver::forward(diagnostics_off_stages, diagnostics_off,
                               initial_step, false) != Status::SUCCESS) {
            std::printf("FAIL: diagnostics-off recursion failed in case %d\n",
                        case_index);
            return false;
        }
        if (Solver::schur_residual != 0.0
            || Solver::riccati_direct_stationarity != 0.0
            || Solver::riccati_direct_stationarity_corr != 0.0) {
            std::printf(
                "FAIL: diagnostics-off path retained optional residuals "
                "in case %d\n",
                case_index);
            return false;
        }

        if (!close(diagnostics_on_regularization, reference_regularization)
            || !close(diagnostics_off_regularization,
                      reference_regularization)) {
            std::printf("FAIL: regularization mismatch in case %d\n",
                        case_index);
            return false;
        }
        if (!compare_workspaces(diagnostics_on, reference, case_index)
            || !compare_workspaces(diagnostics_off, reference, case_index)
            || !compare_workspaces(diagnostics_on, diagnostics_off,
                                   case_index)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    std::printf("  TEST randomized optimized/reference Riccati equivalence ... ");
    if (!run_randomized_equivalence()) return 1;
    std::printf("PASS (%d deterministic cases, diagnostics on/off)\n", kCases);
    return 0;
}
