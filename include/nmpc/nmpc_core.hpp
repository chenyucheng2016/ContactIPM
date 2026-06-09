#pragma once
/**
 * @file    nmpc_core.hpp
 * @brief   Fixed-size linear algebra types and memory pool for embedded NMPC.
 *
 * All hot-path operations use pre-allocated stack/static memory.
 * No heap allocation occurs after the solver is constructed.
 *
 * Design constraints for embedded deployment:
 *   - Templates parametrized on dimensions → compile-time bounds elimination.
 *   - All workspace buffers allocated once at construction.
 *   - No exceptions thrown in hot paths (return codes instead).
 *   - Double precision throughout for numerical robustness.
 */

#include <cstddef>      // size_t
#include <cstdint>      // int32_t
#include <cstring>      // memcpy
#include <cmath>        // fabs, sqrt, fmax, fmin
#include <algorithm>    // max, min, swap
#include <new>          // placement new
#include <type_traits>
#include <cstdlib>      // malloc, free, _aligned_malloc (MSVC)

#ifdef _MSC_VER
#include <malloc.h>     // _aligned_malloc, _aligned_free
#endif

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Status codes – no exceptions
// ─────────────────────────────────────────────────────────────────────────────

enum class Status : int32_t {
    SUCCESS                = 0,
    MAX_ITERATIONS         = 1,   // Reached iteration limit
    LINE_SEARCH_FAILURE    = 2,   // Armijo / filter line-search stalled
    KKT_SINGULAR           = 3,   // KKT matrix near-singular
    NAN_DETECTED           = 4,   // NaN encountered in state/control
    INFEASIBLE             = 5,   // Primal infeasibility detected
    DIVERGING              = 6,   // Cost increasing despite regularization
    BAD_ARGUMENT           = 7,   // Invalid dimensions or nullptr
    NOT_INITIALIZED        = 8,   // Solver not initialized
    INTERNAL_ERROR         = 99
};

inline const char* status_string(Status s) {
    switch (s) {
    case Status::SUCCESS:             return "Success";
    case Status::MAX_ITERATIONS:      return "Max iterations reached";
    case Status::LINE_SEARCH_FAILURE: return "Line search failure";
    case Status::KKT_SINGULAR:        return "KKT matrix singular";
    case Status::NAN_DETECTED:        return "NaN detected";
    case Status::INFEASIBLE:          return "Infeasible";
    case Status::DIVERGING:           return "Diverging";
    case Status::BAD_ARGUMENT:        return "Bad argument";
    case Status::NOT_INITIALIZED:     return "Not initialized";
    default:                          return "Unknown error";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Compile-time dimension constants  (change these for your system)
// ─────────────────────────────────────────────────────────────────────────────

#ifndef NMPC_MAX_NX
#define NMPC_MAX_NX  12     // max state dimension
#endif
#ifndef NMPC_MAX_NU
#define NMPC_MAX_NU   6     // max control dimension
#endif
#ifndef NMPC_MAX_NC
#define NMPC_MAX_NC  20     // max inequality constraints per stage
#endif
#ifndef NMPC_MAX_HORIZON
#define NMPC_MAX_HORIZON 50 // max MPC horizon length
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Fixed-size vector  (stack-allocated, no heap)
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
struct Vec {
    double data[N];

    Vec() { zero(); }

    static constexpr int dim() { return N; }

    double  operator()(int i) const { return data[i]; }
    double& operator()(int i)       { return data[i]; }
    double  operator[](int i) const { return data[i]; }
    double& operator[](int i)       { return data[i]; }

    void zero() { for (int i = 0; i < N; ++i) data[i] = 0.0; }
    void set_constant(double v) { for (int i = 0; i < N; ++i) data[i] = v; }

    double norm2_sq() const {
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += data[i] * data[i];
        return s;
    }
    double norm2() const { return std::sqrt(norm2_sq()); }
    double norm_inf() const {
        double m = 0.0;
        for (int i = 0; i < N; ++i) m = std::max(m, std::fabs(data[i]));
        return m;
    }

    double dot(const Vec<N>& other) const {
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += data[i] * other.data[i];
        return s;
    }

    void axpy(double a, const Vec<N>& x) {
        for (int i = 0; i < N; ++i) data[i] += a * x.data[i];
    }

    void scale(double a) {
        for (int i = 0; i < N; ++i) data[i] *= a;
    }

    void copy_from(const Vec<N>& src) {
        for (int i = 0; i < N; ++i) data[i] = src.data[i];
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Fixed-size matrix  (stack-allocated, column-major for LAPACK compat)
// ─────────────────────────────────────────────────────────────────────────────

template <int Rows, int Cols>
struct Mat {
    double data[Rows * Cols];   // column-major: data[col * Rows + row]

    static constexpr int rows()   { return Rows; }
    static constexpr int cols()   { return Cols; }
    static constexpr int stride() { return Rows; }

    double  operator()(int r, int c) const { return data[c * Rows + r]; }
    double& operator()(int r, int c)       { return data[c * Rows + r]; }

    void zero() {
        for (int i = 0; i < Rows * Cols; ++i) data[i] = 0.0;
    }

    void set_identity() {
        zero();
        for (int i = 0; i < ((Rows < Cols) ? Rows : Cols); ++i)
            (*this)(i, i) = 1.0;
    }

    void set_diag(const Vec<Rows>& d) {
        zero();
        for (int i = 0; i < ((Rows < Cols) ? Rows : Cols); ++i)
            (*this)(i, i) = d[i];
    }

    // y = this * x
    void mul_vec(const Vec<Cols>& x, Vec<Rows>& y) const {
        y.zero();
        for (int c = 0; c < Cols; ++c) {
            double xc = x[c];
            for (int r = 0; r < Rows; ++r)
                y[r] += (*this)(r, c) * xc;
        }
    }

    // y = this^T * x
    void mul_vec_transpose(const Vec<Rows>& x, Vec<Cols>& y) const {
        y.zero();
        for (int c = 0; c < Cols; ++c) {
            double s = 0.0;
            for (int r = 0; r < Rows; ++r)
                s += (*this)(r, c) * x[r];
            y[c] = s;
        }
    }

    // this += a * x * y^T   (rank-1 update)
    void ger(double a, const Vec<Rows>& x, const Vec<Cols>& y) {
        for (int c = 0; c < Cols; ++c) {
            double ayc = a * y[c];
            for (int r = 0; r < Rows; ++r)
                (*this)(r, c) += x[r] * ayc;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Symmetric matrix  (stores lower triangle, column-major)
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
struct SymMat {
    double data[N * N];  // column-major lower triangle, zeros in upper

    static constexpr int dim()    { return N; }
    static constexpr int stride() { return N; }

    double  operator()(int r, int c) const {
        return (r >= c) ? data[c * N + r] : data[r * N + c];
    }
    double& operator()(int r, int c) {
        // Always store in lower triangle
        return (r >= c) ? data[c * N + r] : data[r * N + c];
    }

    void zero() { for (int i = 0; i < N * N; ++i) data[i] = 0.0; }

    void set_identity() {
        zero();
        for (int i = 0; i < N; ++i) (*this)(i, i) = 1.0;
    }

    // Copy upper/lower triangle from a general Mat (reads lower triangle)
    void copy_lower_from(const Mat<N, N>& src) {
        for (int c = 0; c < N; ++c)
            for (int r = c; r < N; ++r)
                (*this)(r, c) = src(r, c);
    }

    // LDL^T factorization (in-place). Returns true if positive definite.
    // After call: L in lower triangle, D on diagonal.
    bool ldlt_factorize(double reg = 1e-12) {
        for (int j = 0; j < N; ++j) {
            // Compute L_{j,0:j-1} * D * L_{j,0:j-1}^T contribution
            double d_jj = (*this)(j, j);
            for (int k = 0; k < j; ++k) {
                double l_jk = (*this)(j, k);  // L(j,k) = A(j,k) - ...
                double d_kk = (*this)(k, k);
                d_jj -= l_jk * l_jk * d_kk;
            }
            // If pivot is substantially negative, matrix is indefinite → fail
            if (d_jj < -1e-12) return false;
            // Regularize near-zero pivots
            if (d_jj < reg) d_jj = reg;
            (*this)(j, j) = d_jj;  // D(j,j)

            // Compute column j of L below diagonal
            for (int i = j + 1; i < N; ++i) {
                double s = (*this)(i, j);  // A(i,j)
                for (int k = 0; k < j; ++k) {
                    s -= (*this)(i, k) * (*this)(k, k) * (*this)(j, k);
                }
                (*this)(i, j) = s / d_jj;  // L(i,j)
            }
        }
        return true;
    }

    // Solve L D L^T x = b  after ldlt_factorize. x overwrites b.
    void ldlt_solve(Vec<N>& b) const {
        // Forward substitution: L z = b
        for (int i = 0; i < N; ++i) {
            double s = b[i];
            for (int j = 0; j < i; ++j)
                s -= (*this)(i, j) * b[j];  // b[j] already holds z_j
            b[i] = s;  // z_i
        }
        // D^{-1} scaling: y = D^{-1} z
        for (int i = 0; i < N; ++i)
            b[i] /= (*this)(i, i);
        // Back substitution: L^T x = y
        for (int i = N - 1; i >= 0; --i) {
            double s = b[i];
            for (int j = i + 1; j < N; ++j)
                s -= (*this)(j, i) * b[j];
            b[i] = s;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Memory pool – single pre-allocation, arena-style sub-allocation at init
// ─────────────────────────────────────────────────────────────────────────────

class MemoryPool {
public:
    MemoryPool() : buf_(nullptr), size_(0), owned_(false) {}

    ~MemoryPool() { release(); }

    // Allocate once. After this, sub-allocate workspaces.
    Status init(size_t total_bytes) {
        release();
        // _aligned_malloc on MSVC, std::aligned_alloc on C++17 POSIX
#if defined(_MSC_VER)
        buf_ = static_cast<char*>(_aligned_malloc(total_bytes, 64));
#elif defined(__cpp_aligned_new) || (__cplusplus >= 201703L)
        buf_ = static_cast<char*>(std::aligned_alloc(64, total_bytes));
#else
        // Fallback: overallocate to guarantee alignment
        buf_ = new char[total_bytes + 64];
        if (buf_) {
            size_t space = total_bytes + 64;
            void* raw = buf_;
            buf_ = static_cast<char*>(std::align(64, total_bytes, raw, space));
        }
#endif
        if (!buf_) return Status::INTERNAL_ERROR;
        size_  = total_bytes;
        owned_ = true;
        return Status::SUCCESS;
    }

    // Use externally-provided buffer (no ownership)
    Status attach(void* external_buf, size_t total_bytes) {
        release();
        buf_   = static_cast<char*>(external_buf);
        size_  = total_bytes;
        owned_ = false;
        return Status::SUCCESS;
    }

    void release() {
#ifdef _MSC_VER
        if (owned_ && buf_) { _aligned_free(buf_); }
#else
        if (owned_ && buf_) { std::free(buf_); }
#endif
        buf_   = nullptr;
        size_  = 0;
        owned_ = false;
    }

    template <typename T>
    T* allocate(size_t count = 1) {
        // Round offset up to alignment of T
        size_t align = alignof(T);
        size_t mask  = align - 1;
        size_t offset = (offset_ + mask) & ~mask;
        if (offset + count * sizeof(T) > size_) return nullptr;
        T* ptr = reinterpret_cast<T*>(buf_ + offset);
        offset_ = offset + count * sizeof(T);
        return ptr;
    }

    void reset() { offset_ = 0; }

    size_t used()     const { return offset_; }
    size_t capacity() const { return size_; }

private:
    char*  buf_;
    size_t size_;
    size_t offset_ = 0;
    bool   owned_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Solver statistics  (for monitoring / watchdog)
// ─────────────────────────────────────────────────────────────────────────────

struct SolverStats {
    int32_t  outer_iterations   = 0;  // barrier parameter updates
    int32_t  inner_iterations   = 0;  // total SQP iterations
    int32_t  riccati_failures   = 0;  // regularization activations
    int32_t  line_search_evals  = 0;
    double   barrier_param      = 0.0;
    double   primal_infeas      = 0.0;  // ||dynamics residual||∞
    double   dual_infeas        = 0.0;  // ||KKT residual||∞
    double   complementarity    = 0.0;  // avg μ for barrier
    double   cost               = 0.0;
    double   merit              = 0.0;
    double   step_norm          = 0.0;
    double   alpha_primal       = 0.0;  // step size accepted
    double   alpha_dual         = 0.0;
    double   regularization     = 0.0;  // last LM regularization used
    double   condition_estimate = 0.0;  // repurposed: s/λ inequality violation (negative = bad)
    int32_t  watchdog_resets    = 0;
    int32_t  soc_steps         = 0;     // total SOC corrections applied
    double   penalty_weight    = 0.0;   // final adaptive penalty nu
    double   linear_kkt_abs    = 0.0;   // max absolute linear KKT residual
    double   linear_kkt_rel    = 0.0;   // max relative linear KKT residual
    int32_t  linear_kkt_quality= 0;     // 0=well_solved, 1=acceptable, 2=marginal, 3=poor

    void reset() { *this = SolverStats{}; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Utility: check for NaN/Inf in a vector
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
bool is_finite(const Vec<N>& v) {
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(v[i])) return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Utility: small symmetric eigenvalue  (power iteration, for conditioning)
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
double estimate_condition(const SymMat<N>& A) {
    // Quick-and-dirty: ratio of max to min diagonal (works for SPD-ish matrices)
    double dmax = 0.0, dmin = 1e100;
    for (int i = 0; i < N; ++i) {
        double d = A(i, i);
        if (d > dmax) dmax = d;
        if (d < dmin) dmin = d;
    }
    if (dmin < 1e-14) dmin = 1e-14;
    return dmax / dmin;
}

} // namespace nmpc
