#pragma once
//
// Leftmost row maxima of an implicit n x m matrix, three ways.
//
// Each solver fills argmax[r] for rows r = 1..n (index 0 unused) with a column
// in 1..m. The matrix is a callable f(row, col) -> double, 1-based, evaluated
// on demand: the DP's matrix is never materialised.
//
// Tie-breaking is copied exactly from the reference -- a strictly greater value
// is needed to move right, so the leftmost maximum wins. On ties it decides
// which of two equally good partitions is reported, and the golden tests
// compare partitions exactly.
//
// full_scan_maxima is exact on any matrix. monotone_maxima needs the row argmax
// to be non-decreasing; smawk needs the matrix to be totally monotone. On a
// matrix that is neither, both still return a column for every row, but not
// necessarily the row's maximum, and not necessarily one with a finite value.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace plbf {

namespace detail {

// Leftmost maximum of row `row` over columns lo..hi. Starts from -inf, so a row
// that is -inf everywhere reports `lo` -- the reference's CalcJ, exactly.
template <class F>
std::uint32_t scan_row(F& f, std::size_t row, std::size_t lo, std::size_t hi) {
    double best = -std::numeric_limits<double>::infinity();
    std::size_t arg = lo;
    for (std::size_t c = lo; c <= hi; ++c) {
        const double v = f(row, c);
        if (v > best) {
            best = v;
            arg = c;
        }
    }
    return static_cast<std::uint32_t>(arg);
}

template <class F>
void monotone_rec(F& f, std::size_t row_lo, std::size_t row_hi, std::size_t col_lo,
                  std::size_t col_hi, std::uint32_t* argmax) {
    // Rows are 1-based, so row_hi = mid - 1 never wraps below row_lo - 1 >= 0.
    if (row_lo > row_hi) return;
    const std::size_t mid = (row_lo + row_hi) / 2;
    const std::uint32_t best = scan_row(f, mid, col_lo, col_hi);
    argmax[mid] = best;
    monotone_rec(f, row_lo, mid - 1, col_lo, best, argmax);
    monotone_rec(f, mid + 1, row_hi, best, col_hi, argmax);
}

// Discard columns that cannot hold a (leftmost) row maximum, leaving at most
// |rows| of them. The reference's reduce, step for step.
template <class F>
void smawk_reduce(F& f, const std::vector<std::uint32_t>& rows,
                  const std::vector<std::uint32_t>& cols, std::vector<std::uint32_t>& out) {
    out.clear();
    for (const std::uint32_t c : cols) {
        while (!out.empty()) {
            const std::uint32_t r = rows[out.size() - 1];
            if (f(r, out.back()) < f(r, c)) {
                out.pop_back();
            } else {
                break;
            }
        }
        if (out.size() < rows.size()) out.push_back(c);
    }
}

// `pos` is scratch indexed by column. Each level writes the positions of its
// own columns after its recursive call returns and reads only those, so one
// array serves the whole recursion.
template <class F>
void smawk_solve(F& f, const std::vector<std::uint32_t>& rows, const std::vector<std::uint32_t>& cols_in,
                 std::uint32_t* argmax, std::vector<std::uint32_t>& pos) {
    if (rows.empty()) return;
    std::vector<std::uint32_t> cols;
    cols.reserve(rows.size());
    smawk_reduce(f, rows, cols_in, cols);

    std::vector<std::uint32_t> odd;
    odd.reserve(rows.size() / 2);
    for (std::size_t i = 1; i < rows.size(); i += 2) odd.push_back(rows[i]);
    smawk_solve(f, odd, cols, argmax, pos);

    for (std::size_t i = 0; i < cols.size(); ++i) pos[cols[i]] = static_cast<std::uint32_t>(i);

    // Even rows: scan only between the answers of the neighbouring odd rows.
    std::size_t k = 0;
    for (std::size_t i = 0; i < rows.size(); i += 2) {
        const std::uint32_t r = rows[i];
        const std::size_t end = i + 1 < rows.size() ? pos[argmax[rows[i + 1]]] : cols.size() - 1;
        std::uint32_t best = cols[k];
        double best_val = f(r, best);
        for (std::size_t t = k + 1; t <= end; ++t) {
            const double v = f(r, cols[t]);
            if (v > best_val) {
                best_val = v;
                best = cols[t];
            }
        }
        argmax[r] = best;
        k = end;
    }
}

}  // namespace detail

// O(n m) evaluations. Exact on any matrix; the oracle for the other two.
template <class F>
void full_scan_maxima(F&& f, std::size_t n, std::size_t m, std::vector<std::uint32_t>& argmax) {
    argmax.assign(n + 1, 0);
    for (std::size_t r = 1; r <= n; ++r) argmax[r] = detail::scan_row(f, r, 1, m);
}

// O((n + m) log n) evaluations: solve the middle row, then each half within
// the columns the monotone argmax leaves it.
template <class F>
void monotone_maxima(F&& f, std::size_t n, std::size_t m, std::vector<std::uint32_t>& argmax) {
    argmax.assign(n + 1, 0);
    detail::monotone_rec(f, 1, n, 1, m, argmax.data());
}

// O(n + m) evaluations (Aggarwal, Klawe, Moran, Shor and Wilber, 1987).
template <class F>
void smawk(F&& f, std::size_t n, std::size_t m, std::vector<std::uint32_t>& argmax) {
    argmax.assign(n + 1, 0);
    std::vector<std::uint32_t> rows(n), cols(m), pos(m + 1, 0);
    for (std::size_t i = 0; i < n; ++i) rows[i] = static_cast<std::uint32_t>(i + 1);
    for (std::size_t i = 0; i < m; ++i) cols[i] = static_cast<std::uint32_t>(i + 1);
    detail::smawk_solve(f, rows, cols, argmax.data(), pos);
}

}  // namespace plbf
