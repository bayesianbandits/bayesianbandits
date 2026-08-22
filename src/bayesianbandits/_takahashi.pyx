# cython: boundscheck=False, wraparound=False, cdivision=True
"""Takahashi diagonal recursion — Cython-accelerated selected inversion.

Compute diag((LL')⁻¹) via backward recursion through L's CSC sparsity
structure, supernodally: columns with identical below-diagonal
structure form one dense block column, and the recursion for the block
is a handful of BLAS calls rather than a scatter/gather per column.
The scalar recursion (à la SuiteSparse sparseinv) remains for
singleton supernodes.
"""
import numpy as np
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_GetName

# BLAS/LAPACK entry points from scipy's ``__pyx_capi__`` capsules rather
# than ``cimport``, which would need scipy at build time.
ctypedef void (*dgemm_t)(char *, char *, int *, int *, int *, double *,
                         double *, int *, double *, int *, double *,
                         double *, int *) noexcept nogil
ctypedef void (*dtrsm_t)(char *, char *, char *, char *, int *, int *,
                         double *, double *, int *, double *, int *) noexcept nogil
ctypedef void (*dtrtri_t)(char *, char *, int *, double *, int *, int *) noexcept nogil

cdef dgemm_t dgemm
cdef dtrsm_t dtrsm
cdef dtrtri_t dtrtri


# Capsule names are C signatures; scipy's ``d`` typedef is ``double``.
_SIGNATURES = {
    "dgemm": "void (char *, char *, int *, int *, int *, d *, d *, int *, "
    "d *, int *, d *, d *, int *)",
    "dtrsm": "void (char *, char *, char *, char *, int *, int *, d *, d *, "
    "int *, d *, int *)",
    "dtrtri": "void (char *, char *, int *, d *, int *, int *)",
}


cdef void *_capsule_pointer(module, str name) except NULL:
    """The function pointer behind ``module.__pyx_capi__[name]``; a
    signature other than the one compiled for raises at import."""
    import re

    cap = module.__pyx_capi__[name]
    signature = (<bytes>PyCapsule_GetName(cap)).decode()
    normalized = re.sub(r"__pyx_t_\w+_d\b", "d", signature)
    if normalized != _SIGNATURES[name]:
        raise ImportError(
            f"scipy's {name} signature {signature!r} does not match the one "
            f"bayesianbandits._takahashi was built for"
        )
    return PyCapsule_GetPointer(cap, PyCapsule_GetName(cap))


def _load_blas():
    import scipy.linalg.cython_blas as _blas
    import scipy.linalg.cython_lapack as _lapack
    global dgemm, dtrsm, dtrtri
    dgemm = <dgemm_t>_capsule_pointer(_blas, "dgemm")
    dtrsm = <dtrsm_t>_capsule_pointer(_blas, "dtrsm")
    dtrtri = <dtrtri_t>_capsule_pointer(_lapack, "dtrtri")


_load_blas()


cdef int _supernode_starts(
    const int[::1] indices, const int[::1] indptr, int p, int[::1] starts
) noexcept nogil:
    """Fundamental supernodes of a sorted lower-triangular CSC ``L``: runs
    of consecutive columns whose sub-diagonal structure is ``{j+1}`` plus
    column ``j+1``'s. Writes start columns (then ``p``) into ``starts``."""
    cdef int j, a0, a1, b0, b1, k, count = 1, same
    starts[0] = 0
    for j in range(1, p):
        a0 = indptr[j - 1]
        a1 = indptr[j]
        b0 = indptr[j]
        b1 = indptr[j + 1]
        same = 0
        if (a1 - a0) == (b1 - b0) + 1 and a1 - a0 >= 2 and indices[a0 + 1] == j:
            same = 1
            for k in range(b1 - b0 - 1):
                if indices[a0 + 2 + k] != indices[b0 + 1 + k]:
                    same = 0
                    break
        if not same:
            starts[count] = j
            count += 1
    starts[count] = p
    return count


cdef inline void _scalar_column(
    int j,
    const double[::1] data,
    const int[::1] indices,
    const int[::1] indptr,
    double[::1] z_data,
    double[::1] z_diag,
    double[::1] z_work,
    double[::1] z_col,
) noexcept nogil:
    """Takahashi recursion for one column by scatter/gather over Z's
    stored columns (à la SuiteSparse sparseinv)."""
    cdef int col_start = indptr[j]
    cdef int col_end = indptr[j + 1]
    cdef int n_sub = col_end - col_start - 1
    cdef int a, b, k, ra, ra_start, ra_end, sub_start
    cdef double l_jj, inv_l_jj, dot, neg_inv, sv_a, z_ab
    if n_sub == 0:
        return
    l_jj = data[col_start]
    inv_l_jj = 1.0 / l_jj
    sub_start = col_start + 1

    for a in range(n_sub):
        z_col[a] = z_diag[indices[sub_start + a]] * data[sub_start + a]

    for a in range(n_sub):
        ra = indices[sub_start + a]
        ra_start = indptr[ra]
        ra_end = indptr[ra + 1]
        for k in range(ra_start + 1, ra_end):
            z_work[indices[k]] = z_data[k]
        sv_a = data[sub_start + a]
        for b in range(a + 1, n_sub):
            z_ab = z_work[indices[sub_start + b]]
            z_col[a] += z_ab * data[sub_start + b]
            z_col[b] += z_ab * sv_a
        for k in range(ra_start + 1, ra_end):
            z_work[indices[k]] = 0.0

    dot = 0.0
    for a in range(n_sub):
        dot += data[sub_start + a] * z_col[a]
    neg_inv = -inv_l_jj
    for a in range(n_sub):
        z_data[sub_start + a] = z_col[a] * neg_inv
    z_diag[j] = inv_l_jj * inv_l_jj * (1.0 + dot)


# Workspace for one supernode's dense block, in doubles, laid out flat and
# Fortran-ordered: L_SS (S x S) | M (S x S) | L_RS (m x S) | W (m x S) |
# Z_B (m x cols). Z_B is a column panel of the symmetric Z_RR, sized so no
# term grows with m * m — that's what keeps the peak workspace bounded
# regardless of fill-in.
cdef inline Py_ssize_t _dense_work(
    int S, int m, Py_ssize_t panel_bytes, int *panel_cols
) noexcept nogil:
    cdef Py_ssize_t cols = 0
    if m > 0:
        cols = panel_bytes // (<Py_ssize_t>sizeof(double) * <Py_ssize_t>m)
        if cols < 1:
            cols = 1
        elif cols > m:
            cols = m
    panel_cols[0] = <int>cols
    return (2 * (<Py_ssize_t>S) * S + 2 * (<Py_ssize_t>m) * S
            + cols * (<Py_ssize_t>m))


def takahashi_diagonal(
    const double[::1] data,
    const int[::1] indices,
    const int[::1] indptr,
    int p,
    int min_dense_cols=4,
    int max_dense_cols=1024,
    Py_ssize_t panel_bytes=8388608,
):
    """Compute diag((LL')⁻¹) from CSC arrays of a lower-triangular L.

    Parameters
    ----------
    data : float64 array — CSC data
    indices : int32 array — CSC row indices
    indptr : int32 array — CSC column pointers
    p : int — matrix dimension
    min_dense_cols : int — supernodes with at least this many columns take
        the dense BLAS path; narrower ones run the scalar recursion.
        Internal: exposed for tests, not a tuning knob.
    max_dense_cols : int — widest block column handed to the dense kernel;
        wider supernodes are split. Bounds the ``S x S`` buffers at
        ``2 * max_dense_cols ** 2`` (~17 MB at the default) regardless of
        fill-in. Internal.
    panel_bytes : int — cache-blocking size for the ``Z_RR`` column panel,
        in bytes. Internal.

    Returns
    -------
    ndarray of float64, length p
    """
    if p == 0:
        return np.zeros(0, dtype=np.float64)

    cdef int nnz = data.shape[0]
    cdef double[::1] z_data = np.zeros(nnz, dtype=np.float64)
    cdef double[::1] z_diag = np.zeros(p, dtype=np.float64)
    cdef double[::1] z_work = np.zeros(p, dtype=np.float64)

    cdef int j, a, b, k, col_len, n_sub, col_start, max_sub
    cdef double l_jj

    # First pass: handle trivial columns, find max_sub for z_col buffer.
    max_sub = 0
    for j in range(p):
        col_len = indptr[j + 1] - indptr[j]
        if col_len == 1:
            l_jj = data[indptr[j]]
            z_diag[j] = 1.0 / (l_jj * l_jj)
        else:
            n_sub = col_len - 1
            if n_sub > max_sub:
                max_sub = n_sub

    cdef double[::1] z_col = np.zeros(max(max_sub, 1), dtype=np.float64)

    # Supernode partition.
    cdef int[::1] starts = np.empty(p + 1, dtype=np.int32)
    cdef int n_super = _supernode_starts(indices, indptr, p, starts)

    # Split wide supernodes into balanced block columns, each itself a valid
    # supernode of the columns to its right, so this bounds the S x S
    # buffers without changing the algorithm.
    cdef int s, f, l, S, m, cols = 0
    cdef int[::1] blocks = np.empty(p + 1, dtype=np.int32)
    cdef int n_blocks = 0, pieces, width
    for s in range(n_super):
        f = starts[s]
        l = starts[s + 1]
        S = l - f
        if S <= max_dense_cols:
            blocks[n_blocks] = f
            n_blocks += 1
            continue
        pieces = (S + max_dense_cols - 1) // max_dense_cols
        width = (S + pieces - 1) // pieces
        j = f
        while j < l:
            blocks[n_blocks] = j
            n_blocks += 1
            j += width
    blocks[n_blocks] = p

    # One flat workspace sized to the largest single dense block (LAPACK
    # ``lwork`` style), not a cross product of maxima over unrelated blocks.
    cdef Py_ssize_t need, lwork = 0
    for s in range(n_blocks):
        f = blocks[s]
        S = blocks[s + 1] - f
        if S < min_dense_cols:
            continue
        m = indptr[f + 1] - indptr[f] - S
        need = _dense_work(S, m, panel_bytes, &cols)
        if need > lwork:
            lwork = need
    cdef double[::1] work = np.zeros(max(lwork, 1), dtype=np.float64)
    cdef double *wbuf = &work[0]
    cdef double *LSS
    cdef double *M
    cdef double *LRS
    cdef double *W
    cdef double *ZB

    cdef int c0, r, r0, r1, i, info, q0, q1, nb
    cdef double one = 1.0, minus_one = -1.0, zd, tmp
    cdef char *side_r = "R"
    cdef char *uplo_l = "L"
    cdef char *trans_n = "N"
    cdef char *trans_t = "T"
    cdef char *diag_n = "N"

    # Backward pass over block columns.
    for s in range(n_blocks - 1, -1, -1):
        f = blocks[s]
        l = blocks[s + 1]
        S = l - f
        c0 = indptr[f]
        m = indptr[f + 1] - c0 - S

        if S < min_dense_cols:
            for j in range(l - 1, f - 1, -1):  # column j reads Z's column j + 1
                _scalar_column(j, data, indices, indptr, z_data, z_diag, z_work, z_col)
            continue

        _dense_work(S, m, panel_bytes, &cols)
        LSS = wbuf
        M = LSS + (<Py_ssize_t>S) * S
        LRS = M + (<Py_ssize_t>S) * S
        W = LRS + (<Py_ssize_t>m) * S
        ZB = W + (<Py_ssize_t>m) * S

        # Column f + k of L holds rows f+k .. l-1 (the lower triangle of
        # L_SS) followed by the m rows R below.
        for k in range(S):
            col_start = indptr[f + k]
            for i in range(k):
                LSS[i + k * S] = 0.0
            for i in range(S - k):
                LSS[k + i + k * S] = data[col_start + i]
            for i in range(m):
                LRS[i + k * m] = data[col_start + S - k + i]

        if m > 0:
            # W = Z_RR L_RS without forming Z_RR: split it into D + Z + Z'
            # (Z strictly lower) and walk Z one column panel at a time.
            for a in range(m):
                zd = z_diag[indices[c0 + S + a]]
                for k in range(S):
                    W[a + k * m] = zd * LRS[a + k * m]
            q0 = 0
            while q0 < m:
                q1 = q0 + cols
                if q1 > m:
                    q1 = m
                nb = q1 - q0
                for b in range(q0, q1):
                    r = indices[c0 + S + b]
                    r0 = indptr[r] + 1
                    r1 = indptr[r + 1]
                    for k in range(r0, r1):
                        z_work[indices[k]] = z_data[k]
                    for a in range(b + 1):
                        ZB[a + (b - q0) * m] = 0.0
                    for a in range(b + 1, m):
                        ZB[a + (b - q0) * m] = z_work[indices[c0 + S + a]]
                    for k in range(r0, r1):
                        z_work[indices[k]] = 0.0
                # W += Z_B L_RS[B, :]
                dgemm(trans_n, trans_n, &m, &S, &nb, &one, ZB, &m,
                      LRS + q0, &m, &one, W, &m)
                # W += (Z_B)' L_RS, landing on rows B of W
                dgemm(trans_t, trans_n, &nb, &S, &m, &one, ZB, &m,
                      LRS, &m, &one, W + q0, &m)
                q0 = q1
            # Z_RS = -W L_SS^{-1}                              (in place in W)
            dtrsm(side_r, uplo_l, trans_n, diag_n, &m, &S, &minus_one,
                  LSS, &S, W, &m)

        # M = L_SS^{-T} - Z_RS' L_RS                          (S x S)
        for i in range((<Py_ssize_t>S) * S):
            M[i] = LSS[i]
        dtrtri(uplo_l, diag_n, &S, M, &S, &info)
        for k in range(S):  # transpose in place; dtrtri left the zeros above
            for i in range(k + 1, S):
                tmp = M[i + k * S]
                M[i + k * S] = M[k + i * S]
                M[k + i * S] = tmp
        if m > 0:
            dgemm(trans_t, trans_n, &S, &S, &m, &minus_one, W, &m,
                  LRS, &m, &one, M, &S)
        # Z_SS = M L_SS^{-1}
        dtrsm(side_r, uplo_l, trans_n, diag_n, &S, &S, &one,
              LSS, &S, M, &S)

        # Store Z's columns in L's pattern.
        for k in range(S):
            col_start = indptr[f + k]
            z_diag[f + k] = M[k + k * S]
            for i in range(1, S - k):
                z_data[col_start + i] = M[k + i + k * S]
            for i in range(m):
                z_data[col_start + S - k + i] = W[i + k * m]

    return np.asarray(z_diag)
