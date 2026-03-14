# cython: boundscheck=False, wraparound=False, cdivision=True
"""Variant: typed memoryviews instead of old ndarray buffer syntax."""
import numpy as np
cimport numpy as cnp

cnp.import_array()


def takahashi_diagonal(
    double[::1] data,
    int[::1] indices,
    int[::1] indptr,
    int p,
):
    cdef int nnz = data.shape[0]
    cdef double[::1] z_data = np.zeros(nnz, dtype=np.float64)
    cdef double[::1] z_diag = np.zeros(p, dtype=np.float64)
    cdef double[::1] z_work = np.zeros(p, dtype=np.float64)

    cdef int j, a, b, k, col_len, n_sub, col_start, col_end, sub_start
    cdef int ra, ra_start, ra_end, idx, max_sub
    cdef double l_jj, inv_l_jj, dot, neg_inv, sv_a, z_ab

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

    cdef double[::1] z_col = np.zeros(max_sub, dtype=np.float64)

    for j in range(p - 1, -1, -1):
        col_start = indptr[j]
        col_end = indptr[j + 1]
        n_sub = col_end - col_start - 1
        if n_sub == 0:
            continue

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
                idx = indices[k]
                z_work[idx] = z_data[k]

            sv_a = data[sub_start + a]
            for b in range(a + 1, n_sub):
                z_ab = z_work[indices[sub_start + b]]
                z_col[a] += z_ab * data[sub_start + b]
                z_col[b] += z_ab * sv_a

            for k in range(ra_start + 1, ra_end):
                idx = indices[k]
                z_work[idx] = 0.0

        dot = 0.0
        for a in range(n_sub):
            dot += data[sub_start + a] * z_col[a]

        neg_inv = -inv_l_jj
        for a in range(n_sub):
            z_data[sub_start + a] = z_col[a] * neg_inv

        z_diag[j] = inv_l_jj * inv_l_jj * (1.0 + dot)

    return np.asarray(z_diag)
