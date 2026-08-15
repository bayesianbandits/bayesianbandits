# cython: boundscheck=False, wraparound=False, cdivision=True
"""Reach-limited sparse triangular solve — Cython-accelerated.

Solve ``L y = b`` for a lower-triangular CSC ``L`` and a sparse ``b``,
touching only the nodes reachable from ``b``'s support in the graph of
``L`` (Gilbert-Peierls). The caller owns persistent workspace so a
solve allocates O(reach).
"""
import numpy as np


def reach_solve(
    const double[::1] data,
    const int[::1] indices,
    const int[::1] indptr,
    const long long[::1] nodes,
    const double[::1] vals,
    double[::1] work,
    long long[::1] stamp,
    long long gen,
    long long node_budget,
    long long entry_budget,
):
    """Reach-limited solve of ``L y = b`` with sparse ``b``.

    Parameters
    ----------
    data, indices, indptr : CSC arrays of lower-triangular L, sorted
        indices with the diagonal first in every column.
    nodes : int64 array — rows of b's support (in L's row order).
    vals : float64 array — values of b on ``nodes``.
    work : float64 array, length p — caller-persistent scratch, zero
        outside this call; restored to zero on the touched entries.
    stamp : int64 array, length p — caller-persistent visit stamps.
    gen : int — fresh generation for this call (greater than any
        stamp value written by earlier generations).
    node_budget, entry_budget : int — bail thresholds on reach nodes
        and traversed entries.

    Returns
    -------
    (reach, values, entries) — sorted reach rows (intp), the solution
    on those rows (float64), and the entries traversed — or None when
    a budget is exceeded (touched state is safe to abandon: the next
    call uses a fresh generation and ``work`` is only written after
    the reach is fully collected).
    """
    cdef Py_ssize_t n_nodes = nodes.shape[0]
    cdef long long[::1] reach = np.empty(node_budget + 1, dtype=np.int64)
    cdef long long[::1] stack = np.empty(node_budget + 1, dtype=np.int64)
    cdef Py_ssize_t nr = 0, sp = 0
    cdef long long entries = 0
    cdef long long j, i
    cdef Py_ssize_t k, idx, s, e
    cdef double xj

    for k in range(n_nodes):
        j = nodes[k]
        if stamp[j] != gen:
            if nr >= node_budget:
                return None
            stamp[j] = gen
            reach[nr] = j
            nr += 1
            stack[sp] = j
            sp += 1

    while sp > 0:
        sp -= 1
        j = stack[sp]
        s = indptr[j] + 1
        e = indptr[j + 1]
        entries += e - s
        if entries > entry_budget:
            return None
        for idx in range(s, e):
            i = indices[idx]
            if stamp[i] != gen:
                if nr >= node_budget:
                    return None
                stamp[i] = gen
                reach[nr] = i
                nr += 1
                stack[sp] = i
                sp += 1

    # lower-triangular: ascending index order is a topological order
    reach_np = np.sort(np.asarray(reach[:nr]))
    cdef const long long[::1] order = reach_np

    for k in range(n_nodes):
        work[nodes[k]] += vals[k]

    for k in range(nr):
        j = order[k]
        s = indptr[j]
        e = indptr[j + 1]
        xj = work[j] / data[s]
        work[j] = xj
        if xj != 0.0:
            for idx in range(s + 1, e):
                work[indices[idx]] -= data[idx] * xj

    out = np.empty(nr, dtype=np.float64)
    cdef double[::1] out_mv = out
    for k in range(nr):
        j = order[k]
        out_mv[k] = work[j]
        work[j] = 0.0

    return reach_np.astype(np.intp, copy=False), out, int(entries)
