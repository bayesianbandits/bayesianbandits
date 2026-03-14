# cython: boundscheck=False, wraparound=False, cdivision=True
"""Microbenchmark: pow() vs manual arithmetic for the operations
used in Takahashi diagonal."""


def bench_manual(int n):
    """1.0 / (x * x) — manual multiplication."""
    cdef double x = 2.5
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += 1.0 / (x * x)
        x += 0.0001
    return result


def bench_pow_neg2(int n):
    """x ** (-2.0) — Cython pow dispatch."""
    cdef double x = 2.5
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += x ** (-2.0)
        x += 0.0001
    return result


def bench_inv_manual(int n):
    """1.0 / x — manual division."""
    cdef double x = 2.5
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += 1.0 / x
        x += 0.0001
    return result


def bench_inv_pow(int n):
    """x ** (-1.0) — Cython pow dispatch."""
    cdef double x = 2.5
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += x ** (-1.0)
        x += 0.0001
    return result


def bench_sq_manual(int n):
    """x * x — manual multiplication."""
    cdef double x = 2.5
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += x * x
        x += 0.0001
    return result


def bench_sq_pow(int n):
    """x ** 2.0 — Cython pow dispatch."""
    cdef double x = 2.5
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += x ** 2.0
        x += 0.0001
    return result
