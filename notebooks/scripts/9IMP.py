import functools
from typing import List

import numpy as np


SIZE = 14


def _identity():
    return np.eye(SIZE, dtype=complex)


def _conj_transpose(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def _replace_part(arr: np.ndarray, positions: List[List[int]], value: complex) -> np.ndarray:
    out = np.array(arr, copy=True)
    for i, j in positions:
        out[i - 1, j - 1] = value
    return out


@functools.lru_cache(maxsize=None)
def beta(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    base = (omega + 1j * delta - eps) * _identity()
    positions = []
    positions.extend([[i, i - 1] for i in range(2, SIZE + 1)])
    positions.extend([[i, i + 1] for i in range(1, SIZE + 1)])
    positions.extend([[2 * n - 1, SIZE - 2 * n + 2] for n in range(1, 5)])
    positions.extend([[SIZE - 2 * n + 2, 2 * n - 1] for n in range(1, 4)])
    return _replace_part(base, positions, -t)


@functools.lru_cache(maxsize=None)
def T1(t: float) -> np.ndarray:
    base = np.zeros((SIZE, SIZE), dtype=complex)
    positions = [[2 * n, SIZE - 2 * n + 1] for n in range(1, 4)]
    return _replace_part(base, positions, t)


def TT(t: float) -> np.ndarray:
    base = np.zeros((SIZE, SIZE), dtype=complex)
    positions = [[2 * n, 2 * n] for n in range(1, 4)]
    return _replace_part(base, positions, t)


@functools.lru_cache(maxsize=None)
def LEFT(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    # Mathematica uses fixed parameters inside LEFT (delta=0.0001, t=1, eps=0)
    j = np.linalg.inv(beta(omega, 0.0001, 1.0, 0.0))
    b = np.linalg.inv(beta(omega, 0.0001, 1.0, 0.0))
    t1 = T1(1.0)
    ident = _identity()
    for _ in range(30000):
        j = np.linalg.inv(ident - b @ _conj_transpose(t1) @ j @ t1) @ b
    return j


def g(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    return np.linalg.inv(beta(omega, delta, t, eps))


@functools.lru_cache(maxsize=None)
def SR(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    gg = g(omega, delta, t, eps)
    t1 = T1(t)
    ident = _identity()
    return np.linalg.inv(ident - gg @ _conj_transpose(t1) @ LEFT(omega, delta, t, eps) @ t1) @ gg


@functools.lru_cache(maxsize=None)
def SL(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    gg = g(omega, delta, t, eps)
    t1 = T1(t)
    ident = _identity()
    return np.linalg.inv(ident - gg @ _conj_transpose(t1) @ LEFT(omega, delta, t, eps) @ t1) @ gg


def IL(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    ident = _identity()
    return np.linalg.inv(ident - SL(omega, delta, t, eps) @ TT(t) @ SR(omega, delta, t, eps) @ TT(t)) @ SL(
        omega, delta, t, eps
    )


def IR(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    ident = _identity()
    return np.linalg.inv(ident - SR(omega, delta, t, eps) @ TT(t) @ SL(omega, delta, t, eps) @ TT(t)) @ SR(
        omega, delta, t, eps
    )


def gdd(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    il = IL(omega, delta, t, eps)
    return il - _conj_transpose(il)


def grr(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    ir = IR(omega, delta, t, eps)
    return ir - _conj_transpose(ir)


def Gnonlocal(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    return SR(omega, delta, t, eps) @ TT(t) @ IL(omega, delta, t, eps)


def GNON(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    gnl = Gnonlocal(omega, delta, t, eps)
    return gnl - _conj_transpose(gnl)


def tr(omega: float, delta: float, t: float, eps: float) -> complex:
    return np.trace(
        gdd(omega, delta, t, eps) @ TT(t) @ grr(omega, delta, t, eps) @ TT(t)
        - TT(t) @ GNON(omega, delta, t, eps) @ TT(t) @ GNON(omega, delta, t, eps)
    )


def _imp_fixed(index: int, omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    kappa = beta(omega, delta, t, eps)
    return np.linalg.inv(_replace_part(kappa, [[index, index]], omega + 1j * delta - eps1))


def imp1(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(1, omega, delta, t, eps, eps1)


def imp2(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(2, omega, delta, t, eps, eps1)


def imp3(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(3, omega, delta, t, eps, eps1)


def imp4(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(4, omega, delta, t, eps, eps1)


def imp5(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(5, omega, delta, t, eps, eps1)


def imp6(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(6, omega, delta, t, eps, eps1)


def imp7(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(7, omega, delta, t, eps, eps1)


def imp8(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(8, omega, delta, t, eps, eps1)


def imp9(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(9, omega, delta, t, eps, eps1)


def imp10(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(10, omega, delta, t, eps, eps1)


def imp11(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(11, omega, delta, t, eps, eps1)


def imp12(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(12, omega, delta, t, eps, eps1)


def imp13(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(13, omega, delta, t, eps, eps1)


def imp14(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    return _imp_fixed(14, omega, delta, t, eps, eps1)


def imp(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    kappa = beta(omega, delta, t, eps)
    # Note: Mathematica uses omega + I*delta + 0 at site 14.
    return np.linalg.inv(_replace_part(kappa, [[14, 14]], omega + 1j * delta + 0.0))


def set1(omega: float, delta: float, t: float, eps: float, eps1: float, num: int, s: int, rng=None):
    rng = rng or np.random.default_rng()
    imp_funcs = [
        imp1,
        imp2,
        imp3,
        imp4,
        imp5,
        imp6,
        imp7,
        imp8,
        imp9,
        imp10,
        imp11,
        imp12,
        imp13,
        imp14,
    ]
    chosen = [rng.choice(imp_funcs)(omega, delta, t, eps, eps1) for _ in range(num)]
    fillers = [imp(omega, delta, t, eps, eps1) for _ in range(100 - num)]
    pool = chosen + fillers
    # RandomChoice[..., 100] yields a list of length 100 sampled with replacement
    return [pool[rng.integers(0, len(pool))] for _ in range(100)]


def tr1(omega: float, delta: float, t: float, eps: float, eps1: float, num: int, s: int, rng=None) -> float:
    rng = rng or np.random.default_rng()
    x = T1(1.0)
    tmat = TT(1.0)

    cc = set1(omega, delta, t, eps, eps1, num, s, rng=rng)

    ident = _identity()
    j = SL(omega, delta, t, eps)
    for m in range(s):
        j = np.linalg.inv(ident - cc[m] @ _conj_transpose(x) @ j @ x) @ cc[m]
    sl1 = j

    il1 = np.linalg.inv(ident - sl1 @ tmat @ SR(omega, delta, t, 0.0) @ tmat) @ sl1
    ir1 = np.linalg.inv(ident - SR(omega, delta, 1.0, 0.0) @ tmat @ sl1 @ tmat) @ SR(omega, delta, 1.0, 0.0)

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, delta, 1.0, 0.0) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)

    val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))
    return float(val)
