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
    for _ in range(8000):
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


def tr(omega: float, delta: float, t: float, eps: float) -> float:
    val = np.trace(
        gdd(omega, delta, t, eps) @ TT(t) @ grr(omega, delta, t, eps) @ TT(t)
        - TT(t) @ GNON(omega, delta, t, eps) @ TT(t) @ GNON(omega, delta, t, eps)
    )
    return float(np.abs(val))


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def imp1(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    i = _randint(rng, 1, 14)
    base = beta(0.0, 0.0, 0.0, 0.0)
    return np.linalg.inv(beta(omega, delta, t, eps) - _replace_part(base, [[i, i]], eps1))


def imp2(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    i = _randint(rng, 1, 7)
    j = _randint(rng, 8, 14)
    base = beta(0.0, 0.0, 0.0, 0.0)
    return np.linalg.inv(beta(omega, delta, t, eps) - _replace_part(base, [[i, i], [j, j]], eps1))


def imp(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    i = _randint(rng, 1, 7)
    j = _randint(rng, 8, 14)
    base = beta(0.0, 0.0, 0.0, 0.0)
    return np.linalg.inv(beta(omega, delta, t, eps) - _replace_part(base, [[i, i], [j, j]], 0.0))


def _build_list(tokens, omega, delta, t, eps, eps1, rng):
    items = []
    for token in tokens:
        if token == 'imp':
            items.append(imp(omega, delta, t, eps, eps1, rng=rng))
        elif token == 'imp1':
            items.append(imp1(omega, delta, t, eps, eps1, rng=rng))
        elif token == 'imp2':
            items.append(imp2(omega, delta, t, eps, eps1, rng=rng))
    rng.shuffle(items)
    return items


TRA_TOKENS = ['imp', 'imp', 'imp', 'imp2', 'imp', 'imp', 'imp', 'imp', 'imp1', 'imp', 'imp1', 'imp', 'imp1', 'imp1', 'imp1', 'imp', 'imp', 'imp', 'imp', 'imp1', 'imp1', 'imp', 'imp1', 'imp', 'imp1', 'imp1', 'imp', 'imp', 'imp', 'imp2', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp1', 'imp', 'imp1', 'imp', 'imp2', 'imp', 'imp1', 'imp2', 'imp', 'imp', 'imp1', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp1', 'imp1', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp1', 'imp', 'imp', 'imp', 'imp2', 'imp2', 'imp', 'imp', 'imp', 'imp2', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp', 'imp1', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp1', 'imp1', 'imp', 'imp1', 'imp1', 'imp1', 'imp', 'imp', 'imp1', 'imp1', 'imp', 'imp', 'imp', 'imp2', 'imp1', 'imp', 'imp']
TRA2_TOKENS = ['imp2', 'imp', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp1', 'imp2', 'imp1', 'imp', 'imp2', 'imp1', 'imp1', 'imp2', 'imp2', 'imp', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp1', 'imp', 'imp1', 'imp', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp', 'imp', 'imp', 'imp1', 'imp2', 'imp2', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp1', 'imp2', 'imp', 'imp', 'imp', 'imp1', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp1', 'imp1', 'imp2', 'imp', 'imp2', 'imp', 'imp2', 'imp', 'imp1', 'imp1']
TRA3_TOKENS = ['imp1', 'imp2', 'imp2', 'imp', 'imp2', 'imp1', 'imp1', 'imp2', 'imp2', 'imp', 'imp1', 'imp2', 'imp1', 'imp', 'imp1', 'imp2', 'imp1', 'imp', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp2', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp1', 'imp2', 'imp1', 'imp2', 'imp', 'imp', 'imp2', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp2', 'imp1', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp1', 'imp', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp', 'imp2', 'imp2', 'imp2', 'imp1', 'imp', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp2', 'imp1', 'imp2', 'imp', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp1', 'imp2', 'imp1', 'imp1', 'imp1']
TRA4_TOKENS = ['imp2', 'imp2', 'imp1', 'imp2', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp', 'imp1', 'imp2', 'imp1', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp1', 'imp', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp2', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp2', 'imp1', 'imp2', 'imp2', 'imp2', 'imp2', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp2', 'imp2', 'imp', 'imp2', 'imp2', 'imp2', 'imp2']


def _tra_common(tokens, omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> float:
    rng = rng or np.random.default_rng()
    tin = T1(1.0)
    tmat = TT(1.0)
    list_items = _build_list(tokens, omega, delta, t, eps, eps1, rng)

    ident = _identity()
    j = SL(omega, 0.0001, 1.0, 0.0)
    for item in list_items:
        j = np.linalg.inv(ident - item @ _conj_transpose(tin) @ j @ tin) @ item
    sl1 = j

    il1 = np.linalg.inv(ident - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0) @ tmat) @ sl1
    ir1 = np.linalg.inv(ident - SR(omega, 0.0001, 1.0, 0.0) @ tmat @ sl1 @ tmat) @ SR(omega, 0.0001, 1.0, 0.0)

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)

    val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))
    return float(3.0 if val > 3.0 else val)


def tra(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> float:
    return _tra_common(TRA_TOKENS, omega, delta, t, eps, eps1, rng=rng)


def tra2(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> float:
    return _tra_common(TRA2_TOKENS, omega, delta, t, eps, eps1, rng=rng)


def tra3(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> float:
    return _tra_common(TRA3_TOKENS, omega, delta, t, eps, eps1, rng=rng)


def tra4(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> float:
    return _tra_common(TRA4_TOKENS, omega, delta, t, eps, eps1, rng=rng)
