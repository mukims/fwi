import functools
from typing import List, Optional

import numpy as np

SIZE = 14


def _identity() -> np.ndarray:
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
    j = np.linalg.inv(beta(omega, 0.0001, 1.0, 0.0))
    b = np.linalg.inv(beta(omega, 0.0001, 1.0, 0.0))
    t1 = T1(1.0)
    ident = _identity()
    for _ in range(10000):
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


# Data helpers

def load_series(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=',')


def sample_k(series: np.ndarray, omega: float) -> float:
    idx = int(round(omega * 100))
    idx = max(0, min(idx, len(series) - 1))
    return float(series[idx][1])


def pri(omega: float) -> float:
    return tr(omega, 0.0001, 1.0, 0.0)


def f_values(omega: float, series_map: dict) -> List[float]:
    return [
        pri(omega),
        sample_k(series_map['k14'], omega),
        sample_k(series_map['k28'], omega),
        sample_k(series_map['k42'], omega),
        sample_k(series_map['k56'], omega),
        sample_k(series_map['k70'], omega),
        sample_k(series_map['k84'], omega),
        sample_k(series_map['k100'], omega),
        sample_k(series_map['k114'], omega),
        sample_k(series_map['k126'], omega),
    ]


def beta_beta(omega: float, j: int, x: int, series_map: dict) -> List[float]:
    vals = f_values(omega, series_map)
    return [-(vals[i]) + vals[i - 1] for i in range(j, x + 1)]


def gamma(omega: float, n1: int, series_map: dict) -> float:
    vals = f_values(omega, series_map)
    return vals[n1] + vals[n1 - 2] - 2 * vals[n1 - 1]


# Impurity helpers (fixed positions)

def _imp_fixed(index: int, omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    kappa = beta(omega, delta, t, eps)
    return np.linalg.inv(_replace_part(kappa, [[index, index]], omega + 1j * delta - eps1))


def imp(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    kappa = beta(omega, delta, t, eps)
    return np.linalg.inv(_replace_part(kappa, [[14, 14]], omega + 1j * delta + 0.0))


def _imp_by_token(token: str, omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    if token == 'imp':
        return imp(omega, delta, t, eps, eps1)
    idx = int(token.replace('imp', ''))
    return _imp_fixed(idx, omega, delta, t, eps, eps1)


LISTA_DELTAE = ['imp10', 'imp2', 'imp3', 'imp13', 'imp5', 'imp7', 'imp1', 'imp14', 'imp11', 'imp1', 'imp2', 'imp8', 'imp7', 'imp12', 'imp8', 'imp5', 'imp1', 'imp5', 'imp7', 'imp8', 'imp6', 'imp10', 'imp12', 'imp3', 'imp4', 'imp9', 'imp14', 'imp12', 'imp13', 'imp1', 'imp6', 'imp1', 'imp11', 'imp9', 'imp10', 'imp14', 'imp10', 'imp3', 'imp11', 'imp1', 'imp3', 'imp12', 'imp9', 'imp9', 'imp4', 'imp10', 'imp14', 'imp7', 'imp3', 'imp14', 'imp3', 'imp6', 'imp2', 'imp3', 'imp5', 'imp8', 'imp12', 'imp12', 'imp5', 'imp10', 'imp5', 'imp1', 'imp10', 'imp2', 'imp5', 'imp14', 'imp2', 'imp3', 'imp11', 'imp13', 'imp6', 'imp10', 'imp2', 'imp3', 'imp2', 'imp10', 'imp10', 'imp12', 'imp9', 'imp3', 'imp1', 'imp7', 'imp1', 'imp12', 'imp8', 'imp3', 'imp5', 'imp13', 'imp2', 'imp14', 'imp14', 'imp1', 'imp12', 'imp10', 'imp13', 'imp12', 'imp9', 'imp4', 'imp5', 'imp8']


def deltae(eps1: float, rng=None) -> List[List[float]]:
    rng = rng or np.random.default_rng()
    tin = T1(1.0)
    tmat = TT(1.0)
    out = []

    tokens = LISTA_DELTAE.copy()
    rng.shuffle(tokens)

    for omega in np.arange(0.0, 4.0 + 1e-12, 0.01):
        lista = [_imp_by_token(tok, omega, 0.0001, 1.0, 0.0, eps1) for tok in tokens]
        j = SL(omega, 0.0001, 1.0, 0.0)
        for item in lista:
            j = np.linalg.inv(_identity() - item @ _conj_transpose(tin) @ j @ tin) @ item
        sl1 = j

        il1 = np.linalg.inv(_identity() - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0) @ tmat) @ sl1
        ir1 = np.linalg.inv(_identity() - SR(omega, 0.0001, 1.0, 0.0) @ tmat @ sl1 @ tmat) @ SR(
            omega, 0.0001, 1.0, 0.0
        )

        gdd1 = il1 - _conj_transpose(il1)
        grr1 = ir1 - _conj_transpose(ir1)
        gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0) @ tmat @ il1
        gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)
        val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))
        out.append([float(omega), float(val)])

    return out
