import functools
from typing import Dict, List, Optional

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


# Simple aliases used in the notebook

def pri(omega: float) -> float:
    return tr(omega, 0.0001, 1.0, 0.0)


def tr_table(start=0.0, stop=3.0, step=0.01) -> List[List[float]]:
    return [[omega, tr(omega, 0.0001, 1.0, 0.0)] for omega in np.arange(start, stop + 1e-12, step)]


# Data interpolation helpers (from 50unitcells files in the notebook)

def _load_series(path: str) -> np.ndarray:
    return np.loadtxt(path)


def _interp_series(series: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.interp(xs, series[:, 0], series[:, 1])


def build_k_from_file(path: str, start=0.0, stop=3.0, step=0.001) -> List[List[float]]:
    xs = np.arange(start, stop + 1e-12, step)
    series = _load_series(path)
    ys = _interp_series(series, xs)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def sample_k(table: List[List[float]], omega: float, step=0.001) -> float:
    idx = int(round(omega / step))
    idx = max(0, min(idx, len(table) - 1))
    return float(table[idx][1])


def build_k00(step=0.001) -> List[List[float]]:
    xs = np.arange(0.0, 3.0 + 1e-12, step)
    return [[float(x), float(tr(x, 0.0001, 1.0, 0.0))] for x in xs]


def f_from_tables(omega: float, tables: Dict[str, List[List[float]]], step=0.001) -> List[float]:
    return [
        sample_k(tables['k00'], omega, step),
        sample_k(tables['k14'], omega, step),
        sample_k(tables['k28'], omega, step),
        sample_k(tables['k42'], omega, step),
        sample_k(tables['k56'], omega, step),
        sample_k(tables['k70'], omega, step),
        sample_k(tables['k84'], omega, step),
        sample_k(tables['k100'], omega, step),
    ]


def beta_beta(omega: float, tables: Dict[str, List[List[float]]]) -> List[List[float]]:
    vals = f_from_tables(omega, tables)
    out = []
    for j in range(1, 5):
        row = []
        for i in range(1, j):
            row.append((vals[j - 1] - vals[i - 1]) / (j - i))
        out.append(row)
    return out


def psi(omega: float, tables: Dict[str, List[List[float]]]) -> float:
    flat = [v for row in beta_beta(omega, tables) for v in row]
    return -float(np.mean(flat)) if flat else 0.0


# Impurity helpers (fixed positions, no randomness)

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


LISTA_DELTAE = ['imp', 'imp9', 'imp4', 'imp3', 'imp6', 'imp2', 'imp11', 'imp13', 'imp13', 'imp', 'imp', 'imp10', 'imp5', 'imp5', 'imp', 'imp5', 'imp14', 'imp14', 'imp6', 'imp', 'imp1', 'imp10', 'imp5', 'imp6', 'imp10', 'imp12', 'imp', 'imp', 'imp2', 'imp14', 'imp', 'imp2', 'imp', 'imp1', 'imp2', 'imp9', 'imp7', 'imp8', 'imp3', 'imp', 'imp', 'imp', 'imp10', 'imp6', 'imp14', 'imp', 'imp6', 'imp1', 'imp', 'imp5', 'imp3', 'imp', 'imp', 'imp13', 'imp', 'imp8', 'imp2', 'imp6', 'imp10', 'imp4', 'imp', 'imp', 'imp4', 'imp13', 'imp', 'imp13', 'imp', 'imp2', 'imp11', 'imp7', 'imp1', 'imp8', 'imp7', 'imp', 'imp', 'imp5', 'imp9', 'imp', 'imp7', 'imp3', 'imp13', 'imp', 'imp', 'imp5', 'imp13', 'imp9', 'imp4', 'imp', 'imp10', 'imp12', 'imp7', 'imp', 'imp6', 'imp', 'imp14', 'imp5', 'imp1', 'imp13', 'imp1', 'imp']
LISTA_MISFIT = ['imp', 'imp3', 'imp11', 'imp', 'imp3', 'imp', 'imp', 'imp5', 'imp', 'imp', 'imp8', 'imp', 'imp', 'imp', 'imp6', 'imp', 'imp8', 'imp', 'imp', 'imp', 'imp1', 'imp', 'imp', 'imp', 'imp', 'imp2', 'imp6', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp8', 'imp', 'imp', 'imp', 'imp', 'imp4', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp6', 'imp', 'imp', 'imp7', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp4', 'imp6', 'imp14', 'imp', 'imp', 'imp', 'imp', 'imp6', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp14', 'imp9', 'imp', 'imp', 'imp', 'imp4', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp8', 'imp13', 'imp', 'imp', 'imp', 'imp', 'imp2', 'imp', 'imp9', 'imp', 'imp13', 'imp', 'imp10', 'imp', 'imp', 'imp12', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp']


def _build_list(tokens: List[str], omega: float, delta: float, t: float, eps: float, eps1: float) -> List[np.ndarray]:
    return [_imp_by_token(tok, omega, delta, t, eps, eps1) for tok in tokens]


def _integral_interp(xs: np.ndarray, ys: np.ndarray, a: float, b: float, num=1000) -> float:
    if a == b:
        return 0.0
    lo, hi = (a, b) if a < b else (b, a)
    grid = np.linspace(lo, hi, num=num)
    vals = np.interp(grid, xs, ys)
    area = np.trapz(vals, grid)
    return area if a < b else -area


def deltae(eps1: float, omega: float) -> float:
    tin = T1(1.0)
    tmat = TT(1.0)
    lista = _build_list(LISTA_DELTAE, omega, 0.0001, 1.0, 0.0, eps1)

    ident = _identity()
    j = SL(omega, 0.0001, 1.0, 0.0)
    for item in lista[:4]:
        j = np.linalg.inv(ident - item @ _conj_transpose(tin) @ j @ tin) @ item
    sl1 = j

    il1 = np.linalg.inv(ident - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0) @ tmat) @ sl1
    ir1 = np.linalg.inv(ident - SR(omega, 0.0001, 1.0, 0.0) @ tmat @ sl1 @ tmat) @ SR(omega, 0.0001, 1.0, 0.0)

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)

    return float(np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1)))


def misfit(eps1: float, x: float, y: float, base_dir: Optional[str] = None):
    tin = T1(1.0)
    tmat = TT(1.0)

    def tra(omega: float) -> float:
        lista = _build_list(LISTA_MISFIT, omega, 0.0001, 1.0, 0.0, eps1)
        ident = _identity()
        j = SL(omega, 0.0001, 1.0, 0.0)
        for item in lista:
            j = np.linalg.inv(ident - item @ _conj_transpose(tin) @ j @ tin) @ item
        sl1 = j

        il1 = np.linalg.inv(ident - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0) @ tmat) @ sl1
        ir1 = np.linalg.inv(ident - SR(omega, 0.0001, 1.0, 0.0) @ tmat @ sl1 @ tmat) @ SR(omega, 0.0001, 1.0, 0.0)

        gdd1 = il1 - _conj_transpose(il1)
        grr1 = ir1 - _conj_transpose(ir1)
        gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0) @ tmat @ il1
        gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)
        return float(np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1)))

    m5 = np.array([[omega, tra(omega)] for omega in np.arange(0.0, 3.0 + 1e-12, 0.01)])

    def _load(path: str) -> np.ndarray:
        return np.loadtxt(path, delimiter=',')

    if base_dir is None:
        base_dir = '/home/shardulmukim/PhD/fwi/7AGNR/100unitcells'

    def _rho(name: str) -> float:
        ref = _load(base_dir + '/' + name)
        xs = m5[:400, 0]
        ys = (m5[:400, 1] - ref[:400, 1]) ** 2
        return _integral_interp(xs, ys, y, x) / (x * 100.0)

    rhos = {
        1: _rho('14imp100.csv'),
        2: _rho('28imp100.csv'),
        3: _rho('42imp100.csv'),
        4: _rho('56imp100.csv'),
        5: _rho('70imp100.csv'),
        6: _rho('84imp100.csv'),
        7: _rho('100imp100.csv'),
        8: _rho('114imp100.csv'),
        9: _rho('126imp100.csv'),
        10: _rho('140imp100.csv'),
    }
    best_k = min(rhos, key=rhos.get)
    return rhos[best_k], best_k, rhos
