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
    j = np.linalg.inv(beta(omega, 0.001, 1.0, 0.0))
    b = np.linalg.inv(beta(omega, 0.001, 1.0, 0.0))
    t1 = T1(1.0)
    ident = _identity()
    for _ in range(3000):
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


# Impurity helpers (random positions)

def imp1(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mu = int(rng.integers(1, 15))
    kappa = beta(omega, 0.0001, t, 0.0)
    return np.linalg.inv(_replace_part(kappa, [[mu, mu]], omega + 1j * 0.0001 - eps1))


def imp2(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mu1 = int(rng.integers(1, 8))
    mu2 = int(rng.integers(8, 15))
    kappa = beta(omega, 0.0001, t, 0.0)
    return np.linalg.inv(_replace_part(kappa, [[mu2, mu2], [mu1, mu1]], omega + 1j * 0.0001 - eps1))


def imp(omega: float, delta: float, t: float, eps: float, eps1: float) -> np.ndarray:
    kappa = beta(omega, 0.0001, t, 0.0)
    return np.linalg.inv(_replace_part(kappa, [[14, 14]], omega + 1j * 0.0001 + 0.0))


TRA_TOKENS = ['imp', 'imp2', 'imp', 'imp1', 'imp', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp2', 'imp', 'imp2', 'imp2', 'imp1', 'imp1', 'imp', 'imp1', 'imp1', 'imp1', 'imp', 'imp2', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp1', 'imp', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp', 'imp', 'imp1', 'imp2', 'imp2', 'imp2', 'imp', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp', 'imp2', 'imp2', 'imp', 'imp', 'imp1', 'imp', 'imp2', 'imp1', 'imp1', 'imp1', 'imp', 'imp', 'imp1', 'imp1', 'imp', 'imp', 'imp', 'imp2', 'imp', 'imp1', 'imp', 'imp2', 'imp2', 'imp1', 'imp', 'imp1', 'imp', 'imp1', 'imp', 'imp2', 'imp', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp2', 'imp2', 'imp1', 'imp1', 'imp1', 'imp', 'imp', 'imp', 'imp2', 'imp2', 'imp2', 'imp1', 'imp2', 'imp1']
MISFIT_TOKENS = ['imp9', 'imp2', 'imp', 'imp14', 'imp', 'imp', 'imp7', 'imp1', 'imp12', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp12', 'imp', 'imp4', 'imp11', 'imp11', 'imp', 'imp10', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp7', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp12', 'imp7', 'imp14', 'imp', 'imp', 'imp2', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp12', 'imp10', 'imp6', 'imp4', 'imp', 'imp', 'imp10', 'imp14', 'imp10', 'imp', 'imp', 'imp', 'imp', 'imp4', 'imp8', 'imp10', 'imp14', 'imp1', 'imp14', 'imp', 'imp', 'imp', 'imp', 'imp14', 'imp2', 'imp', 'imp', 'imp14', 'imp', 'imp2', 'imp9', 'imp', 'imp', 'imp', 'imp10', 'imp', 'imp5', 'imp', 'imp8', 'imp7', 'imp', 'imp', 'imp3', 'imp', 'imp11', 'imp10', 'imp', 'imp', 'imp11']


def _build_list(tokens, omega, delta, t, eps, eps1, rng):
    items = []
    for tok in tokens:
        if tok == 'imp':
            items.append(imp(omega, delta, t, eps, eps1))
        elif tok == 'imp1':
            items.append(imp1(omega, delta, t, eps, eps1, rng=rng))
        elif tok == 'imp2':
            items.append(imp2(omega, delta, t, eps, eps1, rng=rng))
        else:
            # impX fixed
            idx = int(tok.replace('imp', ''))
            kappa = beta(omega, 0.0001, t, 0.0)
            items.append(np.linalg.inv(_replace_part(kappa, [[idx, idx]], omega + 1j * 0.0001 - eps1)))
    rng.shuffle(items)
    return items


def tra(omega: float, delta: float, t: float, eps: float, eps1: float, rng=None) -> float:
    rng = rng or np.random.default_rng()
    tin = T1(1.0)
    tmat = TT(1.0)
    list_items = _build_list(TRA_TOKENS, omega, delta, t, eps, eps1, rng)

    j = SL(omega, 0.0001, 1.0, 0.0)
    for item in list_items:
        j = np.linalg.inv(_identity() - item @ _conj_transpose(tin) @ j @ tin) @ item
    sl1 = j

    il1 = np.linalg.inv(_identity() - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0) @ tmat) @ sl1
    ir1 = np.linalg.inv(_identity() - SR(omega, 0.0001, 1.0, 0.0) @ tmat @ sl1 @ tmat) @ SR(omega, 0.0001, 1.0, 0.0)

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)
    val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))
    return float(3.0 if val > 3.0 else val)


def misfit(eps1: float, x: float, y: float, base_dir: Optional[str] = None, rng=None):
    rng = rng or np.random.default_rng()
    tin = T1(1.0)
    tmat = TT(1.0)

    def tra_local(omega: float) -> float:
        lista = _build_list(MISFIT_TOKENS, omega, 0.0001, 1.0, 0.0, eps1, rng)
        j = SL(omega, 0.0001, 1.0, 0.0)
        for item in lista:
            j = np.linalg.inv(_identity() - item @ _conj_transpose(tin) @ j @ tin) @ item
        sl1 = j

        il1 = np.linalg.inv(_identity() - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0) @ tmat) @ sl1
        ir1 = np.linalg.inv(_identity() - SR(omega, 0.0001, 1.0, 0.0) @ tmat @ sl1 @ tmat) @ SR(omega, 0.0001, 1.0, 0.0)

        gdd1 = il1 - _conj_transpose(il1)
        grr1 = ir1 - _conj_transpose(ir1)
        gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0) @ tmat @ il1
        gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)
        return float(np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1)))

    m5 = np.array([[omega, tra_local(omega)] for omega in np.arange(0.0, 4.0 + 1e-12, 0.01)])

    if base_dir is None:
        base_dir = '/home/shardulmukim/PhD/fwi/7AGNR/100unitcells'

    def _load(name: str) -> np.ndarray:
        return np.loadtxt(f"{base_dir}/{name}", delimiter=',')

    def _rho(name: str) -> float:
        ref = _load(name)
        xs = m5[:400, 0]
        ys = (m5[:400, 1] - ref[:400, 1]) ** 2
        grid = np.linspace(y, x, 1000)
        vals = np.interp(grid, xs, ys)
        return float(np.trapz(vals, grid) / ((x - y) * 100.0))

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
