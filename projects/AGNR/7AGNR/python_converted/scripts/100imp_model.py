import functools
from typing import List, Optional

import numpy as np


def _identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=complex)


def _conj_transpose(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def _replace_part(arr: np.ndarray, positions: List[List[int]], value: complex) -> np.ndarray:
    out = np.array(arr, copy=True)
    nrows, ncols = out.shape
    for i, j in positions:
        if 1 <= i <= nrows and 1 <= j <= ncols:
            out[i - 1, j - 1] = value
    return out


@functools.lru_cache(maxsize=None)
def beta(omega: float, delta: float, t: float, eps: float, m: int) -> np.ndarray:
    size = 2 * m
    base = (omega + 1j * delta - eps) * _identity(size)
    positions = []
    positions.extend([[i, i - 1] for i in range(2, size + 1)])
    positions.extend([[i, i + 1] for i in range(1, size + 1)])
    positions.extend([[2 * n - 1, 2 * m - 2 * n + 2] for n in range(1, (m + 1) // 2 + 1)])
    positions.extend([[2 * m - 2 * n + 2, 2 * n - 1] for n in range(1, (m - 1) // 2 + 1)])
    return _replace_part(base, positions, -t)


@functools.lru_cache(maxsize=None)
def T1(t: float, m: int) -> np.ndarray:
    size = 2 * m
    base = np.zeros((size, size), dtype=complex)
    positions = [[2 * n, 2 * m - 2 * n + 1] for n in range(1, (m - 1) // 2 + 1)]
    return _replace_part(base, positions, t)


@functools.lru_cache(maxsize=None)
def rho(t: float, m: int) -> np.ndarray:
    size = 2 * m
    base = np.zeros((size, size), dtype=complex)
    positions = [[2 * n, 2 * n] for n in range(1, (m - 1) // 2 + 1)]
    return _replace_part(base, positions, t)


@functools.lru_cache(maxsize=None)
def LEFT(omega: float, delta: float, t: float, eps: float, m: int) -> np.ndarray:
    j = np.linalg.inv(beta(omega, delta, t, eps, m))
    b = np.linalg.inv(beta(omega, delta, t, eps, m))
    t1 = T1(1.0, m)
    ident = _identity(2 * m)
    for _ in range(8000):
        j = np.linalg.inv(ident - b @ _conj_transpose(t1) @ j @ t1) @ b
    return j


def g(omega: float, delta: float, t: float, eps: float, m: int) -> np.ndarray:
    return np.linalg.inv(beta(omega, delta, t, eps, m))


@functools.lru_cache(maxsize=None)
def SR(omega: float, delta: float, t: float, eps: float, m: int) -> np.ndarray:
    gg = g(omega, delta, t, eps, m)
    t1 = T1(t, m)
    ident = _identity(2 * m)
    return np.linalg.inv(ident - gg @ _conj_transpose(t1) @ LEFT(omega, delta, t, eps, m) @ t1) @ gg


@functools.lru_cache(maxsize=None)
def SL(omega: float, delta: float, t: float, eps: float, m: int) -> np.ndarray:
    return SR(omega, delta, t, eps, m)


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def tr(omega: float, delta: float, t: float, eps: float, eps1: float, m: int, rng=None) -> float:
    rng = rng or np.random.default_rng()
    tmat = T1(1.0, m)

    j = SR(omega, delta, t, eps, m)
    ident = _identity(2 * m)
    for _ in range(4):
        mu1 = _randint(rng, 1, 2 * m)
        kappa = beta(omega, delta, t, eps, m)
        c = np.linalg.inv(_replace_part(kappa, [[mu1, mu1]], omega + 1j * delta - eps1))
        j = np.linalg.inv(ident - c @ _conj_transpose(tmat) @ j @ tmat) @ c
    sl1 = j

    rho1 = rho(1.0, m)
    il1 = np.linalg.inv(ident - sl1 @ rho1 @ SR(omega, delta, t, eps, m) @ rho1) @ sl1
    ir1 = np.linalg.inv(ident - SR(omega, delta, 1.0, 0.0, m) @ rho1 @ sl1 @ rho1) @ SR(
        omega, delta, 1.0, 0.0, m
    )

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, delta, 1.0, 0.0, m) @ rho1 @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)

    val = np.abs(np.trace(gdd1 @ rho1 @ grr1 @ rho1 - rho1 @ gnon1 @ rho1 @ gnon1))
    return float(val)


# Impurity helpers used in set/tr14

def imp1(omega: float, delta: float, t: float, eps: float, eps1: float, m: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    kappa = beta(0.0, 0.0, 0.0, 0.0, m)
    i = _randint(rng, 1, 14)
    diag = _replace_part(kappa, [[i, i]], eps1)
    return np.linalg.inv(beta(omega, delta, t, eps, m) - diag)


def imp2(omega: float, delta: float, t: float, eps: float, eps1: float, m: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    kappa = beta(0.0, 0.0, 0.0, 0.0, m)
    i = _randint(rng, 1, 7)
    j = _randint(rng, 8, 14)
    diag = _replace_part(kappa, [[i, i], [j, j]], eps1)
    return np.linalg.inv(beta(omega, delta, t, eps, m) - diag)


def imp(omega: float, delta: float, t: float, eps: float, eps1: float, m: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    kappa = beta(omega, delta, t, eps, m)
    mu = _randint(rng, 1, 14)
    return np.linalg.inv(_replace_part(kappa, [[mu, mu]], omega + 1j * delta - eps1))


def set_imp(omega: float, delta: float, t: float, eps: float, eps1: float, m: int, x: int, y: int, z: int, rng=None):
    rng = rng or np.random.default_rng()
    items = []
    for _ in range(x):
        items.append(imp1(omega, delta, t, eps, eps1, m, rng=rng))
    for _ in range(y):
        items.append(imp2(omega, delta, t, eps, eps1, m, rng=rng))
    for _ in range(z):
        items.append(imp(omega, delta, t, eps, 0.0, m, rng=rng))
    rng.shuffle(items)
    return items


def tr14(omega: float, delta: float, t: float, eps: float, eps1: float, m: int, numimp: int, rng=None) -> float:
    rng = rng or np.random.default_rng()
    x = T1(t, m)
    tmat = rho(t, m)
    list_items = set_imp(omega, delta, t, eps, eps1, m, numimp, 0, 100 - numimp, rng=rng)

    ident = _identity(14)
    j = SL(omega, delta, t, eps, m)
    for item in list_items:
        j = np.linalg.inv(ident - item @ _conj_transpose(x) @ j @ x) @ item
    sl1 = j

    il1 = np.linalg.inv(ident - sl1 @ tmat @ SR(omega, delta, t, eps, m) @ tmat) @ sl1
    ir1 = np.linalg.inv(ident - SR(omega, delta, t, eps, m) @ tmat @ sl1 @ tmat) @ SR(omega, delta, t, eps, m)

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, delta, t, eps, m) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)

    val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))
    return float(3.0 if val > 3.0 else val)


def _imp_fixed(index: int, omega: float, delta: float, t: float, eps: float, eps1: float, m: int) -> np.ndarray:
    kappa = beta(omega, delta, t, eps, m)
    return np.linalg.inv(_replace_part(kappa, [[index, index]], omega + 1j * delta - eps1))


def tr1(omega: float, delta: float, t: float, eps: float, eps1: float, m: int, rng=None) -> float:
    rng = rng or np.random.default_rng()
    x = T1(t, m)
    tmat = rho(t, m)

    indices = [
        5, 14, 5, 7, 14, 14, 3, 3, 6, 8, 9, 1, 7, 6, 7, 5, 3, 5, 11, 10,
        2, 9, 8, 7, 7, 8, 14, 2, 3, 6, 7, 12, 1, 6, 7, 8, 8, 13, 4, 10,
        14, 3, 6, 3, 4, 12, 3, 2, 14, 8, 5, 4, 13, 13, 5, 1, 9, 9, 1, 10,
        9, 6, 2, 9, 2, 12, 9, 14, 4, 3, 6, 4, 5, 12, 8, 2, 4, 9, 11, 8,
        11, 8, 4, 4, 1, 12, 8, 8, 1, 1, 1, 13, 1, 10, 5, 7, 1, 3, 1, 8,
    ]
    rng.shuffle(indices)
    list_items = [_imp_fixed(idx, omega, delta, t, eps, eps1, m) for idx in indices]

    ident = _identity(14)
    j = SL(omega, delta, t, eps, m)
    for item in list_items:
        j = np.linalg.inv(ident - item @ _conj_transpose(x) @ j @ x) @ item
    sl1 = j

    il1 = np.linalg.inv(ident - sl1 @ tmat @ SR(omega, delta, t, eps, m) @ tmat) @ sl1
    ir1 = np.linalg.inv(ident - SR(omega, delta, t, eps, m) @ tmat @ sl1 @ tmat) @ SR(omega, delta, t, eps, m)

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, delta, t, eps, m) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)

    val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))
    return float(3.0 if val > 3.0 else val)


def _interp_integral(xs: np.ndarray, ys: np.ndarray, a: float, b: float, num: int = 1000) -> float:
    if a == b:
        return 0.0
    lo, hi = (a, b) if a < b else (b, a)
    grid = np.linspace(lo, hi, num=num)
    vals = np.interp(grid, xs, ys)
    area = np.trapz(vals, grid)
    return area if a < b else -area


def upsilon(num: int, x: float, y: float, base_dir: Optional[str] = None, rng=None):
    rng = rng or np.random.default_rng()
    mu1 = _randint(rng, 1, 1000)
    m5 = np.array([[omega, tr1(omega, 0.0001, 1.0, 0.0, 0.5, 7)] for omega in np.arange(0, 4.0 + 1e-9, 0.01)])

    def _load_csv(name: str) -> np.ndarray:
        path = name
        if base_dir is not None:
            path = f"{base_dir}/{name}"
        return np.loadtxt(path, delimiter=',')

    def _rho(ref_name: str) -> float:
        ref = _load_csv(ref_name)
        xs = m5[:400, 0]
        ys = (m5[:400, 1] - ref[:400, 1]) ** 2
        return _interp_integral(xs, ys, y, x) / ((x - y) * 100.0)

    rhos = {
        2: _rho("14imp100.csv"),
        3: _rho("28imp100.csv"),
        4: _rho("42imp100.csv"),
        5: _rho("56imp100.csv"),
        6: _rho("70imp100.csv"),
        7: _rho("84imp100.csv"),
        8: _rho("100imp100.csv"),
        9: _rho("114imp100.csv"),
        10: _rho("126imp100.csv"),
        11: _rho("140imp100.csv"),
    }
    best_k = min(rhos, key=rhos.get)
    return best_k, rhos[best_k], m5
