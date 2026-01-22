import functools
from typing import List, Sequence

import numpy as np

SIZE = 14


def _identity(n: int = SIZE) -> np.ndarray:
    return np.eye(n, dtype=complex)


def _conj_transpose(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def _replace_part(arr: np.ndarray, positions: List[List[int]], value: complex) -> np.ndarray:
    out = np.array(arr, copy=True)
    for i, j in positions:
        out[i - 1, j - 1] = value
    return out


@functools.lru_cache(maxsize=None)
def beta(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    base = (omega + 1j * delta - eps) * _identity(SIZE)
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


# Leads (from data files)

def load_leads(base_dir: str, omega_start=0.0, omega_stop=3.0, step=0.01) -> List[np.ndarray]:
    omegas = np.arange(omega_start, omega_stop + 1e-12, step)
    leads = []
    for omega in omegas:
        path = f"{base_dir}/lead_{omega:.2f}.dat"
        leads.append(np.loadtxt(path))
    return leads


def left_from_leads(omega: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    idx = int(round(omega * 100))
    return leads[idx]


@functools.lru_cache(maxsize=None)
def g(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    return np.linalg.inv(beta(omega, delta, t, eps))


def SR(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    gg = g(omega, delta, t, eps)
    t1 = T1(t)
    ident = _identity()
    left = left_from_leads(omega, leads)
    return np.linalg.inv(ident - gg @ _conj_transpose(t1) @ left @ t1) @ gg


def SL(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    gg = g(omega, delta, t, eps)
    t1 = T1(t)
    ident = _identity()
    left = left_from_leads(omega, leads)
    return np.linalg.inv(ident - gg @ _conj_transpose(t1) @ left @ t1) @ gg


def IL(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    ident = _identity()
    sl = SL(omega, delta, t, eps, leads)
    return np.linalg.inv(ident - sl @ TT(t) @ SR(omega, delta, t, eps, leads) @ TT(t)) @ sl


def IR(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    ident = _identity()
    sr = SR(omega, delta, t, eps, leads)
    sl = SL(omega, delta, t, eps, leads)
    return np.linalg.inv(ident - sr @ TT(t) @ sl @ TT(t)) @ sr


def gdd(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    il = IL(omega, delta, t, eps, leads)
    return il - _conj_transpose(il)


def grr(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    ir = IR(omega, delta, t, eps, leads)
    return ir - _conj_transpose(ir)


def Gnonlocal(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    return SR(omega, delta, t, eps, leads) @ TT(t) @ IL(omega, delta, t, eps, leads)


def GNON(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> np.ndarray:
    gnl = Gnonlocal(omega, delta, t, eps, leads)
    return gnl - _conj_transpose(gnl)


def tr(omega: float, delta: float, t: float, eps: float, leads: Sequence[np.ndarray]) -> float:
    val = np.trace(
        gdd(omega, delta, t, eps, leads) @ TT(t) @ grr(omega, delta, t, eps, leads) @ TT(t)
        - TT(t) @ GNON(omega, delta, t, eps, leads) @ TT(t) @ GNON(omega, delta, t, eps, leads)
    )
    return float(np.abs(val))


# inputspectra uses random impurity positions; this translates the function

def inputspectra(omega: float, eps1: float, size: int, leads: Sequence[np.ndarray], rng=None) -> float:
    rng = rng or np.random.default_rng()
    tin = T1(1.0)
    tmat = TT(1.0)

    mus = [int(rng.integers(1, 15)) for _ in range(14)]

    def imp(mu: int) -> np.ndarray:
        kappa = beta(omega, 0.0001, 1.0, 0.0)
        return np.linalg.inv(_replace_part(kappa, [[mu, mu]], omega + 1j * 0.0001 - eps1))

    def imp_zero() -> np.ndarray:
        kappa = beta(omega, 0.0001, 1.0, 0.0)
        return np.linalg.inv(_replace_part(kappa, [[14, 14]], omega + 1j * 0.0001 + 0.0))

    # lista was fixed in the notebook; reproduce by mapping tokens to imp/imp_zero
    tokens = [
        'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp12', 'imp', 'imp5', 'imp13',
        'imp', 'imp', 'imp2', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp10', 'imp2', 'imp14', 'imp',
        'imp1', 'imp', 'imp', 'imp6', 'imp5', 'imp', 'imp7', 'imp1', 'imp10', 'imp12', 'imp8', 'imp',
        'imp', 'imp4', 'imp1', 'imp6', 'imp', 'imp11', 'imp7', 'imp4', 'imp', 'imp', 'imp11', 'imp',
        'imp14', 'imp', 'imp', 'imp10', 'imp12', 'imp', 'imp', 'imp', 'imp', 'imp9', 'imp', 'imp', 'imp7',
        'imp14', 'imp9', 'imp13', 'imp', 'imp', 'imp6', 'imp', 'imp', 'imp', 'imp', 'imp1', 'imp', 'imp3',
        'imp8', 'imp', 'imp', 'imp', 'imp5', 'imp3', 'imp', 'imp', 'imp3', 'imp', 'imp', 'imp2', 'imp',
        'imp9', 'imp', 'imp8', 'imp4', 'imp', 'imp', 'imp', 'imp', 'imp13', 'imp', 'imp11'
    ]

    def build_item(tok: str) -> np.ndarray:
        if tok == 'imp':
            return imp_zero()
        idx = int(tok.replace('imp', ''))
        return imp(mus[idx - 1])

    lista = [build_item(tok) for tok in tokens]

    j = SL(omega, 0.0001, 1.0, 0.0, leads)
    for item in lista[:size]:
        j = np.linalg.inv(_identity() - item @ _conj_transpose(tin) @ j @ tin) @ item
    sl1 = j

    il1 = np.linalg.inv(_identity() - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0, leads) @ tmat) @ sl1
    ir1 = np.linalg.inv(_identity() - SR(omega, 0.0001, 1.0, 0.0, leads) @ tmat @ sl1 @ tmat) @ SR(
        omega, 0.0001, 1.0, 0.0, leads
    )

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0, leads) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)
    val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))

    base = tr(omega, 0.0001, 1.0, 0.0, leads)
    return float(base if val > base else val)


# Configurational averaging with random impurities

def impurity(omega: float, delta: float, t: float, eps: float, eps1: float, n: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    b = beta(omega, delta, t, eps)
    positions = [[int(rng.integers(1, 15))] * 2 for _ in range(n)]
    return _replace_part(b, positions, omega + 1j * delta - eps1)


def CAvg(omega: float, delta: float, t: float, eps: float, eps1: float, num: int, size: int, leads: Sequence[np.ndarray], rng=None) -> float:
    rng = rng or np.random.default_rng()
    tin = T1(1.0)
    tmat = TT(1.0)

    lista = []
    for _ in range(num):
        lista.append(impurity(omega, 0.0001, 1.0, 0.0, eps1, 1, rng=rng))
    for _ in range(size - num):
        lista.append(impurity(omega, 0.0001, 1.0, 0.0, 0.0, 1, rng=rng))
    rng.shuffle(lista)

    j = SL(omega, 0.0001, 1.0, 0.0, leads)
    for item in lista[:size]:
        inv_item = np.linalg.inv(item)
        j = np.linalg.inv(_identity() - inv_item @ _conj_transpose(tin) @ j @ tin) @ inv_item
    sl1 = j

    il1 = np.linalg.inv(_identity() - sl1 @ tmat @ SR(omega, 0.0001, 1.0, 0.0, leads) @ tmat) @ sl1
    ir1 = np.linalg.inv(_identity() - SR(omega, 0.0001, 1.0, 0.0, leads) @ tmat @ sl1 @ tmat) @ SR(
        omega, 0.0001, 1.0, 0.0, leads
    )

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0, leads) @ tmat @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)

    val = np.abs(np.trace(gdd1 @ tmat @ grr1 @ tmat - tmat @ gnon1 @ tmat @ gnon1))
    base = tr(omega, 0.0001, 1.0, 0.0, leads)
    return float(base if val > base else val)


# NeuralNetwork / Predict portions omitted (plotting and training).
