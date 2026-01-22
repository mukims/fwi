import functools
from typing import Dict, List, Optional, Sequence

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


# 1x1 lead helper
@functools.lru_cache(maxsize=None)
def beta1(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    return (omega + 1j * delta - eps) * _identity(1)


@functools.lru_cache(maxsize=None)
def T2(t: float) -> np.ndarray:
    return t * _identity(1)


@functools.lru_cache(maxsize=None)
def LEFT1(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    j = np.linalg.inv(beta1(omega, 0.0001, 1.0, 0.0))
    b = np.linalg.inv(beta1(omega, 0.0001, 1.0, 0.0))
    tmat = T2(1.0)
    ident = _identity(1)
    for _ in range(10000):
        j = np.linalg.pinv(ident - b @ tmat @ j @ tmat) @ b
    return j


# 14x14 device helpers
@functools.lru_cache(maxsize=None)
def beta(omega: float, delta: float, t: float, eps: float) -> np.ndarray:
    base = (omega + 1j * delta - eps) * _identity(SIZE)
    positions = []
    positions.extend([[i, i - 1] for i in range(2, SIZE + 1)])
    positions.extend([[i, i + 1] for i in range(1, SIZE + 1)])
    positions.extend([[2 * n - 1, SIZE - 2 * n + 2] for n in range(1, 5)])
    positions.extend([[SIZE - 2 * n + 2, 2 * n - 1] for n in range(1, 4)])
    # Note: notebook uses +t (not -t)
    return _replace_part(base, positions, t)


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
    return np.linalg.inv(ident - sl @ TT(t) @ sl @ TT(t)) @ sl


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


def pris(leads: Sequence[np.ndarray], start=0.0, stop=3.0, step=0.01) -> List[List[float]]:
    return [[omega, tr(omega, 0.0001, 1.0, 0.0, leads)] for omega in np.arange(start, stop + 1e-12, step)]


# Lattice utilities

def f1(y: int, start: int, stop: int) -> List[List[int]]:
    return [[x, y] for x in range(start, stop + 1)]


def s(xstart: int, xstop: int, ystart: int, ystop: int) -> List[List[int]]:
    data = []
    for y in range(ystart, ystop + 1):
        data.extend(f1(y, xstart, xstop))
    return data


@functools.lru_cache(maxsize=None)
def dist(total: int) -> List[np.ndarray]:
    base = s(1, 100, 1, 14)
    rng = np.random.default_rng()
    out = []
    for _ in range(10000):
        sample = rng.choice(len(base), size=total, replace=False)
        arr = np.array([base[i] for i in sample])
        arr = arr[np.argsort(arr[:, 0])]
        out.append(arr)
    return out


def unitcelltest2(omega: float, total: int, number: int, unitcell: int) -> np.ndarray:
    j = beta(omega, 0.0001, 1.0, 0.0)
    for loc in range(total):
        current = dist(total)[number][loc]
        if current[0] == unitcell:
            j = _replace_part(j, [[int(current[1]), int(current[1])]], omega + 1j * 0.0001 - 0.5)
    return j


def inputspectra(omega: float, total: int, number: int, leads: Sequence[np.ndarray]) -> float:
    tin = T1(1.0)
    th = TT(1.0)
    j1 = SL(omega, 0.0001, 1.0, 0.0, leads)
    for unitcell in range(1, 101):
        inv_uc = np.linalg.inv(unitcelltest2(omega, total, number, unitcell))
        j1 = np.linalg.inv(_identity() - inv_uc @ _conj_transpose(tin) @ j1 @ tin) @ inv_uc
    sl1 = j1
    il1 = np.linalg.inv(_identity() - sl1 @ th @ SR(omega, 0.0001, 1.0, 0.0, leads) @ th) @ sl1
    ir1 = np.linalg.inv(_identity() - SR(omega, 0.0001, 1.0, 0.0, leads) @ th @ sl1 @ th) @ SR(
        omega, 0.0001, 1.0, 0.0, leads
    )

    gdd1 = il1 - _conj_transpose(il1)
    grr1 = ir1 - _conj_transpose(ir1)
    gnonlocal1 = SR(omega, 0.0001, 1.0, 0.0, leads) @ th @ il1
    gnon1 = gnonlocal1 - _conj_transpose(gnonlocal1)
    return float(np.abs(np.trace(gdd1 @ th @ grr1 @ th - th @ gnon1 @ th @ gnon1)))


def input_configs(imp: int, leads: Sequence[np.ndarray], n_configs: int = 50) -> List[List[List[float]]]:
    out = []
    for n in range(1, n_configs + 1):
        spectrum = [[omega, inputspectra(omega, imp, n, leads)] for omega in np.arange(0.0, 3.0 + 1e-12, 0.01)]
        out.append(spectrum)
    return out


def transmission(x: int, base_dir: str) -> np.ndarray:
    return np.loadtxt(f"{base_dir}/test{x}.dat")


def misfit(imp: int, x: float, y: float, n: int, input_data: List[List[List[float]]], base_dir: str) -> List[List[float]]:
    m5 = np.array(input_data[n])
    out = []
    for b in range(1, 111, 2):
        tb = transmission(b, base_dir)
        diff = np.abs(m5[:, 1] - tb[: len(m5), 1])
        xs = np.arange(0.0, 3.0 + 1e-12, 0.01)
        val = np.trapz(np.interp(np.linspace(y, x, 1000), xs, diff), np.linspace(y, x, 1000)) / ((x - y) * 100.0)
        out.append([b, float(val)])
    return out


# Configurational averaging utilities

def F(n: int) -> List[int]:
    return [n, n]


def impurity(omega: float, delta: float, t: float, eps: float, eps1: float, n: int) -> np.ndarray:
    b = beta(omega, delta, t, eps)
    rng = np.random.default_rng()
    positions = [F(int(rng.integers(1, 15))) for _ in range(n)]
    return _replace_part(b, positions, omega + 1j * delta - eps1)


def list_ud(up: int, down: int) -> np.ndarray:
    rng = np.random.default_rng()
    vals = [0] * (100 - down - up)
    vals += [int(rng.integers(1, 15)) for _ in range(up)]
    vals += [int(rng.integers(21, 41)) for _ in range(down)]
    rng.shuffle(vals)
    return np.column_stack([np.arange(1, 101), vals])


def dist_ud(up: int, down: int) -> List[np.ndarray]:
    return [list_ud(up, down) for _ in range(5000)]


def CAvg(omega: float, delta: float, t: float, eps: float, eps1: float, up: int, down: int, number: int, leads: Sequence[np.ndarray]) -> float:
    tin = T1(1.0)
    tmat = TT(1.0)
    dist_list = dist_ud(up, down)[number]

    j = SL(omega, 0.0001, 1.0, 0.0, leads)
    for loc in range(100):
        n = int(dist_list[loc, 1])
        kappa = beta(omega, 0.0001, 1.0, 0.0)
        inv_uc = np.linalg.inv(_replace_part(kappa, [[n, n]], omega + 1j * 0.0001 - eps1))
        j = np.linalg.inv(_identity() - inv_uc @ tin @ j @ tin) @ inv_uc
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


# Predict/NeuralNetwork logic in Mathematica is not reproduced here.
# Add a custom model if needed for data(num, omega).
