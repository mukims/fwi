import numpy as np

# Mathematica-like helpers (best-effort)

def mma_range(start, end=None, step=1):
    if end is None:
        start, end = 1, start
    step = 1 if step is None else step
    if step == 0:
        raise ValueError("step cannot be 0")
    if step > 0:
        return np.arange(start, end + 1, step)
    return np.arange(start, end - 1, step)


def mma_table(func, *specs):
    def build(specs, args):
        if not specs:
            return func(*args)
        spec = specs[0]
        if len(spec) == 2:
            iterable = spec[1]
        else:
            iterable = mma_range(spec[1], spec[2], spec[3] if len(spec) > 3 else 1)
        return np.array([build(specs[1:], args + [val]) for val in iterable])

    return build(list(specs), [])


def mma_rule(lhs, rhs):
    return (lhs, rhs)


def mma_join(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    return np.concatenate([np.array(a) for a in args])


def mma_replace_part(arr, rules, value=None):
    out = np.array(arr, copy=True)
    # rules can be (positions, value) or list of (pos, val)
    if value is not None:
        positions = rules
        for pos in positions:
            idx = tuple(int(p) - 1 for p in pos)
            out[idx] = value
        return out
    # list of (pos, val)
    for pos, val in rules:
        idx = tuple(int(p) - 1 for p in pos)
        out[idx] = val
    return out


def mma_conjugate_transpose(a):
    return np.conjugate(np.transpose(a))


def mma_array_flatten(blocks):
    return np.block(blocks)


def mma_flatten(a):
    return np.ravel(a)

