import re
from pathlib import Path
from IPython.core.debugger import set_trace
import pickle, gzip, math, torch, matplotlib as mpl
from typing import *
from torch.nn import init

_camel_re1 = re.compile("(.)([A-Z][a-z]+)")
_camel_re2 = re.compile("([a-z0-9])([A-Z])")


def test(a, b, cmp, cname=None):
    if cname is None:
        cname = cmp.__name__
    assert cmp(a, b), f"{cname}:\n{a}\n{b}"


def test_eq(a, b):
    test(a, b, operator.eq, "==")


def near(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)


def test_near(a, b):
    test(a, b, near)


def normalize(x, m, s):
    return (x - m) / s


def test_near_zero(a, tol=1e-3):
    assert a.abs() < tol, f"Near zero: {a}"


def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


def camel2snake(name):
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


def listify(o):
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]
