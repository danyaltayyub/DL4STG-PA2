"""Mirror / rotation and the 8-element group G (act on the last two dims)."""
import torch


def mirror(x):
    return torch.flip(x, dims=(-1,))


def rot(x, a=1):
    return torch.rot90(x, k=a, dims=(-2, -1))


def apply(g, x):
    """g = 4*m + a: rotate by a, then mirror if m = 1."""
    m, a = divmod(g, 4)
    x = rot(x, a)
    return mirror(x) if m else x


def inverse(g, x):
    m, a = divmod(g, 4)
    return apply(g, x) if m else rot(x, -a)  # mirror(rot(a)) is an involution
