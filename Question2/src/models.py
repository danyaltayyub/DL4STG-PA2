"""ConvLSTM forecaster, reference predictors and the symmetrised wrapper."""
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transforms import apply, inverse


@dataclass
class ArchConfig:
    patch: int = 4
    hidden: int = 64
    kernels: tuple = (3, 3)
    dilations: tuple = (1, 1)

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, hidden, k, d):
        super().__init__()
        self.conv = nn.Conv2d(c_in + hidden, 4 * hidden, k, dilation=d, padding=d * (k - 1) // 2)

    def forward(self, u, h, c):
        i, f, o, g = self.conv(torch.cat([u, h], 1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMForecaster(nn.Module):
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch or ArchConfig()
        a, p = self.arch, self.arch.patch
        assert len(a.kernels) == len(a.dilations)
        cins = [p * p] + [a.hidden] * (len(a.kernels) - 1)
        self.cells = nn.ModuleList(ConvLSTMCell(ci, a.hidden, k, d)
                                   for ci, k, d in zip(cins, a.kernels, a.dilations))
        self.head = nn.Conv2d(a.hidden, p * p, 1)

    def forward(self, X):
        """X [B,5,1,H,W] -> Y_hat [B,5,1,H,W]; 9 steps, own predictions fed back from step 6."""
        B, _, _, H, W = X.shape
        p = self.arch.patch
        hs = [X.new_zeros(B, self.arch.hidden, H // p, W // p) for _ in self.cells]
        cs = [torch.zeros_like(h) for h in hs]
        outs, u = [], None
        for tau in range(1, 10):
            if tau <= 5:
                u = F.pixel_unshuffle(X[:, tau - 1], p)
            inp = u
            for l, cell in enumerate(self.cells):
                hs[l], cs[l] = cell(inp, hs[l], cs[l])
                inp = hs[l]
            if tau >= 5:                       # outputs of tau=1..4 are discarded
                u = torch.sigmoid(self.head(inp))  # folded prediction, fed back as next input
                outs.append(u)
        Y = torch.stack(outs, 1).flatten(0, 1)
        return F.pixel_shuffle(Y, p).view(B, 5, 1, H, W)


class Blank(nn.Module):
    def forward(self, X):
        return torch.zeros_like(X)


class CopyLast(nn.Module):
    def forward(self, X):
        return X[:, 4:5].expand_as(X).contiguous()


class Symmetrised(nn.Module):
    """Average of inverse(g, f(g X)) over the 8 group elements, in one stacked forward pass."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, X):
        B = X.shape[0]
        Yh = self.f(torch.cat([apply(g, X) for g in range(8)]))
        return sum(inverse(g, Yh[g * B:(g + 1) * B]) for g in range(8)) / 8


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
