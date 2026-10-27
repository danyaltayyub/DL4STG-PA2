"""Speed-k windows; frame numbers are 1-based as in the handout."""
import torch


def window_frames(t, k):
    inp = [t + i * k for i in range(5)]
    tgt = [t + (5 + i) * k for i in range(5)]
    return inp, [f for f in tgt if f <= 20]


def make_windows(videos, t, k):
    """videos [B,20,1,64,64] (uint8 or float); t, k ints or LongTensors [B]. Returns X, Y, mask."""
    B, dev = videos.shape[0], videos.device
    t = torch.as_tensor(t, device=dev).expand(B)
    k = torch.as_tensor(k, device=dev).expand(B)
    steps = torch.arange(5, device=dev)
    inp = t[:, None] + k[:, None] * steps                 # 1-based frame numbers
    tgt = t[:, None] + k[:, None] * (steps + 5)
    assert int(inp.max()) <= 20, "input frame beyond 20"
    mask = tgt <= 20
    rows = torch.arange(B, device=dev)[:, None]
    X = videos[rows, inp - 1]
    Y = videos[rows, (tgt - 1).clamp(max=19)]
    if videos.dtype == torch.uint8:
        X, Y = X.float() / 255, Y.float() / 255
    Y = Y * mask[:, :, None, None, None]
    return X, Y, mask
