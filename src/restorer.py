"""EURS Track 1 (DeepSeaSight) - the per-frame restoration network `Restorer3`: a gain-conditioned U-Net of
RCAB (residual channel-attention) blocks. Trained from scratch on the EURS training split only; no pretrained
weights or external data are used for restoration. Trimmed, inference-only extract of `eurs3_rest.py` (the
auxiliary water-type/depth head, the training dataset class and the EMA-training wrapper are training-only and
removed here)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from eurs_core import RCAB, expose_u8


class Restorer3(nn.Module):
    """Input: exposed frame `xn` (B,3,H,W) in [0,1] and its per-channel exposure gain `g` (B,3).
    out = xn * exp(a) + b, with per-pixel gain map a and offset map b (matches the underwater image-formation
    model). Two independently-trained copies (A: never saw the 5 held-out calibration scenes; B: all scenes)
    are averaged at inference for the submitted result."""

    def __init__(self, c=40, nb=2):
        super().__init__()
        ch = [c, 2 * c, 4 * c, 8 * c]
        self.inc = nn.Conv2d(6, ch[0], 3, padding=1)
        self.stages = nn.ModuleList([nn.Sequential(*[RCAB(k) for _ in range(nb)]) for k in ch])
        self.down = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1) for i in range(3)])
        self.up = nn.ModuleList([nn.Conv2d(ch[i + 1], ch[i], 1) for i in range(3)])
        self.dec = nn.ModuleList([nn.Sequential(nn.Conv2d(2 * ch[i], ch[i], 3, padding=1), *[RCAB(ch[i]) for _ in range(nb)]) for i in range(3)])
        self.out = nn.Conv2d(ch[0], 6, 3, padding=1)
        # training-only auxiliary head (water type / depth); kept so a full checkpoint loads with strict=False,
        # never used at inference
        self.aux = nn.Linear(ch[3], 11)

    def forward(self, xn, g):
        H, W = xn.shape[-2:]
        cond = (g.float().log10() / 2).view(g.shape[0], -1, 1, 1).expand(-1, -1, H, W).to(xn.dtype)
        h = F.pad(torch.cat([xn, cond], 1), (0, (-W) % 8, 0, (-H) % 8), mode="reflect")
        f, h = [], self.inc(h)
        for i in range(4):
            h = self.stages[i](h)
            if i < 3:
                f.append(h)
                h = self.down[i](h)
        for i in reversed(range(3)):
            h = F.interpolate(self.up[i](h), size=f[i].shape[-2:], mode="bilinear", align_corners=False)
            h = self.dec[i](torch.cat([h, f[i]], 1))
        a, b = self.out(h)[..., :H, :W].float().chunk(2, 1)
        return xn.float() * torch.exp(2 * torch.tanh(a / 2)) + b


def unsharp(img, amount, sigma):
    """Fixed post-processing unsharp mask applied to the restored uint8 image (never inside the network)."""
    if amount <= 0:
        return img
    bl = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img.astype(np.float32) + amount * (img.astype(np.float32) - bl.astype(np.float32)), 0, 255).round().astype(np.uint8)


@torch.no_grad()
def restore_img(model, dev, img, g, flip=True):
    """uint8 HxWx3 image + its exposure gain -> uint8 HxWx3 restored image. Horizontal-flip TTA (average of the
    image and its flip, both run through the network, flipped back)."""
    model.eval()
    x = torch.from_numpy(expose_u8(img, g)).permute(2, 0, 1)[None].float().div(255).to(dev)
    gt = torch.from_numpy(np.asarray(g, np.float32))[None].to(dev)
    y = model(x, gt).float()
    if flip:
        y = (y + model(x.flip(-1), gt).float().flip(-1)) / 2
    return (y.clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
