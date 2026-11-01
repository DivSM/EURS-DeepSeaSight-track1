"""EURS Track 1 (DeepSeaSight) - shared constants, image I/O, exposure normalisation and the two network
architectures (SegNet, Restorer3 / RCAB). Trimmed, inference-only extract of the team's own training codebase
(`eurs_core.py` + `eurs3_rest.py`): every training-only class (datasets, loss functions, the training loop) has
been removed because `run.py` only loads already-trained weights and runs inference. No code or weights are
copied from any other project; the only third-party component is the pretrained ImageNet encoder used inside
`timm.create_model(..., pretrained=True)` during TRAINING (see METHOD.md) - at inference time weights are always
loaded from the local checkpoint with `pretrained=False`, so no network access is required or attempted.
"""
import re
import math
from pathlib import Path

import numpy as np
import cv2
cv2.setNumThreads(1)
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

NCLS = 5
CLASSES = ["background", "coral", "fish", "elasmobranch", "other"]
PALETTE = np.array([[58, 58, 58], [235, 104, 52], [42, 120, 214], [27, 175, 122], [237, 161, 0]], np.uint8)
CAP = 1024                      # longest side the whole pipeline (and the required output) works at
EXPO_TARGET, EXPO_MAXG = 0.85, 128.0
SMOOTH_G0, SMOOTH_MAX = 10.0, 1.2


# --------------------------------------------------------------------------------------------------------- io
def read_rgb(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    assert im is not None, f"cannot read {p}"
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def cap_size(h, w, cap=CAP):
    s = min(1.0, cap / max(h, w))
    return int(round(h * s)), int(round(w * s))


def to_cap(img, interp, cap=CAP):
    """Downscale-only resize so the longest side is exactly `cap` (the scoring size); keeps size if already <= cap."""
    h, w = img.shape[:2]
    nh, nw = cap_size(h, w, cap)
    return img if (nh, nw) == (h, w) else cv2.resize(img, (nw, nh), interpolation=interp)


def colorize(m):
    return PALETTE[np.clip(m, 0, NCLS - 1)]


# --------------------------------------------------------------------------------------------------- exposure
def expose_gain(img):
    """Per-channel gain that brings the 99.5th percentile (of a 48x48 area-average) to 0.85. img: uint8 HxWx3."""
    x = cv2.resize(np.ascontiguousarray(img), (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32) / 255
    s = np.percentile(x, 99.5, axis=0)
    return np.clip(EXPO_TARGET / (s + 1e-4), 1.0, EXPO_MAXG).astype(np.float32)


def expose_u8(img, g):
    """Applies the exposure gain `g` to a uint8 image; a small blur is blended in first for very dark frames
    (a large gain multiplied onto an 8-bit staircase shows up as flat, contoured plateaus - this is the missing dither)."""
    gm = float(np.max(g))
    if SMOOTH_MAX > 0 and gm > SMOOTH_G0:
        sig = min(SMOOTH_MAX, 0.3 * math.log2(gm / SMOOTH_G0) + 0.4)
        img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), sig)
    return np.clip(np.rint(img.astype(np.float32) * g), 0, 255).astype(np.uint8)


def frame_q(img):
    """Light that survived in a frame: 99th percentile of the weakest channel (thumbnail), 0-1."""
    if img.shape[1] >= img.shape[0]:
        size = (96, int(round(96 * img.shape[0] / img.shape[1])))
    else:
        size = (int(round(96 * img.shape[1] / img.shape[0])), 96)
    t = cv2.resize(np.ascontiguousarray(img), size, interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32) / 255
    return float(np.percentile(t, 99, axis=0).min())


def amp_ctx(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


def load_state(model, path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd, strict=False)
    return model


# --------------------------------------------------------------------------------------------- segmentation net
def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, stride=s, padding=k // 2, bias=False), nn.GroupNorm(8, o), nn.GELU())


def nchw(t, c):
    return t if t.shape[1] == c else t.permute(0, 3, 1, 2)


class SegNet(nn.Module):
    """ViT encoder (DINOv2, timm) -> multi-scale lateral fusion -> two image-guided upsampling stages -> 5-class
    logits. Input: exposed image in [0, 1]. `pretrained` is forced False here: at inference the full state
    (encoder included) is always restored from the team's own checkpoint, never downloaded."""

    def __init__(self, name, freeze=False, d=128, train_blocks=0, pretrained=False):
        super().__init__()
        self.freeze = freeze
        idx = (5, 11, 17, 23) if "large" in name else (2, 5, 8, 11)
        self.enc = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=idx, dynamic_img_size=True)
        self.chs = self.enc.feature_info.channels()
        self.mult = max(self.enc.feature_info.reduction())
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad_(False)
        elif train_blocks > 0:
            idxs = {n_: int(m.group(1)) for n_, _ in self.enc.named_parameters() if (m := re.search(r"blocks\.(\d+)\.", n_))}
            top = max(idxs.values()) if idxs else -1
            for n_, p in self.enc.named_parameters():
                p.requires_grad_(n_ in idxs and idxs[n_] > top - train_blocks or (n_ not in idxs and "norm" in n_ and "patch" not in n_))
        self.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in self.chs])
        self.fuse = nn.Sequential(cbr(d, d), cbr(d, d))
        self.stem4 = nn.Sequential(cbr(3, 24, 3, 2), cbr(24, 48, 3, 2))
        self.stem1 = cbr(3, 32)
        self.up4 = nn.Sequential(cbr(d + 48, 96), cbr(96, 96))
        self.up1 = nn.Sequential(cbr(96 + 32, 64), cbr(64, 64))
        self.head = nn.Conv2d(64, NCLS, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.enc.eval()
        return self

    def forward(self, x):
        H, W = x.shape[-2:]
        xp = F.pad(x, (0, (-W) % self.mult, 0, (-H) % self.mult), mode="reflect")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            feats = [nchw(f, c) for f, c in zip(self.enc((xp - self.mean) / self.std), self.chs)]
        f = self.fuse(sum(l(t) for l, t in zip(self.lat, feats)))
        s4 = self.stem4(xp)
        u = self.up4(torch.cat([F.interpolate(f, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], 1))
        u = self.up1(torch.cat([F.interpolate(u, size=xp.shape[-2:], mode="bilinear", align_corners=False), self.stem1(xp)], 1))
        return self.head(u)[..., :H, :W]


# --------------------------------------------------------------------------------------------------- restoration
class RCAB(nn.Module):
    """Residual channel-attention block."""

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, padding=1))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // 4, 8), 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(max(c // 4, 8), c, 1), nn.Sigmoid())

    def forward(self, x):
        y = self.body(x)
        return x + y * self.se(y)
