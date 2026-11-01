"""EURS Track 1 (DeepSeaSight) - the fixed, per-frame segmentation inference and ensembling used for the
reported submission (#19229, `sharp_lo` restoration + `seg_nocal` masks). Trimmed, inference-only extract of the
team's own `eurs2_base.py` / `eurs2_seg.py` / `eurs3_pf.py`.

Every value below (scale, crop size, ensemble weights, class multipliers, clean-up threshold, black-frame
policy) is a FIXED, already-chosen number - nothing here is computed from or adapted to the test images. See
METHOD.md for where the numbers come from (held-out scenes of the EURS training split) and why this exact
("nocal") combination was selected: it is one of a small, fixed set of variants prepared in advance and chosen
by its testing-phase leaderboard score, which Section 2 of the factsheet records.
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from eurs_core import NCLS, expose_u8, amp_ctx

NCLS_ = NCLS


def work_hw(H, W, base):
    s0 = base / min(H, W)
    return int(round(H * s0)), int(round(W * s0))


@torch.no_grad()
def seg_probs(model, dev, img, g, scale, base, flip=True):
    """Softmax probabilities of ONE model at ONE scale, upsampled back to the input's own resolution.
    `img`: uint8 HxWx3 (already capped to the scoring size). -> float32 (NCLS, H, W)."""
    model.eval()
    x = torch.from_numpy(expose_u8(img, g)).permute(2, 0, 1)[None].float().div(255).to(dev)
    H, W = x.shape[-2:]
    Hw, Ww = work_hw(H, W, base)
    h, w = int(round(Hw * scale)), int(round(Ww * scale))
    t = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=h < H)
    with amp_ctx(torch.device(dev)):
        p = model(t).float().softmax(1)
        if flip:
            p = (p + model(t.flip(-1)).float().softmax(1).flip(-1)) / 2
    return F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0].cpu().numpy()


def combine(probs_per_model, w, mult):
    """probs_per_model: list of (NCLS,H,W) arrays, one per segmentation model. -> (NCLS,H,W) float32."""
    p = sum(pm * wm for pm, wm in zip(probs_per_model, w))
    return p * np.asarray(mult, np.float32)[:, None, None]


def clean_mask(m, min_frac=0.0):
    """Small connected components take the majority label of their surroundings. No-op when `min_frac` <= 0
    (the case used for the reported, `seg_nocal`, submission)."""
    if min_frac <= 0:
        return m
    H, W = m.shape
    thr = max(8, int(min_frac * H * W))
    out = m.copy()
    for c in range(NCLS):
        b = (m == c).astype(np.uint8)
        if not b.any():
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        small = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] < thr][:600]
        for k in small:
            x, y, w, h, _ = st[k]
            y0, y1, x0, x1 = max(0, y - 3), min(H, y + h + 3), max(0, x - 3), min(W, x + w + 3)
            sub = lab[y0:y1, x0:x1] == k
            ring = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & ~sub
            v = m[y0:y1, x0:x1][ring]
            v = v[v != c]
            if len(v):
                out[y0:y1, x0:x1][sub] = np.bincount(v, minlength=NCLS).argmax()
    return out
