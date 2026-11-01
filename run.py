"""Track 1 inference entry point - DeepSeaSight, reported submission #19229
(PNG_01_sharp_lo__seg_nocal, testing-phase composite 0.579).

The organizers call   main(model_dir, input_dir, output_dir, device)   and nothing else.

  model_dir   the weights/ folder of this repository: restorer_A.pt, restorer_B.pt, segnet_ft6.pt, segnet_frozen.pt
  input_dir   a folder of degraded images (<scene>_<type>_<level>.png), all at their original, full size
  output_dir  one file per input image, same file stem:
                  output_dir/restored/<stem>.png    RGB restored image
                  output_dir/seg_masks/<stem>.png   single-channel mask, labels 0..4 (background, coral, fish,
                                                     elasmobranch, other)
              both resized to EXACTLY the scoring size: longest side 1024 px, aspect ratio kept (downscale only).
  device      "cuda" or "cpu"

Method in one paragraph (see METHOD.md for the full description and every external component used in TRAINING):
every image is processed strictly on its own. It is first downsized (if needed) to the scoring size (longest
side 1024 px), then its own per-channel exposure gain is computed and applied before both networks. Restoration
is the average of two independently-trained copies of the same restoration network (`Restorer3`), each run with
horizontal-flip test-time augmentation, followed by a fixed unsharp-mask step (amount 0.2, sigma 1.0 - the
"sharp_lo" variant: half the strength tuned on held-out training scenes, the better of the two strengths
actually uploaded to the testing-phase leaderboard). Segmentation is an un-calibrated, equal-weight average of
two DINOv2 ViT-L/14 segmentation networks ("seg_nocal": this plain average scored higher on the real,
testing-phase leaderboard than the calibrated class-weighted combination that was tuned on 5 held-out training
scenes - evidence that the calibration had overfit those 5 scenes - see METHOD.md and factsheet Section 2).

Nothing here reads the `<scene>` part of the file name, groups frames of the same scene, or looks at any image
other than the one being processed right now: every tensor computed below depends on exactly one input array,
`im`, read fresh inside the loop below.
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from eurs_core import NCLS, to_cap, expose_gain, load_state, read_rgb   # noqa: E402
from restorer import Restorer3, restore_img, unsharp                      # noqa: E402
from pipeline import seg_probs, combine, clean_mask                        # noqa: E402

# ----------------------------------------------------------------------------------------------- fixed settings
# These are the exact, already-chosen numbers of the reported submission; none of them is read from, or fit to,
# the images processed by this script. Provenance: METHOD.md.
SCALE = 1.5           # the single scale kept after calibration on held-out training scenes
CROP = 518            # work-resolution short side at scale 1.0 (DINOv2 patch-14 crop used during training)
SHARP_AMOUNT = 0.2     # "sharp_lo" = round(0.5 * main_amount, 3); main_amount (held-out-chosen) = 0.4
SHARP_SIGMA = 1.0
SEG_WEIGHTS = [0.5, 0.5]             # "nocal": equal weight on both segmentation models
SEG_MULT = [1.0, 1.0, 1.0, 1.0, 1.0]  # "nocal": no per-class multiplier
MIN_FRAC = 0.0                        # "nocal": no small-component clean-up
FLIP_TTA = True


def main(model_dir, input_dir, output_dir, device):
    torch.set_grad_enabled(False)
    dev = torch.device(device)

    # --- restoration ensemble: two independently-trained copies of the same network, averaged -----------------
    rest_models = []
    for fname in ("restorer_A.pt", "restorer_B.pt"):
        path = os.path.join(model_dir, fname)
        if not os.path.isfile(path):
            continue
        m = Restorer3(c=40, nb=2)
        load_state(m, path, map_location=dev)
        rest_models.append(m.to(dev).eval())
    assert rest_models, f"no restorer checkpoint found in {model_dir} (expected restorer_A.pt and/or restorer_B.pt)"

    # --- segmentation ensemble: DINOv2 ViT-L/14, one partially fine-tuned, one fully frozen ---------------------
    from eurs_core import SegNet
    seg_cfgs = [
        dict(file="segnet_ft6.pt", freeze=False, train_blocks=6),
        dict(file="segnet_frozen.pt", freeze=True, train_blocks=0),
    ]
    seg_models, seg_w = [], []
    for cfg, w in zip(seg_cfgs, SEG_WEIGHTS):
        path = os.path.join(model_dir, cfg["file"])
        if not os.path.isfile(path):
            continue
        m = SegNet("vit_large_patch14_dinov2.lvd142m", freeze=cfg["freeze"], train_blocks=cfg["train_blocks"], pretrained=False)
        load_state(m, path, map_location=dev)
        seg_models.append(m.to(dev).eval())
        seg_w.append(w)
    assert seg_models, f"no segmentation checkpoint found in {model_dir} (expected segnet_ft6.pt and/or segnet_frozen.pt)"
    if len(seg_w) < len(seg_cfgs):
        s = sum(seg_w)
        seg_w = [w / s for w in seg_w]   # renormalise if one checkpoint is missing (both are expected to be present)

    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)

    for name in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        # Every array below is derived from THIS file only. The water type / level can be read from the end of
        # `stem` if ever needed (`stem.rsplit("_", 2)`); neither they nor the scene part of `stem` are used by
        # this model, which conditions on pixel content and the per-image exposure gain alone.
        im = to_cap(read_rgb(os.path.join(input_dir, name)), interp=cv2.INTER_AREA)
        g = expose_gain(im)

        # restoration: average of the restorer ensemble, each with flip TTA, then the fixed unsharp step
        outs = [restore_img(m, dev, im, g, flip=FLIP_TTA) for m in rest_models]
        rest = outs[0] if len(outs) == 1 else np.clip(np.rint(np.mean([o.astype(np.float32) for o in outs], axis=0)), 0, 255).astype(np.uint8)
        rest = unsharp(rest, SHARP_AMOUNT, SHARP_SIGMA)

        # segmentation: equal-weight probability average at one fixed scale, each model with flip TTA
        probs = [seg_probs(m, dev, im, g, SCALE, CROP, flip=FLIP_TTA) for m in seg_models]
        p = combine(probs, seg_w, SEG_MULT)
        mask = clean_mask(p.argmax(0).astype(np.uint8), MIN_FRAC)

        Image.fromarray(rest).save(os.path.join(output_dir, "restored", stem + ".png"))
        Image.fromarray(mask).save(os.path.join(output_dir, "seg_masks", stem + ".png"))
