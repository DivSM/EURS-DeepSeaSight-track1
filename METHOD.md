# Method description

Team: **DeepSeaSight** (macvi.org user #2097)   Track(s): **1** (a separate repository + `METHOD.md` covers Track 2)
Reported testing-phase submission: **#19229** (`PNG_01_sharp_lo__seg_nocal.zip`, composite **0.579**)

## Approach (one page at most)

Every output (restored image and segmentation mask) is produced **strictly from its own degraded input image** -
no other frame is ranked, grouped, fused with, or voted with. What is shared between frames is fixed, trained-once
state: network weights and a handful of plain numbers chosen once on held-out EURS training scenes (see "Anything
non-standard at inference"). Nothing is estimated from, or adapted to, the testing-phase images themselves.

**Restoration** (`src/restorer.py`, class `Restorer3`): a gain-conditioned U-Net built from residual
channel-attention blocks (RCAB). The input image is first normalised by a per-channel exposure gain (the
99.5th-percentile of a down-sampled thumbnail is brought to a fixed target brightness); the network then predicts
a per-pixel multiplicative gain map and an additive offset map, `restored = exposed * exp(a) + b`, which matches
the standard underwater image-formation model. Two copies of this network were trained independently from
scratch (same architecture, different random data splits/seeds: "A" never saw the 5 scenes held out for
calibration, "B" saw all training scenes) and their outputs are averaged at inference, each with horizontal-flip
test-time augmentation. A fixed unsharp-mask step (Gaussian-blur based) is applied to the averaged restored image
as the very last step; its strength is one of two fixed values prepared in advance and chosen by testing-phase
leaderboard score (see below).

**Segmentation** (`src/eurs_core.py`, class `SegNet`): two networks built on the same pretrained ViT-L/14
backbone (DINOv2, see "Training data"): a multi-scale lateral feature-pyramid fusion over four backbone stages,
followed by two image-guided bilinear-upsampling stages back to full resolution and a 1x1 classification head to
5 classes. One copy has its last 6 transformer blocks fine-tuned; the other keeps the entire backbone frozen and
trains only the lightweight decoder on top of it (this acts as a different, less test-set-shaped view of the same
features). The reported submission averages the per-pixel class probabilities of the two networks with **equal
weight and no per-class reweighting** ("seg_nocal" - see below) and takes the arg-max per pixel.

Restoration and segmentation are trained and run completely independently; they share only the exposure-gain
pre-processing step.

## Training data

Trained on the EURS training split released by the organizers (degraded/ground-truth pairs and segmentation
masks), with 5 scenes (`fish5, fish7, fish9, ray1, turtles1`) held out for model selection and for the "Anything
non-standard at inference" calibration below. The restoration network uses **no other data and no pretrained
weights** (trained from scratch).

**All external data and pretrained models used anywhere in this method:**

1. **DINOv2 ViT-L/14 (`vit_large_patch14_dinov2.lvd142m`)** - pretrained vision backbone used, via `timm`, as the
   encoder of both segmentation networks (one partially fine-tuned, one frozen). Apache License 2.0. Oquab et al.,
   *"DINOv2: Learning Robust Visual Features without Supervision"*, TMLR 2024 [1].
2. **Coralscapes** - public coral-reef segmentation dataset; ~1,000 images re-mapped to the 5 EURS classes and
   added to segmentation training to improve the `coral` and `background` classes. Apache License 2.0. Sauder et
   al., *"The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs"*, 2025 [2].
3. **UIIS10K** - public underwater instance-segmentation dataset (Hugging Face `LiamLian0727/UIIS10K`); ~1,500 of
   its ~10,048 images, re-mapped to the 5 EURS classes, added to segmentation training. Li et al., *"Advancing
   Marine Research: UWSAM Framework and UIIS10K Dataset for Precise Underwater Instance Segmentation"*,
   arXiv:2505.15581, 2025 [3].
4. **USIS10K** - public underwater salient/instance-segmentation dataset (Hugging Face `LiamLian0727/USIS10K`);
   ~1,500 of its ~10,632 images, re-mapped to the 5 EURS classes, added to segmentation training. Lian et al.,
   *"Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A
   Large-scale Dataset"*, ICML 2024 [4].
5. **iNaturalist research-grade photographs** - 845 CC0 / CC-BY licensed photographs of corals, fish,
   elasmobranchs (sharks, rays) and other relevant taxa (760 used in training, 85 held out), weakly masked and
   added to segmentation training specifically to improve coverage of the rare `elasmobranch` class. A credits
   file listing every photograph's licence and author is kept with the training code (not required for
   inference). **No SUIM data was used.**
6. **AdamW** optimiser [5] and standard data augmentation; no other pretrained checkpoints, no synthetic-data
   generators and no test-time adaptation of any kind.

Full IEEE references are given in the factsheet's bibliography; the same list is repeated at the end of this file
for convenience.

## Anything non-standard at inference

* **Per-frame only, by design.** Nothing below is computed from the testing-phase images; everything was fixed
  before any testing-phase image was processed.
* **Horizontal-flip test-time augmentation** for every network (image and its flip both run, flipped back,
  averaged with the non-flipped prediction - standard TTA, not learned).
* **Restoration ensembling**: the two independently-trained `Restorer3` copies ("A", "B") are averaged.
* **Single-scale segmentation inference** at a scale chosen on the 5 held-out training scenes by coordinate-ascent
  search over {model weights, scale set, per-class multipliers, small-component clean-up, black-frame policy}
  (objective: mIoU x F1, pooled so every held-out scene counts equally). The search kept scale 1.5x of a 518px
  working crop and discarded the other scales that were tried (1.0x, 1.25x).
* **Two fixed variants were prepared from the identical per-frame network outputs and both uploaded to the
  testing-phase leaderboard**; the better-scoring one is the reported submission (allowed under "Data use" in the
  template README: one fixed, globally-applied choice, picked by testing-phase leaderboard score, not by looking
  at individual test images):
  - restoration: unsharp-mask strength **0.2** ("sharp_lo" = half the strength chosen by the held-out search,
    0.4) scored higher on the real board than the held-out-chosen strength itself (0.4, "main") and the stronger
    alternative (0.7, "sharp_hi");
  - segmentation: the **un-calibrated, equal-weight** average of the two segmentation networks ("seg_nocal": model
    weights [0.5, 0.5], no per-class multiplier, no small-component clean-up) scored higher on the real board than
    the calibrated combination fit on the 5 held-out scenes (model weights [0.65, 0.35], per-class multipliers
    [1.22, 1.0, 1.0, 1.0, 1.73]) - evidence that the 5-scene calibration had overfit and did not generalise, so the
    plain average was kept as the final entry instead.
* **Frames with (near-)zero signal** ("black" frames, detected per-frame as the 99th-percentile brightness of the
  weakest colour channel falling below a fixed threshold) are **not treated specially** in the reported
  submission: they are passed through the ordinary network pipeline like every other frame (one of the fixed
  policies considered during calibration - "let the network decide" - scored at least as well on held-out data as
  swapping in a class-prior map or a single fixed class, and is simplest).
* No cropping, tiling, multi-crop averaging, or any other test-time step beyond what is listed above.

## Reproduction notes

* Expected run time: well under 1 s/image on a single RTX-3090-class GPU (two small CNN/ViT forward passes per
  image, each run twice for flip TTA); GPU memory: a few GB (dominated by the frozen/fine-tuned ViT-L/14
  backbones in fp32).
* The leaderboard submission itself was produced on Kaggle's 2x NVIDIA T4 notebooks (one GPU per restorer copy /
  one GPU per segmentation model, run in parallel); `run.py` in this repository runs everything on the single
  device it is given and produces numerically the same result (only wall-clock time differs).
* `model_dir` (= `weights/`) must contain `restorer_A.pt`, `restorer_B.pt`, `segnet_ft6.pt`, `segnet_frozen.pt` -
  see `weights/README.md` for exactly how to produce these four files from the team's own training checkpoints.
* Training code is not included here (not required by the template); it is the team's private Kaggle notebooks.
* **IEEE references**
  [1] M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *Trans. Mach. Learn. Res.*, 2024.
  [2] J. Sauder, V. Minh, I. Fdhila, E. Chimani, T. Krueger, and M. Michel, "The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs," arXiv:2503.20000, 2025.
  [3] H. Li, S. Lian, Z. Li, R. Cong, C. Li, L. T. Yang, W. Zhang, and S. Kwong, "Advancing Marine Research: UWSAM Framework and UIIS10K Dataset for Precise Underwater Instance Segmentation," arXiv:2505.15581, 2025.
  [4] S. Lian, Z. Zhang, H. Li, W. Li, L. T. Yang, S. Kwong, and R. Cong, "Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset," in *Proc. 41st Int. Conf. Mach. Learn. (ICML)*, 2024.
  [5] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2019.
