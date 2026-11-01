# Weights - ACTION NEEDED before this repository is submittable

`run.py` expects **four** checkpoint files directly in this folder (`model_dir`), each a plain
`torch.save(model.state_dict(), ...)` of the matching class in `src/`:

| file               | class (in `src/`)                                                                    |
|---------------------|---------------------------------------------------------------------------------------|
| `restorer_A.pt`     | `restorer.Restorer3(c=40, nb=2)` - restoration copy "A" (never saw the 5 held-out scenes) |
| `restorer_B.pt`     | `restorer.Restorer3(c=40, nb=2)` - restoration copy "B" (trained on all scenes)       |
| `segnet_ft6.pt`     | `eurs_core.SegNet("vit_large_patch14_dinov2.lvd142m", freeze=False, train_blocks=6)`  |
| `segnet_frozen.pt`  | `eurs_core.SegNet("vit_large_patch14_dinov2.lvd142m", freeze=True, train_blocks=0)`   |

These are **not included in this repository** - they were trained on Kaggle and must be exported from there. To
produce them:

1. In the Kaggle notebook/output of the restoration-training run (R3), save each model's EMA weights:
   `torch.save(ema_model.state_dict(), "restorer_A.pt")` (and `_B.pt` for the second GPU's copy) - these are
   exactly the `ckpt_A` / `ckpt_B` files referenced by `rest_cfg.json` in that notebook's own output.
2. In the Kaggle notebook/output of the segmentation-training run (S3), save the **stage-B** ("all scenes")
   checkpoints of both segmentation models (the `ckpt/SL_ft.pt` and `ckpt/SL_frozen.pt` - or equivalently-named -
   files produced by that notebook) as `segnet_ft6.pt` and `segnet_frozen.pt` respectively.
3. Because a DINOv2 ViT-L/14 checkpoint is far larger than GitHub's 100 MB per-file limit, **do not commit these
   files to git**. Instead:
   - put all four files into one archive, e.g. `tar -czf eurs_t1_weights.tar.gz restorer_A.pt restorer_B.pt segnet_ft6.pt segnet_frozen.pt`
   - upload that archive somewhere public (a Hugging Face model repo is the most reliable for files this size;
     a GitHub Release asset or a Dropbox / Google-Drive **file** link also work)
   - compute its hash: `sha256sum eurs_t1_weights.tar.gz`
   - fill in `weights/weights_link.yaml` with that URL and hash (template already in this folder)
   - put the same URL and SHA-256 in the factsheet (Section 3 / Section "Other details")
4. Test it: `python repro_check.py . --download-weights` should download and unpack the archive into this
   folder, then `python repro_check.py . --run --input <some degraded images> --n 5` should run `run.py`
   end-to-end.

Until step 3 is done, `weights/weights_link.yaml` below still has placeholder values and this repository will
**not** pass `repro_check.py` or reproduce on the organizers' side.
