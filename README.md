# EURS @ MaCVi WACV 2027 - Track 1 reproduction repository (team DeepSeaSight)

Reproduces testing-phase submission **#19229** (composite 0.579): a per-frame restoration ensemble
(`Restorer3` A+B, unsharp 0.2) and an un-calibrated, equal-weight ensemble of two DINOv2-backbone segmentation
networks ("seg_nocal"). See **`METHOD.md`** for the full method description, every external dataset / pretrained
model used, and exactly which inference-time choices were fixed in advance and why.

```
repro.yaml       team / track / submission_id / contact
METHOD.md        approach, training data, external data, non-standard inference details
run.py           main(model_dir, input_dir, output_dir, device) - the required entry point
src/             model code imported by run.py (SegNet, Restorer3, inference helpers)
weights/         ACTION NEEDED - see weights/README.md: the trained checkpoints are not yet attached
repro_check.py   the organizers' checker; run before submitting
docker/Dockerfile the organizers' base image
```

Before handing this in: fill `weights/weights_link.yaml` (or commit the checkpoints) per `weights/README.md`,
then run `python repro_check.py . --run --input <some degraded images> --n 5`.
