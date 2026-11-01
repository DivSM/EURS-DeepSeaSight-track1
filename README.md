# EURS @ MaCVi - Track 1 reproduction repository 
Team: OffshorePixels 
Method: DeepSeaSight

Submission **#19229** (composite 0.579)
A per-frame restoration ensemble (`Restorer3` A+B, unsharp 0.2) and an un-calibrated, equal-weight ensemble of two DINOv2-backbone segmentation networks. 
See **`METHOD.md`** for the full method description, every external dataset / pretrained
model used, and exactly which inference-time choices were fixed in advance and why.

```
repro.yaml       team / track / submission_id / contact
METHOD.md        approach, training data, external data, non-standard inference details
run.py           main(model_dir, input_dir, output_dir, device) - the required entry point
src/             model code imported by run.py (SegNet, Restorer3, inference helpers)
weights/         trained checkpoints attached
repro_check.py   the organizers' checker
docker/Dockerfile the organizers' base image
```
