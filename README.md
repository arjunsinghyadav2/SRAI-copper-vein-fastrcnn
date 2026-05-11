# SRAI · Copper-vein detector

Real-time bounding-box detection of copper-bearing veins on a chalcopyrite
specimen, trained from Frangi-vesselness pseudo-labels - no human annotation.

The full write-up is in [`RealTimeCopperVeinDetection.pdf`](RealTimeCopperVeinDetection.pdf).

## What's here

```
.
├── RealTimeCopperVeinDetection.pdf    report
├── specimen_video.mp4                 input clip
├── src/
│   ├── extract_frames.py              ffmpeg  data/frames/
│   ├── auto_annotate.py               Frangi pseudo-label COCO
│   ├── train.py                       Faster R-CNN MobileNetV3-FPN fine-tune
│   ├── eval.py                        threshold sweep on test split
│   ├── detect.py                      run model over the source video
│   ├── qualitative.py                 input | label | prediction panel
│   └── figures.py                     pipeline / training / PR figures
├── data/annotations/auto_coco.json    1 535 boxes from Frangi
├── model/fasterrcnn_best.pt           best-val-F1 checkpoint (~76 MB)
└── outputs/
    ├── detections.mp4                 model output overlaid on source
    ├── metrics.json                   val history + test metrics
    ├── render_stats.json              inference fps
    ├── train.log                      per-epoch loss / val F1
    └── figures/                       qualitative + train + PR + pipeline
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## End-to-end

```bash
.venv/bin/python src/extract_frames.py
.venv/bin/python src/auto_annotate.py
EPOCHS=6 .venv/bin/python src/train.py
.venv/bin/python src/eval.py
.venv/bin/python src/detect.py
.venv/bin/python src/qualitative.py
.venv/bin/python src/figures.py
```

The frame extraction + Frangi pass takes ~2 min, training is ~8 min on a
laptop CPU.

## Test results (24-frame held-out split, score thr 0.20)

| | Box IoU=0.5 | Box IoU=0.3 |
|---|---|---|
| Precision | 0.325 | — |
| Recall | 0.476 | 0.828 |
| F1 | 0.386 | 0.414 |
| AP | 0.347 | 0.551 |

Pixel-level: F1 = 0.736, accuracy = 0.950. Inference 6.6 fps on M-series CPU.

## Open items

- bigger expert-annotated test set (the current ground truth is Frangi itself)
- multispectral / NIR channel as a second cue for chalcopyrite
- multi-class head (chalcopyrite vs quartz vs host rock)
- second specimen for cross-rock generalisation
- swap MobileNet for a real GPU run on ResNet-50 FPN
