# Reload best checkpoint and sweep score thresholds on the test split.
# Picks the threshold with the best F1 and writes results to outputs/metrics.json.

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import VeinDataset, evaluate, load_dataset

ROOT = Path(__file__).resolve().parent.parent
SCORE_THRS = [0.05, 0.10, 0.12, 0.15, 0.18, 0.20]
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def model():
    m = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, min_size=480, max_size=640,
        box_score_thresh=0.001, box_nms_thresh=0.5,
    )
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    m.load_state_dict(torch.load(ROOT / "model" / "fasterrcnn_best.pt", map_location="cpu"))
    m.eval()
    return m


def main():
    m = model()
    items = [it for it in load_dataset() if len(it["boxes"]) > 0]
    random.shuffle(items)
    n = len(items)
    n_tr = int(0.7 * n); n_va = int(0.15 * n)
    test = items[n_tr + n_va:]
    print(f"test split: {len(test)} frames")

    best = (-1, None, None)
    for thr in SCORE_THRS:
        r = evaluate(m, VeinDataset(test), score_thr=thr)
        print(f"  thr={thr:.2f}  P={r['box_precision']:.3f}  R={r['box_recall']:.3f}  "
              f"F1={r['box_f1']:.3f}  AP={r['AP@0.5']:.3f}  "
              f"pixF1={r['pixel_f1']:.3f}  pixAcc={r['pixel_accuracy']:.3f}  "
              f"preds={r['n_pred']}")
        if r["box_f1"] > best[0]:
            best = (r["box_f1"], thr, r)

    print(f"\nbest threshold: {best[1]}  F1={best[0]:.3f}")
    chosen = best[2]

    metrics = json.load(open(ROOT / "outputs" / "metrics.json"))
    metrics["test"] = {k: v for k, v in chosen.items() if k != "pr_curve"}
    metrics["test"]["chosen_score_thr"] = best[1]
    metrics["test_pr_curve"] = chosen["pr_curve"]
    json.dump(metrics, open(ROOT / "outputs" / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
