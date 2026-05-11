# Side-by-side panel: input | Frangi pseudo-label | Faster R-CNN prediction.

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
MASKS = ROOT / "data" / "masks"
ANN = ROOT / "data" / "annotations" / "auto_coco.json"
WEIGHTS = ROOT / "model" / "fasterrcnn_best.pt"
OUT = ROOT / "outputs" / "figures"

PICK_FRAMES = [25, 70, 110, 145]
SCORE_THR = 0.20


def label_strip(img, txt):
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def load_model():
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, min_size=480, max_size=640)
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.eval()
    return m


def main():
    data = json.load(open(ANN))
    by_id = {im["id"]: im["file_name"] for im in data["images"]}
    boxes_by_id = {im["id"]: [] for im in data["images"]}
    for a in data["annotations"]:
        boxes_by_id[a["image_id"]].append(a["bbox"])
    model = load_model()

    cols = []
    for pid in PICK_FRAMES:
        fn = by_id[pid]
        bgr = cv2.imread(str(FRAMES / fn))
        col_in = label_strip(bgr.copy(), "(a) input")

        mask = cv2.imread(str(MASKS / fn.replace(".jpg", "_mask.png")), 0)
        ov = bgr.copy(); ov[mask > 0] = (0, 255, 255)
        mid = cv2.addWeighted(bgr, 0.55, ov, 0.45, 0)
        for x, y, w, h in boxes_by_id.get(pid, []):
            cv2.rectangle(mid, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
        col_mid = label_strip(mid, "(b) Frangi pseudo-label")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            out = model([TF.to_tensor(rgb)])[0]
        pr = bgr.copy()
        for b, s in zip(out["boxes"].numpy(), out["scores"].numpy()):
            if s < SCORE_THR:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            cv2.rectangle(pr, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(pr, f"{s:.2f}", (x1, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
        col_right = label_strip(pr, "(c) Faster R-CNN")
        cols.append(np.vstack([col_in, col_mid, col_right]))

    panel = np.hstack(cols)
    cv2.imwrite(str(OUT / "qualitative_panel.jpg"), panel)
    cv2.imwrite(str(OUT / "qualitative_panel.png"), panel)
    print(f"wrote {OUT}/qualitative_panel.{{jpg,png}}  shape={panel.shape}")


if __name__ == "__main__":
    main()
