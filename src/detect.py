# Run the trained detector over the input video and write an overlay MP4.

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
SRC = next(p for p in ROOT.glob("*.mp4") if "detection" not in p.name.lower())
WEIGHTS = ROOT / "model" / "fasterrcnn_best.pt"
OUT_MP4 = ROOT / "outputs" / "detections.mp4"

# TODO: expose --score-thr CLI flag; 0.20 picked by sweep on val
SCORE_THR = 0.20
DEVICE = torch.device("cpu")


def load_model():
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, min_size=480, max_size=640)
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    m.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    m.eval()
    return m


def main():
    model = load_model()
    cap = cv2.VideoCapture(str(SRC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    print(f"rendering {n} frames @ {W}x{H} -> {OUT_MP4}")

    inf_times = []
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ti = time.time()
        with torch.no_grad():
            out = model([TF.to_tensor(rgb).to(DEVICE)])[0]
        inf_times.append(time.time() - ti)

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        keep = scores >= SCORE_THR
        for b, s in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, f"vein {s:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"copper-vein detector  |  {int(keep.sum())} dets",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        i += 1
        if i % 50 == 0:
            print(f"  {i}/{n}  ({i / (time.time() - t0):.1f} fps end-to-end)")

    cap.release(); writer.release()
    avg_fps = 1.0 / (sum(inf_times) / max(len(inf_times), 1))
    print(f"done. {i} frames, model fps={avg_fps:.1f}")
    json.dump({"frames": i, "avg_model_fps": avg_fps, "device": str(DEVICE)},
              open(ROOT / "outputs" / "render_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
