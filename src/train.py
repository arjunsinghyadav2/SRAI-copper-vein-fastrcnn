# Fine-tune Faster R-CNN on the Frangi pseudo-labels.
#
# TODO: try larger min_size once we have a real GPU
# TODO: maybe Tversky/focal-style box loss reweighting for the long-tail vein sizes

import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as tud
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
ANN = ROOT / "data" / "annotations" / "auto_coco.json"
OUT = ROOT / "outputs"
MODELS = ROOT / "model"

# MPS hangs on torchvision detection models — stick with CPU for now.
# FIXME: revisit when torch ships a fixed Metal backend
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def load_dataset():
    data = json.load(open(ANN))
    by_img = {im["id"]: {"file": im["file_name"], "boxes": []} for im in data["images"]}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
    return sorted(by_img.values(), key=lambda r: r["file"])


class VeinDataset(tud.Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(FRAMES / it["file"])), cv2.COLOR_BGR2RGB)
        boxes = np.array(it["boxes"], dtype=np.float32) if it["boxes"] else np.zeros((0, 4), np.float32)

        if self.augment and random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            if len(boxes):
                W = img.shape[1]
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([i]),
            "area": torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                dtype=torch.float32) if len(boxes) else torch.zeros(0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return TF.to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes=2):
    m = fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT", trainable_backbone_layers=3,
        min_size=480, max_size=640,
        box_score_thresh=0.001, box_nms_thresh=0.5,
    )
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return m


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-8)


def evaluate(model, ds, score_thr=0.5, iou_thr=0.5):
    model.eval()
    tps = fps = fns = 0
    p_tp = p_fp = p_fn = p_tn = 0
    pr_pairs = []
    n_gt = 0

    with torch.no_grad():
        for img, tgt in ds:
            out = model([img.to(DEVICE)])[0]
            scores = out["scores"].cpu().numpy()
            keep = scores >= score_thr
            pred = out["boxes"].cpu().numpy()[keep]
            psc = scores[keep]
            gt = tgt["boxes"].numpy()
            n_gt += len(gt)

            order = np.argsort(-psc)
            matched = set()
            for idx in order:
                ious = box_iou(pred[idx:idx + 1], gt) if len(gt) else np.zeros((1, 0))
                ious = ious[0]
                best, best_iou = -1, 0.0
                for j, v in enumerate(ious):
                    if j in matched: continue
                    if v > best_iou:
                        best, best_iou = j, v
                if best_iou >= iou_thr:
                    tps += 1; matched.add(best)
                    pr_pairs.append((float(psc[idx]), 1))
                else:
                    fps += 1
                    pr_pairs.append((float(psc[idx]), 0))
            fns += len(gt) - len(matched)

            # paint mask version for pixel-level metrics
            H, W = img.shape[1], img.shape[2]
            pm = np.zeros((H, W), np.uint8)
            gm = np.zeros((H, W), np.uint8)
            for b in pred:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                pm[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1
            for b in gt:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                gm[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1
            p_tp += int(((pm == 1) & (gm == 1)).sum())
            p_fp += int(((pm == 1) & (gm == 0)).sum())
            p_fn += int(((pm == 0) & (gm == 1)).sum())
            p_tn += int(((pm == 0) & (gm == 0)).sum())

    P = tps / max(tps + fps, 1)
    R = tps / max(tps + fns, 1)
    F1 = 2 * P * R / max(P + R, 1e-8)
    pp = p_tp / max(p_tp + p_fp, 1)
    pr = p_tp / max(p_tp + p_fn, 1)
    pf1 = 2 * pp * pr / max(pp + pr, 1e-8)
    pacc = (p_tp + p_tn) / max(p_tp + p_fp + p_fn + p_tn, 1)

    # AP @ iou_thr (11-point interpolation)
    pr_pairs.sort(key=lambda x: -x[0])
    tpc = fpc = 0
    pts = []
    for s, t in pr_pairs:
        tpc += t
        fpc += (1 - t)
        rec = tpc / max(n_gt, 1)
        prec = tpc / max(tpc + fpc, 1)
        pts.append((rec, prec))
    ap = 0.0
    for r0 in np.linspace(0, 1, 11):
        rs = [p for rc, p in pts if rc >= r0]
        ap += (max(rs) if rs else 0.0) / 11.0

    return {
        "box_precision": P, "box_recall": R, "box_f1": F1,
        "pixel_precision": pp, "pixel_recall": pr, "pixel_f1": pf1, "pixel_accuracy": pacc,
        "AP@0.5": ap, "n_pred": tps + fps, "n_gt": n_gt,
        "tp": tps, "fp": fps, "fn": fns,
        "pr_curve": pts,
    }


def main():
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    items = [it for it in load_dataset() if len(it["boxes"]) > 0]
    random.shuffle(items)
    n = len(items)
    n_tr = int(0.7 * n); n_va = int(0.15 * n)
    tr_items, va_items, te_items = items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]
    print(f"frames: {n}  train={len(tr_items)} val={len(va_items)} test={len(te_items)}")

    tr_loader = tud.DataLoader(VeinDataset(tr_items, augment=True),
                               batch_size=2, shuffle=True, collate_fn=collate)

    model = build_model().to(DEVICE)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=5e-3, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

    EPOCHS = int(os.environ.get("EPOCHS", "6"))
    THR = float(os.environ.get("VAL_SCORE_THR", "0.10"))
    history = {"train_loss": [], "val_metrics": []}
    best = -1.0

    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for imgs, tgts in tr_loader:
            imgs = [i.to(DEVICE) for i in imgs]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
            losses = model(imgs, tgts)
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss.item())
        sched.step()

        avg = run_loss / max(len(tr_loader), 1)
        m = evaluate(model, VeinDataset(va_items), score_thr=THR)
        history["train_loss"].append(avg)
        history["val_metrics"].append({k: v for k, v in m.items() if k != "pr_curve"})
        print(f"epoch {ep+1}/{EPOCHS}  loss={avg:.4f}  val_f1={m['box_f1']:.3f}  "
              f"val_AP@0.5={m['AP@0.5']:.3f}  ({time.time()-t0:.1f}s)")
        if m["box_f1"] > best:
            best = m["box_f1"]
            torch.save(model.state_dict(), MODELS / "fasterrcnn_best.pt")

    # final test
    model.load_state_dict(torch.load(MODELS / "fasterrcnn_best.pt", map_location=DEVICE))
    test = evaluate(model, VeinDataset(te_items), score_thr=THR)
    test_iou03 = evaluate(model, VeinDataset(te_items), score_thr=THR, iou_thr=0.3)

    print("\nTEST")
    for k, v in test.items():
        if k != "pr_curve":
            print(f"  {k}: {v}")

    json.dump({
        "history": history,
        "test": {k: v for k, v in test.items() if k != "pr_curve"},
        "test_iou03": {k: v for k, v in test_iou03.items() if k != "pr_curve"},
        "test_pr_curve": test["pr_curve"],
        "score_thr": THR,
    }, open(OUT / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
