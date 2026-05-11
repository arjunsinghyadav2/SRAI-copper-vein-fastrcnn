# Generate bbox pseudo-labels from Frangi vesselness on the rock surface.
# Output: COCO-style JSON in data/annotations/.
#
# TODO: try anisotropic ridge filters (Sato) and compare
# TODO: small veins still get dropped by the area filter — revisit min_area

import json
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi, meijering
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
MASKS = ROOT / "data" / "masks"
ANN = ROOT / "data" / "annotations"

MIN_AREA = 200       # px^2 — anything smaller is noise
MIN_DIM = 14         # px   — drop tiny near-square specks


def isolate_rock(bgr):
    """Return uint8 mask (255 inside rock, 0 outside).

    The desk is smooth, the rock is textured. Local std-dev + Otsu, then
    pick the connected component with the best area*centrality score.
    """
    h, w = bgr.shape[:2]
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    blur = cv2.GaussianBlur(L, (0, 0), 9)
    sq = cv2.GaussianBlur(L * L, (0, 0), 9)
    tex = np.sqrt(np.clip(sq - blur * blur, 0, None))
    tex = (tex / max(tex.max(), 1e-6) * 255).astype(np.uint8)

    _, m = cv2.threshold(tex, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)

    cx, cy = w / 2, h / 2
    diag = math.hypot(cx, cy)
    best, best_score = 0, -1.0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 0.02 * h * w:
            continue
        ccx, ccy = cents[i]
        d = math.hypot(ccx - cx, ccy - cy) / diag
        score = area * (1 - 0.5 * d)
        if score > best_score:
            best, best_score = i, score
    if best == 0:
        return np.zeros_like(m)

    out = (labels == best).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    return cv2.erode(out, np.ones((11, 11), np.uint8), iterations=1)


def vein_mask(bgr, rock):
    if rock.sum() == 0:
        return np.zeros(bgr.shape[:2], np.uint8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    Lf = lab[:, :, 0].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # multiscale ridge response
    R = np.maximum.reduce([
        frangi(gray, sigmas=range(1, 5), black_ridges=False),
        frangi(Lf, sigmas=range(1, 5), black_ridges=False),
        meijering(gray, sigmas=range(1, 4), black_ridges=False),
    ])
    inside = rock > 0
    if R[inside].max() <= 0:
        return np.zeros_like(rock)
    R = R / (R[inside].max() + 1e-8)

    # color gates: brass-yellow chalcopyrite, bright quartz, locally bright pixels
    chalco = (H >= 10) & (H <= 45) & (S >= 30) & (V >= 60)
    quartz = (S <= 80) & (V >= 160)
    bright_local = Lf > np.percentile(Lf[inside], 92)
    color_gate = (chalco | quartz | bright_local) & inside

    mu, sd = R[inside].mean(), R[inside].std()
    path_a = (R > mu + 1.0 * sd) & color_gate     # color-aware
    path_b = (R > mu + 2.5 * sd) & inside         # strong ridges, any color
    v = ((path_a | path_b).astype(np.uint8)) * 255

    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return v


def boxes_from_mask(mask):
    boxes = []
    for r in regionprops(label(mask > 0)):
        if r.area < MIN_AREA:
            continue
        y1, x1, y2, x2 = r.bbox
        h, w = y2 - y1, x2 - x1
        if max(h, w) < MIN_DIM:
            continue
        if max(h, w) < 30 and min(h, w) / max(h, w) > 0.7:
            # near-square small blob → likely noise
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def main():
    MASKS.mkdir(parents=True, exist_ok=True)
    ANN.mkdir(parents=True, exist_ok=True)

    frames = sorted(FRAMES.glob("frame_*.jpg"))
    print(f"annotating {len(frames)} frames")

    images, anns = [], []
    aid = 1
    for img_id, fp in enumerate(frames, start=1):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rock = isolate_rock(bgr)
        v = vein_mask(bgr, rock)
        cv2.imwrite(str(MASKS / fp.name.replace(".jpg", "_mask.png")), v)
        bs = boxes_from_mask(v)

        images.append({"id": img_id, "file_name": fp.name, "width": w, "height": h})
        for x1, y1, x2, y2 in bs:
            anns.append({
                "id": aid, "image_id": img_id, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            aid += 1
        if img_id % 25 == 0:
            print(f"  {img_id}/{len(frames)}  total_boxes={aid-1}")

    out = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 1, "name": "copper_vein"}],
    }
    with open(ANN / "auto_coco.json", "w") as f:
        json.dump(out, f)
    print(f"done. {len(images)} images, {len(anns)} boxes")


if __name__ == "__main__":
    main()
