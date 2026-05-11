# Generate the report figures from outputs/metrics.json + the saved arrays.
#
# TODO: add a per-sigma ablation chart for Frangi scales

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def pipeline():
    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.8); ax.axis("off")
    boxes = [
        (0.2, "Video\nframes", "#dde3ee"),
        (1.9, "Rock\nisolation\n(LAB texture\n+ centre prior)", "#cfe3d2"),
        (3.7, "Frangi /\nMeijering\nvesselness", "#f6e2c4"),
        (5.5, "Color-gated\nfusion\n(chalcopyrite\n+ quartz)", "#f6c4c4"),
        (7.3, "Connected\ncomp.\n→ bboxes", "#e5d4f5"),
        (9.0, "Faster R-CNN\nMobileNetV3 FPN", "#c4dff6"),
    ]
    for x, lbl, c in boxes:
        ax.add_patch(plt.Rectangle((x - 0.08, 0.6), 1.5, 1.6, fc=c, ec="#222"))
        ax.text(x + 0.66, 1.4, lbl, ha="center", va="center", fontsize=8.5)
    for x in [1.42, 3.22, 5.02, 6.82, 8.62]:
        ax.annotate("", xy=(x + 0.55, 1.4), xytext=(x, 1.4),
                    arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(5.0, 0.15, "Frangi pseudo-labels  →  supervised detector",
            ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(OUT / "pipeline.pdf", bbox_inches="tight")
    plt.savefig(OUT / "pipeline.png", dpi=180, bbox_inches="tight")
    plt.close()


def train_curves(metrics):
    h = metrics["history"]
    ep = list(range(1, len(h["train_loss"]) + 1))
    fig, a1 = plt.subplots(figsize=(5.0, 3.0))
    a1.plot(ep, h["train_loss"], "o-", color="#cf3030", label="train loss")
    a1.set_xlabel("epoch"); a1.set_ylabel("train loss", color="#cf3030")
    a1.tick_params(axis="y", labelcolor="#cf3030")
    a1.set_xticks(ep)
    a2 = a1.twinx()
    a2.plot(ep, [m["box_f1"] for m in h["val_metrics"]], "s-",
            color="#1d4f9b", label="val F1@0.5")
    a2.plot(ep, [m["AP@0.5"] for m in h["val_metrics"]], "^--",
            color="#2a8a3a", label="val AP@0.5")
    a2.set_ylabel("validation score", color="#1d4f9b")
    a2.tick_params(axis="y", labelcolor="#1d4f9b")
    a2.set_ylim(0, 1)
    fig.legend(loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "train_curves.pdf", bbox_inches="tight")
    plt.savefig(OUT / "train_curves.png", dpi=180, bbox_inches="tight")
    plt.close()


def pr_curve(metrics):
    pts = metrics["test_pr_curve"]
    rs = [r for r, _ in pts]; ps = [p for _, p in pts]
    plt.figure(figsize=(4.6, 3.2))
    if rs:
        plt.plot(rs, ps, "-", color="#1d4f9b", lw=1.6)
        plt.fill_between(rs, 0, ps, alpha=0.12, color="#1d4f9b")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim(0, 1); plt.ylim(0, 1.02); plt.grid(alpha=0.3)
    plt.title(f"Test PR @ IoU=0.5  (AP={metrics['test']['AP@0.5']:.3f})")
    plt.tight_layout()
    plt.savefig(OUT / "pr_curve.pdf", bbox_inches="tight")
    plt.savefig(OUT / "pr_curve.png", dpi=180, bbox_inches="tight")
    plt.close()


def main():
    metrics = json.load(open(ROOT / "outputs" / "metrics.json"))
    pipeline()
    train_curves(metrics)
    pr_curve(metrics)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
