# Pull frames out of the source video at 4 fps, scaled to 800 px wide.
# ffmpeg has to be on $PATH.

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = next(p for p in ROOT.glob("*.mp4") if "detection" not in p.name.lower())
OUT = ROOT / "data" / "frames"
OUT.mkdir(parents=True, exist_ok=True)

# TODO: dedupe near-duplicate frames (Laplacian variance > N) so we don't waste
# annotation effort on motion-blurred ones
subprocess.run([
    "ffmpeg", "-y", "-i", str(SRC),
    "-vf", "fps=4,scale=800:-1",
    "-q:v", "2",
    str(OUT / "frame_%04d.jpg"),
], check=True)
print(f"frames -> {OUT}")
