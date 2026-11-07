"""Render descriptive development outcomes. Saved data only; no fitted policy."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"direct": "Direct", "legacy_no_extra": "Old single / no extra", "legacy_active32": "Old active +32",
          "single_uniform128": "Single / up to 128", "beam_uniform128": "Beam / uniform",
          "beam_geometry128": "Beam / geometry", "beam_active128": "Beam / active"}


def main():
    data = json.loads((HERE / "results.json").read_bytes())
    methods = list(data["methods"])
    names = [LABELS[m] for m in methods]
    colors = ["#8a9ba7"]*4 + ["#4d8caa", "#4d8caa", "#0e766e"]
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    a, b, c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    counts = [data["methods"][m]["correct"] for m in methods]
    a.barh(names, counts, color=colors)
    for i, count in enumerate(counts):
        a.text(count+.07, i, f"{count}/4", va="center")
    a.set(xlim=(0, 4.7), xticks=range(5), xlabel="Correct raw poses / all four positives",
          title="Pose recovery (uncalibrated)")
    a.invert_yaxis()
    times = [data["methods"][m]["median_positive_online_seconds"] for m in methods]
    b.barh(names, times, color=colors)
    for i, value in enumerate(times):
        b.text(value+max(times)*.015, i, f"{value:.1f}s", va="center")
    b.set(xlim=(0, max(times)*1.24), xlabel="Median positive-pair seconds", title="Same local CPU, two-thread cap")
    b.invert_yaxis()
    rows = data["case_results"]
    # 0=no raw pose; 1=correct positive; 2=wrong positive/nonoverlap pose.
    matrix = np.array([[1 if r["methods"][m]["correct"] else
                        2 if r["methods"][m]["pose_returned"] else 0 for r in rows] for m in methods])
    palette = matplotlib.colors.ListedColormap(["#e7ecef", "#65b8a6", "#e4a38d"])
    c.imshow(matrix, cmap=palette, vmin=0, vmax=2, aspect="auto")
    for y, m in enumerate(methods):
        for x, r in enumerate(rows):
            v = matrix[y, x]
            label = "correct" if v == 1 else "wrong" if v == 2 else "no pose"
            if not r["positive"] and v == 0:
                label = "no pose"
            c.text(x, y, label, ha="center", va="center", fontsize=8)
    c.set(yticks=range(len(methods)), yticklabels=names, xticks=range(len(rows)),
          xticklabels=[r["id"].removeprefix("blind138-") + (" +" if r["positive"] else " −") for r in rows],
          title="Every eligible case (+ positive, − nonoverlap); no calibrated acceptance")
    fig.suptitle("Persistent camera/light hypotheses: frozen eight-case development study\n"
                 "Four positive pairs on three exposed terrain components; no final-test inference", fontsize=14)
    fig.savefig(HERE / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
