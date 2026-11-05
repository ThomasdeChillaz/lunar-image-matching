"""Presentation-only rendering of independently replayed native study results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def render():
    result = json.loads((HERE/"results.json").read_text())
    names = ["framed_pooled_parallax", "framed_shared_camera_pooled_parallax", "framed_direct",
             "framed_native_plane_integer", "framed_native_plane_subpixel", "framed_native_terrain_subpixel"]
    labels = ["Pooled patch\ncontrol", "Shared camera\n+ old patches", "Direct\nreference library",
              "Native radiance\ninteger peaks", "Native radiance\nsubpixel peaks", "3D terrain\nsubpixel peaks"]
    rows = [result["methods"][name] for name in names]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2))
    fig.subplots_adjust(left=.09, right=.97, top=.88, bottom=.17, hspace=.58)
    fig.suptitle("Native camera localization: matched development comparison", fontsize=15, y=.97)
    axes[0].bar(x-.17, [r["correct"] for r in rows], .34, color="#2169a5", label="Correct raw pose")
    axes[0].bar(x+.17, [r["strict"] for r in rows], .34, color="#e6a33e", label="Strict raw pose")
    for i, row in enumerate(rows):
        for offset, key in ((-.17, "correct"), (.17, "strict")):
            axes[0].text(i+offset, row[key]+.3, str(row[key]), ha="center", fontsize=10)
    axes[0].set_ylim(0, 17)
    axes[0].set_yticks([0, 4, 8, 12, 16])
    axes[0].set_ylabel("Correct camera poses /16")
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=10)
    times = [r["median_positive_inference_seconds"] for r in rows]
    axes[1].bar(x, times, .55, color=["#8296a7", "#8296a7", "#398970", "#2169a5", "#2169a5", "#2169a5"])
    for i, value in enumerate(times):
        axes[1].text(i, value+max(times)*.025, f"{value:.2f}s", ha="center", fontsize=10)
    axes[1].set_ylim(0, max(times)*1.23)
    axes[1].set_ylabel("Median positive inference (s)")
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=9)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.09, .075, "All methods use the same reference-camera AP3P frame, three-pixel threshold and native 512px inputs.", fontsize=9)
    fig.text(.09, .046, "Four exposed camera conditions / three terrain components; matching light supplied; sixteen positives incl. eight near-uniform cases.", fontsize=8.5)
    fig.text(.09, .02, "Raw poses are uncalibrated. Calibration and final terrain remain closed. Reference-cache workload; same CPU.", fontsize=9)
    fig.savefig(HERE/"comparison.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    render()
