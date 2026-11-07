"""Presentation only: render unchanged saved counts and online timings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDER = Path(__file__).resolve().parent
METHODS = ("direct", "joint_no_extra", "uniform", "appearance", "geometry", "active")
LABELS = ("Direct", "Joint / no extra", "Uniform +32", "Appearance +32", "Geometry +32", "Active +32")


def main():
    result = json.loads((FOLDER/"results.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#213f5b", "text.color": "#213f5b", "savefig.dpi": 180})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), gridspec_kw={"width_ratios": [1.1, 1]}, layout="constrained")
    y = np.arange(len(METHODS))
    counts = [result["methods"][m]["correct"] for m in METHODS]
    wrong = [result["methods"][m]["wrong_positive"] for m in METHODS]
    axes[0].barh(y, counts, color="#087f8c", label="Correct raw pose")
    axes[0].barh(y, wrong, left=counts, color="#c25845", label="Wrong raw pose")
    axes[0].barh(y, 4-np.array(counts)-wrong, left=np.array(counts)+wrong,
                 color="#e5eaf0", label="No raw pose")
    axes[0].set(yticks=y, yticklabels=LABELS, xticks=range(5), xlim=(0, 4),
                xlabel="All four eligible positive pairs", title="Every method recovers the same one pose")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", bbox_to_anchor=(-.03, -.22), fontsize=8, frameon=False)
    times = [result["methods"][m]["median_positive_online_seconds"] for m in METHODS]
    axes[1].barh(y, times, color=["#087f8c" if m == "active" else "#213f5b" for m in METHODS])
    for i, value in enumerate(times):
        axes[1].text(value+.45, i, f"{value:.2f} s", va="center", fontsize=9)
    axes[1].set(yticks=y, yticklabels=[], xlim=(0, max(times)+8), xlabel="Median online seconds / positive pair",
                title="Active selection adds cost without recovery")
    axes[1].invert_yaxis()
    figure.suptitle("Frozen unknown-light comparison | four positives + four nonoverlaps", fontsize=13)
    figure.text(.5, -.065, "Three exposed positive terrain groups; discrete 24-light catalogue. Raw poses are uncalibrated.\n"
                "All methods return no pose on the four nonoverlaps; this does not establish a 1% risk.",
                ha="center", fontsize=8, color="#526579")
    figure.savefig(FOLDER/"comparison.png", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
