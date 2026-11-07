"""Presentation-only figure from frozen scores and the saved-only diagnostic."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
result = json.loads((HERE/"results.json").read_text())
diagnostic = json.loads((HERE/"saved-diagnostic.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
names = ["cached_framed_direct", "hard", "joint", "joint_camera_proposal"]
labels = ["Direct\n(cached)", "Hard peaks\n(fresh control)", "Joint + AP3P\n(primary)", "Soft camera\n(proposal only)"]
correct = [result["methods"][name]["correct"] for name in names]
wrong = [result["methods"][name]["wrong_positive"] for name in names]
missing = [4-a-b for a, b in zip(correct, wrong, strict=True)]
positions = np.arange(len(names))
axes[0].bar(positions, correct, color="#277c65", label="Correct raw pose")
axes[0].bar(positions, wrong, bottom=correct, color="#bd5546", label="Wrong raw pose")
axes[0].bar(positions, missing, bottom=np.array(correct)+wrong, color="#e0e4e8", label="No pose")
axes[0].set(xticks=positions, xticklabels=labels, yticks=range(5), ylim=(0, 4.35), ylabel="Positive pairs (all four retained)")
for x, a, b in zip(positions, correct, wrong, strict=True):
    axes[0].text(x, 4.08, f"{a} correct / {b} wrong", ha="center", fontsize=8)
axes[0].set_title("Primary recovery ties Direct; proposal differs", loc="left", fontweight="bold")
axes[0].legend(loc="lower left", bbox_to_anchor=(-.02, -.33), ncols=3, frameon=False, fontsize=8)
faint = next(row for row in diagnostic["rows"] if row["reference_index"] == 0)
error_names = ["cached_camera_proposal", "hard", "joint_camera_proposal"]
errors = [faint["methods"][name]["pose_error"]["translation_m"] for name in error_names]
axes[1].bar(range(3), errors, color=["#999fa7", "#bd5546", "#437ca3"])
limit = .01*faint["methods"]["hard"]["pose_error"]["reference_range_m"]
axes[1].axhline(limit, color="#292d34", ls="--", lw=1, label=f"Pose criterion: {limit:.2f} m")
for x, value in enumerate(errors):
    axes[1].text(x, value+.5, f"{value:.2f} m", ha="center")
axes[1].set(xticks=range(3), xticklabels=["Initial\ncamera", "Hard-peak\nAP3P", "Soft camera\nproposal"], ylim=(0, 33), ylabel="Translation error (metres)")
axes[1].set_title("Faint case: better camera, only five matches", loc="left", fontweight="bold")
axes[1].legend(frameon=False, fontsize=9)
fig.suptitle("Known-light development pilot · 4 conditions / 3 terrains · 4 nonoverlaps give no poses", y=1.015, fontsize=11)
fig.text(.02, -.06, "Camera-conditioned matches are not independent landmarks. No calibrated acceptance, new-terrain result or complete-runtime comparison.", fontsize=9)
fig.tight_layout()
fig.savefig(HERE/"comparison.png", dpi=180, bbox_inches="tight")
