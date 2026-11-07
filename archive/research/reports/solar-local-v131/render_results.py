"""Presentation-only figure from the independently replayed principal summary."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/"results.json").read_text())
keys=["pooled_parallax","shared_camera_pooled_parallax","direct"]
labels=["Local patches","Shared camera","Direct"]
colors=["#83939e","#337b9b","#c58735"]
fig,axes=plt.subplots(1,3,figsize=(12.3,4.8))
fig.subplots_adjust(left=.07,right=.985,bottom=.24,top=.79,wspace=.34)
fig.set_facecolor("#fafbf9")
for ax in axes:
    ax.set_facecolor("#fafbf9");ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=9);ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
values=[r["methods"][k]["correct"] for k in keys]
bars=axes[0].bar(labels,values,color=colors,width=.62)
axes[0].bar_label(bars,labels=[f"{x}/16" for x in values],padding=4,fontweight="bold")
axes[0].set_ylim(0,16.5);axes[0].set_title("Correct camera poses",loc="left",fontweight="bold",fontsize=11)
axes[0].set_ylabel("Raw poses, all 16 eligible positives")
for i,k in enumerate(keys):
    if values[i]:axes[0].text(i,.5,f'{r["methods"][k]["strict"]} strict',ha="center",fontsize=9,color="white")
precision=[100*r["methods"][k]["native_two_pixel_matches"]/r["methods"][k]["retained_positive_matches"] for k in keys]
bars=axes[1].bar(labels,precision,color=colors,width=.62)
axes[1].bar_label(bars,labels=[f"{x:.1f}%" for x in precision],padding=4,fontweight="bold")
axes[1].set_ylim(0,118);axes[1].set_title("Returned-match precision",loc="left",fontweight="bold",fontsize=11)
axes[1].set_ylabel("Within 2 original pixels (%)")
seconds=[r["methods"][k]["median_positive_inference_seconds"] for k in keys]
bars=axes[2].bar(labels,seconds,color=colors,width=.62)
axes[2].bar_label(bars,labels=[f"{x:.2f} s" for x in seconds],padding=4,fontweight="bold")
axes[2].set_ylim(0,10.5);axes[2].set_title("Median inference time",loc="left",fontweight="bold",fontsize=11)
axes[2].set_ylabel("Seconds on the same CPU workload")
fig.suptitle("Pose recovery restored; strongest baseline remains better",x=.06,y=.96,ha="left",fontsize=15,fontweight="bold")
fig.supxlabel("Known matching light · 16 noise pairings · 4 camera conditions · 3 terrain components\nDevelopment diagnostics; correlated matches and reference-cache timings; no calibrated risk claim.",y=.04,fontsize=9,color="#48545e")
fig.savefig(HERE/"comparison.png",dpi=180,bbox_inches="tight",pad_inches=.15)
plt.close(fig)
