"""Plot independently replayed raw pose comparisons at the original thresholds."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIRECT = "reference_camera_render_match"
MATCH = "MatchAnything-EfficientLoFTR-native512"
SIFT = "SIFT2048"


def read(path):
    return json.loads(path.read_text())


def source_counts(root):
    paired = read(root/"reports/solar_full_paired_development_v94.json")
    withheld = read(root/"reports/solar_offcatalogue_direct_replay_v95.json")
    if (paired["complete"] is not True or paired["accepted_results"] is not False
            or withheld["passed"] is not True or withheld["accepted_results"] is not False
            or paired["positive_cases"] != 192 or len(paired["terrain_groups"]) != 8):
        raise ValueError("Require the independently replayed development cohorts")
    values = []
    for key in (DIRECT, MATCH, SIFT):
        strata = paired["method_results"][key]["strata"]
        values.append([strata[s]["correct"] for s in ("same_light_index","changed_light_index")])
    cuts = withheld["paired_by_query_index"]
    values.insert(1, [cuts[s]["even12"]["correct"] for s in ("even_query_index","odd_query_index")])
    if [cuts[s]["full24"]["correct"] for s in ("even_query_index","odd_query_index")] != values[0]:
        raise ValueError("The full-library controls must agree across reports")
    if [cuts[s]["even12"]["positives"] for s in ("even_query_index","odd_query_index")] != [64,128]:
        raise ValueError("Unexpected cohort size")
    vs = next(item for item in paired["paired_comparisons"] if item["first"]==DIRECT and item["second"]==MATCH)
    changed = vs["strata"]["changed_light_index"]
    interval_main = changed["percentile_2_5_50_97_5"]["raw_correct_pose_difference"]
    interval_withheld = cuts["odd_query_index"]["paired_terrain_percentiles"]["raw_correct_pose_difference"]
    effects = [(changed["correct_pose_delta"], interval_main[0],interval_main[2]),
               (cuts["odd_query_index"]["paired_correct_delta"],interval_withheld[0],interval_withheld[2])]
    return values, np.asarray(effects)*100


def plot(root, target):
    values, effects = source_counts(root)
    labels = ("Relighting, full library","Relighting, 12-light library","MatchAnything","SIFT")
    colors = ("#15796e","#69a998","#416599","#8a96a4")
    denom = np.array([64,128])
    fig, (bars, interval) = plt.subplots(1,2,figsize=(11.5,4.25),
                                         gridspec_kw={"width_ratios":[1.3,1]})
    fig.patch.set_facecolor("white")
    fig.text(.07,.93,"Lunar localization under changed illumination",size=18,weight="bold",color="#172635")
    fig.text(.07,.864,"192 positive simulated pairs across eight terrain groups. Raw poses before calibration.",
             size=10,color="#415063")
    for i,(label,color,counts) in enumerate(zip(labels,colors,values,strict=True)):
        heights = np.asarray(counts)/denom*100
        columns = bars.bar(np.arange(2)+(i-1.5)*.19,heights,.16,label=label,color=color,zorder=3)
        bars.bar_label(columns,labels=[str(n) for n in counts],padding=2,size=9,color="#26364b")
    bars.set_xticks([0,1],["Same light\n64 pairs","Changed light\n128 pairs"])
    bars.set_ylabel("Correct raw poses (%)")
    bars.set_ylim(0,81)
    bars.set_title("A   Pose recovery by lighting",loc="left",fontsize=11,weight="bold",pad=12)
    bars.legend(loc="upper left",frameon=False,fontsize=8,ncol=2,handlelength=1,handletextpad=.55)
    bars.grid(axis="y",color="#e0e7ef",zorder=0)
    interval.axvline(0,color="#8a96a4",linestyle="--",linewidth=1)
    for idx,((point,lo,hi),label,color) in enumerate(zip(
        effects,("Full library vs MatchAnything","Withheld library vs full library"),(colors[0],colors[1]),strict=True
    )):
        y = 1-idx
        interval.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt="o",color=color,
                          linewidth=2,markersize=6,capsize=4,zorder=3)
        interval.text(-56,y+.24,label,color="#172635",size=9,weight="semibold")
        interval.text(point,y-.27,f"{point:+.2f} pp",ha="center",color="#26364b",size=9)
    interval.set_xlim(-60,87)
    interval.set_ylim(-.48,1.5)
    interval.set_yticks([])
    interval.set_title("B   Changed-light paired differences",loc="left",fontsize=11,weight="bold",pad=12)
    interval.set_xlabel("Percentage points")
    interval.grid(axis="x",color="#e0e7ef",zorder=0)
    for axis in (bars,interval):
        for side in ("top","right","left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#b8c2cf")
        axis.tick_params(length=0,labelcolor="#26364b",labelsize=9)
    fig.text(.07,.028,"Bar labels show case counts. Whiskers show 95% paired terrain-group intervals.",
             color="#415063",size=9)
    fig.subplots_adjust(left=.075,right=.98,bottom=.205,top=.748,wspace=.32)
    fig.savefig(target,dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo",required=True,type=Path)
    cli.add_argument("--output",required=True,type=Path)
    args=cli.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous figure output")
    plot(args.repo,args.output)
