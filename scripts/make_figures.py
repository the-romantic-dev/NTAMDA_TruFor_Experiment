"""Rebuild tables, baseline comparison plots and auditable example selections from raw outputs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import evaluate, read_json, write_json


def format_metric(value):
    return "undefined" if value is None or value == "" else f"{float(value):.4f}"


def panel(path, destination, title):
    with np.load(path, allow_pickle=False) as data:
        images = [data["rgb"], data["gt"], data["np++"], data["map"], data["conf"], data["map"] > 0.5]
        labels = ["RGB", "Ground truth", "Noiseprint++", "Anomaly", "Confidence", "Prediction > 0.5"]
        fig, axes = plt.subplots(1, 6, figsize=(16, 3.5))
        for i, (ax, image, label) in enumerate(zip(axes, images, labels)):
            if i == 0:
                ax.imshow(image)
            elif i == 2:
                low, high = np.percentile(image, [1, 99])
                ax.imshow(image, cmap="gray", vmin=low, vmax=max(high, low + 1e-8))
            else:
                ax.imshow(image, cmap="gray" if i in (1, 5) else "viridis", vmin=0, vmax=1)
            ax.set_title(label, fontsize=11)
            ax.axis("off")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(destination, dpi=150)
        plt.close(fig)


def make_figures(output, sanity_count=20):
    output = Path(output)
    summaries = evaluate(output)
    metadata = read_json(output / "metadata.json")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    baseline = summaries[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - .18, [.720, .523], .36, label="Paper")
    values = [baseline["f1_best"], baseline["f1_fixed"]]
    ax.bar(x + .18, [v if v is not None else np.nan for v in values], .36, label="Reproduction")
    ax.set_xticks(x, ["F1 best", "F1 fixed (0.5)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean pixel-level F1")
    ax.set_title(f"CocoGlide: {len(metadata['ids'])}/512 images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "paper_comparison.png", dpi=180)
    plt.close(fig)
    with (output / "metrics/per_image.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["status"] == "ok"]
    clean = [r for r in rows if r["condition"] == "clean"]
    selections = []
    if clean:
        for group, metric, largest in [("successful", "f1_direct", True),
                                       ("false_positives", "false_positive_rate", True),
                                       ("missed_manipulations", "miss_rate", True),
                                       ("boundary_errors", "boundary_error", True),
                                       ("low_confidence", "mean_confidence", False)]:
            eligible = [r for r in clean if r.get(metric)]
            if not eligible:
                continue
            selected = sorted(eligible, key=lambda r: (float(r[metric]), r["id"]), reverse=largest)[0]
            selections.append({"group": group, "criterion": metric, "value": float(selected[metric]), **selected})
    write_json(figures / "selection.json", selections)
    notes = ["# Qualitative evidence", "", "Examples selected automatically using the stated metrics; they may overlap.",
             "These labels denote ranked examples, not guaranteed error categories.",
             "Binary maps use direct orientation at 0.5; official F1 can choose inverse orientation.",
             "Noiseprint++ alone uses per-image 1st/99th percentile display scaling; raw values are retained.", ""]
    for selected in selections:
        name = selected["group"] + ".png"
        panel(output / "raw" / selected["condition"] / (selected["id"] + ".npz"), figures / name,
              f"{selected['group']}: {selected['condition']} / {selected['id']}")
        notes.extend([f"## {selected['group']}",
                      f"Sample `{selected['id']}`, condition `{selected['condition']}`. "
                      f"Selection: {selected['criterion']}={float(selected['value']):.4f}. "
                      f"Direct F1={format_metric(selected['f1_direct'])}, official F1={format_metric(selected['f1_fixed'])}, "
                      f"false-positive rate={format_metric(selected['false_positive_rate'])}, "
                      f"miss rate={format_metric(selected['miss_rate'])}. "
                      "These measurements describe the error; they do not establish its internal cause.",
                      f"![{selected['group']}]({name})", ""])
    (figures / "qualitative_analysis.md").write_text("\n".join(notes), encoding="utf-8")
    sanity = output / "sanity"
    sanity.mkdir(exist_ok=True)
    for sample_id in metadata["ids"][:sanity_count]:
        condition = "clean"
        path = output / "raw" / condition / (sample_id + ".npz")
        panel(path, sanity / (sample_id + ".png"), sample_id)
    lines = ["# Experimental results", "", f"Dataset: CocoGlide; requested images: {len(metadata['ids'])}.",
             "Full published benchmark comparison requires all 512 forged images.", "",
             "| Condition | Requested | F1 best (n) | F1 fixed (n) | Pixel AUC (n) |",
             "|---|---:|---:|---:|---:|"]
    fmt = lambda v: "undefined" if v is None else f"{v:.4f}"
    for s in summaries:
        lines.append(f"| {s['condition']} | {s['n_requested']} | {fmt(s['f1_best'])} ({s['n_f1_best']}) | {fmt(s['f1_fixed'])} ({s['n_f1_fixed']}) | {fmt(s['pixel_auc'])} ({s['n_pixel_auc']}) |")
    comparison_path = output / "metrics/paper_comparison.csv"
    if comparison_path.exists():
        lines.extend(["", "## Comparison with CVPR 2023 Table 1", "",
                      "| Metric | Paper | Reproduction | Absolute delta |", "|---|---:|---:|---:|"])
        with comparison_path.open(encoding="utf-8", newline="") as stream:
            for r in csv.DictReader(stream):
                lines.append(f"| {r['metric']} | {r['paper'] or 'not reported'} | {r['reproduction']} | {r['absolute_delta'] or 'N/A'} |")
    lines.extend(["", "Pixel AUC is an additional localization metric, not the paper's image-level AUC.",
                  "See metrics/undefined.json for per-metric undefined cases; means use finite values only.",
                  "", "[Qualitative analysis](figures/qualitative_analysis.md)"])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sanity-count", type=int, default=20)
    args = parser.parse_args()
    make_figures(args.output, args.sanity_count)
