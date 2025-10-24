import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def plot_severity_curves(per_corruption, title="Severity Curves"):
    severities = [1, 2, 3, 4, 5]
    xs = np.array(severities)
    plt.figure(figsize=(10, 6))
    for cname, sev_map in per_corruption.items():
        ys = [sev_map.get(str(s), np.nan) for s in severities]
        ys = np.array(ys, dtype=float)
        plt.plot(xs, ys, marker="o", label=cname)
    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()


def plot_per_corruption_bars(severity_auc, title="AUC by Corruption"):
    names = list(severity_auc.keys())
    vals = [severity_auc[n] for n in names]
    idx = np.arange(len(names))
    plt.figure(figsize=(12, 5))
    plt.bar(idx, vals)
    plt.xticks(idx, names, rotation=45, ha="right")
    plt.ylabel("AUC")
    plt.title(title)
    plt.tight_layout()


def load_results_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_current_fig(out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot robustness results")
    ap.add_argument("--results", type=str, nargs="+", help="paths to *_robustness.json files")
    ap.add_argument("--out_dir", type=str, default="./results/plots")
    ap.add_argument("--show", action="store_true", help="show figures interactively")
    args = ap.parse_args()

    for res_path in args.results:
        data = load_results_json(res_path)
        model = data.get("model", "model")
        probe = data.get("probe", "probe")
        tag = f"{model}_{probe}"
        title_base = f"{model} [{probe}]"

        plot_severity_curves(data.get("per_corruption", {}), title=f"{title_base} – Severity Curves")
        curves_path = os.path.join(args.out_dir, tag, "severity_curves.png")
        _save_current_fig(curves_path)

        plot_per_corruption_bars(data.get("severity_auc", {}), title=f"{title_base} – AUC")
        auc_path = os.path.join(args.out_dir, tag, "severity_auc.png")
        _save_current_fig(auc_path)

        if args.show:
            plot_severity_curves(data.get("per_corruption", {}), title=f"{title_base} – Severity Curves")
            plot_per_corruption_bars(data.get("severity_auc", {}), title=f"{title_base} – AUC")
            plt.show()


if __name__ == "__main__":
    main()

