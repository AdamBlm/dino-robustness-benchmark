import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_fig(path):
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_bars(results, metric_key, title, out_path):
    names = [f"{r.get('model','model')}\n[{r.get('probe','probe')}]" for r in results]
    vals = [float(r.get(metric_key, np.nan)) for r in results]
    idx = np.arange(len(names))
    plt.figure(figsize=(10, 4))
    plt.bar(idx, vals)
    plt.xticks(idx, names, rotation=30, ha="right")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    save_fig(out_path)


def compute_mean_severity_curve(per_corr):
    severities = [1, 2, 3, 4, 5]
    means = []
    for s in severities:
        vals = []
        for _, sev_map in per_corr.items():
            v = sev_map.get(str(s), np.nan)
            if not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
    return means


def plot_mean_severity_curves(results, out_path):
    xs = np.array([1, 2, 3, 4, 5])
    plt.figure(figsize=(8, 5))
    for r in results:
        means = compute_mean_severity_curve(r.get("per_corruption", {}))
        label = f"{r.get('model','model')} [{r.get('probe','probe')}]"
        plt.plot(xs, means, marker="o", label=label)
    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.title("Mean across corruptions vs severity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(out_path)


def plot_auc_by_corruption_grouped(results, out_path):
    all_names = []
    for r in results:
        aucs = r.get("severity_auc", {})
        for k in aucs.keys():
            if k not in all_names:
                all_names.append(k)
    all_names = sorted(all_names)

    n_models = len(results)
    n = len(all_names)
    x = np.arange(n)
    width = 0.8 / max(n_models, 1)

    plt.figure(figsize=(max(10, n * 0.6), 5))
    for i, r in enumerate(results):
        aucs = r.get("severity_auc", {})
        vals = [float(aucs.get(name, np.nan)) for name in all_names]
        plt.bar(x + i * width, vals, width=width, label=f"{r.get('model')} [{r.get('probe')}]")
    plt.xticks(x + width * (n_models - 1) / 2, all_names, rotation=45, ha="right")
    plt.ylabel("AUC")
    plt.title("Severity AUC by corruption (grouped by model)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    save_fig(out_path)


def main():
    ap = argparse.ArgumentParser(description="Cross-model robustness comparison plots")
    ap.add_argument("--results", type=str, nargs="+", help="paths to *_robustness.json files")
    ap.add_argument("--out_dir", type=str, default="./results/plots_compare")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    results = [load_json(p) for p in args.results]

    plot_metric_bars(results, "clean_acc", "Clean accuracy (Top-1)", os.path.join(args.out_dir, "clean_acc.png"))
    plot_metric_bars(results, "cifar10c_acc", "CIFAR-10-C overall accuracy (Top-1)", os.path.join(args.out_dir, "cifar10c_acc.png"))
    plot_metric_bars(results, "mCA", "Mean Corruption Accuracy (mCA)", os.path.join(args.out_dir, "mCA.png"))
    plot_metric_bars(results, "RR", "Relative Robustness (RR)", os.path.join(args.out_dir, "RR.png"))

    plot_mean_severity_curves(results, os.path.join(args.out_dir, "mean_severity_curves.png"))

    plot_auc_by_corruption_grouped(results, os.path.join(args.out_dir, "auc_by_corruption.png"))


if __name__ == "__main__":
    main()


