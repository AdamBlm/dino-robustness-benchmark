from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean() * 100.0)


def per_group_accuracy(
    y_true,
    y_pred,
    corruption_ids,
    severities,
    num_corruptions,
    severity_levels=[1, 2, 3, 4, 5],
):
    results = {}
    for cid in range(num_corruptions):
        cid_key = str(cid)
        results[cid_key] = {}
        for s in severity_levels:
            mask = (corruption_ids == cid) & (severities == s)
            if mask.any():
                acc = accuracy(y_true[mask], y_pred[mask])
            else:
                acc = float("nan")
            results[cid_key][str(s)] = round(acc, 3) if not np.isnan(acc) else float("nan")
    return results


def mean_corruption_accuracy(per_group):
    vals = []
    for cid in per_group.values():
        for acc in cid.values():
            if not (isinstance(acc, float) and np.isnan(acc)):
                vals.append(acc)
    return round(float(np.mean(vals)) if len(vals) > 0 else float("nan"), 3)


def relative_robustness(c10c_acc, clean_acc):
    if clean_acc <= 0:
        return float("nan")
    return round(float(c10c_acc / clean_acc), 4)


def severity_auc(
    per_group,
    severity_levels=[1, 2, 3, 4, 5],
):
    aucs = {}
    xs = np.array(severity_levels, dtype=float)
    for cid_str, sev_map in per_group.items():
        ys = []
        for s in severity_levels:
            v = sev_map.get(str(s), float("nan"))
            ys.append(np.nan if (isinstance(v, float) and np.isnan(v)) else float(v))
        ys_arr = np.array(ys, dtype=float)
        if np.all(np.isnan(ys_arr)):
            aucs[cid_str] = float("nan")
        else:
            if np.any(np.isnan(ys_arr)):
                mean_val = np.nanmean(ys_arr)
                ys_arr = np.where(np.isnan(ys_arr), mean_val, ys_arr)
            aucs[cid_str] = float(np.trapz(ys_arr, xs))
    return aucs


