import os
import json
import argparse

import numpy as np

from metrics import (
    per_group_accuracy,
    mean_corruption_accuracy,
    relative_robustness,
    severity_auc,
)


def load_cifar10c_features(in_dir, model):
    d = os.path.join(in_dir, model)
    X = np.load(os.path.join(d, "cifar10c_X.npy"))
    y = np.load(os.path.join(d, "cifar10c_y.npy"))
    cid = np.load(os.path.join(d, "cifar10c_corruption_ids.npy"))
    sev = np.load(os.path.join(d, "cifar10c_severities.npy"))
    meta_path = os.path.join(d, "cifar10c_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {"corruptions": []}
    return X, y, cid.astype(int), sev.astype(int), meta


def load_clean_predictions(clean_metrics_path):
    with open(clean_metrics_path, "r") as f:
        return json.load(f)


def load_splits(in_dir, model):
    d = os.path.join(in_dir, model)
    Xtr = np.load(os.path.join(d, "train_X.npy"))
    ytr = np.load(os.path.join(d, "train_y.npy"))
    Xva = np.load(os.path.join(d, "val_X.npy"))
    yva = np.load(os.path.join(d, "val_y.npy"))
    Xte = np.load(os.path.join(d, "test_X.npy"))
    yte = np.load(os.path.join(d, "test_y.npy"))
    return (Xtr, ytr), (Xva, yva), (Xte, yte)


def eval_linear(args, Xc, yc):
    from joblib import load
    clf = load(args.clf_path)
    yhat_c = clf.predict(Xc)
    return {
        "yhat_c": yhat_c,
    }


def eval_knn(args, Xc, yc, in_dir, model):
    from eval_knn import l2_normalize, knn_predict
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_splits(in_dir, model)
    if args.use_val_db:
        Xdb = np.concatenate([Xtr, Xva], axis=0)
        ydb = np.concatenate([ytr, yva], axis=0)
    else:
        Xdb, ydb = Xtr, ytr
    if args.normalize:
        Xdb = l2_normalize(Xdb)
        Xc = l2_normalize(Xc)
    yhat_c = knn_predict(Xdb, ydb, Xc, k=args.k, metric=args.metric, weighted=True)
    return {
        "yhat_c": yhat_c,
    }


def main():
    ap = argparse.ArgumentParser(description="Robustness evaluation on CIFAR-10-C")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--in_dir", type=str, default="./features")
    ap.add_argument("--out_dir", type=str, default="./results")
    ap.add_argument("--probe", type=str, default="linear", choices=["linear", "knn"])
    ap.add_argument("--clean_metrics", type=str, required=True, help="path to results_linear.json or results_knn.json")

    ap.add_argument("--clf_path", type=str, help="joblib for linear classifier")

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--use_val_db", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    Xc, yc, cids, sevs, meta = load_cifar10c_features(args.in_dir, args.model)
    num_corr = len(meta.get("corruptions", [])) if meta.get("corruptions") else int(cids.max()) + 1

    if args.probe == "linear":
        if not args.clf_path:
            raise ValueError("--clf_path is required for linear probe")
        out = eval_linear(args, Xc, yc)
    else:
        out = eval_knn(args, Xc, yc, args.in_dir, args.model)

    yhat_c = out["yhat_c"]

    per_grp = per_group_accuracy(yc, yhat_c, cids, sevs, num_corruptions=num_corr)
    mca = mean_corruption_accuracy(per_grp)
    aucs = severity_auc(per_grp)

    clean = load_clean_predictions(args.clean_metrics)
    if args.probe == "linear":
        clean_acc = float(clean.get("test_acc", float("nan")))
    else:
        kval = str(clean.get("k_values", [args.k])[-1])
        clean_acc = float(clean.get("test", {}).get(kval, float("nan")))
    c10c_overall = float((yc == yhat_c).mean() * 100.0)
    rr = relative_robustness(c10c_overall, clean_acc)

    names = meta.get("corruptions") or [f"corr_{i}" for i in range(num_corr)]

    per_group_named = {names[int(k)]: v for k, v in per_grp.items()}
    aucs_named = {names[int(k)]: v for k, v in aucs.items()}

    results = {
        "model": args.model,
        "probe": args.probe,
        "clean_acc": round(clean_acc, 3) if not np.isnan(clean_acc) else float("nan"),
        "cifar10c_acc": round(c10c_overall, 3),
        "mCA": mca,
        "RR": rr,
        "severity_auc": aucs_named,
        "per_corruption": per_group_named,
    }

    out_path = os.path.join(args.out_dir, f"{args.model}_{args.probe}_robustness.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("[eval_corruptions] Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


