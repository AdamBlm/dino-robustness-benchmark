"""
k-NN evaluation on frozen features.
"""
from __future__ import annotations
import os
import json
import argparse
from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score


def load_split(in_dir: str, model: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    d = os.path.join(in_dir, model)
    X = np.load(os.path.join(d, f"{split}_X.npy"))
    y = np.load(os.path.join(d, f"{split}_y.npy"))
    return X, y


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def knn_predict(
    X_db: np.ndarray,
    y_db: np.ndarray,
    X_q: np.ndarray,
    k: int = 20,
    metric: str = "cosine",
    weighted: bool = True,
    batch_size: int = 8192,
) -> np.ndarray:
    """kNN prediction with cosine or euclidean distance."""
    # Build index
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn.fit(X_db)

    preds = []
    for i in range(0, X_q.shape[0], batch_size):
        Q = X_q[i:i + batch_size]
        dists, idxs = nn.kneighbors(Q, return_distance=True)

        neigh_labels = y_db[idxs]

        if weighted:
            if metric == "cosine":
                weights = 1.0 - dists
            else:
                weights = 1.0 / (dists + 1e-6)
        else:
            weights = np.ones_like(dists)

        # Vote per class
        batch_pred = []
        for labels_row, w_row in zip(neigh_labels, weights):
            c = int(labels_row.max()) + 1
            votes = np.bincount(labels_row.astype(int), weights=w_row, minlength=c)
            batch_pred.append(votes.argmax())
        preds.append(np.asarray(batch_pred, dtype=int))

    return np.concatenate(preds, axis=0)


def main():
    ap = argparse.ArgumentParser(description="k-NN eval on frozen features")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--in_dir", type=str, default="./features")
    ap.add_argument("--out_dir", type=str, default="./experiments")
    ap.add_argument("--k", type=int, nargs="*", default=[1, 5, 20])
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--normalize", action="store_true", help="apply L2 normalization to features")
    ap.add_argument("--use_val_db", action="store_true", help="use train+val as the database (stronger)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exp_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(exp_dir, exist_ok=True)

    # Load features
    Xtr, ytr = load_split(args.in_dir, args.model, "train")
    Xva, yva = load_split(args.in_dir, args.model, "val")
    Xte, yte = load_split(args.in_dir, args.model, "test")

    # Optional CIFAR-10-C
    c10c_path = os.path.join(args.in_dir, args.model, "cifar10c_X.npy")
    has_c10c = os.path.isfile(c10c_path)
    if has_c10c:
        Xc = np.load(c10c_path)
        yc = np.load(os.path.join(args.in_dir, args.model, "cifar10c_y.npy"))

    if args.use_val_db:
        Xdb = np.concatenate([Xtr, Xva], axis=0)
        ydb = np.concatenate([ytr, yva], axis=0)
    else:
        Xdb, ydb = Xtr, ytr

    if args.normalize:
        Xdb = l2_normalize(Xdb)
        Xte = l2_normalize(Xte)
        if has_c10c:
            Xc = l2_normalize(Xc)

    results = {
        "model": args.model,
        "metric": args.metric,
        "normalize": bool(args.normalize),
        "use_val_db": bool(args.use_val_db),
        "k_values": args.k,
        "test": {},
    }
    if has_c10c:
        results["cifar10c"] = {}

    for kval in args.k:
        yhat_test = knn_predict(Xdb, ydb, Xte, k=kval, metric=args.metric, weighted=True)
        acc_test = float(accuracy_score(yte, yhat_test) * 100.0)
        results["test"][str(kval)] = round(acc_test, 3)

        if has_c10c:
            yhat_c = knn_predict(Xdb, ydb, Xc, k=kval, metric=args.metric, weighted=True)
            acc_c = float(accuracy_score(yc, yhat_c) * 100.0)
            results["cifar10c"][str(kval)] = round(acc_c, 3)

    with open(os.path.join(exp_dir, "results_knn.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[eval_knn] Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
