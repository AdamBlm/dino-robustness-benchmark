import os
import json
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from joblib import dump


def load_split(in_dir, model, split):
    d = os.path.join(in_dir, model)
    X = np.load(os.path.join(d, f"{split}_X.npy"))
    y = np.load(os.path.join(d, f"{split}_y.npy"))
    return X, y


def evaluate(clf, X, y):
    yhat = clf.predict(X)
    acc = float(accuracy_score(y, yhat) * 100.0)
    return {"acc": acc, "yhat": yhat}


def main():
    ap = argparse.ArgumentParser(description="Train linear probe on frozen features")
    ap.add_argument("--model", type=str, required=True, help="model name used in features/<model>")
    ap.add_argument("--in_dir", type=str, default="./features")
    ap.add_argument("--out_dir", type=str, default="./experiments")
    ap.add_argument("--Cs", type=float, nargs="*", default=[0.01, 0.1, 1.0, 10.0, 100.0],
                    help="inverse regularization strengths to search")
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-3, help="solver tolerance")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", action="store_true", help="apply StandardScaler before LR")
    ap.add_argument("--refit_on_full", action="store_true",
                    help="refit best hyperparams on train+val before final test")
    ap.add_argument("--verbose", type=int, default=0, help="verbosity level for solver")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exp_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(exp_dir, exist_ok=True)

    Xtr, ytr = load_split(args.in_dir, args.model, "train")
    Xva, yva = load_split(args.in_dir, args.model, "val")
    Xte, yte = load_split(args.in_dir, args.model, "test")

    c10c_path = os.path.join(args.in_dir, args.model, "cifar10c_X.npy")
    has_c10c = os.path.isfile(c10c_path)
    if has_c10c:
        Xc = np.load(c10c_path)
        yc = np.load(os.path.join(args.in_dir, args.model, "cifar10c_y.npy"))

    def make_pipeline(C):
        steps = []
        if args.scale:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        steps.append(("logreg", LogisticRegression(
            penalty="l2",
            C=C,
            solver="saga",
            max_iter=args.max_iter,
            tol=args.tol,
            n_jobs=-1,
            random_state=args.seed,
            verbose=args.verbose,
        )))
        return Pipeline(steps)

    best = {"C": None, "val_acc": -1.0, "clf": None}
    for C in args.Cs:
        print(f"[train_linear] Fitting with C={C}...")
        clf = make_pipeline(C)
        clf.fit(Xtr, ytr)
        train_metrics = evaluate(clf, Xtr, ytr)
        val_metrics = evaluate(clf, Xva, yva)
        print(f"[train_linear]  => Train acc: {train_metrics['acc']:.3f}, Val acc: {val_metrics['acc']:.3f}")
        if val_metrics["acc"] > best["val_acc"]:
            print(f"[train_linear]  New best val acc: {val_metrics['acc']:.3f}. Saving checkpoint.")
            best.update({"C": C, "val_acc": val_metrics['acc'], "clf": clf})
            dump(clf, os.path.join(exp_dir, "linear_clf.joblib"))

    print(f"[train_linear] Best C={best['C']} with val acc {best['val_acc']:.3f}")

    if args.refit_on_full:
        print("[train_linear] Refitting on train+val...")
        X_full = np.concatenate([Xtr, Xva], axis=0)
        y_full = np.concatenate([ytr, yva], axis=0)
        best["clf"] = make_pipeline(best["C"])
        best["clf"].fit(X_full, y_full)

    test_metrics = evaluate(best["clf"], Xte, yte)
    np.save(os.path.join(exp_dir, "predictions_test.npy"), test_metrics["yhat"])

    results = {
        "model": args.model,
        "search_Cs": args.Cs,
        "best_C": best["C"],
        "val_acc": round(best["val_acc"], 3),
        "test_acc": round(test_metrics["acc"], 3),
        "scaled": bool(args.scale),
        "refit_on_full": bool(args.refit_on_full),
        "seed": args.seed,
    }

    if has_c10c:
        c10c_metrics = evaluate(best["clf"], Xc, yc)
        np.save(os.path.join(exp_dir, "predictions_cifar10c.npy"), c10c_metrics["yhat"])
        results["cifar10c_acc"] = round(c10c_metrics["acc"], 3)

        if test_metrics["acc"] > 0:
            RR = c10c_metrics["acc"] / test_metrics["acc"]
            results["relative_robustness"] = round(float(RR), 4)

    dump(best["clf"], os.path.join(exp_dir, "linear_clf.joblib"))
    with open(os.path.join(exp_dir, "results_linear.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[train_linear] Done. Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
