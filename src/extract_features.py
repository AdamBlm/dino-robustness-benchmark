import os
import argparse
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import timm

from datasets import (
    get_cifar10_dataloaders,
    get_cifar10c_loader,
    build_transform_for_model,
)


def load_backbone(model_name, device):
    model = timm.create_model(model_name, pretrained=True)
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)

    model.eval().to(device)

    def _forward_features(x):
        with torch.no_grad():
            feats = model.forward_features(x)
            if isinstance(feats, dict):
                if "x" in feats:
                    z = feats["x"]
                    if z.ndim == 3:
                        cls = z[:, 0]
                        return cls
                    return z
                if "pre_logits" in feats:
                    z = feats["pre_logits"]
                    return z if z.ndim == 2 else z.mean(dim=tuple(range(1, z.ndim)))
            if isinstance(feats, torch.Tensor):
                if feats.ndim == 2:
                    return feats
                if feats.ndim == 3:
                    return feats[:, 0] if feats.size(1) >= 1 else feats.mean(dim=1)
                if feats.ndim == 4:
                    return feats.mean(dim=(2, 3))
            return feats.mean(dim=tuple(range(2, feats.ndim)))

    class Embedder(nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, x):
            return _forward_features(x)

    return Embedder(model).to(device)


@torch.no_grad()
def extract_from_loader(
    model,
    loader,
    device,
    use_amp=True,
    desc="extract",
):
    xs, ys = [], []
    cids, sevs = [], []
    has_meta = False
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            y = batch[1]
            if len(batch) >= 4:
                has_meta = True
                cid = batch[2]
                sev = batch[3]
        else:
            raise RuntimeError("Unexpected batch structure from DataLoader")

        x = x.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                z = model(x)
        else:
            z = model(x)
        xs.append(z.float().cpu())
        ys.append(y.clone().cpu())
        if has_meta:
            cids.append(cid.clone().cpu())
            sevs.append(sev.clone().cpu())

    X = torch.cat(xs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()
    if has_meta:
        C = torch.cat(cids, dim=0).numpy()
        S = torch.cat(sevs, dim=0).numpy()
        return X, Y, C, S
    return X, Y


def save_features(X, Y, out_dir, model_name, split):
    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, f"{split}_X.npy"), X)
    np.save(os.path.join(model_dir, f"{split}_y.npy"), Y)


def parse_args():
    ap = argparse.ArgumentParser(description="Frozen feature extraction for CIFAR-10/10-C")

    ap.add_argument("--model", type=str, required=True, help="timm model name (e.g., dinov2_vits14)")
    ap.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar10c"])

    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./features")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--corruptions", type=str, nargs="*", default=None,
                    help="subset of corruption names; default uses all available")
    ap.add_argument("--severities", type=int, nargs="*", default=None,
                    help="subset like: 1 3 5 ; default uses 1..5")

    ap.add_argument("--no_amp", action="store_true", help="disable autocast mixed precision")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_features] Device: {device}")

    model = load_backbone(args.model, device)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(
            data_root=args.data_root,
            model_name=args.model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_size=args.val_size,
            seed=args.seed,
        )
        Xtr, ytr = extract_from_loader(model, train_loader, device, use_amp=(not args.no_amp), desc="train")
        Xva, yva = extract_from_loader(model, val_loader, device, use_amp=(not args.no_amp), desc="val")
        Xte, yte = extract_from_loader(model, test_loader, device, use_amp=(not args.no_amp), desc="test")

        save_features(Xtr, ytr, args.out_dir, args.model, "train")
        save_features(Xva, yva, args.out_dir, args.model, "val")
        save_features(Xte, yte, args.out_dir, args.model, "test")
        print(f"[extract_features] Saved CIFAR-10 features under {os.path.join(args.out_dir, args.model)}")

    elif args.dataset == "cifar10c":
        c_loader = get_cifar10c_loader(
            data_root=args.data_root,
            model_name=args.model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruptions=args.corruptions,
            severities=args.severities,
        )
        out = extract_from_loader(model, c_loader, device, use_amp=(not args.no_amp), desc="cifar10-c")
        if len(out) == 4:
            Xc, yc, ccids, ssev = out
        else:
            Xc, yc = out
            n = Xc.shape[0]
            ccids = np.full((n,), -1, dtype=np.int32)
            ssev = np.full((n,), -1, dtype=np.int32)

        save_features(Xc, yc, args.out_dir, args.model, "cifar10c")

        model_dir = os.path.join(args.out_dir, args.model)
        np.save(os.path.join(model_dir, "cifar10c_corruption_ids.npy"), ccids)
        np.save(os.path.join(model_dir, "cifar10c_severities.npy"), ssev)
        try:
            ds = c_loader.dataset
            meta = {"corruptions": getattr(ds, "corruptions", [])}
        except Exception:
            meta = {"corruptions": []}
        import json as _json
        with open(os.path.join(model_dir, "cifar10c_meta.json"), "w") as f:
            _json.dump(meta, f, indent=2)

        print(f"[extract_features] Saved CIFAR-10-C features and metadata under {os.path.join(args.out_dir, args.model)}")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
