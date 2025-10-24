"""
Data utilities for CIFAR-10 and CIFAR-10-C.
"""

from __future__ import annotations
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets as tvdatasets
from torchvision import transforms as T

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# ----------------------------
# Transforms / timm utilities
# ----------------------------

def build_transform_for_model(model_name: str, is_train: bool = False) -> T.Compose:
    """
    Creates a transform for a timm model.
    No augmentations, just resizing and normalization.
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    cfg = resolve_data_config({}, model=model)

    tfm = create_transform(
        input_size=cfg["input_size"],
        interpolation=cfg.get("interpolation", "bicubic"),
        mean=cfg["mean"],
        std=cfg["std"],
        crop_pct=cfg.get("crop_pct", 1.0),
        is_training=is_train,
        re_prob=0.0,
        hflip=0.0,
        color_jitter=0.0,
    )

    if isinstance(tfm, T.Compose):
        return tfm
    return T.Compose([tfm])


# ----------------------------
# CIFAR-10 loaders
# ----------------------------

def _stratified_split_indices(
    targets: List[int],
    val_size: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split for CIFAR-10 train set.
    """
    y = np.array(targets)
    n = len(y)
    if isinstance(val_size, float) and 0 < val_size < 1:
        val_prop = val_size
    else:
        val_prop = val_size / n

    if _HAS_SKLEARN:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_prop, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(n), y))
        return train_idx, val_idx

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        v = int(round(len(idx) * val_prop))
        val_idx.extend(idx[:v].tolist())
        train_idx.extend(idx[v:].tolist())
    return np.array(train_idx), np.array(val_idx)


def get_cifar10_dataloaders(
    data_root: str,
    model_name: str,
    batch_size: int = 256,
    num_workers: int = 4,
    val_size: int = 5000,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Gets CIFAR-10 train/val/test DataLoaders.
    """
    os.makedirs(data_root, exist_ok=True)
    tfm = build_transform_for_model(model_name, is_train=False)

    ds_train_full = tvdatasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    ds_test = tvdatasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)

    train_idx, val_idx = _stratified_split_indices(ds_train_full.targets, val_size=val_size, seed=seed)
    ds_train = Subset(ds_train_full, train_idx.tolist())
    ds_val = Subset(ds_train_full, val_idx.tolist())

    # Save split indices for reproducibility
    split_dir = os.path.join(data_root, "cifar10_splits")
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, f"split_seed{seed}_val{val_size}.json"), "w") as f:
        json.dump(
            {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()},
            f,
            indent=2,
        )

    common_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        common_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(ds_train, shuffle=False, drop_last=False, **common_kwargs)
    val_loader = DataLoader(ds_val, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(ds_test, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader, test_loader


# ----------------------------
# CIFAR-10-C dataset / loader
# ----------------------------

_CIFAR10C_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"
]

class CIFAR10C(Dataset):
    """
    CIFAR-10-C dataset.
    Handles loading corrupted images from .npy files.
    """
    def __init__(
        self,
        root: str,
        model_name: str,
        corruptions: Optional[List[str]] = None,
        severities: Optional[List[int]] = None,
    ):
        super().__init__()
        self.root = os.path.join(root, "CIFAR-10-C")
        if not os.path.isdir(self.root):
            # Common alternative folder name
            alt = os.path.join(root, "cifar10-c")
            if os.path.isdir(alt):
                self.root = alt
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"CIFAR-10-C directory not found under {root}. "
                f"Expected '{os.path.join(root, 'CIFAR-10-C')}' or '{os.path.join(root, 'cifar10-c')}'. "
                "Download from https://github.com/hendrycks/robustness."
            )

        labels_path = os.path.join(self.root, "labels.npy")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError("labels.npy not found in CIFAR-10-C directory.")

        self.labels_all = np.load(labels_path)
        self._severity_slices = {s: slice(10000*(s-1), 10000*s) for s in range(1, 6)}

        self.corruptions = corruptions or _CIFAR10C_CORRUPTIONS
        self.severities = severities or [1, 2, 3, 4, 5]

        # Filter to existing corruptions in the folder
        existing = []
        for c in self.corruptions:
            p = os.path.join(self.root, f"{c}.npy")
            if os.path.isfile(p):
                existing.append(c)
        if len(existing) == 0:
            raise RuntimeError("No corruption .npy files found. Check your CIFAR-10-C download.")
        if set(existing) != set(self.corruptions):
            missing = sorted(set(self.corruptions) - set(existing))
            if missing:
                print(f"[datasets.py] Warning: Missing corruptions: {missing}")
            self.corruptions = existing

        # Build an index of (corruption_name, severity, local_idx)
        self.index_map: List[Tuple[str, int, int]] = []
        for cname in self.corruptions:
            for s in self.severities:
                sl = self._severity_slices[s]
                for i in range(sl.start, sl.stop):
                    local = i - sl.start
                    self.index_map.append((cname, s, local))

        self.transform = build_transform_for_model(model_name, is_train=False)

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_corruption_block(self, cname: str) -> np.ndarray:
        path = os.path.join(self.root, f"{cname}.npy")
        return np.load(path, mmap_mode="r")

    def __getitem__(self, idx: int):
        cname, severity, local_idx = self.index_map[idx]
        arr = self._load_corruption_block(cname)
        sl = slice(10000*(severity-1), 10000*severity)
        img_np = arr[sl][local_idx]
        label = int(self.labels_all[sl][local_idx])

        pil = T.functional.to_pil_image(img_np)
        x = self.transform(pil)
        return x, label


def get_cifar10c_loader(
    data_root: str,
    model_name: str,
    batch_size: int = 256,
    num_workers: int = 4,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for CIFAR-10-C.
    """
    ds = CIFAR10C(
        root=data_root,
        model_name=model_name,
        corruptions=corruptions,
        severities=severities,
    )

    kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    return DataLoader(ds, drop_last=False, **kwargs)
