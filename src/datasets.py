import os
import json

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


def build_transform_for_model(model_name, is_train=False):
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


def _stratified_split_indices(
    targets,
    val_size=5000,
    seed=42,
):
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
    data_root,
    model_name,
    batch_size=256,
    num_workers=4,
    val_size=5000,
    seed=42,
    pin_memory=True,
):
    os.makedirs(data_root, exist_ok=True)
    tfm = build_transform_for_model(model_name, is_train=False)

    ds_train_full = tvdatasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    ds_test = tvdatasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)

    train_idx, val_idx = _stratified_split_indices(ds_train_full.targets, val_size=val_size, seed=seed)
    ds_train = Subset(ds_train_full, train_idx.tolist())
    ds_val = Subset(ds_train_full, val_idx.tolist())

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


_CIFAR10C_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"
]

class CIFAR10C(Dataset):
    def __init__(
        self,
        root,
        model_name,
        corruptions=None,
        severities=None,
    ):
        super().__init__()
        self.root = os.path.join(root, "CIFAR-10-C")
        if not os.path.isdir(self.root):
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

        self.index_map = []
        for cid, cname in enumerate(self.corruptions):
            for s in self.severities:
                sl = self._severity_slices[s]
                for i in range(sl.start, sl.stop):
                    local = i - sl.start
                    self.index_map.append((cid, s, local))

        self.transform = build_transform_for_model(model_name, is_train=False)

    def __len__(self):
        return len(self.index_map)

    def _load_corruption_block(self, cname):
        path = os.path.join(self.root, f"{cname}.npy")
        return np.load(path, mmap_mode="r")

    def __getitem__(self, idx):
        cid, severity, local_idx = self.index_map[idx]
        cname = self.corruptions[cid]
        arr = self._load_corruption_block(cname)
        sl = slice(10000*(severity-1), 10000*severity)
        img_np = arr[sl][local_idx]
        label = int(self.labels_all[sl][local_idx])

        pil = T.functional.to_pil_image(img_np)
        x = self.transform(pil)
        return x, label, int(cid), int(severity)


def get_cifar10c_loader(
    data_root,
    model_name,
    batch_size=256,
    num_workers=4,
    corruptions=None,
    severities=None,
    pin_memory=True,
):
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
