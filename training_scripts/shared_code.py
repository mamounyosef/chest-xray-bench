"""
shared_code.py  —  CHEST-XRAY-BENCH shared library
===================================================

Everything that must stay identical across experiments lives here so the
comparison stays controlled: config loading, the data layer (preprocessing /
Dataset / loaders), and the training engine (seed / model / loss / metrics /
loop). A small `shared_config.yaml` holds settings common to all experiments;
each experiment's own `config.yaml` holds what varies and is merged on top.

Sections:
    [1] Config loading   [2] Data layer   [3] Metrics/optim   [4] Engine
    [5] Modal helpers (optional cloud execution)
"""

from __future__ import annotations

import copy
import csv
import functools
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T

# Directory of this file (training_scripts/); experiment folders sit next to it.
PKG_DIR = Path(__file__).resolve().parent
SHARED_CONFIG = PKG_DIR / "shared_config.yaml"


# =============================================================================
# Section 1 — Config loading
# A small shared_config.yaml (next to this file) holds the settings common to
# EVERY experiment: paths, tasks, image, augmentation, reproducibility, output.
# Each experiment's own config.yaml holds what VARIES: experiment, model, labels,
# clahe, dataloader, training, evaluation. The two are deep-merged (experiment
# wins on any shared key), so the returned cfg is complete.
# =============================================================================

def _read_yaml(path: Path) -> dict:
    """Load one YAML file into a dict (empty/blank file -> {})."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` INTO a copy of `base`. Nested dicts merge
    key-by-key; scalars/lists in override replace the base value wholesale."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_config(experiment_dir, config_name: str = "config.yaml",
                shared_config=SHARED_CONFIG, verbose: bool = True) -> dict:
    """Merge shared_config.yaml with one experiment's config.yaml and return it.

    Parameters
    ----------
    experiment_dir : folder of the experiment (holds its config.yaml).
    config_name    : config filename inside that folder.
    shared_config  : path to the shared common-settings YAML.
    verbose        : print a short summary of the key settings.
    """
    experiment_dir = Path(experiment_dir)
    shared_config = Path(shared_config)
    config_path = experiment_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"experiment config not found: {config_path}")

    shared = _read_yaml(shared_config) if shared_config.exists() else {}
    exp = _read_yaml(config_path)
    cfg = _deep_merge(shared, exp)

    if verbose:
        miss = "" if shared_config.exists() else "  (MISSING!)"
        print("=" * 70)
        print("[load_config] loaded experiment configuration")
        print("-" * 70)
        print(f"  shared config   : {shared_config}{miss}")
        print(f"  experiment file : {config_path}")
        print(f"  shared sections : {list(shared.keys())}")
        print(f"  experiment keys : {list(exp.keys())}")
        print(f"  experiment name : {cfg.get('experiment', {}).get('name', '<unset>')}")
        print(f"  model           : {cfg.get('model', {}).get('name', '<unset>')}")
        print(f"  use_clahe       : {cfg.get('clahe', {}).get('use_clahe')}")
        print(f"  u_policy        : {cfg.get('labels', {}).get('u_policy')}")
        print("=" * 70)

    return cfg


# =============================================================================
# Section 2 — Data layer  (promoted from data_code/02_splits_dataset.ipynb)
# Everything is driven by cfg, so each experiment's config.yaml fully determines
# its pipeline (use_clahe, u_policy, augmentation, ...).
# =============================================================================

def resolve_path(rel: str, cfg: dict) -> Path | None:
    """Resolve a CSV `Path` to a real file. train/valid live under data_root;
    the full-res test images live one level up — so try both bases."""
    data_root = Path(cfg["paths"]["data_root"])
    for base in (data_root.parent, data_root):
        p = base / rel
        if p.exists():
            return p
    return None


def resize_pad(img: np.ndarray, out_w: int, out_h: int,
               interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize preserving aspect ratio to fit inside (out_w, out_h), then center
    zero-pad (black) to exactly that size. Padding is required so every image
    stacks into a uniform batch tensor."""
    h0, w0 = img.shape
    scale = min(out_w / w0, out_h / h0)
    new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((out_h, out_w), dtype=img.dtype)        # 0 = black background
    top, left = (out_h - new_h) // 2, (out_w - new_w) // 2    # center placement
    canvas[top:top + new_h, left:left + new_w] = img
    return canvas


@functools.lru_cache(maxsize=8)
def _get_clahe(clip_limit: float, tile_grid: tuple):
    """Cache the CLAHE operator (cheap to reuse across all samples)."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)


def build_labels(df: pd.DataFrame, tasks: list, policy: str = "ones") -> pd.DataFrame:
    """Raw labels (1 / 0 / -1 / blank) -> target vector per uncertainty policy.
    blanks/NaN -> 0 always; policy maps the uncertain (-1) entries."""
    target = df[tasks].fillna(0).copy()
    if policy == "ones":
        target = target.replace(-1, 1)
    elif policy == "zeros":
        target = target.replace(-1, 0)
    else:
        raise ValueError(f"unknown u_policy: {policy!r} (expected 'ones' | 'zeros')")
    return target.astype("float32")


def build_train_transforms(cfg: dict):
    """torchvision v2 augmentation pipeline, TRAIN ONLY. Each aug is wrapped in
    its own RandomApply so it rolls an INDEPENDENT probability `p`. Geometric
    fills with 0 (black) to match the zero-pad. Returns a Compose or None."""
    aug = cfg.get("augmentation", {})
    if not aug.get("enable", False):
        return None
    ops = []
    if "rotation" in aug:
        r = aug["rotation"]
        ops.append(T.RandomApply([T.RandomRotation(degrees=r["deg"], fill=0)], p=r["p"]))
    if "translate" in aug:
        tr = aug["translate"]
        ops.append(T.RandomApply(
            [T.RandomAffine(degrees=0, translate=(tr["frac"], tr["frac"]), fill=0)], p=tr["p"]))
    if "scale" in aug:
        sc = aug["scale"]
        ops.append(T.RandomApply(
            [T.RandomAffine(degrees=0, scale=tuple(sc["range"]), fill=0)], p=sc["p"]))
    if "brightness" in aug:
        br = aug["brightness"]
        ops.append(T.RandomApply([T.ColorJitter(brightness=br["factor"])], p=br["p"]))
    if "contrast" in aug:
        ct = aug["contrast"]
        ops.append(T.RandomApply([T.ColorJitter(contrast=ct["factor"])], p=ct["p"]))
    return T.Compose(ops) if ops else None


def preprocess(rel: str, cfg: dict, use_clahe: bool = False,
               train_transforms=None, clahe=None) -> torch.Tensor:
    """One image: grayscale -> [CLAHE] -> resize_pad -> tensor -> [augment] ->
    3-channel -> /255 -> ImageNet-normalize. Returns (C, H, W) float32."""
    img_cfg = cfg["image"]
    out_w, out_h = img_cfg["width"], img_cfg["height"]
    interp = getattr(cv2, img_cfg["interpolation"])

    path = resolve_path(rel, cfg)
    if path is None:
        raise FileNotFoundError(f"image not found for relative path: {rel}")

    img = np.asarray(Image.open(path).convert("L"))                 # grayscale uint8
    if use_clahe:
        if clahe is None:
            clahe = _get_clahe(cfg["clahe"]["clip_limit"], tuple(cfg["clahe"]["tile_grid"]))
        img = clahe.apply(img)
    img = resize_pad(img, out_w, out_h, interp)                     # (H,W) uint8

    t = torch.from_numpy(img).unsqueeze(0).float().div_(255.0)      # (1,H,W) in [0,1]
    if train_transforms is not None:
        t = train_transforms(t)                                     # augment pre-normalize
    ch = img_cfg.get("channels", 3)
    if ch == 3:
        t = t.repeat(3, 1, 1)                                       # grayscale -> 3ch
    mean = torch.tensor(img_cfg["norm_mean"][:t.shape[0]]).view(-1, 1, 1)
    std = torch.tensor(img_cfg["norm_std"][:t.shape[0]]).view(-1, 1, 1)
    t = (t - mean) / std
    return t.float()                                               # (C,H,W) float32


class CheXpertDataset(Dataset):
    """Ties preprocess() + build_labels() together. `split` decides augmentation
    ('train' augments, 'val' never does). All knobs come from cfg."""

    def __init__(self, df: pd.DataFrame, cfg: dict, split: str):
        assert split in ("train", "val"), f"split must be 'train'|'val', got {split!r}"
        self.cfg = cfg
        self.split = split
        self.tasks = cfg["tasks"]
        self.paths = df["Path"].tolist()
        policy = cfg["labels"]["u_policy"]
        self.labels = torch.from_numpy(build_labels(df, self.tasks, policy).to_numpy())
        self.use_clahe = bool(cfg["clahe"]["use_clahe"])
        self._clahe = (
            _get_clahe(cfg["clahe"]["clip_limit"], tuple(cfg["clahe"]["tile_grid"]))
            if self.use_clahe else None
        )
        self.train_transforms = build_train_transforms(cfg) if split == "train" else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = preprocess(
            self.paths[i], self.cfg, use_clahe=self.use_clahe,
            train_transforms=self.train_transforms, clahe=self._clahe,
        )
        return img, self.labels[i]                                  # (C,H,W) f32, (n_tasks,) f32


def seed_worker(worker_id: int):
    """DataLoader worker init: derive each worker's numpy/random seed from the
    torch seed so shuffling/augmentation are reproducible across workers."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(cfg: dict):
    """Build the train + val DataLoaders from cfg (split CSVs + dataloader.*).
    Seeded generator + worker_init_fn for deterministic shuffling/augmentation."""
    paths = cfg["paths"]
    data_dir = Path(paths["data_dir"])
    train_csv = data_dir / paths["train_csv"]
    val_csv = data_dir / paths["val_csv"]

    print("=" * 70)
    print("[make_loaders] building datasets & loaders")
    print("-" * 70)
    print(f"  train csv : {train_csv}")
    print(f"  val   csv : {val_csv}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"  train rows: {len(train_df)}    val rows: {len(val_df)}")

    train_ds = CheXpertDataset(train_df, cfg, split="train")
    val_ds = CheXpertDataset(val_df, cfg, split="val")

    dl = cfg["dataloader"]
    g = torch.Generator()
    g.manual_seed(cfg["reproducibility"]["seed"])

    common = dict(num_workers=dl["num_workers"], pin_memory=dl["pin_memory"],
                  worker_init_fn=seed_worker)
    extra = {}
    if dl["num_workers"] > 0:   # these kwargs are only valid with real workers
        extra = dict(prefetch_factor=dl["prefetch_factor"],
                     persistent_workers=dl["persistent_workers"])

    train_loader = DataLoader(train_ds, batch_size=dl["batch_size"], shuffle=True,
                              drop_last=dl["drop_last"], generator=g, **common, **extra)
    val_loader = DataLoader(val_ds, batch_size=dl["batch_size"], shuffle=False,
                            drop_last=False, **common, **extra)

    print("-" * 70)
    print(f"  batch_size={dl['batch_size']}  num_workers={dl['num_workers']}  "
          f"use_clahe={train_ds.use_clahe}  u_policy={cfg['labels']['u_policy']}")
    print(f"  augment (train): {'ON' if train_ds.train_transforms is not None else 'OFF'}")
    print(f"  train: {len(train_ds)} imgs -> {len(train_loader)} batches")
    print(f"  val  : {len(val_ds)} imgs -> {len(val_loader)} batches")
    print("=" * 70)
    return train_loader, val_loader


# =============================================================================
# Section 3 — Reproducibility, metrics, and optim builders
# =============================================================================

def set_seed(seed: int, deterministic: bool = True):
    """Seed every RNG (python / numpy / torch / cuda). With deterministic=True
    also pin cuDNN so runs are repeatable (at a small speed cost)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[set_seed] seed={seed}  deterministic={deterministic}")


def compute_metrics(y_true, y_prob, tasks: list, threshold: float = 0.5) -> dict:
    """Multi-label metrics from ground-truth (N,T) 0/1 and probabilities (N,T).

    Per task: AUROC, AUPRC (threshold-free) + F1 / precision / recall /
    specificity @ threshold, plus the positive count. Macro = mean over tasks
    (AUROC/AUPRC use nanmean so a task with a single class present — undefined
    AUROC -> NaN — is skipped instead of poisoning the average).
    Returns {"macro": {...}, "per_task": {task: {...}}}.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    preds = (y_prob >= threshold).astype(int)

    per_task = {}
    cols = {k: [] for k in ("auroc", "auprc", "f1", "precision", "recall", "specificity")}
    for k, t in enumerate(tasks):
        yt, yp, pr = y_true[:, k], y_prob[:, k], preds[:, k]
        n_pos = int(yt.sum())
        n_neg = int((yt == 0).sum())

        auroc = roc_auc_score(yt, yp) if (n_pos > 0 and n_neg > 0) else float("nan")
        auprc = average_precision_score(yt, yp) if n_pos > 0 else float("nan")

        tp = int(((pr == 1) & (yt == 1)).sum())
        fp = int(((pr == 1) & (yt == 0)).sum())
        tn = int(((pr == 0) & (yt == 0)).sum())
        fn = int(((pr == 0) & (yt == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0          # = sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_task[t] = dict(auroc=auroc, auprc=auprc, f1=f1, precision=precision,
                           recall=recall, specificity=specificity, n_pos=n_pos)
        for key, val in (("auroc", auroc), ("auprc", auprc), ("f1", f1),
                         ("precision", precision), ("recall", recall),
                         ("specificity", specificity)):
            cols[key].append(val)

    macro = {
        "mean_auroc":       float(np.nanmean(cols["auroc"])),
        "mean_auprc":       float(np.nanmean(cols["auprc"])),
        "mean_f1":          float(np.mean(cols["f1"])),
        "mean_precision":   float(np.mean(cols["precision"])),
        "mean_recall":      float(np.mean(cols["recall"])),
        "mean_specificity": float(np.mean(cols["specificity"])),
    }
    return {"macro": macro, "per_task": per_task}


def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """AdamW from cfg['training']['optimizer'] (lr, weight_decay)."""
    o = cfg["training"]["optimizer"]
    return optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, steps_per_epoch: int | None = None):
    """cosine (+linear warmup) | plateau | none, stepped PER EPOCH.

    Returns (scheduler_or_None, requires_metric). `requires_metric` is True only
    for plateau, telling the engine to call scheduler.step(val_mean_auroc)."""
    s = cfg["training"]["scheduler"]
    name = s["name"].lower()
    epochs = cfg["training"]["epochs"]

    if name == "none":
        return None, False
    if name == "cosine":
        warmup = int(s.get("warmup_epochs", 0))
        min_lr = s.get("min_lr", 0.0)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup), eta_min=min_lr)
        if warmup > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup)
            sched = optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_sched, cosine], milestones=[warmup])
            return sched, False
        return cosine, False
    if name == "plateau":
        mode = cfg["training"]["early_stopping"].get("mode", "max")
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=s.get("plateau_patience", 2),
            min_lr=s.get("min_lr", 0.0))
        return sched, True
    raise ValueError(f"unknown scheduler: {name!r}")


def build_loss(cfg: dict, device=None) -> nn.Module:
    """Multi-label BCE-with-logits. Optional per-task pos_weight (list) is moved
    onto `device` so it lands on the same device as the logits."""
    l = cfg["training"]["loss"]
    name = l["name"].lower()
    if name != "bce_with_logits":
        raise ValueError(f"unknown loss: {name!r}")
    pw = l.get("pos_weight")
    pos_weight = None
    if pw is not None:
        pos_weight = torch.tensor(pw, dtype=torch.float32)
        if device is not None:
            pos_weight = pos_weight.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# =============================================================================
# Section 4 — Engine: logging, validation, checkpointing, train loop, runner
# =============================================================================


class _Tee:
    """Duplicate everything written to a stream into several streams at once
    (e.g. the real console + a capture file). Flushes on every write so a crash
    still leaves a complete log. isatty()->False so libraries don't expect a TTY."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass        # never let a stream error (e.g. capture file) crash prints

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class CSVLogger:
    """Append-only CSV writer. Writes a header on first row (or when the file is
    empty), keeps the handle open, flushes every row so logs survive a crash."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = None
        self._writer = None

    def log(self, row: dict):
        # Robust: a logging failure must never crash training. On error, drop the
        # row, reset the handle, and let the next call try to reopen.
        try:
            if self._f is None:
                new = (not self.path.exists()) or self.path.stat().st_size == 0
                self._f = open(self.path, "a", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._f, fieldnames=list(row.keys()))
                if new:
                    self._writer.writeheader()
            self._writer.writerow(row)
            self._f.flush()
        except Exception as e:
            print(f"  ⚠️  CSV log failed (continuing): {type(e).__name__}: {e}")
            try:
                if self._f is not None:
                    self._f.close()
            except Exception:
                pass
            self._f, self._writer = None, None

    def close(self):
        try:
            if self._f is not None:
                self._f.close()
        except Exception:
            pass
        self._f = None


def _safe(label: str, fn, *args, **kwargs):
    """Run fn(*args) but never let it crash the run; warn and continue on error."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠️  {label} failed (continuing): {type(e).__name__}: {e}")
        return None


def _write_summary(path: Path, data: dict):
    """Write the best-so-far metrics snapshot as JSON (atomic-ish via flush)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()


def _save_debug_image(imgs: torch.Tensor, cfg: dict, debug_dir: Path,
                      epoch: int, epoch_step: int):
    """Save ONE random sample from the batch exactly as the model sees it
    (un-normalized back to [0,1] for viewing) -> debug_dir/e{epoch}_{step}.png."""
    idx = random.randrange(imgs.size(0))
    t = imgs[idx].detach().float().cpu()                       # (C,H,W) normalized
    mean = torch.tensor(cfg["image"]["norm_mean"][:t.shape[0]]).view(-1, 1, 1)
    std = torch.tensor(cfg["image"]["norm_std"][:t.shape[0]]).view(-1, 1, 1)
    t = (t * std + mean).clamp(0, 1)                           # un-normalize -> [0,1]
    arr = (t * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(debug_dir / f"e{epoch}_{epoch_step}.png")


def _monitor_value(macro: dict, val_loss: float, monitor: str) -> float:
    """Resolve an early_stopping.monitor string to its value.
    'val_loss' -> val_loss; 'val_<macro key>' (e.g. val_mean_auroc) -> macro[...]."""
    m = monitor[4:] if monitor.startswith("val_") else monitor
    if m == "loss":
        return val_loss
    if m in macro:
        return macro[m]
    raise KeyError(f"unknown early_stopping.monitor: {monitor!r} "
                   f"(expected val_loss or one of val_{{{', '.join(macro)}}})")


def fmt_duration(seconds: float) -> str:
    """Human-friendly duration that auto-scales: '45s', '12m 30s', '2h 15m',
    '1d 03h'. NaN/inf -> '?' (e.g. ETA before the first step)."""
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    if seconds < 86400:
        h, rem = divmod(seconds, 3600)
        return f"{h}h {rem // 60:02d}m"
    d, rem = divmod(seconds, 86400)
    return f"{d}d {rem // 3600:02d}h"


def _flatten_val_row(epoch: int, gstep: int, val_loss: float, metrics: dict,
                     tasks: list, elapsed_sec: float, eta_sec: float) -> dict:
    """Flatten validation metrics into one wide CSV row (macro + per-task)."""
    row = {"step": gstep, "epoch": epoch, "elapsed_sec": round(elapsed_sec, 2),
           "eta_sec": round(eta_sec, 2) if math.isfinite(eta_sec) else "",
           "val_loss": round(val_loss, 6)}
    for k, v in metrics["macro"].items():
        row[k] = round(v, 6)
    for t in tasks:
        m = metrics["per_task"][t]
        for key in ("auroc", "auprc", "f1", "precision", "recall", "specificity", "n_pos"):
            val = m[key]
            row[f"{key}/{t}"] = round(val, 6) if isinstance(val, float) else val
    return row


def _rng_state(device) -> dict:
    """Snapshot every RNG so a resumed run continues the same random stream."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore_rng(state: dict, device):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def print_config(cfg: dict):
    """Comprehensive startup dump of every important hyperparameter / setting,
    grouped and aligned so a remote run is fully self-documenting in the log."""
    img, clahe, aug = cfg["image"], cfg["clahe"], cfg["augmentation"]
    dl, tr = cfg["dataloader"], cfg["training"]
    opt, sch, loss, es = tr["optimizer"], tr["scheduler"], tr["loss"], tr["early_stopping"]
    ev, rep, out = cfg["evaluation"], cfg["reproducibility"], cfg["output"]

    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  experiment        : {cfg.get('experiment', {}).get('name', '<unnamed>')}")
    print(f"  model             : {cfg.get('model', {}).get('name', '<unset>')}"
          f"  (pretrained={cfg.get('model', {}).get('pretrained')})")
    print(f"  resume            : {cfg.get('resume')}")
    print(f"  tasks ({len(cfg['tasks'])})         : {cfg['tasks']}")
    print("  --- paths --------------------------------------------------------")
    p = cfg["paths"]
    print(f"  data_root         : {p['data_root']}")
    print(f"  data_dir          : {p['data_dir']}")
    print(f"  train / val csv   : {p['train_csv']}  |  {p['val_csv']}")
    print("  --- data ---------------------------------------------------------")
    print(f"  u_policy          : {cfg['labels']['u_policy']}")
    print(f"  image (W x H)     : {img['width']} x {img['height']}  "
          f"channels={img['channels']}  interp={img['interpolation']}")
    print(f"  normalize mean    : {img['norm_mean']}")
    print(f"  normalize std     : {img['norm_std']}")
    print(f"  CLAHE             : use={clahe['use_clahe']}  clip={clahe['clip_limit']}  "
          f"tiles={clahe['tile_grid']}")
    if aug.get("enable"):
        print(f"  augmentation      : ENABLED (train only, independent p per aug)")
        print(f"      rotation      : p={aug['rotation']['p']}  deg=+/-{aug['rotation']['deg']}")
        print(f"      translate     : p={aug['translate']['p']}  frac=+/-{aug['translate']['frac']}")
        print(f"      scale         : p={aug['scale']['p']}  range={aug['scale']['range']}")
        print(f"      brightness    : p={aug['brightness']['p']}  factor=+/-{aug['brightness']['factor']}")
        print(f"      contrast      : p={aug['contrast']['p']}  factor=+/-{aug['contrast']['factor']}")
    else:
        print(f"  augmentation      : DISABLED")
    print("  --- dataloader ---------------------------------------------------")
    print(f"  batch_size        : {dl['batch_size']}")
    print(f"  num_workers       : {dl['num_workers']}  pin_memory={dl['pin_memory']}  "
          f"drop_last={dl['drop_last']}")
    print(f"  prefetch_factor   : {dl['prefetch_factor']}  persistent={dl['persistent_workers']}")
    print("  --- training -----------------------------------------------------")
    print(f"  epochs            : {tr['epochs']}")
    print(f"  optimizer         : AdamW  lr={opt['lr']}  weight_decay={opt['weight_decay']}")
    print(f"  scheduler         : {sch['name']}  warmup_epochs={sch.get('warmup_epochs')}  "
          f"min_lr={sch.get('min_lr')}")
    print(f"  loss              : {loss['name']}  pos_weight={loss.get('pos_weight')}")
    print(f"  amp               : {tr['amp']}  grad_clip={tr['grad_clip']}")
    print(f"  early_stopping    : enable={es['enable']}  monitor={es['monitor']}  "
          f"mode={es['mode']}  patience={es['patience']}")
    print("  --- evaluation / repro / output ----------------------------------")
    print(f"  metric (headline) : {ev['metric']}  (eval_level={ev['eval_level']})")
    print(f"  eval cadence      : every_epochs={ev['eval_every_epochs']}  "
          f"every_steps={ev['eval_every_steps']}")
    print(f"  seed              : {rep['seed']}  deterministic={rep['deterministic']}")
    print(f"  keep_last_n ckpts : {out['keep_last_n']}  best_name={out['best_name']}")
    print(f"  run_dir           : {out['run_dir']}  "
          f"(logs: {out['train_log_csv']}, {out['val_log_csv']})")
    print(f"  console_log_every : {out['console_log_every']}  "
          f"capture_dir={out['console_capture_dir']}")
    if "modal" in cfg:
        m = cfg["modal"]
        print(f"  modal (cloud)     : gpu={m.get('gpu')}  cpu={m.get('cpu_cores')}  "
              f"ram={m.get('memory_gb')}GB  timeout={m.get('timeout_seconds')}s")
    # Guaranteed-complete fallback: dump the FULL merged config verbatim, so even
    # any key not explicitly formatted above is recorded in the run's txt log.
    print("  --- raw merged config (verbatim) ---------------------------------")
    print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False).rstrip())
    print("=" * 70)


@torch.no_grad()
def validate(model, val_loader, loss_fn, cfg, device, amp: bool) -> tuple:
    """Full pass over the val loader -> (val_loss, metrics dict). No grad, eval mode."""
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.append(labels.detach().cpu().numpy())
        ps.append(torch.sigmoid(logits).float().detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    val_loss = total_loss / max(1, n)
    metrics = compute_metrics(y, p, cfg["tasks"])
    return val_loss, metrics


def save_checkpoint(ckpt_dir: Path, cfg, model, optimizer, scheduler, scaler,
                    epoch: int, global_step: int, best_score: float,
                    is_best: bool, device, elapsed_sec: float) -> Path:
    """Write a fully-resumable checkpoint, prune to keep_last_n rolling copies,
    and (when is_best) refresh best.pt."""
    out = cfg["output"]
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "elapsed_sec": elapsed_sec,
        "rng": _rng_state(device),
        "cfg": cfg,
    }
    path = ckpt_dir / f"ckpt_step{global_step}.pt"
    torch.save(state, path)

    # rolling cleanup: keep only the newest keep_last_n step-checkpoints
    keep = int(out["keep_last_n"])
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"),
                   key=lambda p: int(p.stem.replace("ckpt_step", "")))
    for old in ckpts[:-keep] if keep > 0 else ckpts:
        old.unlink(missing_ok=True)

    if is_best:
        torch.save(state, ckpt_dir / out["best_name"])
    return path


def load_checkpoint(resume, model, ckpt_dir: Path, optimizer=None, scheduler=None,
                    scaler=None, device=None, restore_rng: bool = True) -> dict:
    """Resume from a checkpoint. `resume` may be 'best', 'last', or a path.
    Restores model (+ optimizer/scheduler/scaler/rng if given). Returns the
    bookkeeping dict {epoch, global_step, best_score}."""
    path = _resolve_resume(resume, ckpt_dir)
    print(f"[resume] loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        _restore_rng(ckpt.get("rng"), device)
    info = {"epoch": ckpt["epoch"], "global_step": ckpt["global_step"],
            "best_score": ckpt["best_score"], "elapsed_sec": ckpt.get("elapsed_sec", 0.0)}
    print(f"[resume] restored -> epoch={info['epoch']} step={info['global_step']} "
          f"best_score={info['best_score']:.4f} elapsed={info['elapsed_sec']:.0f}s")
    return info


def _resolve_resume(resume, ckpt_dir: Path) -> Path:
    """Map a resume spec to a concrete checkpoint file.
    Accepts 'best' | 'last' | <int step> (-> ckpt_step{N}.pt) | explicit path."""
    if resume == "best":
        cands = list(ckpt_dir.glob("best*.pt"))
        if not cands:
            raise FileNotFoundError(f"no best checkpoint in {ckpt_dir}")
        return cands[0]
    if resume == "last":
        ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"),
                       key=lambda p: int(p.stem.replace("ckpt_step", "")))
        if not ckpts:
            raise FileNotFoundError(f"no step checkpoints in {ckpt_dir}")
        return ckpts[-1]
    if isinstance(resume, int) or (isinstance(resume, str) and resume.isdigit()):
        path = ckpt_dir / f"ckpt_step{int(resume)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint for step {int(resume)}: {path}")
        return path
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device,
                    epoch: int, global_step: int, train_logger: CSVLogger,
                    amp: bool, grad_clip, eval_every_steps, on_eval,
                    console_log_every: int, elapsed_fn, eta_fn, total_steps: int,
                    log_fn=None, wandb_log_every: int = 50,
                    cfg: dict = None, debug_dir=None, debug_every: int = 0) -> tuple:
    """One epoch of training. Logs every step (loss/lr/grad_norm/throughput/elapsed/eta)
    and fires `on_eval(epoch, step)` every eval_every_steps. Returns (global_step, stop)."""
    model.train()
    clip = grad_clip if grad_clip else math.inf      # inf -> measure norm, no clip
    stop = False
    n_batches = len(loader)

    for i, (imgs, labels) in enumerate(loader, start=1):
        t0 = time.time()
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        # NaN/Inf guard: skip this batch's update instead of crashing the run.
        if not torch.isfinite(loss):
            print(f"  ⚠️  non-finite loss ({loss.item()}) at step ~{global_step + 1}; "
                  f"skipping this batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        imgs_per_s = imgs.size(0) / dt if dt > 0 else 0.0
        el = elapsed_fn()
        eta = eta_fn(global_step)
        _safe("train_log", train_logger.log, {
            "step": global_step, "epoch": epoch + 1,     # 1-indexed for humans
            "elapsed_sec": round(el, 2),
            "eta_sec": round(eta, 2) if math.isfinite(eta) else "",
            "train_loss": round(loss.item(), 6),
            "lr": lr, "grad_norm": round(float(grad_norm), 6),
            "imgs_per_s": round(imgs_per_s, 1), "sec_per_step": round(dt, 4),
        })
        if global_step % console_log_every == 0 or i == 1:
            print(f"  Epoch {epoch + 1}  epoch_step: {i}/{n_batches}  "
                  f"global_step: {global_step}/{total_steps}  "
                  f"loss={loss.item():.4f}  lr={lr:.2e}  gnorm={float(grad_norm):.2f}  "
                  f"{imgs_per_s:.0f} img/s  elapsed={fmt_duration(el)}  ETA={fmt_duration(eta)}")

        if log_fn is not None and (global_step % wandb_log_every == 0 or i == 1):
            _safe("wandb", log_fn, {
                "train/loss": loss.item(), "train/lr": lr,
                "train/grad_norm": float(grad_norm), "train/imgs_per_s": imgs_per_s,
                "train/epoch": epoch + 1,
            }, global_step)

        if debug_dir is not None and debug_every > 0 and global_step % debug_every == 0:
            _safe("debug_image", _save_debug_image, imgs, cfg, debug_dir, epoch + 1, i)

        if eval_every_steps and global_step % eval_every_steps == 0:
            stop = on_eval(epoch, global_step)
            model.train()                            # back to train mode after eval
            if stop:
                break
    return global_step, stop


def _run_experiment(cfg: dict, model, experiment_dir, resume=None, persist_fn=None, log_fn=None):
    """End-to-end training for one experiment. Identical machinery for every run;
    only the injected `model` and cfg overrides differ. Trains on 01_train.csv,
    validates on 01_val.csv, checkpoints (rolling + best by mean AUROC), early-stops.
    NOTE: official 200/500 reporting is intentionally NOT done here.
    Wrapped by run_experiment(), which tees all console output to a per-run txt."""
    experiment_dir = Path(experiment_dir)
    rep = cfg["reproducibility"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("#" * 70)
    print(f"# 🚀 run_experiment: {cfg.get('experiment', {}).get('name', '<unnamed>')}")
    print(f"# 🖥️  device={device}  epochs={cfg['training']['epochs']}")
    print("#" * 70)

    set_seed(rep["seed"], rep["deterministic"])
    print_config(cfg)

    # --- output layout ---
    out = cfg["output"]
    results_dir = experiment_dir / out["run_dir"]
    ckpt_dir = results_dir / out["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_logger = CSVLogger(results_dir / out["train_log_csv"])
    val_logger = CSVLogger(results_dir / out["val_log_csv"])
    print(f"[paths] results -> {results_dir}")

    # --- data / model / optim ---
    train_loader, val_loader = make_loaders(cfg)
    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler, sched_needs_metric = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    loss_fn = build_loss(cfg, device)
    amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    print(f"[setup] optimizer=AdamW  scheduler={cfg['training']['scheduler']['name']}  amp={amp}")

    # --- training params ---
    epochs = cfg["training"]["epochs"]
    grad_clip = cfg["training"]["grad_clip"]
    ev_steps = cfg["evaluation"]["eval_every_steps"]
    ev_epochs = cfg["evaluation"]["eval_every_epochs"]
    es = cfg["training"]["early_stopping"]
    es_enable, es_patience = es["enable"], int(es["patience"])
    monitor = es.get("monitor", "val_mean_auroc")     # which metric drives best + early-stop
    mode = es.get("mode", "max")                       # "max" | "min"
    better = (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)
    summary_path = results_dir / out.get("summary_json", "summary.json")

    # --- resumable state ---
    start_epoch, global_step = 0, 0
    prior_elapsed = 0.0          # wall-clock seconds accumulated before this process
    track = {"best": (-math.inf if mode == "max" else math.inf),
             "no_improve": 0, "last_monitor": 0.0,
             "best_step": 0, "best_epoch": 0, "best_metrics": None}
    if resume is not None:
        info = load_checkpoint(resume, model, ckpt_dir, optimizer, scheduler,
                               scaler, device, restore_rng=True)
        start_epoch = info["epoch"] + 1
        global_step = info["global_step"]
        track["best"] = info["best_score"]
        prior_elapsed = info["elapsed_sec"]

    run_start = time.time()
    console_log_every = int(out["console_log_every"])
    wandb_log_every = int(cfg.get("wandb", {}).get("log_every", 50))   # train-metric cadence
    total_steps = epochs * len(train_loader)     # ETA denominator (full schedule)

    # debug-image saving (assurance that preprocessing/augmentation look right)
    _dbg = cfg.get("debug", {})
    debug_every = int(_dbg.get("save_images_every", 0) or 0)
    debug_dir = (results_dir / _dbg.get("dir", "debugged_images")) if debug_every > 0 else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[debug] saving a batch image every {debug_every} steps -> {debug_dir}")

    def elapsed() -> float:
        """Total training wall-clock seconds (continues across resumes)."""
        return prior_elapsed + (time.time() - run_start)

    def eta(gstep: int) -> float:
        """Estimated seconds remaining = avg sec/step so far x steps left."""
        if gstep <= 0:
            return float("inf")
        return elapsed() / gstep * max(0, total_steps - gstep)

    if resume is not None:
        steps_left = max(0, total_steps - global_step)
        pct = 100 * global_step / total_steps if total_steps else 0.0
        print("=" * 70)
        print("🔁 RESUMING FROM CHECKPOINT")
        print("-" * 70)
        print(f"   continue from   : epoch {start_epoch + 1}/{epochs}  (global step {global_step})")
        print(f"   steps done      : {global_step}/{total_steps}  ({pct:.1f}%)")
        print(f"   steps remaining : {steps_left}")
        print(f"   best so far     : {track['best']:.4f} ({monitor})")
        print(f"   elapsed so far  : {fmt_duration(prior_elapsed)} ({prior_elapsed:.0f}s)")
        print("=" * 70)

    def _summary(status: str) -> dict:
        return {
            "experiment": cfg.get("experiment", {}).get("name"),
            "status": status,                      # running | completed | early_stopped
            "monitor": monitor, "mode": mode,
            "best_value": track["best"] if math.isfinite(track["best"]) else None,
            "best_step": track["best_step"], "best_epoch": track["best_epoch"],
            "best_metrics": track["best_metrics"],
            "last_step": track.get("last_step", 0), "total_steps": total_steps,
            "elapsed_sec": round(elapsed(), 2),
            "elapsed_human": fmt_duration(elapsed()),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }

    # --- validation + checkpoint + early-stop, shared by step- and epoch-eval ---
    def on_eval(epoch: int, gstep: int) -> bool:
        val_loss, metrics = validate(model, val_loader, loss_fn, cfg, device, amp)
        macro = metrics["macro"]
        current = _monitor_value(macro, val_loss, monitor)
        track["last_monitor"] = current
        track["last_step"] = gstep
        _safe("val_log", val_logger.log,
              _flatten_val_row(epoch + 1, gstep, val_loss, metrics,
                               cfg["tasks"], elapsed(), eta(gstep)))

        print("-" * 70)
        print(f"  🔎 [VAL] epoch={epoch + 1} step={gstep}  elapsed={fmt_duration(elapsed())}  "
              f"ETA={fmt_duration(eta(gstep))}  val_loss={val_loss:.4f}")
        print(f"     📊 mean AUROC={macro['mean_auroc']:.4f}  mean AUPRC={macro['mean_auprc']:.4f}  "
              f"F1={macro['mean_f1']:.4f}  P={macro['mean_precision']:.4f}  "
              f"R={macro['mean_recall']:.4f}  Spec={macro['mean_specificity']:.4f}")
        for t in cfg["tasks"]:
            m = metrics["per_task"][t]
            print(f"        {t:<18} AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
                  f"(pos={m['n_pos']})")

        prev_best = track["best"]
        is_best = better(current, track["best"])
        if is_best:
            track["best"], track["no_improve"] = current, 0
            track["best_step"], track["best_epoch"] = gstep, epoch + 1
            track["best_metrics"] = {**macro, "val_loss": val_loss}
        else:
            track["no_improve"] += 1
        _safe("checkpoint", save_checkpoint, ckpt_dir, cfg, model, optimizer,
              scheduler, scaler, epoch, gstep, track["best"], is_best, device, elapsed())
        _safe("summary", _write_summary, summary_path, _summary("running"))
        if log_fn is not None:            # W&B: same scores as the val CSV (no system stats)
            wb = {"val/loss": val_loss}
            wb.update({f"val/{k}": v for k, v in macro.items()})
            for t in cfg["tasks"]:
                pm = metrics["per_task"][t]
                for key in ("auroc", "auprc", "f1", "precision", "recall", "specificity"):
                    wb[f"val/{key}/{t}"] = pm[key]
            _safe("wandb", log_fn, wb, gstep)
        if persist_fn is not None:        # e.g. Modal volume.commit() -> live updates
            _safe("persist", persist_fn)
        if is_best:
            gain = current - prev_best if math.isfinite(prev_best) else current
            sign = "+" if mode == "max" else ""
            print(f"     ⭐ NEW BEST {monitor}={current:.4f} ({sign}{gain:.4f})  💾 saved best.pt")
        else:
            print(f"     ⚠️  no improvement ({track['no_improve']}/{es_patience})  "
                  f"best {monitor}={track['best']:.4f}  💾 ckpt saved")
        print("-" * 70)

        return bool(es_enable and track["no_improve"] >= es_patience)

    # --- epoch loop ---
    stop = False
    for epoch in range(start_epoch, epochs):
        print("=" * 70)
        print(f"📚 EPOCH {epoch + 1}/{epochs}")
        print("=" * 70)
        global_step, stop = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            epoch, global_step, train_logger, amp, grad_clip, ev_steps, on_eval,
            console_log_every, elapsed, eta, total_steps, log_fn, wandb_log_every,
            cfg=cfg, debug_dir=debug_dir, debug_every=debug_every)

        if not stop and ev_epochs and ((epoch + 1) % int(ev_epochs) == 0):
            stop = on_eval(epoch, global_step)

        if scheduler is not None:
            if sched_needs_metric:
                scheduler.step(track["last_monitor"])
            else:
                scheduler.step()

        if stop:
            print(f"🛑 [early-stop] no improvement for {es_patience} validations — stopping.")
            break

    status = "early_stopped" if stop else "completed"
    _safe("summary", _write_summary, summary_path, _summary(status))
    if persist_fn is not None:
        _safe("persist", persist_fn)
    train_logger.close()
    val_logger.close()
    print("#" * 70)
    print(f"# 🏁 done ({status}). best {monitor} = {track['best']:.4f} "
          f"@ step {track['best_step']}  ⏱️  total train time = "
          f"{fmt_duration(elapsed())} ({elapsed():.0f}s)")
    print("#" * 70)
    return track["best"]


def run_experiment(cfg: dict, model, experiment_dir, resume=None, persist_fn=None, log_fn=None):
    """Public entry point. Tees ALL terminal output (stdout + stderr) of this run
    into <experiment>/<console_capture_dir>/<YYYY-MM-DD_HH-MM-SS>.txt so every run
    keeps a complete, timestamped record, then delegates to _run_experiment.
    Streams are always restored and the file closed — even if the run crashes
    (the traceback is captured first).

    resume is config-driven: if not passed explicitly, it is read from
    cfg['resume'] (null/'best'/'last'/<int step>/<path>). An explicit argument
    wins over the config."""
    experiment_dir = Path(experiment_dir)
    if resume is None:
        resume = cfg.get("resume")
    cap_dir = experiment_dir / cfg["output"]["console_capture_dir"]
    cap_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cap_path = cap_dir / f"{ts}.txt"

    try:                                   # UTF-8 on the real streams before teeing
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    orig_out, orig_err = sys.stdout, sys.stderr
    log_f = open(cap_path, "w", encoding="utf-8")
    sys.stdout = _Tee(orig_out, log_f)
    sys.stderr = _Tee(orig_err, log_f)
    print(f"📝 capturing full console output of this run -> {cap_path}")
    try:
        return _run_experiment(cfg, model, experiment_dir, resume, persist_fn, log_fn)
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_f.close()
        print(f"📝 saved run log -> {cap_path}")


# =============================================================================
# Section 5 — Modal helpers (cloud execution)
# All Modal usage is OPTIONAL and `import modal` is lazy (inside modal_image),
# so a plain local `python train.py` never requires the modal package. Each
# experiment's train.py assembles its app from these three helpers.
# =============================================================================

# Packages installed into the Modal container image. torch's default Linux wheel
# is CUDA-enabled; opencv-headless avoids the GUI libs we don't need on a server.
_MODAL_PIP = [
    "torch", "torchvision", "numpy", "pandas", "scikit-learn",
    "opencv-python-headless", "pillow", "pyyaml", "wandb",
]


def modal_image():
    """Build the Modal container image: pip deps + the whole training_scripts/
    source tree (code + YAML configs), excluding local run artifacts."""
    import modal
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*_MODAL_PIP)
        .add_local_dir(
            str(PKG_DIR), remote_path="/root/training_scripts",
            ignore=["**/results/**", "**/train_config/**", "**/__pycache__/**"],
        )
    )


def modal_resources(cfg: dict) -> dict:
    """Map cfg['modal'] to @app.function(...) resource kwargs (GB -> MB for memory)."""
    m = cfg["modal"]
    return dict(
        gpu=m["gpu"],
        cpu=m["cpu_cores"],
        memory=int(m["memory_gb"]) * 1024,
        timeout=int(m["timeout_seconds"]),
    )


def remote_cfg(cfg: dict) -> dict:
    """Copy of cfg with data paths repointed at the mounted Modal data volume."""
    rc = copy.deepcopy(cfg)
    m = cfg["modal"]
    rc["paths"]["data_root"] = m["remote_data_root"]
    rc["paths"]["data_dir"] = m["remote_data_dir"]
    return rc


if __name__ == "__main__":
    # Resolve the baseline experiment's config (config-loading demo).
    demo = PKG_DIR / "resnet50_without_clahe"
    cfg = load_config(demo)
