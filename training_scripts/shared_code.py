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
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
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
        elif val is None and isinstance(merged.get(key), dict):
            # an empty/None section in the override (e.g. a bare "output:") must
            # NOT wipe out the shared dict — keep the base value.
            continue
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


# -----------------------------------------------------------------------------
# Per-task uncertainty policy (head layout)
# Every binary arm uses ONE policy for all 5 tasks (labels.u_policy "ones"|"zeros"):
# a single sigmoid logit per task + BCE. The "mixed" arm instead gives each task
# its OWN policy via labels.per_task, and a task may be 3-class ("multiclass":
# {neg, pos, unc}) with a softmax+CE head. task_layout() turns cfg into the
# per-task plan used everywhere downstream (label encoding, head width, loss,
# inference). For any non-"mixed" u_policy it returns the original all-binary plan,
# so existing arms are byte-for-byte unchanged.
# -----------------------------------------------------------------------------
_BINARY_POLICIES = ("ones", "zeros")


def _check_policy(p, task: str) -> str:
    """Validate + normalize one task's policy string."""
    p = str(p).lower()
    if p not in _BINARY_POLICIES and p not in ("multiclass", "selftrained"):
        raise ValueError(f"labels.per_task[{task!r}]={p!r} "
                         f"(expected ones|zeros|multiclass|selftrained)")
    return p


def task_layout(cfg: dict) -> dict:
    """Resolve cfg -> per-task head plan. Returns a dict with:
        policies : list[str]   per task, 'ones'|'zeros' (binary) | 'multiclass'
        widths   : list[int]   logits per task (1 binary, 3 multiclass)
        slices   : list[slice] each task's columns in the logit vector
        n_logits : int         total output logits the head must produce
        mixed    : bool        True iff any task is multiclass (needs the mixed loss)
        per_task : bool        True iff policies vary by task (drives the per-task print)

    Every experiment specifies its own policy in its config.yaml; this resolves
    all cases uniformly:
      'ones' / 'zeros'  -> that binary policy for every task, BUT labels.per_task
                           (optional) overrides the policy of any task it names
                           (e.g. all U-Ones except Consolidation: 'zeros').
      'mixed'           -> labels.per_task MUST name every task (binary or
                           multiclass) — the fully explicit per-task spec.
    No per_task and a binary u_policy -> the original all-binary plan (existing
    arms unchanged)."""
    tasks = cfg["tasks"]
    lab = cfg.get("labels", {})
    u = str(lab.get("u_policy", "ones")).lower()
    per = lab.get("per_task") or {}
    unknown = [t for t in per if t not in tasks]
    if unknown:
        raise ValueError(f"labels.per_task names unknown task(s): {unknown}")
    if u == "mixed":
        missing = [t for t in tasks if t not in per]
        if missing:
            raise ValueError("labels.u_policy='mixed' requires labels.per_task to "
                             f"name every task; missing: {missing}")
        policies = [_check_policy(per[t], t) for t in tasks]
    elif u in _BINARY_POLICIES or u == "mask":
        # binary default for all tasks; per_task (optional) overrides named tasks.
        # 'mask' is a binary-WIDTH policy (1 logit/task) used for the TRAIN split
        # only (uncertain -> NaN, ignored by the loss); see build_labels.
        policies = [_check_policy(per[t], t) if t in per else u for t in tasks]
    else:
        raise ValueError(f"labels.u_policy={u!r} (expected ones|zeros|mask|mixed)")
    widths = [3 if p == "multiclass" else 1 for p in policies]
    slices, off = [], 0
    for w in widths:
        slices.append(slice(off, off + w))
        off += w
    return {"tasks": list(tasks), "policies": policies, "widths": widths,
            "slices": slices, "n_logits": off, "mixed": any(w == 3 for w in widths),
            "per_task": (u == "mixed") or len(set(policies)) > 1}


def num_output_logits(cfg: dict) -> int:
    """Total output logits the model head must produce for this cfg (5 for an
    all-binary arm; more when any task is multiclass — e.g. 9 for the u_mixed arm
    with two 3-class tasks)."""
    return task_layout(cfg)["n_logits"]


def finetune_ckpt_subdir(cfg: dict) -> str:
    """Subfolder under results/checkpoints that holds the CheXpert model's
    checkpoints. For a two-stage run (pretrain.enable) that's the fine-tune stage's
    folder; calibration/evaluation — which always score the CheXpert model — read
    from there. Returns '' for a single-stage run (the flat checkpoints/ folder),
    so every existing experiment is unchanged."""
    pre = cfg.get("pretrain") or {}
    if pre.get("enable"):
        return pre.get("finetune_stage_name", "finetune_chexpert")
    return ""


def build_labels(df: pd.DataFrame, tasks: list, policies="ones") -> pd.DataFrame:
    """Raw labels (1 / 0 / -1 / blank) -> per-task target column, per that task's
    policy. blanks/NaN -> the NEGATIVE class always (0) for every policy.
        'ones' / 'zeros'  (binary)   : value 0/1; -1 -> 1 / 0.   Float 0.0/1.0.
        'multiclass'      (3-class)  : class index {0=neg, 1=pos, 2=unc};
                                       0/blank -> 0, 1 -> 1, -1 -> 2.
    `policies` may be a single string (same policy for every task — the binary
    arms) or a list aligned with `tasks` (the mixed arm). Returns a
    (N, len(tasks)) float32 frame; multiclass columns hold the class index."""
    if isinstance(policies, str):
        policies = [policies] * len(tasks)
    if len(policies) != len(tasks):
        raise ValueError(f"policies len {len(policies)} != n_tasks {len(tasks)}")
    out = {}
    for t, pol in zip(tasks, policies):
        col = df[t]
        if pol == "multiclass":
            out[t] = col.replace(-1, 2).fillna(0)          # -1 -> unc(2); blank -> neg(0)
        elif pol in ("mask", "selftrained"):
            # uncertain (-1) -> NaN; blank -> 0 (certain negative); 0/1 pass through.
            #   'mask'        : NaN cells are IGNORED by the masked loss.
            #   'selftrained' : NaN cells are FILLED with the model's soft probability
            #                   in CheXpertDataset (train only); on val they stay NaN
            #                   and are handled by binarize_targets per the config
            #                   flag labels.val_ignore_uncertain (default: scored as
            #                   U-Ones, i.e. positive; or excluded if the flag is set).
            out[t] = col.fillna(0).replace(-1, np.nan)
        elif pol in _BINARY_POLICIES:
            enc = col.fillna(0)                            # blank -> 0
            out[t] = enc.replace(-1, 1 if pol == "ones" else 0)
        else:
            raise ValueError(f"unknown policy {pol!r} for task {t!r}")
    target = pd.DataFrame(out, index=df.index)[list(tasks)]
    return target.astype("float32")


def _val_ignore_uncertain(cfg: dict) -> bool:
    """Whether validation/calibration/eval should DROP uncertain rows from the
    metric. Config-driven via labels.val_ignore_uncertain (default False = keep
    them, scored as U-Ones). See binarize_targets."""
    return bool(cfg.get("labels", {}).get("val_ignore_uncertain", False))


def logits_to_probs(logits: torch.Tensor, layout: dict) -> torch.Tensor:
    """Model logits (B, n_logits) -> (B, n_tasks) per-task BINARY probability,
    ready for AUROC. Binary task: sigmoid(its single logit). Multiclass task:
    softmax over {neg,pos,unc}, drop p_unc, renormalize p_pos/(p_pos+p_neg).
    For an all-binary layout this is exactly sigmoid(logits)."""
    cols = []
    for pol, sl in zip(layout["policies"], layout["slices"]):
        if pol == "multiclass":
            sm = torch.softmax(logits[:, sl].float(), dim=1)   # [:,0]=neg [:,1]=pos [:,2]=unc
            cols.append(sm[:, 1] / (sm[:, 0] + sm[:, 1]).clamp_min(1e-12))
        else:
            cols.append(torch.sigmoid(logits[:, sl.start].float()))
    return torch.stack(cols, dim=1)


def binarize_targets(targets, layout: dict, ignore_uncertain: bool = False):
    """Encoded targets (N, n_tasks) -> (y_true_binary, exclude_mask), both
    (N, n_tasks). Binary 'ones'/'zeros' columns pass through (already 0/1 from
    build_labels: uncertain mapped to 1/0 at encode time) -> kept as-is. The
    only tasks whose VALIDATION handling this function decides are the ones that
    leave uncertain un-binarized: 'multiclass' (class index 2) and
    'mask'/'selftrained' (NaN on val, no soft target there).

    `ignore_uncertain` controls those uncertain val cells:
      False (DEFAULT): keep them, scored as POSITIVE (i.e. treat them as U-Ones
        for the metric regardless of the training policy). multiclass class-2 and
        NaN cells -> y_true = 1, nothing excluded. (ones/zeros already encode
        their own 1/0, so this only re-routes multiclass/mask/selftrained.)
      True: drop them from the metric via exclude_mask (multiclass class 2 and
        NaN cells flagged; y_true set to 0 but ignored because excluded).

    For an all-binary ('ones'/'zeros') layout there are no such cells, so the
    flag has no effect: the mask is all-False and y is unchanged either way."""
    y = np.asarray(targets, dtype=float).copy()
    nan_unc = np.isnan(y)                  # 'mask'/'selftrained' uncertain (val)
    mc_unc = np.zeros_like(nan_unc)        # 'multiclass' uncertain (class idx 2)
    for k, pol in enumerate(layout["policies"]):
        if pol == "multiclass":
            col = y[:, k]
            mc_unc[:, k] = (col == 2)
            y[:, k] = (col == 1).astype(float)
    unc = nan_unc | mc_unc
    if ignore_uncertain:
        y = np.nan_to_num(y, nan=0.0)      # excluded cells -> 0 (dropped via excl anyway)
        return y, unc
    # default: keep uncertain, score as positive (U-Ones for the metric).
    y[nan_unc] = 1.0
    y[mc_unc] = 1.0
    y = np.nan_to_num(y, nan=1.0)          # any residual NaN -> positive (defensive)
    return y, np.zeros_like(unc)


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
        # path_column is configurable so a stage with a different CSV schema can be
        # read by the SAME data layer (CheXpert uses 'Path'; ChestX-ray14 'path').
        self.paths = df[cfg["paths"].get("path_column", "Path")].tolist()
        self.layout = task_layout(cfg)            # per-task policy plan (binary or mixed)
        # .copy() -> writable array (silences the non-writable-tensor warning)
        lab = build_labels(df, self.tasks, self.layout["policies"])
        # 'selftrained' tasks: uncertain cells are NaN here -> fill (TRAIN only) with
        # the self-trained soft probability for that (image, task), joined by path.
        if split == "train" and "selftrained" in self.layout["policies"]:
            self._fill_selftrained(df, lab, cfg)
        self.labels = torch.from_numpy(lab.to_numpy().copy())
        self.use_clahe = bool(cfg["clahe"]["use_clahe"])
        self._clahe = (
            _get_clahe(cfg["clahe"]["clip_limit"], tuple(cfg["clahe"]["tile_grid"]))
            if self.use_clahe else None
        )
        self.train_transforms = build_train_transforms(cfg) if split == "train" else None

    def _fill_selftrained(self, df, lab, cfg):
        """In place: replace each 'selftrained' task's uncertain (NaN) train cells
        with the self-trained SOFT probability for that (image, task), read from
        labels.selftrained_csv (Path + one prob column per task) and joined on the
        path column. A path missing from the CSV falls back to U-Ones (1.0)."""
        pcol = cfg["paths"].get("path_column", "Path")
        st_csv = Path(cfg["paths"]["data_dir"]) / cfg["labels"]["selftrained_csv"]
        soft = pd.read_csv(st_csv)
        soft = soft[~soft[pcol].duplicated(keep="first")].set_index(pcol)
        df_paths = df[pcol].to_numpy()
        st_tasks = [t for t, p in zip(self.tasks, self.layout["policies"])
                    if p == "selftrained"]
        n_filled = n_missing = 0
        for t in st_tasks:
            if t not in soft.columns:
                raise ValueError(f"selftrained task {t!r} not a column in {st_csv}")
            col = np.array(lab[t], dtype=float)                    # writable copy
            nan_idx = np.where(np.isnan(col))[0]                    # uncertain cells
            mapped = soft[t].reindex(df_paths).to_numpy()[nan_idx]  # soft prob, aligned
            miss = np.isnan(mapped)
            mapped[miss] = 1.0                                      # fallback: U-Ones
            col[nan_idx] = mapped
            lab[t] = col
            n_filled += int((~miss).sum())
            n_missing += int(miss.sum())
        msg = f"[selftrained] filled {n_filled} uncertain cell(s) from {st_csv.name}"
        if n_missing:
            msg += f"  (WARNING: {n_missing} path(s) missing in CSV -> U-Ones=1)"
        print(msg)

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


class ResumableSampler(torch.utils.data.Sampler):
    """Deterministic shuffling sampler that supports a CLEAN mid-epoch resume.

    Each epoch's permutation is a pure function of (seed, epoch) — torch.randperm
    under a generator seeded `seed + epoch` — so it is identical whether or not
    the run was resumed (no dependence on how many epochs ran before). For a
    mid-epoch resume, `set_epoch(epoch, skip=k)` makes the NEXT iteration yield
    that epoch's permutation with the first `k` SAMPLE indices dropped — so the
    DataLoader workers never load the already-trained samples (no wasted decode).
    Because the dropped indices are just the head of the same permutation, the
    resumed epoch trains on exactly the tail it would have anyway. The skip clears
    after that one epoch; later epochs shuffle fully via their own seed+epoch.

    `__len__` stays the FULL epoch size so steps_per_epoch / the LR-schedule
    budget are unaffected; the resume epoch simply yields fewer batches."""

    def __init__(self, n: int, seed: int):
        self.n = int(n)
        self.seed = int(seed)
        self.epoch = 0
        self.skip = 0

    def set_epoch(self, epoch: int, skip: int = 0):
        """Select which epoch's permutation to produce next, and how many leading
        SAMPLE indices to drop (mid-epoch resume; 0 = full epoch)."""
        self.epoch = int(epoch)
        self.skip = max(0, int(skip))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        if self.skip:
            perm = perm[self.skip:]
            self.skip = 0                 # only the first (resume) epoch is truncated
        return iter(perm)

    def __len__(self):
        return self.n


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

    # Optional TRAIN-ONLY uncertainty policy (labels.train_u_policy). The val split
    # always uses the global labels.u_policy, so validation/calibration/eval are
    # unchanged. Used by the certain-only soft-label run: train_u_policy='mask'
    # (uncertain -> NaN, dropped by the masked loss) while val stays U-Ones.
    lab = cfg.get("labels", {})
    tr_pol = lab.get("train_u_policy")
    if tr_pol:
        tr_cfg = copy.deepcopy(cfg)
        tr_cfg["labels"]["u_policy"] = tr_pol
        train_ds = CheXpertDataset(train_df, tr_cfg, split="train")
        print(f"  [train-only override] u_policy '{lab.get('u_policy')}' -> "
              f"'{tr_pol}' for the TRAIN split (val stays '{lab.get('u_policy')}')")
    else:
        train_ds = CheXpertDataset(train_df, cfg, split="train")
    val_ds = CheXpertDataset(val_df, cfg, split="val")

    dl = cfg["dataloader"]
    g = torch.Generator()
    g.manual_seed(cfg["reproducibility"]["seed"])

    # train settings
    tr_bs = int(dl["batch_size"])
    tr_nw = int(dl["num_workers"])
    tr_pin = bool(dl["pin_memory"])
    tr_pf = dl["prefetch_factor"]
    tr_pw = bool(dl["persistent_workers"])
    # validation settings — each falls back to its train counterpart if omitted.
    # (val runs under no_grad -> can use a bigger batch; doesn't affect metrics.)
    va_bs = int(dl.get("val_batch_size") or tr_bs)
    va_nw = int(dl.get("val_num_workers", tr_nw))
    va_pin = bool(dl.get("val_pin_memory", tr_pin))
    va_pf = dl.get("val_prefetch_factor", tr_pf)
    va_pw = bool(dl.get("val_persistent_workers", tr_pw))

    def _kw(nw, pin, pf, pw):
        # prefetch_factor / persistent_workers are only valid when num_workers>0
        kw = dict(num_workers=nw, pin_memory=pin, worker_init_fn=seed_worker)
        if nw > 0:
            kw.update(prefetch_factor=pf, persistent_workers=pw)
        return kw

    # ResumableSampler (instead of shuffle=True) so a mid-epoch resume can skip
    # already-trained batches at the INDEX level — workers never load them. The
    # loader generator `g` still seeds the workers (base_seed) for augmentation.
    train_sampler = ResumableSampler(len(train_ds), cfg["reproducibility"]["seed"])
    train_loader = DataLoader(train_ds, batch_size=tr_bs, sampler=train_sampler,
                              drop_last=bool(dl["drop_last"]), generator=g,
                              **_kw(tr_nw, tr_pin, tr_pf, tr_pw))
    val_loader = DataLoader(val_ds, batch_size=va_bs, shuffle=False,
                            drop_last=False, **_kw(va_nw, va_pin, va_pf, va_pw))

    print("-" * 70)
    print(f"  train: bs={tr_bs}  workers={tr_nw}  pin={tr_pin}  prefetch={tr_pf}  "
          f"persistent={tr_pw}  drop_last={bool(dl['drop_last'])}")
    print(f"  val  : bs={va_bs}  workers={va_nw}  pin={va_pin}  prefetch={va_pf}  "
          f"persistent={va_pw}")
    print(f"  use_clahe={train_ds.use_clahe}  u_policy={cfg['labels']['u_policy']}  "
          f"augment (train): {'ON' if train_ds.train_transforms is not None else 'OFF'}")
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


def compute_metrics(y_true, y_prob, tasks: list, threshold=0.5, exclude_mask=None) -> dict:
    """Multi-label metrics from ground-truth (N,T) 0/1 and probabilities (N,T).

    Per task: AUROC, AUPRC (threshold-free) + F1 / precision / recall /
    specificity @ threshold, plus the positive count. Macro = mean over tasks
    (AUROC/AUPRC use nanmean so a task with a single class present — undefined
    AUROC -> NaN — is skipped instead of poisoning the average).
    Returns {"macro": {...}, "per_task": {task: {...}}}.

    `threshold` may be a single float (same cut for every task, e.g. 0.5 during
    in-training validation) OR a per-task sequence of length len(tasks) — the
    frozen per-task thresholds calibrated on the validation set. The threshold
    used for each task is recorded back into its per-task dict.

    `exclude_mask` (optional, (N,T) bool): True drops that (row, task) from the
    task's metric. Used by the mixed arm to exclude uncertain (-1) rows from a
    multiclass task's binary AUROC — they have no binary ground truth. The number
    dropped is reported per task as `n_excluded`. None / all-False -> no effect
    (every binary arm), so behavior there is unchanged.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
    thr = np.asarray(threshold, dtype=float)
    if thr.ndim == 0:
        thr_row = np.full(len(tasks), float(thr))
    else:
        thr_row = thr.reshape(-1)
        if len(thr_row) != len(tasks):
            raise ValueError(f"threshold vector len {len(thr_row)} != n_tasks {len(tasks)}")
    preds = (y_prob >= thr_row.reshape(1, -1)).astype(int)

    per_task = {}
    cols = {k: [] for k in ("auroc", "auprc", "f1", "precision", "recall", "specificity")}
    for k, t in enumerate(tasks):
        yt, yp, pr = y_true[:, k], y_prob[:, k], preds[:, k]
        n_excluded = 0
        if exclude_mask is not None:
            keep = ~exclude_mask[:, k]
            n_excluded = int((~keep).sum())
            yt, yp, pr = yt[keep], yp[keep], pr[keep]
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
                           recall=recall, specificity=specificity, n_pos=n_pos,
                           n_excluded=n_excluded, threshold=float(thr_row[k]))
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


def build_optimizer(model: nn.Module, cfg: dict, loss_fn=None) -> optim.Optimizer:
    """Optimizer from cfg['training']['optimizer']. Defaults to AdamW (when 'name'
    is absent or 'adamw' — every existing run); 'pesg' builds LibAUC's PESG, which
    the AUC-margin loss requires.

    AdamW : lr + weight_decay; fused CUDA kernel when performance.fused_optimizer
            and the params are on CUDA.
    PESG  : LibAUC's min-max optimizer for the AUC-margin loss. It jointly updates
            the weights AND the loss's adversarial variables (a, b, alpha), so it
            must be handed the SAME libauc loss object via `loss_fn`. Extra knobs
            (mode/momentum/margin/epoch_decay) come from the optimizer block. PESG
            reads its LR from param_groups, so the usual cosine/warmup scheduler
            drives it exactly like it drives AdamW."""
    o = cfg["training"]["optimizer"]
    name = str(o.get("name", "adamw")).lower()
    if name == "adamw":
        fused = bool(cfg.get("performance", {}).get("fused_optimizer", False)) \
            and next(model.parameters()).is_cuda
        return optim.AdamW(model.parameters(), lr=o["lr"],
                           weight_decay=o["weight_decay"], fused=fused)
    if name == "pesg":
        from libauc.optimizers import PESG
        # unwrap _AUCMLossWrapper -> the underlying libauc loss (holds a, b, alpha)
        aucm = getattr(loss_fn, "aucm", loss_fn)
        device = next(model.parameters()).device
        return PESG(model.parameters(), loss_fn=aucm,
                    lr=o["lr"], mode=str(o.get("mode", "sgd")).lower(),
                    momentum=float(o.get("momentum", 0.9)),
                    margin=float(o.get("margin", 1.0)),
                    epoch_decay=float(o.get("epoch_decay", 2.0e-3)),
                    weight_decay=o["weight_decay"], device=device)
    raise ValueError(f"unknown optimizer: {name!r}")


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, steps_per_epoch: int):
    """Build the LR scheduler. Returns (scheduler_or_None, mode):
        mode == "step"    -> step once per OPTIMIZER STEP (cosine / warmup)
        mode == "plateau" -> step once per validation with the monitored metric
        mode is None      -> no scheduler

    cosine: linear warmup over `warmup_epochs` (expressed in STEPS) from
    `warmup_start_factor`*lr up to lr, then cosine-anneal down to `min_lr` across
    the remaining steps. Stepping per step gives a smooth ramp + decay (not the
    once-per-epoch staircase that left the LR pinned during warmup)."""
    s = cfg["training"]["scheduler"]
    name = s["name"].lower()
    epochs = cfg["training"]["epochs"]
    total_steps = max(1, epochs * steps_per_epoch)

    if name == "none":
        return None, None
    if name == "cosine":
        warmup_steps = int(round(float(s.get("warmup_epochs", 0)) * steps_per_epoch))
        min_lr = s.get("min_lr", 0.0)
        start_factor = float(s.get("warmup_start_factor", 0.01))   # ~0 -> base lr
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr)
        if warmup_steps > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=start_factor, total_iters=warmup_steps)
            sched = optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_sched, cosine], milestones=[warmup_steps])
            return sched, "step"
        return cosine, "step"
    if name == "plateau":
        mode = cfg["training"]["early_stopping"].get("mode", "max")
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=s.get("plateau_patience", 2),
            min_lr=s.get("min_lr", 0.0))
        return sched, "plateau"
    raise ValueError(f"unknown scheduler: {name!r}")


class MixedTaskLoss(nn.Module):
    """Sum of per-task losses for a mixed-policy head. Each binary task contributes
    BCE-with-logits on its single logit vs its 0/1 (or 'selftrained' SOFT in [0,1])
    target; each multiclass task contributes 3-class cross-entropy on its
    {neg,pos,unc} logits vs the class index. Uncertain is a real CE class (kept in
    the loss); blanks were folded into the negative class upstream by build_labels.
    A NaN binary target (a 'selftrained' task's uncertain VAL row — no soft label)
    is masked out of that task's BCE. Per-task losses are batch means, summed over
    the tasks with equal weight ('sum of per-task losses')."""

    def __init__(self, layout: dict):
        super().__init__()
        self.policies = layout["policies"]
        self.slices = layout["slices"]

    def forward(self, logits, targets):
        total = logits.new_zeros(())
        for k, (pol, sl) in enumerate(zip(self.policies, self.slices)):
            tgt = targets[:, k]
            if pol == "multiclass":
                total = total + F.cross_entropy(logits[:, sl], tgt.long())
            else:
                lg = logits[:, sl.start]
                m = ~torch.isnan(tgt)            # drop NaN (selftrained uncertain val)
                if not bool(m.all()):
                    lg, tgt = lg[m], tgt[m]
                if tgt.numel() > 0:
                    total = total + F.binary_cross_entropy_with_logits(lg, tgt)
        return total


class _AUCMLossWrapper(nn.Module):
    """Adapter so the engine keeps calling loss_fn(logits, targets) for the
    AUC-margin loss. LibAUC's MultiLabelAUCMLoss expects PROBABILITIES, so this
    applies sigmoid to the logits first (BCE, by contrast, eats raw logits). The
    underlying libauc loss is exposed as `.aucm` so build_optimizer can hand the
    SAME object — with its a/b/alpha min-max variables — to the PESG optimizer.
    version 'v1' is the paper's AUCM; with auto=True it estimates each task's
    positive prior per-batch, so no imratio needs to be supplied."""

    def __init__(self, num_labels: int, margin: float = 1.0,
                 version: str = "v1", device=None):
        super().__init__()
        from libauc.losses import MultiLabelAUCMLoss
        self.aucm = MultiLabelAUCMLoss(num_labels=num_labels, margin=margin,
                                       version=version, device=device)

    def forward(self, logits, targets):
        return self.aucm(torch.sigmoid(logits), targets)


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits that IGNORES masked target entries (NaN). Used when the
    TRAIN split masks uncertain (-1) labels (labels.train_u_policy='mask'): those
    (sample, task) cells are NaN and dropped from the loss, so the mean is taken
    over the VALID cells only. With no NaN present (e.g. the U-Ones val split, or
    the official sets) it is exactly BCEWithLogitsLoss."""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight        # already on the right device (build_loss)

    def forward(self, logits, targets):
        mask = ~torch.isnan(targets)
        t = torch.nan_to_num(targets, nan=0.0)
        per = F.binary_cross_entropy_with_logits(
            logits, t, pos_weight=self.pos_weight, reduction="none")
        per = per[mask]
        if per.numel() == 0:                # whole batch masked -> 0 (keeps graph)
            return logits.sum() * 0.0
        return per.mean()


def build_loss(cfg: dict, device=None) -> nn.Module:
    """Loss for this cfg. Mixed-policy arm (any multiclass task) -> MixedTaskLoss
    (per-task BCE + 3-class CE, summed). 'auc_margin' -> LibAUC MultiLabelAUCMLoss
    (optimizes mean AUROC directly; pair with the PESG optimizer). Otherwise the
    standard multi-label BCE-with-logits, with an optional per-task pos_weight
    (list) moved onto `device` so it lands on the same device as the logits."""
    layout = task_layout(cfg)
    if layout["mixed"]:
        return MixedTaskLoss(layout)
    l = cfg["training"]["loss"]
    name = l["name"].lower()
    if name == "auc_margin":
        return _AUCMLossWrapper(
            num_labels=num_output_logits(cfg),
            margin=float(l.get("margin", 1.0)),
            version=str(l.get("version", "v1")),
            device=device)
    if name != "bce_with_logits":
        raise ValueError(f"unknown loss: {name!r}")
    pw = l.get("pos_weight")
    pos_weight = None
    if pw is not None:
        pos_weight = torch.tensor(pw, dtype=torch.float32)
        if device is not None:
            pos_weight = pos_weight.to(device)
    # Use the masked BCE (a no-op on NaN-free batches, so plain binary val/eval
    # are unchanged) whenever the targets can contain NaN:
    #   - train_u_policy='mask'  : uncertain TRAIN cells are NaN (ignored by loss).
    #   - any 'selftrained' task : uncertain VAL cells are NaN (no soft label there;
    #     TRAIN cells are filled with soft probs, so they are valid soft targets).
    # (A 'selftrained' task that also coexists with a multiclass task already takes
    #  the MixedTaskLoss path above, which handles the same NaN masking per task.)
    lab = cfg.get("labels", {})
    train_pol = str(lab.get("train_u_policy", lab.get("u_policy", "ones"))).lower()
    if train_pol == "mask" or "selftrained" in layout["policies"]:
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
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
                      epoch: int, epoch_step: int, stage_tag: str = None):
    """Save ONE random sample from the batch exactly as the model sees it
    (un-normalized back to [0,1] for viewing) -> debug_dir/e{epoch}_{step}.png.
    When a stage_tag is given (two-stage runs), it prefixes the filename so the
    two stages' debug images accumulate in the SAME folder without overwriting."""
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
    prefix = f"{stage_tag}_" if stage_tag else ""
    Image.fromarray(arr).save(debug_dir / f"{prefix}e{epoch}_{epoch_step}.png")


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
                     tasks: list, elapsed_sec: float, eta_sec: float,
                     stage_tag: str = None) -> dict:
    """Flatten validation metrics into one wide CSV row (macro + per-task). When a
    stage_tag is given (two-stage runs), a 'stage' column is added so the two
    stages' rows — stacked in the SAME val_log.csv — stay distinguishable. When it
    is None (every single-stage run) the column is omitted, so the schema is
    byte-identical to before."""
    row = {"step": gstep, "epoch": epoch}
    if stage_tag is not None:
        row["stage"] = stage_tag
    row.update({"elapsed_sec": round(elapsed_sec, 2),
                "eta_sec": round(eta_sec, 2) if math.isfinite(eta_sec) else "",
                "val_loss": round(val_loss, 6)})
    for k, v in metrics["macro"].items():
        row[k] = round(v, 6)
    for t in tasks:
        m = metrics["per_task"][t]
        for key in ("auroc", "auprc", "f1", "precision", "recall", "specificity",
                    "n_pos", "n_excluded"):
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


def _as_cpu_byte(t):
    """Coerce a saved RNG-state tensor back to a CPU ByteTensor. load_checkpoint
    loads with map_location=device, so on a CUDA run the saved CPU ByteTensors
    come back as CUDA tensors — which torch.set_rng_state / set_rng_state_all
    reject ('RNG state must be a torch.ByteTensor'). Move to CPU + uint8."""
    return t.cpu().to(torch.uint8) if torch.is_tensor(t) else t


def _restore_rng(state: dict, device):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_cpu_byte(state["torch"]))
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([_as_cpu_byte(s) for s in state["cuda"]])


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
    _pre = cfg.get("pretrain") or {}
    if _pre.get("enable"):
        print(f"  pretraining       : ENABLED  (Stage 1 ChestX-ray14 -> Stage 2 CheXpert)"
              f"  stages: {_pre.get('stage_name', 'pretrain_chestxray14')} -> "
              f"{_pre.get('finetune_stage_name', 'finetune_chexpert')}")
    print(f"  tasks ({len(cfg['tasks'])})         : {cfg['tasks']}")
    print("  --- paths --------------------------------------------------------")
    p = cfg["paths"]
    print(f"  data_root         : {p['data_root']}")
    print(f"  data_dir          : {p['data_dir']}")
    print(f"  train / val csv   : {p['train_csv']}  |  {p['val_csv']}")
    print("  --- data ---------------------------------------------------------")
    print(f"  u_policy          : {cfg['labels']['u_policy']}")
    _lay = task_layout(cfg)
    if _lay["per_task"]:
        print("  per-task u-policy : "
              + ", ".join(f"{t}={p}" for t, p in zip(cfg["tasks"], _lay["policies"])))
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
    print(f"  optimizer         : AdamW  lr={opt['lr']:.4e}  weight_decay={opt['weight_decay']}")
    print(f"  scheduler         : {sch['name']}  warmup_epochs={sch.get('warmup_epochs')}  "
          f"min_lr={sch.get('min_lr')}")
    _loss_desc = "mixed per-task (BCE + 3-class CE, summed)" if _lay["mixed"] else loss["name"]
    print(f"  loss              : {_loss_desc}  pos_weight={loss.get('pos_weight')}")
    print(f"  amp               : {tr['amp']}  grad_clip={tr['grad_clip']}")
    print(f"  early_stopping    : enable={es['enable']}  monitor={es['monitor']}  "
          f"mode={es['mode']}  patience={es['patience']}")
    _perf = cfg.get("performance", {})
    print(f"  performance       : channels_last={_perf.get('channels_last')}  "
          f"fused_adamw={_perf.get('fused_optimizer')}  compile={_perf.get('compile')}"
          f"  (mode={_perf.get('compile_mode')})")
    print("  --- evaluation / repro / output ----------------------------------")
    print(f"  metric (headline) : {ev['metric']}  (eval_level={ev['eval_level']})")
    print(f"  eval cadence      : every_epochs={ev['eval_every_epochs']}  "
          f"every_steps={ev['eval_every_steps']}")
    _sched = ev.get("eval_schedule")
    if _sched:
        _thr = int(_sched["threshold_epoch"])
        _bef = _sched.get("before_steps", ev.get("eval_every_steps"))
        _aft = _sched.get("after_steps", ev.get("eval_every_steps"))
        print(f"  eval schedule     : every {_bef} steps before epoch {_thr}, "
              f"every {_aft} steps from epoch {_thr} on (inclusive)")
    print(f"  seed              : {rep['seed']}  deterministic={rep['deterministic']}")
    print(f"  checkpointing     : periodic every {out.get('checkpoint_every_steps', 0)} steps "
          f"(0=off) + best.pt at validation")
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


def print_model_summary(model, cfg: dict):
    """Comprehensive model summary at startup: class, output logits, and the
    total / trainable / frozen parameter counts. Counting trainable separately
    means any later layer-freezing (e.g. freezing a pretrained backbone) shows
    up here automatically. Uses the unwrapped module so a torch.compile wrapper
    doesn't affect the figures."""
    m = _unwrap(model)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = (100.0 * trainable / total) if total else 0.0
    print("=" * 70)
    print("MODEL")
    print("=" * 70)
    print(f"  architecture      : {cfg.get('model', {}).get('name', type(m).__name__)}"
          f"  ({type(m).__name__})")
    print(f"  pretrained        : {cfg.get('model', {}).get('pretrained')}")
    _lay = task_layout(cfg)
    if _lay["mixed"]:
        widths = ", ".join(f"{t}:{w}" for t, w in zip(cfg["tasks"], _lay["widths"]))
        print(f"  output logits     : {_lay['n_logits']}  (mixed heads -> {widths})")
    else:
        print(f"  output logits     : {_lay['n_logits']}  (one per task)")
    print(f"  total params      : {total:,}  ({total / 1e6:.2f} M)")
    print(f"  trainable params  : {trainable:,}  ({trainable / 1e6:.2f} M, {pct:.1f}%)")
    print(f"  frozen params     : {frozen:,}  ({frozen / 1e6:.2f} M)")
    print("=" * 70)


def _gpu_mem_report(device, reset: bool = True) -> str:
    """Peak RESERVED GPU memory (GB) for the window since the last call that reset.
    Reading max_memory_reserved() then resetting the peak stats yields the peak
    over the interval between this print and the previous one — the OOM-relevant
    footprint. Returns '' on CPU (no-op). Reserved (not allocated) is used because
    that's what the caching allocator actually holds from the GPU."""
    if device is None or device.type != "cuda":
        return ""
    peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return f"  gpu_mem(peak)={peak:.2f}GB"


@torch.no_grad()
def validate(model, val_loader, loss_fn, cfg, device, amp: bool,
             channels_last: bool = False) -> tuple:
    """Full pass over the val loader -> (val_loss, metrics dict). No grad, eval mode.
    Predictions are collapsed to per-task binary probabilities via the cfg head
    layout (sigmoid for binary tasks, renormalized p_pos for multiclass), and
    uncertain rows on multiclass tasks are excluded from those tasks' metrics."""
    layout = task_layout(cfg)
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.append(labels.detach().cpu().numpy())
        ps.append(logits_to_probs(logits, layout).float().detach().cpu().numpy())
    y_enc = np.concatenate(ys)
    p = np.concatenate(ps)
    y_true, excl = binarize_targets(y_enc, layout, _val_ignore_uncertain(cfg))
    val_loss = total_loss / max(1, n)
    metrics = compute_metrics(y_true, p, cfg["tasks"], exclude_mask=excl)
    return val_loss, metrics


def _unwrap(model):
    """The underlying module behind a torch.compile wrapper (or the model itself),
    so checkpoints always use plain (un-prefixed) state-dict keys."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(ckpt_dir: Path, cfg, model, optimizer, scheduler, scaler,
                    epoch: int, global_step: int, best_score: float,
                    device, elapsed_sec: float,
                    rolling: bool = False, is_best: bool = False,
                    track: dict = None):
    """Write a fully-resumable checkpoint. Two independent triggers:
      rolling=True : write ckpt_step{N}.pt and prune to keep_last_n  (periodic)
      is_best=True : (over)write best.pt                              (at validation)
    Both can be true; either writes the same complete state.
    `track` is the full early-stopping / best-so-far tracking dict (no_improve,
    best_step/epoch, best_metrics, delta baselines); it is saved verbatim so a
    resumed run continues the early-stopping counter exactly where it left off."""
    out = cfg["output"]
    state = {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "elapsed_sec": elapsed_sec,
        "track": dict(track) if track is not None else None,
        "rng": _rng_state(device),
        "cfg": cfg,
    }
    if rolling:
        torch.save(state, ckpt_dir / f"ckpt_step{global_step}.pt")
        keep = int(out["keep_last_n"])      # prune to the newest keep_last_n
        ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"),
                       key=lambda p: int(p.stem.replace("ckpt_step", "")))
        for old in ckpts[:-keep] if keep > 0 else ckpts:
            print(f"🗑️  deleting old checkpoint: {old}")
            old.unlink(missing_ok=True)
    if is_best:
        torch.save(state, ckpt_dir / out["best_name"])


def load_checkpoint(resume, model, ckpt_dir: Path, optimizer=None, scheduler=None,
                    scaler=None, device=None, restore_rng: bool = True) -> dict:
    """Resume from a checkpoint. `resume` may be 'best', 'last', or a path.
    Restores model (+ optimizer/scheduler/scaler/rng if given). Returns the
    bookkeeping dict {epoch, global_step, best_score}."""
    path = _resolve_resume(resume, ckpt_dir)
    print(f"[resume] loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    _unwrap(model).load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        _restore_rng(ckpt.get("rng"), device)
    info = {"epoch": ckpt["epoch"], "global_step": ckpt["global_step"],
            "best_score": ckpt["best_score"], "elapsed_sec": ckpt.get("elapsed_sec", 0.0),
            "track": ckpt.get("track")}                  # full early-stopping state (or None)
    ni = (info["track"] or {}).get("no_improve")
    ni_str = f" no_improve={ni}" if ni is not None else " (no saved track — legacy ckpt)"
    print(f"[resume] restored -> epoch={info['epoch']} step={info['global_step']} "
          f"best_score={info['best_score']:.4f} elapsed={info['elapsed_sec']:.0f}s{ni_str}")
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
                    cfg: dict = None, debug_dir=None, debug_every: int = 0,
                    channels_last: bool = False,
                    checkpoint_fn=None, checkpoint_every: int = 0,
                    scheduler=None, n_epochs: int = 0, skip_batches: int = 0,
                    stage_tag: str = None, wandb_prefix: str = "",
                    phase: str = None, wandb_step_offset: int = 0) -> tuple:
    """One epoch of training. Logs every step (loss/lr/grad_norm/throughput/elapsed/eta)
    and fires `on_eval(epoch, step)` every eval_every_steps. Returns (global_step, stop).

    `skip_batches` > 0 (only on the first epoch after a MID-epoch resume) fast-
    forwards past that many already-completed batches so training continues at the
    exact point the checkpoint was taken, keeping global_step / the LR schedule /
    the total step budget aligned."""
    model.train()
    clip = grad_clip if grad_clip else math.inf      # inf -> measure norm, no clip
    stop = False
    n_batches = len(loader)
    # The ResumableSampler has already dropped the first `skip_batches` batches at
    # the index level (workers never loaded them), so enumerate() starts directly
    # at the resume point. We only OFFSET the displayed per-epoch batch number so
    # the logs still read as batch (skip_batches+1) .. n_batches.
    if skip_batches > 0:
        print(f"  ⏩ resuming mid-epoch: sampler fast-forwarded past the first "
              f"{skip_batches}/{n_batches} already-trained batches (no reload) — "
              f"training continues at batch {skip_batches + 1}.")

    for i, (imgs, labels) in enumerate(loader, start=1):
        ep_step = i + skip_batches       # 1-indexed batch number within the FULL epoch
        t0 = time.time()
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
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
        if scheduler is not None:        # cosine/warmup: step once per optimizer step
            scheduler.step()

        global_step += 1
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        imgs_per_s = imgs.size(0) / dt if dt > 0 else 0.0
        el = elapsed_fn()
        eta = eta_fn(global_step)
        train_row = {"step": global_step, "epoch": epoch + 1}   # 1-indexed for humans
        if stage_tag is not None:
            train_row["stage"] = stage_tag                      # two-stage: which stage
        train_row.update({
            "elapsed_sec": round(el, 2),
            "eta_sec": round(eta, 2) if math.isfinite(eta) else "",
            "train_loss": round(loss.item(), 6),
            "lr": lr, "grad_norm": round(float(grad_norm), 6),
            "imgs_per_s": round(imgs_per_s, 1), "sec_per_step": round(dt, 4),
        })
        _safe("train_log", train_logger.log, train_row)
        if global_step % console_log_every == 0 or i == 1:
            cur_loss = loss.item()
            # delta vs the previous CONSOLE-printed train loss (persists across
            # epochs via a function attribute); empty on the very first print.
            d_loss = _fmt_delta(cur_loss, getattr(train_one_epoch, "_last_print_loss", None))
            train_one_epoch._last_print_loss = cur_loss
            # `phase` (e.g. "1/2") marks the two-stage phase on every step line, right
            # after the epoch; empty for a single-stage run. The step print spans two
            # lines (it got long) with a thin rule between consecutive step prints.
            phase_tag = f"  (Phase: {phase})" if phase else ""
            print("─" * 70)                         # thin continuous separator
            print(f"  Epoch {epoch + 1}/{n_epochs}{phase_tag}  epoch_step: {ep_step}/{n_batches}  "
                  f"global_step: {global_step}/{total_steps}")
            print(f"     loss={cur_loss:.4f}{d_loss}  lr={lr:.4e}  gnorm={float(grad_norm):.2f}  "
                  f"{imgs_per_s:.0f} img/s  elapsed={fmt_duration(el)}  ETA={fmt_duration(eta)}"
                  f"{_gpu_mem_report(device)}")

        if log_fn is not None and (global_step % wandb_log_every == 0 or i == 1):
            _safe("wandb", log_fn, {
                f"{wandb_prefix}train/loss": loss.item(), f"{wandb_prefix}train/lr": lr,
                f"{wandb_prefix}train/grad_norm": float(grad_norm),
                f"{wandb_prefix}train/imgs_per_s": imgs_per_s,
            }, global_step + wandb_step_offset)

        if debug_dir is not None and debug_every > 0 and global_step % debug_every == 0:
            _safe("debug_image", _save_debug_image, imgs, cfg, debug_dir,
                  epoch + 1, ep_step, stage_tag)

        if checkpoint_fn is not None and checkpoint_every > 0 \
                and global_step % checkpoint_every == 0:
            checkpoint_fn(epoch, global_step)        # periodic rolling ckpt (no eval)

        if eval_every_steps and global_step % eval_every_steps == 0:
            stop = on_eval(epoch, global_step)
            model.train()                            # back to train mode after eval
            if stop:
                break
    return global_step, stop


def _resolve_eval_every_steps(cfg: dict, epoch: int):
    """Effective `eval_every_steps` for a 0-indexed `epoch`. Default is the flat
    evaluation.eval_every_steps. If evaluation.eval_schedule is present, the step
    cadence is PHASED by epoch so early training (where the model is clearly still
    improving) validates rarely and later training validates often:
        epochs BEFORE threshold_epoch     -> validate every `before_steps`
        epoch threshold_epoch and onward  -> validate every `after_steps`
    threshold_epoch is 1-indexed and INCLUSIVE (threshold_epoch=2 => epochs 2+ use
    after_steps; epoch 1 uses before_steps). Missing before_/after_steps fall back
    to the flat eval_every_steps. No schedule -> unchanged behavior."""
    ev = cfg["evaluation"]
    base = ev.get("eval_every_steps")
    sched = ev.get("eval_schedule")
    if not sched:
        return base
    thr = int(sched["threshold_epoch"])               # 1-indexed, inclusive
    before = sched.get("before_steps", base)
    after = sched.get("after_steps", base)
    return after if (epoch + 1) >= thr else before


def _fmt_delta(curr, prev) -> str:
    """' (+0.0138)' / ' (-0.0042)' vs the previous validation, '' on the first
    validation (no baseline) or when either value is NaN."""
    if prev is None or curr is None or math.isnan(curr) or math.isnan(prev):
        return ""
    return f" ({curr - prev:+.4f})"


def _run_experiment(cfg: dict, model, experiment_dir, resume=None, persist_fn=None, log_fn=None,
                    *, ckpt_subdir: str = "", stage_tag: str = None,
                    stage_header: str = None, init_weights_from=None, phase: str = None,
                    wandb_state: dict = None):
    """End-to-end training for ONE stage. Identical machinery for every run; only
    the injected `model` and cfg overrides differ. Trains on the cfg's train_csv,
    validates on its val_csv, checkpoints (rolling + best by mean AUROC), early-stops.
    NOTE: official 200/500 reporting is intentionally NOT done here.
    Wrapped by run_experiment(), which tees all console output to a per-run txt.

    Stage-aware (optional, all default to the original single-stage behavior):
      ckpt_subdir       : subfolder under results/checkpoints to write this stage's
                          checkpoints into ('' -> the flat folder; every existing run).
      stage_tag         : labels the stage in the train/val CSV ('stage' column),
                          the W&B namespace, the console banners, AND the key under
                          which this stage's snapshot is nested inside the SINGLE
                          shared training_summary.json (None -> flat, as before).
      stage_header      : one-line stage description printed in the run header.
      init_weights_from : path to a checkpoint whose MODEL weights (backbone + head)
                          are loaded before training — used to seed the CheXpert
                          fine-tune stage from the ChestX-ray14 pretrain best.pt."""
    experiment_dir = Path(experiment_dir)
    rep = cfg["reproducibility"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_prefix = f"{stage_tag}/" if stage_tag else ""
    # W&B step must be monotonic across the WHOLE run, but each stage's global_step
    # restarts at 0. Offset this stage's W&B step past everything logged so far
    # (carried in wandb_state["offset"]); CSV logs keep the raw per-stage step.
    wandb_offset = int(wandb_state.get("offset", 0)) if wandb_state else 0

    print("#" * 70)
    print(f"# 🚀 run_experiment: {cfg.get('experiment', {}).get('name', '<unnamed>')}")
    if stage_header:
        print(f"# 🧬 {stage_header}")
    print(f"# 🖥️  device={device}  epochs={cfg['training']['epochs']}")
    print("#" * 70)

    set_seed(rep["seed"], rep["deterministic"])
    print_config(cfg)

    # --- output layout ---
    out = cfg["output"]
    results_dir = experiment_dir / out["run_dir"]
    # Each stage keeps its OWN checkpoints subfolder (ckpt_subdir); single-stage
    # runs pass '' and write straight into results/checkpoints (unchanged).
    ckpt_dir = results_dir / out["checkpoints_dir"]
    if ckpt_subdir:
        ckpt_dir = ckpt_dir / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # train/val CSVs + debug images + plots are SHARED across stages (rows stack,
    # disambiguated by the 'stage' column); only checkpoints + the summary split.
    train_logger = CSVLogger(results_dir / out["train_log_csv"])
    val_logger = CSVLogger(results_dir / out["val_log_csv"])
    print(f"[paths] results -> {results_dir}")
    print(f"[paths] checkpoints -> {ckpt_dir}")

    # --- data / model / optim ---
    train_loader, val_loader = make_loaders(cfg)
    model = model.to(device)
    print_model_summary(model, cfg)

    # Seed this stage's model from a prior stage's weights (e.g. ChestX-ray14
    # pretrain best.pt -> CheXpert fine-tune). Loaded into the un-compiled module
    # BEFORE channels_last/compile so the plain state-dict keys match. Skipped when
    # resuming (load_checkpoint below restores the full state, weights included).
    if init_weights_from is not None and resume is None:
        print("=" * 70)
        print("🧬 [stage-init] initializing model from a previous stage's checkpoint")
        print(f"   source : {init_weights_from}")
        _ck = torch.load(init_weights_from, map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(_ck["model"])
        print(f"   ✅ loaded full model (backbone + head)  "
              f"(pretrain step={_ck.get('global_step')}, epoch={_ck.get('epoch')}, "
              f"best_score={_ck.get('best_score')})")
        print("=" * 70)
        del _ck

    # performance options (CUDA-only)
    perf = cfg.get("performance", {})
    # TF32: let fp32 matmuls use tensor cores (Ampere+). Reproducibility-neutral
    # (still deterministic) and lossless in practice for CNN training; under AMP
    # the heavy ops are already fp16, so this only speeds the residual fp32 path.
    tf32 = device.type == "cuda"
    if tf32:
        torch.set_float32_matmul_precision("high")
    channels_last = bool(perf.get("channels_last")) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    do_compile = bool(perf.get("compile")) and device.type == "cuda"
    if do_compile:
        model = torch.compile(model, mode=perf.get("compile_mode", "default"))

    # Loss BEFORE optimizer: PESG (AUC-margin) needs the loss object's a/b/alpha.
    loss_fn = build_loss(cfg, device)
    optimizer = build_optimizer(model, cfg, loss_fn)
    scheduler, sched_mode = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    _opt_name = str(cfg["training"]["optimizer"].get("name", "adamw")).upper()
    print(f"[setup] optimizer={_opt_name}  scheduler={cfg['training']['scheduler']['name']}  amp={amp}")
    print(f"[setup] channels_last={channels_last}  compile={do_compile}"
          f"{' (mode=' + perf.get('compile_mode', 'default') + ')' if do_compile else ''}"
          f"  fused_adamw={bool(perf.get('fused_optimizer')) and device.type == 'cuda'}"
          f"  tf32={tf32}")

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
    summary_path = results_dir / out.get("summary_json", "training_summary.json")

    # --- resumable state ---
    start_epoch, global_step, resume_skip = 0, 0, 0
    steps_per_epoch = len(train_loader)
    prior_elapsed = 0.0          # wall-clock seconds accumulated before this process
    track = {"best": (-math.inf if mode == "max" else math.inf),
             "no_improve": 0, "last_monitor": 0.0,
             "best_step": 0, "best_epoch": 0, "best_metrics": None,
             "prev_macro": None, "prev_per_task": None, "prev_val_loss": None}
    if resume is not None:
        info = load_checkpoint(resume, model, ckpt_dir, optimizer, scheduler,
                               scaler, device, restore_rng=True)
        global_step = info["global_step"]
        prior_elapsed = info["elapsed_sec"]
        # global_step is the single source of truth for WHERE we are: this handles
        # both epoch-boundary and MID-epoch checkpoints correctly. (The old code
        # used saved_epoch + 1, which skipped the rest of the epoch for a mid-epoch
        # checkpoint.) start_epoch = completed full epochs; resume_skip = batches
        # already done within the current epoch (fast-forwarded in train_one_epoch).
        start_epoch = global_step // steps_per_epoch
        resume_skip = global_step % steps_per_epoch
        track["best"] = info["best_score"]
        # Restore the FULL early-stopping / best-so-far state so the patience
        # counter (no_improve), best_step/epoch, best_metrics and the delta
        # baselines continue exactly where they left off. Legacy checkpoints
        # without a saved track fall back to best_score only (no_improve=0).
        if info.get("track"):
            track.update(info["track"])

    run_start = time.time()
    console_log_every = int(out["console_log_every"])
    wandb_log_every = int(cfg.get("wandb", {}).get("log_every", 50))   # train-metric cadence
    ckpt_every = int(out.get("checkpoint_every_steps", 0) or 0)         # periodic rolling ckpt
    total_steps = epochs * len(train_loader)     # ETA denominator (full schedule)
    print(f"[checkpoint] periodic rolling checkpoint every "
          f"{ckpt_every if ckpt_every > 0 else 'OFF'} steps; best.pt at validation")

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
        _midep = (f"  (mid-epoch: continue at batch {resume_skip + 1}/{steps_per_epoch})"
                  if resume_skip else "  (at epoch boundary)")
        print(f"   continue from   : epoch {start_epoch + 1}/{epochs}  (global step {global_step}){_midep}")
        print(f"   steps done      : {global_step}/{total_steps}  ({pct:.1f}%)")
        print(f"   steps remaining : {steps_left}")
        print(f"   best so far     : {track['best']:.4f} ({monitor})  "
              f"@ step {track['best_step']} epoch {track['best_epoch']}")
        print(f"   early-stop count: {track['no_improve']}/{es_patience} "
              f"(no-improvement validations; resumes from here)")
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

    def _persist_summary(status: str):
        """Write the best-so-far snapshot. Single-stage runs keep the original flat
        JSON. Two-stage runs share ONE training_summary.json: each stage's snapshot
        is nested under stages[<stage_tag>], so the file always holds BOTH stages
        (merged, never overwriting the other stage's record)."""
        snap = _summary(status)
        if stage_tag is None:
            _write_summary(summary_path, snap)
            return
        try:
            doc = json.load(open(summary_path, encoding="utf-8")) if summary_path.exists() else {}
            if not isinstance(doc.get("stages"), dict):
                doc = {}
        except Exception:
            doc = {}
        doc.setdefault("stages", {})[stage_tag] = snap
        doc["experiment"] = snap["experiment"]
        doc["updated"] = snap["updated"]
        _write_summary(summary_path, doc)

    # --- validation + checkpoint + early-stop, shared by step- and epoch-eval ---
    def on_eval(epoch: int, gstep: int) -> bool:
        print("=" * 70)
        _stg = f"[{stage_tag}] " if stage_tag else ""
        print(f"🔎 {_stg}VALIDATION START — epoch {epoch + 1}/{epochs}  step {gstep}  "
              f"({len(val_loader)} batches, {len(val_loader.dataset)} images) ...")
        val_loss, metrics = validate(model, val_loader, loss_fn, cfg, device, amp, channels_last)
        macro = metrics["macro"]
        current = _monitor_value(macro, val_loss, monitor)
        track["last_monitor"] = current
        track["last_step"] = gstep
        _safe("val_log", val_logger.log,
              _flatten_val_row(epoch + 1, gstep, val_loss, metrics,
                               cfg["tasks"], elapsed(), eta(gstep), stage_tag))

        d_val_loss = _fmt_delta(val_loss, track["prev_val_loss"])   # vs last validation
        print("-" * 70)
        print(f"  🔎 [VAL] epoch={epoch + 1}/{epochs} step={gstep}  elapsed={fmt_duration(elapsed())}  "
              f"ETA={fmt_duration(eta(gstep))}  val_loss={val_loss:.4f}{d_val_loss}"
              f"{_gpu_mem_report(device)}")
        pm = track["prev_macro"]            # previous validation's macro (None on 1st)
        def _d(key):                        # delta vs last validation for a macro key
            return _fmt_delta(macro[key], pm[key] if pm is not None else None)
        print(f"     📊 mean AUROC={macro['mean_auroc']:.4f}{_d('mean_auroc')}  "
              f"mean AUPRC={macro['mean_auprc']:.4f}{_d('mean_auprc')}  "
              f"F1={macro['mean_f1']:.4f}{_d('mean_f1')}  "
              f"P={macro['mean_precision']:.4f}{_d('mean_precision')}  "
              f"R={macro['mean_recall']:.4f}{_d('mean_recall')}  "
              f"Spec={macro['mean_specificity']:.4f}{_d('mean_specificity')}")
        ppt = track["prev_per_task"]        # previous validation's per-task metrics
        for t in cfg["tasks"]:
            m = metrics["per_task"][t]
            pt = ppt.get(t) if ppt is not None else None
            d_auroc = _fmt_delta(m['auroc'], pt['auroc'] if pt is not None else None)
            d_auprc = _fmt_delta(m['auprc'], pt['auprc'] if pt is not None else None)
            n_excl = m.get('n_excluded', 0)
            excl_str = f"  excl={n_excl}" if n_excl else ""   # uncertain rows dropped (multiclass)
            print(f"        {t:<18} AUROC={m['auroc']:.4f}{d_auroc}  "
                  f"AUPRC={m['auprc']:.4f}{d_auprc}  (pos={m['n_pos']}{excl_str})")
        track["prev_macro"] = dict(macro)            # baseline for next validation
        track["prev_per_task"] = {t: dict(metrics["per_task"][t]) for t in cfg["tasks"]}
        track["prev_val_loss"] = val_loss

        prev_best = track["best"]
        is_best = better(current, track["best"])
        if is_best:
            track["best"], track["no_improve"] = current, 0
            track["best_step"], track["best_epoch"] = gstep, epoch + 1
            track["best_metrics"] = {**macro, "val_loss": val_loss}
        else:
            track["no_improve"] += 1
        _safe("best-checkpoint", save_checkpoint, ckpt_dir, cfg, model, optimizer,
              scheduler, scaler, epoch, gstep, track["best"], device, elapsed(),
              rolling=False, is_best=is_best, track=track)
        _safe("summary", _persist_summary, "running")
        if log_fn is not None:            # W&B: same scores as the val CSV (no system stats)
            wb = {f"{wandb_prefix}val/loss": val_loss}
            wb.update({f"{wandb_prefix}val/{k}": v for k, v in macro.items()})
            for t in cfg["tasks"]:
                pm = metrics["per_task"][t]
                for key in ("auroc", "auprc", "f1", "precision", "recall", "specificity"):
                    wb[f"{wandb_prefix}val/{key}/{t}"] = pm[key]
            _safe("wandb", log_fn, wb, gstep + wandb_offset)
        if persist_fn is not None:        # e.g. Modal volume.commit() -> live updates
            _safe("persist", persist_fn)
        if is_best:
            gain = current - prev_best if math.isfinite(prev_best) else current
            sign = "+" if mode == "max" else ""
            print(f"     ⭐ NEW BEST {monitor}={current:.4f} ({sign}{gain:.4f})  💾 saved best.pt")
        else:
            print(f"     ⚠️  no improvement ({track['no_improve']}/{es_patience})  "
                  f"best {monitor}={track['best']:.4f}")
        print("-" * 70)

        return bool(es_enable and track["no_improve"] >= es_patience)

    # --- periodic rolling checkpoint, INDEPENDENT of validation ---
    def save_periodic(epoch: int, gstep: int):
        _safe("checkpoint", save_checkpoint, ckpt_dir, cfg, model, optimizer,
              scheduler, scaler, epoch, gstep, track["best"], device, elapsed(),
              rolling=True, is_best=False, track=track)
        _safe("summary", _persist_summary, "running")
        if persist_fn is not None:
            _safe("persist", persist_fn)
        print(f"  💾 periodic checkpoint @ step {gstep} "
              f"(rolling, keep_last_n={out['keep_last_n']})")

    # --- epoch loop ---
    stop = False
    for epoch in range(start_epoch, epochs):
        print("=" * 70)
        _stg = f"  [{stage_tag}]" if stage_tag else ""
        print(f"📚 EPOCH {epoch + 1}/{epochs}{_stg}")
        print("=" * 70)
        # only the first epoch after a mid-epoch resume fast-forwards already-done
        # batches; every later epoch starts clean.
        skip = resume_skip if epoch == start_epoch else 0
        # tell the sampler which epoch's permutation to draw and how many leading
        # samples to drop, so the workers never load already-trained batches.
        _samp = getattr(train_loader, "sampler", None)
        if hasattr(_samp, "set_epoch"):
            _samp.set_epoch(epoch, skip * int(train_loader.batch_size))
        # phased validation cadence: resolve THIS epoch's eval_every_steps
        # (evaluation.eval_schedule splits coarse early vs fine later; absent ->
        # the flat ev_steps, so existing runs are unchanged).
        ev_steps_epoch = _resolve_eval_every_steps(cfg, epoch)
        global_step, stop = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            epoch, global_step, train_logger, amp, grad_clip, ev_steps_epoch, on_eval,
            console_log_every, elapsed, eta, total_steps, log_fn, wandb_log_every,
            cfg=cfg, debug_dir=debug_dir, debug_every=debug_every,
            channels_last=channels_last,
            checkpoint_fn=save_periodic, checkpoint_every=ckpt_every,
            scheduler=(scheduler if sched_mode == "step" else None),
            n_epochs=epochs, skip_batches=skip,
            stage_tag=stage_tag, wandb_prefix=wandb_prefix, phase=phase,
            wandb_step_offset=wandb_offset)

        if not stop and ev_epochs and ((epoch + 1) % int(ev_epochs) == 0):
            stop = on_eval(epoch, global_step)

        # plateau steps once per epoch on the monitored metric; "step"-mode
        # schedulers (cosine/warmup) are stepped inside train_one_epoch.
        if scheduler is not None and sched_mode == "plateau":
            scheduler.step(track["last_monitor"])

        if stop:
            print(f"🛑 [early-stop] no improvement for {es_patience} validations — stopping.")
            break

    status = "early_stopped" if stop else "completed"
    _safe("summary", _persist_summary, status)
    if persist_fn is not None:
        _safe("persist", persist_fn)
    train_logger.close()
    val_logger.close()
    print("#" * 70)
    print(f"# 🏁 done ({status}). best {monitor} = {track['best']:.4f} "
          f"@ step {track['best_step']}  ⏱️  total train time = "
          f"{fmt_duration(elapsed())} ({elapsed():.0f}s)")
    print("#" * 70)
    # advance the shared W&B step cursor so the NEXT stage logs strictly after this
    # one (keeps the single W&B run's step monotonic across stages).
    if wandb_state is not None:
        wandb_state["offset"] = wandb_offset + global_step
    return track["best"]


# -----------------------------------------------------------------------------
# Two-stage training (optional): ChestX-ray14 pre-train -> CheXpert fine-tune.
# Driven entirely by an optional `pretrain:` block in an experiment's config.yaml.
# When that block is absent or pretrain.enable is false, _run_all_stages collapses
# to a single _run_experiment call with the original defaults — so every existing
# run is byte-for-byte unchanged.
# -----------------------------------------------------------------------------

def _build_pretrain_cfg(cfg: dict, pre: dict) -> dict:
    """Derive the Stage-1 (ChestX-ray14) config from the experiment cfg + its
    `pretrain` block. Everything not named in the block is INHERITED from cfg
    (image geometry, augmentation, dataloader, performance, output, repro, modal,
    wandb), so the only differences are the stage's data, tasks and (optionally)
    its training/evaluation/labels/clahe overrides. The pretrain block itself is
    dropped from the derived cfg so the pretrain stage can't recurse."""
    pc = copy.deepcopy(cfg)
    pc.pop("pretrain", None)
    pc["tasks"] = list(pre["tasks"])                          # CXR14 column names
    pc["paths"] = _deep_merge(cfg["paths"], pre.get("paths", {}))
    for key in ("labels", "clahe", "training", "evaluation"):
        if key in pre:
            pc[key] = _deep_merge(cfg[key], pre[key])
    return pc


def _read_status(path: Path, stage: str = None):
    """Read a status from training_summary.json (or None if missing/unreadable).
    With `stage` given, read stages[<stage>].status from the unified two-stage
    file; otherwise the flat top-level 'status'. Used to decide whether Stage 1 is
    already complete (so a later run skips straight to fine-tuning)."""
    try:
        if Path(path).exists():
            doc = json.load(open(path, encoding="utf-8"))
            if stage is not None:
                return (doc.get("stages") or {}).get(stage, {}).get("status")
            return doc.get("status")
    except Exception:
        pass
    return None


def _run_all_stages(cfg: dict, model, experiment_dir, resume=None,
                    persist_fn=None, log_fn=None):
    """Orchestrate the run. Single-stage (no `pretrain` block) -> one plain
    _run_experiment (unchanged behavior). Two-stage (`pretrain.enable: true`):

        Stage 1  pre-train on ChestX-ray14  -> results/checkpoints/<stage_name>/
        Stage 2  fine-tune on CheXpert      -> results/checkpoints/<finetune_stage_name>/

    Stage 2 is seeded from Stage 1's best.pt (full model: backbone + head, since the
    5 tasks map 1:1, Effusion == Pleural Effusion). Resume rules:
      - top-level `resume` resumes the FINE-TUNE stage (Stage 1 assumed complete).
      - `pretrain.resume` resumes the PRE-TRAIN stage.
      - a finished Stage 1 (its summary status is completed/early_stopped) is
        skipped automatically on a later run, going straight to fine-tuning."""
    pre = cfg.get("pretrain") or {}
    if not pre.get("enable", False):
        # No pretraining: the original single-stage run (flat checkpoints/, no
        # 'stage' column, default summary) — every existing experiment unchanged.
        # Optional `init_weights_from` (absent in every prior run) warm-starts the
        # weights from ANOTHER run's checkpoint — e.g. the AUC-margin run seeded
        # from the BCE-trained convnext_base_22k best.pt (LibAUC's recipe). Skipped
        # on resume (load_checkpoint restores the full state instead).
        return _run_experiment(
            cfg, model, experiment_dir, resume, persist_fn, log_fn,
            init_weights_from=(cfg.get("init_weights_from") if resume is None else None))

    experiment_dir = Path(experiment_dir)
    out = cfg["output"]
    results_dir = experiment_dir / out["run_dir"]
    ckpt_root = results_dir / out["checkpoints_dir"]
    pre_name = pre.get("stage_name", "pretrain_chestxray14")
    ft_name = pre.get("finetune_stage_name", "finetune_chexpert")
    best_name = out["best_name"]
    # both stages share ONE training_summary.json (each nested under stages[<tag>]).
    summary_path = results_dir / out.get("summary_json", "training_summary.json")
    pre_resume = pre.get("resume")
    pre_best = ckpt_root / pre_name / best_name

    print("#" * 70)
    print("# 🧬🧬  TWO-STAGE TRAINING ENABLED  🧬🧬")
    print("#   Stage 1/2 : PRE-TRAIN on ChestX-ray14   -> checkpoints/{}/".format(pre_name))
    print("#   Stage 2/2 : FINE-TUNE on CheXpert        -> checkpoints/{}/".format(ft_name))
    print("#   handoff   : Stage-1 best.pt -> Stage-2 model init (backbone + head)")
    print("#" * 70)

    # shared W&B step cursor: both stages log to ONE W&B run, but each stage's
    # global_step restarts at 0 — so Stage 2's step is offset past Stage 1's to keep
    # the run's step monotonic (W&B drops out-of-order steps otherwise).
    wandb_state = {"offset": 0}

    # ---- decide whether Stage 1 runs ----
    pre_done = _read_status(summary_path, pre_name) in ("completed", "early_stopped")
    if resume is not None and pre_resume is None:
        run_stage1 = False
        print(f"# ↪️  top-level resume='{resume}' set -> skipping Stage 1; "
              f"resuming the FINE-TUNE stage.")
    elif pre_done and pre_resume is None:
        run_stage1 = False
        print(f"# ✅ Stage 1 already complete (stages['{pre_name}'].status in "
              f"training_summary.json); skipping pre-training, going to fine-tuning.")
    else:
        run_stage1 = True

    # ---- Stage 1: ChestX-ray14 pre-training ----
    if run_stage1:
        pcfg = _build_pretrain_cfg(cfg, pre)
        _run_experiment(
            pcfg, model, experiment_dir, resume=pre_resume,
            persist_fn=persist_fn, log_fn=log_fn,
            ckpt_subdir=pre_name, stage_tag=pre_name, phase="1/2",
            stage_header="STAGE 1/2 — PRE-TRAIN on ChestX-ray14",
            wandb_state=wandb_state)

    # ---- Stage 2: CheXpert fine-tuning ----
    if resume is None:
        # fresh fine-tune -> seed from Stage 1 best.pt
        if not pre_best.exists():
            raise FileNotFoundError(
                f"Stage-1 best checkpoint not found: {pre_best}. Run/complete the "
                f"ChestX-ray14 pre-training stage first (or set pretrain.resume).")
        init_from = pre_best
    else:
        init_from = None        # resuming fine-tune: full state restored from its ckpt

    return _run_experiment(
        cfg, model, experiment_dir, resume=resume,
        persist_fn=persist_fn, log_fn=log_fn,
        ckpt_subdir=ft_name, stage_tag=ft_name, phase="2/2",
        stage_header="STAGE 2/2 — FINE-TUNE on CheXpert",
        init_weights_from=init_from, wandb_state=wandb_state)


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
        return _run_all_stages(cfg, model, experiment_dir, resume, persist_fn, log_fn)
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
    "torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn",
    "opencv-python-headless", "pillow", "pyyaml", "wandb",
]


def modal_image(python_version: str = "3.11", extra_pip=None, extra_pip_options: str = ""):
    """Build the Modal container image: pip deps + the whole training_scripts/
    source tree (code + YAML configs), excluding local run artifacts.

    python_version : the container's Python. For functions shipped with
    serialized=True (the eval/calibration scripts launched via `python ... +
    app.run()`), this MUST match the LOCAL interpreter's version, so those
    scripts pass their own version. Training (`modal run`) doesn't need a match
    and keeps the 3.11 default.

    extra_pip / extra_pip_options : optional, per-run extra packages installed in
    a SECOND pip layer (so the shared base layer/image cache is unchanged for
    every other run). The AUC-margin run uses this to add 'libauc' with
    extra_pip_options='--no-deps' — libauc's deps (torch/numpy/sklearn/...) are
    already in the base image, and --no-deps avoids pulling its heavy extras
    (transformers, torch_geometric, ...) and any chance of touching torch."""
    import modal
    img = (modal.Image.debian_slim(python_version=python_version)
           .pip_install(*_MODAL_PIP))
    if extra_pip:
        img = img.pip_install(*extra_pip, extra_options=extra_pip_options)
    return img.add_local_dir(
        str(PKG_DIR), remote_path="/root/training_scripts",
        ignore=["**/results/**", "**/train_config/**", "**/__pycache__/**"],
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


def pretrain_data_mount(cfg: dict):
    """For a two-stage run whose ChestX-ray14 images live on their OWN Modal volume
    (separate from chexpert-data — each volume has a 500k-inode cap), return
    (volume_name, mount_path) so train.py can mount it alongside the data + runs
    volumes. Returns None when there's no pretrain block, pretraining is disabled,
    or no separate volume is declared (then the pretrain paths are assumed to sit
    on an already-mounted volume). Single-stage runs always get None — unchanged."""
    pre = cfg.get("pretrain") or {}
    if pre.get("enable") and pre.get("data_volume"):
        return pre["data_volume"], pre.get("data_mount", "/data_cxr14")
    return None


def remote_cfg(cfg: dict) -> dict:
    """Copy of cfg with data paths repointed at the mounted Modal data volume.
    For a two-stage run, the pre-train stage's images/CSVs live on their own
    volume, so its paths are repointed too (from pretrain.remote_data_root /
    remote_data_dir). No-op for single-stage runs (pretrain block absent)."""
    rc = copy.deepcopy(cfg)
    m = cfg["modal"]
    rc["paths"]["data_root"] = m["remote_data_root"]
    rc["paths"]["data_dir"] = m["remote_data_dir"]
    pre = rc.get("pretrain") or {}
    if pre.get("enable"):
        pp = pre.setdefault("paths", {})
        pp["data_root"] = pre["remote_data_root"]
        pp["data_dir"] = pre["remote_data_dir"]
    # A warm-start checkpoint from another run lives on a Modal volume in the
    # cloud; repoint to its in-container path when one is given.
    if cfg.get("init_weights_from_remote"):
        rc["init_weights_from"] = cfg["init_weights_from_remote"]
    return rc


# =============================================================================
# Section 6 — Plots from the training logs (post-hoc, CPU-only)
# Reads the two CSVs a run produces (train_log.csv + val_log.csv) and writes a
# set of PNGs to results/<plots>/.  No torch / GPU needed.  Each plot is wrapped
# so one failure never aborts the rest.  Reused unchanged by every experiment.
# =============================================================================

def _smooth(series, window: int):
    """Trailing rolling mean (min_periods=1 so the head isn't dropped)."""
    if window and window > 1:
        return series.rolling(window, min_periods=1).mean()
    return series


def _epoch_boundary_steps(train_df) -> list:
    """Global steps at which a NEW epoch begins (after the first) — drawn as
    faint vertical guides so the curves are readable per epoch."""
    steps = []
    if "epoch" in train_df.columns:
        prev = None
        for st, ep in zip(train_df["step"], train_df["epoch"]):
            if prev is not None and ep != prev:
                steps.append(st)
            prev = ep
    return steps


def _vlines(ax, steps):
    """Faint grey vertical guides at the given x positions (epoch boundaries)."""
    for s in steps:
        ax.axvline(s, color="0.85", lw=0.8, zorder=0)


def _mark_extreme(ax, x, y, mode: str = "max"):
    """Star + annotate the best point of a metric curve (max or min)."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or np.all(np.isnan(y)):
        return
    idx = int(np.nanargmax(y)) if mode == "max" else int(np.nanargmin(y))
    xi = float(np.asarray(x)[idx])
    ax.scatter([xi], [y[idx]], marker="*", s=220, color="crimson", zorder=6)
    ax.annotate(f"best={y[idx]:.4f}\n@ step {int(xi)}", (xi, y[idx]),
                textcoords="offset points", xytext=(8, -12), fontsize=8,
                color="crimson")


def make_plots(cfg: dict, results_dir, out_subdir: str = "plots",
               smooth_window: int = 50, dpi: int = 130):
    """Render the training-curve PNGs for one finished/running experiment.

    Inputs : results_dir/<train_log_csv> and results_dir/<val_log_csv>.
    Outputs: results_dir/<out_subdir>/*.png  (one concept per file):
        loss_curves, mean_auroc, mean_auprc, per_task_auroc, per_task_auprc,
        macro_f1_precision_recall, training_diagnostics (lr/img-s/grad-norm).

    Two-stage runs (pretrain.enable, logs carry a 'stage' column): the rows are
    split by stage and each stage gets its OWN plots subfolder — exactly like the
    checkpoints layout — so e.g. results/plots/pretrain_chestxray14/ and
    results/plots/finetune_chexpert/. Single-stage runs are unchanged (flat
    results/plots/)."""
    import matplotlib
    matplotlib.use("Agg")                 # headless: write files, never open a window
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    out = cfg["output"]
    tasks = cfg["tasks"]
    name = cfg.get("experiment", {}).get("name", "experiment")

    train_path = results_dir / out["train_log_csv"]
    val_path = results_dir / out["val_log_csv"]
    print("=" * 70)
    print(f"[make_plots] {name}")
    print(f"  train log : {train_path}")
    print(f"  val   log : {val_path}")

    train_df = pd.read_csv(train_path) if train_path.exists() else None
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    if train_df is None:
        print("  ⚠️  no train_log.csv — skipping train-based plots")
    if val_df is None:
        print("  ⚠️  no val_log.csv — skipping validation-based plots")

    # ----- the actual plotting, for ONE (train_df, val_df) into ONE plots_dir -----
    def _render(train_df, val_df, plots_dir, name):
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"  output    : {plots_dir}")
        bsteps = _epoch_boundary_steps(train_df) if train_df is not None else []
        saved = []

        def _save(fig, fname):
            path = plots_dir / fname
            fig.tight_layout()
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            saved.append(fname)
            print(f"  ✅ {fname}")

        # 1) loss curves — train (raw + smoothed) and validation, vs global step
        def _plot_loss():
            if train_df is None and val_df is None:
                return
            fig, ax = plt.subplots(figsize=(9, 5))
            if train_df is not None:
                ax.plot(train_df["step"], train_df["train_loss"], color="tab:blue",
                        alpha=0.25, lw=0.8, label="train (raw)")
                ax.plot(train_df["step"], _smooth(train_df["train_loss"], smooth_window),
                        color="tab:blue", lw=1.8, label=f"train (smooth {smooth_window})")
            if val_df is not None:
                ax.plot(val_df["step"], val_df["val_loss"], color="tab:orange",
                        marker="o", ms=4, lw=1.6, label="validation")
            _vlines(ax, bsteps)
            ax.set_xlabel("global step"); ax.set_ylabel("BCE loss")
            ax.set_title(f"{name} — loss"); ax.legend(); ax.grid(True, alpha=0.3)
            _save(fig, "loss_curves.png")

        # 2/3) a single macro metric vs step (best point starred)
        def _plot_macro_metric(col, title, fname, mode="max"):
            if val_df is None or col not in val_df.columns:
                return
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(val_df["step"], val_df[col], color="tab:green",
                    marker="o", ms=4, lw=1.8)
            _mark_extreme(ax, val_df["step"].values, val_df[col].values, mode)
            _vlines(ax, bsteps)
            ax.set_xlabel("global step"); ax.set_ylabel(col)
            ax.set_title(f"{name} — {title}"); ax.grid(True, alpha=0.3)
            _save(fig, fname)

        # 4/5) per-task curves (one line per pathology) for a given metric prefix.
        # Discover EVERY "{prefix}/<task>" column present, rather than only those whose
        # name matches cfg["tasks"]. In a two-stage run the stages share one CSV but
        # name the 5th task differently (NIH 'Effusion' vs CheXpert 'Pleural Effusion');
        # the header is written once (by the first stage), so a fixed cfg-name lookup
        # would drop that column and show only 4 tasks. We keep cfg-task order for the
        # shared names and append any other present column (labelled by its own name).
        def _plot_per_task(prefix, title, fname):
            if val_df is None:
                return
            present = [c for c in val_df.columns if c.startswith(prefix + "/")]
            cols = [f"{prefix}/{t}" for t in tasks if f"{prefix}/{t}" in present]
            cols += [c for c in present if c not in cols]
            if not cols:
                return
            fig, ax = plt.subplots(figsize=(9, 5))
            for c in cols:
                ax.plot(val_df["step"], val_df[c], marker="o", ms=3, lw=1.5,
                        label=c.split("/", 1)[1])
            _vlines(ax, bsteps)
            ax.set_xlabel("global step"); ax.set_ylabel(prefix.upper())
            ax.set_title(f"{name} — {title}")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            _save(fig, fname)

        # 6) macro F1 / precision / recall together (all 0–1, one figure)
        def _plot_macro_fpr():
            if val_df is None:
                return
            series = [("mean_f1", "F1", "tab:purple"),
                      ("mean_precision", "precision", "tab:red"),
                      ("mean_recall", "recall", "tab:blue")]
            series = [s for s in series if s[0] in val_df.columns]
            if not series:
                return
            fig, ax = plt.subplots(figsize=(9, 5))
            for col, lbl, color in series:
                ax.plot(val_df["step"], val_df[col], marker="o", ms=3, lw=1.6,
                        color=color, label=lbl)
            _vlines(ax, bsteps)
            ax.set_xlabel("global step"); ax.set_ylabel("score @ 0.5")
            ax.set_title(f"{name} — macro F1 / precision / recall (threshold 0.5)")
            ax.legend(); ax.grid(True, alpha=0.3)
            _save(fig, "macro_f1_precision_recall.png")

        # 7) training diagnostics: LR / throughput / grad-norm (3 stacked panels)
        def _plot_diagnostics():
            if train_df is None:
                return
            fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
            axes[0].plot(train_df["step"], train_df["lr"], color="tab:blue", lw=1.5)
            axes[0].set_ylabel("learning rate"); axes[0].set_title(f"{name} — training diagnostics")
            if "imgs_per_s" in train_df.columns:
                axes[1].plot(train_df["step"], train_df["imgs_per_s"], color="tab:gray",
                             alpha=0.3, lw=0.8)
                axes[1].plot(train_df["step"], _smooth(train_df["imgs_per_s"], smooth_window),
                             color="tab:green", lw=1.6)
            axes[1].set_ylabel("images / sec")
            if "grad_norm" in train_df.columns:
                axes[2].plot(train_df["step"], train_df["grad_norm"], color="tab:gray",
                             alpha=0.3, lw=0.8)
                axes[2].plot(train_df["step"], _smooth(train_df["grad_norm"], smooth_window),
                             color="tab:red", lw=1.6)
            axes[2].set_ylabel("grad norm"); axes[2].set_xlabel("global step")
            for ax in axes:
                _vlines(ax, bsteps); ax.grid(True, alpha=0.3)
            _save(fig, "training_diagnostics.png")

        _safe("plot:loss", _plot_loss)
        _safe("plot:mean_auroc", _plot_macro_metric,
              "mean_auroc", "validation mean AUROC", "mean_auroc.png", "max")
        _safe("plot:mean_auprc", _plot_macro_metric,
              "mean_auprc", "validation mean AUPRC", "mean_auprc.png", "max")
        _safe("plot:per_task_auroc", _plot_per_task, "auroc", "per-task AUROC", "per_task_auroc.png")
        _safe("plot:per_task_auprc", _plot_per_task, "auprc", "per-task AUPRC", "per_task_auprc.png")
        _safe("plot:macro_fpr", _plot_macro_fpr)
        _safe("plot:diagnostics", _plot_diagnostics)

        print(f"[make_plots] wrote {len(saved)} plot(s) -> {plots_dir}")
        return saved

    def _stage_slice(df, stage):
        """Rows of df for one stage (None if absent/empty), so each stage plots
        only its own curves with its own (reset) step axis."""
        if df is None or "stage" not in df.columns:
            return None
        sub = df[df["stage"] == stage]
        return sub if len(sub) else None

    # ----- dispatch: per-stage subfolders for a two-stage run, else flat -----
    pre = cfg.get("pretrain") or {}
    has_stage_col = ((train_df is not None and "stage" in train_df.columns) or
                     (val_df is not None and "stage" in val_df.columns))
    saved = []
    if pre.get("enable") and has_stage_col:
        # stage order: pretrain then fine-tune (as named in cfg); append any other
        # stage values that happen to appear in the logs, preserving first-seen order.
        ordered = [pre.get("stage_name", "pretrain_chestxray14"),
                   pre.get("finetune_stage_name", "finetune_chexpert")]
        seen = []
        for df in (train_df, val_df):
            if df is not None and "stage" in df.columns:
                seen += [s for s in df["stage"].dropna().tolist()]
        seen = list(dict.fromkeys(seen))
        stages = [s for s in ordered if s in seen] + [s for s in seen if s not in ordered]
        print(f"  two-stage run -> one plots subfolder per stage: {stages}")
        for st in stages:
            print("-" * 70)
            print(f"  [stage: {st}]")
            saved += [f"{st}/{f}" for f in _render(
                _stage_slice(train_df, st), _stage_slice(val_df, st),
                results_dir / out_subdir / st, f"{name} [{st}]")]
    else:
        saved = _render(train_df, val_df, results_dir / out_subdir, name)

    print("=" * 70)
    return saved


# =============================================================================
# Section 7 — Official-set evaluation (final test)
# Loads best.pt and scores it on the official radiologist sets (valid200 and
# test500) INDEPENDENTLY, using the exact same data layer + metrics as training
# (cfg drives CLAHE / u_policy / image geometry, so each arm is scored fairly).
# Writes results/<set>_results.json (source of truth) + a readable .txt twin.
# =============================================================================

_REPORT_KEYS = ("auroc", "auprc", "f1", "precision", "recall", "specificity")


def _json_safe(obj):
    """Recursively make a dict JSON-valid: non-finite floats -> None (null)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


@torch.no_grad()
def _predict_dataframe(cfg, model, df, device, loss_fn, amp: bool,
                       channels_last: bool, batch_size: int, num_workers: int):
    """Run the model over one dataframe (split='val' -> no augmentation) and
    return (y_true (N,T) binary, y_prob (N,T), exclude_mask (N,T), mean_loss).
    Predictions collapse to per-task binary probabilities via the cfg head layout
    (sigmoid / renormalized p_pos); exclude_mask flags uncertain rows on
    multiclass tasks (all-False for a binary arm)."""
    layout = task_layout(cfg)
    ds = CheXpertDataset(df, cfg, split="val")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.append(labels.detach().cpu().numpy())
        ps.append(logits_to_probs(logits, layout).float().detach().cpu().numpy())
    y_true, excl = binarize_targets(np.concatenate(ys), layout,
                                    _val_ignore_uncertain(cfg))
    return y_true, np.concatenate(ps), excl, total_loss / max(1, n)


def _load_ckpt_weights(model, ckpt_dir: Path, checkpoint, device):
    """Load `checkpoint` (best | last | <step> | path) weights into `model`.
    best.pt holds a plain, un-compiled state dict (see _unwrap). Returns
    (ckpt_dict, ckpt_path)."""
    ckpt_path = _resolve_resume(checkpoint, ckpt_dir)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _unwrap(model).load_state_dict(ckpt["model"])
    return ckpt, ckpt_path


# ---- threshold calibration (max-F1 per task, on the validation set) ----------

def _best_f1_threshold(y_true_col, y_prob_col) -> tuple:
    """Threshold that maximizes F1 for one task, found exactly from the
    precision-recall curve. Degenerate tasks (no pos or no neg) -> 0.5.
    Returns (threshold, f1_at_threshold)."""
    yt = np.asarray(y_true_col)
    yp = np.asarray(y_prob_col)
    n_pos = int(yt.sum())
    if n_pos == 0 or n_pos == len(yt):
        return 0.5, float("nan")
    prec, rec, thr = precision_recall_curve(yt, yp)
    # prec/rec are len(thr)+1; the trailing point (recall=0) has no threshold.
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    f1 = f1[:-1]
    if len(f1) == 0 or np.all(np.isnan(f1)):
        return 0.5, float("nan")
    best = int(np.nanargmax(f1))
    return float(thr[best]), float(f1[best])


def calibrate_thresholds(cfg, model, df_val, device, amp: bool = False,
                         batch_size: int = None, num_workers: int = 0) -> tuple:
    """Run the model over the validation dataframe and pick the per-task max-F1
    threshold for each pathology. Returns (thresholds_dict, y_true, y_prob) so
    the caller can also report the val metrics achieved at those thresholds."""
    tasks = cfg["tasks"]
    if batch_size is None:
        batch_size = int(cfg["dataloader"].get("val_batch_size",
                         cfg["dataloader"]["batch_size"]))
    loss_fn = build_loss(cfg, device)
    print(f"[calibrate] inference over {len(df_val)} validation images "
          f"(bs={batch_size}, workers={num_workers}) ...")
    y_true, y_prob, excl, _ = _predict_dataframe(
        cfg, model, df_val, device, loss_fn, amp, channels_last=False,
        batch_size=batch_size, num_workers=num_workers)
    thresholds = {}
    for k, t in enumerate(tasks):
        keep = ~excl[:, k]                       # drop uncertain rows (multiclass tasks)
        thr, _f1 = _best_f1_threshold(y_true[keep, k], y_prob[keep, k])
        thresholds[t] = thr
    return thresholds, y_true, y_prob, excl


def _threshold_payload(cfg, thresholds, y_true, y_prob, exclude_mask, ckpt, ckpt_path,
                       n_val: int, objective: str = "f1") -> tuple:
    """Build the thresholds.json payload (+ the val metrics at those thresholds
    for printing). Returns (payload_dict, val_metrics_dict)."""
    tasks = cfg["tasks"]
    thr_vec = [thresholds[t] for t in tasks]
    metrics = compute_metrics(y_true, y_prob, tasks, threshold=thr_vec,
                              exclude_mask=exclude_mask)
    payload = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "objective": objective,
        "scope": "per_task",
        "calibration_set": cfg["paths"]["val_csv"],
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("global_step"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "n_val_images": int(n_val),
        "thresholds": {t: float(thresholds[t]) for t in tasks},
        "val_metrics_at_threshold": {"macro": metrics["macro"],
                                     "per_task": metrics["per_task"]},
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return payload, metrics


def _thresholds_path(results_dir: Path, cfg) -> Path:
    return Path(results_dir) / cfg["output"].get("thresholds_json", "thresholds.json")


def load_thresholds(results_dir: Path, cfg) -> tuple:
    """Read frozen per-task thresholds from results/thresholds.json.
    Returns ({task: threshold}, path) or (None, path) if absent/incomplete."""
    path = _thresholds_path(results_dir, cfg)
    if not path.exists():
        return None, path
    data = json.load(open(path, encoding="utf-8"))
    thr = data.get("thresholds", {})
    tasks = cfg["tasks"]
    if not all(t in thr for t in tasks):
        return None, path
    return {t: float(thr[t]) for t in tasks}, path


def run_calibration(cfg, model, experiment_dir, device=None, checkpoint: str = "best",
                    amp: bool = False, batch_size: int = None, num_workers: int = 0,
                    objective: str = "f1", ignore_uncertain: bool = False) -> dict:
    """Calibrate per-task thresholds on the validation set and SAVE them to
    results/thresholds.json (the frozen decision rule applied at test time).
    Prints the chosen thresholds + the val metrics they achieve. Returns the
    {task: threshold} dict.

    ignore_uncertain : when True, uncertain (-1) validation labels are EXCLUDED
    per task (the 'mask' policy) instead of following labels.u_policy — so the
    thresholds (and the reported val metrics) are computed on CERTAIN cells only.

    NOTE: thresholds are model-specific — re-run this for every experiment /
    checkpoint. The PROCEDURE (max-F1 on 01_val.csv) is what stays identical
    across arms, keeping the comparison fair."""
    experiment_dir = Path(experiment_dir)
    out = cfg["output"]
    tasks = cfg["tasks"]
    results_dir = experiment_dir / out["run_dir"]
    ckpt_dir = results_dir / out["checkpoints_dir"]
    _sub = finetune_ckpt_subdir(cfg)          # two-stage runs: the fine-tune subfolder
    if _sub:
        ckpt_dir = ckpt_dir / _sub
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("#" * 70)
    print(f"# 🎚️  threshold calibration: {cfg.get('experiment', {}).get('name')}")
    print(f"# 🖥️  device={device}  objective={objective}  scope=per_task  "
          f"set={cfg['paths']['val_csv']}")
    print("#" * 70)

    ckpt, ckpt_path = _load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    model = model.to(device)
    print(f"[calibrate] checkpoint: {ckpt_path}  (step={ckpt.get('global_step')}, "
          f"epoch={ckpt.get('epoch')})")

    # ignore_uncertain -> build val labels with the 'mask' policy so uncertain (-1)
    # cells become NaN and are dropped per task by binarize_targets/exclude_mask.
    cal_cfg = cfg
    if ignore_uncertain:
        cal_cfg = copy.deepcopy(cfg)
        cal_cfg["labels"]["u_policy"] = "mask"
        print("[calibrate] ignore_uncertain=True -> uncertain (-1) val labels EXCLUDED "
              "per task (calibrating on certain cells only)")

    val_csv = Path(cfg["paths"]["data_dir"]) / cfg["paths"]["val_csv"]
    df_val = pd.read_csv(val_csv)
    thresholds, y_true, y_prob, excl = calibrate_thresholds(
        cal_cfg, model, df_val, device, amp, batch_size, num_workers)
    payload, metrics = _threshold_payload(
        cal_cfg, thresholds, y_true, y_prob, excl, ckpt, ckpt_path, len(df_val), objective)

    # print a tidy per-task table (threshold + the val scores it yields)
    print("-" * 78)
    print(f"  {'task':<18}{'thr':>7}{'F1':>8}{'Prec':>8}{'Rec':>8}{'AUROC':>8}{'AUPRC':>8}")
    for t in tasks:
        m = metrics["per_task"][t]
        def f(x):
            return f"{x:.4f}" if isinstance(x, float) and math.isfinite(x) else "  nan"
        print(f"  {t:<18}{thresholds[t]:>7.3f}{f(m['f1']):>8}{f(m['precision']):>8}"
              f"{f(m['recall']):>8}{f(m['auroc']):>8}{f(m['auprc']):>8}")
    print("-" * 78)
    mac = metrics["macro"]
    print(f"  macro @ thresholds : F1={mac['mean_f1']:.4f}  P={mac['mean_precision']:.4f}  "
          f"R={mac['mean_recall']:.4f}   (AUROC={mac['mean_auroc']:.4f} threshold-free)")

    path = _thresholds_path(results_dir, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2)
    print(f"  💾 saved frozen thresholds -> {path}")
    print("#" * 70)
    print("# 🏁 calibration done.")
    print("#" * 70)
    return thresholds


# ---- official-set scoring (applies the frozen per-task thresholds) -----------

def _render_report_txt(report: dict) -> str:
    """Human-readable rendering of one set's report dict."""
    L = []
    name, s = report["experiment"], report["set"]
    L.append("=" * 86)
    L.append(f"OFFICIAL EVAL — {s}   ({report['csv']})")
    L.append("=" * 86)
    L.append(f"  experiment : {name}")
    L.append(f"  checkpoint : {report['checkpoint']}")
    L.append(f"  images     : {report['n_images']}    device: {report['device']}    "
             f"amp: {report['amp']}")
    L.append(f"  use_clahe  : {report['use_clahe']}    u_policy: {report['u_policy']}    "
             f"val_loss: {report['val_loss']:.5f}")
    L.append(f"  thresholds : per-task ({report['threshold_objective']}-optimal on "
             f"{report['threshold_source']})")
    macro = report["macro"]
    L.append("-" * 86)
    L.append("  MACRO (mean over tasks)   [AUROC / AUPRC are threshold-free]")
    L.append(f"    mean AUROC : {macro['mean_auroc']:.4f}        mean AUPRC : {macro['mean_auprc']:.4f}")
    L.append(f"    mean F1    : {macro['mean_f1']:.4f}    "
             f"mean precision: {macro['mean_precision']:.4f}    "
             f"mean recall: {macro['mean_recall']:.4f}    "
             f"mean spec: {macro['mean_specificity']:.4f}")
    L.append("-" * 86)
    L.append("  PER-TASK")
    L.append(f"    {'task':<18}{'thr':>7}{'AUROC':>8}{'AUPRC':>8}{'F1':>8}"
             f"{'Prec':>8}{'Rec':>8}{'Spec':>8}{'n_pos':>8}")
    for t, m in report["per_task"].items():
        def f(x):
            return f"{x:.4f}" if isinstance(x, float) and math.isfinite(x) else "  nan"
        L.append(f"    {t:<18}{m['threshold']:>7.3f}{f(m['auroc']):>8}{f(m['auprc']):>8}"
                 f"{f(m['f1']):>8}{f(m['precision']):>8}{f(m['recall']):>8}"
                 f"{f(m['specificity']):>8}{m['n_pos']:>8}")
    L.append("=" * 86)
    return "\n".join(L)


def evaluate_official(cfg: dict, model, experiment_dir, device=None,
                      checkpoint: str = "best", amp: bool = False,
                      batch_size: int = None, num_workers: int = 0,
                      objective: str = "f1"):
    """Load `checkpoint` (default best.pt) and score it on the official sets
    (valid200 + test500) INDEPENDENTLY, applying the FROZEN per-task thresholds
    from results/thresholds.json (calibrated on the validation set). If that file
    is missing it is calibrated now (on 01_val.csv) and saved, so eval is always
    self-contained. AUROC/AUPRC are threshold-free. For each set writes:
        results/<set>_results.json   (source of truth)
        results/<set>_results.txt    (readable twin)
    Returns {set_name: report_dict}. num_workers defaults to 0 (the official
    sets are tiny and 0 avoids Windows multiprocessing re-import issues)."""
    experiment_dir = Path(experiment_dir)
    out = cfg["output"]
    tasks = cfg["tasks"]
    results_dir = experiment_dir / out["run_dir"]
    ckpt_dir = results_dir / out["checkpoints_dir"]
    _sub = finetune_ckpt_subdir(cfg)          # two-stage runs: the fine-tune subfolder
    if _sub:
        ckpt_dir = ckpt_dir / _sub
    data_dir = Path(cfg["paths"]["data_dir"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = int(cfg["dataloader"].get("val_batch_size",
                         cfg["dataloader"]["batch_size"]))

    print("#" * 70)
    print(f"# 🧪 official evaluation: {cfg.get('experiment', {}).get('name')}")
    print(f"# 🖥️  device={device}  amp={amp}  "
          f"use_clahe={cfg['clahe']['use_clahe']}  u_policy={cfg['labels']['u_policy']}")
    print("#" * 70)

    ckpt, ckpt_path = _load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    model = model.to(device)
    print(f"[eval] loaded checkpoint: {ckpt_path}")
    print(f"[eval] checkpoint was step={ckpt.get('global_step')}  "
          f"epoch={ckpt.get('epoch')}  best_score={ckpt.get('best_score')}")

    # frozen per-task thresholds: load the saved file, else calibrate now + save.
    thr_map, thr_path = load_thresholds(results_dir, cfg)
    if thr_map is None:
        print(f"  ⚠️  no usable {thr_path.name} — calibrating per-task thresholds now "
              f"on {cfg['paths']['val_csv']} ...")
        df_val = pd.read_csv(data_dir / cfg["paths"]["val_csv"])
        thr_map, y_t, y_p, excl_v = calibrate_thresholds(
            cfg, model, df_val, device, amp, batch_size, num_workers)
        payload, _ = _threshold_payload(
            cfg, thr_map, y_t, y_p, excl_v, ckpt, ckpt_path, len(df_val), objective)
        thr_path.parent.mkdir(parents=True, exist_ok=True)
        with open(thr_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(payload), fh, indent=2)
        print(f"  💾 saved frozen thresholds -> {thr_path}")
    else:
        print(f"[eval] using frozen thresholds from {thr_path}")
    thr_vec = [thr_map[t] for t in tasks]
    print(f"[eval] per-task thresholds: "
          + "  ".join(f"{t}={thr_map[t]:.3f}" for t in tasks))

    loss_fn = build_loss(cfg, device)
    sets = [("valid200", cfg["paths"]["valid200_csv"]),
            ("test500", cfg["paths"]["test500_csv"])]

    reports = {}
    for set_name, csv_name in sets:
        csv_path = data_dir / csv_name
        print("=" * 70)
        print(f"🧪 evaluating on {set_name}  ({csv_path})")
        if not csv_path.exists():
            print(f"  ⚠️  CSV not found — skipping {set_name}")
            continue
        df = pd.read_csv(csv_path)
        print(f"  rows: {len(df)}  ->  running inference ...")
        y_true, y_prob, excl, val_loss = _predict_dataframe(
            cfg, model, df, device, loss_fn, amp, channels_last=False,
            batch_size=batch_size, num_workers=num_workers)
        metrics = compute_metrics(y_true, y_prob, tasks, threshold=thr_vec,
                                  exclude_mask=excl)

        report = {
            "experiment": cfg.get("experiment", {}).get("name"),
            "set": set_name,
            "csv": csv_name,
            "checkpoint": str(ckpt_path),
            "checkpoint_step": ckpt.get("global_step"),
            "checkpoint_epoch": ckpt.get("epoch"),
            "n_images": int(len(df)),
            "device": str(device),
            "amp": bool(amp),
            "threshold_objective": objective,
            "threshold_source": cfg["paths"]["val_csv"],
            "thresholds": {t: float(thr_map[t]) for t in tasks},
            "use_clahe": bool(cfg["clahe"]["use_clahe"]),
            "u_policy": cfg["labels"]["u_policy"],
            "val_loss": float(val_loss),
            "macro": metrics["macro"],
            "per_task": metrics["per_task"],
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        json_path = results_dir / f"{set_name}_results.json"
        txt_path = results_dir / f"{set_name}_results.txt"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(report), f, indent=2)
        txt = _render_report_txt(report)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(txt)
        print(f"  💾 {json_path}")
        print(f"  💾 {txt_path}")
        reports[set_name] = report

    print("#" * 70)
    print(f"# 🏁 official evaluation done — {len(reports)} set(s) scored.")
    print("#" * 70)
    return reports


if __name__ == "__main__":
    # Resolve the baseline experiment's config (config-loading demo).
    demo = PKG_DIR / "resnet50_without_clahe"
    cfg = load_config(demo)
