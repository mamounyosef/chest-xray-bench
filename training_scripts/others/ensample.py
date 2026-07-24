"""
ensample.py
===========
Probability-averaging ENSEMBLE, scored on the valid200 split ONLY (never test500).

Each member loads its own best.pt, runs inference with its OWN cfg (its own
CLAHE / image geometry / u-policy), and produces per-class probabilities. The
members' 5-class probability matrices are averaged (equal weight), then mean
AUROC (+ AUPRC, per-class) is computed on valid200.

The five per-disease Stage-2 AUC-M runs are given as ONE entry (STAGE2_GROUP) and
handled as ONE composite 5-class member: for each class, only its dedicated Stage-2
model contributes that class's probability. (This is mathematically identical to
treating them as five separate members, since their single-class outputs never
overlap — so it's specified as one line for convenience.)

Writes (overwritten each run):
Each run writes into its OWN timestamped subfolder (so different ensembles don't
overwrite each other):
    local :  training_scripts/others/ensembling_results/<timestamp>/ensemble_<SET>_summary.{json,txt}
    modal :  /runs/ensembling_results/<timestamp>/ensemble_<SET>_summary.{json,txt}
On a Modal run the freshly written <timestamp> folder is AUTOMATICALLY pulled back
down to training_scripts/others/ensembling_results/<timestamp>/ (via `modal volume
get`) as soon as the remote run finishes, so local + remote stay in sync with no
manual fetch.

Run:  python training_scripts/others/ensample.py   (honours RUN_ON below)
"""

import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
RUN_ON = "local"        # "modal" | "local" 
SET    = "valid200"      # scored split — "valid200" or "test500" or "val"
GPU    = "H200"         # Modal GPU for the ensemble run: T4|L4|A10G|A100|A100-80GB|
                        # H100|H200. Overrides the reference run's gpu; None -> use its.
BATCH_SIZE = 512        # 176
                        # own dataloader.val_batch_size). valid200 is 200 imgs, so
                        # any value >=200 is one batch; lower it only to cap VRAM.

# Modal container compute for the ensemble run (None -> inherit the reference run's
# modal.cpu_cores / modal.memory_gb). These size what `modal` allocates.
CPU_CORES = 14        # requested CPU cores for the Modal container
MEMORY_GB = 50        # requested RAM in GB for the Modal container

# DataLoader knobs applied to EVERY member's inference pass (None -> that run's own
# dataloader.val_* value). Raise workers toward the available cores to speed decoding.
# Separate worker counts per environment (Modal has many vCPUs; a local run is bound
# by this PC's cores). The active one is picked from RUN_ON below.
NUM_WORKERS_MODAL = 24   # DataLoader workers per member when RUN_ON == "modal"
NUM_WORKERS_LOCAL = 2    # DataLoader workers per member when RUN_ON == "local"
NUM_WORKERS = NUM_WORKERS_MODAL if RUN_ON == "modal" else NUM_WORKERS_LOCAL
PREFETCH_FACTOR = 4   # batches prefetched per worker (only used when workers > 0)

# Per-member prob cache (cache_runs/<set>/<run>_<ckpt>.npy). A present file is a
# HIT and is reused as-is (no GPU, no checkpoint needed). Set True for ONE run to
# ignore + overwrite existing cache entries — do this after RETRAINING a checkpoint.
REFRESH_CACHE = False

# Full 5-class models — each contributes ALL five classes. Just list run names.
FULL_MODELS = [
    "convnext_base_22k_1600x1312",
    "medmae_vitb_nih_B_768_s2",
    "rad_dino_vitB_768",
    # "rad_dino_vitB_1064x896",
    # "medmae_vitb_nih",
    # "convnext_base_22k_768x640",
    # "convnext_base_22k_final_stage1",

    # "medmae_vitb_raw",
    # "convnext_base_22k_seed1337",
    # "convnext_base_22k_seed7",
]

# Per-run checkpoint file under results/checkpoints/ (default "best.pt"). Two-stage
# runs keep their fine-tune best.pt in a stage subfolder.
CKPT_SUBPATH = {
    # "convnext_large_22k_cxr14_pretrain": "finetune_chexpert/best.pt",
    # "convnext_base_22k_cxr14_pretrain_lowlr_all": "finetune_chexpert/best.pt",
}

# Extra members from SPECIFIC checkpoints — each entry is its own separate 5-class
# member, ADDED on top of FULL_MODELS. Lets you ensemble several checkpoints of the
# SAME run as independent voters (e.g. best.pt via FULL_MODELS + step 7500 here).
#   Each entry: {"run": <run name>, "checkpoint": <"best" | "last" | <int step> | filename>}
#   resolved in that run's checkpoints dir like `resume` (7500 -> ckpt_step7500.pt;
#   two-stage subfolder from CKPT_SUBPATH honored). Labeled "<run> @ step<N>" so it
#   never collides with the plain "<run>" best.pt member. Empty list = off.
CHECKPOINT_MEMBERS = [
    # {"run": "convnext_base_22k_1600x1312", "checkpoint": 7500},
    # {"run": "convnext_base_22k_1600x1312", "checkpoint": 8700},
    # {"run": "medmae_vitb_nih", "checkpoint": 7500},
    # {"run": "medmae_vitb_nih_B_768_s2", "checkpoint": 4400},
]

# Per-class thresholds for the F1/precision/recall/specificity of the ENSEMBLE come
# from this run's results/thresholds.json (AUROC/AUPRC stay threshold-free). Defaults
# to the FIRST member in FULL_MODELS (its thresholds.json is on the runs volume because
# it's an ensemble member). Set to any run name to override. Missing file/task -> 0.5.
THRESHOLDS_FROM = FULL_MODELS[0]

# WEIGHTED blend (per class, per member) instead of the flat 1/M probability average.
# Point WEIGHTS_FROM at a weighted_ensemble summary produced by
# others/weighted_ensemble/weighted_ensemble.py — either the json file itself or the
# folder containing it (weighted_<set>_summary.json); its "weights_per_class" block
# ({task: {member: weight}}) drives the blend. None -> equal-weight average (default).
# Resolved on the launcher and passed in, so it also works on Modal. Every ensemble
# member's label must have a weight for every class; weights are renormalized per class
# over the members actually present (so dropping/adding a member stays well-defined).
#   e.g. WEIGHTS_FROM = "weighted_ensemble/results/2026-07-22_14-30-05"
WEIGHTS_FROM = "weighted_ensemble/results/2026-07-22_15-22-14"

# Each ensemble run writes into its OWN timestamped subfolder under
# ensembling_results/, so trying different ensembles never overwrites the last.
# RUN_TAG (optional) is appended to the timestamp to make a run self-describing,
# e.g. RUN_TAG="5models" -> 2026-07-01_14-30-05_5models.
RUN_TAG = ""

# The five per-disease Stage-2 models, as ONE composite member (one entry).
# Set USE_STAGE2 = False to drop it from the ensemble.
USE_STAGE2 = False
STAGE2_GROUP = {
    "Atelectasis":      "convnext_base_22k_final_stage2_atelectasis",
    "Cardiomegaly":     "convnext_base_22k_final_stage2_cardiomegaly",
    "Consolidation":    "convnext_base_22k_final_stage2_consolidation",
    "Edema":            "convnext_base_22k_final_stage2_edema",
    "Pleural Effusion": "convnext_base_22k_final_stage2_pleural_effusion",
}
# ===========================================================================

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _resolve_pkg_root() -> Path:
    # this script lives in training_scripts/others/, so shared_code.py is one level up
    here = Path(__file__).resolve().parent
    for cand in (Path("/root/training_scripts"), here.parent, here):
        if (cand / "shared_code.py").exists():
            return cand
    return here.parent


PKG_ROOT = _resolve_pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc          # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402


def build_model_generic(cfg: dict):
    """Rebuild a run's model EXACTLY as its own train.py does, so best.pt loads.
    timm ids carry a pretrained tag (a '.'); the plain torchvision names
    (densenet121, convnext_tiny, resnet50, ...) are built from torchvision with
    the same head swap the runs use. Weights=None (best.pt supplies them)."""
    import timm
    import torch.nn as nn
    import torchvision
    name = cfg["model"]["name"]
    n = sc.num_output_logits(cfg)
    # Medical-MAE ViT runs (model.arch == "medmae_vitb") are NOT plain timm ViTs:
    # they use global_pool='avg' (fc_norm head) at this run's img_size (non-square
    # pos-embed). Build them EXACTLY as train.py does so best.pt loads; pretrained=
    # False -> best.pt supplies the weights (no Google-Drive checkpoint reload).
    if cfg["model"].get("arch") == "medmae_vitb":
        return sc.build_medmae_vit(cfg, load_pretrained=False)
    # RAD-DINO runs (model.arch == "raddino") are a HF Dinov2 backbone wrapped in
    # sc.RadDinoClassifier (NOT a timm model) — build the identical architecture from
    # the HF config (load_pretrained=False -> no weight download; best.pt supplies them).
    if cfg["model"].get("arch") == "raddino":
        return sc.build_raddino_vit(cfg, load_pretrained=False)
    if "." in name:                                  # timm id (e.g. convnext_base.fb_in22k...)
        return timm.create_model(name, pretrained=False, num_classes=n)
    low = name.lower()
    if low.startswith("densenet"):
        m = getattr(torchvision.models, name)(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, n)
        return m
    if low.startswith("convnext"):
        m = getattr(torchvision.models, name)(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
        return m
    if low.startswith("resnet"):
        m = getattr(torchvision.models, name)(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n)
        return m
    return timm.create_model(name, pretrained=False, num_classes=n)   # fallback


def _out_subdir(base: Path) -> Path:
    """A fresh timestamped subfolder under base/ensembling_results/ (+ RUN_TAG)."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{RUN_TAG}" if RUN_TAG else stamp
    return base / "ensembling_results" / name


def _dummy_loss(logits, targets):
    return logits.sum() * 0.0        # we only need probabilities, not the loss


def _fetch_results_from_modal(runs_volume: str, remote_sub: str, local_base: Path):
    """Pull the run's freshly written <timestamp> folder off the runs volume down to
    local `others/ensembling_results/`, so a Modal ensemble ends up on disk exactly
    like a local one. `remote_sub` is the volume-relative POSIX path the remote
    function returned (e.g. 'ensembling_results/2026-07-06_14-30-05').

    IMPORTANT: `modal volume get` maps each downloaded file to
    (local_destination / entry_path.relative_to(remote_path.parent)) ONLY when
    local_destination already exists AS A DIRECTORY; otherwise it writes EVERY file
    to local_destination itself (last write wins -> a single file, not a folder). So
    we pass the PARENT dir (which we pre-create) as the destination and let modal
    recreate the <ts>/ subfolder + its files under it. Invoked via `python -m modal`
    so it works regardless of PATH."""
    import subprocess
    from pathlib import PurePosixPath
    remote_posix = remote_sub.replace("\\", "/")               # e.g. ensembling_results/<ts>
    dest_parent = local_base / PurePosixPath(remote_posix).parent   # others/ensembling_results
    dest_parent.mkdir(parents=True, exist_ok=True)             # MUST exist as a dir (see above)
    final_dir = dest_parent / PurePosixPath(remote_posix).name      # others/ensembling_results/<ts>
    cmd = [sys.executable, "-m", "modal", "volume", "get",
           runs_volume, remote_posix, str(dest_parent)]
    print(f"[fetch] modal volume get {runs_volume} {remote_posix} -> {dest_parent}")
    subprocess.run(cmd, check=True)
    print(f"[fetch] downloaded ensemble results -> {final_dir}")
    return final_dir


def _sync_cache_up(runs_volume: str, local_base: Path):
    """Push the LOCAL others/cache_runs/ up to the runs volume BEFORE a Modal run,
    so remote members reuse probs already computed on this PC. Uploading the folder to
    the volume ROOT lands it at /cache_runs (dir basename), matching where the
    remote reads/writes (/runs/cache_runs). --force overwrites unchanged dupes;
    check=False so a first run (nothing to push) never aborts the ensemble."""
    import subprocess
    L = Path(local_base) / "cache_runs"
    if not L.exists() or not any(L.rglob("*.npy")):
        print("[cache-sync] up: no local cache_runs yet (skip)")
        return
    cmd = [sys.executable, "-m", "modal", "volume", "put", "--force",
           runs_volume, str(L), "/"]
    print(f"[cache-sync] up: {L} -> {runs_volume}:/cache_runs")
    subprocess.run(cmd, check=False)


def _sync_cache_down(runs_volume: str, local_base: Path):
    """Pull the runs volume's /cache_runs/ back down to others/cache_runs/ AFTER
    a Modal run, so members freshly computed on the GPU are cached on this PC too.
    local_base already exists as a dir, so `get` maps entries under it (-> its
    cache_runs/ subtree). check=False: an empty/absent remote cache isn't fatal."""
    import subprocess
    dest = Path(local_base)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "modal", "volume", "get", "--force",
           runs_volume, "cache_runs", str(dest)]
    print(f"[cache-sync] down: {runs_volume}:/cache_runs -> {dest}")
    subprocess.run(cmd, check=False)


def _member_ckpt_stem(run: str, checkpoint=None):
    """The checkpoint-file stem used in a member's cache name (f'{run}_{stem}'),
    WITHOUT touching any volume. Returns None when it can't be known offline (e.g.
    'last', which needs a directory listing) -> caller must then assume 'not cached'.
    Mirrors ckpt_path_of + _predict_member's naming."""
    sub = CKPT_SUBPATH.get(run, "best.pt")
    if checkpoint is None or checkpoint == "best":
        return Path(sub).stem
    if isinstance(checkpoint, int) or (isinstance(checkpoint, str) and checkpoint.isdigit()):
        return f"ckpt_step{int(checkpoint)}"
    if isinstance(checkpoint, str) and checkpoint.endswith(".pt"):
        return Path(checkpoint).stem
    return None                                   # e.g. "last" -> undecidable offline


def _expected_cache_names():
    """(names, undecidable): the cache basenames every member of the CURRENT config
    will look up, and whether any member's name can't be resolved offline."""
    names, undecidable = [], False

    def add(run, checkpoint=None):
        nonlocal undecidable
        stem = _member_ckpt_stem(run, checkpoint)
        if stem is None:
            undecidable = True
        else:
            names.append(f"{run}_{stem}")

    for run in FULL_MODELS:
        add(run)
    for spec in CHECKPOINT_MEMBERS:
        add(spec["run"], spec["checkpoint"])
    if USE_STAGE2:
        for run in STAGE2_GROUP.values():
            add(run)
    return names, undecidable


def _members_all_cached(local_base: Path, set_name: str) -> bool:
    """True iff EVERY member's probs (and the labels) are already cached locally for
    `set_name` — so the Modal VM can run CPU-only (no GPU). Name-based existence check
    (a present .npy is reused as-is). Any undecidable member ("last") -> False, and
    REFRESH_CACHE -> False (a refresh recomputes, so it needs the GPU)."""
    if REFRESH_CACHE:
        return False
    names, undecidable = _expected_cache_names()
    if undecidable or not names:
        return False
    cdir = Path(local_base) / "cache_runs" / set_name
    needed = [f"{n}.npy" for n in names] + ["_y_true.npy"]
    missing = [f for f in needed if not (cdir / f).exists()]
    if missing:
        print(f"[gpu] not all cached for {set_name} — missing: {missing}")
    return not missing


# --------------------- local "best" tracker (per SET) ----------------------
# After every run the freshly written results folder is on local disk. We keep, PER
# scored SET, ONE winning folder tagged with a set-qualified suffix " (best <set>)"
# (e.g. "2026-07-15_17-09-13 (best test500)"). If the new run's ensemble mean AUROC
# beats the current winner for THAT set, the old winner is demoted (suffix stripped)
# and the new folder is promoted. valid200 and test500 keep INDEPENDENT winners,
# distinguished by the set named in the suffix.
import re as _re                                          # noqa: E402
# matches a trailing "(best)" or "(best <set>)" marker on a folder name. <set> is any
# split token (valid200, test500, val, ...) so the tracker works for every SET.
_BEST_RE = _re.compile(r"\s*\(best(?:\s+(?P<set>\w+))?\)\s*$")


def _strip_best(name: str) -> str:
    """Folder name without any trailing '(best)'/'(best <set>)' marker."""
    return _BEST_RE.sub("", name).rstrip()


def _folder_mean_auroc(folder: Path, set_name: str):
    """ensemble_mean_auroc for `set_name`, read from folder's summary json (or None)."""
    import json
    j = Path(folder) / f"ensemble_{set_name}_summary.json"
    if not j.exists():
        return None
    try:
        return float(json.loads(j.read_text(encoding="utf-8"))["ensemble_mean_auroc"])
    except Exception:
        return None


def _rename_dir(d: Path, new_name: str) -> Path:
    """Rename folder d -> its parent/new_name (no-op if unchanged; refuse to clobber)."""
    tgt = d.parent / new_name
    if tgt.resolve() == d.resolve():
        return d
    if tgt.exists():
        print(f"[best] WARNING: '{tgt.name}' already exists — leaving '{d.name}' as-is")
        return d
    d.rename(tgt)
    return tgt


def _migrate_legacy_best(root: Path):
    """Normalize any legacy bare '<ts> (best)' folder to the set-qualified
    '<ts> (best <set>)', inferring <set> from whichever summary json it contains."""
    for d in list(root.iterdir()):
        if not d.is_dir():
            continue
        m = _BEST_RE.search(d.name)
        if not m or m.group("set") is not None:          # only bare "(best)" (no set)
            continue
        base = _strip_best(d.name)
        for summ in sorted(d.glob("ensemble_*_summary.json")):   # infer set from the summary file
            s = summ.name[len("ensemble_"):-len("_summary.json")]
            print(f"[best] migrating legacy marker: '{d.name}' -> '{base} (best {s})'")
            _rename_dir(d, f"{base} (best {s})")
            break


def _update_best_tracker(set_name: str, new_dir: Path):
    """Promote new_dir to the '(best <set_name>)' winner IFF its ensemble mean AUROC
    beats the current winner for THIS set. Independent tracker per set; also heals
    any legacy bare '(best)' folder into the set-qualified form first."""
    new_dir = Path(new_dir)
    root = new_dir.parent                                 # others/ensembling_results
    if not root.exists():
        return
    _migrate_legacy_best(root)

    new_auroc = _folder_mean_auroc(new_dir, set_name)
    if new_auroc is None:
        print(f"[best] {set_name}: no ensemble_{set_name}_summary.json in "
              f"'{new_dir.name}' — best tracker skipped")
        return

    # current winner for THIS set = a folder whose name ends in "(best <set_name>)"
    cur_best, cur_auroc = None, None
    for d in root.iterdir():
        if not d.is_dir() or d.resolve() == new_dir.resolve():
            continue
        m = _BEST_RE.search(d.name)
        if m and m.group("set") == set_name:
            a = _folder_mean_auroc(d, set_name)
            if a is not None and (cur_auroc is None or a > cur_auroc):
                cur_best, cur_auroc = d, a

    if cur_best is None:
        p = _rename_dir(new_dir, f"{_strip_best(new_dir.name)} (best {set_name})")
        print(f"[best] {set_name}: no previous best -> promoted '{p.name}' "
              f"(AUROC={new_auroc:.4f})")
    elif new_auroc > cur_auroc:
        _rename_dir(cur_best, _strip_best(cur_best.name))          # demote old winner
        p = _rename_dir(new_dir, f"{_strip_best(new_dir.name)} (best {set_name})")
        print(f"[best] {set_name}: NEW BEST {new_auroc:.4f} > {cur_auroc:.4f} "
              f"-> '{p.name}'  (demoted '{cur_best.name}')")
    else:
        print(f"[best] {set_name}: kept '{cur_best.name}' "
              f"(best AUROC={cur_auroc:.4f} >= new {new_auroc:.4f})")


# ------------------------ per-member prediction cache ----------------------
# A member's (N, 5) PROBABILITY matrix depends only on (run cfg, checkpoint, SET) —
# the run's cfg (geometry/CLAHE/u-policy) is fixed, so the cache key is just
# cache_runs/<set>/<run>_<ckpt>.npy. This lets a re-run reuse a member's probs
# instead of recomputing on the GPU. The cache is NAME-based: a present .npy is a hit,
# regardless of environment (a probe built on Modal is reused as-is when run locally,
# and vice-versa). On a HIT the checkpoint is never resolved/loaded/stat'd at all — so
# a fully-cached run needs neither the GPU nor even the .pt files present. If you
# RETRAIN a checkpoint, set REFRESH_CACHE=True (below) for one run to overwrite.
def _cache_npy(cache_dir: Path, set_name: str, cache_name: str) -> Path:
    return Path(cache_dir) / set_name / f"{cache_name}.npy"


def _predict_member(cfg, ckpt_resolver, df, device, cache_dir, set_name, cache_name):
    """_predict with a NAME-based on-disk cache. `ckpt_resolver` is a 0-arg callable
    returning the checkpoint path — it is invoked ONLY on a cache miss, so a hit never
    touches (or requires) the checkpoint file. Set REFRESH_CACHE to force recompute."""
    npy = _cache_npy(cache_dir, set_name, cache_name)
    if not REFRESH_CACHE and npy.exists():
        arr = np.load(npy)
        print(f"[cache] HIT  {set_name}/{cache_name}  {arr.shape}  (no compute)")
        return arr
    print(f"[cache] MISS {set_name}/{cache_name} -> computing on {device.type.upper()}", flush=True)
    ckpt_path = ckpt_resolver()                       # resolve/require the .pt only now
    p = _predict(cfg, ckpt_path, df, device, desc=cache_name)
    try:
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, p)
        print(f"[cache] {'REFRESH' if REFRESH_CACHE else 'save'} {set_name}/{cache_name}")
    except Exception as e:
        print(f"[cache] WARNING: could not save {set_name}/{cache_name}: {e}")
    return p


def _predict(cfg: dict, ckpt_path: Path, df, device, desc: str = None) -> np.ndarray:
    """Probabilities (N, len(cfg tasks)) for one checkpoint (ckpt_path) over `df`.
    `desc` labels the throttled per-batch progress line printed during inference."""
    print(f"     [load] building model + weights from {Path(ckpt_path).name} ...", flush=True)
    model = build_model_generic(cfg).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sc._unwrap(model).load_state_dict(ck["model"])
    bs = int(BATCH_SIZE or cfg["dataloader"].get("val_batch_size",
                                                  cfg["dataloader"]["batch_size"]))
    nw = int(NUM_WORKERS) if NUM_WORKERS is not None \
        else int(cfg["dataloader"].get("val_num_workers", 4))
    eff_cfg = cfg
    if PREFETCH_FACTOR is not None:                   # _predict_dataframe reads it off cfg
        eff_cfg = dict(cfg)
        eff_cfg["dataloader"] = dict(cfg["dataloader"])
        eff_cfg["dataloader"]["val_prefetch_factor"] = int(PREFETCH_FACTOR)
    _, y_prob, _, _ = sc._predict_dataframe(
        eff_cfg, model, df, device, _dummy_loss, amp=False, channels_last=False,
        batch_size=bs, num_workers=nw, progress_desc=(desc or "member"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y_prob


def _ckpt_member_label(run: str, checkpoint) -> str:
    """Distinct member label for a specific checkpoint (never collides with the plain
    '<run>' best.pt member): '<run> @ step7500' for an int, '<run> @ <spec>' else."""
    if isinstance(checkpoint, int) or (isinstance(checkpoint, str) and checkpoint.isdigit()):
        return f"{run} @ step{int(checkpoint)}"
    return f"{run} @ {checkpoint}"


def run_ensemble(load_cfg, ckpt_path_of, out_dir: Path, thr_map: dict = None,
                 thr_source: str = "", class_weights: dict = None,
                 weights_source: str = ""):
    """Core routine. `load_cfg(run)` -> that run's cfg (data paths already correct
    for this environment); `ckpt_path_of(run, checkpoint=None)` -> a checkpoint path
    (None -> that run's best.pt/CKPT_SUBPATH; else a specific best|last|<step>|file);
    `thr_map` -> {task: threshold} per-class thresholds for F1/precision/recall/
    specificity (AUROC/AUPRC are threshold-free); `thr_source` -> a human label of
    where they came from (for the summary/logs). The thresholds are resolved by the
    LAUNCHER from the local repo and passed in, so they always load regardless of what
    happens to be on the runs volume."""
    import json, pandas as pd
    from datetime import datetime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reference cfg (first full model) fixes the canonical task order + the split df.
    ref = load_cfg(FULL_MODELS[0])
    tasks = list(ref["tasks"])
    data_dir = Path(ref["paths"]["data_dir"])
    df = pd.read_csv(data_dir / ref["paths"][f"{SET}_csv"])
    print(f"[ensemble] {SET}: {len(df)} images  |  tasks={tasks}  device={device}")
    # per-member prob cache, next to ensembling_results/ (out_dir = <base>/ensembling_results/<ts>)
    cache_dir = Path(out_dir).parent.parent / "cache_runs"
    print(f"[cache] dir: {cache_dir / SET}")

    # per-class thresholds for F1/P/R/Spec (AUROC/AUPRC ignore them). Resolved by the
    # launcher from the local repo and passed in as a dict -> never depends on the vol.
    thr_map = thr_map or {}
    if thr_map:
        print(f"[ensemble] thresholds from {thr_source or '(provided)'}")
    else:
        print(f"[ensemble] WARNING: no thresholds ({thr_source or 'none'}) -> F1/P/R/Spec at 0.5")
    thr_vec = [float(thr_map.get(t, 0.5)) for t in tasks]

    # ground truth labels are FIXED per SET -> cache them too, so an ALL-cached run
    # needs NO model build at all (truly CPU-only). _y_true.npy sits beside members.
    _yt_npy = Path(cache_dir) / SET / "_y_true.npy"
    y_true = np.load(_yt_npy) if _yt_npy.exists() else None
    if y_true is not None:
        print(f"[cache] HIT  {SET}/_y_true  {y_true.shape} (labels)")
    members = {}          # label -> (N, 5) prob matrix

    # total members for the "i/M" progress counter in the headers below
    _n_members = len(FULL_MODELS) + len(CHECKPOINT_MEMBERS) + (1 if USE_STAGE2 else 0)
    _mi = 0

    # --- full 5-class models ---
    for run in FULL_MODELS:
        cfg = load_cfg(run)
        _mi += 1
        print(f"[member {_mi}/{_n_members}] {run}", flush=True)
        cname = f"{run}_{_member_ckpt_stem(run)}"          # name only — no file access
        p = _predict_member(cfg, (lambda run=run: ckpt_path_of(run)),
                            df, device, cache_dir, SET, cname)
        members[run] = p
        if y_true is None:
            yt, _, _, _ = sc._predict_dataframe(   # labels once (cheap, same df)
                cfg, build_model_generic(cfg).to(device).eval(),
                df, device, _dummy_loss, amp=False, channels_last=False,
                batch_size=8, num_workers=0)
            y_true = yt
            arr = yt.detach().cpu().numpy() if hasattr(yt, "detach") else np.asarray(yt)
            _yt_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(_yt_npy, arr)                 # cache labels for future CPU-only runs
            print(f"[cache] save {SET}/_y_true  {arr.shape} (labels)")

    # --- extra members from specific checkpoints (each its own 5-class voter) ---
    for spec in CHECKPOINT_MEMBERS:
        run, ckpt = spec["run"], spec["checkpoint"]
        cfg = load_cfg(run)
        label = _ckpt_member_label(run, ckpt)
        _mi += 1
        print(f"[member {_mi}/{_n_members}] {label}", flush=True)
        resolver = (lambda run=run, ckpt=ckpt: ckpt_path_of(run, ckpt))
        stem = _member_ckpt_stem(run, ckpt) or Path(resolver()).stem   # resolve only if "last"
        members[label] = _predict_member(cfg, resolver, df, device, cache_dir, SET,
                                         f"{run}_{stem}")

    # --- Stage-2 composite (one member, per-class dedicated model) ---
    if USE_STAGE2:
        _mi += 1
        comp = np.zeros((len(df), len(tasks)), dtype=float)
        for disease, run in STAGE2_GROUP.items():
            cfg = load_cfg(run)
            print(f"[member {_mi}/{_n_members}] stage2/{disease}: {run}", flush=True)
            cname = f"{run}_{_member_ckpt_stem(run)}"
            p = _predict_member(cfg, (lambda run=run: ckpt_path_of(run)),
                                df, device, cache_dir, SET, cname)   # (N,1)
            comp[:, tasks.index(disease)] = p[:, 0]
        members["final_stage2 (5 per-disease composite)"] = comp

    # --- combine members + metrics (F1/P/R/Spec at the calibrated per-class thresholds).
    # class_weights (from a weighted_ensemble summary) -> per-class weighted blend;
    # otherwise the flat equal-weight probability average. ---
    _labels_order = list(members.keys())
    stack = np.stack([members[l] for l in _labels_order], axis=0)   # (M, N, 5)
    if class_weights:
        ens, _used_weights = _weighted_blend(stack, _labels_order, tasks, class_weights)
        _blend = "weighted"
        print(f"[ensemble] WEIGHTED per-class blend from {weights_source}")
    else:
        ens = stack.mean(axis=0)
        _used_weights = None
        _blend = "prob-average"
    ens_metrics = sc.compute_metrics(y_true, ens, tasks, threshold=thr_vec)

    # each member's own mean AUROC (for reference; AUROC is threshold-free)
    per_member = {lab: sc.compute_metrics(y_true, p, tasks)["macro"]["mean_auroc"]
                  for lab, p in members.items()}

    _pc_keys = ("auroc", "auprc", "f1", "precision", "recall", "specificity")
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "set": SET, "n_images": int(len(df)),
        "members": list(members.keys()),
        "blend": _blend,
        "weights_source": weights_source or None,
        "class_weights": _used_weights,          # per-class normalized weights, or None (flat)
        "thresholds_source": thr_source or "(none -> 0.5)",
        "thresholds": {t: thr_vec[i] for i, t in enumerate(tasks)},
        "ensemble_mean_auroc": ens_metrics["macro"]["mean_auroc"],
        "ensemble_macro": ens_metrics["macro"],
        "ensemble_per_class": {t: {k: ens_metrics["per_task"][t][k] for k in _pc_keys}
                               for t in tasks},
        "per_member_mean_auroc": per_member,
    }

    mac = ens_metrics["macro"]
    lines = ["=" * 78,
             f"ENSEMBLE ({_blend})  —  set={SET}  images={len(df)}",
             f"generated: {summary['generated']}",
             f"thresholds (F1/P/R/Spec): {thr_source or '(none -> 0.5)'}"]
    if _used_weights is not None:
        lines.append(f"weights: {weights_source}")
    lines += ["=" * 78,
              f"  ENSEMBLE mean AUROC={mac['mean_auroc']:.4f}  AUPRC={mac['mean_auprc']:.4f}  "
              f"F1={mac['mean_f1']:.4f}  P={mac['mean_precision']:.4f}  "
              f"R={mac['mean_recall']:.4f}  Spec={mac['mean_specificity']:.4f}",
              "  per-class:"]
    for t in tasks:
        pc = summary["ensemble_per_class"][t]
        lines.append(f"    {t:<18} AUROC={pc['auroc']:.4f}  AUPRC={pc['auprc']:.4f}  "
                     f"F1={pc['f1']:.4f}  P={pc['precision']:.4f}  R={pc['recall']:.4f}  "
                     f"Spec={pc['specificity']:.4f}  (thr={summary['thresholds'][t]:.4f})")
    if _used_weights is not None:
        lines += ["  " + "-" * 74, "  per-class weights (renormalized over members):"]
        for t in tasks:
            lines.append(f"    {t:<18} " + "  ".join(f"{lab}={w:.3f}"
                         for lab, w in _used_weights[t].items()))
    lines += ["  " + "-" * 74, "  members (own mean AUROC):"]
    for lab, a in per_member.items():
        lines.append(f"    {a:.4f}   {lab}")
    lines += ["=" * 78]
    txt = "\n".join(lines) + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"ensemble_{SET}_summary.json"
    txt_path  = out_dir / f"ensemble_{SET}_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote -> {json_path}")
    print(f"wrote -> {txt_path}")


def _load_class_weights(spec):
    """Load per-class member weights from a weighted_ensemble summary (its json file, or
    a folder containing weighted_*_summary.json). Runs on the LAUNCHER; the resolved dict
    is passed into run_ensemble (local) / handed to the remote (modal), so it loads
    regardless of the runs volume. Returns (weights_per_class, source_label), or
    (None, "") when spec is falsy. weights_per_class = {task: {member_label: weight}}."""
    if not spec:
        return None, ""
    import json
    p = Path(spec)
    if not p.is_absolute():                       # try relative to others/, then CWD
        cand = Path(__file__).resolve().parent / p
        p = cand if cand.exists() else p
    if p.is_dir():
        hits = sorted(p.glob("weighted_*_summary.json"))
        if not hits:
            raise FileNotFoundError(f"no weighted_*_summary.json in {p}")
        p = hits[0]
    data = json.load(open(p, encoding="utf-8"))
    w = data.get("weights_per_class")
    if not w:
        raise ValueError(f"{p} has no 'weights_per_class' block")
    return w, str(p)


def _weighted_blend(stack, labels_order, tasks, class_weights):
    """Per-class weighted average of the members. stack is (M, N, T) with axis 0 aligned
    to labels_order; class_weights = {task: {member_label: weight}}. Weights are
    renormalized per class over the members present (all-zero -> uniform). Returns
    (ens (N, T), used_weights {task: {label: normalized_weight}})."""
    M, N, T = stack.shape
    ens = np.zeros((N, T), dtype=float)
    used = {}
    for c, t in enumerate(tasks):
        wmap = class_weights.get(t)
        if wmap is None:
            raise KeyError(f"weights file has no class '{t}' (has: {list(class_weights)})")
        raw = np.empty(M, dtype=float)
        for i, lab in enumerate(labels_order):
            if lab not in wmap:
                raise KeyError(f"weights file class '{t}' has no member '{lab}' "
                               f"(file members: {list(wmap)})")
            raw[i] = float(wmap[lab])
        s = raw.sum()
        w = raw / s if s > 0 else np.full(M, 1.0 / M)      # renormalize; degenerate -> uniform
        ens[:, c] = np.tensordot(w, stack[:, :, c], axes=(0, 0))
        used[t] = {lab: float(w[i]) for i, lab in enumerate(labels_order)}
    return ens, used


def _load_thresholds(run_name: str):
    """Read {task: threshold} from a run's LOCAL results/thresholds.json (the repo
    always ships it, unlike the runs volume). Runs on the LAUNCHER; the resolved dict
    is then passed to run_ensemble (local) or handed to the remote function (modal), so
    the thresholds always load even if this run isn't a member / isn't on the volume.
    Returns (thr_map, source_label); a missing file -> ({}, "<...> (MISSING)")."""
    import json
    p = PKG_ROOT / run_name / "results" / "thresholds.json"
    if p.exists():
        thr_map = json.load(open(p, encoding="utf-8")).get("thresholds", {})
        return thr_map, f"{run_name}/results/thresholds.json (local)"
    return {}, f"{run_name}/results/thresholds.json (MISSING)"


# ----------------------------- local execution -----------------------------
def run_local():
    def load_cfg(run):
        return sc.load_config(PKG_ROOT / run, verbose=False)

    def ckpt_path_of(run, checkpoint=None):
        base = PKG_ROOT / run / "results" / "checkpoints"
        sub = CKPT_SUBPATH.get(run, "best.pt")            # file, maybe in a stage subfolder
        if checkpoint is None:                            # default member = CKPT_SUBPATH file
            p = base / sub
            if not p.exists():
                raise FileNotFoundError(f"missing {p} — fetch this run's best.pt first "
                                        f"(or use RUN_ON='modal')")
            return p
        return sc._resolve_resume(checkpoint, base / Path(sub).parent)   # honor stage subfolder

    thr_map, thr_src = _load_thresholds(THRESHOLDS_FROM)
    cw_map, cw_src = _load_class_weights(WEIGHTS_FROM)
    out_dir = _out_subdir(Path(__file__).resolve().parent)   # others/ensembling_results/<stamp>
    run_ensemble(load_cfg, ckpt_path_of, out_dir, thr_map, thr_src, cw_map, cw_src)
    return out_dir


# ----------------------------- modal execution -----------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

# Build the app ONLY on the launching (local) side. In the remote container this
# module is imported (to reuse run_ensemble), and modal.is_local() is False there,
# so we skip rebuilding the app/image/volumes.
if _MODAL_OK and modal.is_local():
    _ref_cfg = sc.load_config(PKG_ROOT / FULL_MODELS[0], verbose=False)
    app = modal.App(f"ensemble-{SET}")
    _runs_vol = modal.Volume.from_name(_ref_cfg["modal"]["runs_volume"], create_if_missing=True)

    # Mount EVERY distinct data volume any member uses, each at its own mount point,
    # so members resolve images at their own remote_data_root. Members can live on
    # different volumes (e.g. small-res runs -> chexpert-data /data; native-res runs
    # -> chexpert-native-data /data_native), so one mount is not enough.
    _all_runs = (list(FULL_MODELS)
                 + [m["run"] for m in CHECKPOINT_MEMBERS]
                 + (list(STAGE2_GROUP.values()) if USE_STAGE2 else []))
    _volumes = {_ref_cfg["modal"]["runs_mount"]: _runs_vol}
    _needs_transformers = False
    for _run in _all_runs:
        _rcfg = sc.load_config(PKG_ROOT / _run, verbose=False)
        _mc = _rcfg["modal"]
        _mount, _vname = _mc["data_mount"], _mc["data_volume"]
        if _mount not in _volumes:
            _volumes[_mount] = modal.Volume.from_name(_vname, create_if_missing=True)
        if _rcfg["model"].get("arch") == "raddino":
            _needs_transformers = True   # RAD-DINO is a HF model — timm can't load it

    # RAD-DINO members need `transformers` in the image (added in a second pip layer so
    # the shared base layer/cache is untouched when no raddino member is present).
    _image = sc.modal_image(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        extra_pip=["transformers"] if _needs_transformers else None)

    # If EVERY member (and the labels) is already cached locally, the remote run is
    # pure CPU averaging -> reserve NO GPU (don't idle-book a B200). Otherwise use the
    # configured GPU. Decided at VM-build time from the LOCAL cache (kept in sync by
    # _sync_cache_down); the up-sync then mirrors it to the volume the remote reads.
    _all_cached = _members_all_cached(Path(__file__).resolve().parent, SET)
    _resources = sc.modal_resources(_ref_cfg)
    if _all_cached:
        _resources.pop("gpu", None)           # all members cached -> CPU-only VM
        print(f"[gpu] all members cached for {SET} -> CPU-only VM (no GPU reserved)")
    else:
        if GPU:
            _resources["gpu"] = GPU           # explicit GPU override (see CONFIG at top)
        print(f"[gpu] some members need compute -> GPU={_resources.get('gpu')}")
    if CPU_CORES is not None:
        _resources["cpu"] = CPU_CORES         # requested CPU cores override
    if MEMORY_GB is not None:
        _resources["memory"] = int(MEMORY_GB) * 1024   # GB -> MB (modal_resources' unit)

    @app.function(
        image=_image,
        volumes=_volumes,
        serialized=True,
        **_resources,
    )
    def ensemble_remote(thr_map=None, thr_source="", class_weights=None, weights_source=""):
        # Self-contained: import everything INSIDE so nothing from __main__ (which
        # references shared_code) gets cloudpickled. Reuse run_ensemble from the
        # MOUNTED module (imported here, where shared_code is importable).
        import sys as _sys
        from pathlib import Path as _P
        # shared_code is at /root/training_scripts; THIS module at /root/training_scripts/others
        for _p in ("/root/training_scripts", "/root/training_scripts/others"):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        import shared_code as _sc
        import ensample as _E                   # mounted; app not rebuilt (is_local False)
        runs_mount = _P("/runs")

        def load_cfg(run):
            return _sc.remote_cfg(_sc.load_config(_P("/root/training_scripts") / run, verbose=False))

        def ckpt_path_of(run, checkpoint=None):
            base = runs_mount / run / "results" / "checkpoints"
            sub = _E.CKPT_SUBPATH.get(run, "best.pt")
            if checkpoint is None:
                return base / sub
            return _sc._resolve_resume(checkpoint, base / _P(sub).parent)   # honor stage subfolder

        out_dir = _E._out_subdir(runs_mount)         # /runs/ensembling_results/<timestamp>
        try:
            # thresholds AND per-class weights are resolved locally by the launcher and
            # passed in, so they load regardless of what's on the runs volume.
            _E.run_ensemble(load_cfg, ckpt_path_of, out_dir, thr_map, thr_source,
                            class_weights, weights_source)
        finally:
            _runs_vol.commit()                        # persist before the local fetch reads it
        # hand the volume-relative POSIX subpath back so the launcher can download it
        return out_dir.relative_to(runs_mount).as_posix()


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but modal isn't installed; set RUN_ON='local'.")
        _base_local = Path(__file__).resolve().parent
        _runs_volume = _ref_cfg["modal"]["runs_volume"]
        _sync_cache_up(_runs_volume, _base_local)    # push local cache so remote can reuse it
        _thr_map, _thr_src = _load_thresholds(THRESHOLDS_FROM)   # resolve locally -> pass to remote
        _cw_map, _cw_src = _load_class_weights(WEIGHTS_FROM)     # weighted blend (or None)
        with modal.enable_output():
            with app.run():
                remote_sub = ensemble_remote.remote(_thr_map, _thr_src,
                                                    _cw_map, _cw_src)   # volume-relative POSIX subpath
        _sync_cache_down(_runs_volume, _base_local)  # pull GPU-computed cache back to this PC
        # remote run + volume commit are done; pull the results folder down locally.
        if remote_sub:
            local_dir = _fetch_results_from_modal(_runs_volume, remote_sub, _base_local)
            _update_best_tracker(SET, local_dir)     # promote if it beats the local best
    elif RUN_ON == "local":
        out_dir = run_local()
        _update_best_tracker(SET, out_dir)           # promote if it beats the local best
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
