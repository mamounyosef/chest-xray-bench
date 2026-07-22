"""
weighted_ensemble.py
====================
PER-CLASS WEIGHTED ensemble (not a flat 1/3 average), scored on ONE split
(valid200 by default; never test500 unless you explicitly set it).

For each of the 5 tasks INDEPENDENTLY we search a weight vector over the members
(non-negative, summing to 1, on a discrete grid) that MAXIMIZES that class's
AUROC — so every class can lean on whichever members are strongest for it. The
weights are fit with 5-fold CROSS-VALIDATION over the split (fit on 4 folds,
repeat 5×, average the 5 fold-weight vectors) so we don't overfit 3×5 free
parameters to ~234 images. A leave-one-fold-out (OOF) AUROC is also reported as
the honest generalization estimate, alongside the in-sample number and the flat
1/3 baseline.

This is PURELY a post-processing step on the SAME per-member probability cache
that ensample.py writes: others/cache_runs/<set>/<run>_best.npy (+ _y_true.npy).
If every member is already cached (they are, once ensample.py has run these
members on this split), NOTHING is recomputed and this runs instantly, locally.
Only a cache MISS needs a member's checkpoint + a GPU — handled exactly like
ensample.py (Modal or local, with the same up/down cache sync).

Writes into its OWN timestamped subfolder:
    local :  training_scripts/others/weighted_ensemble/results/<timestamp>/weighted_<SET>_summary.{json,txt}
    modal :  /runs/weighted_ensemble_results/<timestamp>/  (auto-pulled down to the local path above)

Run:  python training_scripts/others/weighted_ensemble/weighted_ensemble.py   (honours RUN_ON below)
"""

import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
RUN_ON = "local"        # "modal" | "local". If every member is cached, "local" is instant.
SET    = "valid200"     # scored/searched split — "valid200" | "test500" | "val"

# The members to blend (each contributes ALL five classes from its best.pt). The
# weight search is over THESE, in this order. Cache names match ensample.py's
# (<run>_best.npy) so the shared others/cache_runs is reused verbatim.
MEMBERS = [
    "convnext_base_22k_1600x1312",
    "medmae_vitb_nih_B_768_s2",
    "rad_dino_vitB_768",
]

# ---- weight-search knobs --------------------------------------------------
GRID_STEP = 0.1         # each member weight ∈ {0, STEP, 2*STEP, ..., 1}, summing to 1.
                        # 0.1 -> 66 weight vectors per class (3 members). Finer = more.
N_FOLDS   = 5           # cross-validation folds over the split (fit on N_FOLDS-1, repeat).
SEED      = 42          # KFold shuffle seed (reproducible fold assignment).
TIE_EPS   = 1e-9        # AUROC ties within this are broken toward the most-uniform weights.

# Per-class thresholds for the F1/precision/recall/specificity columns (AUROC/AUPRC
# are threshold-free). Taken from this run's results/thresholds.json; defaults to the
# FIRST member. Missing file/task -> 0.5.
THRESHOLDS_FROM = MEMBERS[0]

# Optional tag appended to the timestamped output folder (self-describing runs).
RUN_TAG = ""

# ---- compute knobs (ONLY used on a cache MISS; a full cache ignores all of these) ----
GPU        = "H200"     # Modal GPU when a member must be computed: T4|L4|A10G|A100|A100-80GB|H100|H200.
BATCH_SIZE = 512        # inference batch size per member (None -> that run's val_batch_size).
CPU_CORES  = 14         # Modal container CPU cores (None -> reference run's).
MEMORY_GB  = 50         # Modal container RAM in GB (None -> reference run's).
NUM_WORKERS_MODAL = 24  # DataLoader workers per member when RUN_ON == "modal"
NUM_WORKERS_LOCAL = 2   # DataLoader workers per member when RUN_ON == "local"
NUM_WORKERS = NUM_WORKERS_MODAL if RUN_ON == "modal" else NUM_WORKERS_LOCAL
PREFETCH_FACTOR = 4     # batches prefetched per worker (workers > 0 only)
REFRESH_CACHE = False   # True -> ignore + overwrite cached member probs (after RETRAINING a ckpt)
CKPT_SUBPATH = {}       # {run: "stage_subdir/best.pt"} for two-stage runs (none here)
# ===========================================================================

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _resolve_pkg_root() -> Path:
    # this script lives in training_scripts/others/weighted_ensemble/, so
    # shared_code.py is TWO levels up (training_scripts/).
    here = Path(__file__).resolve().parent
    for cand in (Path("/root/training_scripts"), here.parent.parent, here.parent, here):
        if (cand / "shared_code.py").exists():
            return cand
    return here.parent.parent


PKG_ROOT = _resolve_pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc          # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402


# ======================= model rebuild (cache MISS only) ====================
def build_model_generic(cfg: dict):
    """Rebuild a run's model EXACTLY as its own train.py does, so best.pt loads.
    (Copied from ensample.py — same members, same rules.) Only invoked on a cache
    miss; a fully-cached run never calls this."""
    import timm
    import torch.nn as nn
    import torchvision
    name = cfg["model"]["name"]
    n = sc.num_output_logits(cfg)
    if cfg["model"].get("arch") == "medmae_vitb":
        return sc.build_medmae_vit(cfg, load_pretrained=False)
    if cfg["model"].get("arch") == "raddino":
        return sc.build_raddino_vit(cfg, load_pretrained=False)
    if "." in name:                                  # timm id
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


def _dummy_loss(logits, targets):
    return logits.sum() * 0.0        # we only need probabilities, not the loss


# ==================== per-member prob cache (shared with ensample) ==========
# SAME cache as ensample.py: others/cache_runs/<set>/<run>_best.npy (+ _y_true.npy).
# NAME-based: a present .npy is a HIT reused as-is (no GPU, no checkpoint needed).
def _member_ckpt_stem(run: str) -> str:
    return Path(CKPT_SUBPATH.get(run, "best.pt")).stem       # "best" for best.pt


def _cache_npy(cache_dir: Path, set_name: str, cache_name: str) -> Path:
    return Path(cache_dir) / set_name / f"{cache_name}.npy"


def _predict(cfg: dict, ckpt_path: Path, df, device, desc: str = None) -> np.ndarray:
    """Probabilities (N, len(tasks)) for one checkpoint over `df`. `desc` labels the
    throttled per-batch progress/ETA line printed by shared_code._predict_dataframe."""
    print(f"     [load] building model + weights from {Path(ckpt_path).name} ...", flush=True)
    model = build_model_generic(cfg).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sc._unwrap(model).load_state_dict(ck["model"])
    bs = int(BATCH_SIZE or cfg["dataloader"].get("val_batch_size",
                                                 cfg["dataloader"]["batch_size"]))
    nw = int(NUM_WORKERS) if NUM_WORKERS is not None \
        else int(cfg["dataloader"].get("val_num_workers", 4))
    eff_cfg = cfg
    if PREFETCH_FACTOR is not None:
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


def _predict_member(cfg, ckpt_resolver, df, device, cache_dir, set_name, cache_name):
    """_predict with the shared NAME-based cache. `ckpt_resolver` (0-arg) is invoked
    ONLY on a miss, so a hit never touches the checkpoint file."""
    npy = _cache_npy(cache_dir, set_name, cache_name)
    if not REFRESH_CACHE and npy.exists():
        arr = np.load(npy)
        print(f"[cache] HIT  {set_name}/{cache_name}  {arr.shape}  (no compute)")
        return arr
    print(f"[cache] MISS {set_name}/{cache_name} -> computing on {device.type.upper()}", flush=True)
    ckpt_path = ckpt_resolver()
    p = _predict(cfg, ckpt_path, df, device, desc=cache_name)
    try:
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, p)
        print(f"[cache] {'REFRESH' if REFRESH_CACHE else 'save'} {set_name}/{cache_name}")
    except Exception as e:
        print(f"[cache] WARNING: could not save {set_name}/{cache_name}: {e}")
    return p


def _expected_cache_names():
    return [f"{run}_{_member_ckpt_stem(run)}" for run in MEMBERS]


def _members_all_cached(cache_dir: Path, set_name: str) -> bool:
    """True iff EVERY member's probs AND the labels are already cached for `set_name`
    -> the Modal VM can run CPU-only (no GPU). REFRESH_CACHE -> False."""
    if REFRESH_CACHE:
        return False
    cdir = Path(cache_dir) / set_name
    needed = [f"{n}.npy" for n in _expected_cache_names()] + ["_y_true.npy"]
    missing = [f for f in needed if not (cdir / f).exists()]
    if missing:
        print(f"[gpu] not all cached for {set_name} — missing: {missing}")
    return not missing


# --------- Modal cache sync (identical contract to ensample.py) ------------
def _sync_cache_up(runs_volume: str, cache_dir: Path):
    """Push local others/cache_runs/ up to /cache_runs BEFORE a Modal run."""
    import subprocess
    L = Path(cache_dir)
    if not L.exists() or not any(L.rglob("*.npy")):
        print("[cache-sync] up: no local cache_runs yet (skip)")
        return
    cmd = [sys.executable, "-m", "modal", "volume", "put", "--force",
           runs_volume, str(L), "/"]
    print(f"[cache-sync] up: {L} -> {runs_volume}:/cache_runs")
    subprocess.run(cmd, check=False)


def _sync_cache_down(runs_volume: str, cache_parent: Path):
    """Pull /cache_runs/ back down to others/cache_runs/ AFTER a Modal run."""
    import subprocess
    dest = Path(cache_parent)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "modal", "volume", "get", "--force",
           runs_volume, "cache_runs", str(dest)]
    print(f"[cache-sync] down: {runs_volume}:/cache_runs -> {dest}")
    subprocess.run(cmd, check=False)


def _fetch_results_from_modal(runs_volume: str, remote_sub: str, dest_parent: Path):
    """Pull the freshly written remote results folder down. `modal volume get` maps
    each file to dest_parent / entry.relative_to(remote_path.parent), so passing the
    pre-created dest_parent recreates the <ts>/ subfolder under it (see ensample.py)."""
    import subprocess
    from pathlib import PurePosixPath
    remote_posix = remote_sub.replace("\\", "/")
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    final_dir = dest_parent / PurePosixPath(remote_posix).name
    cmd = [sys.executable, "-m", "modal", "volume", "get",
           runs_volume, remote_posix, str(dest_parent)]
    print(f"[fetch] modal volume get {runs_volume} {remote_posix} -> {dest_parent}")
    subprocess.run(cmd, check=True)
    print(f"[fetch] downloaded weighted-ensemble results -> {final_dir}")
    return final_dir


def _out_subdir(results_root: Path) -> Path:
    """A fresh timestamped subfolder under results_root (+ RUN_TAG)."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{RUN_TAG}" if RUN_TAG else stamp
    return Path(results_root) / name


def _load_thresholds(run_name: str):
    """Read {task: threshold} from a run's LOCAL results/thresholds.json (the repo
    always ships it). Resolved on the launcher; passed into the core routine so it
    loads regardless of what's on the runs volume."""
    import json
    p = PKG_ROOT / run_name / "results" / "thresholds.json"
    if p.exists():
        thr_map = json.load(open(p, encoding="utf-8")).get("thresholds", {})
        return thr_map, f"{run_name}/results/thresholds.json (local)"
    return {}, f"{run_name}/results/thresholds.json (MISSING)"


# ========================= weight grid + CV search =========================
def _weight_grid(n_members: int, step: float):
    """All weight vectors of length n_members, entries in {0, step, ..., 1}, summing
    to 1 (stars-and-bars over k = round(1/step) units). For 3 members & step 0.1 this
    is 66 vectors."""
    k = int(round(1.0 / step))
    out = []

    def rec(parts_left, remaining, cur):
        if parts_left == 1:
            out.append(tuple((v / k) for v in (cur + [remaining])))
            return
        for v in range(remaining + 1):
            rec(parts_left - 1, remaining - v, cur + [v])

    rec(n_members, k, [])
    return out


def _blend_col(probs, w, c):
    """Blend column c of the (M, N, T) member-prob stack with weight vector w (len M)."""
    return np.tensordot(np.asarray(w), probs[:, :, c], axes=(0, 0))   # (N,)


def _safe_auroc(y_col, p_col):
    """roc_auc_score guarded against a single-class subset (returns np.nan)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_col)
    if y.min() == y.max():                 # only one class present -> undefined
        return np.nan
    return float(roc_auc_score(y, np.asarray(p_col)))


def _pick_weights_on(idx, probs, y_true, c, grid, uniform):
    """The grid weight vector maximizing class-c AUROC on rows `idx`. Ties (within
    TIE_EPS) are broken toward the most-uniform vector (min L2 to `uniform`). Returns
    (best_w, best_auc); if AUROC is undefined for every vector on `idx` -> (uniform, nan)."""
    best_auc, cands = -np.inf, []
    for w in grid:
        a = _safe_auroc(y_true[idx, c], _blend_col(probs, w, c)[idx])
        if np.isnan(a):
            continue
        if a > best_auc + TIE_EPS:
            best_auc, cands = a, [w]
        elif a >= best_auc - TIE_EPS:
            cands.append(w)
    if not cands:
        return np.asarray(uniform), np.nan
    best = min(cands, key=lambda w: float(np.sum((np.asarray(w) - uniform) ** 2)))
    return np.asarray(best), best_auc


def _cv_weight_search(probs, y_true, tasks):
    """Per-class 5-fold-CV weighted search. Returns a dict per task with the final
    (fold-averaged) weight vector, the 5 per-fold vectors, and the OOF AUROC (each
    fold's own weights scored on its held-out fold — the honest estimate)."""
    from sklearn.model_selection import KFold
    M, N, T = probs.shape
    uniform = np.full(M, 1.0 / M)
    grid = _weight_grid(M, GRID_STEP)
    print(f"[search] grid: {len(grid)} weight vectors/class (step={GRID_STEP}, {M} members)  "
          f"| {N_FOLDS}-fold CV (seed={SEED})  | {N} images")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(N)))

    result = {}
    for c, t in enumerate(tasks):
        fold_ws, oof_scores = [], []
        for train_idx, val_idx in folds:
            w, _ = _pick_weights_on(train_idx, probs, y_true, c, grid, uniform)
            fold_ws.append(w)
            oof_scores.append(_safe_auroc(y_true[val_idx, c], _blend_col(probs, w, c)[val_idx]))
        fold_ws = np.stack(fold_ws, axis=0)                  # (n_folds, M)
        final_w = fold_ws.mean(axis=0)                       # sums to 1 (each row does)
        oof = float(np.nanmean(oof_scores)) if np.any(~np.isnan(oof_scores)) else float("nan")
        result[t] = {"weights": final_w, "fold_weights": fold_ws, "oof_auroc": oof}
        _wtxt = "  ".join(f"{MEMBERS[m]}={final_w[m]:.3f}" for m in range(M))
        print(f"[search] {t:<18} OOF AUROC={oof:.4f}  weights: {_wtxt}")
    return result


def _apply_weights(probs, per_class):
    """Assemble the (N, T) weighted-blend probability matrix from per-class weights."""
    M, N, T = probs.shape
    ens = np.zeros((N, T), dtype=float)
    for c, t in enumerate(list(per_class.keys())):
        ens[:, c] = _blend_col(probs, per_class[t]["weights"], c)
    return ens


# ============================ core routine =================================
def run_weighted_ensemble(load_cfg, ckpt_path_of, out_dir: Path, cache_dir: Path,
                          thr_map: dict = None, thr_source: str = ""):
    """Load/compute each member's probs (shared cache), run the per-class CV weighted
    search, and write a comprehensive summary. `load_cfg`/`ckpt_path_of` mirror
    ensample.py; `cache_dir` is the SHARED others/cache_runs (or /runs/cache_runs
    remotely); `thr_map` are the per-class thresholds for F1/P/R/Spec."""
    import json, pandas as pd
    from datetime import datetime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref = load_cfg(MEMBERS[0])
    tasks = list(ref["tasks"])
    data_dir = Path(ref["paths"]["data_dir"])
    df = pd.read_csv(data_dir / ref["paths"][f"{SET}_csv"])
    print(f"[weighted] {SET}: {len(df)} images  |  tasks={tasks}  device={device}")
    print(f"[cache] dir: {Path(cache_dir) / SET}")

    thr_map = thr_map or {}
    if thr_map:
        print(f"[weighted] thresholds from {thr_source or '(provided)'}")
    else:
        print(f"[weighted] WARNING: no thresholds ({thr_source or 'none'}) -> F1/P/R/Spec at 0.5")
    thr_vec = [float(thr_map.get(t, 0.5)) for t in tasks]

    # cached labels (fixed per SET) — an all-cached run needs no model build at all.
    _yt_npy = Path(cache_dir) / SET / "_y_true.npy"
    y_true = np.load(_yt_npy) if _yt_npy.exists() else None
    if y_true is not None:
        print(f"[cache] HIT  {SET}/_y_true  {y_true.shape} (labels)")

    # --- gather every member's (N, T) prob matrix (cache or compute) ---
    member_probs = {}
    for i, run in enumerate(MEMBERS, 1):
        cfg = load_cfg(run)
        print(f"[member {i}/{len(MEMBERS)}] {run}", flush=True)
        cname = f"{run}_{_member_ckpt_stem(run)}"
        p = _predict_member(cfg, (lambda run=run: ckpt_path_of(run)),
                            df, device, cache_dir, SET, cname)
        member_probs[run] = np.asarray(p)
        if y_true is None:                       # compute + cache labels once (cheap)
            yt, _, _, _ = sc._predict_dataframe(
                cfg, build_model_generic(cfg).to(device).eval(),
                df, device, _dummy_loss, amp=False, channels_last=False,
                batch_size=8, num_workers=0)
            y_true = yt.detach().cpu().numpy() if hasattr(yt, "detach") else np.asarray(yt)
            _yt_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(_yt_npy, y_true)
            print(f"[cache] save {SET}/_y_true  {y_true.shape} (labels)")

    probs = np.stack([member_probs[r] for r in MEMBERS], axis=0)      # (M, N, T)
    y_true = np.asarray(y_true)

    # --- per-class CV weighted search ---
    per_class = _cv_weight_search(probs, y_true, tasks)

    # --- assemble blends + metrics ---
    _PC = ("auroc", "auprc", "f1", "precision", "recall", "specificity")

    def _metrics(prob):
        m = sc.compute_metrics(y_true, prob, tasks, threshold=thr_vec)
        return {"macro": m["macro"],
                "per_class": {t: {k: m["per_task"][t][k] for k in _PC} for t in tasks}}

    weighted = _metrics(_apply_weights(probs, per_class))            # per-class weighted
    flat = _metrics(probs.mean(axis=0))                             # 1/M baseline
    per_member = {r: _metrics(member_probs[r]) for r in MEMBERS}    # each model, per class

    oof_mean = float(np.nanmean([per_class[t]["oof_auroc"] for t in tasks]))

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "set": SET, "n_images": int(len(df)),
        "members": list(MEMBERS),
        "grid_step": GRID_STEP, "n_folds": N_FOLDS, "seed": SEED,
        "thresholds_source": thr_source or "(none -> 0.5)",
        "thresholds": {t: thr_vec[i] for i, t in enumerate(tasks)},
        # the searched decision rule (per class, per member) + fold detail
        "weights_per_class": {t: {MEMBERS[m]: float(per_class[t]["weights"][m])
                                  for m in range(len(MEMBERS))} for t in tasks},
        "fold_weights_per_class": {
            t: [{MEMBERS[m]: float(per_class[t]["fold_weights"][f, m])
                 for m in range(len(MEMBERS))} for f in range(N_FOLDS)] for t in tasks},
        # honest (OOF) vs in-sample vs baseline
        "oof_mean_auroc": oof_mean,
        "oof_auroc_per_class": {t: per_class[t]["oof_auroc"] for t in tasks},
        "weighted_mean_auroc": weighted["macro"]["mean_auroc"],
        "weighted_macro": weighted["macro"],
        "weighted_per_class": weighted["per_class"],
        "flat_mean_auroc": flat["macro"]["mean_auroc"],
        "flat_macro": flat["macro"],
        "flat_per_class": flat["per_class"],
        "gain_mean_auroc_vs_flat": weighted["macro"]["mean_auroc"] - flat["macro"]["mean_auroc"],
        # comprehensive per-member × per-class breakdown
        "per_member": {r: {"macro": per_member[r]["macro"],
                           "per_class": per_member[r]["per_class"]} for r in MEMBERS},
    }

    txt = _render_txt(summary, tasks, _PC)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"weighted_{SET}_summary.json"
    txt_path  = out_dir / f"weighted_{SET}_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote -> {json_path}")
    print(f"wrote -> {txt_path}")


def _render_txt(s: dict, tasks, PC) -> str:
    """Comprehensive readable twin of the summary json."""
    def row(m):
        def f(x):
            import math
            return f"{x:.4f}" if isinstance(x, (int, float)) and math.isfinite(x) else "  nan"
        return (f"AUROC={f(m['auroc'])}  AUPRC={f(m['auprc'])}  F1={f(m['f1'])}  "
                f"P={f(m['precision'])}  R={f(m['recall'])}  Spec={f(m['specificity'])}")

    def macro(mac):
        return (f"AUROC={mac['mean_auroc']:.4f}  AUPRC={mac['mean_auprc']:.4f}  "
                f"F1={mac['mean_f1']:.4f}  P={mac['mean_precision']:.4f}  "
                f"R={mac['mean_recall']:.4f}  Spec={mac['mean_specificity']:.4f}")

    L = ["=" * 92,
         f"PER-CLASS WEIGHTED ENSEMBLE  —  set={s['set']}  images={s['n_images']}",
         f"generated: {s['generated']}",
         f"members ({len(s['members'])}): {', '.join(s['members'])}",
         f"grid step={s['grid_step']}   folds={s['n_folds']}   seed={s['seed']}",
         f"thresholds (F1/P/R/Spec): {s['thresholds_source']}",
         "=" * 92,
         "HEADLINE",
         f"  WEIGHTED mean AUROC = {s['weighted_mean_auroc']:.4f}   "
         f"(OOF honest = {s['oof_mean_auroc']:.4f})",
         f"  FLAT 1/{len(s['members'])} mean AUROC = {s['flat_mean_auroc']:.4f}   "
         f"-> gain = {s['gain_mean_auroc_vs_flat']:+.4f}",
         "-" * 92,
         "CHOSEN PER-CLASS WEIGHTS  (fold-averaged) + OOF AUROC"]
    for t in tasks:
        w = s["weights_per_class"][t]
        wtxt = "  ".join(f"{m}={w[m]:.3f}" for m in s["members"])
        L.append(f"    {t:<18} OOF={s['oof_auroc_per_class'][t]:.4f}   {wtxt}")
    L += ["-" * 92, "WEIGHTED ENSEMBLE",
          f"  macro : {macro(s['weighted_macro'])}", "  per-class:"]
    for t in tasks:
        L.append(f"    {t:<18} {row(s['weighted_per_class'][t])}  "
                 f"(thr={s['thresholds'][t]:.3f})")
    L += ["-" * 92, f"FLAT 1/{len(s['members'])} ENSEMBLE  (baseline)",
          f"  macro : {macro(s['flat_macro'])}", "  per-class:"]
    for t in tasks:
        L.append(f"    {t:<18} {row(s['flat_per_class'][t])}")
    L += ["-" * 92, "PER-MEMBER  (own scores — macro then per-class)"]
    for r in s["members"]:
        pm = s["per_member"][r]
        L.append(f"  {r}")
        L.append(f"    macro : {macro(pm['macro'])}")
        for t in tasks:
            L.append(f"      {t:<18} {row(pm['per_class'][t])}")
    L += ["=" * 92]
    return "\n".join(L) + "\n"


# ----------------------------- local execution -----------------------------
def _cache_dir_local() -> Path:
    # shared with ensample.py: training_scripts/others/cache_runs
    return Path(__file__).resolve().parent.parent / "cache_runs"


def _results_root_local() -> Path:
    # training_scripts/others/weighted_ensemble/results
    return Path(__file__).resolve().parent / "results"


def run_local():
    def load_cfg(run):
        return sc.load_config(PKG_ROOT / run, verbose=False)

    def ckpt_path_of(run, checkpoint=None):
        base = PKG_ROOT / run / "results" / "checkpoints"
        sub = CKPT_SUBPATH.get(run, "best.pt")
        if checkpoint is None:
            p = base / sub
            if not p.exists():
                raise FileNotFoundError(f"missing {p} — fetch this run's best.pt first "
                                        f"(or use RUN_ON='modal'). A cached member never "
                                        f"needs the .pt.")
            return p
        return sc._resolve_resume(checkpoint, base / Path(sub).parent)

    thr_map, thr_src = _load_thresholds(THRESHOLDS_FROM)
    out_dir = _out_subdir(_results_root_local())
    run_weighted_ensemble(load_cfg, ckpt_path_of, out_dir, _cache_dir_local(),
                          thr_map, thr_src)
    return out_dir


# ----------------------------- modal execution -----------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

if _MODAL_OK and modal.is_local():
    _ref_cfg = sc.load_config(PKG_ROOT / MEMBERS[0], verbose=False)
    app = modal.App(f"weighted-ensemble-{SET}")
    _runs_vol = modal.Volume.from_name(_ref_cfg["modal"]["runs_volume"], create_if_missing=True)

    # Mount every distinct data volume any member uses (members can live on different
    # volumes: small-res -> chexpert-data, native-res -> chexpert-native-data).
    _volumes = {_ref_cfg["modal"]["runs_mount"]: _runs_vol}
    _needs_transformers = False
    for _run in MEMBERS:
        _rcfg = sc.load_config(PKG_ROOT / _run, verbose=False)
        _mc = _rcfg["modal"]
        if _mc["data_mount"] not in _volumes:
            _volumes[_mc["data_mount"]] = modal.Volume.from_name(_mc["data_volume"],
                                                                 create_if_missing=True)
        if _rcfg["model"].get("arch") == "raddino":
            _needs_transformers = True       # RAD-DINO is a HF model — timm can't load it

    _image = sc.modal_image(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        extra_pip=["transformers"] if _needs_transformers else None)

    # All members cached -> pure CPU search: reserve NO GPU. Decided from the LOCAL
    # cache (kept in sync by _sync_cache_down); the up-sync mirrors it to the volume.
    _all_cached = _members_all_cached(_cache_dir_local(), SET)
    _resources = sc.modal_resources(_ref_cfg)
    if _all_cached:
        _resources.pop("gpu", None)
        print(f"[gpu] all members cached for {SET} -> CPU-only VM (no GPU reserved)")
    else:
        if GPU:
            _resources["gpu"] = GPU
        print(f"[gpu] some members need compute -> GPU={_resources.get('gpu')}")
    if CPU_CORES is not None:
        _resources["cpu"] = CPU_CORES
    if MEMORY_GB is not None:
        _resources["memory"] = int(MEMORY_GB) * 1024

    @app.function(image=_image, volumes=_volumes, serialized=True, **_resources)
    def weighted_remote(thr_map=None, thr_source=""):
        import sys as _sys
        from pathlib import Path as _P
        for _p in ("/root/training_scripts", "/root/training_scripts/others/weighted_ensemble"):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        import shared_code as _sc
        import weighted_ensemble as _W          # mounted; app not rebuilt (is_local False)
        runs_mount = _P("/runs")

        def load_cfg(run):
            return _sc.remote_cfg(_sc.load_config(_P("/root/training_scripts") / run, verbose=False))

        def ckpt_path_of(run, checkpoint=None):
            base = runs_mount / run / "results" / "checkpoints"
            sub = _W.CKPT_SUBPATH.get(run, "best.pt")
            if checkpoint is None:
                return base / sub
            return _sc._resolve_resume(checkpoint, base / _P(sub).parent)

        out_dir = _W._out_subdir(runs_mount / "weighted_ensemble_results")
        cache_dir = runs_mount / "cache_runs"          # shared remote cache (ensample's too)
        try:
            _W.run_weighted_ensemble(load_cfg, ckpt_path_of, out_dir, cache_dir,
                                     thr_map, thr_source)
        finally:
            _runs_vol.commit()
        return out_dir.relative_to(runs_mount).as_posix()


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but modal isn't installed; set RUN_ON='local'.")
        _cache_local = _cache_dir_local()
        _runs_volume = _ref_cfg["modal"]["runs_volume"]
        _sync_cache_up(_runs_volume, _cache_local)       # push local cache so remote reuses it
        _thr_map, _thr_src = _load_thresholds(THRESHOLDS_FROM)
        with modal.enable_output():
            with app.run():
                remote_sub = weighted_remote.remote(_thr_map, _thr_src)
        _sync_cache_down(_runs_volume, _cache_local.parent)   # pull GPU-computed cache back
        if remote_sub:
            _fetch_results_from_modal(_runs_volume, remote_sub, _results_root_local())
    elif RUN_ON == "local":
        run_local()
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
