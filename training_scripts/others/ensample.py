"""
ensample.py
===========
ONE ensembling script: flat probability averaging AND the per-class weighted search,
scored on ONE split (valid200 by default; never test500 unless you set it).

Every member loads its own checkpoint, runs inference with its OWN cfg (its own
CLAHE / image geometry / u-policy) and produces per-class probabilities. What happens
to those probabilities is decided by the single trigger WEIGHTS:

    WEIGHTS = "flat"        equal 1/M average of the members            (the default)
    WEIGHTS = "search"      FIT fresh per-class weights here and now (5-fold CV over
                            the split), score with them, and SAVE them
    WEIGHTS = "<path>"      REUSE the weights saved by an earlier "search" run

"search" and "<path>" are the whole weighted story: one fits and saves, the other
loads and applies. A fit NEVER overwrites an earlier one — every run writes its own
timestamped folder, so old weights stay exactly where they were.

The five per-disease Stage-2 AUC-M runs are given as ONE entry (STAGE2_GROUP) and
handled as ONE composite 5-class member: for each class, only its dedicated Stage-2
model contributes that class's probability. (Mathematically identical to five separate
members, since their single-class outputs never overlap.)

Writes — EVERY run, whatever the mode, into its OWN timestamped subfolder of ONE
results folder, so nothing is ever overwritten and each split keeps ONE "(best <set>)"
marker across all of them:
    local :  others/ensembling_results/<timestamp>/
    modal :  /runs/ensembling_results/<timestamp>/
The summary inside is named for what the run did: ensemble_<SET>_summary.{json,txt} for
a flat or reused-weights run, weighted_<SET>_summary.{json,txt} for a "search" run (it
carries the fitted weights). On a Modal run the freshly written folder is AUTOMATICALLY
pulled back down to the local path above (via `modal volume get`), so local + remote
stay in sync. others/weighted_ensemble/results/ holds the searches run before the two
scripts were merged — nothing new is written there, but WEIGHTS can still point at it.

Run:  python training_scripts/others/ensample.py   (honours RUN_ON below)
"""

import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
# --- what to run -----------------------------------------------------------
RUN_ON          = "auto"       # "auto" | "modal" | "local"   (auto: all cached -> local)
SET             = "test500"   # scored split: "valid200" | "test500" | "val"
FRONTAL_ONLY    = True         # drop lateral rows from the scored split. valid200 is
                               # 202 of 234 frontal and test500 is 518 of 668, and no
                               # model ever trained on a lateral. Caches and summaries
                               # are tagged separately, so the mixed-view artifacts of
                               # earlier runs are neither read nor overwritten.
WEIGHTS         = "flat"
                               # "flat" | "search" | a saved fit: its folder/json, just
                               # the folder NAME, or "best_valid200" (the marked winner)
COMBINE_SPACE   = "prob"       # blend space: "prob" | "logit" | "rank"
THRESHOLDS_FROM = None         # run supplying thresholds.json; None -> the first member
RUN_TAG         = ""           # appended to the output folder's timestamp

# Cache and summary names carry this, so frontal-only artifacts never collide with
# the mixed-view ones cached under the plain set name.
SET_TAG         = SET + ("_frontal" if FRONTAL_ONLY else "")

# --- members ---------------------------------------------------------------
# "run" -> its best.pt. ("run", 8000) -> that run's step-8000 checkpoint.
# A nested LIST is a sub-ensemble: averaged flat among itself, ONE vote/weight above.

FULL_MODELS = [
    "convnext_base_22k_1600x1312",
    "medmae_vitb_nih_B_768_s2",
    "rad_dino_vitB_768",
]

CKPT_SUBPATH = {}              # {run: "stage_subdir/best.pt"} for two-stage runs
USE_STAGE2 = False             # add the 5 per-disease Stage-2 models as ONE composite member
STAGE2_GROUP = {
    "Atelectasis":      "convnext_base_22k_final_stage2_atelectasis",
    "Cardiomegaly":     "convnext_base_22k_final_stage2_cardiomegaly",
    "Consolidation":    "convnext_base_22k_final_stage2_consolidation",
    "Edema":            "convnext_base_22k_final_stage2_edema",
    "Pleural Effusion": "convnext_base_22k_final_stage2_pleural_effusion",
}

# --- WEIGHTS = "search" only: the per-class CV weight search ---------------
SEARCH_MODE = "greedy"         # "greedy" (forward selection) | "grid" (exhaustive)
GRID_STEP   = 0.1              # weights in {0, STEP, ..., 1}, summing to 1
                               # greedy explores the SAME lattice, as 1/STEP rounds
N_FOLDS     = 5                # CV folds over the split
SEED        = 42               # KFold shuffle seed
TIE_EPS     = 1e-9             # AUROC ties within this break toward uniform weights
SEARCH_WORKERS = 4             # CPU threads for the search; 0 -> half this PC's logical CPUs
SEARCH_DEVICE  = "auto"        # where the grid search runs: "auto" | "cuda" | "cpu"
BEST_METRIC = "weighted_mean_auroc"   # what ranks a search run for the "(best <set>)"
                               # marker: the headline WEIGHTED mean AUROC.
                               # "oof_mean_auroc" to rank on the held-out number instead

# --- compute — ONLY used on a cache MISS; a fully cached run ignores all of it.
GPU        = "B300"            # T4|L4|A10G|A100|A100-80GB|H100|H200; None -> ref run's
BATCH_SIZE = 176               # None -> that run's dataloader.val_batch_size
CPU_CORES  = 16                # Modal container cores; None -> ref run's
MEMORY_GB  = 50                # Modal container RAM in GB; None -> ref run's
NUM_WORKERS_MODAL = 30         # DataLoader workers per member on Modal
NUM_WORKERS_LOCAL = 2          # DataLoader workers per member locally
NUM_WORKERS = NUM_WORKERS_LOCAL   # rebound below once RUN_ON="auto" resolves to MODE
PREFETCH_FACTOR = 4            # batches prefetched per worker (workers > 0 only)
# ===========================================================================

# ---------------------------------------------------------------------------
# REFERENCE — what the knobs above mean
#
# WEIGHTS — the ONE trigger deciding how the members are combined:
#   "flat"     equal 1/M average. Writes ensembling_results/<ts>/ensemble_<set>_summary.*
#   "search"   fits per-class weights RIGHT NOW: for each of the 5 tasks INDEPENDENTLY a
#              weight vector over the members (non-negative, summing to 1, on a GRID_STEP
#              grid) maximizing that class's AUROC, fit with N_FOLDS-fold CV (fit on
#              N_FOLDS-1, repeat, average the fold vectors) so ~234 images don't overfit
#              the free parameters. Reports the OOF AUROC (the honest generalization
#              estimate) next to the in-sample number and the flat 1/M baseline, and
#              SAVES the fitted weights inside ensembling_results/<ts>/
#              weighted_<set>_summary.json. A new fit is a NEW timestamped folder — it
#              never touches, moves or overwrites weights fitted earlier.
#   "<path>"   reuses weights from such a run. Any of these name it:
#                "best_valid200" / "best test500" / "best"  -> whatever folder currently
#                     carries that split's "(best <set>)" marker ("best" uses SET). Only
#                     folders that actually hold fitted weights qualify, and a shorthand
#                     must agree on the number, so "val200" finds "(best valid200)" but
#                     never "(best val)".
#                "2026-07-27_20-36-20 (best valid200)"  -> just the folder NAME; the
#                     results roots are searched for it, so console output pastes in.
#                "ensembling_results/2026-07-27_20-36-20" or an absolute path, or the
#                     summary json itself. A folder from the pre-merge
#                     weighted_ensemble/results/ works just as well.
#              Its "weights_per_class" ({task: {member: weight}}) drives the blend and
#              the result is written like a flat run (ensembling_results/<ts>/). Every
#              member's label needs a weight for every class; weights are renormalized
#              per class over the members actually present, so adding/dropping a member
#              stays well-defined. Weights apply ONLY to the final layer — a group gets
#              one weight, never one per run inside it. Resolved on the launcher and
#              passed in, so it works on Modal too.
#
# RUN_ON "auto": if EVERY member's probs are cached locally the ensemble is just a
#   CPU average over .npy files -> run here. A missing member needs its checkpoint AND
#   the images (both on the Modal volumes) -> run on Modal.
#
# FULL_MODELS / hierarchy. Each entry gets ONE vote (or ONE weight). Two orthogonal
#   things per entry:
#   WHICH CHECKPOINT — a leaf is a run name, or a (run, checkpoint) TUPLE:
#       "run_a"                 # that run's best.pt (CKPT_SUBPATH honored)
#       ("run_a", 8000)         # ckpt_step8000.pt  ("best"|"last"|<step>|"<file>.pt")
#     A tuple leaf is labeled "<run> @ step<N>", which never collides with the plain
#     "<run>" member — so several checkpoints of the SAME run can vote independently.
#     Resolved in that run's checkpoints dir exactly like `resume`.
#   HOW IT NESTS — a nested LIST is a sub-ensemble, averaged flat among itself FIRST
#     (equal weight, no per-run weights, by design), voting ONCE above:
#       ["run_b", "run_c"],                      # ONE member = mean(run_b, run_c)
#       [("run_a", 8000), ("run_a", 9000)],      # ONE member = mean of two checkpoints
#       ["run_d", ["run_e", "run_f"]],           # nests to any depth
#     LIST = group, TUPLE = one leaf's (run, checkpoint) — that is the whole rule.
#   A group may also be written {"members": [...], "space": "prob"|"logit"|"rank",
#   "name": "..."} when it needs a space or a label override, and a leaf {"run": ...,
#   "checkpoint": ...}. A group's label is built from the EXACT run names —
#   avg(run_b + run_c) — so nothing hides behind a nickname. A saved weights file keys
#   on these labels, so keep the member list identical when reusing one.
#
# COMBINE_SPACE: fixed, parameter-free per-member transform applied BEFORE the blend
#   (and before the search, so weights are fitted in the space they are used in).
#   AUROC/AUPRC are ranking metrics, so it can only help when members are calibrated
#   differently (e.g. ConvNeXt@1600 vs the two ViTs).
#     "prob"  : average probabilities (default).
#     "logit" : average log-odds, sigmoid back. F1/P/R/Spec still use the frozen
#               thresholds (sigmoid(mean logit) is a valid probability).
#     "rank"  : average per-class normalized ranks in (0,1], ties averaged. Best for
#               AUROC/AUPRC when scales differ, but the result is NOT a probability, so
#               the frozen thresholds don't transfer -> F1/P/R/Spec report n/a.
#
# THRESHOLDS_FROM: that run's results/thresholds.json supplies the per-class thresholds
#   for the F1/precision/recall/specificity (AUROC/AUPRC are threshold-free).
#   None -> the first member. Missing file/task -> 0.5.
#
# The member cache (cache_runs/<set>/<run>_<ckpt>.npy) is NAME-based: a present file is
#   a HIT reused as-is, needing no GPU and no checkpoint. To recompute one member (e.g.
#   after RETRAINING its checkpoint), DELETE that one .npy — the next run recomputes and
#   re-saves exactly it, leaving every other member cached.
#
# RUN_TAG: every run writes its own timestamped folder, so trying different ensembles
#   never overwrites the last. RUN_TAG is appended to make it self-describing:
#   RUN_TAG="5models" -> 2026-07-01_14-30-05_5models.
#
# The "(best <set>)" marker: after each run the winning folder for THAT split is renamed
#   with a " (best <set>)" suffix and the previous winner is demoted. valid200 / test500
#   / val keep INDEPENDENT winners. ONE marker per split covers every run in
#   ensembling_results/, flat and search alike: a flat run is ranked on its ensemble mean
#   AUROC, a search run on BEST_METRIC — the headline WEIGHTED mean AUROC, i.e. the same
#   number printed in the banner. Note that number is IN-SAMPLE for a search (the weights
#   were fitted on the images they are scored on), so it sits above what the same weights
#   would score on fresh data; BEST_METRIC="oof_mean_auroc" ranks on the held-out
#   estimate instead.
# ---------------------------------------------------------------------------


# --------------------------- WEIGHTS mode -----------------------------------
_W = (WEIGHTS or "flat")
_W = _W.strip() if isinstance(_W, str) else _W
if not isinstance(_W, str):
    raise SystemExit(f"WEIGHTS must be 'flat' | 'search' | '<path>', got {WEIGHTS!r}")
SEARCH_WEIGHTS = (_W.lower() == "search")          # fit fresh weights (and save them)
LOAD_WEIGHTS = None if _W.lower() in ("flat", "search", "") else _W   # reuse a saved fit
# The summary this run writes: weighted_<set>_summary.* for a search (it carries the
# fitted weights), ensemble_<set>_summary.* otherwise. Both land in ensembling_results/.
SUMMARY_PREFIX = "weighted" if SEARCH_WEIGHTS else "ensemble"
# Keys ranking the "(best <set>)" marker, per summary kind — BEST_METRIC first, then a
# fallback chain so a summary written before a metric existed still ranks.
BEST_KEYS_SEARCH = (BEST_METRIC, "weighted_mean_auroc", "oof_mean_auroc", "flat_mean_auroc")
BEST_KEYS_FLAT = ("ensemble_mean_auroc",)


# ---- member-tree helpers (a nested list in FULL_MODELS = a sub-ensemble) ----
def _is_group(node) -> bool:
    """A node is a GROUP when it's a nested LIST, or the explicit dict form
    {"members": [...]} used when a space or a name override is needed. A TUPLE is
    never a group — it is one leaf's (run, checkpoint) pair."""
    return isinstance(node, list) or (isinstance(node, dict) and "members" in node)


def _group_children(node):
    """The children of a group node (list form or dict form)."""
    kids = list(node) if isinstance(node, list) else list(node.get("members") or [])
    if not kids:
        raise ValueError(f"empty sub-ensemble: {node!r}")
    return kids


def _group_space(node) -> str:
    """Space a group averages its children in. A plain nested list -> "prob"."""
    return "prob" if isinstance(node, list) else str(node.get("space", "prob"))


def _leaf_spec(node):
    """(run, checkpoint) of a LEAF node. A leaf is exactly one run+checkpoint — i.e.
    one cached prob matrix. Accepted forms:
        "run"                            -> the run's default ckpt (best.pt/CKPT_SUBPATH)
        ("run", 8000)                    -> "best"|"last"|<step>|"<file>.pt"
        {"run": ..., "checkpoint": ...}  -> same, explicit form
    "best"/None are the same checkpoint, so both normalize to None and share one label
    and one cache entry."""
    if isinstance(node, str):
        return node, None
    if isinstance(node, tuple):
        if not 1 <= len(node) <= 2 or not isinstance(node[0], str):
            raise ValueError(f"bad (run, checkpoint) member entry: {node!r}")
        run, ckpt = node[0], (node[1] if len(node) == 2 else None)
    elif isinstance(node, dict) and "run" in node:
        run, ckpt = node["run"], node.get("checkpoint")
    else:
        raise ValueError(f"bad member entry: {node!r}")
    return run, (None if ckpt == "best" else ckpt)


def _node_leaves(node):
    """(run, checkpoint) for every LEAF under `node`, depth-first in config order."""
    if _is_group(node):
        for child in _group_children(node):
            yield from _node_leaves(child)
    else:
        yield _leaf_spec(node)


def _top_nodes():
    """Every FINAL-layer member, in blend order: the FULL_MODELS entries (a nested
    list among them is one sub-ensemble member). Each contributes ONE column to the
    final blend — and, in search mode, ONE weight per class."""
    return list(FULL_MODELS)


def _all_member_runs():
    """Distinct run names referenced anywhere in the tree (+ the Stage-2 composite),
    in first-seen order. Used to mount every data volume a member needs."""
    seen = []
    for node in _top_nodes():
        for run, _ in _node_leaves(node):
            if run not in seen:
                seen.append(run)
    if USE_STAGE2:
        for run in STAGE2_GROUP.values():
            if run not in seen:
                seen.append(run)
    return seen


# Reference run: the FIRST leaf of the first member. Supplies the canonical task order,
# the split CSV, the Modal settings — and the default THRESHOLDS_FROM.
_first_nodes = _top_nodes()
if not _first_nodes:
    raise SystemExit("no members: fill FULL_MODELS.")
REF_RUN = next(run for run, _ in _node_leaves(_first_nodes[0]))
if THRESHOLDS_FROM is None:
    THRESHOLDS_FROM = REF_RUN


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


def _stamp_name() -> str:
    """<timestamp> (+ RUN_TAG) — the per-run folder name, unique to this run."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_{RUN_TAG}" if RUN_TAG else stamp


def _out_subdir(base: Path) -> Path:
    """A fresh timestamped subfolder under base/ensembling_results/ (+ RUN_TAG)."""
    return Path(base) / "ensembling_results" / _stamp_name()


def _out_subdir_in(root: Path) -> Path:
    """A fresh timestamped subfolder directly under `root` (+ RUN_TAG)."""
    return Path(root) / _stamp_name()


def _results_root_local() -> Path:
    """Where EVERY run's folder goes locally — flat, reused-weights and search alike.
    One history, one "(best <set>)" marker per split. (weighted_ensemble/results/ still
    holds the runs made before the scripts were merged; nothing new is written there,
    and a WEIGHTS path pointing into it still works.)"""
    return Path(__file__).resolve().parent / "ensembling_results"


def _results_root_remote(runs_mount: Path) -> Path:
    """The same, on the runs volume."""
    return Path(runs_mount) / "ensembling_results"


def _dummy_loss(logits, targets):
    return logits.sum() * 0.0        # we only need probabilities, not the loss


def _fetch_results_from_modal(runs_volume: str, remote_sub: str, local_base: Path):
    """Pull the run's freshly written <timestamp> folder off the runs volume down to
    the local results root, so a Modal ensemble ends up on disk exactly like a local
    one. `remote_sub` is the volume-relative POSIX path the remote function returned
    (e.g. 'ensembling_results/2026-07-06_14-30-05'), and `local_base` is the LOCAL
    parent the <ts> folder should land in.

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
    dest_parent = Path(local_base)
    dest_parent.mkdir(parents=True, exist_ok=True)             # MUST exist as a dir (see above)
    final_dir = dest_parent / PurePosixPath(remote_posix).name
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
    """Pull /cache_runs/ back down to others/cache_runs/ AFTER a Modal run, so probs
    computed on the GPU are reusable locally (a later run then needs no GPU at all)."""
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

    # every LEAF of the member tree — a group needs each of its children cached, since
    # the group's own value is computed from them on the fly (never cached itself).
    for node in _top_nodes():
        for run, ckpt in _node_leaves(node):
            add(run, ckpt)
    if USE_STAGE2:
        for run in STAGE2_GROUP.values():
            add(run)
    return names, undecidable


def _members_all_cached(local_base: Path, set_name: str) -> bool:
    """True iff EVERY member's probs (and the labels) are already cached locally for
    `set_name` — so the Modal VM can run CPU-only (no GPU). Name-based existence check
    (a present .npy is reused as-is). Any undecidable member ("last") -> False."""
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
# (e.g. "2026-07-15_17-09-13 (best test500)"). If the new run beats the current winner
# for THAT set, the old winner is demoted (suffix stripped) and the new folder promoted.
# valid200 and test500 keep INDEPENDENT winners, distinguished by the set named in the
# suffix. Every run lives in ensembling_results/, so ONE marker per split ranks flat and
# search runs together — each on its honest number (see _folder_score).
import re as _re                                          # noqa: E402
# matches a trailing "(best)" or "(best <set>)" marker on a folder name. <set> is any
# split token (valid200, test500, val, ...) so the tracker works for every SET.
_BEST_RE = _re.compile(r"\s*\(best(?:\s+(?P<set>\w+))?\)\s*$")


def _strip_best(name: str) -> str:
    """Folder name without any trailing '(best)'/'(best <set>)' marker."""
    return _BEST_RE.sub("", name).rstrip()


def _folder_score(folder: Path, set_name: str):
    """(score, metric name) for `set_name` from whichever summary the folder holds, or
    (None, None). Flat/reused-weights runs rank on their ensemble mean AUROC, search runs
    on BEST_METRIC (the headline WEIGHTED mean AUROC). The key list is tried in order, so
    a summary written before a metric existed still ranks instead of 'no score'."""
    import json
    for prefix, keys in (("ensemble", BEST_KEYS_FLAT), ("weighted", BEST_KEYS_SEARCH)):
        j = Path(folder) / f"{prefix}_{set_name}_summary.json"
        if not j.exists():
            continue
        try:
            doc = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in keys:
            v = doc.get(key)
            if isinstance(v, (int, float)):
                # a reused-weights run has no honest number of its own: its score was
                # produced by weights someone fitted earlier (on this very split, if the
                # fit came from here) — say so rather than pass it off as clean.
                if prefix == "ensemble" and doc.get("class_weights"):
                    key += " (reused weights)"
                return float(v), key
    return None, None


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
        summs = sorted(d.glob("ensemble_*_summary.json")) + \
            sorted(d.glob("weighted_*_summary.json"))
        for summ in summs:                                     # infer the set
            s = summ.name.split("_", 1)[1][:-len("_summary.json")]
            print(f"[best] migrating legacy marker: '{d.name}' -> '{base} (best {s})'")
            _rename_dir(d, f"{base} (best {s})")
            break


def _update_best_tracker(set_name: str, new_dir: Path):
    """Promote new_dir to the '(best <set_name>)' winner IFF its score beats the current
    winner for THIS set. ONE marker per split over the whole ensembling_results/ folder,
    flat and search runs alike (see _folder_score for what each is ranked on); also
    heals any legacy bare '(best)' folder into the set-qualified form first."""
    new_dir = Path(new_dir)
    root = new_dir.parent
    if not root.exists():
        return
    _migrate_legacy_best(root)

    new_score, metric = _folder_score(new_dir, set_name)
    if new_score is None:
        print(f"[best] {set_name}: no *_{set_name}_summary.json in '{new_dir.name}' "
              f"— best tracker skipped")
        return

    # current winner for THIS set = a folder whose name ends in "(best <set_name>)"
    cur_best, cur_score, cur_metric = None, None, None
    for d in root.iterdir():
        if not d.is_dir() or d.resolve() == new_dir.resolve():
            continue
        m = _BEST_RE.search(d.name)
        if m and m.group("set") == set_name:
            a, k = _folder_score(d, set_name)
            if a is not None and (cur_score is None or a > cur_score):
                cur_best, cur_score, cur_metric = d, a, k

    # No marker yet for this set (first run after adopting the tracker, or the marked
    # folder was deleted/renamed by hand): SEED from every folder that has a summary
    # for this set, so the marker lands on the genuinely best run rather than simply on
    # whichever run happened to come first.
    if cur_best is None:
        for d in root.iterdir():
            if not d.is_dir() or d.resolve() == new_dir.resolve():
                continue
            a, k = _folder_score(d, set_name)
            if a is not None and (cur_score is None or a > cur_score):
                cur_best, cur_score, cur_metric = d, a, k
        if cur_best is not None:
            print(f"[best] {set_name}: no marker found — seeding across existing folders "
                  f"(leader '{cur_best.name}' {cur_metric}={cur_score:.4f})")
            if cur_score >= new_score:      # an OLD folder is the true best -> mark it
                p = _rename_dir(cur_best, f"{_strip_best(cur_best.name)} (best {set_name})")
                print(f"[best] {set_name}: kept '{p.name}' ({cur_metric}={cur_score:.4f} "
                      f">= new {metric}={new_score:.4f})")
                return

    if cur_best is None:
        p = _rename_dir(new_dir, f"{_strip_best(new_dir.name)} (best {set_name})")
        print(f"[best] {set_name}: no previous best -> promoted '{p.name}' "
              f"({metric}={new_score:.4f})")
    elif new_score > cur_score:
        _rename_dir(cur_best, _strip_best(cur_best.name))          # demote old winner
        p = _rename_dir(new_dir, f"{_strip_best(new_dir.name)} (best {set_name})")
        print(f"[best] {set_name}: NEW BEST {metric}={new_score:.4f} > "
              f"{cur_metric}={cur_score:.4f} -> '{p.name}'  (demoted '{cur_best.name}')")
    else:
        print(f"[best] {set_name}: kept '{cur_best.name}' ({cur_metric}={cur_score:.4f} "
              f">= new {metric}={new_score:.4f})")


# ------------------------ per-member prediction cache ----------------------
# A member's (N, 5) PROBABILITY matrix depends only on (run cfg, checkpoint, SET) —
# the run's cfg (geometry/CLAHE/u-policy) is fixed, so the cache key is just
# cache_runs/<set>/<run>_<ckpt>.npy. This lets a re-run reuse a member's probs
# instead of recomputing on the GPU. The cache is NAME-based: a present .npy is a hit,
# regardless of environment (a probe built on Modal is reused as-is when run locally,
# and vice-versa). On a HIT the checkpoint is never resolved/loaded/stat'd at all — so
# a fully-cached run needs neither the GPU nor even the .pt files present. To recompute
# a member (e.g. after RETRAINING its checkpoint), DELETE its .npy.
def _cache_npy(cache_dir: Path, set_name: str, cache_name: str) -> Path:
    return Path(cache_dir) / set_name / f"{cache_name}.npy"


def _predict_member(cfg, ckpt_resolver, df, device, cache_dir, set_name, cache_name):
    """_predict with a NAME-based on-disk cache. `ckpt_resolver` is a 0-arg callable
    returning the checkpoint path — it is invoked ONLY on a cache miss, so a hit never
    touches (or requires) the checkpoint file. Delete a .npy to force its recompute."""
    npy = _cache_npy(cache_dir, set_name, cache_name)
    if npy.exists():
        arr = np.load(npy)
        print(f"[cache] HIT  {set_name}/{cache_name}  {arr.shape}  (no compute)")
        return arr
    print(f"[cache] MISS {set_name}/{cache_name} -> computing on {device.type.upper()}", flush=True)
    ckpt_path = ckpt_resolver()                       # resolve/require the .pt only now
    p = _predict(cfg, ckpt_path, df, device, desc=cache_name)
    try:
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, p)
        print(f"[cache] save {set_name}/{cache_name}")
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


def _node_label(node) -> str:
    """Label of a member-tree node — what the blend, the summary and any weights file
    key on. A group's label is derived from the EXACT run names of its children:
        avg(medmae_vitb_nih_B_768_s2_seed1337 + medmae_vitb_nih_B_448_s1_seed1337)
    so nothing hides behind a nickname and nested groups nest visibly. A non-prob
    group space is named too (avg-logit(...)). An explicit "name" wins if given.
    A leaf on its run's default checkpoint is just "<run>"; a leaf pinned to a specific
    checkpoint is "<run> @ step<N>", so the two never collide."""
    if _is_group(node):
        if isinstance(node, dict) and node.get("name"):
            return str(node["name"])
        space = _group_space(node)
        tag = "avg" if space == "prob" else f"avg-{space}"
        return f"{tag}(" + " + ".join(_node_label(c) for c in _group_children(node)) + ")"
    run, ckpt = _leaf_spec(node)
    return run if ckpt is None else _ckpt_member_label(run, ckpt)


def _node_probs(node, load_cfg, ckpt_path_of, df, device, cache_dir, depth: int = 0):
    """(N, T) probabilities for one member-tree node.

    A LEAF loads (or computes + caches) that run+checkpoint's probs. A GROUP resolves
    each child, then averages them FLAT — equal weight, no per-child weights, by
    design — in the group's `space`, and maps the result back to a [0,1] score. So a
    group is indistinguishable from a plain member to everything above it, and only
    leaves ever touch the cache."""
    pad = "   " * depth
    if _is_group(node):
        print(f"{pad}[group] {_node_label(node)}", flush=True)
        subs = [_node_probs(c, load_cfg, ckpt_path_of, df, device, cache_dir, depth + 1)
                for c in _group_children(node)]
        space = _group_space(node)
        stack = _to_space(np.stack(subs, axis=0), space)
        avg = _from_space(stack.mean(axis=0), space)
        print(f"{pad}[group] -> averaged {len(subs)} member(s) in {space} space", flush=True)
        return avg
    run, ckpt = _leaf_spec(node)
    cfg = load_cfg(run)
    resolver = (lambda run=run, ckpt=ckpt: ckpt_path_of(run, ckpt))
    stem = _member_ckpt_stem(run, ckpt) or Path(resolver()).stem   # resolve only if "last"
    if depth:
        print(f"{pad}[leaf ] {_node_label(node)}", flush=True)
    return np.asarray(_predict_member(cfg, resolver, df, device, cache_dir, SET_TAG,
                                      f"{run}_{stem}"))


def _gather(load_cfg, ckpt_path_of, df, device, cache_dir, tasks):
    """Every FINAL-layer member's (N, T) probability matrix, in blend order.

    One member per FULL_MODELS entry (a nested list resolves its children and averages
    them flat first, so from here on a group is just one more column), plus the Stage-2
    composite when USE_STAGE2. Returns (members {label: probs}, groups {label: {...}})
    — the single source of member identity for both the flat blend and the search."""
    members, group_tree = {}, {}
    nodes = _top_nodes()
    n_total = len(nodes) + (1 if USE_STAGE2 else 0)
    for i, node in enumerate(nodes, 1):
        label = _node_label(node)
        print(f"[member {i}/{n_total}] {label}", flush=True)
        if label in members:
            raise ValueError(f"duplicate member label {label!r} — labels must be unique")
        members[label] = _node_probs(node, load_cfg, ckpt_path_of, df, device, cache_dir)
        if _is_group(node):
            group_tree[label] = {"space": _group_space(node),
                                 "members": [_node_label(c) for c in _group_children(node)]}

    # --- Stage-2 composite (one member, per-class dedicated model) ---
    if USE_STAGE2:
        comp = np.zeros((len(df), len(tasks)), dtype=float)
        for disease, run in STAGE2_GROUP.items():
            cfg = load_cfg(run)
            print(f"[member {n_total}/{n_total}] stage2/{disease}: {run}", flush=True)
            cname = f"{run}_{_member_ckpt_stem(run)}"
            p = _predict_member(cfg, (lambda run=run: ckpt_path_of(run)),
                                df, device, cache_dir, SET_TAG, cname)   # (N,1)
            comp[:, tasks.index(disease)] = p[:, 0]
        members["final_stage2 (5 per-disease composite)"] = comp
    return members, group_tree


def _load_y_true(load_cfg, df, device, cache_dir):
    """The split's (N, T) labels. FIXED per SET and model-independent, so they are
    cached beside the members (_y_true.npy) — an all-cached run then needs no model
    build at all (truly CPU-only)."""
    npy = Path(cache_dir) / SET_TAG / "_y_true.npy"
    if npy.exists():
        y = np.load(npy)
        print(f"[cache] HIT  {SET_TAG}/_y_true  {y.shape} (labels)")
        return y
    rcfg = load_cfg(REF_RUN)
    yt, _, _, _ = sc._predict_dataframe(
        rcfg, build_model_generic(rcfg).to(device).eval(),
        df, device, _dummy_loss, amp=False, channels_last=False,
        batch_size=8, num_workers=0)
    arr = yt.detach().cpu().numpy() if hasattr(yt, "detach") else np.asarray(yt)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, arr)
    print(f"[cache] save {SET_TAG}/_y_true  {arr.shape} (labels)")
    return arr


def _split_df(load_cfg):
    """(ref cfg, tasks, dataframe) for the scored split, from the reference run.

    With FRONTAL_ONLY the lateral rows are dropped here, which is the single place
    every mode (flat, search, reused weights) gets its dataframe from. The view is
    read off the filename because test500 carries no Frontal/Lateral column."""
    import pandas as pd
    ref = load_cfg(REF_RUN)
    tasks = list(ref["tasks"])
    data_dir = Path(ref["paths"]["data_dir"])
    df = pd.read_csv(data_dir / ref["paths"][f"{SET}_csv"])
    if FRONTAL_ONLY:
        n_all = len(df)
        df = df[df["Path"].str.contains("frontal", case=False)].reset_index(drop=True)
        print(f"[split] {SET}: {len(df)} frontal rows of {n_all}")
    return ref, tasks, df


def run_ensemble(load_cfg, ckpt_path_of, out_dir: Path, thr_map: dict = None,
                 thr_source: str = "", class_weights: dict = None,
                 weights_source: str = ""):
    """FLAT (or reused-weights) ensemble. `load_cfg(run)` -> that run's cfg (data paths
    already correct for this environment); `ckpt_path_of(run, checkpoint=None)` -> a
    checkpoint path (None -> that run's best.pt/CKPT_SUBPATH; else a specific
    best|last|<step>|file); `thr_map` -> {task: threshold} per-class thresholds for
    F1/precision/recall/specificity (AUROC/AUPRC are threshold-free); `thr_source` -> a
    human label of where they came from (for the summary/logs); `class_weights` ->
    {task: {member: weight}} from a saved search (None -> flat 1/M). Thresholds and
    weights are resolved by the LAUNCHER from the local repo and passed in, so they
    always load regardless of what happens to be on the runs volume."""
    import json, math
    from datetime import datetime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref, tasks, df = _split_df(load_cfg)
    print(f"[ensemble] {SET}: {len(df)} images  |  tasks={tasks}  device={device}")
    # per-member prob cache, shared with every mode: others/cache_runs (or /runs/cache_runs)
    cache_dir = _cache_dir_of(out_dir)
    print(f"[cache] dir: {cache_dir / SET_TAG}")

    # per-class thresholds for F1/P/R/Spec (AUROC/AUPRC ignore them). Resolved by the
    # launcher from the local repo and passed in as a dict -> never depends on the vol.
    thr_map = thr_map or {}
    if thr_map:
        print(f"[ensemble] thresholds from {thr_source or '(provided)'}")
    else:
        print(f"[ensemble] WARNING: no thresholds ({thr_source or 'none'}) -> F1/P/R/Spec at 0.5")
    thr_vec = [float(thr_map.get(t, 0.5)) for t in tasks]

    y_true = _load_y_true(load_cfg, df, device, cache_dir)
    members, _group_tree = _gather(load_cfg, ckpt_path_of, df, device, cache_dir, tasks)

    # --- combine members + metrics (F1/P/R/Spec at the calibrated per-class thresholds).
    # Members are combined in COMBINE_SPACE (prob | logit | rank): a fixed per-member
    # transform, blended (flat or per-class weighted from a saved search), then mapped
    # back to a [0,1] decision score. ---
    _labels_order = list(members.keys())
    stack = _to_space(np.stack([members[l] for l in _labels_order], axis=0), COMBINE_SPACE)
    if class_weights:
        ens_t, _used_weights = _weighted_blend(stack, _labels_order, tasks, class_weights)
        _blend = "weighted"
        print(f"[ensemble] WEIGHTED per-class blend from {weights_source}")
    else:
        ens_t = stack.mean(axis=0)
        _used_weights = None
        _blend = "prob-average"
    if COMBINE_SPACE != "prob":
        print(f"[ensemble] combine space: {COMBINE_SPACE}")
    ens = _from_space(ens_t, COMBINE_SPACE)
    ens_metrics = sc.compute_metrics(y_true, ens, tasks, threshold=thr_vec)
    if COMBINE_SPACE == "rank":                 # thresholds don't transfer to the rank scale
        _null_threshold_metrics(ens_metrics, tasks)

    # each member's own mean AUROC (for reference; AUROC is threshold-free)
    per_member = {lab: sc.compute_metrics(y_true, p, tasks)["macro"]["mean_auroc"]
                  for lab, p in members.items()}

    _pc_keys = ("auroc", "auprc", "f1", "precision", "recall", "specificity")
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "set": SET, "n_images": int(len(df)),
        "members": list(members.keys()),
        # sub-ensembles: which children each group averaged, and in what space. Empty
        # when the ensemble is flat (no nested lists).
        "groups": _group_tree,
        "blend": _blend,
        "combine_space": COMBINE_SPACE,
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

    def _f4(x):                                 # None (rank mode) / NaN -> "n/a"
        return f"{x:.4f}" if isinstance(x, float) and math.isfinite(x) else " n/a"

    mac = ens_metrics["macro"]
    _blend_lbl = _blend if COMBINE_SPACE == "prob" else f"{_blend}, {COMBINE_SPACE}-space"
    lines = ["=" * 78,
             f"ENSEMBLE ({_blend_lbl})  —  set={SET_TAG}  images={len(df)}",
             f"generated: {summary['generated']}",
             f"thresholds (F1/P/R/Spec): {thr_source or '(none -> 0.5)'}"
             + ("   [n/a in rank space]" if COMBINE_SPACE == "rank" else "")]
    if _used_weights is not None:
        lines.append(f"weights: {weights_source}")
    lines += ["=" * 78,
              f"  ENSEMBLE mean AUROC={_f4(mac['mean_auroc'])}  AUPRC={_f4(mac['mean_auprc'])}  "
              f"F1={_f4(mac['mean_f1'])}  P={_f4(mac['mean_precision'])}  "
              f"R={_f4(mac['mean_recall'])}  Spec={_f4(mac['mean_specificity'])}",
              "  per-class:"]
    for t in tasks:
        pc = summary["ensemble_per_class"][t]
        lines.append(f"    {t:<18} AUROC={_f4(pc['auroc'])}  AUPRC={_f4(pc['auprc'])}  "
                     f"F1={_f4(pc['f1'])}  P={_f4(pc['precision'])}  R={_f4(pc['recall'])}  "
                     f"Spec={_f4(pc['specificity'])}  (thr={summary['thresholds'][t]:.4f})")
    if _used_weights is not None:
        lines += ["  " + "-" * 74,
                  "  per-class weights (renormalized over members; '·' = unused):"]
        lines += _weights_table(_labels_order, tasks,
                                [[_used_weights[t][lab] for t in tasks]
                                 for lab in _labels_order], indent="    ")
    lines += ["  " + "-" * 74, "  members (own mean AUROC):"]
    for lab, a in per_member.items():
        lines.append(f"    {a:.4f}   {lab}")
    if _group_tree:
        lines += ["  " + "-" * 74,
                  "  sub-ensembles (averaged FLAT among themselves, then one member above):"]
        for lab, g in _group_tree.items():
            lines.append(f"    {lab}")
            for child in g["members"]:
                lines.append(f"        + {child}")
            lines.append(f"      averaged in {g['space']} space, equal weight "
                         f"(1/{len(g['members'])} each)")
    lines += ["=" * 78]
    txt = "\n".join(lines) + "\n"

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"ensemble_{SET_TAG}_summary.json"
    txt_path  = out_dir / f"ensemble_{SET_TAG}_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote -> {json_path}")
    print(f"wrote -> {txt_path}")

    # Final banner: restate the ONE number that matters so it is impossible to miss at
    # the bottom of a long console dump.
    _big = f"{'WEIGHTED' if _used_weights else 'ENSEMBLE'} mean AUROC   {_f4(mac['mean_auroc'])}"
    _pad = 74
    print("\n" * 2 + "╔" + "═" * _pad + "╗")
    print("║" + f"{SET_TAG}  ·  {len(df)} images  ·  {len(members)} members".center(_pad) + "║")
    print("║" + " " * _pad + "║")
    print("║" + _big.center(_pad) + "║")
    print("║" + " " * _pad + "║")
    print("║" + f"blend: {_blend_lbl}".center(_pad) + "║")
    print("╚" + "═" * _pad + "╝\n")


def _cache_dir_of(out_dir: Path) -> Path:
    """The shared per-member prob cache for a run writing into `out_dir`. Local runs
    put it at others/cache_runs; on Modal it is /runs/cache_runs. out_dir is always
    <root>/<results dir>/<ts>, so the cache sits beside <results dir>'s parent."""
    p = Path(out_dir)
    for anc in p.parents:                       # nearest ancestor first
        if anc.name in ("others", "runs") or (anc / "cache_runs").is_dir():
            return anc / "cache_runs"
    return p.parent.parent / "cache_runs"       # fallback: beside the results root


def _to_space(stack, space):
    """Transform each member's (M, N, T) probability stack into the combine space.
    prob -> identity; logit -> log-odds (clipped); rank -> per-member, per-class
    normalized ranks in (0,1] (ties averaged). The transform is per member & per class,
    so it never mixes members or classes."""
    if space == "prob":
        return stack
    if space == "logit":
        eps = 1e-6
        p = np.clip(stack, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))
    if space == "rank":
        from scipy.stats import rankdata
        M, N, T = stack.shape
        out = np.empty_like(stack, dtype=float)
        for m in range(M):
            for c in range(T):
                out[m, :, c] = rankdata(stack[m, :, c], method="average") / N
        return out
    raise ValueError(f"COMBINE_SPACE must be 'prob' | 'logit' | 'rank', got {space!r}")


def _from_space(ens, space):
    """Map the blended score back to a [0,1] decision score for the thresholded metrics.
    logit -> sigmoid (a valid probability); prob/rank are already in [0,1] (rank is a
    normalized rank, NOT a probability — the caller nulls F1/P/R/Spec in rank mode)."""
    if space == "logit":
        return 1.0 / (1.0 + np.exp(-ens))
    return ens


def _null_threshold_metrics(metrics, tasks):
    """In rank mode the blended score isn't on the probability scale, so the frozen
    prob-thresholds don't transfer — void the threshold-dependent metrics (AUROC/AUPRC
    stay, being threshold-free)."""
    for t in tasks:
        for k in ("f1", "precision", "recall", "specificity"):
            metrics["per_task"][t][k] = None
    for k in ("mean_f1", "mean_precision", "mean_recall", "mean_specificity"):
        metrics["macro"][k] = None


def _same_split(a: str, b: str) -> bool:
    """Do two split tokens name the same split, allowing an abbreviated word?
    "val200" == "valid200" (same number, one word abbreviates the other), but
    "val200" != "val" and "valid200" != "test500" — the number has to agree, so a
    shorthand can never silently select a DIFFERENT split."""
    pa = _re.fullmatch(r"([a-z]*)(\d*)", a.lower())
    pb = _re.fullmatch(r"([a-z]*)(\d*)", b.lower())
    if not pa or not pb or pa.group(2) != pb.group(2):
        return False
    wa, wb = pa.group(1), pb.group(1)
    return bool(wa) and bool(wb) and (wa.startswith(wb) or wb.startswith(wa))


def _resolve_best_alias(spec):
    """A WEIGHTS value of "best", "best_valid200", "best test500" (any separator) ->
    the folder currently carrying that split's "(best <set>)" marker, so you can point
    at the reigning fit without pasting a timestamp. The set defaults to SET when only
    "best" is given, and a shorthand ("val200") matches the marker's set by prefix.
    Only folders that actually HOLD fitted weights qualify — a flat run can hold the
    marker, and it has no weights to reuse. Returns a Path, or None when `spec` isn't
    one of these aliases."""
    m = _re.fullmatch(r"best[\s_-]*(?P<set>\w*)", str(spec).strip(), _re.IGNORECASE)
    if not m:
        return None
    want = (m.group("set") or SET).lower()
    here = Path(__file__).resolve().parent
    roots = [here / "ensembling_results", here / "weighted_ensemble" / "results"]

    marked = []                       # (set token, folder) for every "(best <set>)" dir
    for root in roots:
        if not root.exists():
            continue
        for d in root.iterdir():
            mm = _BEST_RE.search(d.name) if d.is_dir() else None
            if mm and mm.group("set"):
                marked.append((mm.group("set").lower(), d))
    if not marked:
        raise FileNotFoundError(
            f"WEIGHTS={spec!r}: no folder carries a '(best <set>)' marker yet in "
            + " or ".join(str(r) for r in roots))

    hits = [d for s, d in marked if s == want] \
        or [d for s, d in marked if _same_split(s, want)]
    if not hits:
        raise FileNotFoundError(
            f"WEIGHTS={spec!r}: no '(best {want})' folder. Marked splits: "
            + ", ".join(sorted({s for s, _ in marked})))

    withw = [d for d in hits if any(d.glob("weighted_*_summary.json"))]
    if not withw:
        raise FileNotFoundError(
            f"WEIGHTS={spec!r}: '{hits[0].name}' holds the marker but no fitted weights "
            f"(it is a flat or reused-weights run). Point WEIGHTS at a SEARCH run's "
            f"folder, or set WEIGHTS='search' to fit fresh ones.")
    pick = max(withw, key=lambda d: _folder_score(d, SET)[0] or float("-inf"))
    print(f"[weights] {spec!r} -> {pick.name}")
    return pick


def _load_class_weights(spec):
    """Load per-class member weights SAVED by an earlier WEIGHTS="search" run (its json
    file, or a folder containing weighted_*_summary.json). Runs on the LAUNCHER; the
    resolved dict is passed into run_ensemble (local) / handed to the remote (modal), so
    it loads regardless of the runs volume. Returns (weights_per_class, source_label),
    or (None, "") when spec is falsy. weights_per_class = {task: {member_label: weight}}.

    `spec` may be:
      "best" / "best_valid200" / "best test500"  -> whatever folder currently carries
                                                    that split's "(best <set>)" marker
      an absolute path, or one relative to others/ or the CWD
      just the NAME of a run folder — the results roots are searched for it, so pasting
      "2026-07-28_00-41-58 (best valid200)" straight off the console works."""
    if not spec:
        return None, ""
    import json
    here = Path(__file__).resolve().parent
    alias = _resolve_best_alias(spec)
    if alias is not None:
        spec = str(alias)
    tried = [Path(spec)] if Path(spec).is_absolute() else [
        here / spec,                                  # others/<spec>
        here / "ensembling_results" / spec,           # the results root
        here / "weighted_ensemble" / "results" / spec,  # pre-merge search runs
        Path(spec),                                   # CWD-relative
    ]
    p = next((c for c in tried if c.exists()), None)
    if p is None:
        raise FileNotFoundError(
            f"WEIGHTS={spec!r}: no such file or folder. Looked in:\n  "
            + "\n  ".join(str(c) for c in tried))
    if p.is_dir():
        hits = sorted(p.glob("weighted_*_summary.json"))
        if not hits:
            raise FileNotFoundError(
                f"{p} holds no weighted_*_summary.json — only a WEIGHTS='search' run "
                f"saves fitted weights (a flat run writes ensemble_*_summary.json).")
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


# ==================== WEIGHTS = "search": grid + CV search =================
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


def _auc_rows(scores, y_col):
    """AUROC of EVERY row of `scores` (G, n) against the binary labels y_col (n,), in
    one shot. Uses the rank identity AUC = (sum of the positives' ranks - npos(npos+1)/2)
    / (npos*nneg) — identical to roc_auc_score, but computed for all G candidate blends
    at once instead of G separate Python calls.

    The ranks come from one argsort rather than scipy.stats.rankdata: rankdata's generic
    array-API path allocates several int64 temporaries the size of the whole block, which
    is both slow and the thing that blew up memory when several threads ran at once. That
    shortcut assumes DISTINCT scores, which is not guaranteed (a saturated member pins
    several images to exactly 1.0), so rows that actually contain a tie — normally none —
    are recomputed with averaged ranks. A single-class y_col -> all-NaN (AUROC undefined).
    """
    y = np.asarray(y_col).astype(bool)
    npos = int(y.sum())
    nneg = int(y.size - npos)
    if npos == 0 or nneg == 0:
        return np.full(scores.shape[0], np.nan)
    n = scores.shape[1]
    half = npos * (npos + 1) / 2.0
    denom = float(npos * nneg)

    order = np.argsort(scores, axis=1, kind="stable")       # (G, n)
    ranks = np.arange(1, n + 1, dtype=np.int64)
    auc = ((y[order] * ranks).sum(axis=1) - half) / denom

    srt = np.take_along_axis(scores, order, axis=1)
    tied = (srt[:, 1:] == srt[:, :-1]).any(axis=1)          # rows needing averaged ranks
    if tied.any():
        from scipy.stats import rankdata
        r = rankdata(scores[tied], method="average", axis=1)
        auc[tied] = (r[:, y].sum(axis=1) - half) / denom
    return auc


# Cap on the (grid rows x images) block held at once, so a big grid streams in chunks
# instead of allocating one huge array. The peak is a few arrays of this many cells PER
# SEARCH THREAD; ~4e6 keeps the whole search inside a few hundred MB.
_SEARCH_CHUNK_CELLS = 4_000_000


def _pick_best(auc, dist2_chunk, gi0):
    """(best auc, its tie-break distance, its GLOBAL grid index) for one chunk of
    candidates: the max AUROC, ties within TIE_EPS broken toward the most-uniform
    vector. NaN everywhere (single-class fold) -> (nan, inf, -1)."""
    if not np.any(np.isfinite(auc)):
        return float("nan"), float("inf"), -1
    best = float(np.nanmax(auc))
    cand = np.flatnonzero(auc >= best - TIE_EPS)
    j = int(cand[int(np.argmin(dist2_chunk[cand]))])
    return best, float(dist2_chunk[j]), gi0 + j


def _fit_chunk(Gc, gi0, P_all, y, fold_masks, fold_stats, dist2_chunk):
    """Fit ONE chunk of grid vectors for ONE class, across ALL folds at once.

    The folds are nested subsets of the same images, and a subset's internal ordering is
    just the full ordering filtered — so the expensive sort is done ONCE over all N
    images and every fold reuses it, instead of each fold sorting its own 4/5 of the
    rows. A row's rank WITHIN a fold is then the running count of that fold's members up
    to it (a cumsum), which is far cheaper than another argsort.

    Returns [(best_auc, best_dist, best_global_index)] per fold."""
    S = Gc @ P_all                                     # (g, N) every candidate blended
    order = np.argsort(S, axis=1)                      # unstable: fine, ties handled below
    yo = y[order]
    srt = np.take_along_axis(S, order, axis=1)
    tied = (srt[:, 1:] == srt[:, :-1]).any(axis=1)     # rows needing averaged ranks
    del srt

    out = []
    for f, tm in enumerate(fold_masks):
        npos, nneg = fold_stats[f]
        if npos == 0 or nneg == 0:
            out.append((float("nan"), float("inf"), -1))
            continue
        half, denom = npos * (npos + 1) / 2.0, float(npos * nneg)
        tmo = tm[order]                                # fold membership in sorted order
        rank = np.cumsum(tmo, axis=1, dtype=np.int32)  # rank within the fold
        auc = (np.einsum("gn,gn->g", (yo & tmo), rank, dtype=np.int64) - half) / denom
        _fix_tied_rows(auc, tied, S, y, tm, half, denom)
        out.append(_pick_best(auc, dist2_chunk, gi0))
    return out


def _fix_tied_rows(auc, tied, S, y, tm, half, denom):
    """Recompute the AUROC of `tied` rows with averaged ranks (the exact definition).
    In place; a no-op when nothing tied, which is the normal case."""
    if not tied.any():
        return
    from scipy.stats import rankdata
    idx = np.flatnonzero(tm)
    r = rankdata(np.asarray(S)[np.ix_(tied, idx)], method="average", axis=1)
    auc[tied] = (r[:, y[idx]].sum(axis=1) - half) / denom


# Two blended scores closer than this in float32 are treated as an ordering the GPU
# cannot be trusted to get right, and those rows are redone exactly on the CPU.
_FP32_TIE_ATOL = 1e-6


def _fit_chunk_gpu(Gc, gi0, P_all, y, fold_masks, fold_stats, dist2_chunk, dev):
    """_fit_chunk on the GPU. Same algorithm — one sort shared by the folds, ranks by
    cumsum — but the (grid x images) blocks are exactly the dense, regular work a GPU
    eats for breakfast.

    On EXACTNESS: the blend runs in float32 (the 4060 is ~1/64 rate at float64), but the
    AUROC numerator is an INTEGER count of ranks, so it carries no rounding error — the
    only thing float32 can get wrong is the ORDER of two nearly-equal scores. Rows with
    any adjacent gap under _FP32_TIE_ATOL are therefore recomputed on the CPU in float64
    with averaged ranks, and the winner is picked from the same float64 AUROCs the CPU
    path uses. So the result matches the CPU path, not merely approximates it."""
    g = torch.as_tensor(Gc, dtype=torch.float32, device=dev)
    p = torch.as_tensor(P_all, dtype=torch.float32, device=dev)
    S = g @ p                                          # (rows, N)
    order = torch.argsort(S, dim=1)
    srt = torch.gather(S, 1, order)
    near = ((srt[:, 1:] - srt[:, :-1]).abs() <= _FP32_TIE_ATOL).any(dim=1)
    del srt, S, g
    yo = torch.as_tensor(np.ascontiguousarray(y), device=dev)[order]
    near_np = near.cpu().numpy()
    S_cpu = (Gc @ P_all) if near_np.any() else None    # float64, only if needed

    out = []
    for f, tm in enumerate(fold_masks):
        npos, nneg = fold_stats[f]
        if npos == 0 or nneg == 0:
            out.append((float("nan"), float("inf"), -1))
            continue
        half, denom = npos * (npos + 1) / 2.0, float(npos * nneg)
        tmo = torch.as_tensor(np.ascontiguousarray(tm), device=dev)[order]
        rank = torch.cumsum(tmo.to(torch.int32), dim=1)
        num = ((yo & tmo) * rank).sum(dim=1, dtype=torch.int64).cpu().numpy()
        auc = (num - half) / denom
        if near_np.any():
            _fix_tied_rows(auc, near_np, S_cpu, y, tm, half, denom)
        out.append(_pick_best(auc, dist2_chunk, gi0))
    return out


def _greedy_fold(P_all, y, mask, rounds):
    """Forward selection (Caruana) of a weight vector for ONE class on ONE fold.

    Start from nothing and, `rounds` times, add whichever member most improves the
    running sum's AUROC — the same member may be picked again, which is how a member
    earns more weight. Dividing the final counts by `rounds` gives weights in
    {0, 1/rounds, ..., 1} summing to 1: EXACTLY the lattice the exhaustive grid
    enumerates at GRID_STEP = 1/rounds. The difference is only how it is explored —
    rounds*M candidate blends instead of C(rounds+M-1, M-1). It is a hill climb, so a
    local optimum is possible; on a few hundred images that gap is inside the noise,
    and it is the only tractable option once M grows.

    Ties (within TIE_EPS) go to the member with the LOWEST current count, which pulls
    toward a uniform vector — the same tie-break rule the grid search uses.
    Returns (weights (M,), auc of the final blend)."""
    P = np.ascontiguousarray(P_all[:, mask])            # (M, n) this fold's rows
    yf = y[mask]
    M, n = P.shape
    counts = np.zeros(M, dtype=np.int64)
    S = np.zeros(n, dtype=np.float64)
    last = np.nan
    for _ in range(rounds):
        auc = _auc_rows(P + S, yf)                      # (M,) add each member in turn
        if not np.any(np.isfinite(auc)):
            return np.full(M, 1.0 / M), np.nan
        best = float(np.nanmax(auc))
        cand = np.flatnonzero(auc >= best - TIE_EPS)
        m = int(cand[int(np.argmin(counts[cand]))])
        counts[m] += 1
        S += P[m]
        last = best
    return counts / float(rounds), last


def _greedy_search(probs, y_true, tasks, member_labels, fold_masks, folds, rounds,
                   n_grid, time):
    """SEARCH_MODE='greedy': _greedy_fold for every (class, fold), then the same
    fold-averaging, OOF scoring and report as the exhaustive path."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    M, N, T = probs.shape
    y_bool = y_true.astype(bool)
    jobs = [(c, f) for c in range(T) for f in range(N_FOLDS)]
    workers = min(SEARCH_WORKERS or max(1, (os.cpu_count() or 2) // 2), len(jobs))
    print(f"[search]   method    : GREEDY forward selection — {rounds} rounds x {M} "
          f"members = {rounds * M:,} evaluations per class and fold, vs {n_grid:,} "
          f"for the full lattice ({workers} threads)", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        picked = list(pool.map(
            lambda j: _greedy_fold(probs[:, :, j[0]], y_bool[:, j[0]],
                                   fold_masks[j[1]], rounds), jobs))
    elapsed = time.time() - t0

    result = {}
    for c, t in enumerate(tasks):
        fold_ws = np.stack([picked[c * N_FOLDS + f][0] for f in range(N_FOLDS)], axis=0)
        result[t] = _fold_result(fold_ws, probs, y_true, c, folds)
        print(f"[search]   {c + 1}/{T} {t:<18} OOF AUROC={result[t]['oof_auroc']:.4f}")
    print(f"[search]   fitted {len(jobs)} (class, fold) weight vectors in {elapsed:.1f}s",
          flush=True)
    _print_weights(result, tasks, member_labels, M)
    return result


def _fold_result(fold_ws, probs, y_true, c, folds):
    """Average the per-fold weight vectors and score each fold's own vector on its
    held-out rows — the OOF (honest) AUROC for this class."""
    oof_scores = [_safe_auroc(y_true[folds[f][1], c],
                              _blend_col(probs, fold_ws[f], c)[folds[f][1]])
                  for f in range(len(folds))]
    oof = float(np.nanmean(oof_scores)) if np.any(~np.isnan(oof_scores)) else float("nan")
    return {"weights": fold_ws.mean(axis=0), "fold_weights": fold_ws, "oof_auroc": oof}


def _print_weights(result, tasks, member_labels, M):
    """The chosen per-class weights, as the members x classes matrix."""
    wmat = [[result[t]["weights"][m] for t in tasks] for m in range(M)]
    print("\n[search] CHOSEN WEIGHTS  (fold-averaged; each column sums to 1, "
          "'·' = member unused for that class)")
    for line in _weights_table(member_labels, tasks, wmat,
                               oof={t: result[t]["oof_auroc"] for t in tasks}):
        print(line)
    print()


def _weights_table(member_labels, tasks, wmat, oof=None, indent="  "):
    """The per-class weights as a MATRIX: one row per member, one column per class —
    the readable shape for this. A run label is long and there are 5 classes, so the
    one-line-per-class form ("a=0.2  b=0.0  ...") is a wall of text; here a member's
    whole profile reads across, and a class's split reads down. wmat is (M, T) aligned
    to (member_labels, tasks). Zeros print as a dot so the non-zero picks stand out."""
    M, T = len(member_labels), len(tasks)
    wide = max([len(l) for l in member_labels] + [len("sum (per class)")])   # footers fit too
    col = 8
    L = [indent + f"{'#':<3} {'member':<{wide}}" + "".join(f"{t[:col - 1]:>{col}}" for t in tasks),
         indent + "-" * (3 + 1 + wide + col * T)]
    for m in range(M):
        cells = "".join((f"{'·':>{col}}" if wmat[m][c] <= 0 else f"{wmat[m][c]:>{col}.3f}")
                        for c in range(T))
        L.append(indent + f"{'M' + str(m + 1):<3} {member_labels[m]:<{wide}}" + cells)
    L.append(indent + "-" * (3 + 1 + wide + col * T))
    L.append(indent + f"{'':<3} {'sum (per class)':<{wide}}"
             + "".join(f"{sum(wmat[m][c] for m in range(M)):>{col}.3f}" for c in range(T)))
    if oof is not None:
        L.append(indent + f"{'':<3} {'OOF AUROC':<{wide}}"
                 + "".join(f"{oof[t]:>{col}.4f}" for t in tasks))
    return L


def _cv_weight_search(probs, y_true, tasks, member_labels):
    """Per-class N_FOLDS-fold-CV weight search. Returns a dict per task with the final
    (fold-averaged) weight vector, the per-fold vectors, and the OOF AUROC (each fold's
    own weights scored on its held-out fold — the honest estimate)."""
    import time
    from sklearn.model_selection import KFold
    from math import comb
    M, N, T = probs.shape
    probs = np.ascontiguousarray(probs, dtype=np.float64)
    if SEARCH_MODE not in ("greedy", "grid"):
        raise SystemExit(f"SEARCH_MODE must be 'greedy' | 'grid', got {SEARCH_MODE!r}")
    k = int(round(1.0 / GRID_STEP))
    n_grid = comb(k + M - 1, M - 1)                   # size of the FULL lattice
    allowed = ", ".join(f"{i / k:g}" for i in range(min(k, 3) + 1)) + (", ... 1" if k > 3 else "")
    print(f"\n[search] fitting per-class weights over {M} members on {N} images")
    print(f"[search]   weights   : each in {{{allowed}}} (step {GRID_STEP}), summing to 1 "
          f"— a lattice of {n_grid:,} vectors")
    print(f"[search]   validation: {N_FOLDS}-fold CV (seed={SEED}) — fit on {N_FOLDS - 1} "
          f"folds, score the held-out one, average the {N_FOLDS} vectors")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(N)))
    fold_masks = []
    for train_idx, _ in folds:
        m = np.zeros(N, dtype=bool)
        m[train_idx] = True
        fold_masks.append(m)

    if SEARCH_MODE == "greedy":
        return _greedy_search(probs, y_true, tasks, member_labels, fold_masks, folds, k,
                              n_grid, time)

    if n_grid * M * 8 > 4e9:
        raise SystemExit(
            f"SEARCH_MODE='grid' with {M} members at step {GRID_STEP} needs a "
            f"{n_grid:,}-row lattice ({n_grid * M * 8 / 1e9:.1f} GB just to hold it). "
            f"Use SEARCH_MODE='greedy' (same lattice, ~{k * M:,} evaluations per class "
            f"and fold), a coarser GRID_STEP, or group members with a nested list.")
    G = np.asarray(_weight_grid(M, GRID_STEP), dtype=np.float64)   # (n_grid, M)
    dist2 = ((G - 1.0 / M) ** 2).sum(axis=1)          # L2² to uniform — the tie-breaker

    # One job = one class x one slab of grid rows. All N_FOLDS folds are fitted inside
    # the job off a SINGLE sort of that slab (see _fit_chunk), so the folds cost one sort
    # between them instead of one each. Threads, not processes: numpy's argsort/cumsum
    # and BLAS release the GIL, and the prob stack never has to be pickled. Half the
    # logical CPUs is the sweet spot — BLAS is already threaded inside each job.
    from concurrent.futures import ThreadPoolExecutor
    import os
    use_gpu = (SEARCH_DEVICE == "cuda"
               or (SEARCH_DEVICE == "auto" and torch.cuda.is_available()))
    if SEARCH_DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("SEARCH_DEVICE='cuda' but no CUDA device is visible.")
    # Bigger slabs on the GPU (its whole advantage is width); one CUDA stream, so no
    # thread pool there — the CPU path instead spreads slabs over cores.
    cells = 20_000_000 if use_gpu else _SEARCH_CHUNK_CELLS
    rows = max(1, int(cells // max(N, 1)))                     # grid rows per slab
    slabs = [(i, min(i + rows, len(G))) for i in range(0, len(G), rows)]
    workers = 1 if use_gpu else min(SEARCH_WORKERS or max(1, (os.cpu_count() or 2) // 2),
                                    T * len(slabs))
    dev = torch.device("cuda") if use_gpu else None
    where = f"GPU ({torch.cuda.get_device_name(0)})" if use_gpu else f"CPU, {workers} threads"
    print(f"[search]   work      : {len(G) * T * N_FOLDS:,} candidate AUROCs in "
          f"{T * len(slabs)} batched jobs ({len(slabs)} slab(s)/class) on {where}",
          flush=True)

    def _job(job):
        c, (a, b) = job
        args = (G[a:b], a, np.ascontiguousarray(probs[:, :, c]),
                y_true[:, c].astype(bool), fold_masks, fold_stats[c], dist2[a:b])
        return _fit_chunk_gpu(*args, dev) if use_gpu else _fit_chunk(*args)

    y_bool = y_true.astype(bool)
    fold_stats = [[(int(y_bool[m, c].sum()), int(m.sum() - y_bool[m, c].sum()))
                   for m in fold_masks] for c in range(T)]
    jobs = [(c, s) for c in range(T) for s in slabs]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(_job, jobs))
    elapsed = time.time() - t0

    # reduce the per-slab winners into one winner per (class, fold): highest AUROC, ties
    # within TIE_EPS broken toward the most-uniform vector — the same rule as within a slab
    best = {}
    for (c, _s), res in zip(jobs, done):
        for f, (auc, dist, gi) in enumerate(res):
            cur = best.get((c, f))
            if gi < 0:
                continue
            if cur is None or auc > cur[0] + TIE_EPS or \
                    (auc >= cur[0] - TIE_EPS and dist < cur[1]):
                best[(c, f)] = (max(auc, cur[0]) if cur else auc, dist, gi)

    result = {}
    for c, t in enumerate(tasks):
        fold_ws = np.stack([G[best[(c, f)][2]] if (c, f) in best else np.full(M, 1.0 / M)
                            for f in range(N_FOLDS)], axis=0)
        result[t] = _fold_result(fold_ws, probs, y_true, c, folds)
        print(f"[search]   {c + 1}/{T} {t:<18} OOF AUROC={result[t]['oof_auroc']:.4f}")
    print(f"[search]   searched {len(G) * T * N_FOLDS:,} candidates in {elapsed:.1f}s", flush=True)
    _print_weights(result, tasks, member_labels, M)
    return result


def _apply_weights(probs, per_class):
    """Assemble the (N, T) weighted-blend matrix from the searched per-class weights."""
    M, N, T = probs.shape
    ens = np.zeros((N, T), dtype=float)
    for c, t in enumerate(list(per_class.keys())):
        ens[:, c] = _blend_col(probs, per_class[t]["weights"], c)
    return ens


def run_weight_search(load_cfg, ckpt_path_of, out_dir: Path, thr_map: dict = None,
                      thr_source: str = ""):
    """WEIGHTS = "search": load/compute each member's probs (same shared cache), FIT
    per-class weights by CV, score with them, and SAVE everything into out_dir — a
    fresh timestamped folder, so an earlier fit is never overwritten. Reuse the result
    later with WEIGHTS = "<that folder>"."""
    import json
    from datetime import datetime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref, tasks, df = _split_df(load_cfg)
    print(f"[weighted] {SET}: {len(df)} images  |  tasks={tasks}  device={device}")
    cache_dir = _cache_dir_of(out_dir)
    print(f"[cache] dir: {cache_dir / SET_TAG}")

    thr_map = thr_map or {}
    if thr_map:
        print(f"[weighted] thresholds from {thr_source or '(provided)'}")
    else:
        print(f"[weighted] WARNING: no thresholds ({thr_source or 'none'}) -> F1/P/R/Spec at 0.5")
    thr_vec = [float(thr_map.get(t, 0.5)) for t in tasks]

    y_true = np.asarray(_load_y_true(load_cfg, df, device, cache_dir))
    members, _group_tree = _gather(load_cfg, ckpt_path_of, df, device, cache_dir, tasks)
    member_labels = list(members.keys())
    probs = np.stack([members[lab] for lab in member_labels], axis=0)   # (M, N, T)

    # COMBINE_SPACE: fixed per-member, per-class transform applied BEFORE the search,
    # so the weights are fitted in the same space the blend happens in. Members are
    # untouched (their own scores stay on the probability scale).
    probs_t = _to_space(probs, COMBINE_SPACE)
    if COMBINE_SPACE != "prob":
        print(f"[weighted] combine space: {COMBINE_SPACE}")

    per_class = _cv_weight_search(probs_t, y_true, tasks, member_labels)

    # --- assemble blends + metrics ---
    _PC = ("auroc", "auprc", "f1", "precision", "recall", "specificity")

    def _metrics(prob, blended=False):
        m = sc.compute_metrics(y_true, prob, tasks, threshold=thr_vec)
        out = {"macro": m["macro"],
               "per_class": {t: {k: m["per_task"][t][k] for k in _PC} for t in tasks}}
        if blended and COMBINE_SPACE == "rank":
            # a mean rank is NOT on the probability scale, so the frozen thresholds
            # don't transfer -> void the threshold-dependent metrics (AUROC/AUPRC are
            # threshold-free and stay valid).
            for t in tasks:
                for k in ("f1", "precision", "recall", "specificity"):
                    out["per_class"][t][k] = None
            for k in ("mean_f1", "mean_precision", "mean_recall", "mean_specificity"):
                out["macro"][k] = None
        return out

    weighted = _metrics(_from_space(_apply_weights(probs_t, per_class),        # searched weights
                                    COMBINE_SPACE), blended=True)
    flat = _metrics(_from_space(probs_t.mean(axis=0), COMBINE_SPACE),          # 1/M baseline
                    blended=True)
    per_member = {lab: _metrics(members[lab]) for lab in member_labels}

    oof_mean = float(np.nanmean([per_class[t]["oof_auroc"] for t in tasks]))
    _M = len(member_labels)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "set": SET, "n_images": int(len(df)),
        "members": list(member_labels),
        # nested-list sub-ensembles averaged flat BEFORE the weight search (one weight
        # per group, never per run inside it). Empty when the ensemble is flat.
        "groups": _group_tree,
        "combine_space": COMBINE_SPACE,
        "search_mode": SEARCH_MODE,
        "grid_step": GRID_STEP, "n_folds": N_FOLDS, "seed": SEED,
        "thresholds_source": thr_source or "(none -> 0.5)",
        "thresholds": {t: thr_vec[i] for i, t in enumerate(tasks)},
        # the searched decision rule (per class, per member) + fold detail. THIS is what
        # WEIGHTS="<this folder>" reads back.
        "weights_per_class": {t: {member_labels[m]: float(per_class[t]["weights"][m])
                                  for m in range(_M)} for t in tasks},
        "fold_weights_per_class": {
            t: [{member_labels[m]: float(per_class[t]["fold_weights"][f, m])
                 for m in range(_M)} for f in range(N_FOLDS)] for t in tasks},
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
        "per_member": {lab: {"macro": per_member[lab]["macro"],
                             "per_class": per_member[lab]["per_class"]} for lab in member_labels},
    }

    txt = _render_search_txt(summary, tasks, _PC)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"weighted_{SET_TAG}_summary.json"
    txt_path  = out_dir / f"weighted_{SET_TAG}_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote -> {json_path}")
    print(f"wrote -> {txt_path}")

    # Final banner: after all that detail, restate the ONE number that matters so it
    # is impossible to miss at the bottom of a long console dump.
    _w, _o = summary["weighted_mean_auroc"], summary["oof_mean_auroc"]
    _big = f"WEIGHTED mean AUROC   {_w:.4f}"
    _pad = 74
    print("\n" * 2 + "╔" + "═" * _pad + "╗")
    print("║" + f"{SET_TAG}  ·  {len(df)} images  ·  {len(member_labels)} members".center(_pad) + "║")
    print("║" + " " * _pad + "║")
    print("║" + _big.center(_pad) + "║")
    print("║" + " " * _pad + "║")
    print("║" + f"OOF (honest, generalization) {_o:.4f}".center(_pad) + "║")
    print("║" + f"flat 1/{len(member_labels)} baseline {summary['flat_mean_auroc']:.4f}"
          f"   ({summary['gain_mean_auroc_vs_flat']:+.4f})".center(_pad) + "║")
    print("╚" + "═" * _pad + "╝\n")
    # the fitted weights live in this folder and nowhere else — say how to reuse them
    print(f"[weights] fitted weights SAVED (earlier fits untouched). To reuse them:\n"
          f'    WEIGHTS = "{_weights_ref(out_dir)}"\n')


def _resolve_ckpt(ckpt_dir: Path, checkpoint):
    """A member's `checkpoint` spec -> a concrete .pt inside ckpt_dir.

    A STEP resolves to ckpt_step<N>.pt (the rolling checkpoints) or, failing that,
    top_step<N>.pt — the top-K keeper's naming, which is what the top_checkpoints
    report lists. A bare "<file>.pt" is taken relative to ckpt_dir (shared_code's
    _resolve_resume would read it relative to the CWD); "best"/"last" defer to it."""
    d = Path(ckpt_dir)
    if isinstance(checkpoint, int) or (isinstance(checkpoint, str) and checkpoint.isdigit()):
        n = int(checkpoint)
        cands = [d / f"ckpt_step{n}.pt", d / f"top_step{n}.pt"]
        for p in cands:
            if p.exists():
                return p
        raise FileNotFoundError(f"no checkpoint for step {n} in {d} "
                                f"(looked for {', '.join(p.name for p in cands)})")
    if isinstance(checkpoint, str) and checkpoint.endswith(".pt"):
        p = Path(checkpoint)
        return p if p.is_absolute() else d / p
    return sc._resolve_resume(checkpoint, d)


def _weights_ref(out_dir: Path) -> str:
    """The string to paste into WEIGHTS to reuse this run's fitted weights — relative to
    others/ when the folder is under it, else the absolute path."""
    p = Path(out_dir)
    here = Path(__file__).resolve().parent
    try:
        return p.resolve().relative_to(here).as_posix()
    except Exception:
        return str(p)


def _render_search_txt(s: dict, tasks, PC) -> str:
    """Comprehensive readable twin of the search summary json."""
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
         f"search={s.get('search_mode', 'grid')}   weight step={s['grid_step']}   "
         f"folds={s['n_folds']}   seed={s['seed']}",
         f"thresholds (F1/P/R/Spec): {s['thresholds_source']}",
         "=" * 92,
         "HEADLINE",
         f"  WEIGHTED mean AUROC = {s['weighted_mean_auroc']:.4f}   "
         f"(in-sample; fit and scored on the same {s['n_images']} images)",
         f"  OOF      mean AUROC = {s['oof_mean_auroc']:.4f}   "
         f"<- the HONEST number: each image scored by weights fit WITHOUT it",
         f"  FLAT 1/{len(s['members'])} mean AUROC = {s['flat_mean_auroc']:.4f}   "
         f"-> gain = {s['gain_mean_auroc_vs_flat']:+.4f}",
         "-" * 92,
         "CHOSEN PER-CLASS WEIGHTS  (fold-averaged; each column sums to 1, "
         "'·' = unused for that class)"]
    L += _weights_table(s["members"], tasks,
                        [[s["weights_per_class"][t][m] for t in tasks] for m in s["members"]],
                        oof=s["oof_auroc_per_class"])
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
                                        f"(or use RUN_ON='modal'). A cached member never "
                                        f"needs the .pt.")
            return p
        return _resolve_ckpt(base / Path(sub).parent, checkpoint)   # honor stage subfolder

    thr_map, thr_src = _load_thresholds(THRESHOLDS_FROM)
    out_dir = _out_subdir_in(_results_root_local())
    if SEARCH_WEIGHTS:
        run_weight_search(load_cfg, ckpt_path_of, out_dir, thr_map, thr_src)
    else:
        cw_map, cw_src = _load_class_weights(LOAD_WEIGHTS)
        run_ensemble(load_cfg, ckpt_path_of, out_dir, thr_map, thr_src, cw_map, cw_src)
    return out_dir


# ----------------------------- modal execution -----------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

# --------------------- resolve RUN_ON ("auto") -> MODE ---------------------
# A member is FREE if its probs are already cached: the ensemble is then a CPU average
# over .npy files, no checkpoint and no images needed. A missing member needs BOTH its
# checkpoint and the image tree — which live on the Modal volumes, not on this PC — so
# "auto" sends the run to Modal exactly when something has to be computed.
# In the REMOTE container this module is re-imported; there the answer is always
# "modal" (never re-decide from a container-local cache dir).
_IS_REMOTE = _MODAL_OK and not modal.is_local()
if _IS_REMOTE:
    MODE, _ALL_CACHED = "modal", False
else:
    _ALL_CACHED = _members_all_cached(Path(__file__).resolve().parent, SET_TAG)
    if RUN_ON == "auto":
        MODE = "local" if _ALL_CACHED else "modal"
        print(f"[auto] {SET}: "
              + ("every member is cached -> running LOCAL (CPU, no GPU, no images)"
                 if _ALL_CACHED else
                 "some members need compute -> running on MODAL"))
        if MODE == "modal" and not _MODAL_OK:
            raise SystemExit(
                "RUN_ON='auto' needs Modal (some members aren't cached) but modal isn't "
                "installed. Install it, or cache those members first.")
    elif RUN_ON in ("local", "modal"):
        MODE = RUN_ON
        if MODE == "local" and not _ALL_CACHED:
            print("[warn] RUN_ON='local' but some members aren't cached — this needs the "
                  "image tree, which is NOT on this PC. Use RUN_ON='auto' or 'modal'.")
    else:
        raise SystemExit(f"RUN_ON must be 'auto' | 'local' | 'modal', got {RUN_ON!r}")
NUM_WORKERS = NUM_WORKERS_MODAL if MODE == "modal" else NUM_WORKERS_LOCAL

# Build the app ONLY on the launching (local) side, and only when we're actually going
# to Modal. In the remote container this module is imported (to reuse the cores), and
# modal.is_local() is False there, so we skip rebuilding the app/image/volumes.
if _MODAL_OK and modal.is_local() and MODE == "modal":
    _ref_cfg = sc.load_config(PKG_ROOT / REF_RUN, verbose=False)
    app = modal.App(f"ensemble-{SET}")
    _runs_vol = modal.Volume.from_name(_ref_cfg["modal"]["runs_volume"], create_if_missing=True)

    # Mount EVERY distinct data volume any member uses, each at its own mount point,
    # so members resolve images at their own remote_data_root. Members can live on
    # different volumes (e.g. small-res runs -> chexpert-data /data; native-res runs
    # -> chexpert-native-data /data_native), so one mount is not enough.
    _all_runs = _all_member_runs()      # every LEAF run in the tree, groups included
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
    _all_cached = _ALL_CACHED             # already decided above (single check)
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
        # references shared_code) gets cloudpickled. Reuse the cores from the MOUNTED
        # module (imported here, where shared_code is importable).
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
            return _E._resolve_ckpt(base / _P(sub).parent, checkpoint)   # honor stage subfolder

        out_dir = _E._out_subdir_in(_E._results_root_remote(runs_mount))
        try:
            # thresholds AND any reused per-class weights are resolved locally by the
            # launcher and passed in, so they load regardless of what's on the volume.
            if _E.SEARCH_WEIGHTS:
                _E.run_weight_search(load_cfg, ckpt_path_of, out_dir, thr_map, thr_source)
            else:
                _E.run_ensemble(load_cfg, ckpt_path_of, out_dir, thr_map, thr_source,
                                class_weights, weights_source)
        finally:
            _runs_vol.commit()                        # persist before the local fetch reads it
        # hand the volume-relative POSIX subpath back so the launcher can download it
        return out_dir.relative_to(runs_mount).as_posix()


if __name__ == "__main__":
    print(f"[mode] WEIGHTS={WEIGHTS!r} -> "
          + ("FIT fresh per-class weights (CV search) and save them" if SEARCH_WEIGHTS
             else f"reuse saved weights from {LOAD_WEIGHTS}" if LOAD_WEIGHTS
             else "flat 1/M average"))
    if MODE == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but modal isn't installed; set RUN_ON='local'.")
        _base_local = Path(__file__).resolve().parent
        _runs_volume = _ref_cfg["modal"]["runs_volume"]
        _sync_cache_up(_runs_volume, _base_local)    # push local cache so remote can reuse it
        _thr_map, _thr_src = _load_thresholds(THRESHOLDS_FROM)   # resolve locally -> pass to remote
        _cw_map, _cw_src = _load_class_weights(LOAD_WEIGHTS)     # reused weights (or None)
        with modal.enable_output():
            with app.run():
                remote_sub = ensemble_remote.remote(_thr_map, _thr_src,
                                                    _cw_map, _cw_src)   # volume-relative POSIX subpath
        _sync_cache_down(_runs_volume, _base_local)  # pull GPU-computed cache back to this PC
        # remote run + volume commit are done; pull the results folder down locally.
        if remote_sub:
            local_dir = _fetch_results_from_modal(_runs_volume, remote_sub,
                                                  _results_root_local())
            _update_best_tracker(SET_TAG, local_dir)     # promote if it beats the local best
    else:
        out_dir = run_local()
        _update_best_tracker(SET_TAG, out_dir)           # promote if it beats the local best
