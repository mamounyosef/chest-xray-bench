"""
label_uncertain.py  —  experiment: convnext_base_22k_5pct_val
============================================================
One-off soft-label generator for the 95/5 (02_*) split. This is a copy of
convnext_base_22k_certain_only/label_uncertain.py retargeted at the 02 split; it
still uses the SAME teacher (the certain_only checkpoint) so the self-trained
Consolidation labels are identical in origin to final_stage1's — only the split
(and hence the row coverage) differs.

What it does:
  1. loads the CERTAIN-only teacher checkpoint
     (convnext_base_22k_certain_only/results/checkpoints/best.pt),
  2. runs it over the 95/5 TRAIN split (02_train.csv),
  3. keeps only the rows that have >=1 UNCERTAIN (-1) label among the 5 tasks,
  4. writes  Path + the model's positive probability for ALL 5 tasks  to
     02_train_softlabels.csv.

Because the teacher's per-image prediction is deterministic, the rows shared with
01_train_softlabels.csv are byte-identical; this file simply ALSO covers the ~9k
rows that moved into 02_train (from the old 01_val), so convnext_base_22k_5pct_val
gets a real self-trained target for every uncertain Consolidation cell (no U-Ones
fallback). Reuses the shared data layer (same preprocessing/geometry/CLAHE).

EXP_NAME stays convnext_base_22k_certain_only ON PURPOSE: load_config + the
checkpoint path resolve against the TEACHER run. Only train_csv (-> 02_train.csv)
and OUT_NAME (-> 02_train_softlabels.csv) are overridden below.

Run:  python label_uncertain.py     (honours RUN_ON below: "modal" or "local")
build_model here MIRRORS train.py so the architecture matches the checkpoint.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import timm

# ============================ CONFIG (edit here) ============================
RUN_ON     = "modal"     # "modal" -> run on Modal GPU ; "local" -> this machine
CHECKPOINT = "best"      # which checkpoint to use: best | last | <int step> | path
AMP        = False       # False -> full fp32 for the cleanest probabilities
OUT_NAME   = "02_train_softlabels.csv"   # output file name (95/5 split)
TRAIN_CSV_OVERRIDE = "02_train.csv"      # label the 02 split, not the teacher's 01_train.csv
# ===========================================================================

for _s in (sys.stdout, sys.stderr):     # UTF-8 so emoji prints never crash on Windows
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXP_NAME = "convnext_base_22k_certain_only"


def _resolve_pkg_root() -> Path:
    """training_scripts/ (holds shared_code.py) in BOTH environments."""
    for cand in (Path("/root/training_scripts"),
                 Path(__file__).resolve().parent.parent):
        if (cand / "shared_code.py").exists():
            return cand
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _resolve_pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc          # noqa: E402

cfg = sc.load_config(EXP_DIR, verbose=False)
# Retarget the teacher at the 95/5 split. This mutates the module-level cfg BEFORE
# the Modal remote function captures it (serialized=True pickles cfg BY VALUE), so
# the override reaches the remote worker. soft_label reads paths.train_csv to pick
# the split; the checkpoint still resolves against the certain_only teacher.
cfg["paths"]["train_csv"] = TRAIN_CSV_OVERRIDE


def build_model(cfg: dict):
    """ConvNeXt-B (timm), head sized to one logit per task. MUST match train.py so
    the checkpoint loads cleanly. pretrained=False — the checkpoint supplies the
    weights."""
    return timm.create_model(cfg["model"]["name"], pretrained=False,
                             num_classes=sc.num_output_logits(cfg))


def soft_label(cfg, model, experiment_dir, out_csv, *,
               checkpoint="best", amp=False, num_workers=4):
    """Predict soft labels for the uncertain TRAIN rows and write them to out_csv.
    Columns: the path column + one positive-probability column per task.

    NOTE: every dependency (incl. shared_code) is imported INSIDE this function on
    purpose. On Modal the remote function is shipped with serialized=True (pickled
    BY VALUE); a module-level `import shared_code` would be captured by reference
    and fail to deserialize remotely ('No module named shared_code'). Local imports
    keep the pickled function self-contained."""
    import shared_code as sc
    import numpy as np
    import pandas as pd
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader
    experiment_dir = Path(experiment_dir)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = cfg["output"]
    ckpt_dir = experiment_dir / out["run_dir"] / out["checkpoints_dir"]
    sub = sc.finetune_ckpt_subdir(cfg)           # "" for this single-stage run
    if sub:
        ckpt_dir = ckpt_dir / sub

    model = model.to(device)
    perf = cfg.get("performance", {})
    channels_last = bool(perf.get("channels_last")) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    _, ckpt_path = sc._load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    model.eval()
    print(f"[soft-labels] loaded checkpoint: {ckpt_path}")

    tasks = cfg["tasks"]
    pcol = cfg["paths"].get("path_column", "Path")
    train_csv = Path(cfg["paths"]["data_dir"]) / cfg["paths"]["train_csv"]
    df = pd.read_csv(train_csv)
    unc = (df[tasks] == -1).any(axis=1).to_numpy()       # >=1 uncertain among tasks
    df_unc = df.loc[unc].reset_index(drop=True)
    print(f"[soft-labels] train rows: {len(df)}   uncertain rows "
          f"(>=1 of {tasks} == -1): {len(df_unc)}")

    layout = sc.task_layout(cfg)
    ds = sc.CheXpertDataset(df_unc, cfg, split="val")    # 'val' -> no augmentation
    dl = cfg["dataloader"]
    bs = int(dl.get("val_batch_size") or dl["batch_size"])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    probs = []
    for imgs, _ in loader:
        imgs = sc.prepare_batch(imgs, cfg, device, channels_last)   # to GPU + (uint8) normalize
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(imgs)
        probs.append(sc.logits_to_probs(logits, layout).float().cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, len(tasks)), dtype=float)

    out_df = pd.DataFrame({pcol: df_unc[pcol].tolist()})
    for k, t in enumerate(tasks):
        out_df[t] = probs[:, k]                          # task column = soft P(positive)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[soft-labels] wrote {len(out_df)} rows x {len(tasks)} task-probs -> {out_csv}")
    return out_csv


# -----------------------------------------------------------------------------
# Local execution: read the checkpoint from this folder's results/ and write the
# CSV straight into the local CheXpert data folder (cfg paths.data_dir).
# -----------------------------------------------------------------------------
def run_local():
    model = build_model(cfg)
    out_csv = Path(cfg["paths"]["data_dir"]) / OUT_NAME
    soft_label(cfg, model, EXP_DIR, out_csv, checkpoint=CHECKPOINT, amp=AMP)


# -----------------------------------------------------------------------------
# Modal execution: mount data + runs volumes; read the checkpoint + train images
# from the volumes; write the CSV into this run's results/ on the runs volume
# (download it to data/chexpert/ afterward — see the modal commands file).
# -----------------------------------------------------------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

if _MODAL_OK:
    app = modal.App(cfg["experiment"]["name"] + "-softlabels")
    _image = sc.modal_image(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    _data_vol = modal.Volume.from_name(cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(
        image=_image,
        volumes={cfg["modal"]["data_mount"]: _data_vol,
                 cfg["modal"]["runs_mount"]: _runs_vol},
        serialized=True,
        **sc.modal_resources(cfg),
    )
    def label_remote():
        import sys as _sys
        from pathlib import Path as _P
        if "/root/training_scripts" not in _sys.path:
            _sys.path.insert(0, "/root/training_scripts")
        import shared_code as _sc
        import timm as _timm

        rcfg = _sc.remote_cfg(cfg)                       # repoint data paths at /data
        model = _timm.create_model(rcfg["model"]["name"], pretrained=False,
                                   num_classes=_sc.num_output_logits(rcfg))
        out_dir = _P(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        out_csv = out_dir / rcfg["output"]["run_dir"] / OUT_NAME
        try:
            soft_label(rcfg, model, out_dir, out_csv, checkpoint=CHECKPOINT, amp=AMP,
                       num_workers=int(rcfg["dataloader"]["val_num_workers"]))
        finally:
            _runs_vol.commit()                           # persist the CSV


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but the modal package isn't installed.")
        with modal.enable_output():
            with app.run():
                label_remote.remote()
    elif RUN_ON == "local":
        run_local()
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
