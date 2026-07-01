"""
ensemble_valid200.py
====================
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
    local :  training_scripts/others/ensembling_results/<timestamp>/ensemble_valid200_summary.{json,txt}
    modal :  /runs/ensembling_results/<timestamp>/ensemble_valid200_summary.{json,txt}  (fetch after)

Run:  python training_scripts/others/ensemble_valid200.py   (honours RUN_ON below)
"""

import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
RUN_ON = "modal"        # "modal" -> Modal GPU (reads best.pt from /runs) | "local"
SET    = "valid200"      # scored split — "valid200" or "test500"
BATCH_SIZE = 256        # inference batch size for EVERY member (None -> each run's
                        # own dataloader.val_batch_size). valid200 is 200 imgs, so
                        # any value >=200 is one batch; lower it only to cap VRAM.

# Full 5-class models — each contributes ALL five classes. Just list run names.
FULL_MODELS = [
    "convnext_base_22k_final_stage1",
    "densenet121",
    "swin_base_22k",
    "convnext_tiny",
    "convnext_large_22k_cxr14_pretrain",
    "efficientnetv2_m",
]

# Per-run checkpoint file under results/checkpoints/ (default "best.pt"). Two-stage
# runs keep their fine-tune best.pt in a stage subfolder.
CKPT_SUBPATH = {
    "convnext_large_22k_cxr14_pretrain": "finetune_chexpert/best.pt",
}

# Per-class thresholds for the F1/precision/recall/specificity of the ENSEMBLE come
# from this run's results/thresholds.json (AUROC/AUPRC stay threshold-free). Missing
# file/task -> 0.5 fallback.
THRESHOLDS_FROM = "convnext_base_22k_final_stage1"

# Each ensemble run writes into its OWN timestamped subfolder under
# ensembling_results/, so trying different ensembles never overwrites the last.
# RUN_TAG (optional) is appended to the timestamp to make a run self-describing,
# e.g. RUN_TAG="5models" -> 2026-07-01_14-30-05_5models.
RUN_TAG = ""

# The five per-disease Stage-2 models, as ONE composite member (one entry).
# Set USE_STAGE2 = False to drop it from the ensemble.
USE_STAGE2 = True
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


def _predict(cfg: dict, ckpt_path: Path, df, device) -> np.ndarray:
    """Probabilities (N, len(cfg tasks)) for one run's best.pt over `df`."""
    model = build_model_generic(cfg).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sc._unwrap(model).load_state_dict(ck["model"])
    bs = int(BATCH_SIZE or cfg["dataloader"].get("val_batch_size",
                                                  cfg["dataloader"]["batch_size"]))
    _, y_prob, _, _ = sc._predict_dataframe(
        cfg, model, df, device, _dummy_loss, amp=False, channels_last=False,
        batch_size=bs, num_workers=int(cfg["dataloader"].get("val_num_workers", 4)))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y_prob


def run_ensemble(load_cfg, ckpt_path_of, out_dir: Path, thr_json: Path = None):
    """Core routine. `load_cfg(run)` -> that run's cfg (data paths already correct
    for this environment); `ckpt_path_of(run)` -> its best.pt path; `thr_json` ->
    a thresholds.json whose per-class thresholds are used for F1/precision/recall/
    specificity (AUROC/AUPRC are threshold-free)."""
    import json, pandas as pd
    from datetime import datetime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reference cfg (first full model) fixes the canonical task order + the split df.
    ref = load_cfg(FULL_MODELS[0])
    tasks = list(ref["tasks"])
    data_dir = Path(ref["paths"]["data_dir"])
    df = pd.read_csv(data_dir / ref["paths"][f"{SET}_csv"])
    print(f"[ensemble] {SET}: {len(df)} images  |  tasks={tasks}  device={device}")

    # per-class thresholds for F1/P/R/Spec (AUROC/AUPRC ignore them).
    thr_map = {}
    if thr_json is not None and Path(thr_json).exists():
        thr_map = json.load(open(thr_json, encoding="utf-8")).get("thresholds", {})
        print(f"[ensemble] thresholds from {thr_json}")
    else:
        print(f"[ensemble] WARNING: {thr_json} not found -> F1/P/R/Spec at 0.5")
    thr_vec = [float(thr_map.get(t, 0.5)) for t in tasks]

    # ground truth (from the reference model; valid200 has no uncertain labels).
    y_true, _ = None, None
    members = {}          # label -> (N, 5) prob matrix

    # --- full 5-class models ---
    for run in FULL_MODELS:
        cfg = load_cfg(run)
        print(f"[member] {run} ...")
        p = _predict(cfg, ckpt_path_of(run), df, device)
        members[run] = p
        if y_true is None:
            yt, _, _, _ = sc._predict_dataframe(   # labels once (cheap, same df)
                cfg, build_model_generic(cfg).to(device).eval(),
                df, device, _dummy_loss, amp=False, channels_last=False,
                batch_size=8, num_workers=0)
            y_true = yt

    # --- Stage-2 composite (one member, per-class dedicated model) ---
    if USE_STAGE2:
        comp = np.zeros((len(df), len(tasks)), dtype=float)
        for disease, run in STAGE2_GROUP.items():
            cfg = load_cfg(run)
            print(f"[member] stage2/{disease}: {run} ...")
            p = _predict(cfg, ckpt_path_of(run), df, device)   # (N,1)
            comp[:, tasks.index(disease)] = p[:, 0]
        members["final_stage2 (5 per-disease composite)"] = comp

    # --- average + metrics (F1/P/R/Spec at the calibrated per-class thresholds) ---
    stack = np.stack(list(members.values()), axis=0)           # (M, N, 5)
    ens = stack.mean(axis=0)
    ens_metrics = sc.compute_metrics(y_true, ens, tasks, threshold=thr_vec)

    # each member's own mean AUROC (for reference; AUROC is threshold-free)
    per_member = {lab: sc.compute_metrics(y_true, p, tasks)["macro"]["mean_auroc"]
                  for lab, p in members.items()}

    _pc_keys = ("auroc", "auprc", "f1", "precision", "recall", "specificity")
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "set": SET, "n_images": int(len(df)),
        "members": list(members.keys()),
        "thresholds_source": str(thr_json),
        "thresholds": {t: thr_vec[i] for i, t in enumerate(tasks)},
        "ensemble_mean_auroc": ens_metrics["macro"]["mean_auroc"],
        "ensemble_macro": ens_metrics["macro"],
        "ensemble_per_class": {t: {k: ens_metrics["per_task"][t][k] for k in _pc_keys}
                               for t in tasks},
        "per_member_mean_auroc": per_member,
    }

    mac = ens_metrics["macro"]
    lines = ["=" * 78,
             f"ENSEMBLE (prob-average)  —  set={SET}  images={len(df)}",
             f"generated: {summary['generated']}",
             f"thresholds (F1/P/R/Spec): {thr_json}", "=" * 78,
             f"  ENSEMBLE mean AUROC={mac['mean_auroc']:.4f}  AUPRC={mac['mean_auprc']:.4f}  "
             f"F1={mac['mean_f1']:.4f}  P={mac['mean_precision']:.4f}  "
             f"R={mac['mean_recall']:.4f}  Spec={mac['mean_specificity']:.4f}",
             "  per-class:"]
    for t in tasks:
        pc = summary["ensemble_per_class"][t]
        lines.append(f"    {t:<18} AUROC={pc['auroc']:.4f}  AUPRC={pc['auprc']:.4f}  "
                     f"F1={pc['f1']:.4f}  P={pc['precision']:.4f}  R={pc['recall']:.4f}  "
                     f"Spec={pc['specificity']:.4f}  (thr={summary['thresholds'][t]:.4f})")
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


# ----------------------------- local execution -----------------------------
def run_local():
    def load_cfg(run):
        return sc.load_config(PKG_ROOT / run, verbose=False)

    def ckpt_path_of(run):
        p = PKG_ROOT / run / "results" / "checkpoints" / CKPT_SUBPATH.get(run, "best.pt")
        if not p.exists():
            raise FileNotFoundError(f"missing {p} — fetch this run's best.pt first "
                                    f"(or use RUN_ON='modal')")
        return p

    thr_json = PKG_ROOT / THRESHOLDS_FROM / "results" / "thresholds.json"
    out_dir = _out_subdir(Path(__file__).resolve().parent)   # others/ensembling_results/<stamp>
    run_ensemble(load_cfg, ckpt_path_of, out_dir, thr_json)


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
    _image = sc.modal_image(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    _data_vol = modal.Volume.from_name(_ref_cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(_ref_cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(
        image=_image,
        volumes={_ref_cfg["modal"]["data_mount"]: _data_vol,
                 _ref_cfg["modal"]["runs_mount"]: _runs_vol},
        serialized=True,
        **sc.modal_resources(_ref_cfg),
    )
    def ensemble_remote():
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
        import ensemble_valid200 as _E          # mounted; app not rebuilt (is_local False)
        runs_mount = _P("/runs")

        def load_cfg(run):
            return _sc.remote_cfg(_sc.load_config(_P("/root/training_scripts") / run, verbose=False))

        def ckpt_path_of(run):
            return runs_mount / run / "results" / "checkpoints" / _E.CKPT_SUBPATH.get(run, "best.pt")

        thr_json = runs_mount / _E.THRESHOLDS_FROM / "results" / "thresholds.json"
        try:
            _E.run_ensemble(load_cfg, ckpt_path_of, _E._out_subdir(runs_mount), thr_json)
        finally:
            _runs_vol.commit()


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but modal isn't installed; set RUN_ON='local'.")
        with modal.enable_output():
            with app.run():
                ensemble_remote.remote()
    elif RUN_ON == "local":
        run_local()
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
