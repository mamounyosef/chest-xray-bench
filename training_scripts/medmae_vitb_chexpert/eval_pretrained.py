"""
eval_pretrained.py  —  experiment: medmae_vitb_chexpert
=======================================================
ZERO-SHOT baseline: score the Medical-MAE CheXpert checkpoint EXACTLY as it comes
off Google Drive (already fine-tuned on CheXpert, 5-class head KEPT) on OUR two
validation subsets — WITHOUT any training of our own:
    val       : the primary 10% val split (01_val.csv)
    valid200  : the official 200-study radiologist set (01_valid200.csv)

Nothing is trained and no run checkpoint is loaded — the weights come straight from
model.checkpoint_path (/data/pretrained/vit-b_CXR_0.5M_mae_chexpert.pth, downloaded
by modal_scripts/modal_download_medmae.py). This is the "before we fine-tune"
reference for the medmae_vitb_chexpert warm-start.

Uses the SAME data layer + metrics as in-training validation (cfg drives image
geometry / u_policy), so the numbers are directly comparable to this run's later
in-training validations. AUROC/AUPRC are the meaningful, threshold-free numbers;
F1/P/R/Spec are at the default 0.5 threshold. Writes results/pretrained_baseline.json.

NOTE (read the per-class AUROCs): we score at OUR 384x320 (the checkpoint was trained
at 224, pos-embed is resampled at load) with our ImageNet norm, so this is NOT the
paper's reported 89.3 — it is the baseline IN OUR PIPELINE. Also, keeping the
checkpoint's head assumes its 5-class order equals cfg['tasks']; a class whose AUROC
is ~0.5 would flag an order mismatch.

Run:  python eval_pretrained.py        (honours RUN_ON below: "modal" or "local")
"""

import sys
from pathlib import Path

import torch.nn as nn

# ============================ CONFIG (edit here) ============================
RUN_ON = "modal"     # "modal" -> run on Modal GPU ; "local" -> this machine
AMP    = False       # False -> full fp32 for the cleanest baseline metrics
# ===========================================================================

# Force UTF-8 so the emoji in the prints never crash a cp1252 console.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXP_NAME = "medmae_vitb_chexpert"


def _resolve_pkg_root() -> Path:
    """training_scripts/ (holds shared_code.py) in BOTH environments: locally it's
    this file's parent's parent; on Modal it's mounted at /root/training_scripts."""
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


# -----------------------------------------------------------------------------
# Local execution (RUN_ON = "local"): needs the checkpoint at model.checkpoint_path
# reachable on THIS machine (normally it lives on the Modal data volume, so use
# RUN_ON='modal').
# -----------------------------------------------------------------------------
def run_local():
    model = sc.build_medmae_vit(cfg, load_pretrained=True)   # Google-Drive CheXpert weights
    sc.evaluate_baseline(cfg, model, out_dir=EXP_DIR / cfg["output"]["run_dir"], amp=AMP)


# -----------------------------------------------------------------------------
# Modal execution (RUN_ON = "modal"): mounts the data + runs volumes, builds the
# model from the checkpoint on the data volume, writes the baseline JSON back.
# -----------------------------------------------------------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

if _MODAL_OK:
    app = modal.App(cfg["experiment"]["name"] + "-eval-pretrained")
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
    def eval_pretrained_remote():
        import sys as _sys
        from pathlib import Path as _P
        if "/root/training_scripts" not in _sys.path:
            _sys.path.insert(0, "/root/training_scripts")
        import shared_code as _sc

        rcfg = _sc.remote_cfg(cfg)                       # repoint data paths at /data
        # load_pretrained=True -> seed weights from model.checkpoint_path
        # (/data/pretrained/vit-b_CXR_0.5M_mae_chexpert.pth), head KEPT (config).
        model = _sc.build_medmae_vit(rcfg, load_pretrained=True)
        out_dir = _P(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"] / rcfg["output"]["run_dir"]
        try:
            _sc.evaluate_baseline(rcfg, model, out_dir=out_dir, amp=AMP,
                                  persist_fn=_runs_vol.commit)
        finally:
            _runs_vol.commit()                          # persist the baseline JSON


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but the modal package isn't installed "
                             "(pip install modal), or set RUN_ON='local'.")
        with modal.enable_output():
            with app.run():
                eval_pretrained_remote.remote()
    elif RUN_ON == "local":
        run_local()
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
