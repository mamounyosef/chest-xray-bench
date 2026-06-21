"""
evaluate.py  —  experiment: densenet121
========================================
Final evaluation. Loads a checkpoint and scores it on the two official
radiologist sets — valid200 and test500 — INDEPENDENTLY, using the exact same
data layer + metrics as training (cfg drives CLAHE / u_policy / image geometry,
so every arm is scored identically and fairly).

Decision rule: AUROC & AUPRC are threshold-free. F1 / precision / recall /
specificity are reported at the FROZEN per-task thresholds in
results/thresholds.json (produced by calibrate_threshold.py). If that file is
missing, it is calibrated now on 01_val.csv and saved, so eval is self-contained.

For each set it writes (into results/):
    <set>_results.json   <- source of truth (machine-readable; aggregate later)
    <set>_results.txt    <- readable twin rendered from the same numbers

Run:  python evaluate.py        (honours RUN_ON below: "modal" or "local")
build_model here MIRRORS train.py so the architecture matches the checkpoint.
"""

import sys
from pathlib import Path

import torch.nn as nn
import torchvision

# ============================ CONFIG (edit here) ============================
RUN_ON     = "modal"     # "modal" -> run on Modal GPU ; "local" -> this machine
CHECKPOINT = "best"      # which checkpoint to score: best | last | <int step> | path
AMP        = False       # False -> full fp32 for the cleanest final metrics
OBJECTIVE  = "f1"        # threshold objective if thresholds.json must be (re)calibrated
# ===========================================================================

# Windows consoles default to cp1252 and choke on the emoji in the logs; force
# UTF-8 so prints never crash (same thing run_experiment does for training).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXP_NAME = "densenet121"


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


def build_model(cfg: dict) -> nn.Module:
    """DenseNet-121 with the final classifier swapped to one logit per task. MUST
    match train.py so the checkpoint's state dict loads cleanly. Pretrained weights
    are irrelevant here (the checkpoint overwrites them) — weights=None for speed."""
    model = torchvision.models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(cfg["tasks"]))
    return model


# -----------------------------------------------------------------------------
# Local execution (RUN_ON = "local"): scores using the local dataset + the
# checkpoint already in this folder's results/.
# -----------------------------------------------------------------------------
def run_local():
    model = build_model(cfg)
    sc.evaluate_official(cfg, model, EXP_DIR,
                         checkpoint=CHECKPOINT, amp=AMP, objective=OBJECTIVE)


# -----------------------------------------------------------------------------
# Modal execution (RUN_ON = "modal"): mounts the data + runs volumes, reads the
# checkpoint from the runs volume, writes the result files back there.
# -----------------------------------------------------------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

if _MODAL_OK:
    app = modal.App(cfg["experiment"]["name"] + "-eval")
    # serialized=True requires the image's Python to match THIS interpreter's.
    _image = sc.modal_image(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    _data_vol = modal.Volume.from_name(cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    # serialized=True -> ship this function by value (cloudpickle), since it is
    # defined in __main__ when launched via `python evaluate.py` + app.run().
    # The body is self-contained: it puts the mounted source tree on sys.path and
    # imports shared_code + builds the model INSIDE, so nothing __main__-specific
    # (other than the picklable cfg dict + the config constants) needs to travel.
    @app.function(
        image=_image,
        volumes={cfg["modal"]["data_mount"]: _data_vol,
                 cfg["modal"]["runs_mount"]: _runs_vol},
        serialized=True,
        **sc.modal_resources(cfg),
    )
    def evaluate_remote():
        import sys as _sys
        from pathlib import Path as _P
        if "/root/training_scripts" not in _sys.path:
            _sys.path.insert(0, "/root/training_scripts")
        import shared_code as _sc
        import torchvision as _tv
        import torch.nn as _nn

        rcfg = _sc.remote_cfg(cfg)                       # repoint data paths at /data
        model = _tv.models.densenet121(weights=None)
        model.classifier = _nn.Linear(model.classifier.in_features, len(rcfg["tasks"]))
        out_dir = _P(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        try:
            _sc.evaluate_official(rcfg, model, out_dir,
                                  checkpoint=CHECKPOINT, amp=AMP, objective=OBJECTIVE,
                                  num_workers=int(rcfg["dataloader"]["val_num_workers"]))
        finally:
            _runs_vol.commit()                          # persist result files


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but the modal package isn't installed "
                             "(pip install modal), or set RUN_ON='local'.")
        with modal.enable_output():                     # stream remote logs locally
            with app.run():                             # ephemeral app (== `modal run`)
                evaluate_remote.remote()
    elif RUN_ON == "local":
        run_local()
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
