"""
calibrate_threshold.py  —  experiment: densenet121_seed123
==========================================================
Threshold calibration. Runs the trained checkpoint over the in-training
validation set (01_val.csv) and picks, FOR EACH pathology, the decision
threshold that maximizes F1 on that set. Prints the 5 thresholds + the val
scores they achieve, and writes them to results/thresholds.json — the frozen
decision rule that evaluate.py then applies, unchanged, to valid200 / test500.

Why per experiment? The threshold operates on THIS model's score distribution,
so it must be recalibrated for every checkpoint / experiment. What stays fixed
across all arms is the PROCEDURE (max-F1 per task on 01_val.csv) — that is what
keeps the comparison fair. AUROC / AUPRC are threshold-free and need none of this.

Run:  python calibrate_threshold.py    (honours RUN_ON below: "modal" or "local")
build_model here MIRRORS train.py so the architecture matches the checkpoint.
"""

import sys
from pathlib import Path

import torch.nn as nn
import torchvision

# ============================ CONFIG (edit here) ============================
RUN_ON     = "modal"     # "modal" -> run on Modal GPU ; "local" -> this machine
CHECKPOINT = "best"      # which checkpoint to calibrate: best | last | <int step> | path
AMP        = False       # False -> full fp32 (calibration is cheap; keep it exact)
OBJECTIVE  = "f1"        # threshold-selection objective (per-task, max-F1)
# ===========================================================================

# Windows consoles default to cp1252 and choke on the emoji in the logs; force
# UTF-8 so prints never crash (same thing run_experiment does for training).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXP_NAME = "densenet121_seed123"


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
    match train.py so the checkpoint's state dict loads cleanly."""
    model = torchvision.models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(cfg["tasks"]))
    return model


# -----------------------------------------------------------------------------
# Local execution (RUN_ON = "local").
# -----------------------------------------------------------------------------
def run_local():
    model = build_model(cfg)
    sc.run_calibration(cfg, model, EXP_DIR,
                       checkpoint=CHECKPOINT, amp=AMP, objective=OBJECTIVE)


# -----------------------------------------------------------------------------
# Modal execution (RUN_ON = "modal"): mounts the data + runs volumes, reads the
# checkpoint from the runs volume, writes thresholds.json back there.
# -----------------------------------------------------------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

if _MODAL_OK:
    app = modal.App(cfg["experiment"]["name"] + "-calibrate")
    # serialized=True requires the image's Python to match THIS interpreter's.
    _image = sc.modal_image(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    _data_vol = modal.Volume.from_name(cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    # serialized=True -> ship this function by value (cloudpickle), since it is
    # defined in __main__ when launched via `python calibrate_threshold.py` +
    # app.run(). The body is self-contained: it puts the mounted source tree on
    # sys.path and imports shared_code + builds the model INSIDE.
    @app.function(
        image=_image,
        volumes={cfg["modal"]["data_mount"]: _data_vol,
                 cfg["modal"]["runs_mount"]: _runs_vol},
        serialized=True,
        **sc.modal_resources(cfg),
    )
    def calibrate_remote():
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
            _sc.run_calibration(rcfg, model, out_dir,
                                checkpoint=CHECKPOINT, amp=AMP, objective=OBJECTIVE,
                                num_workers=int(rcfg["dataloader"]["val_num_workers"]))
        finally:
            _runs_vol.commit()                          # persist thresholds.json


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but the modal package isn't installed "
                             "(pip install modal), or set RUN_ON='local'.")
        with modal.enable_output():                     # stream remote logs locally
            with app.run():                             # ephemeral app (== `modal run`)
                calibrate_remote.remote()
    elif RUN_ON == "local":
        run_local()
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
