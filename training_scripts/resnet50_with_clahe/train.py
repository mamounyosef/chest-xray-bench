"""
train.py  —  experiment: resnet50_with_clahe
============================================
Thin entry point. The shared engine does all the heavy lifting; this file only:
  1. loads the config (../shared_config.yaml merged with ./config.yaml),
  2. builds the model (the one thing that differs between experiments),
  3. hands off to shared_code.run_experiment.

This is the CLAHE arm: IDENTICAL to resnet50_without_clahe in every respect
EXCEPT clahe.use_clahe: true (set in this folder's config.yaml). Keeping
everything else fixed is what makes the CLAHE comparison controlled.

Resume is config-driven: set `resume:` in config.yaml (null/'best'/'last'/<step>/<path>).

Run locally:   python train.py
Run on Modal:  modal run training_scripts/resnet50_with_clahe/train.py
"""

import sys
from pathlib import Path

import torch.nn as nn
import torchvision

EXP_NAME = "resnet50_with_clahe"


def _resolve_pkg_root() -> Path:
    """Locate the training_scripts/ dir (holds shared_code.py + shared_config.yaml)
    in BOTH environments: locally it's this file's parent's parent; on Modal the
    whole tree is mounted at /root/training_scripts by shared_code.modal_image().
    Pick whichever actually contains shared_code.py."""
    for cand in (Path("/root/training_scripts"),
                 Path(__file__).resolve().parent.parent):
        if (cand / "shared_code.py").exists():
            return cand
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _resolve_pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc          # noqa: E402  (import after sys.path tweak)

cfg = sc.load_config(EXP_DIR, verbose=False)


def build_model(cfg: dict) -> nn.Module:
    """ResNet-50 with the final FC swapped to one logit per task.
    ImageNet-1k V1 weights (kept consistent with the other models' available
    pretraining) unless cfg['model']['pretrained'] is false."""
    pretrained = cfg.get("model", {}).get("pretrained", True)
    weights = "IMAGENET1K_V1" if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(cfg["tasks"]))
    return model


# -----------------------------------------------------------------------------
# Local execution:  python train.py   (outputs go in THIS experiment folder)
# -----------------------------------------------------------------------------
def run_local():
    # Seed BEFORE building the model so the new FC head's random init is
    # reproducible (run_experiment re-seeds later, but that's after the model
    # already exists). set_seed inside the engine then makes data/training match.
    sc.set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
    model = build_model(cfg)
    sc.run_experiment(cfg, model, EXP_DIR)


# -----------------------------------------------------------------------------
# Modal execution:  modal run train.py   (data + outputs on Modal volumes)
# Guarded so a modal-free local environment can still `python train.py`.
# -----------------------------------------------------------------------------
try:
    import modal

    app = modal.App(cfg["experiment"]["name"])
    _image = sc.modal_image()
    _data_vol = modal.Volume.from_name(cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)
    # W&B API key comes from a Modal secret; only attach it if W&B is enabled.
    _secrets = ([modal.Secret.from_name(cfg["modal"]["wandb_secret"])]
                if cfg.get("wandb", {}).get("enabled") else [])

    @app.function(
        image=_image,
        volumes={cfg["modal"]["data_mount"]: _data_vol,
                 cfg["modal"]["runs_mount"]: _runs_vol},
        secrets=_secrets,
        **sc.modal_resources(cfg),
    )
    def train_remote():
        # Repoint data paths at the mounted data volume; write outputs to the
        # runs volume so checkpoints/logs persist after the container exits.
        rcfg = sc.remote_cfg(cfg)
        out_dir = Path(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        # Seed BEFORE building the model (reproducible FC-head init); see run_local.
        sc.set_seed(rcfg["reproducibility"]["seed"], rcfg["reproducibility"]["deterministic"])
        model = build_model(rcfg)

        # Optional W&B: same scores as the CSV logs, system stats disabled.
        wb = cfg.get("wandb", {})
        run, log_fn = None, None
        if wb.get("enabled"):
            import wandb
            try:
                _settings = wandb.Settings(x_disable_stats=True)   # no system/GPU metrics
            except Exception:
                _settings = None
            run = wandb.init(project=wb.get("project", "chest-xray-bench"),
                             name=cfg["experiment"]["name"], config=cfg, settings=_settings)
            log_fn = lambda data, step: run.log(data, step=step)

        try:
            # persist_fn=_runs_vol.commit -> checkpoints + CSV logs committed to the
            # volume live at every validation (not just at the end).
            sc.run_experiment(rcfg, model, out_dir,
                              persist_fn=_runs_vol.commit, log_fn=log_fn)
        finally:
            if run is not None:
                run.finish()
            _runs_vol.commit()                  # final commit after the run

    @app.local_entrypoint()
    def modal_main():
        train_remote.remote()

except ImportError:
    pass        # modal not installed -> local-only mode


if __name__ == "__main__":
    run_local()
