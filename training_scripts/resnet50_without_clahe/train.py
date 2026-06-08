"""
train.py  —  experiment: resnet50_without_clahe
================================================
Thin entry point. The shared engine does all the heavy lifting; this file only:
  1. loads the config (../shared_config.yaml merged with ./config.yaml),
  2. builds the model (the one thing that differs between experiments),
  3. hands off to shared_code.run_experiment.

Resume is config-driven: set `resume:` in config.yaml (null/'best'/'last'/<step>/<path>).

Run locally:   python train.py
Run on Modal:  modal run training_scripts/resnet50_without_clahe/train.py
"""

import sys
from pathlib import Path

import torch.nn as nn
import torchvision

# Make shared_code.py (one level up, in training_scripts/) importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shared_code as sc          # noqa: E402  (import after sys.path tweak)

cfg = sc.load_config(HERE, verbose=False)


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
    model = build_model(cfg)
    sc.run_experiment(cfg, model, HERE)


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
