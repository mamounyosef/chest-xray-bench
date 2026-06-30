"""
train.py  —  experiment: convnext_base_22k_final_stage2_edema
==============================================
Thin entry point. The shared engine does all the heavy lifting; this file only:
  1. loads the config (../shared_config.yaml merged with ./config.yaml),
  2. builds the model (the one thing that differs between experiments),
  3. hands off to shared_code.run_experiment.

Resume is config-driven: set `resume:` in config.yaml (null/'best'/'last'/<step>/<path>).

Run locally:   python train.py
Run on Modal:  modal run training_scripts/convnext_base_22k_final_stage2_edema/train.py
"""

import sys
from pathlib import Path

import torch.nn as nn
import timm

EXP_NAME = "convnext_base_22k_final_stage2_edema"


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
    """ConvNeXt-B (timm) with the head sized to one logit per task.
    timm.create_model(num_classes=N) replaces the head with a fresh
    Linear(num_features, N), giving an N-logit head directly. The model id comes
    from cfg['model']['name'] and carries an explicit pretrained tag — here
    'convnext_base.fb_in22k_ft_in1k_384' (ImageNet-22k pretrained @224 -> IN-1k
    fine-tuned @384), loaded via timm rather than torchvision's IN-1k-only
    weights. ConvNeXt pools adaptively, so our 384x320 input is fine. Weights
    download via timm unless cfg['model']['pretrained'] is false."""
    pretrained = cfg.get("model", {}).get("pretrained", True)
    return timm.create_model(cfg["model"]["name"], pretrained=pretrained,
                             num_classes=sc.num_output_logits(cfg))


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
    # AUC-margin run: add LibAUC (--no-deps; torch/numpy/sklearn already present).
    _image = sc.modal_image(extra_pip=["libauc"], extra_pip_options="--no-deps")
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
