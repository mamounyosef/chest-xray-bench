"""
Source of the `modeling.py` that ships inside the Hugging Face repo.

Kept here as a plain string so the exported file stays standalone: the copy that
users download must not import this project's shared_code.py.
"""

MODELING_PY = '''"""
modeling.py — build any model in this repo and load its weights.

    from modeling import load_model
    model, cfg = load_model("rad_dino_vitB_768")     # downloads from the Hub
    model, cfg = load_model("./rad_dino_vitB_768")   # or from a local folder

Every model returns raw logits. See config.json -> head for how the logits map
onto the five findings.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

REPO_ID = "mamounyosef/chest-xray-bench"

TORCHVISION = {"resnet50", "densenet121", "densenet201",
               "convnext_tiny", "convnext_small"}


def _fetch(run, filename):
    """Local folder if it exists, otherwise the Hub."""
    local = Path(run) / filename
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, f"{Path(run).name}/{filename}")


class RadDinoClassifier(nn.Module):
    """RAD-DINO (HF Dinov2) backbone with a linear head over
    concat[CLS, mean(patch tokens)]. The pretrained 37x37 position grid is
    resampled to this run's non-square grid at every forward."""

    def __init__(self, n_out, head_pool="cls_patch"):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        conf = AutoConfig.from_pretrained("microsoft/rad-dino")
        self.backbone = AutoModel.from_config(conf)     # architecture only
        self.head_pool = head_pool
        hidden = self.backbone.config.hidden_size
        in_dim = hidden * (2 if head_pool == "cls_patch" else 1)
        self.head = nn.Linear(in_dim, n_out)

    def forward(self, x):
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        seq = out.last_hidden_state
        cls = seq[:, 0]
        feat = cls if self.head_pool == "cls" else \\
            torch.cat([cls, seq[:, 1:].mean(dim=1)], dim=1)
        return self.head(feat)


def build_model(cfg):
    """The architecture for one config.json, with random weights."""
    name = cfg["backbone"]
    arch = cfg.get("arch")
    n_out = cfg["head"]["n_logits"]
    h, w = cfg["image"]["height"], cfg["image"]["width"]

    if arch == "raddino":
        return RadDinoClassifier(n_out, cfg.get("head_pool", "cls_patch"))

    if arch == "medmae_vitb":
        import timm
        return timm.create_model(name, pretrained=False, num_classes=n_out,
                                 img_size=(h, w), global_pool="avg")

    if name in TORCHVISION:
        import torchvision
        model = getattr(torchvision.models, name)(weights=None)
        if hasattr(model, "fc"):                       # resnet
            model.fc = nn.Linear(model.fc.in_features, n_out)
        elif isinstance(getattr(model, "classifier", None), nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, n_out)
        else:                                          # convnext
            head = model.classifier[-1]
            model.classifier[-1] = nn.Linear(head.in_features, n_out)
        return model

    import timm
    return timm.create_model(name, pretrained=False, num_classes=n_out)


def load_model(run, device="cpu"):
    """Build the architecture and load this run's weights. Returns (model, cfg)."""
    cfg = json.loads(Path(_fetch(run, "config.json")).read_text(encoding="utf-8"))
    model = build_model(cfg)
    model.load_state_dict(load_file(_fetch(run, "model.safetensors")), strict=True)
    model.eval().to(device)
    return model, cfg


def preprocess(image, cfg):
    """One HxW uint8 or float array (grayscale or RGB) -> a (1,3,H,W) tensor.

    Resize to fit inside the target box keeping the aspect ratio, pad the short
    side with zeros, then normalize. Matching this exactly is what reproduces the
    reported scores.
    """
    import cv2
    import numpy as np

    H, W = cfg["image"]["height"], cfg["image"]["width"]
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scale = min(W / img.shape[1], H / img.shape[0])
    new = (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale)))
    img = cv2.resize(img, new, interpolation=cv2.INTER_AREA)

    canvas = np.zeros((H, W), dtype=img.dtype)
    y0, x0 = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img

    x = torch.from_numpy(canvas).float().div(255.0)
    x = x.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(cfg["image"]["norm_mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["image"]["norm_std"]).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)
'''
