"""
export_hf_readme.py  —  write the repo card and modeling.py into the upload folder.

Reads <OUT>/models.json (written by export_hf.py) and renders <OUT>/README.md plus
<OUT>/modeling.py. Kept separate from the exporter so the card can be rewritten
without touching 12 GB of weights.

Run:  python training_scripts/others/export_hf_readme.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hf_modeling import MODELING_PY      # noqa: E402

OUT = Path(r"D:\chexpert-bench-hf")
HF_REPO = "mamounyosef/chest-xray-bench"
GITHUB = "https://github.com/mamounyosef/chest-xray-bench"

TASKS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
         "Pleural Effusion"]

ENSEMBLE = [
    "rad_dino_vitB_1064x896", "convnext_base_22k_1600x1312",
    "medmae_vitb_nih_B_768_s2", "rad_dino_vitB_768",
    "medmae_vitb_nih_B_768_s2_seed1337", "medmae_vitb_nih_B_448_s1_seed1337",
]
ENSEMBLE_TEST500 = 0.9130       # the six blended, per class weights, logit space

# These two runs have no stored valid200_results.json, so their val200 comes from
# the training log at the selected step. That is the same quantity: for every other
# member the two sources agree to four decimals.
VAL200_FALLBACK = {
    "medmae_vitb_nih_B_768_s2_seed1337": 0.8964,
    "medmae_vitb_nih_B_448_s1_seed1337": 0.8979,
}

# Grouping for the full listing, in the order the report discusses them.
GROUPS = [
    ("Chest X-ray pretrained", lambda m: m.get("arch") in ("raddino", "medmae_vitb")),
    ("High resolution ConvNeXt", lambda m: m["image"]["width"] > 384),
    ("Uncertainty and objective", lambda m: any(
        k in m["run"] for k in ("selftrain", "certain", "zeros", "mixed", "aucm",
                                "curriculum", "final_stage1"))),
    ("Single pathology", lambda m: m["head"]["n_logits"] == 1),
    ("Backbone comparison", lambda m: True),
]

FRONT_MATTER = """---
license: cc-by-nc-4.0
tags:
  - medical
  - chest-xray
  - radiology
  - image-classification
  - multi-label-classification
  - chexpert
library_name: pytorch
pipeline_tag: image-classification
---"""


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "--"


def t5(m):
    return m.get("results", {}).get("test500", {}).get("mean_auroc")


def v2(m):
    return (m.get("results", {}).get("valid200", {}).get("mean_auroc")
            or VAL200_FALLBACK.get(m["run"]))


def geo(m):
    return f"{m['image']['width']}x{m['image']['height']}"


def row(m):
    return (f"| [`{m['run']}`](./{m['run']}) | `{m['backbone']}` | {geo(m)} | "
            f"{m['n_parameters'] / 1e6:.1f}M | {fmt(v2(m))} | {fmt(t5(m))} |")


def main():
    metas = json.loads((OUT / "models.json").read_text(encoding="utf-8"))
    by_run = {m["run"]: m for m in metas}
    L = [FRONT_MATTER, ""]

    L += [
        "# CheXpert model zoo",
        "",
        f"**{len(metas)} chest X-ray classifiers**, all trained on CheXpert under one "
        "fixed pipeline so their scores are directly comparable. Released alongside a "
        "technical report that compares the design choices behind them.",
        "",
        "Most models predict five findings: "
        + ", ".join(f"**{t}**" for t in TASKS) + ".",
        "",
        "---",
        "",
        "## The ensemble members",
        "",
        f"Blended together these six score **{ENSEMBLE_TEST500:.4f} mean AUROC on "
        "test500**, the best result in the study. Each one also works on its own, at "
        "the scores below. Sorted by valid200, the split the blend weights were "
        "fitted on.",
        "",
        "| Model | valid200 | test500 | Input size | Params |",
        "|---|:---:|:---:|:---:|---:|",
    ]
    ens = sorted((by_run[r] for r in ENSEMBLE if r in by_run),
                 key=lambda m: v2(m) or 0, reverse=True)
    for m in ens:
        L.append(f"| [`{m['run']}`](./{m['run']}) | {fmt(v2(m))} | "
                 f"**{fmt(t5(m))}** | {geo(m)} | {m['n_parameters'] / 1e6:.1f}M |")

    L.append(f"| **All six blended** | | **{ENSEMBLE_TEST500:.4f}** | | |")

    L += [
        "",
        "Notice that the valid200 order is almost the reverse of the test500 order. "
        "That is the point the report keeps returning to: with 234 and 668 images, "
        "neither radiologist split ranks these models reliably.",
        "",
        "> **Read the scores carefully.** Differences below about **0.01** on test500 "
        "are inside the noise of retraining the same model with a different seed. "
        "Treat them as ties.",
        "",
        "---",
        "",
        "## Quick start",
        "",
        "```bash",
        "pip install torch timm transformers safetensors huggingface_hub opencv-python",
        "```",
        "",
        "Download `modeling.py` from this repo, then:",
        "",
        "```python",
        "import cv2, torch",
        "from modeling import load_model, preprocess",
        "",
        'model, cfg = load_model("rad_dino_vitB_768")     # any folder name below',
        "",
        'img = cv2.imread("frontal.jpg", cv2.IMREAD_GRAYSCALE)',
        "x = preprocess(img, cfg)",
        "",
        "with torch.no_grad():",
        "    probs = model(x).sigmoid()[0]",
        "",
        'for task, p in zip(cfg["tasks"], probs.tolist()):',
        '    print(f"{task:18} {p:.3f}")',
        "```",
        "",
        "`load_model` picks the right builder for each backbone, so the same two lines "
        "work for every model here. `preprocess` reproduces the training pipeline: "
        "resize to fit the target box keeping the aspect ratio, zero pad the short "
        "side, normalize. **Never mirror a chest X-ray** at inference, it moves the "
        "heart to the wrong side.",
        "",
        "---",
        "",
        "## What is in each folder",
        "",
        "| File | Description |",
        "|---|---|",
        "| `model.safetensors` | Weights only, fp32. No optimizer state. |",
        "| `config.json` | Backbone, input size, normalization, label policy, head "
        "layout, scores. |",
        "| `thresholds.json` | Per finding decision thresholds, tuned for F1 on the "
        "large validation split then frozen. Only needed for hard yes/no predictions. |",
        "",
        "### A note on outputs",
        "",
        "Most models emit one logit per finding, so `sigmoid` gives five "
        "probabilities. A few trained with the three way uncertainty head emit **9 "
        "logits**, and the single pathology models emit **1**. `config.json` always "
        "states which, under `head`:",
        "",
        "```json",
        '"head": {',
        '  "n_logits": 5,',
        '  "layout": "one sigmoid logit per task",',
        '  "task_slices": {"Atelectasis": [0, 1], "Cardiomegaly": [1, 2], "...": []}',
        "}",
        "```",
        "",
        "---",
        "",
        "## How the ensemble was combined",
        "",
        "The report's headline, **0.9130 mean AUROC** on the official test set, "
        "averages the six models above in **logit space**, with a separate set of "
        "member weights fitted per finding on the 234 image validation split.",
        "",
        "That is **+0.0103** over the best single member, with a 95% bootstrap "
        "interval of **[+0.0022, +0.0186]** over 10,000 resamples. The lesson from the "
        "report: *diversity beats count*. Seven runs of the same backbone averaged to "
        "0.8972, below the best single model, while three genuinely different "
        "backbones reached 0.9115.",
        "",
        "---",
        "",
        "## All models",
        "",
        "`valid200` and `test500` are the official radiologist labelled splits, 234 and "
        "668 frontal images. Neither was trained on.",
        "",
    ]
    seen = set()
    for title, pred in GROUPS:
        block = [m for m in metas if m["run"] not in seen and pred(m)]
        if not block:
            continue
        seen |= {m["run"] for m in block}
        L += [
            f"### {title}",
            "",
            "| Model | Backbone | Input size | Params | valid200 | test500 |",
            "|---|---|:---:|---:|:---:|:---:|",
        ]
        L += [row(m) for m in sorted(block, key=lambda m: t5(m) or 0, reverse=True)]
        L.append("")

    L += [
        "---",
        "",
        "## Training setup",
        "",
        "| | |",
        "|---|---|",
        "| Data | CheXpert train split, frontal views only, split 90/10 by patient |",
        "| Loss | Binary cross entropy over the five findings, masked where a target "
        "is undefined |",
        "| Optimizer | AdamW, batch 64, cosine schedule with a one epoch warmup |",
        "| Augmentation | Rotation, translation, scale, brightness, contrast. "
        "No horizontal flip |",
        "| Metric | Mean AUROC over the five findings, scored per image |",
        "",
        f"Full configurations, training code and the report are on "
        f"[GitHub]({GITHUB}).",
        "",
        "---",
        "",
        "## Intended use and limits",
        "",
        "These are **research artifacts**, released to support a technical report.",
        "",
        "- **Not a medical device.** Do not use them to make clinical decisions.",
        "- **No external validation.** Every number here comes from CheXpert's own "
        "splits. Performance on images from other hospitals, scanners or populations "
        "is unknown.",
        "- **The test sets are small.** 234 and 668 images, so per model rankings are "
        "unstable and confidence intervals are wide.",
        "- Labels come from an automatic labeler applied to radiology reports, so the "
        "models learn that labeler's conventions along with the findings.",
        "",
        "## License and data",
        "",
        "**CC BY-NC 4.0**: free to use, share and build on with attribution, "
        "non-commercial only. This matches CheXpert's Stanford University Dataset "
        "Research Use Agreement, which permits research use and forbids commercial use.",
        "",
        "The CheXpert data is **not** redistributed here, in this repo or on GitHub. "
        "Request it from [Stanford AIMI](https://stanfordaimi.azurewebsites.net/) "
        "directly.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@techreport{yosef2026chexpert,",
        "  title       = {A Systematic Study of Design Choices for Multi-Label Chest "
        "X-ray Classification on CheXpert},",
        "  author      = {Yosef, Ma'moun},",
        "  year        = {2026},",
        "  institution = {University of Jordan}",
        "}",
        "```",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "modeling.py").write_text(MODELING_PY, encoding="utf-8")
    print(f"wrote README.md ({len(metas)} models) and modeling.py to {OUT}")


if __name__ == "__main__":
    main()
