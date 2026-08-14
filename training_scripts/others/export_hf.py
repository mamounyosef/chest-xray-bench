"""
export_hf.py  —  build a Hugging Face upload folder from the local checkpoints.

For every run that has a checkpoint it writes  <OUT>/<run>/  holding:
    model.safetensors   weights only (fp32), no optimizer / scheduler / RNG state
    config.json         backbone id, geometry, normalization, label policy, scores
    thresholds.json     the frozen per-task F1 thresholds, when the run has them

and one top level README.md listing every model with its scores.

Run:  python training_scripts/others/export_hf.py
Then: hf auth login  &&  hf upload <user>/<repo> <OUT>
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# ============================ CONFIG (edit here) ============================
OUT = Path(r"D:\chexpert-bench-hf")     # staging folder for the upload
SKIP = {"densenet121_temp"}             # scratch runs never reported
DTYPE = torch.float32                   # fp32 reproduces the reported AUROC
SKIP_EXISTING = True                    # keep already written .safetensors files
# ===========================================================================

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc          # noqa: E402

TASKS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
         "Pleural Effusion"]


def find_ckpt(run: Path) -> Path | None:
    """best.pt, preferring the CheXpert stage of the two stage runs."""
    hits = sorted((run / "results" / "checkpoints").rglob("best.pt"))
    if not hits:
        return None
    staged = [h for h in hits if "finetune_chexpert" in h.parts]
    return staged[0] if staged else hits[0]


def scores(run: Path) -> dict:
    """mean and per task AUROC on each official set this run was scored on."""
    out = {}
    for split in ("valid200", "test500"):
        f = run / "results" / f"{split}_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        out[split] = {
            "mean_auroc": round(d["macro"]["mean_auroc"], 4),
            "per_task_auroc": {k: round(v["auroc"], 4)
                               for k, v in d["per_task"].items()},
        }
    return out


def weights(ckpt: Path) -> dict:
    """The state dict alone, contiguous and detached, ready for safetensors."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    clean = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        k = k[7:] if k.startswith("module.") else k
        clean[k] = v.detach().to(DTYPE).contiguous()
    return clean


def n_logits(sd: dict) -> int:
    """Output width, read off the classifier weight rather than guessed."""
    for k in ("head.weight", "fc.weight", "classifier.weight",
              "head.fc.weight", "classifier.1.weight"):
        if k in sd:
            return sd[k].shape[0]
    cands = [v.shape[0] for k, v in sd.items()
             if k.endswith(".weight") and v.ndim == 2]
    return cands[-1] if cands else -1


def head_layout(cfg: dict, n_logits: int) -> dict:
    """How the output logits map onto the five findings. A task under the mixed
    policy can carry a 3-way head (negative / positive / uncertain), so a model is
    not always 5 logits wide."""
    per_task = (cfg.get("labels", {}) or {}).get("per_task") or {}
    tasks = cfg.get("tasks", TASKS)
    widths = {t: (3 if per_task.get(t) == "multiclass" else 1) for t in tasks}
    if sum(widths.values()) != n_logits:      # single task runs, or an odd head
        return {"n_logits": n_logits, "layout": "unknown", "tasks": tasks}
    slices, i = {}, 0
    for t in tasks:
        slices[t] = [i, i + widths[t]]
        i += widths[t]
    return {
        "n_logits": n_logits,
        "layout": "one sigmoid logit per task"
                  if n_logits == len(tasks) else
                  "per task: 1 logit (binary, sigmoid) or 3 logits "
                  "(softmax over negative / positive / uncertain)",
        "task_slices": slices,
    }


def export(run: Path) -> dict | None:
    ckpt = find_ckpt(run)
    if ckpt is None:
        return None
    cfg = sc.load_config(run, verbose=False)
    img, model_cfg = cfg.get("image", {}), cfg.get("model", {})
    dest = OUT / run.name
    dest.mkdir(parents=True, exist_ok=True)

    sd = weights(ckpt)
    st = dest / "model.safetensors"
    if not (SKIP_EXISTING and st.exists() and st.stat().st_size > 0):
        save_file(sd, st, metadata={"format": "pt", "run": run.name})

    sc_ = scores(run)
    meta = {
        "run": run.name,
        "backbone": model_cfg.get("name"),
        "arch": model_cfg.get("arch"),
        "tasks": cfg.get("tasks", TASKS),
        "image": {
            "width": img.get("width"), "height": img.get("height"),
            "norm_mean": img.get("norm_mean"), "norm_std": img.get("norm_std"),
            "interpolation": img.get("interpolation"),
        },
        "labels": cfg.get("labels", {}),
        "label_smoothing": (cfg.get("training", {}).get("loss", {})
                            .get("label_smoothing")),
        "source_checkpoint": str(ckpt.relative_to(run)),
        "n_parameters": sum(v.numel() for v in sd.values()),
        "head": head_layout(cfg, n_logits(sd)),
        "results": sc_,
    }
    (dest / "config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    thr = run / "results" / "thresholds.json"
    if thr.exists():
        shutil.copy2(thr, dest / "thresholds.json")

    size = (dest / "model.safetensors").stat().st_size / 2**20
    print(f"  {run.name:46} {size:7.0f} MB  "
          f"{sc_.get('test500', {}).get('mean_auroc', '--')}")
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in PKG_ROOT.iterdir()
                  if p.is_dir() and p.name != "others" and p.name not in SKIP)
    metas = [m for m in (export(r) for r in runs) if m]
    (OUT / "models.json").write_text(
        json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.rglob("*.safetensors")) / 2**30
    print(f"\n{len(metas)} models exported to {OUT}  ({total:.1f} GB)")


if __name__ == "__main__":
    main()
