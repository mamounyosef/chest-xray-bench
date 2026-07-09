"""
evaluate_tta.py  —  experiment: convnext_base_22k_pleural_effusion_only
=============================================================
Test-Time Augmentation (TTA) twin of evaluate.py. Same job, same data layer,
same metrics, same frozen-threshold decision rule — the ONLY difference is that
each image is scored as the MEAN probability over a small set of light,
label-preserving augmented views (see TTA_VIEWS below) instead of a single
forward pass. Everything else (checkpoint loading, thresholds, report layout,
Modal/local execution) is identical to evaluate.py.

Views are DETERMINISTIC (fixed magnitudes, not random) so the score is
reproducible. Probabilities are averaged AFTER logits->prob so tasks with
different heads still combine correctly.

Outputs (into results/, with a distinct suffix so evaluate.py's files stay put):
    <set>_results_tta.json   <- source of truth
    <set>_results_tta.txt    <- readable twin
    thresholds_tta.json      <- per-task max-F1 thresholds calibrated on the
                                TTA score distribution (val), so the decision
                                rule matches the distribution it is applied to.

Run:  python evaluate_tta.py     (honours RUN_ON below: "modal" or "local")
build_model here MIRRORS train.py so the architecture matches the checkpoint.
"""

import copy
import sys
from pathlib import Path

import torch.nn as nn
import timm

# ============================ CONFIG (edit here) ============================
RUN_ON        = "modal"   # "modal" -> run on Modal GPU ; "local" -> this machine
CHECKPOINT    = "best"    # which checkpoint to score: best | last | <int step> | path
AMP           = False     # False -> full fp32 for the cleanest final metrics
OBJECTIVE     = "f1"      # threshold objective if thresholds_tta.json must be (re)calibrated
OUTPUT_SUFFIX = "_tta"    # appended to result/threshold filenames (keeps evaluate.py's untouched)

# --- compute knobs (TTA does len(TTA_VIEWS) full passes, so these matter a lot) ---
GPU         = "A100"      # Modal GPU string (T4|L4|A10G|A100|A100-80GB|H100|H200); ignored when RUN_ON="local"
BATCH_SIZE  = 672         # inference batch size (bigger = fewer, larger forward passes)
NUM_WORKERS = 16          # DataLoader workers PER view. Modal CPU request is bumped to match;
                          # locally, >0 on Windows can be slow to spawn — set 0 if so.

# TTA views — each dict is one deterministic forward pass; probs are averaged
# over all of them. Keys (all optional; omit = identity for that view):
#   rotate     : degrees (+ccw / -cw), fill=0 (black, matches the zero-pad)
#   translate  : [fx, fy] shift as a FRACTION of width/height (+right/+down)
#   scale      : zoom factor (>1 in, <1 out), fill=0
#   brightness : multiplier (1.0 = unchanged)
#   contrast   : multiplier (1.0 = unchanged)
#   clahe      : override cfg CLAHE for THIS view (True/False)
# Geometry is applied first, then photometric. Keep magnitudes small so every
# view stays a plausible chest X-ray (label-preserving).
TTA_VIEWS = [
    {"name": "identity"},
    {"name": "zoom_in",     "scale": 1.06},
    {"name": "zoom_out",    "scale": 0.94},
    {"name": "shift_right", "translate": [0.03, 0.0]},
    {"name": "shift_left",  "translate": [-0.03, 0.0]},
    {"name": "shift_down",  "translate": [0.0, 0.03]},
    {"name": "shift_up",    "translate": [0.0, -0.03]},
    {"name": "rotate_+5",   "rotate": 5.0},
    {"name": "rotate_-5",   "rotate": -5.0},
    {"name": "brighter",    "brightness": 1.10},
    {"name": "darker",      "brightness": 0.90},
    {"name": "contrast_up", "contrast": 1.10},
    {"name": "contrast_dn", "contrast": 0.90},
    {"name": "clahe",       "clahe": True},
]
# ===========================================================================

# Windows consoles default to cp1252 and choke on the emoji in the logs; force
# UTF-8 so prints never crash (same thing run_experiment does for training).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXP_NAME = "convnext_base_22k_pleural_effusion_only"


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

# Apply the compute knobs above onto the loaded cfg so BOTH paths honour them:
# the Modal function reads modal.gpu/cpu_cores via sc.modal_resources(cfg), and
# cpu_cores is raised to cover NUM_WORKERS (10 workers on 8 cores just thrashes).
cfg["modal"]["gpu"] = GPU
cfg["modal"]["cpu_cores"] = max(int(cfg["modal"].get("cpu_cores", 8)), NUM_WORKERS + 2)


def build_model(cfg: dict) -> nn.Module:
    """ConvNeXt-B (timm), head sized to one logit per task. MUST match train.py so
    the checkpoint's state dict loads cleanly. pretrained=False — the checkpoint
    supplies the weights — but the model id still defines the exact architecture."""
    return timm.create_model(cfg["model"]["name"], pretrained=False,
                             num_classes=sc.num_output_logits(cfg))


# =============================================================================
# TTA core — self-contained here (not in shared_code) so this stays a drop-in
# beside evaluate.py. Every function takes `sc` explicitly (no module-global
# reference to shared_code) so cloudpickle can ship them to Modal by value;
# shared_code is imported INSIDE the remote body and passed in.
# =============================================================================

def _make_transform(spec: dict, W: int, H: int):
    """Build the deterministic view transform for one TTA spec, operating on a
    (B, 1, H, W) uint8 batch. Returns a callable, or None for the identity view.
    Uses torchvision v2 functional so it matches the train-aug backend exactly."""
    import torchvision.transforms.v2.functional as TF
    from torchvision.transforms.v2 import InterpolationMode

    angle = float(spec.get("rotate", 0.0))
    scale = float(spec.get("scale", 1.0))
    tr = spec.get("translate", [0.0, 0.0])
    tx, ty = int(round(float(tr[0]) * W)), int(round(float(tr[1]) * H))
    brightness = spec.get("brightness", None)
    contrast = spec.get("contrast", None)

    has_geo = angle != 0.0 or scale != 1.0 or tx != 0 or ty != 0
    has_photo = brightness is not None or contrast is not None
    if not has_geo and not has_photo:
        return None

    def _fn(imgs):
        out = imgs
        if has_geo:
            out = TF.affine(out, angle=angle, translate=[tx, ty], scale=scale,
                            shear=[0.0, 0.0],
                            interpolation=InterpolationMode.BILINEAR, fill=0)
        if brightness is not None:
            out = TF.adjust_brightness(out, float(brightness))
        if contrast is not None:
            out = TF.adjust_contrast(out, float(contrast))
        return out
    return _fn


def _tta_predict(sc, cfg, model, df, device, loss_fn, amp, channels_last,
                 batch_size, num_workers, views):
    """Run the model over `df` once PER view and return the MEAN-probability
    prediction: (y_true (N,T), y_prob (N,T), exclude_mask (N,T), identity_loss).
    Mirrors shared_code._predict_dataframe (split='val', same loader settings)
    but injects each view's deterministic transform on the uint8 batch before the
    GPU normalize, and toggles CLAHE per-view via a shallow cfg copy. Assumes the
    gpu_normalize (uint8) data path so fill=0 matches the pad and brightness/
    contrast act in [0,255]."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    img_cfg = cfg["image"]
    W, H = int(img_cfg["width"]), int(img_cfg["height"])
    layout = sc.task_layout(cfg)
    dl = cfg["dataloader"]
    pin = bool(dl.get("val_pin_memory", dl.get("pin_memory", True))) and device.type == "cuda"

    model.eval()
    prob_sum, y_true, excl, ident_loss = None, None, None, 0.0

    for vi, spec in enumerate(views):
        clahe_override = spec.get("clahe", None)
        cfg_v = cfg
        if clahe_override is not None:
            cfg_v = copy.deepcopy(cfg)
            cfg_v["clahe"]["use_clahe"] = bool(clahe_override)
        transform = _make_transform(spec, W, H)

        ds = sc.CheXpertDataset(df, cfg_v, split="val")
        kwargs = dict(batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=num_workers, pin_memory=pin)
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(dl.get("val_prefetch_factor",
                                                   dl.get("prefetch_factor", 2)))
            kwargs["persistent_workers"] = bool(dl.get("val_persistent_workers", False))
        loader = DataLoader(ds, **kwargs)

        ys, ps = [], []
        vloss, n = 0.0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                if transform is not None:
                    imgs = transform(imgs)
                imgs = sc.prepare_batch(imgs, cfg_v, device, channels_last)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(imgs)
                    loss = loss_fn(logits, labels)
                bs = imgs.size(0)
                vloss += loss.item() * bs
                n += bs
                ys.append(labels.detach().cpu().numpy())
                ps.append(sc.logits_to_probs(logits, layout).float().detach().cpu().numpy())
        p = np.concatenate(ps)
        prob_sum = p if prob_sum is None else prob_sum + p
        if vi == 0:                          # y_true is view-invariant; take it once
            y_true, excl = sc.binarize_targets(np.concatenate(ys), layout,
                                               sc._val_ignore_uncertain(cfg))
            ident_loss = vloss / max(1, n)
        print(f"    [tta] view {vi + 1}/{len(views)}: {spec.get('name', 'view')}  "
              f"(clahe={cfg_v['clahe']['use_clahe']})")

    y_prob = prob_sum / float(len(views))
    return y_true, y_prob, excl, ident_loss


def _calibrate_tta(sc, cfg, model, df_val, device, amp, batch_size, num_workers, views):
    """Per-task max-F1 thresholds computed on the TTA-averaged val predictions —
    same PROCEDURE as shared_code.calibrate_thresholds, but on the TTA scores so
    the frozen rule matches the distribution evaluate_tta applies it to."""
    import numpy as np                                          # noqa: F401 (parity)
    tasks = cfg["tasks"]
    loss_fn = sc.build_loss(cfg, device)
    print(f"[calibrate-tta] TTA inference over {len(df_val)} val images "
          f"x {len(views)} views (bs={batch_size}, workers={num_workers}) ...")
    y_true, y_prob, excl, _ = _tta_predict(
        sc, cfg, model, df_val, device, loss_fn, amp, False,
        batch_size, num_workers, views)
    thresholds = {}
    for k, t in enumerate(tasks):
        keep = ~excl[:, k]
        thr, _f1 = sc._best_f1_threshold(y_true[keep, k], y_prob[keep, k])
        thresholds[t] = thr
    return thresholds, y_true, y_prob, excl


def _load_thresholds_file(path, tasks):
    """Read a frozen thresholds JSON (thresholds_tta.json). Returns {task: thr}
    or None if absent/incomplete."""
    import json
    path = Path(path)
    if not path.exists():
        return None
    data = json.load(open(path, encoding="utf-8"))
    thr = data.get("thresholds", {})
    if not all(t in thr for t in tasks):
        return None
    return {t: float(thr[t]) for t in tasks}


def evaluate_official_tta(sc, cfg, model, experiment_dir, device=None,
                          checkpoint="best", amp=False, batch_size=None,
                          num_workers=0, objective="f1",
                          eval_sets=("valid200", "test500"),
                          views=None, output_suffix="_tta"):
    """TTA version of shared_code.evaluate_official: load `checkpoint`, score the
    official sets with MEAN-over-views probabilities, apply frozen per-task
    thresholds (calibrated on the TTA val distribution, saved to
    thresholds{suffix}.json), and write <set>_results{suffix}.{json,txt}."""
    import json
    from datetime import datetime

    import pandas as pd
    import torch

    views = views or [{"name": "identity"}]
    experiment_dir = Path(experiment_dir)
    out = cfg["output"]
    tasks = cfg["tasks"]
    results_dir = experiment_dir / out["run_dir"]
    ckpt_dir = results_dir / out["checkpoints_dir"]
    _sub = sc.finetune_ckpt_subdir(cfg)          # two-stage runs: fine-tune subfolder
    if _sub:
        ckpt_dir = ckpt_dir / _sub
    data_dir = Path(cfg["paths"]["data_dir"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = int(cfg["dataloader"].get("val_batch_size",
                         cfg["dataloader"]["batch_size"]))

    print("#" * 70)
    print(f"# 🧪 official evaluation (TTA): {cfg.get('experiment', {}).get('name')}")
    print(f"# 🖥️  device={device}  amp={amp}  base_use_clahe={cfg['clahe']['use_clahe']}  "
          f"u_policy={cfg['labels']['u_policy']}")
    print(f"# 🔁 TTA views ({len(views)}): "
          + ", ".join(v.get("name", "view") for v in views))
    print("#" * 70)

    ckpt, ckpt_path = sc._load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    model = model.to(device)
    print(f"[eval] loaded checkpoint: {ckpt_path}")
    print(f"[eval] checkpoint was step={ckpt.get('global_step')}  "
          f"epoch={ckpt.get('epoch')}  best_score={ckpt.get('best_score')}")

    # frozen TTA thresholds: separate file so evaluate.py's thresholds.json stays put.
    thr_path = results_dir / f"thresholds{output_suffix}.json"
    thr_map = _load_thresholds_file(thr_path, tasks)
    if thr_map is None:
        print(f"  ⚠️  no usable {thr_path.name} — calibrating per-task TTA thresholds "
              f"now on {cfg['paths']['val_csv']} ...")
        df_val = pd.read_csv(data_dir / cfg["paths"]["val_csv"])
        thr_map, y_t, y_p, excl_v = _calibrate_tta(
            sc, cfg, model, df_val, device, amp, batch_size, num_workers, views)
        payload, _ = sc._threshold_payload(
            cfg, thr_map, y_t, y_p, excl_v, ckpt, ckpt_path, len(df_val), objective)
        payload["tta"] = True
        payload["tta_views"] = [v.get("name", "view") for v in views]
        thr_path.parent.mkdir(parents=True, exist_ok=True)
        with open(thr_path, "w", encoding="utf-8") as fh:
            json.dump(sc._json_safe(payload), fh, indent=2)
        print(f"  💾 saved frozen TTA thresholds -> {thr_path}")
    else:
        print(f"[eval] using frozen TTA thresholds from {thr_path}")
    thr_vec = [thr_map[t] for t in tasks]
    print("[eval] per-task thresholds: "
          + "  ".join(f"{t}={thr_map[t]:.3f}" for t in tasks))

    loss_fn = sc.build_loss(cfg, device)
    _all_sets = {"valid200": cfg["paths"]["valid200_csv"],
                 "test500":  cfg["paths"]["test500_csv"]}
    sets = [(n, _all_sets[n]) for n in eval_sets if n in _all_sets]
    print(f"[eval] scoring sets: {[n for n, _ in sets]}")

    reports = {}
    for set_name, csv_name in sets:
        csv_path = data_dir / csv_name
        print("=" * 70)
        print(f"🧪 TTA evaluating on {set_name}  ({csv_path})")
        if not csv_path.exists():
            print(f"  ⚠️  CSV not found — skipping {set_name}")
            continue
        df = pd.read_csv(csv_path)
        print(f"  rows: {len(df)}  ->  running {len(views)}-view TTA inference ...")
        y_true, y_prob, excl, val_loss = _tta_predict(
            sc, cfg, model, df, device, loss_fn, amp, False,
            batch_size, num_workers, views)
        metrics = sc.compute_metrics(y_true, y_prob, tasks, threshold=thr_vec,
                                     exclude_mask=excl)

        report = {
            "experiment": cfg.get("experiment", {}).get("name"),
            "set": set_name,
            "csv": csv_name,
            "checkpoint": str(ckpt_path),
            "checkpoint_step": ckpt.get("global_step"),
            "checkpoint_epoch": ckpt.get("epoch"),
            "n_images": int(len(df)),
            "device": str(device),
            "amp": bool(amp),
            "tta": True,
            "tta_aggregate": "mean_prob",
            "tta_views": [v.get("name", "view") for v in views],
            "threshold_objective": objective,
            "threshold_source": cfg["paths"]["val_csv"] + " (TTA)",
            "thresholds": {t: float(thr_map[t]) for t in tasks},
            "use_clahe": bool(cfg["clahe"]["use_clahe"]),
            "u_policy": cfg["labels"]["u_policy"],
            "val_loss": float(val_loss),
            "macro": metrics["macro"],
            "per_task": metrics["per_task"],
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        json_path = results_dir / f"{set_name}_results{output_suffix}.json"
        txt_path = results_dir / f"{set_name}_results{output_suffix}.txt"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sc._json_safe(report), f, indent=2)
        txt = sc._render_report_txt(report)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(txt)
        print(f"  💾 {json_path}")
        print(f"  💾 {txt_path}")
        reports[set_name] = report

    print("#" * 70)
    print(f"# 🏁 official TTA evaluation done — {len(reports)} set(s) scored.")
    print("#" * 70)
    return reports


# -----------------------------------------------------------------------------
# Local execution (RUN_ON = "local").
# -----------------------------------------------------------------------------
def run_local():
    model = build_model(cfg)
    evaluate_official_tta(sc, cfg, model, EXP_DIR,
                          checkpoint=CHECKPOINT, amp=AMP, objective=OBJECTIVE,
                          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                          views=TTA_VIEWS, output_suffix=OUTPUT_SUFFIX)


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
    app = modal.App(cfg["experiment"]["name"] + "-eval-tta")
    # serialized=True requires the image's Python to match THIS interpreter's.
    _image = sc.modal_image(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
    _data_vol = modal.Volume.from_name(cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    # serialized=True -> ship this function by value (cloudpickle). It calls the
    # __main__-defined TTA driver, which cloudpickle serializes by value too (its
    # helpers take `sc` as an argument, so nothing references shared_code as a
    # module global before it is importable on the remote).
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
        import timm as _timm

        rcfg = _sc.remote_cfg(cfg)                       # repoint data paths at /data
        model = _timm.create_model(rcfg["model"]["name"], pretrained=False,
                                   num_classes=_sc.num_output_logits(rcfg))
        out_dir = _P(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        try:
            evaluate_official_tta(
                _sc, rcfg, model, out_dir,
                checkpoint=CHECKPOINT, amp=AMP, objective=OBJECTIVE,
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                views=TTA_VIEWS, output_suffix=OUTPUT_SUFFIX)
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
