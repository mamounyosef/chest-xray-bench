"""
ensample_tta.py
===============
Test-Time Augmentation (TTA) twin of ensample.py. IDENTICAL ensemble in every
respect — same RUN_ON / SET / GPU / BATCH_SIZE, same FULL_MODELS (same hierarchy
and same per-member checkpoints), the same CKPT_SUBPATH / THRESHOLDS_FROM /
USE_STAGE2 / STAGE2_GROUP — all imported straight from ensample.py so there is ONE source of
truth. The ONLY difference: each member is scored as the MEAN probability over a
small set of light, label-preserving augmented views (TTA_VIEWS below) instead of
a single forward pass.

So the work is  len(TTA_VIEWS)  x  (number of ensemble members): every member run
does 14 full inference passes (one per view), those 14 are averaged into that
member's probability matrix, and THEN the members are averaged together exactly
as in ensample.py. Everything downstream (thresholds, metrics, summary layout,
timestamped output folder, auto-fetch from Modal) is ensample.py's, unchanged.

To keep results separate, the output folder is tagged `..._tta<N>` (N = #views),
so a TTA ensemble never overwrites a plain ensemble.

Run:  python training_scripts/others/ensample_tta.py   (honours ensample.RUN_ON)
"""

import copy
import sys
from pathlib import Path

# TTA views — each dict is one deterministic forward pass; probs are averaged over
# all of them. Keys (all optional; omit = identity for that view):
#   rotate     : degrees (+ccw / -cw), fill=0 (black, matches the zero-pad)
#   translate  : [fx, fy] shift as a FRACTION of width/height (+right/+down)
#   scale      : zoom factor (>1 in, <1 out), fill=0
#   brightness : multiplier (1.0 = unchanged)
#   contrast   : multiplier (1.0 = unchanged)
#   clahe      : override the run's CLAHE for THIS view (True/False)
# Geometry first, then photometric. Kept identical to evaluate_tta.py's 14 views.
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

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _resolve_pkg_root() -> Path:
    # this script lives in training_scripts/others/, so shared_code.py is one level up
    here = Path(__file__).resolve().parent
    for cand in (Path("/root/training_scripts"), here.parent, here):
        if (cand / "shared_code.py").exists():
            return cand
    return here.parent


PKG_ROOT = _resolve_pkg_root()
_HERE = Path(__file__).resolve().parent
for _p in (str(PKG_ROOT), str(_HERE)):        # need others/ on the path too, for `import ensample`
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared_code as sc          # noqa: E402
import ensample as E             # noqa: E402  — the ONE source of config + the ensemble core
import numpy as np               # noqa: E402
import torch                     # noqa: E402

# Config is ensample's, verbatim (used by __main__ / the launcher).
RUN_ON = E.RUN_ON


# =============================================================================
# TTA member prediction — same signature as ensample._predict so it can be swapped
# straight in (ensample.run_ensemble calls the module-global `_predict`). Instead
# of one pass it averages probabilities over TTA_VIEWS.
# =============================================================================

def _make_transform(spec: dict, W: int, H: int):
    """Deterministic view transform for one TTA spec on a (B,1,H,W) uint8 batch.
    Returns a callable, or None for the identity view. torchvision v2 functional,
    matching the train-aug backend so geometry/fill semantics are identical."""
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


def _tta_predict(cfg: dict, ckpt_path: Path, df, device) -> np.ndarray:
    """Drop-in TTA replacement for ensample._predict: probabilities (N, #tasks)
    for one checkpoint over `df`, averaged over TTA_VIEWS. Rebuilds/loads the model
    exactly like ensample._predict, then runs one val pass per view (each with its
    own deterministic transform + CLAHE toggle) and means the per-view probs.
    Assumes the gpu_normalize (uint8) data path so fill=0 matches the pad and
    brightness/contrast act in [0,255]."""
    from torch.utils.data import DataLoader

    model = E.build_model_generic(cfg).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sc._unwrap(model).load_state_dict(ck["model"])

    bs = int(E.BATCH_SIZE or cfg["dataloader"].get("val_batch_size",
                                                   cfg["dataloader"]["batch_size"]))
    nw = int(E.NUM_WORKERS) if E.NUM_WORKERS is not None \
        else int(cfg["dataloader"].get("val_num_workers", 4))
    dl = cfg["dataloader"]
    pin = bool(dl.get("val_pin_memory", dl.get("pin_memory", True))) and device.type == "cuda"
    layout = sc.task_layout(cfg)
    img_cfg = cfg["image"]
    W, H = int(img_cfg["width"]), int(img_cfg["height"])

    n_views = len(TTA_VIEWS)
    print(f"        TTA: {n_views} views to average — {ckpt_path.name}", flush=True)
    prob_sum = None
    for i, spec in enumerate(TTA_VIEWS, start=1):
        print(f"        [TTA view {i}/{n_views}] {spec.get('name', '?')} ...",
              end="", flush=True)
        clahe_override = spec.get("clahe", None)
        cfg_v = cfg
        if clahe_override is not None:
            cfg_v = copy.deepcopy(cfg)
            cfg_v["clahe"]["use_clahe"] = bool(clahe_override)
        transform = _make_transform(spec, W, H)

        ds = sc.CheXpertDataset(df, cfg_v, split="val")
        kwargs = dict(batch_size=bs, shuffle=False, drop_last=False,
                      num_workers=nw, pin_memory=pin)
        if nw > 0:
            kwargs["prefetch_factor"] = int(E.PREFETCH_FACTOR) if E.PREFETCH_FACTOR is not None \
                else int(dl.get("val_prefetch_factor", dl.get("prefetch_factor", 2)))
            kwargs["persistent_workers"] = bool(dl.get("val_persistent_workers", False))
        loader = DataLoader(ds, **kwargs)

        ps = []
        with torch.no_grad():
            for imgs, _labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                if transform is not None:
                    imgs = transform(imgs)
                imgs = sc.prepare_batch(imgs, cfg_v, device, False)
                logits = model(imgs)
                ps.append(sc.logits_to_probs(logits, layout).float().detach().cpu().numpy())
        p = np.concatenate(ps)
        prob_sum = p if prob_sum is None else prob_sum + p
        print(f" done ({p.shape[0]} samples)", flush=True)

    y_prob = prob_sum / float(n_views)
    print(f"        (TTA: all {n_views} views averaged)", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y_prob


def _out_subdir_tta(base: Path) -> Path:
    """ensample._out_subdir, but the folder name carries a `tta<N>` tag (+ RUN_TAG)
    so a TTA ensemble is self-describing and never overwrites a plain one."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"tta{len(TTA_VIEWS)}"
    if E.RUN_TAG:
        tag = f"{E.RUN_TAG}_{tag}"
    return base / "ensembling_results" / f"{stamp}_{tag}"


# ----------------------------- local execution -----------------------------
def run_local():
    E._predict = _tta_predict              # make the shared ensemble core score with TTA

    def load_cfg(run):
        return sc.load_config(PKG_ROOT / run, verbose=False)

    def ckpt_path_of(run, checkpoint=None):
        base = PKG_ROOT / run / "results" / "checkpoints"
        sub = E.CKPT_SUBPATH.get(run, "best.pt")
        if checkpoint is None:
            p = base / sub
            if not p.exists():
                raise FileNotFoundError(f"missing {p} — fetch this run's best.pt first "
                                        f"(or use RUN_ON='modal')")
            return p
        return sc._resolve_resume(checkpoint, base / Path(sub).parent)

    thr_json = PKG_ROOT / E.THRESHOLDS_FROM / "results" / "thresholds.json"
    out_dir = _out_subdir_tta(Path(__file__).resolve().parent)
    E.run_ensemble(load_cfg, ckpt_path_of, out_dir, thr_json)


# ----------------------------- modal execution -----------------------------
try:
    import modal
    _MODAL_OK = True
except ImportError:
    _MODAL_OK = False

# Build the app ONLY on the launching (local) side, reusing the image / volumes /
# resources ensample.py already built (imported above) so the mount layout stays
# byte-for-byte identical to a plain ensemble run.
if _MODAL_OK and modal.is_local():
    _ref_cfg = E._ref_cfg
    _runs_vol = E._runs_vol                  # captured by value into ensemble_remote (picklable)
    app = modal.App(f"ensemble-tta-{E.SET}")

    @app.function(
        image=E._image,
        volumes=E._volumes,
        serialized=True,
        **E._resources,
    )
    def ensemble_remote():
        # Self-contained: import the MOUNTED modules INSIDE (where shared_code is
        # importable) and patch ensample._predict -> the TTA predictor, so the
        # UNCHANGED ensample.run_ensemble scores every member with TTA.
        import sys as _sys
        from pathlib import Path as _P
        for _p in ("/root/training_scripts", "/root/training_scripts/others"):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        import shared_code as _sc
        import ensample as _E
        import ensample_tta as _T
        _E._predict = _T._tta_predict        # the ONLY change vs a plain ensemble
        runs_mount = _P("/runs")

        def load_cfg(run):
            return _sc.remote_cfg(_sc.load_config(_P("/root/training_scripts") / run, verbose=False))

        def ckpt_path_of(run, checkpoint=None):
            base = runs_mount / run / "results" / "checkpoints"
            sub = _E.CKPT_SUBPATH.get(run, "best.pt")
            if checkpoint is None:
                return base / sub
            return _sc._resolve_resume(checkpoint, base / _P(sub).parent)

        thr_json = runs_mount / _E.THRESHOLDS_FROM / "results" / "thresholds.json"
        out_dir = _T._out_subdir_tta(runs_mount)
        try:
            _E.run_ensemble(load_cfg, ckpt_path_of, out_dir, thr_json)
        finally:
            _runs_vol.commit()               # persist before the local fetch reads it
        return out_dir.relative_to(runs_mount).as_posix()


if __name__ == "__main__":
    if RUN_ON == "modal":
        if not _MODAL_OK:
            raise SystemExit("RUN_ON='modal' but modal isn't installed; set RUN_ON='local'.")
        with modal.enable_output():
            with app.run():
                remote_sub = ensemble_remote.remote()   # volume-relative POSIX subpath
        if remote_sub:
            E._fetch_results_from_modal(_ref_cfg["modal"]["runs_volume"], remote_sub,
                                        Path(__file__).resolve().parent)
    elif RUN_ON == "local":
        run_local()
    else:
        raise SystemExit(f"RUN_ON must be 'modal' or 'local', got {RUN_ON!r}")
