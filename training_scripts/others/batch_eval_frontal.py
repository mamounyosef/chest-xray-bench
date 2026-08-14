"""
batch_eval_frontal.py  —  re-score many runs on the FRONTAL-ONLY subsets, in a
few containers instead of one container per run.

Why this exists
---------------
01_valid200.csv holds 234 rows (202 frontal + 32 lateral) and 01_test500.csv holds
668 rows (518 frontal + 150 lateral). The models never saw a lateral in training,
so this script rescores every run on the frontal rows alone and writes the result
next to the existing files, under a separate name, leaving the mixed-view numbers
untouched for comparison:

    results/valid200_frontal_results.json / .txt
    results/test500_frontal_results.json  / .txt

Efficiency
----------
One Modal app, SHARDS containers, each looping over its slice of the run list.
The image, the volume mounts and the CUDA context are paid for once per container
rather than once per run. All three data volumes are mounted, so a single
container can score a 384x320 run off /data and a 768x640 run off /data_native
without a second app.

Run:  python training_scripts/others/batch_eval_frontal.py
"""

import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
RUNS = None          # None -> every run that has a checkpoint; else a list of names
SKIP = {"densenet121_temp"}          # scratch runs never reported
EVAL_SETS = ("valid200", "test500")
CHECKPOINT = "best"
AMP = False          # fp32, to match how the reported numbers were produced
SUFFIX = "_frontal"  # output files: <set><SUFFIX>_results.json
TIMEOUT_S = 60 * 60 * 4

# Runs are split by input size, because that is what decides the memory a forward
# pass needs. Each tier gets its own GPU and its own number of containers, so the
# 1600x1312 model does not force everything else onto an expensive card.
#   (name, min megapixels, GPU, containers, per-batch images)
TIERS = [
    ("heavy", 0.60, "H200",       1,  32),  # 1600x1312, 1064x896
    ("mid",   0.15, "A100-80GB",  2, 128),  # 768x640, 784x644, 448x384
    ("light", 0.00, "A100",       4, 256),  # everything at 384x320
]
# ===========================================================================

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc          # noqa: E402

TORCHVISION = {"resnet50", "densenet121", "densenet201",
               "convnext_tiny", "convnext_small"}


def discover_runs() -> list:
    """Every experiment folder holding a usable checkpoint, with its input size."""
    out = []
    for d in sorted(p for p in PKG_ROOT.iterdir() if p.is_dir()):
        if d.name in ("others", "__pycache__") or d.name in SKIP:
            continue
        if not (d / "config.yaml").exists():
            continue
        if not list((d / "results" / "checkpoints").rglob("best.pt")):
            continue
        img = sc.load_config(d, verbose=False).get("image", {})
        out.append((d.name, img["width"] * img["height"] / 1e6))
    return out


def assign_tiers(runs: list) -> dict:
    """{tier name: [run names]}, each run in the first tier its size clears."""
    groups = {t[0]: [] for t in TIERS}
    for name, mp in runs:
        for tier, min_mp, *_ in TIERS:
            if mp >= min_mp:
                groups[tier].append(name)
                break
    return groups


def shard(items: list, n: int) -> list:
    """Round robin, so a slow run does not always land in the same shard."""
    parts = [items[i::n] for i in range(n)]
    return [p for p in parts if p]


# --------------------------------------------------------------------------
# The remote worker. Everything it needs is imported inside, so the function
# body ships by value and nothing from __main__ has to travel with it.
# --------------------------------------------------------------------------
def _score_one(run_name, eval_sets, checkpoint, amp, suffix, batch_size):
    """Score one run on the frontal rows of each set. Returns a small summary."""
    import json as _json
    import sys as _sys
    from datetime import datetime as _dt
    from pathlib import Path as _P

    import pandas as _pd
    import torch as _torch
    import torch.nn as _nn

    # the image mounts the source tree here; it is not on the path by default
    if "/root/training_scripts" not in _sys.path:
        _sys.path.insert(0, "/root/training_scripts")
    import shared_code as _sc

    exp_dir = _P("/root/training_scripts") / run_name
    cfg = _sc.load_config(exp_dir, verbose=False)
    rcfg = _sc.remote_cfg(cfg)

    # ---- architecture, keyed on the same fields the run's own train.py uses
    arch = str(cfg.get("model", {}).get("arch", "")).lower()
    name = cfg["model"]["name"]
    n_out = _sc.num_output_logits(rcfg)
    if arch == "raddino":
        model = _sc.build_raddino_vit(rcfg, load_pretrained=False)
    elif arch == "medmae_vitb" or name.startswith("vit_base_patch16"):
        model = _sc.build_medmae_vit(rcfg, load_pretrained=False)
    elif name in {"resnet50", "densenet121", "densenet201",
                  "convnext_tiny", "convnext_small"}:
        import torchvision as _tv
        model = getattr(_tv.models, name)(weights=None)
        if hasattr(model, "fc"):
            model.fc = _nn.Linear(model.fc.in_features, n_out)
        elif isinstance(getattr(model, "classifier", None), _nn.Linear):
            model.classifier = _nn.Linear(model.classifier.in_features, n_out)
        else:
            head = model.classifier[-1]
            model.classifier[-1] = _nn.Linear(head.in_features, n_out)
    else:
        import timm as _timm
        model = _timm.create_model(name, pretrained=False, num_classes=n_out)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    out_dir = _P("/runs") / run_name
    results_dir = out_dir / rcfg["output"]["run_dir"]
    ckpt_dir = results_dir / rcfg["output"]["checkpoints_dir"]
    sub = _sc.finetune_ckpt_subdir(rcfg)
    if sub:
        ckpt_dir = ckpt_dir / sub

    # best.pt when the volume has it, otherwise the stripped weights uploaded
    # alongside it. Both hold the same tensors; the .safetensors file is a third
    # of the size, which is why it is the one this script ships.
    st_path = ckpt_dir / "model.safetensors"
    if (ckpt_dir / "best.pt").exists():
        ckpt, ckpt_path = _sc._load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    elif st_path.exists():
        from safetensors.torch import load_file as _load_st
        missing, unexpected = model.load_state_dict(_load_st(str(st_path)),
                                                    strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{run_name}: state dict mismatch, "
                               f"missing={list(missing)[:4]} "
                               f"unexpected={list(unexpected)[:4]}")
        ckpt, ckpt_path = {}, st_path
    else:
        raise FileNotFoundError(f"{run_name}: no best.pt or model.safetensors "
                                f"under {ckpt_dir}")
    model = model.to(device).eval()

    # frozen thresholds only; never calibrate here, that is a separate concern
    thr_map, thr_path = _sc.load_thresholds(results_dir, rcfg)
    tasks = rcfg["tasks"]
    thr_source = str(thr_path)
    if thr_map is None:
        thr_map = {t: 0.5 for t in tasks}
        thr_source = "uncalibrated (0.5)"
    thr_vec = [thr_map[t] for t in tasks]

    loss_fn = _sc.build_loss(rcfg, device)
    data_dir = _P(rcfg["paths"]["data_dir"])
    csvs = {"valid200": rcfg["paths"]["valid200_csv"],
            "test500": rcfg["paths"]["test500_csv"]}
    # the tier's cap, not the run's own val_batch_size: those were tuned for the
    # card each run trained on, and this script deliberately uses cheaper GPUs
    bs = int(batch_size)
    # the runs' own worker counts were set for their training containers and
    # oversubscribe the cores these get, which torch warns about
    nw = min(int(rcfg["dataloader"].get("val_num_workers", 4)), 8)

    summary = {"run": run_name}
    for set_name in eval_sets:
        csv_path = data_dir / csvs[set_name]
        if not csv_path.exists():
            summary[set_name] = "csv missing"
            continue
        df = _pd.read_csv(csv_path)
        n_all = len(df)
        # the frontal filter: the view is in the filename for every CheXpert split,
        # and it is the only signal test500 carries (it has no Frontal/Lateral column)
        df = df[df["Path"].str.contains("frontal", case=False)].reset_index(drop=True)
        print(f"[{run_name}] {set_name}: {len(df)} frontal of {n_all} rows")

        y_true, y_prob, excl, val_loss = _sc._predict_dataframe(
            rcfg, model, df, device, loss_fn, amp, channels_last=False,
            batch_size=bs, num_workers=nw,
            progress_desc=f"{run_name}/{set_name}-frontal")
        metrics = _sc.compute_metrics(y_true, y_prob, tasks, threshold=thr_vec,
                                      exclude_mask=excl)
        # the key set _render_report_txt expects, plus the two frontal fields
        report = {
            "experiment": run_name,
            "set": set_name,
            "view_filter": "frontal only",
            "csv": csvs[set_name],
            "n_images": int(len(df)),
            "n_images_all_views": int(n_all),
            "checkpoint": str(ckpt_path),
            "checkpoint_step": ckpt.get("global_step"),
            "checkpoint_epoch": ckpt.get("epoch"),
            "device": str(device),
            "amp": bool(amp),
            "threshold_objective": "f1",
            "threshold_source": thr_source,
            "thresholds": {t: float(thr_map[t]) for t in tasks},
            "use_clahe": bool(rcfg["clahe"]["use_clahe"]),
            "u_policy": rcfg["labels"]["u_policy"],
            "val_loss": float(val_loss),
            "macro": metrics["macro"],
            "per_task": metrics["per_task"],
            "evaluated_at": _dt.now().isoformat(timespec="seconds"),
        }
        (results_dir / f"{set_name}{suffix}_results.json").write_text(
            _json.dumps(_sc._json_safe(report), indent=2), encoding="utf-8")
        (results_dir / f"{set_name}{suffix}_results.txt").write_text(
            _sc._render_report_txt(report) + "\n", encoding="utf-8")
        summary[set_name] = round(float(metrics["macro"]["mean_auroc"]), 4)
        print(f"[{run_name}] {set_name} frontal mean AUROC = {summary[set_name]}")
    return summary


def fetch_results(run_names, suffix=SUFFIX, eval_sets=EVAL_SETS):
    """Download each run's frontal result files into its own local results folder.

    training_scripts/<run>/results/<set><suffix>_results.{json,txt}
    """
    import modal

    vol = modal.Volume.from_name("chexpert-runs")
    got, missed = 0, []
    for name in run_names:
        dest = PKG_ROOT / name / "results"
        dest.mkdir(parents=True, exist_ok=True)
        for set_name in eval_sets:
            for ext in ("json", "txt"):
                remote = f"{name}/results/{set_name}{suffix}_results.{ext}"
                try:
                    data = b"".join(vol.read_file(remote))
                except Exception:
                    missed.append(remote)
                    continue
                (dest / f"{set_name}{suffix}_results.{ext}").write_bytes(data)
                got += 1
    print(f"\nfetched {got} files into training_scripts/<run>/results/")
    if missed:
        print(f"{len(missed)} not on the volume:")
        for m in missed[:10]:
            print("  ", m)
    return got


def main():
    import modal

    runs = RUNS or discover_runs()
    if len(sys.argv) > 1:                  # names on the command line win, for a
        wanted = set(sys.argv[1:])         # single run smoke test before the fleet
        runs = [r for r in runs if r[0] in wanted]
        unknown = wanted - {r[0] for r in runs}
        if unknown:
            raise SystemExit(f"unknown run(s): {sorted(unknown)}")
    groups = assign_tiers(runs)

    app = modal.App("chexpert-batch-eval-frontal")
    # Same layers as sc.modal_image(), built here because this app needs two extra
    # packages with different pip options, and add_local_dir has to come last.
    #   transformers : RAD-DINO's backbone is an HF Dinov2 built from its config
    #                  (the rad_dino runs pin >=4.45), installed WITH its deps
    #   libauc       : build_loss needs it for the AUC-M runs, --no-deps since
    #                  everything it wants is already installed above
    image = (modal.Image
             .debian_slim(python_version=f"{sys.version_info.major}."
                                         f"{sys.version_info.minor}")
             .pip_install(*sc._MODAL_PIP)
             .pip_install("transformers>=4.45")
             .pip_install("libauc", extra_options="--no-deps")
             .add_local_dir(str(sc.PKG_DIR), remote_path="/root/training_scripts",
                            ignore=["**/results/**", "**/train_config/**",
                                    "**/__pycache__/**"]))
    vols = {
        "/data": modal.Volume.from_name("chexpert-data", create_if_missing=True),
        "/data_native": modal.Volume.from_name("chexpert-native-data",
                                               create_if_missing=True),
        "/runs": modal.Volume.from_name("chexpert-runs", create_if_missing=True),
    }
    runs_vol = vols["/runs"]

    def make_worker(gpu, batch_size):
        """One Modal function per GPU tier. The body is identical; only the card
        and the batch size differ, so a tier can be resized without touching it."""
        @app.function(image=image, volumes=vols, gpu=gpu, timeout=TIMEOUT_S,
                      serialized=True, name=f"worker_{gpu.replace('-', '_')}")
        def worker(names):
            import traceback
            out = []
            for i, run_name in enumerate(names, 1):
                print("#" * 70)
                print(f"# [{i}/{len(names)}] {run_name}  (gpu={gpu}, bs={batch_size})")
                print("#" * 70)
                try:
                    out.append(_score_one(run_name, EVAL_SETS, CHECKPOINT, AMP,
                                          SUFFIX, batch_size))
                except Exception as exc:
                    traceback.print_exc()
                    out.append({"run": run_name, "error": repr(exc)})
                finally:
                    runs_vol.commit()
            return out
        return worker

    workers, plan = {}, []
    for tier, _min_mp, gpu, n_containers, batch_size in TIERS:
        names = groups[tier]
        if not names:
            continue
        workers[tier] = make_worker(gpu, batch_size)
        plan.append((tier, gpu, n_containers, names))
        print(f"{tier:6} {gpu:11} {len(names):2} runs over "
              f"{min(n_containers, len(names))} container(s), bs={batch_size}")
    print(f"total {sum(len(n) for _, _, _, n in plan)} runs, "
          f"sets={list(EVAL_SETS)}\n")

    results = []
    with modal.enable_output():
        with app.run():
            handles = [(tier, workers[tier].map(shard(names, n)))
                       for tier, _gpu, n, names in plan]
            for _tier, h in handles:
                results.extend(list(h))

    flat = [r for chunk in results for r in chunk]
    ok = [r for r in flat if "error" not in r]
    bad = [r for r in flat if "error" in r]

    fetch_results([r["run"] for r in ok])

    print(f"\n{len(ok)} scored, {len(bad)} failed")
    for r in sorted(ok, key=lambda r: r.get("test500") or 0, reverse=True):
        print(f"  {r['run']:46} valid200={r.get('valid200')}  "
              f"test500={r.get('test500')}")
    for r in bad:
        print(f"  FAILED {r['run']}: {r['error']}")


if __name__ == "__main__":
    main()
