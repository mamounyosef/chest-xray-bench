"""
seed_noise_analysis.py
======================
Seed-to-seed variability of the headline metric — no retraining, no re-evaluation.
Reads only the EXISTING result JSONs on disk and reports, per config, the mean AUROC
across seeds with its sample std (ddof=1), on both validation sets. This quantifies
the run-to-run "noise floor" from random seeds alone (init + shuffling + augmentation
order), so real differences between models can be judged against it.

Configs (seeds default / 7 / …):
  convnext_base_22k : convnext_base_22k, _seed7, _seed1337
  densenet121       : densenet121, _seed7, _seed123

Sources per run (training_scripts/<run>/results/):
  training_summary.json  -> val19k mean AUROC (in-training monitored val set); flat
                            schema: best_metrics.mean_auroc (or best_value if best_metrics
                            is null); two-split schema: best_metrics.val19k.mean_auroc.
  valid200_results.json  -> val200 mean AUROC (radiologist 200-set): macro.mean_auroc.

Run:  python training_scripts/others/seed_noise_analysis/seed_noise_analysis.py
Writes:  seed_noise_analysis.txt  (next to this script)
"""

import json
import statistics
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent.parent      # training_scripts/
OUT_TXT = Path(__file__).resolve().parent / "seed_noise_analysis.txt"

CONFIGS = {
    "convnext_base_22k": ["convnext_base_22k", "convnext_base_22k_seed7",
                          "convnext_base_22k_seed1337"],
    "densenet121": ["densenet121", "densenet121_seed7", "densenet121_seed123"],
}


def _val19k_auroc(run: str) -> float:
    """Mean AUROC on the in-training validation set (val19k) from training_summary.json.
    Handles the flat (best_metrics.mean_auroc), two-split (best_metrics.val19k.mean_auroc),
    and null-best_metrics (fall back to best_value, which equals best_metrics.mean_auroc
    on the runs that carry both) cases."""
    d = json.load(open(PKG_ROOT / run / "results" / "training_summary.json", encoding="utf-8"))
    bm = d.get("best_metrics")
    if isinstance(bm, dict) and "val19k" in bm:
        return float(bm["val19k"]["mean_auroc"])
    if isinstance(bm, dict) and bm.get("mean_auroc") is not None:
        return float(bm["mean_auroc"])
    return float(d["best_value"])


def _val200_auroc(run: str) -> float:
    """Mean AUROC on the radiologist valid200 set from valid200_results.json."""
    d = json.load(open(PKG_ROOT / run / "results" / "valid200_results.json", encoding="utf-8"))
    return float(d["macro"]["mean_auroc"])


def _fmt_mean_std(vals) -> str:
    """mean ± sample std (ddof=1); std is 'n/a' for a single value."""
    m = statistics.fmean(vals)
    s_txt = f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "n/a"
    return f"{m:.4f} ± {s_txt}"


def main():
    lines = ["=" * 78,
             "SEED NOISE ANALYSIS — mean AUROC across seeds (std = sample, ddof=1)",
             "=" * 78]
    for cfg, runs in CONFIGS.items():
        v200 = [_val200_auroc(r) for r in runs]
        v19k = [_val19k_auroc(r) for r in runs]
        seeds = ["default"] + [r.split("_seed")[-1] for r in runs[1:]]
        lines.append(f"\n{cfg}   (n={len(runs)} seeds: {', '.join(seeds)})")
        lines.append(f"  runs: {', '.join(runs)}")
        lines.append(f"  val200  mean AUROC : {_fmt_mean_std(v200)}     "
                     f"[{', '.join(f'{x:.4f}' for x in v200)}]")
        lines.append(f"  val19k  mean AUROC : {_fmt_mean_std(v19k)}     "
                     f"[{', '.join(f'{x:.4f}' for x in v19k)}]")
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote -> {OUT_TXT}")


if __name__ == "__main__":
    main()
