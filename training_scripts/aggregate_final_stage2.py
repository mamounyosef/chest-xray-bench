"""
aggregate_final_stage2.py
=========================
Combine the FIVE per-disease Stage-2 AUC-M models into one final summary, the way
papers report it: each model is evaluated on the full split but only its OWN class
matters, so the headline numbers are the MEAN of the five per-class values.

For each split (valid200, test500) it pulls every model's own-class metrics from its
results JSON, then averages across the five diseases:
    mean AUROC / AUPRC          (threshold-free)
    mean F1 / precision / recall / specificity   (at each model's own threshold)

Writes (into this folder):
    final_stage2_summary.json   <- machine-readable
    final_stage2_summary.txt    <- readable twin

Run:  python training_scripts/aggregate_final_stage2.py
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# disease -> its Stage-2 run folder (each model predicts ONLY this class)
RUNS = {
    "Atelectasis":      "convnext_base_22k_final_stage2_atelectasis",
    "Cardiomegaly":     "convnext_base_22k_final_stage2_cardiomegaly",
    "Consolidation":    "convnext_base_22k_final_stage2_consolidation",
    "Edema":            "convnext_base_22k_final_stage2_edema",
    "Pleural Effusion": "convnext_base_22k_final_stage2_pleural_effusion",
}
SPLITS = ["valid200", "test500"]
METRICS = ["auroc", "auprc", "f1", "precision", "recall", "specificity"]


def _load_class_metrics(run: str, split: str, disease: str) -> dict:
    """Read one model's results JSON and return ITS OWN class's metric row."""
    f = HERE / run / "results" / f"{split}_results.json"
    if not f.exists():
        raise FileNotFoundError(f"missing {f} — run evaluate.py for {run} first")
    d = json.load(open(f, encoding="utf-8"))
    pt = d.get("per_task", {})
    if disease not in pt:
        raise KeyError(f"{disease!r} not in per_task of {f} (keys={list(pt)})")
    row = dict(pt[disease])
    row["threshold"] = row.get("threshold")
    row["checkpoint_step"] = d.get("checkpoint_step")
    return row


def _aggregate_split(split: str) -> dict:
    per_class = {}
    for disease, run in RUNS.items():
        per_class[disease] = _load_class_metrics(run, split, disease)
    mean = {f"mean_{m}": statistics.fmean(per_class[d][m] for d in RUNS)
            for m in METRICS}
    return {"set": split, "n_models": len(RUNS),
            "mean": mean, "per_class": per_class}


def _render_txt(summary: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("FINAL Stage-2 summary  —  5 per-disease AUC-M models (mean over the 5)")
    lines.append(f"generated: {summary['generated']}")
    lines.append("=" * 78)
    for split in SPLITS:
        s = summary["splits"][split]
        m = s["mean"]
        lines.append("")
        lines.append(f"[{split}]  ({s['n_models']} models)")
        lines.append(f"  mean AUROC={m['mean_auroc']:.4f}  AUPRC={m['mean_auprc']:.4f}  "
                     f"F1={m['mean_f1']:.4f}  P={m['mean_precision']:.4f}  "
                     f"R={m['mean_recall']:.4f}  Spec={m['mean_specificity']:.4f}")
        lines.append("  " + "-" * 74)
        lines.append(f"  {'disease':<18} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} "
                     f"{'Prec':>7} {'Recall':>7} {'Spec':>7} {'thr':>7}")
        for disease in RUNS:
            r = s["per_class"][disease]
            lines.append(f"  {disease:<18} {r['auroc']:>7.4f} {r['auprc']:>7.4f} "
                         f"{r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} "
                         f"{r['specificity']:>7.4f} {(r.get('threshold') or 0):>7.4f}")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def main():
    summary = {"generated": datetime.now().isoformat(timespec="seconds"),
               "runs": RUNS, "splits": {}}
    for split in SPLITS:
        try:
            summary["splits"][split] = _aggregate_split(split)
        except FileNotFoundError as e:
            print(f"[skip {split}] {e}")
    if not summary["splits"]:
        raise SystemExit("no results found — evaluate the 5 Stage-2 runs first.")

    out_json = HERE / "final_stage2_summary.json"
    out_txt = HERE / "final_stage2_summary.txt"
    json.dump(summary, open(out_json, "w", encoding="utf-8"), indent=2)
    txt = _render_txt(summary)
    open(out_txt, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
