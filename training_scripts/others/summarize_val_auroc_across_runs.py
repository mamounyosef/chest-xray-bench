"""
summarize_val_auroc_across_runs.py
==================================
Scan every run folder under training_scripts/ and print each run's mean AUROC on
the validation set (valid200), into ONE txt file that is REWRITTEN every run.

The five Stage-2 per-disease AUC-M runs are NOT listed individually — instead a
single combined row (the mean over the five) is taken from
final_stage2_summary.json (produced by aggregate_final_stage2.py).

Output (overwritten each run):
    training_scripts/others/val_auroc_summary_across_runs.txt

Run:  python training_scripts/others/summarize_val_auroc_across_runs.py
"""

import json
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent          # training_scripts/others (outputs here)
RUNS_DIR = OUT_DIR.parent                            # training_scripts (run folders here)
SET = "valid200"                      # validation set to read (valid200 | test500)
OUT = OUT_DIR / "val_auroc_summary_across_runs.txt"
STAGE2_SUMMARY = OUT_DIR / "final_stage2_summary.json"

# the five per-disease Stage-2 runs are collapsed into ONE combined row, not listed
STAGE2_RUNS = {
    "convnext_base_22k_final_stage2_atelectasis",
    "convnext_base_22k_final_stage2_cardiomegaly",
    "convnext_base_22k_final_stage2_consolidation",
    "convnext_base_22k_final_stage2_edema",
    "convnext_base_22k_final_stage2_pleural_effusion",
}


def _run_auroc(run_dir: Path):
    """mean AUROC for one run from its <SET>_results.json (or None if missing)."""
    f = run_dir / "results" / f"{SET}_results.json"
    if not f.exists():
        return None
    try:
        return json.load(open(f, encoding="utf-8"))["macro"]["mean_auroc"]
    except Exception:
        return None


def _stage2_combined():
    """combined mean AUROC over the 5 Stage-2 models from the aggregate summary."""
    if not STAGE2_SUMMARY.exists():
        return None
    try:
        d = json.load(open(STAGE2_SUMMARY, encoding="utf-8"))
        return d["splits"][SET]["mean"]["mean_auroc"]
    except Exception:
        return None


def main():
    rows = []                                    # (name, auroc)
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        if run_dir.name in STAGE2_RUNS:
            continue                             # collapsed below, not listed here
        auroc = _run_auroc(run_dir)
        if auroc is not None:
            rows.append((run_dir.name, auroc))

    s2 = _stage2_combined()
    if s2 is not None:
        rows.append(("final_stage2 (mean of 5 per-disease models)", s2))

    rows.sort(key=lambda r: r[1], reverse=True)  # best first

    lines = []
    lines.append("=" * 78)
    lines.append(f"Mean AUROC per run  —  set={SET}")
    lines.append(f"generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 78)
    if not rows:
        lines.append(f"(no {SET}_results.json found in any run folder)")
    for name, auroc in rows:
        lines.append(f"  {auroc:.4f}   {name}")
    lines.append("=" * 78)
    txt = "\n".join(lines) + "\n"

    OUT.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
