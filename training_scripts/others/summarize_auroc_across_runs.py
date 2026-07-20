"""
summarize_auroc_across_runs.py
==============================
ONE unified cross-run mean-AUROC leaderboard. Pick which validation/test splits to
rank on (SPLITS below); the script emits ONE table per chosen split into ONE txt
file, each table sorted best->worst by THAT split, and each row also shows the run's
score on the OTHER chosen splits in parentheses. Runs that have no value for a
table's split are NOT dropped — they are listed at the BOTTOM of that table, marked
"n/a", so a missing number is always stated explicitly rather than silently absent.

Choose splits by editing SPLITS. Each split's number comes from the source in
SPLIT_SOURCE:
  * "summary"  -> the run's results/training_summary.json best snapshot.
  * "results:<file>" -> results/<file> ("macro","mean_auroc") from a post-training
                        evaluation (e.g. test500_results.json).

--- where each split's number comes from (see SPLIT_SOURCE) -------------------
  val19k  : training_summary.json — best_metrics.mean_auroc. In the OLD/flat schema
            this is ALWAYS the primary ~19k val (that never changed in the old
            engine); in the NEW two-split schema it is best_metrics["val19k"]
            ["mean_auroc"]. There is no post-eval file for the 19k val, so the
            in-training best snapshot is the only source. Staged runs are read from
            their finetune/chexpert stage.
  val200  : results/valid200_results.json — the CANONICAL post-training eval on the
            200-study radiologist set (what the old summarize_val_auroc script read).
            Deliberately NOT taken from training_summary.best_value: for the runs that
            monitored valid200 the summary's best_value ~= this file anyway, but the
            post-eval file is the authoritative, uniformly-sourced number. A run
            without this file simply has no val200 here (-> listed as n/a).
The extractor still understands both training_summary schemas for the "summary"
source, so any split can be repointed to the summary via SPLIT_SOURCE if desired.

Single-task runs (the per-class "_only" arms and the per-disease Stage-2 arms) are
OMITTED: their mean_auroc is a SINGLE class, not the 5-task mean, so ranking them
against full 5-task runs would be misleading. They are listed under an "omitted"
footer for transparency.

Output (overwritten each run):
    training_scripts/others/auroc_summary_across_runs.txt

Run:  python training_scripts/others/summarize_auroc_across_runs.py
"""

import json
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent          # training_scripts/others (outputs here)
RUNS_DIR = OUT_DIR.parent                            # training_scripts (run folders here)
OUT = OUT_DIR / "auroc_summary_across_runs.txt"

# ============================ CONFIG (edit here) ============================
# Which splits to rank on. A table is emitted for EACH, in this order. Drop or add
# entries freely (each must be a key of SPLIT_SOURCE below).
SPLITS = ["val19k", "val200"]

# Where each split's mean AUROC comes from:
#   "summary"          -> training_summary.json (extracted by the two-schema rule).
#   "results:<file>"   -> results/<file>, ("macro","mean_auroc").
SPLIT_SOURCE = {
    "val19k":  "summary",                         # no post-eval file exists for the 19k val
    "val200":  "results:valid200_results.json",   # canonical post-eval on the 200-study set
    "test500": "results:test500_results.json",
}
# ===========================================================================

# single-task runs — omitted from the tables (mean_auroc is one class, not the
# 5-task mean); listed in an "omitted" footer instead.
EXCLUDE = {
    "convnext_base_22k_final_stage2_atelectasis",
    "convnext_base_22k_final_stage2_cardiomegaly",
    "convnext_base_22k_final_stage2_consolidation",
    "convnext_base_22k_final_stage2_edema",
    "convnext_base_22k_final_stage2_pleural_effusion",
    "convnext_base_22k_atelectasis_only",
    "convnext_base_22k_cardiomegaly_only",
    "convnext_base_22k_consolidation_only",
    "convnext_base_22k_edema_only",
    "convnext_base_22k_pleural_effusion_only",
}


def _pick_snap(doc: dict) -> dict:
    """The snapshot to score. Staged runs nest per-stage snapshots under
    doc['stages']; the CheXpert number is the finetune/chexpert stage (else the last
    stage). Single-stage runs return the doc itself."""
    stages = doc.get("stages")
    if isinstance(stages, dict) and stages:
        for tag, snap in stages.items():
            if "finetune" in tag or "chexpert" in tag:
                return snap
        return list(stages.values())[-1]
    return doc


def _summary_scores(snap: dict) -> dict:
    """{split_label: mean_auroc} read from one training_summary snapshot, handling
    BOTH the new two-split best_metrics and the old/flat schema (see module docstring
    for the val19k / val200 rule)."""
    bm = snap.get("best_metrics")
    if not isinstance(bm, dict):
        return {}
    if "monitored" in bm:                                    # --- new schema ---
        return {k: v.get("mean_auroc")
                for k, v in bm.items()
                if k != "monitored" and isinstance(v, dict)}
    # --- old / flat schema ---
    out = {"val19k": bm.get("mean_auroc")}                   # always the primary 19k val
    bv, ma = snap.get("best_value"), bm.get("mean_auroc")
    if (snap.get("monitor") == "val_mean_auroc" and bv is not None
            and ma is not None and abs(bv - ma) > 1e-9):
        out["val200"] = bv                                   # monitored valid200
    return out


def _results_auroc(path: Path):
    """('macro','mean_auroc') from a post-training <set>_results.json, or None."""
    if not path.exists():
        return None
    try:
        return json.load(open(path, encoding="utf-8"))["macro"]["mean_auroc"]
    except Exception:
        return None


def _run_scores(run_dir: Path) -> dict:
    """{split: mean_auroc-or-None} for every split in SPLITS."""
    summ = {}
    f = run_dir / "results" / "training_summary.json"
    if f.exists():
        try:
            summ = _summary_scores(_pick_snap(json.load(open(f, encoding="utf-8"))))
        except Exception:
            summ = {}
    scores = {}
    for split in SPLITS:
        src = SPLIT_SOURCE.get(split, "summary")
        if src == "summary":
            scores[split] = summ.get(split)
        elif src.startswith("results:"):
            scores[split] = _results_auroc(run_dir / "results" / src.split(":", 1)[1])
        else:
            scores[split] = None
    return scores


def _fmt(v):
    return f"{v:.4f}" if v is not None else "n/a"


def _table(primary: str, rows):
    """Return the text lines of one table sorted best-first on `primary`, with runs
    that lack a `primary` value parked at the bottom under an explicit marker. Every
    row also shows the OTHER splits in parentheses."""
    others = [s for s in SPLITS if s != primary]

    def _paren(sc):
        return ", ".join(f"{o}={_fmt(sc.get(o))}" for o in others)

    have = sorted((r for r in rows if r[1].get(primary) is not None),
                  key=lambda r: r[1][primary], reverse=True)
    miss = sorted((r for r in rows if r[1].get(primary) is None), key=lambda r: r[0])

    width = len(str(len(have))) if have else 1        # rank-number column width
    lines = ["=" * 88,
             f"Ranked by  {primary}   (mean AUROC, best first)",
             "=" * 88]
    for i, (name, sc) in enumerate(have, start=1):
        lines.append(f"  {i:>{width}}.  {sc[primary]:.4f}   {name:<46}  ({_paren(sc)})")
    if miss:
        lines.append(f"  ---- no {primary} value recorded (listed for completeness) ----")
        pad = " " * (width + 1)                        # align under the rank column
        for name, sc in miss:
            lines.append(f"  {pad} n/a      {name:<46}  ({_paren(sc)})")
    return lines


def main():
    rows, omitted = [], []                       # rows: (name, scores); omitted: names
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        if run_dir.name in EXCLUDE:
            omitted.append(run_dir.name)
            continue
        # a run with no results/ dir at all isn't a training run — skip silently.
        if not (run_dir / "results").is_dir():
            continue
        rows.append((run_dir.name, _run_scores(run_dir)))

    lines = ["#" * 88,
             "Cross-run mean AUROC leaderboard   (source: training_summary.json + results/*.json)",
             f"generated: {datetime.now().isoformat(timespec='seconds')}",
             f"splits: {', '.join(SPLITS)}    runs listed: {len(rows)}"]
    for s in SPLITS:
        n = sum(1 for _, sc in rows if sc.get(s) is not None)
        lines.append(f"  - {s}: {n} run(s) have a value")
    lines.append("#" * 88)
    lines.append("")

    for i, split in enumerate(SPLITS):
        lines += _table(split, rows)
        lines.append("")

    if omitted:
        lines.append("=" * 88)
        lines.append(f"omitted (single-task runs - single-class AUROC, not the 5-task mean): "
                     f"{len(omitted)}")
        lines.append("=" * 88)
        for name in sorted(omitted):
            lines.append(f"  - {name}")
    txt = "\n".join(lines) + "\n"

    OUT.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
