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
            radiologist set — and, ONLY when that file is absent, the run's
            training_summary.json in-training best snapshot. The post-eval file is
            preferred because it is the authoritative, uniformly-sourced number; the
            summary is the fallback so recent runs (which record the set inside
            best_metrics) still appear instead of showing n/a. A value that came from
            the fallback is marked with FALLBACK_MARK, so the two sources are never
            silently mixed in one column. Neither available -> n/a.
            NOTE the summary keys that set by IMAGE COUNT — "val234" for the 234-image
            valid200 — so the extractor maps any "val<N>" (no trailing k) onto val200,
            and "val<N>k" onto val19k, rather than matching a hardcoded name.
The extractor understands both training_summary schemas for the "summary" source, so
any split can be repointed to the summary via SPLIT_SOURCE if desired.

Single-task runs (the per-class "_only" arms and the per-disease Stage-2 arms) are
OMITTED: their mean_auroc is a SINGLE class, not the 5-task mean, so ranking them
against full 5-task runs would be misleading. They are listed under an "omitted"
footer for transparency.

Output (overwritten each run):
    training_scripts/others/auroc_summary_across_runs.txt

Run:  python training_scripts/others/summarize_auroc_across_runs.py
"""

import json
import re as _re
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent          # training_scripts/others (outputs here)
RUNS_DIR = OUT_DIR.parent                            # training_scripts (run folders here)
OUT = OUT_DIR / "auroc_summary_across_runs.txt"

# ============================ CONFIG (edit here) ============================
# Which splits to rank on. A table is emitted for EACH, in this order. Drop or add
# entries freely (each must be a key of SPLIT_SOURCE below).
SPLITS = ["val19k", "val200"]

# Where each split's mean AUROC comes from. A LIST is a fallback chain, tried in
# order until one yields a value:
#   "summary"          -> training_summary.json (extracted by the two-schema rule).
#   "results:<file>"   -> results/<file>, ("macro","mean_auroc").
SPLIT_SOURCE = {
    "val19k":  "summary",                          # no post-eval file exists for the 19k val
    "val200":  ["results:valid200_results.json",   # canonical post-eval on the 200-study set
                "summary"],                        # else the in-training best snapshot
    "test500": "results:test500_results.json",
}

# Mark values that came from a FALLBACK source (not the first one listed), so a table
# never silently mixes post-eval numbers with in-training best snapshots. "" -> off.
FALLBACK_MARK = "*"
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
    """{split: mean_auroc} read from one training_summary snapshot, handling BOTH the
    new two-split best_metrics and the old/flat schema (see the module docstring).

    In the new schema best_metrics is keyed by each subset's SIZE label, which the
    training engine derives from that subset's image count — so the radiologist set
    shows up as "val234" (234 images) in one run and could be "val200" in another,
    while the primary val is "val19k". Both are normalized to the canonical split
    names here: a "val<N>k" key -> val19k, a "val<N>" key -> val200 (the sub-1000
    subset IS the radiologist set; the smallest wins if a run ever has several)."""
    bm = snap.get("best_metrics")
    if not isinstance(bm, dict):
        return {}
    if "monitored" in bm:                                    # --- new schema ---
        out, small = {}, []
        for k, v in bm.items():
            if k == "monitored" or not isinstance(v, dict):
                continue
            auroc = v.get("mean_auroc")
            m = _re.fullmatch(r"val(\d+)(k?)", k)
            if m and m.group(2):                             # val19k -> the primary val
                out["val19k"] = auroc
            elif m:                                          # val234 / val200 -> radiologist
                small.append((int(m.group(1)), auroc))
            else:
                out[k] = auroc                               # anything else: pass through
        if small:
            out["val200"] = min(small)[1]
        return out
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


def _run_scores(run_dir: Path):
    """({split: mean_auroc-or-None}, {split: source-label}) for every split in SPLITS.

    SPLIT_SOURCE may name a single source or a FALLBACK CHAIN: each is tried in order
    and the first that yields a value wins, so e.g. val200 prefers the post-eval
    results/valid200_results.json and only reads training_summary.json when that file
    isn't there. The source actually used is returned alongside, so the table can mark
    values that did not come from the preferred source."""
    summ = {}
    f = run_dir / "results" / "training_summary.json"
    if f.exists():
        try:
            summ = _summary_scores(_pick_snap(json.load(open(f, encoding="utf-8"))))
        except Exception:
            summ = {}

    def _from(src, split):
        if src == "summary":
            return summ.get(split)
        if src.startswith("results:"):
            return _results_auroc(run_dir / "results" / src.split(":", 1)[1])
        return None

    scores, used = {}, {}
    for split in SPLITS:
        chain = SPLIT_SOURCE.get(split, "summary")
        chain = [chain] if isinstance(chain, str) else list(chain)
        scores[split], used[split] = None, None
        for i, src in enumerate(chain):
            v = _from(src, split)
            if v is not None:
                scores[split], used[split] = v, (src, i)
                break
    return scores, used


def _mark(used, split) -> str:
    """FALLBACK_MARK when this split's value came from a fallback source, else ''."""
    u = (used or {}).get(split)
    return FALLBACK_MARK if (u and u[1] > 0) else ""


def _fmt(v, used=None, split=None):
    if v is None:
        return "n/a"
    return f"{v:.4f}{_mark(used, split)}"


def _table(primary: str, rows):
    """Return the text lines of one table sorted best-first on `primary`, with runs
    that lack a `primary` value parked at the bottom under an explicit marker. Every
    row also shows the OTHER splits in parentheses. A value taken from a FALLBACK
    source carries FALLBACK_MARK."""
    others = [s for s in SPLITS if s != primary]

    def _paren(sc, us):
        return ", ".join(f"{o}={_fmt(sc.get(o), us, o)}" for o in others)

    have = sorted((r for r in rows if r[1].get(primary) is not None),
                  key=lambda r: r[1][primary], reverse=True)
    miss = sorted((r for r in rows if r[1].get(primary) is None), key=lambda r: r[0])

    width = len(str(len(have))) if have else 1        # rank-number column width
    lines = ["=" * 88,
             f"Ranked by  {primary}   (mean AUROC, best first)",
             "=" * 88]
    for i, (name, sc, us) in enumerate(have, start=1):
        val = f"{sc[primary]:.4f}{_mark(us, primary)}"
        lines.append(f"  {i:>{width}}.  {val:<7}  {name:<46}  ({_paren(sc, us)})")
    if miss:
        lines.append(f"  ---- no {primary} value recorded (listed for completeness) ----")
        pad = " " * (width + 1)                        # align under the rank column
        for name, sc, us in miss:
            lines.append(f"  {pad} n/a      {name:<46}  ({_paren(sc, us)})")
    return lines


def main():
    rows, omitted = [], []                  # rows: (name, scores, used); omitted: names
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        if run_dir.name in EXCLUDE:
            omitted.append(run_dir.name)
            continue
        # a run with no results/ dir at all isn't a training run — skip silently.
        if not (run_dir / "results").is_dir():
            continue
        sc, us = _run_scores(run_dir)
        rows.append((run_dir.name, sc, us))

    lines = ["#" * 88,
             "Cross-run mean AUROC leaderboard   (source: training_summary.json + results/*.json)",
             f"generated: {datetime.now().isoformat(timespec='seconds')}",
             f"splits: {', '.join(SPLITS)}    runs listed: {len(rows)}"]
    for s in SPLITS:
        n = sum(1 for _, sc, _ in rows if sc.get(s) is not None)
        nb = sum(1 for _, _, us in rows if (us.get(s) or (None, 0))[1] > 0)
        extra = f"  ({nb} via fallback{FALLBACK_MARK})" if nb else ""
        lines.append(f"  - {s}: {n} run(s) have a value{extra}")
    # spell out each split's source chain, so a marked value is self-explanatory
    for s in SPLITS:
        chain = SPLIT_SOURCE.get(s, "summary")
        chain = [chain] if isinstance(chain, str) else list(chain)
        if len(chain) > 1:
            lines.append(f"  - {s} source: {chain[0]}, else "
                         + ", else ".join(chain[1:])
                         + f"   ({FALLBACK_MARK} = not the first source)")
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
