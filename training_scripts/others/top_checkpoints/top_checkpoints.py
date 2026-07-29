"""
top_checkpoints.py
==================
Identify the TOP-N validation points of a run and map each one to the CLOSEST
checkpoint that was actually SAVED to disk.

WHY AN APPROXIMATION IS NEEDED
  Validation ran on one cadence (rad_dino_vitB_768: every 500 steps) but checkpoints
  were written on another (every 200 steps), and only ONE "best" weight file is kept —
  best.pt is repeatedly OVERWRITTEN, so the 2nd/3rd/... best validation points have no
  weight file of their own. The periodic ckpt_step<N>.pt files are therefore the only
  way to get near those points, and "near" is the honest word: a top validation at step
  4500 has no saved 4500 checkpoint (4500 is not a multiple of 200) — the closest saved
  states are 4400 and 4600, each 100 steps away. This script quantifies that gap for
  every entry in the top-N so you know exactly how good the approximation is.

  The single exception is the top-1: best.pt IS that exact step, gap 0, by definition.

RANKING METRIC
  METRIC below defaults to valid200_mean_auroc — the column this run actually monitored
  (training_summary.json: monitor tracks valid200, best_value 0.89292 at step 4500). The
  script cross-checks its own #1 against training_summary.json's best_step and warns
  loudly if they disagree (which would mean METRIC is not the monitored column).

DISTINCT
  Two top-N validation points can round to the same saved checkpoint. With DISTINCT=True
  the assignment is greedy in rank order — a taken checkpoint is skipped and the next
  nearest unused one is used instead — so N ranks yield N DISTINCT weight files, which is
  what you need if these are going to be ensemble members. The realised gap is reported
  either way, so a forced second choice is always visible.

Output: one folder per run, results/<YYYY-MM-DD_HH-MM-SS>_<RUN>/, holding
  top_checkpoints.txt   human-readable table + the gap breakdown
  top_checkpoints.json  the same, machine-readable
Nothing is overwritten, so every ranking ever produced (any RUN, any METRIC) is kept.

Run:  python training_scripts/others/top_checkpoints/top_checkpoints.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================ CONFIG (edit here) ============================
RUN = "convnext_base_22k_1600x1312"   # run whose val_log.csv is ranked

# Column of val_log.csv to rank by. This run monitored valid200, so that is the
# default; "mean_auroc" would rank by the ~19k in-training val split instead.
METRIC = "mean_auroc"
MODE = "max"                     # "max" (AUROC) | "min" (loss)

TOP_N = 10                       # how many validation points to report
MARK_AT = 5                      # draw the "top-5" line here (top-5 ⊂ top-10)

# Which ckpt_step<N>.pt files actually exist on the runs volume. Two ways:
#   AVAILABLE_STEPS = <list>  -> use exactly these steps (needed when the save cadence
#                                CHANGED mid-run, which no single grid can express);
#   AVAILABLE_STEPS = None    -> build range(CKPT_FIRST, CKPT_LAST+1, CKPT_EVERY).
# Either way the list is extended with any ckpt_step*.pt found locally.
# rad_dino_vitB_1064x896: a clean every-200 grid, all present. The run ended at
# step 13500 (early-stopped), so 13400 is the last saved checkpoint.
CKPT_FIRST, CKPT_LAST, CKPT_EVERY = 200, 13400, 200
AVAILABLE_STEPS = None
# medmae_vitb_nih was irregular — keep the recipe here for when you switch back:
#   AVAILABLE_STEPS = list(range(3300, 10801, 300)) + list(range(11000, 13601, 200))

# best.pt is the single kept copy of the best-monitored step; None -> read best_step
# from training_summary.json. Set to False if the run has no best.pt.
BEST_PT_STEP = None

DISTINCT = True                  # force N distinct checkpoints (see docstring)
TIE_PREFER = "earlier"           # equidistant candidates: "earlier" | "later"
# ===========================================================================

HERE = Path(__file__).resolve().parent               # others/top_checkpoints
PKG_ROOT = HERE.parent.parent                        # training_scripts/
RUN_DIR = PKG_ROOT / RUN
RESULTS = RUN_DIR / "results"
RESULTS_DIR = HERE / "results"   # each run gets its own <timestamp>_<RUN> folder


def _available_steps():
    """Sorted list of steps that have a saved ckpt_step<N>.pt, plus where it came from."""
    if AVAILABLE_STEPS is not None:
        s = sorted(set(AVAILABLE_STEPS))
        return s, (f"AVAILABLE_STEPS (explicit): {len(s)} checkpoints, "
                   f"{s[0]}..{s[-1]}")
    grid = list(range(CKPT_FIRST, CKPT_LAST + 1, CKPT_EVERY))
    found = sorted(int(p.stem.replace("ckpt_step", ""))
                   for p in (RESULTS / "checkpoints").glob("ckpt_step*.pt"))
    src = (f"grid {CKPT_FIRST}..{CKPT_LAST} every {CKPT_EVERY}"
           f" ({len(grid)} checkpoints)")
    if found:
        extra = sorted(set(found) - set(grid))
        src += f" + {len(found)} found locally" + (f" (extra: {extra})" if extra else "")
    return sorted(set(grid) | set(found)), src


def _spacing(steps):
    """Describe the gaps between consecutive saved checkpoints. The cadence can CHANGE
    mid-run, so this reports every distinct spacing with its count rather than assuming
    one interval. Returns (description, tightest spacing)."""
    from collections import Counter
    diffs = Counter(b - a for a, b in zip(steps, steps[1:]))
    if not diffs:
        return "single checkpoint", None
    return (", ".join(f"{d} steps x{n}" for d, n in sorted(diffs.items())), min(diffs))


def _best_pt_step(summary):
    if BEST_PT_STEP is False:
        return None
    if BEST_PT_STEP is not None:
        return int(BEST_PT_STEP)
    return int(summary["best_step"]) if summary and summary.get("best_step") is not None else None


def _nearest(target, candidates, taken):
    """Closest candidate step to `target`, skipping `taken` when DISTINCT.
    Ties (equidistant on both sides) resolve by TIE_PREFER. Returns
    (step, gap, tied_with) where gap = chosen - target."""
    pool = [s for s in candidates if not (DISTINCT and s in taken)]
    if not pool:
        return None, None, None
    best_d = min(abs(s - target) for s in pool)
    ties = sorted(s for s in pool if abs(s - target) == best_d)
    chosen = ties[0] if TIE_PREFER == "earlier" else ties[-1]
    other = [s for s in ties if s != chosen]
    return chosen, chosen - target, (other[0] if other else None)


def main():
    log_path = RESULTS / "val_log.csv"
    df = pd.read_csv(log_path)
    if METRIC not in df.columns:
        raise KeyError(f"'{METRIC}' not in {log_path.name}. available metric columns: "
                       + ", ".join(c for c in df.columns if "mean_" in c))
    df = df[["step", "epoch", METRIC]].dropna(subset=[METRIC])

    summary = None
    sp = RESULTS / "training_summary.json"
    if sp.exists():
        summary = json.load(open(sp, encoding="utf-8"))

    ranked = df.sort_values(METRIC, ascending=(MODE == "min")).head(TOP_N).reset_index(drop=True)
    steps, steps_src = _available_steps()
    space_desc, tight = _spacing(steps)
    best_step = _best_pt_step(summary)

    # --- sanity: our #1 must be the step training_summary calls the best ------------
    warns = []
    top1 = int(ranked.loc[0, "step"])
    if best_step is not None and top1 != best_step:
        warns.append(f"top-1 by '{METRIC}' is step {top1}, but training_summary.json says "
                     f"best_step={best_step}. METRIC is probably NOT the monitored column "
                     f"-> best.pt does NOT hold the #1 weights below.")
    if summary and summary.get("best_value") is not None and best_step == top1:
        d = abs(float(summary["best_value"]) - float(ranked.loc[0, METRIC]))
        if d > 1e-6:
            warns.append(f"top-1 {METRIC}={ranked.loc[0, METRIC]:.6f} differs from "
                         f"training_summary best_value={summary['best_value']:.6f} by {d:.2e}")

    # --- map each rank to a saved checkpoint ---------------------------------------
    rows, taken = [], set()
    top1_val = float(ranked.loc[0, METRIC])
    for i, r in ranked.iterrows():
        vstep, val = int(r["step"]), float(r[METRIC])
        if best_step is not None and vstep == best_step and "best.pt" not in taken:
            file_, gap, tie, exact = "best.pt", 0, None, True
            taken.add("best.pt")
            csteps = vstep
        else:
            csteps, gap, tie = _nearest(vstep, steps, taken)
            if csteps is None:
                raise RuntimeError(f"no checkpoint left to assign to rank {i+1} "
                                   f"(step {vstep}); set DISTINCT=False")
            taken.add(csteps)
            file_, exact = f"ckpt_step{csteps}.pt", gap == 0
        rows.append({
            "rank": i + 1, "val_step": vstep, "epoch": int(r["epoch"]),
            "metric": val, "delta_vs_top1": val - top1_val,
            "checkpoint": file_, "checkpoint_step": csteps,
            "gap_steps": gap, "exact": exact,
            "equidistant_alternative": (f"ckpt_step{tie}.pt" if tie is not None else None),
        })

    gaps = [abs(r["gap_steps"]) for r in rows]
    n_exact = sum(r["exact"] for r in rows)

    # --- report --------------------------------------------------------------------
    now = datetime.now()
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    # one folder per run: results/<YYYY-MM-DD_HH-MM-SS>_<RUN>/ — nothing is ever
    # overwritten, so every ranking that was ever produced stays on disk.
    out_dir = RESULTS_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{RUN}"
    L = []
    w = L.append
    w("=" * 96)
    w(f"TOP-{TOP_N} CHECKPOINTS  —  {RUN}")
    w("=" * 96)
    w(f"generated   : {ts}")
    w(f"ranked by   : {METRIC}  ({MODE}) from results/val_log.csv "
      f"({len(df)} validation points, every {int(df['step'].diff().mode()[0])} steps)")
    w(f"saved ckpts : {steps_src}"
      + (f"  +  best.pt @ step {best_step}" if best_step is not None else ""))
    w(f"  spacing   : {space_desc}")
    w(f"assignment  : closest saved checkpoint"
      + (", forced DISTINCT per rank" if DISTINCT else "")
      + f", ties -> {TIE_PREFER}")
    for m in warns:
        w(f"!! WARNING   : {m}")
    w("=" * 96)
    w("")
    w(f"  #   val step  ep   {METRIC[:18]:>18}   vs #1        checkpoint            gap")
    w("  " + "-" * 92)
    for r in rows:
        if r["rank"] == MARK_AT + 1:
            w("  " + "- " * 20 + f" top-{MARK_AT} ends here " + "- " * 20)
        gap_txt = ("exact" if r["gap_steps"] == 0
                   else f"{r['gap_steps']:+d} steps")
        w(f"  {r['rank']:>2}   {r['val_step']:>8}  {r['epoch']:>2}   {r['metric']:>18.6f}   "
          f"{r['delta_vs_top1']:+.4f}    {r['checkpoint']:<20} {gap_txt:>10}"
          + (f"   (tie with {r['equidistant_alternative']})"
             if r["equidistant_alternative"] else ""))
    w("  " + "-" * 92)
    w("")
    w("APPROXIMATION QUALITY")
    w(f"  exact matches      : {n_exact}/{len(rows)}  "
      f"(rank 1 is exact by definition — best.pt IS that step)")
    w(f"  gap 0 steps        : {sum(g == 0 for g in gaps)}")
    for g in sorted(set(gaps) - {0}):
        w(f"  gap {g:>3} steps      : {sum(x == g for x in gaps)}")
    w(f"  max |gap|          : {max(gaps)} steps"
      + (f"  ({max(gaps) / tight:.2f} x the tightest {tight}-step checkpoint interval)"
         if tight else ""))
    w(f"  ckpt spacing       : {space_desc}")
    w(f"  mean |gap|         : {sum(gaps) / len(gaps):.1f} steps")
    w(f"  worst-case drift   : {max(gaps)} steps is "
      f"{max(gaps) / int(df['step'].diff().mode()[0]):.2f} of one validation interval, "
      f"so the approximated weights sit strictly between two measured validations.")
    w("")
    w("COPY-PASTE  —  members for ensample.py's FULL_MODELS. Each ('<run>', <step>) is")
    w("one independent voter; wrap the whole block in [ ... ] to make them ONE averaged")
    w("member instead.")
    for n in sorted({MARK_AT, TOP_N}):
        w(f"  # top-{n}")
        for r in rows[:n]:
            c = "best" if r["checkpoint"] == "best.pt" else r["checkpoint_step"]
            w(f'      ("{RUN}", {json.dumps(c)}),'
              f'   # rank {r["rank"]}, {METRIC}={r["metric"]:.4f}'
              + ("" if r["exact"] else f", {r['gap_steps']:+d} steps"))
    w("=" * 96)

    report = "\n".join(L) + "\n"
    print(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "top_checkpoints.txt"
    out_json = out_dir / "top_checkpoints.json"
    out_txt.write_text(report, encoding="utf-8")
    out_json.write_text(json.dumps({
        "generated": ts, "run": RUN, "metric": METRIC, "mode": MODE,
        "top_n": TOP_N, "mark_at": MARK_AT,
        "val_log": str(log_path), "n_validation_points": int(len(df)),
        "checkpoint_grid": {"source": steps_src, "spacing": space_desc,
                            "tightest_spacing": tight, "n_steps": len(steps),
                            "first_step": steps[0], "last_step": steps[-1],
                            "best_pt_step": best_step},
        "distinct": DISTINCT, "tie_prefer": TIE_PREFER,
        "warnings": warns,
        "summary": {"n_exact": n_exact, "max_gap_steps": max(gaps),
                    "mean_abs_gap_steps": sum(gaps) / len(gaps)},
        "top": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote -> {out_txt}")
    print(f"wrote -> {out_json}")


if __name__ == "__main__":
    main()
