"""
topk_sweep.py  —  how many members should the ensemble hold?

Takes the runs ranked by frontal valid200, and for each K in KS scores a FLAT 1/K
ensemble of the top K on that same split. Flat, never fitted: the only thing chosen
on the validation set is the single integer K, which is about as little fitting as
an ensemble decision can involve.

test500 is not touched. Pick K here, then score it once on test500 separately.

Each K edits ensample.py's FULL_MODELS / SET / WEIGHTS and runs it. The first K
computes every member's probabilities; the rest hit the cache, so they are quick
and run locally.

Run:  python training_scripts/others/topk_sweep.py
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
KS = [6, 8]                       # ensemble sizes to compare
SET = "valid200"                  # scored split (FRONTAL_ONLY lives in ensample.py)
SPACES = ["logit", "prob", "rank"]  # blend spaces to compare, one pass each
MODE = "search"                   # "flat"   -> 1/K average, nothing fitted
                                  # "search" -> fit per-class weights per K, and
                                  #             report flat / weighted / OOF together

# Ranked by frontal valid200, best first. Single-task runs are excluded, and so is
# medmae_vitb_nih_B_448_s1, whose val19k/val200 numbers were never recorded.
RANKED = [
    "medmae_vitb_nih_B_768_s2_seed1337",   # 0.9105
    "medmae_vitb_nih_B_768_s2",            # 0.9064
    "convnext_base_22k_1600x1312",         # 0.9063
    "medmae_vitb_nih_B_448_s1_seed1337",   # 0.9052
    "medmae_vitb_nih_B_768_s2_seed7",      # 0.9045
    "medmae_vitb_raw",                     # 0.9030
    "rad_dino_vitB_768",                   # 0.9016
    "convnext_large_22k_768x640",          # 0.9009
    "rad_dino_vitB_1064x896",              # 0.9000
    "convnext_base_22k_768x640",           # 0.8994
    "medmae_vitb_nih_B_448_s1_seed7",      # 0.8972
    "medmae_vitb_nih",                     # 0.8970
]
# ===========================================================================

OTHERS = Path(__file__).resolve().parent
ENS = OTHERS / "ensample.py"
RESULTS = OTHERS / "ensembling_results"


def patch(members: list, space: str = "logit") -> None:
    """Point ensample.py at this K's member list, this blend space, the chosen split."""
    t = ENS.read_text(encoding="utf-8")
    block = "FULL_MODELS = [\n" + "".join(f'    "{m}",\n' for m in members) + "]\n"
    t, n = re.subn(r"^FULL_MODELS = \[.*?^\]\n", block, t, count=1,
                   flags=re.M | re.S)
    if not n:
        raise SystemExit("could not find the FULL_MODELS block")
    t = re.sub(r'^SET             = "[^"]*"', f'SET             = "{SET}"',
               t, count=1, flags=re.M)
    t = re.sub(r'^WEIGHTS         = "[^"]*"', f'WEIGHTS         = "{MODE}"',
               t, count=1, flags=re.M)
    t = re.sub(r'^COMBINE_SPACE   = "[^"]*"',
               f'COMBINE_SPACE   = "{space}"', t, count=1, flags=re.M)
    ast.parse(t)
    ENS.write_text(t, encoding="utf-8")


def newest_summary() -> Path:
    """The summary written by the run that just finished. A flat run writes
    ensemble_*, a search run writes weighted_*."""
    prefix = "weighted" if MODE == "search" else "ensemble"
    hits = sorted(RESULTS.glob(f"*/{prefix}_*_summary.json"),
                  key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no {prefix}_*_summary.json found")
    return hits[-1]


def main():
    original = ENS.read_text(encoding="utf-8")
    scores = []
    try:
        for space in SPACES:
            for k in KS:
                members = RANKED[:k]
                print("=" * 78)
                print(f"K={k}  space={space}: {', '.join(members)}")
                print("=" * 78, flush=True)
                patch(members, space)
                r = subprocess.run([sys.executable, "-u", str(ENS)],
                                   cwd=str(OTHERS.parent.parent))
                if r.returncode != 0:
                    print(f"K={k} space={space} FAILED (exit {r.returncode})")
                    scores.append(((k, space), None))
                    continue
                doc = json.loads(newest_summary().read_text(encoding="utf-8"))
                scores.append(((k, space), {
                    "flat": doc.get("flat_mean_auroc") or doc.get("ensemble_mean_auroc"),
                    "weighted": doc.get("weighted_mean_auroc"),
                    "oof": doc.get("oof_mean_auroc"),
                }))
                print(f"\nK={k} space={space}  {scores[-1][1]}\n", flush=True)
    finally:
        ENS.write_text(original, encoding="utf-8")   # leave the file as we found it
        print("ensample.py restored")

    def fmt(v):
        return "  --  " if v is None else f"{v:.4f}"

    print("\n" + "=" * 56)
    print(f"top-K by blend space on frontal {SET}   (mode={MODE})")
    print("=" * 56)
    print(f"  {'K':>3} {'space':>6}   {'flat':>6} {'weighted':>9} {'OOF':>7}")
    # OOF is the honest one: weighted_mean_auroc is scored by weights fitted on
    # these very images, so it is optimistic by construction
    ok = [(key, s) for key, s in scores if s and (s.get("oof") or s.get("flat"))]
    best_oof = max(ok, key=lambda p: p[1].get("oof") or -1, default=(None, {}))
    best_flat = max(ok, key=lambda p: p[1].get("flat") or -1, default=(None, {}))
    for key, s in scores:
        k, space = key
        if not s:
            print(f"  {k:>3} {space:>6}   failed")
            continue
        mark = ""
        if key == best_oof[0]:
            mark += "  <- best OOF"
        if key == best_flat[0]:
            mark += "  <- best flat"
        print(f"  {k:>3} {space:>6}   {fmt(s['flat'])} {fmt(s['weighted']):>9} "
              f"{fmt(s['oof']):>7}{mark}")


if __name__ == "__main__":
    main()
