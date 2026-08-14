"""
compare_frontal.py  —  mixed-view vs frontal-only scores, side by side.

Reads each run's <set>_results.json (all views, as originally scored) and
<set>_results_frontal / <set>_frontal_results.json, and reports the change plus
whether the ranking of the runs moved.

Run:  python training_scripts/others/compare_frontal.py
"""

import json
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
SETS = ("valid200", "test500")


def score(run: Path, set_name: str, suffix: str = ""):
    f = run / "results" / f"{set_name}{suffix}_results.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["macro"]["mean_auroc"]


def main():
    rows = []
    for d in sorted(p for p in PKG_ROOT.iterdir() if p.is_dir()):
        if d.name in ("others", "__pycache__"):
            continue
        rec = {"run": d.name}
        for s in SETS:
            rec[f"{s}_all"] = score(d, s)
            rec[f"{s}_frontal"] = score(d, s, "_frontal")
        if rec.get("test500_frontal") is not None:
            rows.append(rec)

    print(f"{'run':46} {'valid200':>19}   {'test500':>19}")
    print(f"{'':46} {'all':>7}{'frontal':>8}{'Δ':>7}   "
          f"{'all':>7}{'frontal':>8}{'Δ':>7}")
    print("-" * 92)
    deltas = {s: [] for s in SETS}
    for r in sorted(rows, key=lambda r: -(r["test500_frontal"] or 0)):
        line = f"{r['run']:46}"
        for s in SETS:
            a, f = r[f"{s}_all"], r[f"{s}_frontal"]
            if a is None:
                line += f" {'--':>7}{f:>8.4f}{'':>7}  "
            else:
                d = f - a
                deltas[s].append(d)
                line += f" {a:>7.4f}{f:>8.4f}{d:>+7.4f}  "
        print(line)

    print("-" * 92)
    for s in SETS:
        v = deltas[s]
        if v:
            print(f"{s}: mean {sum(v)/len(v):+.4f}   min {min(v):+.4f}   "
                  f"max {max(v):+.4f}   n={len(v)}")

    # did the ordering move?
    for s in SETS:
        pairs = [(r["run"], r[f"{s}_all"], r[f"{s}_frontal"]) for r in rows
                 if r[f"{s}_all"] is not None]
        by_all = [p[0] for p in sorted(pairs, key=lambda p: -p[1])]
        by_fro = [p[0] for p in sorted(pairs, key=lambda p: -p[2])]
        moved = sum(1 for a, b in zip(by_all, by_fro) if a != b)
        print(f"{s}: {moved} of {len(by_all)} positions changed;  "
              f"top all-views = {by_all[0]};  top frontal = {by_fro[0]}")


if __name__ == "__main__":
    main()
