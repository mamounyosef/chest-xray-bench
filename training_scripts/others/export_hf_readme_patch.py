"""
export_hf_readme_patch.py  —  update the numbers in the LIVE Hugging Face card.

The card on the Hub has been edited by hand since it was generated, so this patches
the file in place rather than regenerating it: the front matter, the title and the
section layout are left exactly as they are, and only the scores, the split sizes and
the ensemble paragraphs change.

Input  : D:\\chexpert-bench-hf\\README_live.md, downloaded from the Hub
Output : the same file, rewritten

Run:  python training_scripts/others/export_hf_readme_patch.py
"""

import json
import re
from pathlib import Path

LIVE = Path(r"D:\chexpert-bench-hf\README_live.md")
RUNS = Path(__file__).resolve().parent.parent

ENSEMBLE = ["convnext_base_22k_1600x1312", "medmae_vitb_nih_B_768_s2",
            "rad_dino_vitB_768"]
BLENDED = 0.9174
BEST_SINGLE = "medmae_vitb_nih_B_768_s2_seed1337"
BEST_SINGLE_SCORE = 0.9113


def frontal(run: str, split: str):
    p = RUNS / run / "results" / f"{split}_frontal_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["macro"]["mean_auroc"]


def geometry(run: str):
    """Input size and parameter count, taken from the published config."""
    p = Path(r"D:\chexpert-bench-hf") / run / "config.json"
    if not p.exists():
        return "", ""
    c = json.loads(p.read_text(encoding="utf-8"))
    return (f"{c['image']['width']}x{c['image']['height']}",
            f"{c['n_parameters'] / 1e6:.1f}M")


def patch_table_rows(text: str) -> tuple:
    """Rewrite the last two score cells of every row that names a run."""
    row = re.compile(r"^\| \[`([^`]+)`\]\(\./\1\) \|(.*)\|\s*$", re.M)
    changed = []

    def repl(m):
        run, rest = m.group(1), m.group(2)
        v2, t5 = frontal(run, "valid200"), frontal(run, "test500")
        if v2 is None and t5 is None:
            return m.group(0)
        cells = rest.split("|")
        if len(cells) < 2:
            return m.group(0)
        bold = "**" if "**" in cells[-1] else ""
        cells[-2] = f" {v2:.4f} " if v2 is not None else " -- "
        cells[-1] = (f" {bold}{t5:.4f}{bold} " if t5 is not None else " -- ")
        changed.append(run)
        return f"| [`{run}`](./{run}) |" + "|".join(cells) + "|"

    return row.sub(repl, text), changed


def main():
    t = LIVE.read_text(encoding="utf-8")

    # --- the ensemble member table and its lead sentence
    rows = []
    for run in ENSEMBLE:
        v2, t5 = frontal(run, "valid200"), frontal(run, "test500")
        geo, par = geometry(run)
        rows.append(f"| [`{run}`](./{run}) | {v2:.4f} | **{t5:.4f}** | {geo} | {par} |")
    rows.append(f"| **All three blended** | | **{BLENDED:.4f}** | | |")

    block = (
        "Blended together these three score "
        f"**{BLENDED:.4f} mean AUROC on test500**, the best result in the study.\n\n"
        "| Model | valid200 | test500 | Input size | Params |\n"
        "|---|:---:|:---:|:---:|---:|\n" + "\n".join(rows) + "\n"
    )
    t, n = re.subn(
        r"Blended together these six score.*?\| \*\*All six blended\*\* \| \| \*\*0\.9130\*\* \| \| \|\n",
        block, t, count=1, flags=re.S)
    print(f"ensemble table replaced: {n}")

    # --- how the ensemble was combined
    combined = (
        f"The report's headline, **{BLENDED:.4f} mean AUROC** on the official test "
        "set, is a plain 1/3 probability average of the three models above. Fitting "
        "per finding weights on the validation split did not improve on it.\n\n"
        f"That is **+0.0061** over the best single model, "
        f"[`{BEST_SINGLE}`](./{BEST_SINGLE}) at {BEST_SINGLE_SCORE:.4f}, with a 95% "
        "bootstrap interval of **[+0.0005, +0.0118]** over 10,000 resamples. The "
        "lesson from the report: *diversity beats count*. Seven runs of the same "
        "backbone averaged to 0.9095, below the best single model, while three "
        "genuinely different backbones reached 0.9174.\n"
    )
    t, n = re.subn(
        r"The report's headline, \*\*0\.9130 mean AUROC\*\*.*?reached 0\.9115\.\n",
        combined, t, count=1, flags=re.S)
    print(f"combination paragraph replaced: {n}")

    # --- the "All models" tables only. The row patcher rewrites the last two cells,
    # which are the scores there but are Input size and Params in the member table
    # above, so that part of the document must be left alone.
    head, sep, tail = t.partition("## All models")
    if not sep:
        raise SystemExit("could not find the All models heading")
    tail, changed = patch_table_rows(tail)
    t = head + sep + tail
    print(f"table rows updated: {len(changed)}")

    # --- split sizes, now that the lateral rows are out
    t, n1 = re.subn(r"234 and 668 frontal images", "202 and 518 frontal images", t)
    t, n2 = re.subn(r"\*\*The test sets are small\.\*\* 234 and 668 images",
                    "**The test sets are small.** 202 and 518 images", t)
    t, n3 = re.subn(r"the 234 image validation split",
                    "the 202 image validation split", t)
    print(f"split sizes updated: {n1 + n2 + n3}")

    LIVE.write_text(t, encoding="utf-8")
    left = re.findall(r"0\.9130|0\.9027|\+0\.0103|0\.0022, \+0\.0186|\b668\b|\b234\b", t)
    print(f"\nwritten. stale markers still present: {sorted(set(left)) or 'none'}")


if __name__ == "__main__":
    main()
