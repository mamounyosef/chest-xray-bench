"""
member_correlation_frontal.py  —  how different are the candidate members really?

Reads the cached (N, 5) probability matrices for the frontal valid200 split and
reports, for the top candidates:

  * the mean pairwise Spearman correlation of their predictions
  * a full correlation matrix
  * each member's mean correlation with the rest, which is what decides whether it
    brings anything an existing member does not

Ensembling pays when members disagree. Two runs of the same configuration under
different seeds tend to sit above 0.98, so counting them as separate members
overstates how diverse a set is.

Nothing is computed here: everything comes off the cache written by ensample.py.

Run:  python training_scripts/others/member_correlation_frontal.py
"""

import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

OTHERS = Path(__file__).resolve().parent
CACHE = OTHERS / "cache_runs" / "valid200_frontal"

TASKS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# ranked by frontal valid200, best first
MEMBERS = [
    "medmae_vitb_nih_B_768_s2_seed1337",
    "medmae_vitb_nih_B_768_s2",
    "convnext_base_22k_1600x1312",
    "medmae_vitb_nih_B_448_s1_seed1337",
    "medmae_vitb_nih_B_768_s2_seed7",
    "medmae_vitb_raw",
    "rad_dino_vitB_768",
    "convnext_large_22k_768x640",
    "rad_dino_vitB_1064x896",
]
SHORT = {
    "medmae_vitb_nih_B_768_s2_seed1337": "mae768-s1337",
    "medmae_vitb_nih_B_768_s2": "mae768",
    "convnext_base_22k_1600x1312": "cnxtB-1600",
    "medmae_vitb_nih_B_448_s1_seed1337": "mae448-s1337",
    "medmae_vitb_nih_B_768_s2_seed7": "mae768-s7",
    "medmae_vitb_raw": "mae-raw",
    "rad_dino_vitB_768": "dino-784",
    "convnext_large_22k_768x640": "cnxtL-768",
    "rad_dino_vitB_1064x896": "dino-1064",
}


def load(run: str) -> np.ndarray:
    p = CACHE / f"{run}_best.npy"
    if not p.exists():
        raise SystemExit(f"missing cache: {p}")
    return np.load(p)


def main():
    probs = {m: load(m) for m in MEMBERS}
    n = len(MEMBERS)

    # Spearman per task, averaged over the five tasks: rank correlation is the right
    # one here because AUROC only cares about ordering, not calibration
    corr = np.zeros((n, n))
    for i, a in enumerate(MEMBERS):
        for j, b in enumerate(MEMBERS):
            rs = [spearmanr(probs[a][:, t], probs[b][:, t]).statistic
                  for t in range(len(TASKS))]
            corr[i, j] = float(np.mean(rs))

    names = [SHORT[m] for m in MEMBERS]
    print("mean Spearman correlation over the 5 tasks, frontal valid200 "
          f"({probs[MEMBERS[0]].shape[0]} images)\n")
    print(f"{'':14}" + "".join(f"{x:>13}" for x in names))
    for i, row in enumerate(corr):
        print(f"{names[i]:14}" + "".join(
            f"{v:>13.3f}" if i != j else f"{'-':>13}"
            for j, v in enumerate(row)))

    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    print("\nmean correlation with every other member (lower = more to add):")
    for i, m in enumerate(MEMBERS):
        print(f"  {np.nanmean(off[i]):.3f}   {m}")

    print("\nclosest pairs:")
    pairs = sorted(((corr[i, j], names[i], names[j])
                    for i in range(n) for j in range(i + 1, n)), reverse=True)
    for v, a, b in pairs[:5]:
        print(f"  {v:.3f}   {a} ~ {b}")
    print("\nmost different pairs:")
    for v, a, b in pairs[-5:]:
        print(f"  {v:.3f}   {a} ~ {b}")


if __name__ == "__main__":
    main()
