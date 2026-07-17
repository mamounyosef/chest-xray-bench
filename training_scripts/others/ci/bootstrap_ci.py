"""
bootstrap_ci.py
===============
Bootstrap 95% confidence intervals for CheXpert mean-AUROC on a chosen split (set
SET below — e.g. the ~19k patient-grouped 10% val split, or the valid200 / test500
radiologist sets), to answer ONE question: is the ENSEMBLE genuinely better than the
best SINGLE model, or is the gap within noise?

It reads the per-image predicted probabilities + true labels straight from the
ensemble prob-cache (training_scripts/others/cache_runs/<set>/), so it never
loads a model or an image — nothing to recompute. Those .npy files are produced by
ensample.py (each member's (N,5) probabilities; _y_true.npy holds the (N,5) labels).

Procedure (paired bootstrap):
  - Resample the N rows WITH replacement, N_BOOT times. The SAME resampled indices
    are applied to every model each iteration (paired), so the difference is on the
    same patients.
  - Per iteration compute per-class AUROC and the mean across the 5 classes for the
    single model, the ensemble, and their difference (ensemble - single).
  - A class whose resample has only one label present (AUROC undefined) is set to
    NaN for that iteration and excluded from that iteration's mean (nanmean); its
    per-class CI then uses only the valid iterations.
  - Report median + 95% CI (2.5 / 97.5 pct) for each, the CI of the DIFFERENCE, and
    the fraction of iterations where ensemble > single (the key number) — overall
    and per class, so you can see which classes are resolvable at this sample size.

Output: a clean table printed to the console AND written to a fresh timestamped file
under ci/results/<timestamp>.txt (one file per run).

Run:  python training_scripts/others/ci/bootstrap_ci.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================ CONFIG (edit here) ============================
SET     = "val"                # scored split whose cache to read: "val" = the ~19k
                              # patient-grouped 10% val split (01_val.csv); also
                              # "valid200" / "test500" (radiologist sets)
N_BOOT  = 2000                 # bootstrap iterations
SEED    = 42                   # RNG seed (reproducible)
CI_LOW, CI_HIGH = 2.5, 97.5    # percentiles for the 95% CI

# The 5 competition tasks, in the column order of every cached (N,5) matrix.
TASKS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# Cache basenames (without .npy) under cache_runs/<SET>/. The ensemble is the
# equal-weight probability-average of ENSEMBLE_MEMBERS.
SINGLE_MEMBER    = "convnext_base_22k_1600x1312_best"
ENSEMBLE_MEMBERS = ["convnext_base_22k_1600x1312_best", "medmae_vitb_nih_B_768_s2_best"]
# ===========================================================================

CI_DIR    = Path(__file__).resolve().parent          # training_scripts/others/ci
CACHE_DIR = CI_DIR.parent / "cache_runs" / SET   # .../others/cache_runs/<set>
RESULTS_DIR = CI_DIR / "results"


def _load(name: str) -> np.ndarray:
    p = CACHE_DIR / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"missing cache file: {p}\n"
            f"    run ensample.py for SET='{SET}' with this member first so its "
            f"probabilities get cached.")
    return np.load(p)


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(C,) per-class AUROC; NaN for any class with <2 label values present."""
    C = p.shape[1]
    out = np.full(C, np.nan)
    for c in range(C):
        yc = y[:, c]
        if np.unique(yc).size >= 2:
            out[c] = roc_auc_score(yc, p[:, c])
    return out


def summarize(dist: np.ndarray):
    """(low, median, high) over a 1-D bootstrap distribution, ignoring NaNs."""
    return (np.nanpercentile(dist, CI_LOW),
            np.nanmedian(dist),
            np.nanpercentile(dist, CI_HIGH))


def frac_positive(diff: np.ndarray) -> float:
    """Fraction of finite iterations with diff > 0 (ensemble strictly beats single)."""
    finite = np.isfinite(diff)
    return float(np.mean(diff[finite] > 0)) if finite.any() else float("nan")


def main():
    # --- load probabilities + labels (all (N, 5)) -------------------------------
    y_true = _load("_y_true")
    single = _load(SINGLE_MEMBER)
    ens = np.mean([_load(m) for m in ENSEMBLE_MEMBERS], axis=0)   # prob-average
    N, C = y_true.shape
    assert single.shape == (N, C) and ens.shape == (N, C), "shape mismatch vs labels"

    # --- point estimates on the full set ---------------------------------------
    single_obs_pc = per_class_auroc(y_true, single)
    ens_obs_pc = per_class_auroc(y_true, ens)
    single_obs = np.nanmean(single_obs_pc)
    ens_obs = np.nanmean(ens_obs_pc)

    # --- paired bootstrap ------------------------------------------------------
    rng = np.random.default_rng(SEED)
    single_pc = np.empty((N_BOOT, C))     # per-class AUROC per iteration
    ens_pc = np.empty((N_BOOT, C))
    for b in range(N_BOOT):
        idx = rng.integers(0, N, N)       # resample rows WITH replacement
        yb = y_true[idx]
        single_pc[b] = per_class_auroc(yb, single[idx])
        ens_pc[b] = per_class_auroc(yb, ens[idx])   # SAME idx (paired)

    single_mean = np.nanmean(single_pc, axis=1)     # (N_BOOT,) mean-across-5 per iter
    ens_mean = np.nanmean(ens_pc, axis=1)
    diff_mean = ens_mean - single_mean
    diff_pc = ens_pc - single_pc                    # (N_BOOT, C)

    # --- build the report ------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    L = []
    def w(line=""):
        L.append(line)

    w("=" * 92)
    w(f"BOOTSTRAP 95% CI  —  mean AUROC on {SET} ({N} images)")
    w(f"generated : {ts}   |   iterations : {N_BOOT}   |   seed : {SEED}")
    w(f"single    : {SINGLE_MEMBER}")
    w(f"ensemble  : prob-average of {ENSEMBLE_MEMBERS}")
    w("=" * 92)

    # 1) headline: mean AUROC across the 5 classes
    s_lo, s_md, s_hi = summarize(single_mean)
    e_lo, e_md, e_hi = summarize(ens_mean)
    d_lo, d_md, d_hi = summarize(diff_mean)
    w("MEAN AUROC (across 5 classes)")
    w(f"  {'model':<20}{'point':>9}{'median':>9}{'2.5%':>9}{'97.5%':>9}")
    w(f"  {'single':<20}{single_obs:>9.4f}{s_md:>9.4f}{s_lo:>9.4f}{s_hi:>9.4f}")
    w(f"  {'ensemble':<20}{ens_obs:>9.4f}{e_md:>9.4f}{e_lo:>9.4f}{e_hi:>9.4f}")
    w("  " + "-" * 66)
    w(f"  {'diff (ens-single)':<20}{ens_obs - single_obs:>9.4f}{d_md:>9.4f}"
      f"{d_lo:>9.4f}{d_hi:>9.4f}")
    frac = frac_positive(diff_mean)
    verdict = ("CI excludes 0 -> ensemble genuinely better" if d_lo > 0 else
               "CI excludes 0 -> ensemble genuinely WORSE" if d_hi < 0 else
               "CI includes 0 -> within noise (not distinguishable)")
    w(f"  P(ensemble > single) = {frac:6.1%}     [{verdict}]")
    w("=" * 92)

    # 2) per-class breakdown (which classes are resolvable at this sample size)
    w("PER-CLASS AUROC  (median [2.5%, 97.5%]);  diff = ensemble - single")
    w(f"  {'class':<18}{'single':>22}{'ensemble':>22}{'diff (95% CI)':>24}{'P>0':>7}")
    for c, t in enumerate(TASKS):
        s_lo, s_md, s_hi = summarize(single_pc[:, c])
        e_lo, e_md, e_hi = summarize(ens_pc[:, c])
        dc_lo, dc_md, dc_hi = summarize(diff_pc[:, c])
        fc = frac_positive(diff_pc[:, c])
        n_bad = int(np.isnan(diff_pc[:, c]).sum())     # degenerate iterations
        s_str = f"{s_md:.3f}[{s_lo:.3f},{s_hi:.3f}]"
        e_str = f"{e_md:.3f}[{e_lo:.3f},{e_hi:.3f}]"
        d_str = f"{dc_md:+.3f}[{dc_lo:+.3f},{dc_hi:+.3f}]"
        w(f"  {t:<18}{s_str:>22}{e_str:>22}{d_str:>24}{fc:>7.0%}"
          + (f"   (NaN x{n_bad})" if n_bad else ""))
    w("=" * 92)
    w("notes: paired bootstrap (same resampled patients for both models each iter).")
    w("       a class with one label present in a resample -> NaN that iter (excluded")
    w("       from that iter's mean). 'P>0' = fraction of iters with ensemble>single.")
    w("=" * 92)

    report = "\n".join(L) + "\n"
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{ts}.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
