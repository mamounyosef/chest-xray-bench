"""
bootstrap_ci.py
===============
Bootstrap 95% confidence intervals for the FINAL per-class-WEIGHTED ensemble,
computed independently on valid200 (234 images) and test500 (668 images).
No retraining, no re-evaluation: it reads the per-image probabilities + labels straight
from the ensemble prob-cache (training_scripts/others/cache_runs/<set>/), which
ensample.py already wrote.

Procedure (paired, patient-level bootstrap), per set:
  - Each iteration draws N row indices WITH replacement (N = 234 on valid200, 668 on
    test500). The SAME indices are applied to ALL members before blending, so every
    member sees exactly the same resampled patients — the resample is of PATIENTS, not
    of predictions.
  - The members are blended in COMBINE_SPACE with the FROZEN per-class weights read
    from WEIGHTS_FROM (an ensample.py summary json). Weights are fixed — they are
    NOT re-fit inside the bootstrap (re-fitting would leak and inflate the CI).
  - Per iteration: per-class AUROC + the mean across the 5 classes.
  - A class whose resample contains only one label value has an undefined AUROC -> NaN
    for that iteration; it is dropped from that iteration's mean (nanmean) and from
    that class's percentiles.
  - Reported: the POINT ESTIMATE on the full set (not the bootstrap median) together
    with the 2.5 / 97.5 percentile CI, overall and per class.

PAIRED_SINGLE (optional) additionally bootstraps the best single member on the SAME
resampled indices and reports the CI of the DIFFERENCE (ensemble - single), i.e.
whether the ensemble's gain survives resampling noise.

COMBINE_SPACE must match the ensample.py run being reported, or the point estimate
here will not equal the one in that run's summary. "prob" and "logit" are supported;
"rank" is not (a bootstrap resample changes the ranks themselves).

Output: one folder per run, results/<YYYY-MM-DD_HH-MM-SS>_<set>/  (the set(s) in SETS,
joined by '+' if more than one), holding
        bootstrap_ci.json  (machine-readable)
        bootstrap_ci.txt   (the printed summary)

Run:  python training_scripts/others/ci/bootstrap_ci.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================ CONFIG (edit here) ============================
SETS = ["test500_frontal"]   # scored splits, each bootstrapped SEPARATELY

# Frozen per-class weights: path to an ensample.py summary json (or its folder) —
# either a weighted_*_summary.json (a fresh fit) or an ensemble_*_summary.json (a run
# that reused a saved fit). This is the 6-member fit itself, searched on valid200 —
# the same weights ensample.py currently loads, and the ones behind the 0.9130 test500
# run. They never saw test500.
WEIGHTS_FROM = "../ensembling_results/2026-08-15_11-50-49"

# Members, as the labels used in the weights file (a best.pt member is just the run
# name; a checkpoint member is "<run> @ step<N>"). None -> take them from the weights
# file itself, in its stored order.
MEMBERS = None

COMBINE_SPACE = "logit"          # "prob" | "logit" — MUST match the ensample.py run

N_BOOT = 10000                   # bootstrap iterations
SEED = 42                        # RNG seed (reproducible)
CI_LOW, CI_HIGH = 2.5, 97.5      # percentiles for the 95% CI

# Optional paired comparison: bootstrap this single member on the SAME indices and
# report the CI of (ensemble - single). None -> skip.
PAIRED_SINGLE = "medmae_vitb_nih_B_768_s2_seed1337"   # best single member on frontal test500 (0.9113)

# The 5 competition tasks, in the column order of every cached (N,5) matrix.
TASKS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
# ===========================================================================

CI_DIR = Path(__file__).resolve().parent             # training_scripts/others/ci
CACHE_ROOT = CI_DIR.parent / "cache_runs"            # .../others/cache_runs
RESULTS_DIR = CI_DIR / "results"     # each run gets its own <timestamp>_<set> folder


# --------------------------------------------------------------------------- cache
def _cache_stem(label: str) -> str:
    """Cache basename for a member label, matching ensample.py's naming:
    'run' -> 'run_best';  'run @ step7500' -> 'run_ckpt_step7500'."""
    if " @ " not in label:
        return f"{label}_best"
    run, ckpt = label.split(" @ ", 1)
    ckpt = ckpt.strip()
    if ckpt.startswith("step") and ckpt[4:].isdigit():
        return f"{run}_ckpt_step{int(ckpt[4:])}"
    if ckpt in ("best", ""):
        return f"{run}_best"
    return f"{run}_{ckpt}"


def _load(set_name: str, stem: str) -> np.ndarray:
    p = CACHE_ROOT / set_name / f"{stem}.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"missing cache file: {p}\n"
            f"    run ensample.py for SET='{set_name}' with this member first so its "
            f"probabilities get cached.")
    return np.load(p)


def _load_weights():
    """-> (weights_per_class dict, member label order, resolved source path)."""
    p = Path(WEIGHTS_FROM)
    if not p.is_absolute():
        p = (CI_DIR / p).resolve()
    if p.is_dir():
        cands = sorted(p.glob("weighted_*_summary.json")) + \
                sorted(p.glob("ensemble_*_summary.json"))
        if not cands:
            raise FileNotFoundError(
                f"no weighted_*_summary.json / ensemble_*_summary.json under {p}")
        p = cands[0]
    d = json.load(open(p, encoding="utf-8"))
    # a fresh fit stores "weights_per_class"; a run that REUSED a fit stores the
    # resolved weights as "class_weights" — both are {task: {member: weight}}
    wpc = d.get("weights_per_class") or d.get("class_weights")
    if wpc is None:
        raise KeyError(f"{p} has neither 'weights_per_class' nor 'class_weights'")
    if d.get("combine_space") and d["combine_space"] != COMBINE_SPACE:
        raise ValueError(
            f"COMBINE_SPACE={COMBINE_SPACE!r} but {p.name} was produced with "
            f"{d['combine_space']!r} — the point estimate would not match that run.")
    labels = list(MEMBERS) if MEMBERS else list(d["members"])
    for t in TASKS:
        if t not in wpc:
            raise KeyError(f"weights file has no class '{t}'")
        missing = [m for m in labels if m not in wpc[t]]
        if missing:
            raise KeyError(f"weights file has no weight for {missing} on '{t}'")
    return wpc, labels, p


def _to_space(stack: np.ndarray) -> np.ndarray:
    """Probabilities -> COMBINE_SPACE, per member and per class (matches ensample.py)."""
    if COMBINE_SPACE == "prob":
        return stack
    if COMBINE_SPACE == "logit":
        eps = 1e-6
        p = np.clip(stack, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))
    raise ValueError(f"COMBINE_SPACE must be 'prob' | 'logit', got {COMBINE_SPACE!r}")


def _from_space(ens: np.ndarray) -> np.ndarray:
    """Blended score -> [0,1]. AUROC is invariant under this monotone map; it is applied
    so the cached scores stay comparable to ensample.py's."""
    return 1.0 / (1.0 + np.exp(-ens)) if COMBINE_SPACE == "logit" else ens


def _weighted_blend(stack: np.ndarray, labels, wpc) -> np.ndarray:
    """(M,N,T) member probabilities -> (N,T) per-class weighted average, taken in
    COMBINE_SPACE. Weights are renormalized per class over the members present; an
    all-zero class falls back to the uniform average."""
    M, N, T = stack.shape
    S = _to_space(stack)
    out = np.empty((N, T))
    for c, t in enumerate(TASKS):
        w = np.array([float(wpc[t][m]) for m in labels])
        w = w / w.sum() if w.sum() > 0 else np.full(M, 1.0 / M)
        out[:, c] = np.tensordot(w, S[:, :, c], axes=(0, 0))
    return _from_space(out)


# --------------------------------------------------------------------------- metric
def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via the rank (Mann-Whitney U) identity — identical to
    sklearn.roc_auc_score including tie handling, but fast enough for 10k iterations.
    NaN when only one label value is present."""
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([_auroc(y[:, c], p[:, c]) for c in range(p.shape[1])])


def _ci(dist: np.ndarray):
    """(2.5%, 97.5%) over a 1-D bootstrap distribution, ignoring NaNs."""
    return (float(np.nanpercentile(dist, CI_LOW)),
            float(np.nanpercentile(dist, CI_HIGH)))


# --------------------------------------------------------------------------- core
def bootstrap_set(set_name, wpc, labels, single_label):
    y_true = _load(set_name, "_y_true")
    N, C = y_true.shape
    stack = np.stack([_load(set_name, _cache_stem(m)) for m in labels], axis=0)
    if stack.shape[1:] != (N, C):
        raise ValueError(f"{set_name}: member shape {stack.shape[1:]} != labels {(N, C)}")
    ens = _weighted_blend(stack, labels, wpc)
    single = _load(set_name, _cache_stem(single_label)) if single_label else None

    # point estimates on the FULL set (cross-checked against sklearn once)
    ens_pt_pc = per_class_auroc(y_true, ens)
    ref = np.array([roc_auc_score(y_true[:, c], ens[:, c]) for c in range(C)])
    assert np.allclose(ens_pt_pc, ref, atol=1e-12), "fast AUROC != sklearn"
    ens_pt = float(np.nanmean(ens_pt_pc))
    sgl_pt_pc = per_class_auroc(y_true, single) if single is not None else None
    sgl_pt = float(np.nanmean(sgl_pt_pc)) if single is not None else None

    # paired bootstrap: one index draw per iteration, shared by every member
    rng = np.random.default_rng(SEED)
    ens_pc = np.empty((N_BOOT, C))
    sgl_pc = np.empty((N_BOOT, C)) if single is not None else None
    for b in range(N_BOOT):
        idx = rng.integers(0, N, N)          # resample PATIENTS with replacement
        yb = y_true[idx]
        ens_pc[b] = per_class_auroc(yb, ens[idx])
        if single is not None:
            sgl_pc[b] = per_class_auroc(yb, single[idx])
    ens_mean = np.nanmean(ens_pc, axis=1)

    res = {
        "set": set_name, "n_images": int(N), "n_boot": N_BOOT, "seed": SEED,
        "combine_space": COMBINE_SPACE, "blend": "weighted (per-class, frozen)",
        "members": list(labels),
        "weights_per_class": {t: {m: float(wpc[t][m]) for m in labels} for t in TASKS},
        "ensemble": {
            "mean_auroc": {"point": ens_pt,
                           "ci_low": _ci(ens_mean)[0], "ci_high": _ci(ens_mean)[1],
                           "boot_median": float(np.nanmedian(ens_mean)),
                           "boot_std": float(np.nanstd(ens_mean, ddof=1))},
            "per_class": {},
        },
    }
    for c, t in enumerate(TASKS):
        lo, hi = _ci(ens_pc[:, c])
        res["ensemble"]["per_class"][t] = {
            "point": float(ens_pt_pc[c]), "ci_low": lo, "ci_high": hi,
            "boot_median": float(np.nanmedian(ens_pc[:, c])),
            "boot_std": float(np.nanstd(ens_pc[:, c], ddof=1)),
            "n_degenerate_iters": int(np.isnan(ens_pc[:, c]).sum()),
        }

    if single is not None:
        sgl_mean = np.nanmean(sgl_pc, axis=1)
        d_mean = ens_mean - sgl_mean
        d_pc = ens_pc - sgl_pc
        fin = np.isfinite(d_mean)
        lo, hi = _ci(d_mean)
        res["paired_single"] = {
            "member": single_label,
            "mean_auroc": {"point": sgl_pt,
                           "ci_low": _ci(sgl_mean)[0], "ci_high": _ci(sgl_mean)[1]},
            "per_class": {t: dict(zip(("point", "ci_low", "ci_high"),
                                      (float(sgl_pt_pc[c]), *_ci(sgl_pc[:, c]))))
                          for c, t in enumerate(TASKS)},
            "diff_mean_auroc": {
                "point": ens_pt - sgl_pt, "ci_low": lo, "ci_high": hi,
                "p_ensemble_better": float(np.mean(d_mean[fin] > 0)) if fin.any() else None,
            },
            "diff_per_class": {},
        }
        for c, t in enumerate(TASKS):
            dlo, dhi = _ci(d_pc[:, c])
            f = np.isfinite(d_pc[:, c])
            res["paired_single"]["diff_per_class"][t] = {
                "point": float(ens_pt_pc[c] - sgl_pt_pc[c]), "ci_low": dlo, "ci_high": dhi,
                "p_ensemble_better": float(np.mean(d_pc[f, c] > 0)) if f.any() else None,
            }
    return res


# --------------------------------------------------------------------------- report
def format_report(all_res, weights_path, ts):
    L = []
    w = L.append
    w("=" * 96)
    w(f"BOOTSTRAP 95% CI  —  {len(all_res[0]['members'])}-member per-class WEIGHTED "
      f"ensemble ({COMBINE_SPACE}-average)")
    w(f"generated : {ts}   |   iterations : {N_BOOT}   |   seed : {SEED}")
    w(f"members   : {', '.join(all_res[0]['members'])}")
    w(f"weights   : {weights_path}  (FROZEN — not re-fit inside the bootstrap)")
    w("=" * 96)
    for r in all_res:
        e = r["ensemble"]["mean_auroc"]
        w("")
        w(f"[{r['set']}]  N={r['n_images']} images")
        w(f"  MEAN AUROC : {e['point']:.4f}   95% CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]"
          f"   (width {e['ci_high'] - e['ci_low']:.4f}, boot sd {e['boot_std']:.4f})")
        w(f"  {'class':<18}{'point':>9}   {'95% CI':^18}{'width':>9}")
        for t in TASKS:
            d = r["ensemble"]["per_class"][t]
            w(f"  {t:<18}{d['point']:>9.4f}   [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
              f"{d['ci_high'] - d['ci_low']:>9.4f}"
              + (f"   (NaN x{d['n_degenerate_iters']})" if d["n_degenerate_iters"] else ""))
        if "paired_single" in r:
            ps = r["paired_single"]
            dm = ps["diff_mean_auroc"]
            verdict = ("CI excludes 0 -> ensemble genuinely better" if dm["ci_low"] > 0 else
                       "CI excludes 0 -> ensemble genuinely WORSE" if dm["ci_high"] < 0 else
                       "CI includes 0 -> within noise")
            w("  " + "-" * 90)
            w(f"  vs single '{ps['member']}' (same resampled patients):")
            w(f"    single   mean AUROC : {ps['mean_auroc']['point']:.4f}"
              f"   95% CI [{ps['mean_auroc']['ci_low']:.4f}, {ps['mean_auroc']['ci_high']:.4f}]")
            w(f"    diff (ens - single) : {dm['point']:+.4f}"
              f"   95% CI [{dm['ci_low']:+.4f}, {dm['ci_high']:+.4f}]"
              f"   P(ens>single)={dm['p_ensemble_better']:.1%}")
            w(f"    -> {verdict}")
            for t in TASKS:
                d = ps["diff_per_class"][t]
                w(f"      {t:<18}{d['point']:>+8.4f}   [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
                  f"   P>0={d['p_ensemble_better']:.0%}")
    w("")
    w("=" * 96)
    w("notes: each iteration resamples image rows with replacement and applies the SAME")
    w("       rows to every member before blending (paired / patient-level). Weights are")
    w("       frozen. A class with one label value in a resample -> NaN that iteration.")
    w("=" * 96)
    return "\n".join(L) + "\n"


def main():
    wpc, labels, weights_path = _load_weights()
    now = datetime.now()
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    # one folder per run: results/<YYYY-MM-DD_HH-MM-SS>_<set>/  (sets joined by '+'
    # if SETS holds more than one), containing bootstrap_ci.json + bootstrap_ci.txt
    out_dir = RESULTS_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{'+'.join(SETS)}"
    print(f"[boot] members: {labels}")
    print(f"[boot] weights: {weights_path}")
    all_res = []
    for s in SETS:
        print(f"[boot] bootstrapping {s} ({N_BOOT} iters) ...", flush=True)
        all_res.append(bootstrap_set(s, wpc, labels, PAIRED_SINGLE))

    report = format_report(all_res, weights_path, ts)
    print()
    print(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "bootstrap_ci.json"
    out_txt = out_dir / "bootstrap_ci.txt"
    out_json.write_text(json.dumps({
        "generated": ts, "n_boot": N_BOOT, "seed": SEED,
        "ci_percentiles": [CI_LOW, CI_HIGH],
        "combine_space": COMBINE_SPACE,
        "weights_source": str(weights_path),
        "tasks": TASKS,
        "sets": {r["set"]: r for r in all_res},
    }, indent=2), encoding="utf-8")
    out_txt.write_text(report, encoding="utf-8")
    print(f"wrote -> {out_json}")
    print(f"wrote -> {out_txt}")


if __name__ == "__main__":
    main()
