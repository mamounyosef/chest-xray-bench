"""
plot_results.py  —  experiment: convnext_base_22k_final_stage2_edema
======================================================
Post-hoc plotting. Reads this experiment's results/train_log.csv and
results/val_log.csv and writes the training-curve PNGs to results/plots/.

CPU-only, no torch/GPU needed — just pandas + matplotlib. Safe to run while a
training run is still going (it just plots whatever is logged so far).

Run:  python plot_results.py
"""

import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on the emoji in the logs; force
# UTF-8 so prints never crash (same thing run_experiment does for training).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXP_DIR = Path(__file__).resolve().parent
PKG_ROOT = EXP_DIR.parent                       # training_scripts/ (holds shared_code.py)
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc                          # noqa: E402

cfg = sc.load_config(EXP_DIR, verbose=False)
results_dir = EXP_DIR / cfg["output"]["run_dir"]


if __name__ == "__main__":
    sc.make_plots(cfg, results_dir)
