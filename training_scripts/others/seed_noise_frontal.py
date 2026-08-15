"""
seed_noise_frontal.py  —  the run-to-run noise floor, on the frontal splits.

Every number in the report is read against this: a difference smaller than the
spread seen when the SAME configuration is retrained under a different seed is not
a difference. Recomputed here on the frontal-only scores, since the mixed-view ones
included 150 lateral test images no model was trained for.

Run:  python training_scripts/others/seed_noise_frontal.py
"""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent

# each group is one configuration trained under different seeds
GROUPS = {
    "ConvNeXt-B 384x320": [
        ("convnext_base_22k_final_stage1", 42),
        ("convnext_base_22k_seed7", 7),
        ("convnext_base_22k_seed1337", 1337),
    ],
    "DenseNet-121 384x320": [
        ("densenet121", 42),
        ("densenet121_seed7", 7),
        ("densenet121_seed123", 123),
    ],
    "Medical-MAE ViT-B 768x640": [
        ("medmae_vitb_nih_B_768_s2", 42),
        ("medmae_vitb_nih_B_768_s2_seed7", 7),
        ("medmae_vitb_nih_B_768_s2_seed1337", 1337),
    ],
    "Medical-MAE ViT-B 448x384": [
        ("medmae_vitb_nih_B_448_s1_seed7", 7),
        ("medmae_vitb_nih_B_448_s1_seed1337", 1337),
    ],
}

SETS = [("valid200_frontal", "valid200_frontal_results.json"),
        ("test500_frontal", "test500_frontal_results.json"),
        ("valid200 (all views)", "valid200_results.json"),
        ("test500 (all views)", "test500_results.json")]


def score(run: str, fname: str):
    p = RUNS / run / "results" / fname
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["macro"]["mean_auroc"]


def main():
    spreads = {name: [] for name, _ in SETS}
    for title, members in GROUPS.items():
        print(f"\n{title}")
        for set_name, fname in SETS:
            vals = [(seed, score(run, fname)) for run, seed in members]
            got = [v for _, v in vals if v is not None]
            if len(got) < 2:
                print(f"  {set_name:22} not enough values")
                continue
            spread = max(got) - min(got)
            spreads[set_name].append(spread)
            cells = "  ".join(f"seed{s}={v:.4f}" if v is not None else f"seed{s}=--"
                              for s, v in vals)
            print(f"  {set_name:22} spread {spread:.4f}   {cells}")

    print("\n" + "=" * 62)
    print("noise floor per split (spread of identical configs, different seeds)")
    print("=" * 62)
    for set_name, _ in SETS:
        v = spreads[set_name]
        if v:
            print(f"  {set_name:22} min {min(v):.4f}  max {max(v):.4f}  "
                  f"mean {sum(v)/len(v):.4f}   over {len(v)} group(s)")


if __name__ == "__main__":
    main()
