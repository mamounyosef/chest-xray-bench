# Paper outline and progress

**Title:** A Systematic Study of Design Choices for Multi-Label Chest X-ray Classification on CheXpert

**Author:** Ma'moun Yosef, Department of Artificial Intelligence, University of Jordan, Amman, Jordan

**Target:** technical report on ResearchGate. Source in `technical_report.tex`, references in `references.bib`.

Status key: `done` written, `next` the chunk being worked on now, `todo` not started.

---

## Writing order

Sections are written in this order, not in the order they are read:
Methods, then Results, then Discussion, then Introduction and Related work, then Abstract, and the Title is revisited at the end.

---

## 1. Abstract

`todo` Written last, once the results and the claims are settled.

## 2. Introduction

`todo` What the paper does and why. The gap it addresses: published CheXpert numbers are hard to compare because setups differ in many ways at once. States the contributions.

## 3. Related work

`todo` CheXpert and its uncertainty policies, chest X-ray classification backbones, domain pretraining, ensembling.

## 4. Methods

| Subsection | Status | Notes |
|---|---|---|
| 4.1 Dataset and splits | `done` | Table 1. Frontal only, 5 tasks, 90/10 patient grouped, val19k for stability, test500 held back |
| 4.2 Uncertain labels | `done` | Five policies defined. Comparison deferred to Results |
| 4.3 Preprocessing and augmentation | `done` | Table 2. Aspect preserving fit and pad, CLAHE ablation, no horizontal flip |
| 4.4 Backbones and pretraining | `done` | Table 3. Twelve backbones, real parameter counts, resolutions |
| 4.5 Training setup | `done` | Prose, no table. Optimizer, learning rate, layer-wise decay, schedule, label smoothing, early stopping, two-stage resolution ramps, the AUC-M objective |
| 4.6 Evaluation | `done` | Prose. Mean AUROC headline, AUPRC, F1 thresholds calibrated on val19k then frozen, checkpoint selection, paired bootstrap |

## 5. Results

| Subsection | Status | Notes |
|---|---|---|
| 5.1 Architecture comparison | `done` | Table 4. All backbones at the shared 384x320 setting |
| 5.2 Pretraining source | `done` | Table 5. ImageNet-1k vs 22k, ChestX-ray14 stage, Medical-MAE variants, all at 384x320. RAD-DINO and the 768 Medical-MAE runs moved to 5.3 because they were never run at the shared resolution |
| 5.3 Resolution | `done` | Table 6. Gains up to 768x640, flat above it, cost keeps climbing. Carries RAD-DINO and the two-stage Medical-MAE runs |
| 5.4 Training choices | `done` | Table 7. Merged. Uncertainty policies, BCE vs AUC-M, label smoothing. Policies barely matter, masking uncertain cells and self-training Consolidation are the only real gains, AUC-M alone collapses |
| 5.5 How much of this is noise | `done` | Table 8. Seed spread is 0.001 on val19k and 0.005 to 0.012 on the radiologist sets, so only test500 differences above about 0.01 are worth reading. Carries the labels `sec:results-noise`, `sec:results-seed`, `sec:results-valsize` |
| 5.6 Ensembling and final result | `done` | Table 9. Headline test500 mean AUROC **0.9130** [0.8990, 0.9262], six member weighted logit average, weights fitted on val200. Paired bootstrap vs the best single member: +0.0103 [+0.0022, +0.0186]. Diversity beats member count: 3 different backbones 0.9115 vs 7 ConvNeXt-B runs 0.8972 |

## 6. Discussion

`done` Resolution, domain pretraining and diverse ensembling paid. Architecture, ImageNet label set, uncertainty policy, label smoothing and AUC-M did not. Seed noise exceeds most measured effects, and val19k measures agreement with the labeler rather than with a radiologist.

## 7. Limitations

`done` One dataset, no external validation. Ensemble selected and weighted on 234 val200 images, so the bootstrap does not cover selection. Some ablations span backbones or vary two things at once. Noise floor from three repeated configurations.

## 8. Conclusion

`done`

## References

`in progress` All entries currently in `references.bib` are verified against the source.

---

## Run inventory

Extracted programmatically from each run's `config.yaml`, `results/val_log.csv`,
`results/training_summary.json` and the `train_config/` console captures. Fields:
resolution, learning rate, epochs, total optimizer steps, selected step, batch size,
GPU, label smoothing, uncertainty policy, wall clock time, val19k, val200, test500.

Notes on the extraction, so the numbers are not misread later:

- val19k is read from `val_log.csv` at the selected step. It is **not** the
  `best_metrics` field of `training_summary.json`, which holds the monitored subset
  and is therefore val200 for the later runs.
- GPU is taken from each run's console banner, so it reflects what actually ran.
- The ChestX-ray14 pretraining runs write both stages into one `val_log.csv` with
  overlapping step numbers, so the CheXpert stage is the last matching row.
- `medmae_vitb_nih_A_512_s1`, `A_768_s2` and the original `medmae_vitb_nih_B_448_s1`
  have no local logs.

## Open items

- `01_test500.csv` has 667 data rows but the evaluation output reports 668 images. Confirm which is correct before the number is quoted in Results.
- Decide whether the two-phase curriculum run belongs under uncertain labels or training setup.
- `medmae_vitb_nih_B_448_s1_seed1337` scores 0.8935 on test500 in its own run output but
  0.8997 as an ensemble member, and `medmae_vitb_nih_B_448_s1_seed7` shows 0.8969 in both.
  Find out which checkpoint the ensemble loaded before either number is quoted as the run's.
- In the appendix run table, the LR columns for the three ChestX-ray14 pretraining rows
  report the ChestX-ray14 stage learning rate, while the epoch and step counts mix the
  two stages. Either split those rows in two or say so in the table note.
- Decide whether the per-disease AUC-M stage-2 runs get their own results subsection or a single line.
