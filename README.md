# chest-xray-bench

A controlled comparison of roughly 50 training runs on CheXpert under one fixed
pipeline, isolating backbone, input resolution, pretraining source, uncertain label
policy, loss function and ensembling one factor at a time to find what actually
improves multi-label chest X-ray diagnosis.

Every run shares the same splits, preprocessing, augmentation and evaluation, so any
difference between two runs comes from the design choice under test and not from the
setup around it.

**[43 trained models on Hugging Face](https://huggingface.co/mamounyosef/chest-xray-bench)**
· technical report in [`paper/`](paper/)

## Results

Mean AUROC over the five competition findings, on the official test split.

| | test500 |
|---|---|
| Best single model, `medmae_vitb_nih_B_768_s2_seed1337` | 0.9113 |
| Plain average of three different backbones | **0.9174** |

The ensemble gains +0.0061 over its best member, with a 95% bootstrap interval of
[+0.0005, +0.0118] over 10,000 paired resamples.

### What moved the metric

- **Input resolution**, up to about 768x640. Every backbone improved over that first
  step. Past it the gains stop while the cost keeps climbing.
- **Pretraining on chest radiographs.** Medical-MAE and RAD-DINO backbones beat every
  ImageNet backbone at equal size and equal resolution.
- **How uncertain labels are handled.** On ConvNeXt-B, self-training Consolidation's
  uncertain cells is worth +0.0227 over reading them all as positive.
- **Ensembling, when the members differ.** Seven runs of one backbone averaged to
  0.9095, below the best single model; three different backbones reached 0.9174.

### What did not

The ImageNet label set (1k against 22k), label smoothing, and optimizing AUROC
directly with an AUC margin loss all fall inside the noise. Architecture matters
about three times the seed noise between the best and worst backbone, but capacity
does not order the results: ConvNeXt-T at 27.8M finishes above ConvNeXt-L at 196.2M.

### How much of this is noise

Retraining one configuration under a different seed moves mean AUROC by 0.001 on the
19k validation split and by 0.004 to 0.017 on the two radiologist sets. That is wider
than many of the differences this study set out to measure, so anything below roughly
0.015 on test500 is treated as a tie.

## Splits

Frontal views only, split by patient so no patient appears in two splits.

| Split | Source | Images | Labels |
|---|---|---|---|
| train | official train, 90% | 171,927 | automatic labeler |
| val19k | official train, 10% | 19,100 | automatic labeler |
| valid200 | official validation | 202 | radiologist consensus |
| test500 | official test | 518 | radiologist consensus |

## Layout

```
data_code/            notebooks that build the splits from the raw CheXpert release
training_scripts/
  shared_code.py      the shared engine: data, training loop, metrics, Modal wiring
  <run>/              one folder per experiment
    config.yaml       everything that varies for this run
    train.py          builds the model, calls the shared trainer
    evaluate.py       scores valid200 and test500
    calibrate_threshold.py
    results/          logs, metrics, thresholds (checkpoints are not tracked)
  others/             cross-run tooling: ensembling, bootstrap CI, exports, sweeps
paper/                the technical report
```

## Reproducing a run

Runs execute on [Modal](https://modal.com) GPUs, configured per run under `modal:` in
each `config.yaml`.

```bash
modal run training_scripts/<run>/train.py
python training_scripts/<run>/calibrate_threshold.py
python training_scripts/<run>/evaluate.py
```

The ensembling, confidence intervals and figures live in `training_scripts/others/`.

## Data

**CheXpert is not included here.** Its research use agreement does not permit
redistributing the dataset, and the label tables are part of it. Request access from
[Stanford AIMI](https://stanfordaimi.azurewebsites.net/), then run
[`data_code/chexpert/02_splits_dataset.ipynb`](data_code/chexpert/02_splits_dataset.ipynb)
to regenerate the exact splits used here. See
[`data/chexpert/README.md`](data/chexpert/README.md).

## License

Code is Apache-2.0. The trained models are released under CC BY-NC 4.0, matching
CheXpert's research use terms. These are research artifacts, not medical devices, and
have never been validated outside CheXpert's own splits.
