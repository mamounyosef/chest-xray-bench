# CheXpert splits

The split files are **not** included here. CheXpert's Stanford University Dataset
Research Use Agreement allows research use only and does not permit redistributing
the dataset, and its label tables are part of the dataset.

## Getting them

1. Request access and download CheXpert from
   [Stanford AIMI](https://stanfordaimi.azurewebsites.net/), and agree to the
   research use agreement yourself.
2. Run [`data_code/02_splits_dataset.ipynb`](../../data_code/02_splits_dataset.ipynb).
   It reads the official `train.csv` and `valid.csv`, keeps frontal views, and writes
   the files every training script expects:

   | File | What it is |
   |---|---|
   | `01_train.csv` | official train, 90%, grouped by patient |
   | `01_val.csv` | official train, 10%, the val19k split |
   | `01_valid200.csv` | the official validation studies, frontal only |
   | `01_test500.csv` | the official test studies, frontal only |

The split is by patient with a fixed seed, so the same command reproduces the same
partition, and any run in this repository is comparable to yours.

## What is included

`01_train_softlabels.csv` stays in the repository. It holds probabilities predicted by
`convnext_base_22k_certain_only` for the uncertain training cells, used by the
U-SelfTrained runs. Those numbers are this project's model output rather than CheXpert
labels, though the image paths they are keyed on come from the dataset.
