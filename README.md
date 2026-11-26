# Cross-dataset three-class Alzheimer's staging from structural MRI

Code for the benchmark *Linear-Time Sequence Models and Convolutional Networks for
Three-Class Alzheimer's Staging from Structural MRI*. It compares four backbones
(ConvNeXt, MambAD, Vision-xLSTM, Vision-RWKV) on ADNI and OASIS-3 under a single
honest protocol: three classes (CN / MCI / AD), subject-level cross-validation and
cross-cohort evaluation.

The label scheme throughout is CDR 0 -> CN, CDR 0.5 -> MCI, CDR >= 1 -> AD (ADNI uses
its own clinical diagnosis directly).

## Data

The image data cannot be redistributed here. ADNI is available to registered
researchers through the LONI IDA (adni.loni.usc.edu) and OASIS-3 through NITRC-IR
(nitrc.org/ir), each under its own data use agreement. The scripts expect the
downloaded data under a local `data/` folder; edit the paths at the top of each
labelling and preprocessing script to match your layout.

## Pipeline

Run the scripts in order. Steps 1-7 prepare the data, 8-16 train and run the
experiments, 17 makes the figures and tables.

| Script | Purpose |
|---|---|
| `01_label_adni.py` | Build the ADNI inventory (subject, label, image path). |
| `02_build_oasis3_shortlist.py` | Select a balanced OASIS-3 cohort from the CDR table (most severe stage within two years). |
| `03_oasis3_inventory.py` | Build the OASIS-3 inventory from the shortlist. |
| `04_clean_and_extract.py` | Preprocess both cohorts (N4 bias correction, MNI registration, skull-strip, three hippocampal-band coronal slices). |
| `05_clean_light_oasis3.py` | Light OASIS-3 preprocessing (no registration or skull-strip) for the standardisation ablation. |
| `06_clean_adni_kaggle.py` | ADNI label and clean on a GPU runtime (same steps as `04`). |
| `07_clean_oasis3_kaggle.py` | OASIS-3 label and clean on a GPU runtime (same steps as `04`). |
| `08_models.py` | The four model builders. |
| `09_train_cv.py` | Subject-level 5-fold cross-validation, within-cohort and cross-cohort, for all four models. |
| `10_train_cv_365.py` | One-year-window sensitivity run (OASIS-3 relabelled with a 365-day pairing window). |
| `11_experiments.py` | Within-cohort experiment runner (multi-seed and slice-count), reused by the ablations. |
| `12_ablation_slicecount.py` | Slice-count ablation (one vs three slices). |
| `13_ablation_spatialstd.py` | Spatial-standardisation ablation (registered vs light pipeline). |
| `14_pretraining_control.py` | ConvNeXt trained from scratch, to isolate the ImageNet pretraining effect. |
| `15_ml_baseline.py` | Conventional machine-learning baseline (structural features + gradient-boosted trees). |
| `16_additional_ablations.py` | Training-set size, learning rate, batch size and augmentation ablations. |
| `17_make_figures_tables.py` | Figures and table values from the 5-fold results. |

`oasis365_labels.csv` holds the subject labels for the one-year-window run.

## Requirements

See `requirements.txt`. Training uses a single GPU (the models are sized to fit a 16 GB
T4). MambAD and Vision-RWKV are plain PyTorch; Vision-xLSTM is installed from
github.com/NX-AI/vision-lstm.

## Notes

The training and experiment scripts are resumable: each writes per-fold results and
skips anything already present, so a run that stops can be restarted and continues.
