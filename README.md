The section below is tailored to the repository structure and exact experimental workflow used in the manuscript. It distinguishes public benchmark data from the code and derived artifacts you will release.

Recommended README section
## Data and Code Availability

### Source datasets

This study uses two publicly available network intrusion detection datasets maintained by the Canadian Institute for Cybersecurity:

1. **CSE-CIC-IDS2018**, used for model development, internal validation, and holdout testing.
2. **CIC-IDS2017**, used exclusively for independent cross-dataset validation.

The original CSV and packet-capture files are not redistributed through this repository because of their size and the source providers' applicable terms. Researchers should obtain the datasets from the Canadian Institute for Cybersecurity:

- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html

After downloading the datasets, users can verify that the selected source files match those used in the study by running the SHA-256 checks recorded in the `manifests/` directory.

### Repository contents

This repository provides the materials needed to reproduce the study's data preparation, model-development, internal-evaluation, and cross-dataset-validation procedures:

- dataset audit and provenance scripts;
- SHA-256 file-integrity manifests;
- label and attack-family mappings;
- non-finite-value detection and removal procedures;
- global exact-deduplication scripts;
- fixed train, validation, and test partition specifications;
- training-only preprocessing and fitted-scaler specifications;
- 78-feature internal ANN experiments;
- 77-feature cross-dataset ANN experiments;
- CIC-IDS2017-to-CSE-CIC-IDS2018 feature-name mappings;
- model-training and frozen-evaluation scripts;
- model configurations and random-seed settings;
- confusion matrices, classification metrics, and disposition reports;
- software-environment specifications; and
- manuscript figures generated from the experimental results.

The repository does not include the large raw datasets, cleaned data files, NumPy feature arrays, packet captures, or temporary deduplication databases.

### Experimental environment

The experiments were conducted using:

- Python 3.12.16
- TensorFlow 2.21.0
- Keras 3.15.1
- macOS 15.8.3 on Apple ARM64

Exact dependency information is available in:

- `requirements.txt`
- `environment.yml`
- `environment-complete.yml`

To recreate the Conda environment from the portable environment definition, run:

```bash
conda env create -f environment.yml
conda activate ann-web-intrusion


Alternatively, create a Python environment and install the pinned packages:

python -m pip install -r requirements.txt


The environment-complete.yml file records the complete Conda environment used for the experiments. Any machine-specific prefix: entry has been removed from the public version.

Reproduction workflow

After downloading and placing the datasets in the directory structure described in documentation/DATASET_PROVENANCE.md, execute the scripts in the following order:

# CSE-CIC-IDS2018 audit, cleaning, and partitioning
python scripts/dataset_audit.py
python scripts/clean_dataset.py
python scripts/partition_dataset.py
python scripts/preprocess_partitions.py

# Primary 78-feature internal experiments
python scripts/train_binary_ann.py
python scripts/evaluate_binary_ann_final.py
python scripts/train_attack_family_ann.py
python scripts/evaluate_attack_family_ann_final.py

# Feature-compatibility and 77-feature cross-dataset experiments
python scripts/audit_cicids2017_compatibility.py
python scripts/preprocess_77_feature_cross_dataset.py
python scripts/train_binary_ann_77.py
python scripts/train_attack_family_ann_77.py
python scripts/prepare_cicids2017_77.py
python scripts/evaluate_binary_ann_77_external.py
python scripts/evaluate_attack_family_ann_77_external.py


The scripts use fixed random seed 42. StandardScaler parameters and class weights are estimated from the CSE-CIC-IDS2018 training partitions only. CIC-IDS2017 is not used for model training, scaler fitting, threshold selection, hyperparameter selection, or model selection.

The binary models use a prespecified decision threshold of 0.50. The attack-family models classify observations using the maximum three-class softmax probability.

Reproducibility safeguards

The workflow incorporates the following controls:

SHA-256 verification of source and derived files;
read-only preservation of raw dataset files;
removal of embedded header records;
documented handling of missing and infinite values;
global exact deduplication before partitioning;
fixed and non-overlapping train, validation, and test assignments;
exclusion of labels, timestamps, source filenames, and row identifiers from model predictors;
fitting of scaling parameters on training data only;
preservation of validation and holdout boundaries;
one-time evaluation of frozen holdout and external datasets; and
separate 77-feature models for cross-dataset evaluation because the CIC-IDS2017 MachineLearningCVE files do not contain the Protocol feature.
Expected principal results

A successful reproduction should approximate the following results, subject to minor platform-level numerical variation:

Experiment	Accuracy	Recall	F1-scoreInternal binary, CSE-CIC-IDS2018	99.991%	99.958%	99.962%
External binary, CIC-IDS2017	94.057%	76.925%	77.611%
Internal attack-family, CSE-CIC-IDS2018	100.000%	100.000% macro recall	100.000% macro F1
External attack-family, CIC-IDS2017	52.614%	58.719% macro recall	29.176% macro F1

The externally validated attack-family recalls were:

Web Attack: 89.174%
DoS: 86.953%
DDoS: 0.0305%

These cross-dataset results should not be replaced by results obtained through retraining, target-dataset scaling, threshold adjustment, or hyperparameter tuning on CIC-IDS2017.

Trained models and derived artifacts

The archived release includes the final trained-model artifacts and corresponding training-fitted scalers where repository size permits. Large model files may be distributed through the associated Zenodo release rather than stored directly in GitHub.

Important artifacts include:

finalized_binary_ann_before_test.keras
finalized_attack_family_ann_before_test.keras
finalized_binary_ann_77_before_external.keras
finalized_attack_family_ann_77_before_external.keras
standard_scaler.joblib
standard_scaler_77.joblib


The release also includes model configurations, predictor-order files, feature mappings, confusion matrices, evaluation reports, and SHA-256 manifests.

Citation

If you use the code, models, feature mappings, or reproducibility materials, cite the associated article and archived software release:

Ogubuike A. Generalization-Aware Neural Intrusion Detection for Web and Web-Server Traffic. Journal of Web Engineering & Technology. [Year; volume, issue, pages to be added].

Archived reproducibility package:

GitHub: [GITHUB REPOSITORY URL]
Zenodo: [ZENODO DOI URL]
Version: 1.0.0

A machine-readable citation is provided in CITATION.cff.

License

The original code and documentation in this repository are released under the MIT License unless otherwise stated.

The MIT License does not apply to CSE-CIC-IDS2018 or CIC-IDS2017. The source datasets remain subject to the terms established by the Canadian Institute for Cybersecurity and the University of New Brunswick.


## Before publishing

Replace these placeholders after creating the GitHub release and Zenodo archive:

```text
[GITHUB REPOSITORY URL]
[ZENODO DOI URL]
[Year; volume, issue, pages to be added]


Also confirm the exact current URL for CSE-CIC-IDS2018 before publishing the README.

Main writing choices: The section separates source data from your original materials, documents the exact execution order, prevents target-data leakage, provides expected results for validation, and clearly limits the MIT License to your code rather than the CIC datasets.
