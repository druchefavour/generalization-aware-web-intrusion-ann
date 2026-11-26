# Reproduction Guide

## 1. Purpose

This guide describes how to reproduce the experiments reported in:

> **Generalization-Aware Neural Intrusion Detection for Web and Web-Server Traffic**

The reproduction workflow covers:

1. source-file integrity verification;
2. CSE-CIC-IDS2018 auditing and cleaning;
3. global exact deduplication;
4. fixed train, validation, and test partitioning;
5. training-only feature scaling;
6. 78-feature internal binary and attack-family ANN experiments;
7. CIC-IDS2017 feature-compatibility auditing;
8. 77-feature cross-dataset model development;
9. frozen external evaluation on CIC-IDS2017; and
10. verification of generated artifacts.

The workflow is designed to prevent:

- test-set leakage;
- target-dataset scaling;
- duplicate records across partitions;
- threshold tuning on external data;
- accidental use of labels or collection metadata as predictors; and
- modification of raw source files.

---

## 2. Experimental Overview

The study contains two related experimental tracks.

### Track A: Internal 78-feature experiments

```text
CSE-CIC-IDS2018
        ↓
Audit and cleaning
        ↓
Global deduplication
        ↓
70% training / 15% validation / 15% test
        ↓
78-feature binary ANN
        +
78-feature attack-family ANN
        ↓
Frozen internal holdout evaluation

Track B: Cross-dataset 77-feature experiments
CSE-CIC-IDS2018
        ↓
Remove Protocol
        ↓
77 common predictors
        ↓
Train and freeze new ANN models
        ↓
CIC-IDS2017
        ↓
Map the same 77 predictors
        ↓
Apply CSE-CIC-IDS2018 training-fitted scaler
        ↓
External evaluation without retraining


The 78-feature models are used only for the primary internal experiments.

The 77-feature models are separately trained because the CIC-IDS2017 MachineLearningCVE files do not contain the Protocol predictor.

3. System Requirements
3.1 Tested environment

The experiments were conducted using:

Operating system: macOS 15.8.3
Architecture:     Apple ARM64
Python:           3.12.16
TensorFlow:       2.21.0
Keras:            3.15.1


Exact dependency information is provided in:

requirements.txt
environment.yml
environment-complete.yml

3.2 Hardware considerations

The complete workflow processes several million flow records.

Recommended minimum resources:

Memory:       16 GB RAM
Storage:      20 GB free space
Processor:    Modern 64-bit multicore CPU
Python:       3.12-compatible environment


Thirty-two GB of memory is preferable when preparing scaled NumPy arrays and globally deduplicating the datasets.

Native macOS TensorFlow execution is CPU-based unless a separately supported acceleration configuration is used.

4. Repository Setup

Clone the repository:

git clone https://github.com/druchefavour/generalization-aware-web-intrusion-ann.git
cd generalization-aware-web-intrusion-ann


Create the expected local data structure:

mkdir -p Data/raw/CSE-CIC-IDS2018
mkdir -p Data/processed
mkdir -p Data/external-validation/CIC-IDS2017/raw/MachineLearningCVE


The large local data directories are excluded from Git tracking.

5. Environment Creation
5.1 Conda method

Create the environment:

conda env create -f environment.yml
conda activate ann-web-intrusion


Verify the active interpreter:

which python
python --version


Verify TensorFlow:

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

5.2 Pip method

Alternatively:

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

5.3 Package verification
python - <<'PY'
import tensorflow
import keras
import numpy
import pandas
import sklearn
import joblib

print("TensorFlow:", tensorflow.__version__)
print("Keras:", keras.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)
PY


Minor numerical differences may occur across operating systems and TensorFlow builds. The confusion matrices and principal metrics should remain substantively consistent.

6. Obtain the Source Datasets

The repository does not redistribute the original benchmark data.

6.1 CSE-CIC-IDS2018

Download the processed machine-learning CSV files from the official public source.

Official dataset information:

https://www.unb.ca/cic/datasets/ids-2018.html


Public AWS bucket:

s3://cse-cic-ids2018/


Required AWS directory:

Processed Traffic Data for ML Algorithms/


Required files:

Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv


Place the files in:

Data/raw/CSE-CIC-IDS2018/


Example AWS CLI download:

aws s3 cp \
  --no-sign-request \
  --cli-connect-timeout 60 \
  --cli-read-timeout 300 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv" \
  "Data/raw/CSE-CIC-IDS2018/"


Repeat the command for every required file.

6.2 CIC-IDS2017

Download and extract:

MachineLearningCSV.zip


Official dataset information:

https://www.unb.ca/cic/datasets/ids-2017.html


Copy the eight extracted MachineLearningCVE CSV files into:

Data/external-validation/CIC-IDS2017/raw/MachineLearningCVE/


Expected files:

Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv

7. Verify Raw-File Integrity

Run the following commands from the repository root.

7.1 CSE-CIC-IDS2018
shasum -a 256 -c manifests/CSE_CIC_IDS2018_SHA256SUMS.txt


Every required file must return:

OK

7.2 CIC-IDS2017
shasum -a 256 -c manifests/CICIDS2017_SHA256SUMS.txt


All eight CSV files must return:

OK


Do not proceed if a checksum fails.

Possible causes include:

an incomplete download;
a different processed-data release;
browser-mediated file alteration;
incorrect directory placement; or
manual editing of the raw CSV.

Raw files must not be opened and resaved through spreadsheet software.

8. Working Directory

The original experiment ran the Python scripts from the local Data directory.

If the repository preserves that structure:

cd Data


Ensure the scripts are available under:

../scripts/


Commands in this guide use:

python ../scripts/script_name.py


If the scripts are copied directly into Data/, remove the ../scripts/ prefix.

Part I: Primary 78-Feature Internal Experiments
9. Audit CSE-CIC-IDS2018

Run:

python ../scripts/dataset_audit.py


Expected output directory:

processed/audit/


Important outputs:

dataset_audit.json
file_summary.csv
label_counts.csv
primary_class_counts.csv
missing_values.csv
infinite_values.csv
constant_candidates.csv
leakage_candidates.csv


Expected prespecified class counts before cleaning:

Benign:       3,067,644
Web Attack:         362
DoS:            601,802
DDoS:           687,742


Review:

cat processed/audit/primary_class_counts.csv
cat processed/audit/missing_values.csv
cat processed/audit/infinite_values.csv
cat processed/audit/leakage_candidates.csv


Expected non-finite findings:

Flow Byts/s
Flow Pkts/s


Expected leakage-review fields:

Label
Timestamp

10. Clean and Deduplicate CSE-CIC-IDS2018

Run:

python ../scripts/clean_dataset.py


Expected output directory:

processed/cleaning/


Important outputs:

cse_cic_ids2018_primary_clean.csv
stage_disposition.csv
stage_disposition_wide.csv
excluded_labels.csv
nonfinite_by_column.csv
final_counts_by_source.csv
cleaning_manifest.json


Inspect:

cat processed/cleaning/stage_disposition_wide.csv
cat processed/cleaning/nonfinite_by_column.csv
cat processed/cleaning/final_counts_by_source.csv


Expected final modeling population:

Benign:       2,666,145
Web Attack:         338
DoS:            145,254
DDoS:           200,591
Total:        3,012,328


Expected records removed:

Non-finite rows:      14,962
Global duplicates: 1,330,260


All non-finite retained-scope rows should belong to the Benign class.

Freeze the clean dataset
shasum -a 256 \
  processed/cleaning/cse_cic_ids2018_primary_clean.csv \
  > processed/cleaning/CLEAN_DATASET_SHA256.txt


Verify:

shasum -a 256 -c processed/cleaning/CLEAN_DATASET_SHA256.txt

11. Create Frozen Internal Partitions

Run:

python ../scripts/partition_dataset.py


Expected output:

processed/partitions/
├── binary/
└── attack_family/

Expected binary partitions
Training:
  Benign       1,866,301
  Intrusion      242,328

Validation:
  Benign         399,922
  Intrusion       51,927

Test:
  Benign         399,922
  Intrusion       51,928

Expected attack-family partitions
Training:
  Web Attack         236
  DoS            101,678
  DDoS           140,414

Validation:
  Web Attack          51
  DoS             21,788
  DDoS            30,088

Test:
  Web Attack          51
  DoS             21,788
  DDoS            30,089


Inspect:

cat processed/partitions/binary/split_class_counts.csv
cat processed/partitions/attack_family/split_class_counts.csv


Freeze the six partition files:

shasum -a 256 \
  processed/partitions/binary/train.csv \
  processed/partitions/binary/validation.csv \
  processed/partitions/binary/test.csv \
  processed/partitions/attack_family/train.csv \
  processed/partitions/attack_family/validation.csv \
  processed/partitions/attack_family/test.csv \
  > processed/partitions/PARTITION_SHA256SUMS.txt


Verify:

shasum -a 256 -c processed/partitions/PARTITION_SHA256SUMS.txt


Do not regenerate the partitions after model development begins.

12. Perform Training-Only Preprocessing

Run:

python ../scripts/preprocess_partitions.py


Expected output:

processed/preprocessing/
├── binary/
├── attack_family/
└── PREPROCESSING_SHA256SUMS.txt


The script:

loads the frozen partitions;
selects the approved predictors;
fits StandardScaler on training predictors only;
transforms validation and test predictors;
encodes targets deterministically;
calculates class weights from training labels only; and
excludes metadata and target fields from X.

Verify:

shasum -a 256 -c \
  processed/preprocessing/PREPROCESSING_SHA256SUMS.txt


Expected binary class weights:

Benign:      0.564922003471037
Intrusion:   4.350774570004292


Expected attack-family class weights:

Web Attack: 342.271186440678
DoS:          0.7944294734357481
DDoS:         0.5752702721950803

13. Train the Primary Binary ANN

Run:

python ../scripts/train_binary_ann.py


The test arrays must not be loaded during this stage.

Expected console statement:

holdout test arrays: NOT LOADED


Expected output:

results/binary_ann_baseline/


Important artifacts:

best_binary_ann.keras
finalized_binary_ann_before_test.keras
history.csv
training_log.csv
validation_metrics.json
training_manifest.json
TRAINING_SHA256SUMS.txt


Verify:

shasum -a 256 -c \
  results/binary_ann_baseline/TRAINING_SHA256SUMS.txt


Approximate validation metrics:

Accuracy:   0.999880
Precision:  0.999499
Recall:     0.999461
ROC-AUC:    0.999967
PR-AUC:     0.999634


Small numerical deviations may occur across platforms.

14. Evaluate the Frozen Binary Holdout

Run only after binary model development is complete:

python ../scripts/evaluate_binary_ann_final.py


Expected output:

results/binary_ann_final_test/


Verify:

shasum -a 256 -c \
  results/binary_ann_final_test/FINAL_EVALUATION_SHA256SUMS.txt


Expected confusion matrix:

                 Predicted Benign   Predicted Intrusion
Actual Benign          399,904                 18
Actual Intrusion            22             51,906


Expected metrics:

Accuracy:       99.9911%
Precision:      99.9653%
Recall:         99.9576%
Specificity:    99.9955%
F1-score:       99.9615%
ROC-AUC:        0.999999
PR-AUC:         0.999989


Do not change the model or threshold after observing these test results.

15. Train the Primary Attack-Family ANN

Run:

python ../scripts/train_attack_family_ann.py


The script uses:

0 = Web Attack
1 = DoS
2 = DDoS


Expected output:

results/attack_family_ann_baseline/


Verify:

shasum -a 256 -c \
  results/attack_family_ann_baseline/TRAINING_SHA256SUMS.txt


Inspect:

cat results/attack_family_ann_baseline/validation_metrics.json
cat results/attack_family_ann_baseline/validation_confusion_matrix.csv


The reported experiment produced perfect validation classification.

16. Evaluate the Frozen Attack-Family Holdout

Run:

python ../scripts/evaluate_attack_family_ann_final.py


Verify:

shasum -a 256 -c \
  results/attack_family_ann_final_test/FINAL_EVALUATION_SHA256SUMS.txt


Expected holdout counts:

Web Attack:     51/51 correct
DoS:        21,788/21,788 correct
DDoS:       30,089/30,089 correct


The Web Attack result should be interpreted with its Wilson interval because only 51 unique Web Attack records were present in the internal holdout.

Part II: Cross-Dataset 77-Feature Experiments
17. Audit CIC-IDS2017 Compatibility

Run:

python ../scripts/audit_cicids2017_compatibility.py


Expected output:

external-validation/CIC-IDS2017/audit/


The first compatibility audit should report that numerous columns have different names and that the frozen 78-feature model expects:

Protocol


The CIC-IDS2017 MachineLearningCVE files do not contain Protocol.

Do not insert a constant value or infer protocol from destination port.

18. Create the 77-Feature CSE-CIC-IDS2018 Preprocessing

Run:

python ../scripts/preprocess_77_feature_cross_dataset.py


Expected output:

processed/cross_dataset_77/preprocessing/
├── binary/
└── attack_family/


Verify:

shasum -a 256 -c \
  processed/cross_dataset_77/preprocessing/SHA256SUMS.txt


Both manifests should show:

dropped_feature: Protocol
predictor_count: 77
external_data_used: false
existing_78_feature_artifacts_modified: false

19. Train the 77-Feature Binary ANN

Run:

python ../scripts/train_binary_ann_77.py


Expected output:

results/cross_dataset_77/binary_ann/


Verify:

shasum -a 256 -c \
  results/cross_dataset_77/binary_ann/SHA256SUMS.txt


Expected validation performance:

Accuracy:   99.9878%
Precision:  99.9442%
Recall:     99.9499%
ROC-AUC:    0.999966
PR-AUC:     0.999615


The similarity to the 78-feature internal model indicates that removing Protocol had negligible internal effect.

Freeze:

finalized_binary_ann_77_before_external.keras

20. Train the 77-Feature Attack-Family ANN

Run:

python ../scripts/train_attack_family_ann_77.py


Expected output:

results/cross_dataset_77/attack_family_ann/


Verify:

shasum -a 256 -c \
  results/cross_dataset_77/attack_family_ann/SHA256SUMS.txt


The reported validation confusion matrix was:

                 Predicted Web   Predicted DoS   Predicted DDoS
Actual Web              51              0                0
Actual DoS               0         21,788                0
Actual DDoS              0              0           30,088


Freeze:

finalized_attack_family_ann_77_before_external.keras


Do not retrain either 77-feature model after CIC-IDS2017 results are observed.

21. Prepare CIC-IDS2017 With 77 Common Features

Run the corrected script:

python ../scripts/prepare_cicids2017_77.py


The script must read the raw CSV in chunks and normalize headers after reading.

The current implementation should use:

for chunk in pd.read_csv(file, chunksize=CHUNK, low_memory=False):
    chunk.columns = [normalize(column) for column in chunk.columns]


It should not pass normalized header names through usecols before parsing because the original CIC-IDS2017 headers contain leading whitespace.

Expected output:

external-validation/CIC-IDS2017/prepared_77/


Expected final shapes:

Binary:
  2,418,959 rows × 77 predictors

Attack family:
  323,902 rows × 77 predictors


Expected final counts:

Benign:       2,095,057
Web Attack:       2,143
DoS:            193,745
DDoS:           128,014


Verify:

shasum -a 256 -c \
  external-validation/CIC-IDS2017/prepared_77/SHA256SUMS.txt


Inspect:

cat external-validation/CIC-IDS2017/prepared_77/disposition.csv
cat external-validation/CIC-IDS2017/prepared_77/manifest.json


No ANN prediction should occur during this preparation step.

22. Perform External Binary Evaluation

Run:

python ../scripts/evaluate_binary_ann_77_external.py


Expected output:

results/cross_dataset_77/external_binary_cicids2017/


Verify:

shasum -a 256 -c \
  results/cross_dataset_77/external_binary_cicids2017/SHA256SUMS.txt


Expected confusion matrix:

                 Predicted Benign   Predicted Intrusion
Actual Benign        2,026,046              69,011
Actual Intrusion        74,742             249,160


Expected metrics:

Accuracy:             94.0572%
Precision:            78.3101%
Recall:               76.9245%
Specificity:          96.7060%
F1-score:             77.6111%
Balanced accuracy:    86.8153%
ROC-AUC:              0.942403
PR-AUC:               0.864027
False-negative rate:  23.0755%


The binary threshold must remain:

0.50


Do not optimize the threshold using CIC-IDS2017.

23. Perform External Attack-Family Evaluation

Run:

python ../scripts/evaluate_attack_family_ann_77_external.py


Expected output:

results/cross_dataset_77/external_attack_family_cicids2017/


Verify:

shasum -a 256 -c \
  results/cross_dataset_77/external_attack_family_cicids2017/SHA256SUMS.txt


Expected confusion matrix:

                  Predicted Web   Predicted DoS   Predicted DDoS
Actual Web Attack       1,911            232               0
Actual DoS             25,026        168,467             252
Actual DDoS            90,606         37,369              39


Expected summary metrics:

Accuracy:             52.6138%
Balanced accuracy:    58.7192%
Macro precision:      32.2603%
Macro recall:         58.7192%
Macro F1-score:       29.1757%
Weighted F1-score:    50.4538%
Macro ROC-AUC:         0.793294


Expected class recall:

Web Attack:  89.1741%
DoS:         86.9530%
DDoS:         0.0305%


Do not change class weights or retrain after observing these external results.

24. Probability-Sum Audit

The external attack-family probability outputs generated a scikit-learn warning caused by floating-point precision.

Audit the saved probabilities:

python - <<'PY'
import pandas as pd
import numpy as np

p = pd.read_csv(
    "results/cross_dataset_77/external_attack_family_cicids2017/predictions.csv",
    usecols=["prob_Web_Attack", "prob_DoS", "prob_DDoS"]
).to_numpy()

s = p.sum(axis=1)

print("Minimum probability sum:", s.min())
print("Maximum probability sum:", s.max())
print("Maximum absolute deviation:", np.max(np.abs(s - 1.0)))
print(
    "Rows deviating by more than 1e-6:",
    np.sum(np.abs(s - 1.0) > 1e-6)
)
PY


Expected output:

Minimum probability sum: approximately 0.999999851
Maximum probability sum: approximately 1.000000177
Maximum absolute deviation: approximately 1.77e-07
Rows deviating by more than 1e-6: 0


This deviation does not alter the softmax argmax classifications.

25. Generate or Verify Figures

Repository figures should correspond to the frozen results:

Figure 1: Reproducible experimental pipeline
Figure 2: Class disposition before and after cleaning
Figure 3: Internal versus external binary performance
Figure 4: CIC-IDS2017 external attack-family confusion matrix


For Figure 3, place the legend outside the data region:

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.92])


The legend must not cover the bars.

26. Complete Workflow Summary

From the local Data directory:

python ../scripts/dataset_audit.py
python ../scripts/clean_dataset.py
python ../scripts/partition_dataset.py
python ../scripts/preprocess_partitions.py

python ../scripts/train_binary_ann.py
python ../scripts/evaluate_binary_ann_final.py

python ../scripts/train_attack_family_ann.py
python ../scripts/evaluate_attack_family_ann_final.py

python ../scripts/audit_cicids2017_compatibility.py
python ../scripts/preprocess_77_feature_cross_dataset.py

python ../scripts/train_binary_ann_77.py
python ../scripts/train_attack_family_ann_77.py

python ../scripts/prepare_cicids2017_77.py
python ../scripts/evaluate_binary_ann_77_external.py
python ../scripts/evaluate_attack_family_ann_77_external.py

27. Reproduction Success Criteria

A reproduction is considered successful when:

[ ] All source files pass SHA-256 verification.
[ ] The clean CSE-CIC-IDS2018 dataset contains 3,012,328 records.
[ ] The partitions match the documented class counts.
[ ] No RowID appears in more than one partition.
[ ] Training-only StandardScaler fitting is confirmed.
[ ] Internal binary metrics approximate the reported results.
[ ] Internal attack-family predictions reproduce the reported confusion matrix.
[ ] Protocol is excluded from all 77-feature experiments.
[ ] CIC-IDS2017 preparation produces 2,418,959 binary records.
[ ] CIC-IDS2017 attack-family preparation produces 323,902 records.
[ ] CIC-IDS2017 is not used for training or scaler fitting.
[ ] The external binary confusion matrix matches or closely approximates the reported matrix.
[ ] The external family confusion matrix matches or closely approximates the reported matrix.
[ ] Generated artifacts pass their SHA-256 checks.

28. Troubleshooting
TensorFlow fails to import with a protobuf error

Do not repair the Conda base environment by repeatedly changing package versions.

Create the isolated environment:

conda env create -f environment.yml
conda activate ann-web-intrusion


Then verify:

python -c "import tensorflow as tf; print(tf.__version__)"

AWS download closes unexpectedly

Use increased timeouts and retry settings:

AWS_RETRY_MODE=standard AWS_MAX_ATTEMPTS=10 aws s3 cp \
  --no-sign-request \
  --cli-connect-timeout 60 \
  --cli-read-timeout 300 \
  "S3_OBJECT_PATH" \
  "LOCAL_DESTINATION"


Verify the downloaded byte size and SHA-256 digest before use.

Usecols do not match columns

The CIC-IDS2017 CSV headers contain leading spaces.

Read the full chunk first, then normalize:

chunk = pd.read_csv(file, chunksize=CHUNK, low_memory=False)
chunk.columns = [column.strip() for column in chunk.columns]


Do not pass normalized names to usecols before parsing the original header.

Protocol is missing

This is expected for CIC-IDS2017 MachineLearningCVE.

Do not insert or infer Protocol.

Use the separate 77-feature cross-dataset workflow.

Shell displays for> or a parse error

A Python statement was likely pasted directly into the shell.

Press:

Control + C


Then run the Python code from a script or through a properly closed heredoc:

python - <<'PY'
print("Valid Python heredoc")
PY

A checksum fails

Stop the workflow.

Do not proceed until the discrepancy is explained. Redownload or restore the affected artifact and regenerate downstream products only if preprocessing has not yet been frozen.

29. Reporting Reproduced Results

Report reproduced results separately as:

Internal 78-feature evaluation
Internal 77-feature compatibility evaluation
External 77-feature CIC-IDS2017 evaluation

Do not combine internal and external records into a single test population.

Do not use the external results to revise the frozen reported models.

Any model retrained or recalibrated after CIC-IDS2017 access must be described as a separate domain-adaptation or sensitivity experiment rather than a reproduction of the primary external validation.

30. Support and Issues

For reproducibility questions, open a GitHub issue and include:

operating system;
processor architecture;
Python version;
TensorFlow and Keras versions;
script name;
complete error message;
whether all preceding checksums passed; and
the relevant manifest entry.

Do not upload or attach the raw CIC dataset files to GitHub issues.


## Commit the guide

Save the content to:

```text
documentation/REPRODUCTION_GUIDE.md


Then run:

git add documentation/REPRODUCTION_GUIDE.md
git commit -m "Add end-to-end reproduction guide"
git push origin main

Final quality checks

Before committing, confirm that:

script names match the repository exactly;
the scripts expect the same working directory described in the guide;
manifest filenames match the files actually committed;
the corrected CIC-IDS2017 preparation script no longer uses normalized names in read_csv(usecols=...);
the GitHub repository does not contain raw datasets, large NumPy arrays, or SQLite databases.

