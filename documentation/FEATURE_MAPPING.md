
# Cross-Dataset Feature Mapping

## 1. Purpose

This document describes the feature-compatibility procedure used to evaluate artificial neural network intrusion-detection models across **CSE-CIC-IDS2018** and **CIC-IDS2017**.

The primary internal ANN models were developed using **78 numeric CICFlowMeter predictors** from CSE-CIC-IDS2018.

During preparation of CIC-IDS2017 for independent external validation, the MachineLearningCVE files were found to contain 78 flow features plus the class label. However, they did **not** contain the `Protocol` predictor used by the CSE-CIC-IDS2018 models.

The study did not infer, reconstruct, impute, or fabricate the missing `Protocol` feature.

Instead, a separate cross-dataset experiment was created using the **77 predictors directly available in both datasets**.

This design preserves the distinction between:

1. the original 78-feature internal experiments; and
2. the 77-feature cross-dataset generalization experiments.

---

## 2. Feature-Compatibility Principle

Feature mapping followed three rules.

### Rule 1: Match semantically equivalent CICFlowMeter measurements

Differences caused by changes in CICFlowMeter naming conventions were harmonized where the underlying flow measurement was equivalent.

For example:

```text
CSE-CIC-IDS2018: Flow Byts/s
CIC-IDS2017:     Flow Bytes/s

Rule 2: Do not manufacture missing predictors

If a CSE-CIC-IDS2018 feature did not have a defensible CIC-IDS2017 equivalent, the pipeline did not:

substitute an unrelated variable;
fill the feature with zero;
derive the feature from other predictors;
infer the feature from ports or attack labels; or
fit a target-dataset transformation to compensate for its absence.
Rule 3: Preserve predictor order

The final 77 predictors were arranged in exactly the same order for:

CSE-CIC-IDS2018 training;
CSE-CIC-IDS2018 validation;
CSE-CIC-IDS2018 internal testing; and
CIC-IDS2017 external validation.

The authoritative machine-readable ordering is stored in:

configurations/predictor_order_77.csv

3. Why Protocol Was Removed

The CSE-CIC-IDS2018 predictor set contained:

Protocol


The CIC-IDS2017 MachineLearningCVE files used in this study did not.

The CIC-IDS2017 header contained features beginning with:

Destination Port
Flow Duration
Total Fwd Packets
Total Backward Packets
...


and ending with:

Idle Mean
Idle Std
Idle Max
Idle Min
Label


No Protocol column was present.

Inferring protocol from destination port would not have been methodologically defensible. For example, a destination port alone does not reliably reproduce the protocol value extracted by the original flow-processing pipeline.

Therefore:

78-feature CSE-CIC-IDS2018 models
                 ↓
     retained for internal study

Cross-dataset experiment
                 ↓
        remove Protocol
                 ↓
      77 common predictors
                 ↓
 retrain using CSE-CIC-IDS2018 only
                 ↓
 freeze model and scaler
                 ↓
 evaluate on CIC-IDS2017


The existing 78-feature models were not modified.

4. Primary Feature Mapping

The following table documents the naming harmonization required between the CSE-CIC-IDS2018 predictor names and CIC-IDS2017 MachineLearningCVE column names.

CSE-CIC-IDS2018 predictor	CIC-IDS2017 columnDst Port	Destination Port
Flow Duration	Flow Duration
Tot Fwd Pkts	Total Fwd Packets
Tot Bwd Pkts	Total Backward Packets
TotLen Fwd Pkts	Total Length of Fwd Packets
TotLen Bwd Pkts	Total Length of Bwd Packets
Fwd Pkt Len Max	Fwd Packet Length Max
Fwd Pkt Len Min	Fwd Packet Length Min
Fwd Pkt Len Mean	Fwd Packet Length Mean
Fwd Pkt Len Std	Fwd Packet Length Std
Bwd Pkt Len Max	Bwd Packet Length Max
Bwd Pkt Len Min	Bwd Packet Length Min
Bwd Pkt Len Mean	Bwd Packet Length Mean
Bwd Pkt Len Std	Bwd Packet Length Std
Flow Byts/s	Flow Bytes/s
Flow Pkts/s	Flow Packets/s
Flow IAT Mean	Flow IAT Mean
Flow IAT Std	Flow IAT Std
Flow IAT Max	Flow IAT Max
Flow IAT Min	Flow IAT Min
Fwd IAT Tot	Fwd IAT Total
Fwd IAT Mean	Fwd IAT Mean
Fwd IAT Std	Fwd IAT Std
Fwd IAT Max	Fwd IAT Max
Fwd IAT Min	Fwd IAT Min
Bwd IAT Tot	Bwd IAT Total
Bwd IAT Mean	Bwd IAT Mean
Bwd IAT Std	Bwd IAT Std
Bwd IAT Max	Bwd IAT Max
Bwd IAT Min	Bwd IAT Min
Fwd PSH Flags	Fwd PSH Flags
Bwd PSH Flags	Bwd PSH Flags
Fwd URG Flags	Fwd URG Flags
Bwd URG Flags	Bwd URG Flags
Fwd Header Len	Fwd Header Length
Bwd Header Len	Bwd Header Length
Fwd Pkts/s	Fwd Packets/s
Bwd Pkts/s	Bwd Packets/s
Pkt Len Min	Min Packet Length
Pkt Len Max	Max Packet Length
Pkt Len Mean	Packet Length Mean
Pkt Len Std	Packet Length Std
Pkt Len Var	Packet Length Variance
FIN Flag Cnt	FIN Flag Count
SYN Flag Cnt	SYN Flag Count
RST Flag Cnt	RST Flag Count
PSH Flag Cnt	PSH Flag Count
ACK Flag Cnt	ACK Flag Count
URG Flag Cnt	URG Flag Count
CWE Flag Count	CWE Flag Count
ECE Flag Cnt	ECE Flag Count
Down/Up Ratio	Down/Up Ratio
Pkt Size Avg	Average Packet Size
Fwd Seg Size Avg	Avg Fwd Segment Size
Bwd Seg Size Avg	Avg Bwd Segment Size
Fwd Byts/b Avg	Fwd Avg Bytes/Bulk
Fwd Pkts/b Avg	Fwd Avg Packets/Bulk
Fwd Blk Rate Avg	Fwd Avg Bulk Rate
Bwd Byts/b Avg	Bwd Avg Bytes/Bulk
Bwd Pkts/b Avg	Bwd Avg Packets/Bulk
Bwd Blk Rate Avg	Bwd Avg Bulk Rate
Subflow Fwd Pkts	Subflow Fwd Packets
Subflow Fwd Byts	Subflow Fwd Bytes
Subflow Bwd Pkts	Subflow Bwd Packets
Subflow Bwd Byts	Subflow Bwd Bytes
Init Fwd Win Byts	Init_Win_bytes_forward
Init Bwd Win Byts	Init_Win_bytes_backward
Fwd Act Data Pkts	act_data_pkt_fwd
Fwd Seg Size Min	min_seg_size_forward
Active Mean	Active Mean
Active Std	Active Std
Active Max	Active Max
Active Min	Active Min
Idle Mean	Idle Mean
Idle Std	Idle Std
Idle Max	Idle Max
Idle Min	Idle Min

The repository's machine-readable mapping should be treated as authoritative if a discrepancy is discovered between this explanatory document and the executable configuration.

5. Features Not Used as ANN Predictors

The following variables were excluded from ANN predictor matrices:

Label
ModelClass
BinaryTarget
AttackFamilyTarget
SourceFile
RowID
Timestamp


These fields could disclose the target directly or encode information about the collection scenario.

For example, attack families in CSE-CIC-IDS2018 were associated with particular collection days. Providing Timestamp or SourceFile to the ANN could allow the model to distinguish attack scenarios by collection context instead of learning traffic characteristics.

6. 78-Feature Internal Models

The primary internal experiments used 78 predictors, including Protocol.

These models were evaluated only against frozen CSE-CIC-IDS2018 holdout partitions.

The 78-feature models were:

finalized_binary_ann_before_test.keras
finalized_attack_family_ann_before_test.keras


These models were not used for CIC-IDS2017 external validation because CIC-IDS2017 could not supply the complete 78-feature vector.

7. 77-Feature Cross-Dataset Models

Separate models were trained after removing only:

Protocol


from the previously frozen CSE-CIC-IDS2018 partitions.

No CIC-IDS2017 observations were involved in this redevelopment.

The resulting models were:

finalized_binary_ann_77_before_external.keras
finalized_attack_family_ann_77_before_external.keras


The 77-feature training procedure preserved:

the same CSE-CIC-IDS2018 partition assignments;
random seed 42;
the same target definitions;
training-only scaling;
training-only class weights;
the same core ANN architectures; and
validation-based model development without CIC-IDS2017 access.
8. Scaling Across Datasets

Separate StandardScaler objects were fitted for the 77-feature binary and attack-family experiments using only their respective CSE-CIC-IDS2018 training partitions.

Conceptually:

scaler.fit(X_cse_cic_ids2018_train)

X_cse_validation_scaled = scaler.transform(X_cse_validation)
X_cse_test_scaled = scaler.transform(X_cse_test)

X_cicids2017_external_scaled = scaler.transform(X_cicids2017_external)


The following operation was explicitly prohibited:

scaler.fit(X_cicids2017_external)


or:

scaler.fit_transform(X_cicids2017_external)


Fitting the scaler on CIC-IDS2017 would incorporate information from the target distribution and weaken the independence of the external validation.

9. Label Compatibility

Feature compatibility and class compatibility were handled independently.

Only attack classes with a predefined correspondence between the two datasets were included in the primary external evaluation.

Binary task
BENIGN                         → Benign

Web Attack - Brute Force       → Intrusion
Web Attack - XSS               → Intrusion
Web Attack - Sql Injection     → Intrusion
DoS Hulk                       → Intrusion
DoS slowloris                  → Intrusion
DoS Slowhttptest               → Intrusion
DoS GoldenEye                  → Intrusion
DDoS                           → Intrusion

Attack-family task
Web Attack - Brute Force       → Web Attack
Web Attack - XSS               → Web Attack
Web Attack - Sql Injection     → Web Attack

DoS Hulk                       → DoS
DoS slowloris                  → DoS
DoS Slowhttptest               → DoS
DoS GoldenEye                  → DoS

DDoS                           → DDoS


The following CIC-IDS2017 classes were outside the primary comparable-family experiment:

PortScan
Bot
Infiltration
FTP-Patator
SSH-Patator
Heartbleed


They were excluded rather than relabeled as benign or forced into an unrelated attack family.

10. Non-Finite Values

After mapping the 77 common predictors, rows containing non-finite values were identified before external prediction.

CIC-IDS2017 rows removed for non-finite predictors:

Benign:       1,777
Web Attack:       0
DoS:            949
DDoS:             2


No imputation was performed.

This rule was selected before the external ANN evaluation.

11. Exact Deduplication

Global exact deduplication was performed after class mapping and numeric conversion.

CIC-IDS2017 duplicates removed:

Benign:       176,263
Web Attack:        37
DoS:            57,967
DDoS:              11


The final unique external population was:

Benign:       2,095,057
Web Attack:       2,143
DoS:            193,745
DDoS:           128,014


For the binary experiment:

Benign:       2,095,057
Intrusion:      323,902
Total:        2,418,959


For the attack-family experiment:

Web Attack:       2,143
DoS:            193,745
DDoS:           128,014
Total:          323,902

12. External Evaluation Controls

After creation of the 77-feature arrays:

the arrays were frozen;
SHA-256 checksums were generated;
the 77-feature CSE-CIC-IDS2018-trained models were frozen;
the binary decision threshold remained 0.50;
the family classifier used softmax argmax;
no CIC-IDS2017 threshold optimization was performed;
no external class balancing was applied;
no retraining occurred after external results were observed.

These controls are intended to ensure that reported CIC-IDS2017 performance represents genuine cross-dataset transfer rather than target-dataset adaptation.

13. Observed Effect of Cross-Dataset Transfer

The 77-feature binary ANN retained strong but substantially reduced performance when transferred to CIC-IDS2017:

Accuracy:             94.057%
Precision:            78.310%
Recall:               76.925%
F1-score:             77.611%
ROC-AUC:               0.9424
PR-AUC:                0.8640
False-negative rate:  23.075%


The attack-family model showed much greater dataset sensitivity:

Overall accuracy:     52.614%
Balanced accuracy:    58.719%
Macro F1-score:       29.176%


Per-class external recall:

Web Attack:           89.174%
DoS:                  86.953%
DDoS:                  0.0305%


This large internal-to-external performance gap is the principal reason feature compatibility is documented explicitly rather than treating the benchmark datasets as interchangeable.

14. Reproduction Checks

Before external evaluation, researchers should confirm that:

[ ] CSE-CIC-IDS2018 raw files pass SHA-256 verification.
[ ] CIC-IDS2017 raw files pass SHA-256 verification.
[ ] Protocol is absent from the CIC-IDS2017 MachineLearningCVE predictors.
[ ] predictor_order_77.csv contains exactly 77 entries.
[ ] feature_mapping_77.csv maps all 77 features one-to-one.
[ ] No predictor column is mapped more than once.
[ ] No model target or source metadata is present in X.
[ ] CSE-CIC-IDS2018 training-fitted scalers are loaded rather than refitted.
[ ] CIC-IDS2017 contains no rows used in training or model selection.
[ ] The binary decision threshold remains 0.50.
[ ] Attack-family predictions use softmax argmax.


If any of these conditions is not satisfied, the resulting experiment should not be treated as a direct reproduction of the reported cross-dataset validation.

15. Machine-Readable Files

The following repository artifacts accompany this document:

configurations/predictor_order_78.csv
configurations/predictor_order_77.csv
configurations/feature_mapping_77.csv

manifests/CSE_CIC_IDS2018_SHA256SUMS.txt
manifests/CICIDS2017_SHA256SUMS.txt
manifests/CLEAN_DATASET_SHA256.txt
manifests/PARTITION_SHA256SUMS.txt
manifests/PREPROCESSING_SHA256SUMS.txt


The CSV files, rather than this Markdown narrative, should be used by automated reproduction scripts.

16. Interpretation

Feature-name harmonization makes compatible measurements syntactically consistent. It does not make the source and target datasets statistically equivalent.

Traffic generation, topology, attack tools, attack timing, benign behavior, and feature distributions can remain substantially different after column alignment.

Consequently:

Feature compatibility is necessary for cross-dataset evaluation, but feature compatibility does not guarantee model generalization.

The external results in this study demonstrate precisely this distinction.


### One change I recommend before committing

Check `configurations/feature_mapping_77.csv` against this Markdown table. **The CSV generated by the experiment should remain authoritative.** If any naming differs, change the Markdown to match the CSV, not vice versa.

Suggested commit:

```bash
git add documentation/FEATURE_MAPPING.md configurations/feature_mapping_77.csv
git commit -m "Document 77-feature cross-dataset mapping"
git push origin main
