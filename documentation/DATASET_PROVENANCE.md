The provenance file should let another researcher obtain the same source files, verify integrity, understand why each file was selected, and reproduce the class mappings without redistributing the large datasets. The draft below is ready to paste into documentation/DATASET_PROVENANCE.md.

# Dataset Provenance

## 1. Purpose

This document records the provenance, selection, integrity controls, class mappings, and permitted use of the datasets analyzed in the study:

> **Generalization-Aware Neural Intrusion Detection for Web and Web-Server Traffic**

The experiments use:

1. **CSE-CIC-IDS2018** for model development and internal evaluation.
2. **CIC-IDS2017** for independent cross-dataset validation.

The original dataset files are not included in this repository. Researchers must obtain them from the Canadian Institute for Cybersecurity and verify the downloaded files against the SHA-256 manifests supplied in this repository.

---

## 2. Dataset Providers and Sources

### 2.1 CSE-CIC-IDS2018

- **Dataset:** CSE-CIC-IDS2018
- **Provider:** Canadian Institute for Cybersecurity, University of New Brunswick
- **Collaborating organization:** Communications Security Establishment
- **Role in this study:** Model development, validation, and internal holdout testing
- **Format used:** Processed CICFlowMeter CSV files
- **Official information page:**  
  https://www.unb.ca/cic/datasets/ids-2018.html
- **Official public AWS bucket:**  
  `s3://cse-cic-ids2018/`
- **AWS directory used:**  
  `Processed Traffic Data for ML Algorithms/`

The complete CSE-CIC-IDS2018 dataset contains traffic and logs generated from multiple benign and attack scenarios. This study did not use every available attack family. The primary scope was restricted to benign traffic and attacks relevant to web applications, web-accessible servers, and service availability.

### 2.2 CIC-IDS2017

- **Dataset:** CIC-IDS2017
- **Provider:** Canadian Institute for Cybersecurity, University of New Brunswick
- **Role in this study:** Independent cross-dataset validation
- **Format used:** MachineLearningCVE labeled CICFlowMeter CSV files
- **Official information page:**  
  https://www.unb.ca/cic/datasets/ids-2017.html
- **Archive used:**  
  `MachineLearningCSV.zip`
- **Extracted directory:**  
  `MachineLearningCVE/`

CIC-IDS2017 was not used for model training, validation-based model selection, threshold selection, hyperparameter tuning, feature selection, or scaler fitting.

---

## 3. CSE-CIC-IDS2018 Source Files

The following processed CSV files were selected and preserved without modification:

| Source file | Principal traffic used in the study |
|---|---|
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | Benign traffic; FTP and SSH brute-force classes excluded from the primary study |
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | Benign, DoS Hulk, and DoS SlowHTTPTest |
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | Benign, DDoS HOIC, and DDoS LOIC-UDP |
| `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | Benign, Web Brute Force, XSS, and SQL Injection |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | Benign traffic; Infiltration excluded from the primary study |

The official AWS bucket contains the filename:

```text
Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv


The spelling Thuesday is the filename supplied by the source repository. This approximately 4 GB file was not required for the finalized primary modeling population because the selected files already provided the prespecified DoS, DDoS, Web Attack, and benign classes.

4. CSE-CIC-IDS2018 File Integrity

The selected raw files were hashed before auditing or preprocessing.

The SHA-256 values used in the experiment were:

1a4919faa0c49c7af97230b0c2d076eba23ee6dd81103a3801d51ac316355d8b  Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
da33c927018274f9d49b145baa00e4ce0526c25b3b890b34c489e247b5e24544  Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
acff8bc61376ee031d80878ee6099e0b1a87a1bd711d8068298421418c9f8147  Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
a5f4a1c2689e0aa6566c03a58466de9c407c0be0cbd3cc69306544026611be04  Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
f15e2a12304446058a0186c8ad67de2bd15735a9ba5c70c9a1f4c4242ab06771  Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv


The repository copy of these checksums is located at:

manifests/CSE_CIC_IDS2018_SHA256SUMS.txt


Researchers can verify downloaded files with:

shasum -a 256 -c manifests/CSE_CIC_IDS2018_SHA256SUMS.txt


An OK result for every file indicates that the downloaded files match the source files used in the study.

5. CSE-CIC-IDS2018 Label Mapping

The following mapping was fixed before ANN training:

Raw label	Modeling classBenign	Benign
Brute Force -Web	Web Attack
Brute Force -XSS	Web Attack
SQL Injection	Web Attack
DoS attacks-Hulk	DoS
DoS attacks-SlowHTTPTest	DoS
DDOS attack-HOIC	DDoS
DDOS attack-LOIC-UDP	DDoS

The following labels were excluded from the primary modeling population:

FTP-BruteForce
SSH-Bruteforce
Infilteration


The spelling Infilteration is retained here because it is the raw label present in the selected CSE-CIC-IDS2018 file.

Rows containing the literal value Label in the label column were treated as embedded header records rather than attack observations.

6. CSE-CIC-IDS2018 Data Disposition

The initial four-class candidate population contained:

Class	Initial recordsBenign	3,067,644
Web Attack	362
DoS	601,802
DDoS	687,742
Total	4,357,550
6.1 Non-finite-value handling

The audit identified:

14,962 retained-scope rows containing non-finite predictor values;
affected columns: Flow Byts/s and Flow Pkts/s;
all affected rows belonged to the Benign class;
no Web Attack, DoS, or DDoS rows were removed at this stage.

Rows containing at least one non-finite predictor were removed. Values were not imputed.

6.2 Global exact deduplication

Exact global deduplication was performed across all selected CSE-CIC-IDS2018 files using the numeric predictor values and mapped class.

Duplicates removed:

Class	Duplicates removedBenign	386,537
Web Attack	24
DoS	456,548
DDoS	487,151
Total	1,330,260

Final unique modeling population:

Class	Final unique recordsBenign	2,666,145
Web Attack	338
DoS	145,254
DDoS	200,591
Total	3,012,328

The cleaned dataset was hashed and frozen before partitioning. The corresponding checksum is recorded in:

manifests/CLEAN_DATASET_SHA256.txt

7. Internal Dataset Partitioning

The cleaned CSE-CIC-IDS2018 population was partitioned using:

Random seed: 42
Training:    70%
Validation:  15%
Test:        15%


Target stratification was applied separately to the binary and attack-family experiments.

7.1 Binary experiment
Split	Benign	IntrusionTraining	1,866,301	242,328
Validation	399,922	51,927
Test	399,922	51,928

For the binary task:

Benign   = Benign
Intrusion = Web Attack + DoS + DDoS

7.2 Attack-family experiment
Split	Web Attack	DoS	DDoSTraining	236	101,678	140,414
Validation	51	21,788	30,088
Test	51	21,788	30,089

The partition files were frozen and verified using:

manifests/PARTITION_SHA256SUMS.txt


No row identifier appeared in more than one partition.

8. Predictors Excluded From Modeling

The following fields were retained only for auditing or target construction and were not ANN inputs:

SourceFile
Label
ModelClass
RowID
BinaryTarget
AttackFamilyTarget
Timestamp


Timestamp was removed to reduce the risk that the ANN would learn collection schedules or attack-day characteristics rather than flow behavior.

For the primary internal experiments, the ANN used 78 numeric CICFlowMeter predictors.

9. CIC-IDS2017 Source Files

The following CIC-IDS2017 MachineLearningCVE files were obtained and preserved:

Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv


All eight raw CSV files were recorded in:

manifests/CICIDS2017_SHA256SUMS.txt


Verification command:

shasum -a 256 -c manifests/CICIDS2017_SHA256SUMS.txt


The source CSV files were not edited or resaved before auditing.

10. CIC-IDS2017 Comparable Labels

The primary external validation was restricted to attack families that corresponded to the CSE-CIC-IDS2018 modeling scope.

CIC-IDS2017 raw label	External modeling classBENIGN	Benign
Web Attack - Brute Force	Web Attack
Web Attack - XSS	Web Attack
Web Attack - Sql Injection	Web Attack
DoS Hulk	DoS
DoS slowloris	DoS
DoS Slowhttptest	DoS
DoS GoldenEye	DoS
DDoS	DDoS

Some extracted CSV files displayed a replacement character between Web Attack and the subtype. The preparation pipeline normalized this encoding artifact before applying the prespecified mapping.

The following CIC-IDS2017 labels were excluded from the primary comparable-class evaluation:

PortScan
Bot
Infiltration
FTP-Patator
SSH-Patator
Heartbleed


These labels were excluded because they were outside the finalized CSE-CIC-IDS2018 class scope. They were not treated as benign.

11. Cross-Dataset Feature Compatibility

The primary CSE-CIC-IDS2018 models used 78 numeric predictors.

The CIC-IDS2017 MachineLearningCVE files contained 78 flow features plus the label field, but they did not include the CSE-CIC-IDS2018 predictor:

Protocol


The study did not infer Protocol from port numbers, set it to zero, or otherwise fabricate the missing predictor.

Instead, separate cross-dataset models were developed using the 77 predictors common to CSE-CIC-IDS2018 and CIC-IDS2017.

The 77-feature models were:

trained only on CSE-CIC-IDS2018;
scaled using parameters fitted only to the CSE-CIC-IDS2018 training partition;
frozen before CIC-IDS2017 prediction; and
externally evaluated without retraining, threshold adjustment, or target-side scaling.

The complete mapping is available at:

configurations/feature_mapping_77.csv


Examples include:

CSE-CIC-IDS2018 predictor	CIC-IDS2017 equivalentDst Port	Destination Port
Tot Fwd Pkts	Total Fwd Packets
Tot Bwd Pkts	Total Backward Packets
Flow Byts/s	Flow Bytes/s
Flow Pkts/s	Flow Packets/s
Pkt Len Mean	Packet Length Mean
Pkt Len Var	Packet Length Variance
Init Fwd Win Byts	Init_Win_bytes_forward
Init Bwd Win Byts	Init_Win_bytes_backward
Fwd Act Data Pkts	act_data_pkt_fwd
Fwd Seg Size Min	min_seg_size_forward

The mapping performs name harmonization only. It does not derive new measurements or substitute semantically different features.

12. CIC-IDS2017 External-Validation Disposition

Initial comparable-class counts:

Class	Raw comparable recordsBenign	2,273,097
Web Attack	2,180
DoS	252,661
DDoS	128,027

Rows removed because of non-finite predictors:

Class	Non-finite rows removedBenign	1,777
Web Attack	0
DoS	949
DDoS	2

Exact duplicate rows removed:

Class	Duplicates removedBenign	176,263
Web Attack	37
DoS	57,967
DDoS	11

Final unique external-validation population:

Class	Final unique recordsBenign	2,095,057
Web Attack	2,143
DoS	193,745
DDoS	128,014

The binary external-validation population contained:

Benign:     2,095,057
Intrusion:    323,902
Total:      2,418,959


The attack-family external-validation population contained:

Web Attack:   2,143
DoS:        193,745
DDoS:       128,014
Total:      323,902


Prepared external arrays were hashed before model evaluation.

13. Training and External-Validation Boundaries

The following controls were enforced:

CIC-IDS2017 was not used during ANN training.
CIC-IDS2017 was not used for validation-based model selection.
CIC-IDS2017 was not used for feature selection.
CIC-IDS2017 was not used to fit or update StandardScaler parameters.
The binary threshold remained fixed at 0.50.
Attack-family predictions used softmax argmax.
Class weights were calculated from CSE-CIC-IDS2018 training labels only.
No SMOTE, oversampling, undersampling, or synthetic external observations were used.
Models were frozen before external prediction.
External predictions and metrics were preserved with SHA-256 manifests.
The external results were not used to retrain or revise the reported models.
14. Dataset Directory Expectations

The scripts expect a local structure similar to:

Data/
├── raw/
│   └── CSE-CIC-IDS2018/
│       ├── Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
│       ├── Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
│       ├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
│       ├── Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
│       └── Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
│
└── external-validation/
    └── CIC-IDS2017/
        └── raw/
            └── MachineLearningCVE/
                ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
                ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
                ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
                ├── Monday-WorkingHours.pcap_ISCX.csv
                ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
                ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
                ├── Tuesday-WorkingHours.pcap_ISCX.csv
                └── Wednesday-workingHours.pcap_ISCX.csv


The Data/ directory is excluded from Git tracking.

15. Redistribution and Licensing

This repository does not redistribute the original CSE-CIC-IDS2018 or CIC-IDS2017 files.

The datasets remain subject to the terms, conditions, and citation requirements established by the Canadian Institute for Cybersecurity and the University of New Brunswick.

The repository's MIT License applies only to original code and documentation created for this study. It does not transfer ownership of or relicense the CIC datasets.

Researchers using the datasets should cite the applicable dataset publications and official dataset pages.

16. Provenance Limitations

The following limitations should be considered when reproducing or interpreting the experiments:

Both datasets were generated in controlled experimental environments.
Attack families were associated with specific collection days and attack tools.
Flow labels do not independently prove successful compromise of a production web application.
The CSE-CIC-IDS2018 Web Attack class was small, containing only 338 unique records after cleaning.
The 77-feature compatibility experiment omitted Protocol because it was absent from CIC-IDS2017 MachineLearningCVE.
Differences in traffic-generation procedures, tools, topology, and feature distributions may contribute to cross-dataset performance degradation.
Near-perfect internal holdout performance should not be interpreted as equivalent production or cross-environment performance.

These limitations are central to the study's motivation for independent cross-dataset validation.


### Final checks before committing

1. Copy the exact CIC-IDS2017 SHA-256 values from your local `CICIDS2017_SHA256SUMS.txt` into the repository manifest.
2. Confirm that `CSE_CIC_IDS2018_SHA256SUMS.txt` uses paths appropriate for the repository instructions or only filenames.
3. Ensure `feature_mapping_77.csv` contains all 77 rows.
4. Check whether the official CSE-CIC-IDS2018 page remains:  
   `https://www.unb.ca/cic/datasets/ids-2018.html`
5. Commit with a descriptive message:

```bash
git add documentation/DATASET_PROVENANCE.md
git commit -m "Document dataset provenance, class mappings, and integrity controls"
git push origin main


