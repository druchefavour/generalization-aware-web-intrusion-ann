#!/usr/bin/env python3
"""Create reproducible 70/15/15 partitions for the cleaned CSE-CIC-IDS2018 subset.
Run from ANN-Web-Intrusion/Data:
    python partition_dataset.py

Creates TWO independent experiments:
1) binary: Benign vs Intrusion
2) attack_family: Web Attack vs DoS vs DDoS

Important: no scaling, balancing, imputation, SMOTE, feature selection, or model fitting.
SourceFile, Label, and ModelClass are retained only as audit metadata and are excluded
from the predictor lists recorded in the manifests.
"""
from pathlib import Path
import hashlib, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

INPUT = Path("processed/cleaning/cse_cic_ids2018_primary_clean.csv")
OUT = Path("processed/partitions")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TRAIN, VAL, TEST = 0.70, 0.15, 0.15
AUDIT_COLS = ["SourceFile", "Label", "ModelClass"]

if not INPUT.exists():
    raise SystemExit(f"Missing cleaned input: {INPUT}")

def sha256(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def safe_split(df, stratify_col):
    # First: 70% train, 30% temporary. Second: 15% val, 15% test.
    train_df, temp_df = train_test_split(
        df, test_size=(VAL + TEST), random_state=SEED,
        stratify=df[stratify_col], shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED,
        stratify=temp_df[stratify_col], shuffle=True
    )
    return train_df, val_df, test_df

def counts_table(parts, target):
    rows=[]
    for split, frame in parts.items():
        counts=frame[target].value_counts().sort_index()
        for cls,n in counts.items():
            rows.append({"split":split,"class":str(cls),"count":int(n)})
    return pd.DataFrame(rows)

def source_table(parts, target):
    rows=[]
    for split, frame in parts.items():
        g=frame.groupby(["SourceFile",target], dropna=False).size()
        for (source,cls),n in g.items():
            rows.append({"split":split,"source_file":source,"class":str(cls),"count":int(n)})
    return pd.DataFrame(rows)

def write_experiment(name, df, target, predictor_cols, target_definition):
    edir=OUT/name
    edir.mkdir(parents=True, exist_ok=True)
    parts=dict(zip(["train","validation","test"], safe_split(df, target)))

    files={}
    for split, frame in parts.items():
        p=edir/f"{split}.csv"
        frame.to_csv(p,index=False)
        files[split]={"path":str(p),"rows":len(frame),"sha256":sha256(p)}

    ctab=counts_table(parts,target)
    ctab.to_csv(edir/"split_class_counts.csv",index=False)
    stab=source_table(parts,target)
    stab.to_csv(edir/"split_source_class_counts.csv",index=False)

    # Verify no row-index overlap. RowID originates from frozen cleaned file.
    sets={k:set(v["RowID"].tolist()) for k,v in parts.items()}
    overlap={
        "train_validation":len(sets["train"] & sets["validation"]),
        "train_test":len(sets["train"] & sets["test"]),
        "validation_test":len(sets["validation"] & sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Partition overlap detected: {overlap}")

    manifest={
        "experiment":name,
        "source_clean_dataset":str(INPUT),
        "source_clean_dataset_sha256":sha256(INPUT),
        "random_seed":SEED,
        "split_ratios":{"train":TRAIN,"validation":VAL,"test":TEST},
        "split_method":"Two-stage sklearn train_test_split with shuffle=True and target stratification; fixed random seed.",
        "target_column":target,
        "target_definition":target_definition,
        "audit_metadata_not_predictors":AUDIT_COLS + ["RowID"],
        "predictor_columns":predictor_cols,
        "predictor_count":len(predictor_cols),
        "files":files,
        "rowid_overlap_check":overlap,
        "important_limitation":"Attack families are each concentrated in a single source collection day. This split preserves class proportions but is not an independent collection-day holdout. Report this limitation and use external/cross-dataset validation for generalization claims.",
        "not_performed":["scaling","class balancing","undersampling","oversampling","SMOTE","feature selection","ANN training"]
    }
    (edir/"split_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return parts

print("Reading frozen clean dataset...")
df=pd.read_csv(INPUT,low_memory=False)
df.insert(0,"RowID",np.arange(len(df),dtype=np.int64))

required=set(AUDIT_COLS)
missing=required-set(df.columns)
if missing: raise ValueError(f"Missing expected audit columns: {sorted(missing)}")

predictors=[c for c in df.columns if c not in set(AUDIT_COLS+["RowID"])]
if not predictors: raise ValueError("No predictor columns found")

# Binary experiment: all retained rows.
binary=df.copy()
binary["BinaryTarget"]=np.where(binary["ModelClass"].eq("Benign"),"Benign","Intrusion")
binary_predictors=predictors.copy()
write_experiment(
    "binary", binary, "BinaryTarget", binary_predictors,
    {"Benign":"ModelClass == Benign","Intrusion":"ModelClass in {Web Attack, DoS, DDoS}"}
)

# Attack-family experiment: malicious rows only.
attack=df[df["ModelClass"].isin(["Web Attack","DoS","DDoS"])].copy()
attack["AttackFamilyTarget"]=attack["ModelClass"]
write_experiment(
    "attack_family", attack, "AttackFamilyTarget", predictors.copy(),
    {"Web Attack":"Brute Force -Web / Brute Force -XSS / SQL Injection mapped during cleaning","DoS":"Hulk / SlowHTTPTest mapped during cleaning","DDoS":"HOIC / LOIC-UDP mapped during cleaning"}
)

# Overall manifest linking both experiments.
overall={
    "source_clean_dataset":str(INPUT),
    "source_clean_dataset_sha256":sha256(INPUT),
    "seed":SEED,
    "ratios":"70/15/15",
    "experiments":["binary","attack_family"],
    "predictor_exclusions":AUDIT_COLS+["RowID"],
    "note":"Partition files intentionally retain audit metadata. Modeling code must select only predictor_columns from each split_manifest.json."
}
(OUT/"partition_manifest.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")

print("Partitioning complete.")
for exp in ["binary","attack_family"]:
    print(f"\n{exp.upper()}")
    print(pd.read_csv(OUT/exp/"split_class_counts.csv").to_string(index=False))
print(f"\nOutputs: {OUT}")
