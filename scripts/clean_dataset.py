#!/usr/bin/env python3
"""Clean the locked CSE-CIC-IDS2018 subset and produce a stage-by-stage disposition report.

Run from ANN-Web-Intrusion/Data:
    python clean_dataset.py

Raw CSVs are read only. Outputs are written to processed/cleaning/.

Design choices fixed before modeling:
- Primary classes: Benign, Web Attack, DoS, DDoS.
- Exclude FTP-BruteForce, SSH-Bruteforce, Infilteration, and embedded header rows.
- Drop Timestamp from predictors; retain Label only as audit metadata/target source.
- Convert predictors to numeric; +/-Inf becomes NaN.
- Drop rows containing any non-finite predictor (reported by class before removal).
- Perform GLOBAL exact deduplication across all retained rows and all source files.
- No sampling, balancing, imputation, scaling, feature selection, or modeling.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sqlite3
import pandas as pd
import numpy as np

RAW = Path("raw/CSE-CIC-IDS2018")
OUT = Path("processed/cleaning")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 100_000

CLASS_MAP = {
    "Benign": "Benign",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
}
PRIMARY = ["Benign", "Web Attack", "DoS", "DDoS"]

files = sorted(RAW.glob("*.csv"))
if not files:
    raise SystemExit(f"No CSVs found in {RAW}")

# Use SQLite as a disk-backed exact fingerprint set, avoiding a huge in-memory Python set.
db_path = OUT / "dedupe_index.sqlite"
if db_path.exists():
    db_path.unlink()
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("CREATE TABLE seen (h BLOB PRIMARY KEY)")

stage = {
    "01_raw_primary_mapping": Counter(),
    "02_after_embedded_header_and_scope_filter": Counter(),
    "03_nonfinite_rows_identified": Counter(),
    "04_after_nonfinite_row_removal": Counter(),
    "05_global_duplicates_removed": Counter(),
    "06_final_modeling_pool": Counter(),
}
raw_labels = Counter()
excluded_labels = Counter()
nonfinite_by_column = Counter()
source_final = Counter()
written_header = False
clean_csv = OUT / "cse_cic_ids2018_primary_clean.csv"
if clean_csv.exists():
    clean_csv.unlink()

predictor_columns = None

for f in files:
    print(f"Cleaning {f.name} ...")
    for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if "Label" not in chunk.columns:
            raise ValueError(f"Label missing in {f.name}")
        chunk["Label"] = chunk["Label"].astype("string").str.strip()

        # Raw label accounting.
        vc = chunk["Label"].value_counts(dropna=False)
        for k, v in vc.items():
            key = "<NA>" if pd.isna(k) else str(k)
            raw_labels[key] += int(v)
            if key in CLASS_MAP:
                stage["01_raw_primary_mapping"][CLASS_MAP[key]] += int(v)
            else:
                excluded_labels[key] += int(v)

        # Embedded headers and out-of-scope attack families are excluded.
        scoped = chunk[chunk["Label"].isin(CLASS_MAP)].copy()
        scoped["ModelClass"] = scoped["Label"].map(CLASS_MAP)
        for k, v in scoped["ModelClass"].value_counts().items():
            stage["02_after_embedded_header_and_scope_filter"][k] += int(v)

        # Timestamp is intentionally excluded to reduce collection-schedule leakage.
        drop_cols = [c for c in ["Timestamp"] if c in scoped.columns]
        scoped.drop(columns=drop_cols, inplace=True)

        # Determine predictor set once. Label and ModelClass are metadata/targets.
        current_predictors = [c for c in scoped.columns if c not in ("Label", "ModelClass")]
        if predictor_columns is None:
            predictor_columns = current_predictors
        elif current_predictors != predictor_columns:
            raise ValueError(f"Predictor schema mismatch in {f.name}")

        # Numeric coercion. Embedded-header pollution is already removed via Label filtering.
        for c in predictor_columns:
            scoped[c] = pd.to_numeric(scoped[c], errors="coerce")

        # Mark all non-finite predictors; report class and column effects before dropping rows.
        numeric = scoped[predictor_columns]
        arr = numeric.to_numpy(dtype=float, copy=False)
        finite_mask = np.isfinite(arr)
        row_ok = finite_mask.all(axis=1)
        bad = scoped.loc[~row_ok]
        for k, v in bad["ModelClass"].value_counts().items():
            stage["03_nonfinite_rows_identified"][k] += int(v)
        bad_counts = (~finite_mask).sum(axis=0)
        for c, v in zip(predictor_columns, bad_counts):
            if v:
                nonfinite_by_column[c] += int(v)

        scoped = scoped.loc[row_ok].copy()
        for k, v in scoped["ModelClass"].value_counts().items():
            stage["04_after_nonfinite_row_removal"][k] += int(v)

        # Exact global dedupe fingerprint includes all numeric predictors + ModelClass.
        # float64 bytes are stable within this preprocessing run and preserve exact numeric equality.
        keep_positions = []
        dup_classes = Counter()
        pred_arr = scoped[predictor_columns].to_numpy(dtype=np.float64, copy=False)
        classes = scoped["ModelClass"].astype(str).to_numpy()
        cur = conn.cursor()
        for i, (row, cls) in enumerate(zip(pred_arr, classes)):
            h = hashlib.sha256(row.tobytes() + b"\0" + cls.encode("utf-8")).digest()
            try:
                cur.execute("INSERT INTO seen(h) VALUES (?)", (h,))
                keep_positions.append(i)
            except sqlite3.IntegrityError:
                dup_classes[cls] += 1
        conn.commit()
        for k, v in dup_classes.items():
            stage["05_global_duplicates_removed"][k] += int(v)

        final = scoped.iloc[keep_positions].copy()
        final.insert(0, "SourceFile", f.name)
        for k, v in final["ModelClass"].value_counts().items():
            stage["06_final_modeling_pool"][k] += int(v)
            source_final[(f.name, k)] += int(v)

        # Preserve raw label for auditability; ModelClass is the modeling target.
        final.to_csv(clean_csv, mode="a", header=not written_header, index=False)
        written_header = True

conn.close()

# Long-form disposition report, with removals clearly separated from remaining counts.
rows = []
for stage_name, counts in stage.items():
    for cls in PRIMARY:
        rows.append({"stage": stage_name, "class": cls, "count": int(counts.get(cls, 0))})
pd.DataFrame(rows).to_csv(OUT / "stage_disposition.csv", index=False)

# Wide report for manuscript-friendly reading.
wide_rows = []
for stage_name, counts in stage.items():
    rec = {"stage": stage_name}
    for cls in PRIMARY:
        rec[cls] = int(counts.get(cls, 0))
    rec["Total"] = sum(rec[c] for c in PRIMARY)
    wide_rows.append(rec)
pd.DataFrame(wide_rows).to_csv(OUT / "stage_disposition_wide.csv", index=False)

pd.DataFrame(sorted(raw_labels.items()), columns=["raw_label", "count"]).to_csv(OUT / "raw_label_counts.csv", index=False)
pd.DataFrame(sorted(excluded_labels.items()), columns=["excluded_label", "count"]).to_csv(OUT / "excluded_labels.csv", index=False)
pd.DataFrame(sorted(nonfinite_by_column.items()), columns=["column", "nonfinite_count"]).to_csv(OUT / "nonfinite_by_column.csv", index=False)
pd.DataFrame([
    {"source_file": f, "class": c, "final_count": n}
    for (f, c), n in sorted(source_final.items())
]).to_csv(OUT / "final_counts_by_source.csv", index=False)

manifest = {
    "raw_directory": str(RAW),
    "output_clean_csv": str(clean_csv),
    "input_files": [f.name for f in files],
    "chunk_size": CHUNK,
    "class_map": CLASS_MAP,
    "dropped_predictor_columns": ["Timestamp"],
    "target_columns_not_used_as_predictors": ["Label", "ModelClass"],
    "nonfinite_policy": "Rows with any NaN, +Inf, or -Inf after numeric coercion are removed; effects are reported by class and feature.",
    "dedupe_policy": "Exact global deduplication across retained rows and across all input files using all numeric predictor values plus ModelClass; SourceFile is not part of duplicate identity.",
    "not_performed": ["imputation", "sampling", "class balancing", "SMOTE", "scaling", "feature selection", "ANN training"],
    "predictor_columns": predictor_columns,
    "final_counts": {c: int(stage["06_final_modeling_pool"].get(c, 0)) for c in PRIMARY},
}
(OUT / "cleaning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("\nCleaning complete. Raw files were not modified.")
print(f"Clean dataset: {clean_csv}")
print(f"Disposition report: {OUT / 'stage_disposition_wide.csv'}")
print("\nFINAL MODELING POOL")
for cls in PRIMARY:
    print(f"  {cls:12s} {stage['06_final_modeling_pool'].get(cls, 0):,}")
print("\nNON-FINITE ROWS IDENTIFIED (before removal)")
for cls in PRIMARY:
    print(f"  {cls:12s} {stage['03_nonfinite_rows_identified'].get(cls, 0):,}")
print("\nGLOBAL DUPLICATES REMOVED")
for cls in PRIMARY:
    print(f"  {cls:12s} {stage['05_global_duplicates_removed'].get(cls, 0):,}")
