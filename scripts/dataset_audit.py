#!/usr/bin/env python3
"""Read-only audit for selected CSE-CIC-IDS2018 CICFlowMeter CSV files.

Run from ANN-Web-Intrusion/Data:
    python dataset_audit.py

Inputs are never modified. Reports are written to processed/audit/.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, math
import pandas as pd
import numpy as np

RAW = Path("raw/CSE-CIC-IDS2018")
OUT = Path("processed/audit")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 100_000

FILES = sorted(RAW.glob("*.csv"))
if not FILES:
    raise SystemExit(f"No CSV files found in {RAW}")

# Labels intentionally fixed before modeling.
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

summary = []
labels_all = Counter()
missing_all = Counter()
inf_all = Counter()
dtypes_by_file = {}
columns_by_file = {}
constant_candidates = defaultdict(lambda: {"seen": False, "value": None, "constant": True})

for f in FILES:
    print(f"Auditing {f.name} ...")
    rows = 0
    duplicate_rows = 0
    embedded_headers = 0
    file_labels = Counter()
    file_missing = Counter()
    file_inf = Counter()
    first_dtypes = None
    columns = None

    for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False):
        # Normalize whitespace in column names only in memory; raw file is untouched.
        chunk.columns = [str(c).strip() for c in chunk.columns]
        columns = list(chunk.columns)
        if "Label" not in chunk.columns:
            raise ValueError(f"Label column missing in {f.name}")

        if first_dtypes is None:
            first_dtypes = {c: str(t) for c, t in chunk.dtypes.items()}

        rows += len(chunk)
        labels = chunk["Label"].astype("string").str.strip()
        embedded_headers += int((labels == "Label").sum())
        counts = labels.value_counts(dropna=False)
        for k, v in counts.items():
            key = "<NA>" if pd.isna(k) else str(k)
            file_labels[key] += int(v)
            labels_all[key] += int(v)

        miss = chunk.isna().sum()
        for c, v in miss.items():
            if v:
                file_missing[c] += int(v)
                missing_all[c] += int(v)

        # Count +/- Inf only after numeric coercion. This also exposes numeric columns
        # contaminated by embedded header strings without changing source data.
        for c in chunk.columns:
            if c == "Label":
                continue
            numeric = pd.to_numeric(chunk[c], errors="coerce")
            ninf = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
            if ninf:
                file_inf[c] += ninf
                inf_all[c] += ninf

            vals = chunk[c].dropna().unique()
            state = constant_candidates[c]
            if len(vals):
                if not state["seen"]:
                    state["seen"] = True
                    state["value"] = str(vals[0])
                    if len(vals) > 1:
                        state["constant"] = False
                elif state["constant"]:
                    if len(vals) > 1 or str(vals[0]) != state["value"]:
                        state["constant"] = False

        # NOTE: duplicated() here is exact duplicates within each chunk. A global
        # duplicate count is computed separately below to avoid a misleading claim.
        duplicate_rows += int(chunk.duplicated().sum())

    columns_by_file[f.name] = columns
    dtypes_by_file[f.name] = first_dtypes
    summary.append({
        "file": f.name,
        "rows": rows,
        "columns": len(columns or []),
        "embedded_header_rows": embedded_headers,
        "within_chunk_exact_duplicates_lower_bound": duplicate_rows,
        "labels": dict(file_labels),
        "missing_values": dict(file_missing),
        "infinite_values": dict(file_inf),
    })

# Check schema consistency.
base_name = summary[0]["file"]
base_cols = columns_by_file[base_name]
schema_report = {}
for name, cols in columns_by_file.items():
    schema_report[name] = {
        "same_as_first_file": cols == base_cols,
        "missing_vs_first": sorted(set(base_cols) - set(cols)),
        "extra_vs_first": sorted(set(cols) - set(base_cols)),
    }

# Aggregate model-class counts under the predetermined mapping.
model_counts = Counter()
excluded_counts = Counter()
for label, count in labels_all.items():
    if label in CLASS_MAP:
        model_counts[CLASS_MAP[label]] += count
    else:
        excluded_counts[label] += count

constant_cols = sorted(c for c, s in constant_candidates.items() if s["seen"] and s["constant"])

# Candidate leakage/non-predictive identifiers are FLAGGED, not automatically removed.
leakage_terms = ("label", "timestamp", "flow id", "src ip", "source ip", "dst ip", "destination ip")
leakage_candidates = sorted({
    c for cols in columns_by_file.values() for c in cols
    if any(term in c.lower() for term in leakage_terms)
})

report = {
    "input_directory": str(RAW),
    "files_audited": [f.name for f in FILES],
    "chunk_size": CHUNK,
    "per_file": summary,
    "schema_consistency": schema_report,
    "aggregate_raw_labels": dict(labels_all),
    "predetermined_class_mapping": CLASS_MAP,
    "aggregate_primary_model_classes_before_cleaning": dict(model_counts),
    "excluded_or_unmapped_labels_before_cleaning": dict(excluded_counts),
    "aggregate_missing_values": dict(missing_all),
    "aggregate_infinite_values": dict(inf_all),
    "constant_columns_candidate": constant_cols,
    "candidate_leakage_or_identifier_columns_for_review": leakage_candidates,
    "important_note": "No raw CSV was modified. Duplicate counts in per_file are lower bounds because they count exact duplicates within chunks only. Do not cite them as global duplicate counts.",
}

(OUT / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# Convenient tabular outputs.
pd.DataFrame([
    {
        "file": r["file"], "rows": r["rows"], "columns": r["columns"],
        "embedded_header_rows": r["embedded_header_rows"],
        "within_chunk_exact_duplicates_lower_bound": r["within_chunk_exact_duplicates_lower_bound"]
    } for r in summary
]).to_csv(OUT / "file_summary.csv", index=False)

pd.DataFrame(sorted(labels_all.items()), columns=["raw_label", "count"]).to_csv(OUT / "label_counts.csv", index=False)
pd.DataFrame(sorted(model_counts.items()), columns=["model_class", "count"]).to_csv(OUT / "primary_class_counts.csv", index=False)
pd.DataFrame(sorted(missing_all.items()), columns=["column", "missing_count"]).to_csv(OUT / "missing_values.csv", index=False)
pd.DataFrame(sorted(inf_all.items()), columns=["column", "infinite_count"]).to_csv(OUT / "infinite_values.csv", index=False)
pd.DataFrame({"constant_candidate": constant_cols}).to_csv(OUT / "constant_candidates.csv", index=False)
pd.DataFrame({"review_for_leakage_or_identifier": leakage_candidates}).to_csv(OUT / "leakage_candidates.csv", index=False)

print("\nAudit complete. Raw files were not modified.")
print(f"Reports: {OUT}")
print("\nPrimary model-class counts before cleaning:")
for k, v in sorted(model_counts.items()):
    print(f"  {k:12s} {v:,}")
print("\nUnmapped/excluded labels:")
for k, v in sorted(excluded_counts.items()):
    print(f"  {k:24s} {v:,}")
