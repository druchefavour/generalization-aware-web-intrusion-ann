#!/usr/bin/env python3
"""Training-only preprocessing for frozen ANN-Web-Intrusion partitions.

Run from ANN-Web-Intrusion/Data:
    python preprocess_partitions.py

For each experiment (binary, attack_family), this script:
- reads the frozen train/validation/test CSVs;
- reads predictor order from the split manifest;
- verifies forbidden metadata/targets are not predictors;
- fits StandardScaler ONLY on training predictors;
- applies that fitted scaler to validation and test predictors;
- computes balanced class weights ONLY from training targets;
- encodes targets deterministically;
- saves scaler parameters/artifact, class weights, target mapping, predictor order;
- writes scaled arrays as .npy files and hashes all artifacts;
- never modifies the frozen partition CSVs;
- performs no ANN training, sampling, SMOTE, or feature selection.
"""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PARTITIONS = Path("processed/partitions")
OUT = Path("processed/preprocessing")
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "binary": {
        "target": "BinaryTarget",
        "class_order": ["Benign", "Intrusion"],
    },
    "attack_family": {
        "target": "AttackFamilyTarget",
        "class_order": ["Web Attack", "DoS", "DDoS"],
    },
}

FORBIDDEN_BASE = {
    "SourceFile", "Label", "ModelClass", "RowID",
    "BinaryTarget", "AttackFamilyTarget"
}

def sha256(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_manifest(exp):
    p = PARTITIONS / exp / "split_manifest.json"
    if not p.exists():
        raise SystemExit(f"Missing split manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8")), p

def verify_partition_hashes(manifest):
    for split in ("train", "validation", "test"):
        p = Path(manifest["files"][split]["path"])
        expected = manifest["files"][split]["sha256"]
        actual = sha256(p)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {p}: expected {expected}, got {actual}")

def save_npy(path, arr):
    np.save(path, arr, allow_pickle=False)
    return {"path": str(path), "sha256": sha256(path), "shape": list(arr.shape), "dtype": str(arr.dtype)}

for exp, cfg in EXPERIMENTS.items():
    print(f"\nPreprocessing {exp} experiment...")
    manifest, manifest_path = load_manifest(exp)
    verify_partition_hashes(manifest)

    predictors = list(manifest["predictor_columns"])
    forbidden_found = sorted(set(predictors) & FORBIDDEN_BASE)
    if forbidden_found:
        raise RuntimeError(f"Forbidden columns found in predictor list for {exp}: {forbidden_found}")
    if not predictors:
        raise RuntimeError(f"No predictors recorded for {exp}")

    target = cfg["target"]
    class_order = cfg["class_order"]
    class_to_int = {c: i for i, c in enumerate(class_order)}

    edir = OUT / exp
    edir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for split in ("train", "validation", "test"):
        p = Path(manifest["files"][split]["path"])
        usecols = predictors + [target]
        df = pd.read_csv(p, usecols=usecols, low_memory=False)
        missing_cols = sorted(set(usecols) - set(df.columns))
        if missing_cols:
            raise RuntimeError(f"Missing columns in {p}: {missing_cols}")
        # Predictors should already be finite numeric values after cleaning.
        X = df[predictors].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(X).all():
            raise RuntimeError(f"Non-finite predictor found in frozen {split} split for {exp}")
        y_text = df[target].astype(str)
        unexpected = sorted(set(y_text.unique()) - set(class_order))
        if unexpected:
            raise RuntimeError(f"Unexpected target classes in {split}/{exp}: {unexpected}")
        y = y_text.map(class_to_int).to_numpy(dtype=np.int64)
        frames[split] = (X, y, y_text)

    # Fit ONLY on training X.
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(frames["train"][0])

    scaler_path = edir / "standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Compute balanced class weights ONLY on training labels.
    y_train = frames["train"][1]
    present_classes = np.unique(y_train)
    expected_classes = np.arange(len(class_order), dtype=np.int64)
    if not np.array_equal(present_classes, expected_classes):
        raise RuntimeError(f"Training classes incomplete for {exp}: {present_classes}")
    weights = compute_class_weight(class_weight="balanced", classes=expected_classes, y=y_train)
    class_weights = {str(int(i)): float(w) for i, w in zip(expected_classes, weights)}

    artifacts = {}
    distributions = {}
    for split, (X, y, y_text) in frames.items():
        Xs = scaler.transform(X).astype(np.float32)
        artifacts[f"X_{split}"] = save_npy(edir / f"X_{split}_scaled.npy", Xs)
        artifacts[f"y_{split}"] = save_npy(edir / f"y_{split}.npy", y.astype(np.int64))
        vc = y_text.value_counts()
        distributions[split] = {c: int(vc.get(c, 0)) for c in class_order}

        # Sanity diagnostics only; validation/test are NOT used to fit scaler.
        del Xs

    # Human-readable scaler parameters in exact predictor order.
    scaler_params = pd.DataFrame({
        "feature_index": np.arange(len(predictors)),
        "feature": predictors,
        "train_mean": scaler.mean_,
        "train_scale": scaler.scale_,
        "train_variance": scaler.var_,
    })
    scaler_params_path = edir / "scaler_parameters.csv"
    scaler_params.to_csv(scaler_params_path, index=False)

    predictor_path = edir / "predictor_order.csv"
    pd.DataFrame({"feature_index": range(len(predictors)), "feature": predictors}).to_csv(predictor_path, index=False)

    target_map_path = edir / "target_mapping.json"
    target_map_path.write_text(json.dumps(class_to_int, indent=2), encoding="utf-8")

    weights_path = edir / "class_weights.json"
    weights_payload = {
        "method": "sklearn.compute_class_weight(class_weight='balanced') on training labels only",
        "integer_class_weights": class_weights,
        "named_class_weights": {class_order[i]: float(weights[i]) for i in range(len(class_order))},
        "formula": "n_samples / (n_classes * np.bincount(y))"
    }
    weights_path.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

    # Diagnostics useful for leakage auditing.
    train_scaled = np.load(edir / "X_train_scaled.npy", mmap_mode="r")
    # Float32 output means near-zero/near-one is expected rather than mathematical exactness.
    diagnostics = {
        "training_scaled_mean_abs_max": float(np.max(np.abs(np.mean(train_scaled, axis=0, dtype=np.float64)))),
        "training_scaled_std_min": float(np.min(np.std(train_scaled, axis=0, dtype=np.float64))),
        "training_scaled_std_max": float(np.max(np.std(train_scaled, axis=0, dtype=np.float64))),
    }

    fixed_artifacts = {
        "scaler_joblib": {"path": str(scaler_path), "sha256": sha256(scaler_path)},
        "scaler_parameters": {"path": str(scaler_params_path), "sha256": sha256(scaler_params_path)},
        "predictor_order": {"path": str(predictor_path), "sha256": sha256(predictor_path)},
        "target_mapping": {"path": str(target_map_path), "sha256": sha256(target_map_path)},
        "class_weights": {"path": str(weights_path), "sha256": sha256(weights_path)},
    }

    prep_manifest = {
        "experiment": exp,
        "partition_manifest": str(manifest_path),
        "partition_manifest_sha256": sha256(manifest_path),
        "partition_csv_checksums_verified_before_preprocessing": True,
        "target_column": target,
        "class_order": class_order,
        "target_mapping": class_to_int,
        "predictor_count": len(predictors),
        "predictor_columns": predictors,
        "forbidden_columns_excluded_from_X": sorted(FORBIDDEN_BASE),
        "scaler": "sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)",
        "scaler_fit_data": "training predictors only",
        "validation_and_test_policy": "transform only using training-fitted scaler; never used in fit",
        "class_weight_fit_data": "training labels only",
        "class_weights": class_weights,
        "class_distributions": distributions,
        "output_array_dtype": {"X": "float32", "y": "int64"},
        "diagnostics": diagnostics,
        "artifacts": {**artifacts, **fixed_artifacts},
        "not_performed": ["imputation", "sampling", "undersampling", "oversampling", "SMOTE", "feature selection", "ANN training", "test-set model selection"],
    }
    pm = edir / "preprocessing_manifest.json"
    pm.write_text(json.dumps(prep_manifest, indent=2), encoding="utf-8")

    print(f"  predictors: {len(predictors)}")
    print(f"  class weights: {weights_payload['named_class_weights']}")
    print(f"  output: {edir}")

# Consolidated artifact checksums (excluding manifests that would self-reference).
checksum_lines=[]
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name not in {"PREPROCESSING_SHA256SUMS.txt"}:
        checksum_lines.append(f"{sha256(p)}  {p}")
checksum_path = OUT / "PREPROCESSING_SHA256SUMS.txt"
checksum_path.write_text("\n".join(checksum_lines)+"\n", encoding="utf-8")

print("\nTraining-only preprocessing complete.")
print(f"Checksums: {checksum_path}")
print("No ANN was trained and no frozen partition CSV was modified.")
