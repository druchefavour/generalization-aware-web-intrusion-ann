#!/usr/bin/env python3
"""One-time final evaluation of the frozen attack-family ANN holdout test set.

Classes:
    0 = Web Attack
    1 = DoS
    2 = DDoS

Run once from ANN-Web-Intrusion/Data:
    python evaluate_attack_family_ann_final.py

The script performs no training, retuning, resampling, or model selection.
Predictions use argmax over the three softmax probabilities.
"""
from pathlib import Path
import hashlib, json, math, platform, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)
from sklearn.preprocessing import label_binarize

MODEL = Path("results/attack_family_ann_baseline/finalized_attack_family_ann_before_test.keras")
TRAIN_SUMS = Path("results/attack_family_ann_baseline/TRAINING_SHA256SUMS.txt")
PREP = Path("processed/preprocessing/attack_family")
PREP_SUMS = Path("processed/preprocessing/PREPROCESSING_SHA256SUMS.txt")
XTEST = PREP / "X_test_scaled.npy"
YTEST = PREP / "y_test.npy"
OUT = Path("results/attack_family_ann_final_test")
BATCH_SIZE = 4096
CLASS_NAMES = ["Web Attack", "DoS", "DDoS"]
LABELS = [0, 1, 2]

if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(
        f"Refusing to overwrite existing final-test outputs in {OUT}. "
        "This safeguard preserves the one-time holdout evaluation."
    )
OUT.mkdir(parents=True, exist_ok=True)

for p in [MODEL, TRAIN_SUMS, PREP_SUMS, XTEST, YTEST]:
    if not p.exists():
        raise SystemExit(f"Missing required artifact: {p}")

def sha256(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def parse_sums(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out

def verify(checksum_file, target):
    entries = parse_sums(checksum_file)
    key = str(target)
    if key not in entries:
        raise RuntimeError(f"{key} is absent from {checksum_file}")
    actual = sha256(target)
    if entries[key] != actual:
        raise RuntimeError(f"Checksum mismatch for {target}")
    return actual

def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return [float(max(0, center-half)), float(min(1, center+half))]

model_sha = verify(TRAIN_SUMS, MODEL)
x_sha = verify(PREP_SUMS, XTEST)
y_sha = verify(PREP_SUMS, YTEST)

X = np.load(XTEST, mmap_mode="r")
y = np.load(YTEST, mmap_mode="r")
if X.ndim != 2 or len(X) != len(y):
    raise ValueError("Invalid holdout array shapes")
if set(np.unique(y).tolist()) != {0, 1, 2}:
    raise ValueError(f"Unexpected test labels: {np.unique(y)}")

model = keras.models.load_model(MODEL)
if model.input_shape[-1] != X.shape[1] or model.output_shape[-1] != 3:
    raise ValueError("Model shape does not match the frozen attack-family test data")

print("FINAL ATTACK-FAMILY HOLDOUT EVALUATION")
print(f"  rows: {len(y):,}")
print(f"  predictors: {X.shape[1]}")
print("  prediction rule: argmax of 3-class softmax")
print("  no training or tuning will occur")

start = time.time()
prob = model.predict(X, batch_size=BATCH_SIZE, verbose=1).astype(np.float64)
elapsed = time.time() - start
if prob.shape != (len(y), 3):
    raise ValueError(f"Unexpected probability shape: {prob.shape}")
pred = np.argmax(prob, axis=1).astype(np.int64)

cm = confusion_matrix(y, pred, labels=LABELS)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in CLASS_NAMES],
    columns=[f"Predicted_{c}" for c in CLASS_NAMES],
)
cm_df.to_csv(OUT / "confusion_matrix.csv")

report = classification_report(
    y, pred, labels=LABELS, target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)
(OUT / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# One-vs-rest probabilities for multiclass AUC/AP.
y_bin = label_binarize(y, classes=LABELS)
per_class = {}
for i, name in enumerate(CLASS_NAMES):
    tp = int(cm[i, i])
    fn = int(cm[i, :].sum() - tp)
    fp = int(cm[:, i].sum() - tp)
    tn = int(cm.sum() - tp - fn - fp)
    support = tp + fn
    predicted_positive = tp + fp
    recall = tp / support if support else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / predicted_positive if predicted_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    per_class[name] = {
        "support": support,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity_one_vs_rest": specificity,
        "f1_score": f1,
        "roc_auc_one_vs_rest": float(roc_auc_score(y_bin[:, i], prob[:, i])),
        "average_precision_one_vs_rest": float(average_precision_score(y_bin[:, i], prob[:, i])),
        "recall_wilson_95_ci": wilson(tp, support),
        "precision_wilson_95_ci": wilson(tp, predicted_positive),
        "specificity_wilson_95_ci": wilson(tn, tn + fp),
    }

metrics = {
    "n_test": int(len(y)),
    "accuracy": float(accuracy_score(y, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
    "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    "weighted_precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
    "weighted_recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
    "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    "multiclass_log_loss": float(log_loss(y, prob, labels=LABELS)),
    "macro_roc_auc_ovr": float(roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")),
    "weighted_roc_auc_ovr": float(roc_auc_score(y_bin, prob, average="weighted", multi_class="ovr")),
    "macro_average_precision_ovr": float(average_precision_score(y_bin, prob, average="macro")),
    "weighted_average_precision_ovr": float(average_precision_score(y_bin, prob, average="weighted")),
    "accuracy_wilson_95_ci": wilson(int(np.trace(cm)), int(cm.sum())),
    "prediction_seconds": elapsed,
    "per_class": per_class,
}
(OUT / "final_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

pd.DataFrame({
    "RowIndex": np.arange(len(y), dtype=np.int64),
    "y_true": np.asarray(y, dtype=np.int64),
    "y_pred": pred,
    "prob_Web_Attack": prob[:, 0],
    "prob_DoS": prob[:, 1],
    "prob_DDoS": prob[:, 2],
}).to_csv(OUT / "test_predictions.csv", index=False)

summary = {
    "n_test": len(y),
    "correct": int(np.trace(cm)),
    "incorrect": int(cm.sum() - np.trace(cm)),
    "accuracy": metrics["accuracy"],
    "balanced_accuracy": metrics["balanced_accuracy"],
    "macro_f1": metrics["macro_f1"],
    "weighted_f1": metrics["weighted_f1"],
    "macro_roc_auc_ovr": metrics["macro_roc_auc_ovr"],
    "macro_average_precision_ovr": metrics["macro_average_precision_ovr"],
}
pd.DataFrame([summary]).to_csv(OUT / "final_test_summary.csv", index=False)

manifest = {
    "evaluation": "one-time frozen attack-family holdout evaluation",
    "class_mapping": {"Web Attack": 0, "DoS": 1, "DDoS": 2},
    "prediction_rule": "argmax over three softmax probabilities",
    "model": str(MODEL),
    "model_sha256": model_sha,
    "X_test": str(XTEST),
    "X_test_sha256": x_sha,
    "y_test": str(YTEST),
    "y_test_sha256": y_sha,
    "batch_size": BATCH_SIZE,
    "tensorflow_version": tf.__version__,
    "python_version": sys.version,
    "platform": platform.platform(),
    "training_performed": False,
    "hyperparameter_tuning_performed": False,
    "confidence_intervals": "Wilson score 95% intervals for accuracy and per-class precision, recall, and one-vs-rest specificity. AUC confidence intervals are not computed.",
    "post_test_policy": "Do not modify this baseline model based on holdout results. Any alternative weighting or architecture must be reported as a separate sensitivity experiment.",
    "interpretation_limit": "Attack families are scenario/day-specific within the selected CSE-CIC-IDS2018 subset. Internal holdout performance does not establish cross-dataset or cross-environment generalization.",
}
(OUT / "final_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

lines = []
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name != "FINAL_EVALUATION_SHA256SUMS.txt":
        lines.append(f"{sha256(p)}  {p}")
(OUT / "FINAL_EVALUATION_SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\nFINAL ATTACK-FAMILY HOLDOUT RESULTS")
print(cm_df.to_string())
print(f"\n  accuracy: {metrics['accuracy']:.8f}")
print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.8f}")
print(f"  macro_f1: {metrics['macro_f1']:.8f}")
print(f"  weighted_f1: {metrics['weighted_f1']:.8f}")
print(f"  macro_roc_auc_ovr: {metrics['macro_roc_auc_ovr']:.8f}")
print(f"  macro_average_precision_ovr: {metrics['macro_average_precision_ovr']:.8f}")
for name in CLASS_NAMES:
    pc = per_class[name]
    print(f"  {name} recall: {pc['recall_sensitivity']:.8f} ({pc['true_positives']}/{pc['support']})")
print(f"\nArtifacts: {OUT}")
print("No training, tuning, resampling, or model selection was performed on the holdout set.")
