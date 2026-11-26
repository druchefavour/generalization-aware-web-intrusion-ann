#!/usr/bin/env python3
"""Train the baseline attack-family ANN using frozen preprocessed arrays.

Classes (fixed by preprocessing manifest):
    0 = Web Attack
    1 = DoS
    2 = DDoS

Run from ANN-Web-Intrusion/Data:
    python train_attack_family_ann.py

Uses TRAIN + VALIDATION only. It deliberately does NOT load X_test/y_test.
No sampling, SMOTE, threshold tuning, or test-set evaluation is performed.
"""
from pathlib import Path
import hashlib, json, platform, sys, time
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as e:
    raise SystemExit(f"TensorFlow is unavailable in this environment: {e}")

BASE = Path("processed/preprocessing/attack_family")
OUT = Path("results/attack_family_ann_baseline")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
BATCH_SIZE = 512
MAX_EPOCHS = 75
PATIENCE = 8
LEARNING_RATE = 5e-4
CLASS_NAMES = ["Web Attack", "DoS", "DDoS"]

REQUIRED = [
    "X_train_scaled.npy", "y_train.npy",
    "X_validation_scaled.npy", "y_validation.npy",
    "class_weights.json", "preprocessing_manifest.json",
    "predictor_order.csv", "target_mapping.json"
]
for name in REQUIRED:
    if not (BASE / name).exists():
        raise SystemExit(f"Missing required preprocessing artifact: {BASE/name}")

keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
    deterministic_ops = True
except Exception:
    deterministic_ops = False

# Training and validation only. Holdout test arrays are intentionally not loaded.
X_train = np.load(BASE / "X_train_scaled.npy", mmap_mode="r")
y_train = np.load(BASE / "y_train.npy", mmap_mode="r")
X_val = np.load(BASE / "X_validation_scaled.npy", mmap_mode="r")
y_val = np.load(BASE / "y_validation.npy", mmap_mode="r")

if X_train.ndim != 2 or X_val.ndim != 2:
    raise ValueError("Expected 2-D predictor arrays")
if X_train.shape[1] != X_val.shape[1]:
    raise ValueError("Training and validation feature counts differ")
if len(X_train) != len(y_train) or len(X_val) != len(y_val):
    raise ValueError("X/y row count mismatch")
if set(np.unique(y_train).tolist()) != {0, 1, 2}:
    raise ValueError(f"Unexpected training classes: {np.unique(y_train)}")

mapping = json.loads((BASE / "target_mapping.json").read_text(encoding="utf-8"))
expected_mapping = {"Web Attack": 0, "DoS": 1, "DDoS": 2}
if mapping != expected_mapping:
    raise ValueError(f"Unexpected target mapping: {mapping}")

cw_payload = json.loads((BASE / "class_weights.json").read_text(encoding="utf-8"))
class_weight = {int(k): float(v) for k, v in cw_payload["integer_class_weights"].items()}

predictors = pd.read_csv(BASE / "predictor_order.csv")
if len(predictors) != X_train.shape[1]:
    raise ValueError("Predictor order does not match X_train feature count")

# Modest MLP. L2 and dropout are retained because Web Attack has only 236 training rows.
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],), name="cicflowmeter_features"),
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5), name="dense_128"),
    layers.Dropout(0.30, name="dropout_30"),
    layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5), name="dense_64"),
    layers.Dropout(0.25, name="dropout_25"),
    layers.Dense(32, activation="relu", name="dense_32"),
    layers.Dense(3, activation="softmax", name="attack_family_probabilities"),
], name="attack_family_ann")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

best_model_path = OUT / "best_attack_family_ann.keras"
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path, monitor="val_loss", mode="min",
        save_best_only=True, save_weights_only=False, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min", factor=0.5, patience=3,
        min_lr=1e-6, verbose=1
    ),
    keras.callbacks.CSVLogger(OUT / "training_log.csv", append=False),
    keras.callbacks.TerminateOnNaN(),
]

print("Training baseline attack-family ANN")
print(f"  train rows: {len(y_train):,}")
print(f"  validation rows: {len(y_val):,}")
print(f"  predictors: {X_train.shape[1]}")
print(f"  class mapping: {expected_mapping}")
print(f"  class weights: {class_weight}")
print("  holdout test arrays: NOT LOADED")

start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    shuffle=True,
    verbose=2,
)
elapsed = time.time() - start

final_dev_model = OUT / "finalized_attack_family_ann_before_test.keras"
model.save(final_dev_model)

hist = pd.DataFrame(history.history)
hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
hist.to_csv(OUT / "history.csv", index=False)

# Development-stage validation predictions and metrics only.
val_prob = model.predict(X_val, batch_size=2048, verbose=0)
val_pred = np.argmax(val_prob, axis=1)
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
cm = confusion_matrix(y_val, val_pred, labels=[0,1,2])
pd.DataFrame(cm, index=[f"Actual_{x}" for x in CLASS_NAMES], columns=[f"Predicted_{x}" for x in CLASS_NAMES]).to_csv(OUT / "validation_confusion_matrix.csv")
report = classification_report(y_val, val_pred, labels=[0,1,2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
(OUT / "validation_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
val_metrics = {
    "accuracy": float(np.mean(val_pred == y_val)),
    "balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
    "macro_precision": float(precision_score(y_val, val_pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(y_val, val_pred, average="macro", zero_division=0)),
    "macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(y_val, val_pred, average="weighted", zero_division=0)),
    "per_class_recall": {CLASS_NAMES[i]: float(recall_score(y_val == i, val_pred == i, zero_division=0)) for i in range(3)},
}
(OUT / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

with open(OUT / "model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
(OUT / "model_config.json").write_text(model.to_json(), encoding="utf-8")

manifest = {
    "experiment": "attack_family_ann_baseline",
    "status": "model development finalized before holdout-test evaluation",
    "seed": SEED,
    "tensorflow_version": tf.__version__,
    "keras_version": getattr(keras, "__version__", "bundled_with_tensorflow"),
    "python_version": sys.version,
    "platform": platform.platform(),
    "deterministic_tensorflow_ops_requested": deterministic_ops,
    "input_features": int(X_train.shape[1]),
    "train_rows": int(len(y_train)),
    "validation_rows": int(len(y_val)),
    "test_arrays_loaded": False,
    "class_mapping": expected_mapping,
    "training_class_counts": {CLASS_NAMES[i]: int(np.sum(y_train == i)) for i in range(3)},
    "class_weights": class_weight,
    "architecture": [
        "Dense(128, relu, L2=1e-5)", "Dropout(0.30)",
        "Dense(64, relu, L2=1e-5)", "Dropout(0.25)",
        "Dense(32, relu)", "Dense(3, softmax)"
    ],
    "optimizer": "Adam(clipnorm=1.0)",
    "initial_learning_rate": LEARNING_RATE,
    "loss": "sparse_categorical_crossentropy",
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "reduce_lr_on_plateau": {"factor": 0.5, "patience": 3, "min_lr": 1e-6},
    "epochs_completed": int(len(history.history["loss"])),
    "training_seconds": elapsed,
    "validation_metrics": val_metrics,
    "important_limitation": "Only 236 unique Web Attack observations are available for training and 51 for validation. The balanced Web Attack class weight is approximately 342.27, so validation macro metrics and the Web Attack confusion pattern are more informative than overall accuracy.",
    "development_policy": "Training and validation only. Frozen test arrays were not loaded. No SMOTE, oversampling, undersampling, or test-set tuning was performed.",
}
(OUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def sha256(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

lines=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name != "TRAINING_SHA256SUMS.txt":
        lines.append(f"{sha256(p)}  {p}")
(OUT / "TRAINING_SHA256SUMS.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")

print("\nTraining complete.")
print(f"Epochs completed: {len(history.history['loss'])}")
print(f"Training time: {elapsed/60:.2f} minutes")
print("Validation metrics:")
for k,v in val_metrics.items():
    print(f"  {k}: {v}")
print(f"Best checkpoint: {best_model_path}")
print(f"Finalized pre-test model: {final_dev_model}")
print(f"Checksums: {OUT/'TRAINING_SHA256SUMS.txt'}")
print("Frozen attack-family holdout test set was NOT evaluated.")
