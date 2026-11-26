#!/usr/bin/env python3
"""Train the baseline binary ANN using frozen, training-only preprocessed arrays.

Run from ANN-Web-Intrusion/Data:
    python train_binary_ann.py

Uses TRAIN + VALIDATION only. It deliberately does NOT load X_test/y_test.
The frozen holdout test set remains untouched until a separate final-evaluation step.
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
    raise SystemExit(
        "TensorFlow is not installed in this Python environment. Install TensorFlow first, then rerun. "
        f"Original error: {e}"
    )

BASE = Path("processed/preprocessing/binary")
OUT = Path("results/binary_ann_baseline")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
BATCH_SIZE = 1024
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-3

REQUIRED = [
    "X_train_scaled.npy", "y_train.npy",
    "X_validation_scaled.npy", "y_validation.npy",
    "class_weights.json", "preprocessing_manifest.json",
    "predictor_order.csv"
]
for name in REQUIRED:
    if not (BASE / name).exists():
        raise SystemExit(f"Missing required preprocessing artifact: {BASE/name}")

# Reproducibility controls. Keras documents set_random_seed as setting Python,
# NumPy, and backend seeds. Deterministic TF ops are requested where supported.
keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
    deterministic_ops = True
except Exception:
    deterministic_ops = False

# Load only training and validation arrays. Test arrays are intentionally untouched.
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
if set(np.unique(y_train).tolist()) != {0, 1}:
    raise ValueError(f"Unexpected training labels: {np.unique(y_train)}")

with open(BASE / "class_weights.json", "r", encoding="utf-8") as f:
    cw_payload = json.load(f)
class_weight = {int(k): float(v) for k, v in cw_payload["integer_class_weights"].items()}

predictors = pd.read_csv(BASE / "predictor_order.csv")
if len(predictors) != X_train.shape[1]:
    raise ValueError("Predictor-order manifest does not match X_train feature count")

model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],), name="cicflowmeter_features"),
    layers.Dense(128, activation="relu", name="dense_128"),
    layers.Dropout(0.30, name="dropout_30"),
    layers.Dense(64, activation="relu", name="dense_64"),
    layers.Dropout(0.20, name="dropout_20"),
    layers.Dense(32, activation="relu", name="dense_32"),
    layers.Dense(1, activation="sigmoid", name="intrusion_probability"),
], name="binary_intrusion_ann")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="roc_auc", curve="ROC"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)

best_model_path = OUT / "best_binary_ann.keras"
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path, monitor="val_loss", mode="min",
        save_best_only=True, save_weights_only=False, verbose=1
    ),
    keras.callbacks.CSVLogger(OUT / "training_log.csv", append=False),
    keras.callbacks.TerminateOnNaN(),
]

print("Training baseline binary ANN")
print(f"  train rows: {len(y_train):,}")
print(f"  validation rows: {len(y_val):,}")
print(f"  predictors: {X_train.shape[1]}")
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

# The in-memory model has best validation-loss weights due to restore_best_weights=True.
# Save a distinct finalized-development artifact; no test result is consulted here.
final_dev_model = OUT / "finalized_binary_ann_before_test.keras"
model.save(final_dev_model)

pd.DataFrame(history.history).assign(epoch=lambda d: np.arange(1, len(d)+1)).to_csv(
    OUT / "history.csv", index=False
)

# Validation-only metrics are allowed during development.
val_values = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
val_metrics = {k: float(v) for k, v in val_values.items()}
(OUT / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

# Architecture details.
with open(OUT / "model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
(OUT / "model_config.json").write_text(model.to_json(), encoding="utf-8")

# Record environment and experimental configuration.
run_manifest = {
    "experiment": "binary_ann_baseline",
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
    "architecture": [
        "Dense(128, relu)", "Dropout(0.30)",
        "Dense(64, relu)", "Dropout(0.20)",
        "Dense(32, relu)", "Dense(1, sigmoid)"
    ],
    "optimizer": "Adam",
    "initial_learning_rate": LEARNING_RATE,
    "loss": "binary_crossentropy",
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_monitor": "val_loss",
    "early_stopping_patience": PATIENCE,
    "restore_best_weights": True,
    "class_weights": class_weight,
    "epochs_completed": int(len(history.history["loss"])),
    "training_seconds": elapsed,
    "validation_metrics": val_metrics,
    "development_policy": "Architecture/training uses train and validation only. Frozen holdout test data is reserved for a separate one-time final evaluation.",
}
(OUT / "training_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

# SHA-256 all result artifacts except checksum file itself.
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
    print(f"  {k}: {v:.6f}")
print(f"Best checkpoint: {best_model_path}")
print(f"Finalized pre-test model: {final_dev_model}")
print(f"Checksums: {OUT/'TRAINING_SHA256SUMS.txt'}")
print("Frozen holdout test set was NOT evaluated.")
