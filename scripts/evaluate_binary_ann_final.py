#!/usr/bin/env python3
"""One-time final evaluation of the frozen binary ANN holdout test set.

Run once from ANN-Web-Intrusion/Data:
    python evaluate_binary_ann_final.py

This script does not train or tune the model. It uses the prespecified threshold 0.50,
saves test probabilities/predictions, performance metrics, Wilson 95% CIs for key
proportions, and SHA-256 hashes for all evaluation artifacts.
"""
from pathlib import Path
import hashlib, json, math, platform, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score
)
import tensorflow as tf
from tensorflow import keras

MODEL = Path("results/binary_ann_baseline/finalized_binary_ann_before_test.keras")
PREP = Path("processed/preprocessing/binary")
XTEST = PREP / "X_test_scaled.npy"
YTEST = PREP / "y_test.npy"
TRAIN_CHECKSUMS = Path("results/binary_ann_baseline/TRAINING_SHA256SUMS.txt")
PREP_CHECKSUMS = Path("processed/preprocessing/PREPROCESSING_SHA256SUMS.txt")
OUT = Path("results/binary_ann_final_test")
THRESHOLD = 0.50
BATCH_SIZE = 4096

if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(
        f"Refusing to overwrite existing final-test outputs in {OUT}. "
        "This safeguard helps preserve the one-time holdout evaluation."
    )
OUT.mkdir(parents=True, exist_ok=True)

for p in [MODEL, XTEST, YTEST, TRAIN_CHECKSUMS, PREP_CHECKSUMS]:
    if not p.exists():
        raise SystemExit(f"Missing required artifact: {p}")

def sha256(path, block=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def parse_checksums(path):
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        digest, name = line.split(None,1)
        out[name.strip()] = digest
    return out

def verify_from_manifest(checksum_file, target):
    entries=parse_checksums(checksum_file)
    key=str(target)
    if key not in entries:
        raise RuntimeError(f"{target} not found in {checksum_file}")
    actual=sha256(target)
    if actual != entries[key]:
        raise RuntimeError(f"Checksum mismatch for {target}")
    return actual

# Verify frozen test arrays and finalized model before first use.
model_sha = verify_from_manifest(TRAIN_CHECKSUMS, MODEL)
x_sha = verify_from_manifest(PREP_CHECKSUMS, XTEST)
y_sha = verify_from_manifest(PREP_CHECKSUMS, YTEST)

X=np.load(XTEST, mmap_mode="r")
y=np.load(YTEST, mmap_mode="r")
if X.ndim != 2 or len(X) != len(y):
    raise ValueError("Invalid holdout array shapes")
if set(np.unique(y).tolist()) != {0,1}:
    raise ValueError(f"Unexpected test labels: {np.unique(y)}")

model=keras.models.load_model(MODEL)
if model.input_shape[-1] != X.shape[1]:
    raise ValueError("Model input feature count does not match test predictors")

print("FINAL HOLDOUT EVALUATION")
print(f"  rows: {len(y):,}")
print(f"  predictors: {X.shape[1]}")
print(f"  threshold: {THRESHOLD:.2f} (prespecified)")
print("  no training or tuning will occur")

start=time.time()
prob=model.predict(X, batch_size=BATCH_SIZE, verbose=1).reshape(-1).astype(np.float64)
elapsed=time.time()-start
pred=(prob >= THRESHOLD).astype(np.int64)

tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()

def safe_div(a,b): return float(a/b) if b else float("nan")

def wilson(k,n,z=1.959963984540054):
    if n == 0: return [float("nan"), float("nan")]
    p=k/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [float(max(0,center-half)), float(min(1,center+half))]

accuracy=accuracy_score(y,pred)
precision=precision_score(y,pred,zero_division=0)
recall=recall_score(y,pred,zero_division=0)
specificity=safe_div(tn,tn+fp)
f1=f1_score(y,pred,zero_division=0)
bal_acc=balanced_accuracy_score(y,pred)
fpr=safe_div(fp,fp+tn)
fnr=safe_div(fn,fn+tp)
roc=roc_auc_score(y,prob)
pr=average_precision_score(y,prob)
npv=safe_div(tn,tn+fn)

metrics={
    "threshold":THRESHOLD,"n_test":int(len(y)),
    "true_negatives":int(tn),"false_positives":int(fp),
    "false_negatives":int(fn),"true_positives":int(tp),
    "accuracy":float(accuracy),"precision_ppv":float(precision),
    "recall_sensitivity":float(recall),"specificity":float(specificity),
    "negative_predictive_value":float(npv),"f1_score":float(f1),
    "balanced_accuracy":float(bal_acc),"false_positive_rate":float(fpr),
    "false_negative_rate":float(fnr),"roc_auc":float(roc),"pr_auc_average_precision":float(pr),
    "prediction_seconds":elapsed,
    "wilson_95_ci":{
        "accuracy":wilson(tp+tn,len(y)),
        "precision_ppv":wilson(tp,tp+fp),
        "recall_sensitivity":wilson(tp,tp+fn),
        "specificity":wilson(tn,tn+fp),
        "false_positive_rate":wilson(fp,fp+tn),
        "false_negative_rate":wilson(fn,fn+tp),
    }
}
(OUT/"final_test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")

pd.DataFrame([[tn,fp],[fn,tp]], index=["Actual_Benign","Actual_Intrusion"], columns=["Predicted_Benign","Predicted_Intrusion"]).to_csv(OUT/"confusion_matrix.csv")
pd.DataFrame({"RowIndex":np.arange(len(y)),"y_true":np.asarray(y,dtype=np.int64),"intrusion_probability":prob,"y_pred":pred}).to_csv(OUT/"test_predictions.csv",index=False)

summary=pd.DataFrame([{
    "n_test":len(y),"TN":tn,"FP":fp,"FN":fn,"TP":tp,
    "accuracy":accuracy,"precision":precision,"recall":recall,"specificity":specificity,
    "f1":f1,"balanced_accuracy":bal_acc,"FPR":fpr,"FNR":fnr,"ROC_AUC":roc,"PR_AUC":pr
}])
summary.to_csv(OUT/"final_test_summary.csv",index=False)

manifest={
    "evaluation":"one-time frozen binary holdout evaluation",
    "model":str(MODEL),"model_sha256":model_sha,
    "X_test":str(XTEST),"X_test_sha256":x_sha,
    "y_test":str(YTEST),"y_test_sha256":y_sha,
    "threshold":THRESHOLD,"threshold_policy":"prespecified before holdout evaluation; no test-set threshold tuning",
    "batch_size":BATCH_SIZE,"tensorflow_version":tf.__version__,"python_version":sys.version,"platform":platform.platform(),
    "training_performed":False,"hyperparameter_tuning_performed":False,
    "post_test_policy":"Do not modify the baseline binary model based on these holdout results. Alternative models must be reported as separate experiments.",
    "confidence_intervals":"Wilson score 95% intervals for binomial proportions; AUC confidence intervals are not computed by this script.",
}
(OUT/"final_evaluation_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

lines=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name != "FINAL_EVALUATION_SHA256SUMS.txt":
        lines.append(f"{sha256(p)}  {p}")
(OUT/"FINAL_EVALUATION_SHA256SUMS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

print("\nFINAL HOLDOUT RESULTS")
print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
for k in ["accuracy","precision_ppv","recall_sensitivity","specificity","f1_score","balanced_accuracy","false_positive_rate","false_negative_rate","roc_auc","pr_auc_average_precision"]:
    print(f"  {k}: {metrics[k]:.8f}")
print(f"\nArtifacts: {OUT}")
print("No training, threshold tuning, or model selection was performed on the holdout set.")
