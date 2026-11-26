#!/usr/bin/env python3
"""One-time external validation of frozen 77-feature binary ANN on CIC-IDS2017."""
from pathlib import Path
import hashlib,json,math,platform,sys,time
import numpy as np,pandas as pd,tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,roc_auc_score,average_precision_score
MODEL=Path('results/cross_dataset_77/binary_ann/finalized_binary_ann_77_before_external.keras')
XPATH=Path('external-validation/CIC-IDS2017/prepared_77/X_external_binary_77_scaled.npy')
YPATH=Path('external-validation/CIC-IDS2017/prepared_77/y_external_binary_77.npy')
OUT=Path('results/cross_dataset_77/external_binary_cicids2017'); TH=0.5; BATCH=4096
if OUT.exists() and any(OUT.iterdir()): raise SystemExit(f'Refusing to overwrite {OUT}')
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def wilson(k,n,z=1.959963984540054):
 p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d; return [max(0,c-h),min(1,c+h)]
X=np.load(XPATH,mmap_mode='r'); y=np.load(YPATH,mmap_mode='r')
if X.shape[1]!=77 or set(np.unique(y).tolist())!={0,1}: raise RuntimeError('External arrays invalid')
m=keras.models.load_model(MODEL)
print('CIC-IDS2017 EXTERNAL BINARY VALIDATION'); print(f' rows: {len(y):,}'); print(' threshold: 0.50 frozen'); print(' no training/tuning')
t=time.time(); prob=m.predict(X,batch_size=BATCH,verbose=1).reshape(-1); sec=time.time()-t; pred=(prob>=TH).astype(np.int64)
tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
spec=tn/(tn+fp); fpr=fp/(tn+fp); fnr=fn/(fn+tp)
metrics={'n':int(len(y)),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred)),'recall':float(recall_score(y,pred)),'specificity':float(spec),'f1':float(f1_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'FPR':float(fpr),'FNR':float(fnr),'ROC_AUC':float(roc_auc_score(y,prob)),'PR_AUC':float(average_precision_score(y,prob)),'prediction_seconds':sec,'95CI':{'accuracy':wilson(tp+tn,len(y)),'recall':wilson(tp,tp+fn),'specificity':wilson(tn,tn+fp)}}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
pd.DataFrame([[tn,fp],[fn,tp]],index=['Actual_Benign','Actual_Intrusion'],columns=['Predicted_Benign','Predicted_Intrusion']).to_csv(OUT/'confusion_matrix.csv')
pd.DataFrame({'y_true':np.asarray(y),'prob_intrusion':prob,'y_pred':pred}).to_csv(OUT/'predictions.csv',index=False)
manifest={'evaluation':'CIC-IDS2017 external validation','features':77,'model_sha256':sha(MODEL),'X_sha256':sha(XPATH),'y_sha256':sha(YPATH),'threshold':TH,'training':False,'tuning':False,'external_scaler_fit':False,'tensorflow':tf.__version__,'python':sys.version,'platform':platform.platform()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
lines=[f'{sha(p)}  {p}' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt']; (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print('\nRESULTS'); print(json.dumps(metrics,indent=2)); print('Artifacts:',OUT)
