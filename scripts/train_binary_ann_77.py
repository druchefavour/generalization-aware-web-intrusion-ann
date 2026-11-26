#!/usr/bin/env python3
"""Train the 77-feature binary ANN for cross-dataset validation.
Drops Protocol by consuming the already-frozen 77-feature preprocessing artifacts.
Uses train + validation only. Does NOT load test data or CIC-IDS2017.
"""
from pathlib import Path
import hashlib, json, platform, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE=Path('processed/cross_dataset_77/preprocessing/binary')
OUT=Path('results/cross_dataset_77/binary_ann')
OUT.mkdir(parents=True,exist_ok=True)
SEED=42; BATCH=1024; MAX_EPOCHS=50; PATIENCE=5; LR=1e-3

keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism(); deterministic=True
except Exception: deterministic=False

Xtr=np.load(BASE/'X_train_scaled.npy',mmap_mode='r'); ytr=np.load(BASE/'y_train.npy',mmap_mode='r')
Xv=np.load(BASE/'X_validation_scaled.npy',mmap_mode='r'); yv=np.load(BASE/'y_validation.npy',mmap_mode='r')
if Xtr.shape[1]!=77 or Xv.shape[1]!=77: raise RuntimeError('Expected 77 predictors')
if set(np.unique(ytr).tolist())!={0,1}: raise RuntimeError('Unexpected labels')
weights={int(k):float(v) for k,v in json.loads((BASE/'class_weights.json').read_text()).items()}
preds=pd.read_csv(BASE/'predictor_order_77.csv')
if len(preds)!=77 or 'Protocol' in preds['feature'].tolist(): raise RuntimeError('77-feature manifest invalid')

model=keras.Sequential([
 keras.Input(shape=(77,),name='common_77_features'),
 layers.Dense(128,activation='relu'), layers.Dropout(0.30),
 layers.Dense(64,activation='relu'), layers.Dropout(0.20),
 layers.Dense(32,activation='relu'), layers.Dense(1,activation='sigmoid')
],name='binary_ann_77_cross_dataset')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),loss='binary_crossentropy',metrics=[
 keras.metrics.BinaryAccuracy(name='accuracy'),keras.metrics.Precision(name='precision'),
 keras.metrics.Recall(name='recall'),keras.metrics.AUC(name='roc_auc',curve='ROC'),keras.metrics.AUC(name='pr_auc',curve='PR')])

best=OUT/'best_binary_ann_77.keras'
cbs=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=PATIENCE,mode='min',restore_best_weights=True,verbose=1),
     keras.callbacks.ModelCheckpoint(best,monitor='val_loss',mode='min',save_best_only=True,verbose=1),
     keras.callbacks.CSVLogger(OUT/'training_log.csv'),keras.callbacks.TerminateOnNaN()]
print('Training 77-feature binary ANN')
print(f'  train rows: {len(ytr):,}'); print(f'  validation rows: {len(yv):,}'); print('  CIC-IDS2017 used: NO'); print('  test arrays loaded: NO')
start=time.time(); hist=model.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=MAX_EPOCHS,batch_size=BATCH,class_weight=weights,callbacks=cbs,shuffle=True,verbose=2); elapsed=time.time()-start
final=OUT/'finalized_binary_ann_77_before_external.keras'; model.save(final)
pd.DataFrame(hist.history).assign(epoch=lambda d:np.arange(1,len(d)+1)).to_csv(OUT/'history.csv',index=False)
vm={k:float(v) for k,v in model.evaluate(Xv,yv,batch_size=BATCH,verbose=0,return_dict=True).items()}; (OUT/'validation_metrics.json').write_text(json.dumps(vm,indent=2))
manifest={'experiment':'77_feature_binary_cross_dataset_model','seed':SEED,'features':77,'excluded_feature':'Protocol','train_rows':int(len(ytr)),'validation_rows':int(len(yv)),'test_loaded':False,'CIC_IDS2017_used':False,'class_weights':weights,'architecture':['Dense128 ReLU','Dropout0.30','Dense64 ReLU','Dropout0.20','Dense32 ReLU','Dense1 Sigmoid'],'optimizer':'Adam','learning_rate':LR,'batch_size':BATCH,'max_epochs':MAX_EPOCHS,'early_stopping_patience':PATIENCE,'epochs_completed':len(hist.history['loss']),'training_seconds':elapsed,'validation_metrics':vm,'tensorflow':tf.__version__,'python':sys.version,'platform':platform.platform(),'deterministic_ops':deterministic}
(OUT/'training_manifest.json').write_text(json.dumps(manifest,indent=2))
with open(OUT/'model_summary.txt','w') as f:model.summary(print_fn=lambda x:f.write(x+'\n'))

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
lines=[f'{sha(p)}  {p}' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt']; (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print('\nTraining complete'); print('epochs:',len(hist.history['loss'])); print('validation:',json.dumps(vm,indent=2)); print('final model:',final); print('CIC-IDS2017 remains unused.')
