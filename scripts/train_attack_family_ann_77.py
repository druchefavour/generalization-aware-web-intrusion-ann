#!/usr/bin/env python3
"""Train the 77-feature attack-family ANN for cross-dataset validation.

Classes:
  0 = Web Attack
  1 = DoS
  2 = DDoS

Consumes only the already-frozen 77-feature CSE-CIC-IDS2018 preprocessing
artifacts. Uses training + validation only. It does NOT load either test set
and does NOT use CIC-IDS2017.
"""
from pathlib import Path
import hashlib, json, platform, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, precision_score, recall_score, f1_score

BASE=Path('processed/cross_dataset_77/preprocessing/attack_family')
OUT=Path('results/cross_dataset_77/attack_family_ann')
OUT.mkdir(parents=True,exist_ok=True)
SEED=42; BATCH=512; MAX_EPOCHS=75; PATIENCE=8; LR=5e-4
CLASS_NAMES=['Web Attack','DoS','DDoS']

keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism(); deterministic=True
except Exception: deterministic=False

required=['X_train_scaled.npy','y_train.npy','X_validation_scaled.npy','y_validation.npy','class_weights.json','predictor_order_77.csv','manifest.json']
for n in required:
 p=BASE/n
 if not p.exists(): raise SystemExit(f'Missing required artifact: {p}')

Xtr=np.load(BASE/'X_train_scaled.npy',mmap_mode='r'); ytr=np.load(BASE/'y_train.npy',mmap_mode='r')
Xv=np.load(BASE/'X_validation_scaled.npy',mmap_mode='r'); yv=np.load(BASE/'y_validation.npy',mmap_mode='r')
if Xtr.shape[1]!=77 or Xv.shape[1]!=77: raise RuntimeError('Expected 77 predictors')
if set(np.unique(ytr).tolist())!={0,1,2}: raise RuntimeError(f'Unexpected labels: {np.unique(ytr)}')
preds=pd.read_csv(BASE/'predictor_order_77.csv')
if len(preds)!=77 or 'Protocol' in preds['feature'].tolist(): raise RuntimeError('77-feature predictor manifest invalid')
weights={int(k):float(v) for k,v in json.loads((BASE/'class_weights.json').read_text()).items()}

model=keras.Sequential([
 keras.Input(shape=(77,),name='common_77_features'),
 layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(1e-5)),
 layers.Dropout(0.30),
 layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(1e-5)),
 layers.Dropout(0.25),
 layers.Dense(32,activation='relu'),
 layers.Dense(3,activation='softmax')
],name='attack_family_ann_77_cross_dataset')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR,clipnorm=1.0),loss='sparse_categorical_crossentropy',metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

best=OUT/'best_attack_family_ann_77.keras'
cbs=[
 keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=PATIENCE,restore_best_weights=True,verbose=1),
 keras.callbacks.ModelCheckpoint(best,monitor='val_loss',mode='min',save_best_only=True,verbose=1),
 keras.callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min',factor=0.5,patience=3,min_lr=1e-6,verbose=1),
 keras.callbacks.CSVLogger(OUT/'training_log.csv'), keras.callbacks.TerminateOnNaN()
]
print('Training 77-feature attack-family ANN')
print(f'  train rows: {len(ytr):,}'); print(f'  validation rows: {len(yv):,}'); print('  predictors: 77'); print(f'  class weights: {weights}'); print('  CIC-IDS2017 used: NO'); print('  holdout test arrays loaded: NO')
start=time.time(); hist=model.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=MAX_EPOCHS,batch_size=BATCH,class_weight=weights,callbacks=cbs,shuffle=True,verbose=2); elapsed=time.time()-start
final=OUT/'finalized_attack_family_ann_77_before_external.keras'; model.save(final)
pd.DataFrame(hist.history).assign(epoch=lambda d:np.arange(1,len(d)+1)).to_csv(OUT/'history.csv',index=False)

prob=model.predict(Xv,batch_size=2048,verbose=0); pred=np.argmax(prob,axis=1)
cm=confusion_matrix(yv,pred,labels=[0,1,2]); pd.DataFrame(cm,index=[f'Actual_{x}' for x in CLASS_NAMES],columns=[f'Predicted_{x}' for x in CLASS_NAMES]).to_csv(OUT/'validation_confusion_matrix.csv')
rep=classification_report(yv,pred,labels=[0,1,2],target_names=CLASS_NAMES,output_dict=True,zero_division=0); (OUT/'validation_classification_report.json').write_text(json.dumps(rep,indent=2))
vm={'accuracy':float(np.mean(pred==yv)),'balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'macro_precision':float(precision_score(yv,pred,average='macro',zero_division=0)),'macro_recall':float(recall_score(yv,pred,average='macro',zero_division=0)),'macro_f1':float(f1_score(yv,pred,average='macro',zero_division=0)),'weighted_f1':float(f1_score(yv,pred,average='weighted',zero_division=0)),'per_class_recall':{CLASS_NAMES[i]:float(recall_score(yv==i,pred==i,zero_division=0)) for i in range(3)}}
(OUT/'validation_metrics.json').write_text(json.dumps(vm,indent=2))
with open(OUT/'model_summary.txt','w') as f:model.summary(print_fn=lambda x:f.write(x+'\n'))
manifest={'experiment':'77_feature_attack_family_cross_dataset_model','seed':SEED,'features':77,'excluded_feature':'Protocol','train_rows':int(len(ytr)),'validation_rows':int(len(yv)),'test_loaded':False,'CIC_IDS2017_used':False,'class_names':CLASS_NAMES,'class_weights':weights,'architecture':['Dense128 ReLU L2=1e-5','Dropout0.30','Dense64 ReLU L2=1e-5','Dropout0.25','Dense32 ReLU','Dense3 Softmax'],'optimizer':'Adam clipnorm=1.0','learning_rate':LR,'batch_size':BATCH,'max_epochs':MAX_EPOCHS,'early_stopping_patience':PATIENCE,'epochs_completed':len(hist.history['loss']),'training_seconds':elapsed,'validation_metrics':vm,'tensorflow':tf.__version__,'python':sys.version,'platform':platform.platform(),'deterministic_ops':deterministic}
(OUT/'training_manifest.json').write_text(json.dumps(manifest,indent=2))

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
lines=[f'{sha(p)}  {p}' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt']; (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print('\nTraining complete'); print('epochs:',len(hist.history['loss'])); print('validation:',json.dumps(vm,indent=2)); print('final model:',final); print('CIC-IDS2017 remains unused.')
