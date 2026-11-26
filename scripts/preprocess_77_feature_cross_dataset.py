#!/usr/bin/env python3
"""Create 77-feature training-compatible preprocessing for cross-dataset validation.
Drops ONLY Protocol from the frozen 78-feature internal partitions.
Fits new StandardScalers on CSE-CIC-IDS2018 TRAINING splits only.
Does not touch existing 78-feature artifacts and does not use CIC-IDS2017.
"""
from pathlib import Path
import hashlib, json, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SRC=Path('processed/partitions'); OUT=Path('processed/cross_dataset_77/preprocessing'); OUT.mkdir(parents=True,exist_ok=True)
EXPS={'binary':('BinaryTarget',['Benign','Intrusion']), 'attack_family':('AttackFamilyTarget',['Web Attack','DoS','DDoS'])}
DROP={'SourceFile','Label','ModelClass','RowID','BinaryTarget','AttackFamilyTarget','Protocol'}

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

for exp,(target,classes) in EXPS.items():
 print('Preparing 77-feature',exp,'...')
 edir=OUT/exp; edir.mkdir(parents=True,exist_ok=True)
 frames={}
 for split in ['train','validation','test']:
  p=SRC/exp/f'{split}.csv'; df=pd.read_csv(p,low_memory=False)
  predictors=[c for c in df.columns if c not in DROP]
  if 'Protocol' in predictors: raise RuntimeError('Protocol leakage')
  if len(predictors)!=77: raise RuntimeError(f'Expected 77 predictors, got {len(predictors)}')
  X=df[predictors].apply(pd.to_numeric,errors='raise').to_numpy(np.float64)
  if not np.isfinite(X).all(): raise RuntimeError('Non-finite value in frozen split')
  mp={c:i for i,c in enumerate(classes)}; y=df[target].map(mp).to_numpy(np.int64)
  frames[split]=(X,y)
 scaler=StandardScaler().fit(frames['train'][0])
 joblib.dump(scaler,edir/'standard_scaler_77.joblib')
 for split,(X,y) in frames.items():
  np.save(edir/f'X_{split}_scaled.npy',scaler.transform(X).astype(np.float32),allow_pickle=False)
  np.save(edir/f'y_{split}.npy',y,allow_pickle=False)
 y=frames['train'][1]; ids=np.arange(len(classes)); w=compute_class_weight('balanced',classes=ids,y=y)
 (edir/'class_weights.json').write_text(json.dumps({str(i):float(v) for i,v in zip(ids,w)},indent=2))
 pd.DataFrame({'feature_index':range(77),'feature':predictors}).to_csv(edir/'predictor_order_77.csv',index=False)
 (edir/'manifest.json').write_text(json.dumps({'experiment':exp,'dropped_feature':'Protocol','predictor_count':77,'scaler_fit':'CSE-CIC-IDS2018 training partition only','classes':classes,'class_weights':{str(i):float(v) for i,v in zip(ids,w)},'existing_78_feature_artifacts_modified':False,'external_data_used':False},indent=2))
lines=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name!='SHA256SUMS.txt': lines.append(f'{sha(p)}  {p}')
(OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print('Done:',OUT)
