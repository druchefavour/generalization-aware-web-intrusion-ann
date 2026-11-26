#!/usr/bin/env python3
"""Prepare CIC-IDS2017 with the 77 predictors shared with CSE-CIC-IDS2018.

Run from ANN-Web-Intrusion/Data:
    python prepare_cicids2017_77.py

No model prediction, training, tuning, resampling, or scaler fitting occurs.
The CSE-CIC-IDS2018 TRAINING-fitted 77-feature scalers are used via transform only.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, sqlite3
import numpy as np
import pandas as pd
import joblib

RAW=Path('external-validation/CIC-IDS2017/raw/MachineLearningCVE')
OUT=Path('external-validation/CIC-IDS2017/prepared_77')
OUT.mkdir(parents=True,exist_ok=True)
BP=Path('processed/cross_dataset_77/preprocessing/binary')
FP=Path('processed/cross_dataset_77/preprocessing/attack_family')
PRED=BP/'predictor_order_77.csv'
BSC=BP/'standard_scaler_77.joblib'; FSC=FP/'standard_scaler_77.joblib'
CHUNK=100_000
files=sorted(RAW.glob('*.csv'))
if not files: raise SystemExit(f'No CSVs found in {RAW}')
expected=pd.read_csv(PRED)['feature'].astype(str).str.strip().tolist()
if len(expected)!=77 or 'Protocol' in expected: raise RuntimeError('Frozen predictor list is not the required 77-feature set')

ALIASES={
'Dst Port':'Destination Port','Tot Fwd Pkts':'Total Fwd Packets','Tot Bwd Pkts':'Total Backward Packets',
'TotLen Fwd Pkts':'Total Length of Fwd Packets','TotLen Bwd Pkts':'Total Length of Bwd Packets',
'Fwd Pkt Len Max':'Fwd Packet Length Max','Fwd Pkt Len Min':'Fwd Packet Length Min','Fwd Pkt Len Mean':'Fwd Packet Length Mean','Fwd Pkt Len Std':'Fwd Packet Length Std',
'Bwd Pkt Len Max':'Bwd Packet Length Max','Bwd Pkt Len Min':'Bwd Packet Length Min','Bwd Pkt Len Mean':'Bwd Packet Length Mean','Bwd Pkt Len Std':'Bwd Packet Length Std',
'Flow Byts/s':'Flow Bytes/s','Flow Pkts/s':'Flow Packets/s','Fwd IAT Tot':'Fwd IAT Total','Bwd IAT Tot':'Bwd IAT Total',
'Fwd Header Len':'Fwd Header Length','Bwd Header Len':'Bwd Header Length','Fwd Pkts/s':'Fwd Packets/s','Bwd Pkts/s':'Bwd Packets/s',
'Pkt Len Min':'Min Packet Length','Pkt Len Max':'Max Packet Length','Pkt Len Mean':'Packet Length Mean','Pkt Len Std':'Packet Length Std','Pkt Len Var':'Packet Length Variance',
'FIN Flag Cnt':'FIN Flag Count','SYN Flag Cnt':'SYN Flag Count','RST Flag Cnt':'RST Flag Count','PSH Flag Cnt':'PSH Flag Count','ACK Flag Cnt':'ACK Flag Count','URG Flag Cnt':'URG Flag Count','ECE Flag Cnt':'ECE Flag Count',
'Pkt Size Avg':'Average Packet Size','Fwd Seg Size Avg':'Avg Fwd Segment Size','Bwd Seg Size Avg':'Avg Bwd Segment Size',
'Fwd Byts/b Avg':'Fwd Avg Bytes/Bulk','Fwd Pkts/b Avg':'Fwd Avg Packets/Bulk','Fwd Blk Rate Avg':'Fwd Avg Bulk Rate','Bwd Byts/b Avg':'Bwd Avg Bytes/Bulk','Bwd Pkts/b Avg':'Bwd Avg Packets/Bulk','Bwd Blk Rate Avg':'Bwd Avg Bulk Rate',
'Subflow Fwd Pkts':'Subflow Fwd Packets','Subflow Fwd Byts':'Subflow Fwd Bytes','Subflow Bwd Pkts':'Subflow Bwd Packets','Subflow Bwd Byts':'Subflow Bwd Bytes',
'Init Fwd Win Byts':'Init_Win_bytes_forward','Init Bwd Win Byts':'Init_Win_bytes_backward','Fwd Act Data Pkts':'act_data_pkt_fwd','Fwd Seg Size Min':'min_seg_size_forward'}

def norm(s): return ' '.join(str(s).replace('\ufeff','').strip().split())
def key(s): return ''.join(ch.lower() for ch in norm(s) if ch.isalnum())
def map_label(x):
 s=norm(x).replace('\ufffd',' ').replace('�',' '); s=' '.join(s.split())
 if s.upper()=='BENIGN': return 'Benign'
 if s=='DDoS': return 'DDoS'
 if s.startswith('DoS '): return 'DoS'
 if s.startswith('Web Attack '): return 'Web Attack'
 return None

header=[norm(c) for c in pd.read_csv(files[0],nrows=0).columns]; hk={key(c):c for c in header}
feature_map={}
for e in expected:
 candidate=e if key(e) in hk else ALIASES.get(e)
 if candidate is None or key(candidate) not in hk: raise RuntimeError(f'No defensible mapping for {e}')
 feature_map[e]=hk[key(candidate)]
if len(set(feature_map.values()))!=77: raise RuntimeError('Feature mapping is not one-to-one')
pd.DataFrame([{'feature_index':i,'model_feature':e,'external_column':feature_map[e],'basis':'canonical exact' if key(e)==key(feature_map[e]) else 'documented naming alias'} for i,e in enumerate(expected)]).to_csv(OUT/'feature_mapping_77.csv',index=False)

bs=joblib.load(BSC); fs=joblib.load(FSC)
if bs.n_features_in_!=77 or fs.n_features_in_!=77: raise RuntimeError('Scaler incompatibility')

db=OUT/'dedupe.sqlite'
if db.exists(): db.unlink()
con=sqlite3.connect(db); con.execute('PRAGMA journal_mode=WAL'); con.execute('PRAGMA synchronous=NORMAL'); con.execute('CREATE TABLE seen(h BLOB PRIMARY KEY)')
raw=Counter(); bad=Counter(); dup=Counter(); final=Counter(); bysource=Counter()
XB=[];YB=[];XF=[];YF=[]
for f in files:
 print('Preparing',f.name,'...')
 cols=[norm(c) for c in pd.read_csv(f,nrows=0).columns]; actual={key(c):c for c in cols}; lc=actual['label']
 use=[actual[key(feature_map[e])] for e in expected]+[lc]
 for ch in pd.read_csv(f,chunksize=CHUNK,low_memory=False):
  ch.columns=[norm(c) for c in ch.columns]; labelcol=next(c for c in ch.columns if key(c)=='label')
  fam=ch[labelcol].astype(str).map(map_label); keep=fam.notna()
  if not keep.any(): continue
  sub=ch.loc[keep]; fam=fam.loc[keep].to_numpy(object)
  for c in fam: raw[c]+=1
  arr=np.column_stack([pd.to_numeric(sub[next(c for c in sub.columns if key(c)==key(feature_map[e]))],errors='coerce').to_numpy() for e in expected]).astype(np.float64)
  ok=np.isfinite(arr).all(axis=1)
  for c in fam[~ok]: bad[c]+=1
  arr=arr[ok]; fam=fam[ok]
  idx=[]; cur=con.cursor()
  for i,(row,c) in enumerate(zip(arr,fam)):
   h=hashlib.sha256(row.tobytes()+b'\0'+c.encode()).digest()
   try: cur.execute('INSERT INTO seen(h) VALUES(?)',(h,)); idx.append(i)
   except sqlite3.IntegrityError: dup[c]+=1
  con.commit()
  if not idx: continue
  arr=arr[idx]; fam=fam[idx]
  for c in fam: final[c]+=1; bysource[(f.name,c)]+=1
  XB.append(bs.transform(arr).astype(np.float32)); YB.append(np.array([0 if c=='Benign' else 1 for c in fam],np.int64))
  am=np.array([c!='Benign' for c in fam]); fmap={'Web Attack':0,'DoS':1,'DDoS':2}
  if am.any(): XF.append(fs.transform(arr[am]).astype(np.float32)); YF.append(np.array([fmap[c] for c in fam[am]],np.int64))
con.close()
xb=np.concatenate(XB); yb=np.concatenate(YB); xf=np.concatenate(XF); yf=np.concatenate(YF)
np.save(OUT/'X_external_binary_77_scaled.npy',xb,allow_pickle=False); np.save(OUT/'y_external_binary_77.npy',yb,allow_pickle=False)
np.save(OUT/'X_external_attack_family_77_scaled.npy',xf,allow_pickle=False); np.save(OUT/'y_external_attack_family_77.npy',yf,allow_pickle=False)
pd.DataFrame([{'class':c,'raw_comparable':raw[c],'nonfinite_removed':bad[c],'duplicates_removed':dup[c],'final_unique':final[c]} for c in ['Benign','Web Attack','DoS','DDoS']]).to_csv(OUT/'disposition.csv',index=False)
pd.DataFrame([{'source_file':s,'class':c,'final_unique':n} for (s,c),n in sorted(bysource.items())]).to_csv(OUT/'counts_by_source.csv',index=False)
manifest={'predictors':77,'Protocol_present':False,'binary_shape':list(xb.shape),'attack_family_shape':list(xf.shape),'binary_counts':{'Benign':int((yb==0).sum()),'Intrusion':int((yb==1).sum())},'attack_counts':{'Web Attack':int((yf==0).sum()),'DoS':int((yf==1).sum()),'DDoS':int((yf==2).sum())},'scaler_policy':'CSE-CIC-IDS2018 training-fitted 77-feature scalers; transform only','models_used':False,'external_scaler_fit':False,'training_or_tuning':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
lines=[f'{sha(p)}  {p}' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt']; (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print('\nPrepared:',json.dumps(manifest,indent=2)); print('No ANN prediction performed.')
