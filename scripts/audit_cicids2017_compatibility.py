#!/usr/bin/env python3
"""Read-only CIC-IDS2017 external-validation compatibility audit.
Run from ANN-Web-Intrusion/Data:
    python audit_cicids2017_compatibility.py
No ANN predictions or scaler fitting are performed.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

RAW=Path('external-validation/CIC-IDS2017/raw/MachineLearningCVE')
OUT=Path('external-validation/CIC-IDS2017/audit')
PRED=Path('processed/preprocessing/binary/predictor_order.csv')
OUT.mkdir(parents=True,exist_ok=True)
CHUNK=100_000
files=sorted(RAW.glob('*.csv'))
if not files: raise SystemExit(f'No CSVs in {RAW}')
expected=pd.read_csv(PRED)['feature'].astype(str).str.strip().tolist()

def norm(s):
    return ' '.join(str(s).replace('\ufeff','').strip().split())

def key(s):
    return ''.join(ch.lower() for ch in norm(s) if ch.isalnum())

labels=Counter(); missing=Counter(); inf=Counter(); per_file=[]; schemas={}
for f in files:
    print('Auditing',f.name,'...')
    rows=0; flabels=Counter(); cols=None
    for chunk in pd.read_csv(f,chunksize=CHUNK,low_memory=False):
        chunk.columns=[norm(c) for c in chunk.columns]
        cols=list(chunk.columns)
        label_cols=[c for c in cols if key(c)=='label']
        if len(label_cols)!=1: raise ValueError(f'Expected one Label column in {f.name}; got {label_cols}')
        lc=label_cols[0]; rows+=len(chunk)
        vc=chunk[lc].astype('string').str.strip().value_counts(dropna=False)
        for k,v in vc.items():
            k='<NA>' if pd.isna(k) else str(k); labels[k]+=int(v); flabels[k]+=int(v)
        for c in cols:
            if c==lc: continue
            n=pd.to_numeric(chunk[c],errors='coerce')
            original_missing=int(chunk[c].isna().sum())
            coerced_missing=int(n.isna().sum())
            if coerced_missing: missing[c]+=coerced_missing
            arr=n.to_numpy(dtype=float,na_value=np.nan)
            ni=int(np.isinf(arr).sum())
            if ni: inf[c]+=ni
    schemas[f.name]=cols
    per_file.append({'file':f.name,'rows':rows,'columns':len(cols),'labels':dict(flabels)})

# Build canonical-key match from first file, while checking schema consistency.
base=schemas[files[0].name]
base_key={key(c):c for c in base}
match=[]; missing_expected=[]
for e in expected:
    k=key(e)
    if k in base_key:
        match.append({'model_feature':e,'external_column':base_key[k],'match':'canonical_exact'})
    else:
        missing_expected.append(e)
extras=[c for c in base if key(c) not in {key(e) for e in expected} and key(c)!='label']

schema_consistency={name: [key(c) for c in cols]==[key(c) for c in base] for name,cols in schemas.items()}

report={
 'raw_directory':str(RAW),'files':[f.name for f in files],'per_file':per_file,
 'aggregate_labels':dict(labels),'expected_model_predictor_count':len(expected),
 'canonical_matches':len(match),'missing_expected_features':missing_expected,
 'external_extra_nonlabel_columns':extras,'schema_consistency_vs_first_file':schema_consistency,
 'aggregate_numeric_coercion_missing':dict(missing),'aggregate_infinite_values':dict(inf),
 'decision':'COMPATIBLE' if not missing_expected and all(schema_consistency.values()) else 'REVIEW_REQUIRED',
 'note':'Canonical matching ignores whitespace, punctuation, and capitalization only. No semantic feature substitutions are invented. No scaler or ANN is used.'
}
(OUT/'compatibility_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
pd.DataFrame(match).to_csv(OUT/'feature_mapping.csv',index=False)
pd.DataFrame(sorted(labels.items()),columns=['raw_label','count']).to_csv(OUT/'label_counts.csv',index=False)
pd.DataFrame(per_file).drop(columns=['labels']).to_csv(OUT/'file_summary.csv',index=False)
pd.DataFrame({'missing_expected_feature':missing_expected}).to_csv(OUT/'missing_expected_features.csv',index=False)
pd.DataFrame({'external_extra_column':extras}).to_csv(OUT/'external_extra_columns.csv',index=False)
pd.DataFrame(sorted(missing.items()),columns=['column','numeric_coercion_missing']).to_csv(OUT/'missing_values.csv',index=False)
pd.DataFrame(sorted(inf.items()),columns=['column','infinite_count']).to_csv(OUT/'infinite_values.csv',index=False)
print('\nCompatibility decision:',report['decision'])
print(f"Matched frozen predictors: {len(match)}/{len(expected)}")
print('Missing expected features:',len(missing_expected))
print('External extras:',len(extras))
print('Reports:',OUT)
