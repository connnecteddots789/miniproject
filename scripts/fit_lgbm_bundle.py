"""Fit All / Fixed / Dynamic LightGBM models in one isolated process."""
import argparse, json, time
from pathlib import Path
import numpy as np
import joblib
from lightgbm import LGBMClassifier

p=argparse.ArgumentParser()
p.add_argument('--matrix',required=True)
p.add_argument('--labels',required=True)
p.add_argument('--config-file',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
X=np.load(a.matrix,mmap_mode='r')
y=np.load(a.labels,allow_pickle=True)
cfg=json.loads(Path(a.config_file).read_text())
results={}
for key in ('all','fixed','router'):
    idx=np.asarray(cfg[key]['indices'],dtype=int)
    rows=np.asarray(cfg[key].get('rows',np.arange(len(y))),dtype=int)
    model=LGBMClassifier(**cfg[key]['params'])
    t=time.perf_counter(); model.fit(np.asarray(X[rows][:,idx],dtype=np.float32),y[rows]); elapsed=time.perf_counter()-t
    results[key]={'model':model,'indices':idx.tolist(),'training_seconds':elapsed,'fit_records':int(len(rows))}
    print(f'{key} model fit: {elapsed:.2f}s, features={len(idx)}',flush=True)
experts={}
for item in cfg['experts']:
    family=item['family']; rows=np.asarray(item['rows'],dtype=int); idx=np.asarray(item['indices'],dtype=int)
    target=np.where(y[rows]==family,family,'__OTHER__').astype(object)
    model=LGBMClassifier(**item['params'])
    t=time.perf_counter(); model.fit(np.asarray(X[rows][:,idx],dtype=np.float32),target); elapsed=time.perf_counter()-t
    pos=int(np.flatnonzero(np.asarray(model.classes_,dtype=object)==family)[0])
    experts[family]={'model':model,'indices':idx.tolist(),'positive_class':family,'positive_index':pos,
                     'fit_records':int(len(rows)),'training_seconds':elapsed}
    print(f'expert {family} fit: {elapsed:.2f}s, features={len(idx)}, rows={len(rows)}',flush=True)
results['experts']=experts
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
joblib.dump(results,a.output,compress=3)
print(json.dumps({'saved':a.output,'classes':list(experts)}),flush=True)
