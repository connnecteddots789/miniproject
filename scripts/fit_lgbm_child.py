"""Fresh-process LightGBM fit helper for reproducible final refits."""
import argparse, json, os
from pathlib import Path
import numpy as np
import joblib
from lightgbm import LGBMClassifier

p=argparse.ArgumentParser()
p.add_argument('--matrix',required=True)
p.add_argument('--labels',required=True)
p.add_argument('--output',required=True)
p.add_argument('--config',required=True)
p.add_argument('--indices',required=True)
p.add_argument('--rows',default='')
a=p.parse_args()
X=np.load(a.matrix,mmap_mode='r')
y=np.load(a.labels,allow_pickle=True)
idx=np.asarray(json.loads(a.indices),dtype=int)
if a.rows:
    rows=np.load(a.rows)
    data=np.asarray(X[rows][:,idx],dtype=np.float32)
    target=y[rows]
else:
    data=np.asarray(X[:,idx],dtype=np.float32)
    target=y
cfg=json.loads(a.config)
model=LGBMClassifier(**cfg)
model.fit(data,target)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
joblib.dump(model,a.output,compress=3)
print(json.dumps({'output':a.output,'rows':len(target),'features':len(idx)}),flush=True)
