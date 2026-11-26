"""Fit individual final LightGBM model(s) in a fresh process."""
import argparse, json, time
from pathlib import Path
import numpy as np
import joblib
from lightgbm import LGBMClassifier
p=argparse.ArgumentParser();p.add_argument('--key',choices=['all','fixed','router','experts'],required=True);p.add_argument('--cache',required=True)
a=p.parse_args();d=Path(a.cache);X=np.load(d/'matrix.npy',mmap_mode='r');y=np.load(d/'labels.npy',allow_pickle=True);cfg=json.loads((d/'worker_config.json').read_text())
if a.key=='experts':
    out={}
    for item in cfg['experts']:
        fam=item['family']; rows=np.asarray(item['rows'],dtype=int); idx=np.asarray(item['indices'],dtype=int)
        target=np.where(y[rows]==fam,fam,'__OTHER__').astype(object);m=LGBMClassifier(**item['params'])
        t=time.perf_counter();m.fit(np.asarray(X[rows][:,idx],dtype=np.float32),target);elapsed=time.perf_counter()-t
        pos=int(np.flatnonzero(np.asarray(m.classes_,dtype=object)==fam)[0])
        out[fam]={'model':m,'indices':idx.tolist(),'positive_class':fam,'positive_index':pos,'fit_records':int(len(rows)),'training_seconds':elapsed}
        print(f'expert {fam}: {elapsed:.2f}s rows={len(rows)} features={len(idx)}',flush=True)
    joblib.dump(out,d/'experts.joblib',compress=3)
else:
    item=cfg[a.key];idx=np.asarray(item['indices'],dtype=int);m=LGBMClassifier(**item['params'])
    t=time.perf_counter();m.fit(np.asarray(X[:,idx],dtype=np.float32),y);elapsed=time.perf_counter()-t
    joblib.dump({'model':m,'indices':idx.tolist(),'training_seconds':elapsed},d/f'{a.key}.joblib',compress=3)
    print(f'{a.key}: {elapsed:.2f}s rows={len(y)} features={len(idx)}',flush=True)
