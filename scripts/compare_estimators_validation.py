"""Training-only model family screen on group-separated validation; not final test results."""
import argparse, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]; BACKEND=ROOT/'backend';sys.path.insert(0,str(BACKEND));sys.path.insert(0,str(ROOT/'scripts'))
from app.datasets import load_nsl,load_unsw,dataset_files
from app.ml import make_preprocessor,stratified_sample_indices,LabelEncodedClassifier
from optimize_adaptive_research import get_frames,feature_rankings

def score(y,p):
    f=precision_recall_fscore_support(y,p,average='macro',zero_division=0)[2]
    return float(accuracy_score(y,p)),float(f)

def run(ds,task):
    X,y,Xte,yte,*_=get_frames(ds,task)
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    gf=pd.DataFrame({'g':groups,'y':y}); gl=gf.groupby('g',sort=False).y.agg(lambda q:q.value_counts().index[0])
    a,b=train_test_split(gl.index.to_numpy(),test_size=.15,random_state=42,stratify=gl.to_numpy())
    aset=set(a.tolist()); tr=np.flatnonzero(np.fromiter((g in aset for g in groups),dtype=bool,count=len(groups))); va=np.flatnonzero(~np.fromiter((g in aset for g in groups),dtype=bool,count=len(groups)))
    Xtr=X.iloc[tr].reset_index(drop=True); Xv=X.iloc[va].reset_index(drop=True); ytr=y[tr]; yv=y[va]
    pre=make_preprocessor(Xtr); A=np.asarray(pre.fit_transform(Xtr),dtype=np.float32); B=np.asarray(pre.transform(Xv),dtype=np.float32)
    si=stratified_sample_indices(ytr,8000,51,min_per_class=10); A=A[si]; ys=np.asarray(ytr,dtype=object)[si]
    ranks,_=feature_rankings(A,ys,52); idx=np.asarray(ranks['Hybrid (gain ratio + MI + LightGBM)'][:max(3,int(A.shape[1]*.35))],dtype=int)
    candidates={
      'LightGBM-300':LGBMClassifier(n_estimators=300,num_leaves=31,learning_rate=.05,min_child_samples=25,class_weight='balanced',n_jobs=1,random_state=42,verbosity=-1,force_col_wise=True,max_bin=63),
      'ExtraTrees-180':ExtraTreesClassifier(n_estimators=180,max_depth=None,min_samples_leaf=1,max_features='sqrt',class_weight='balanced',n_jobs=2,random_state=42),
      'RandomForest-180':RandomForestClassifier(n_estimators=180,max_depth=None,min_samples_leaf=1,max_features='sqrt',class_weight='balanced_subsample',n_jobs=2,random_state=42),
      'XGBoost-200':LabelEncodedClassifier(XGBClassifier(n_estimators=200,max_depth=7,learning_rate=.05,subsample=.9,colsample_bytree=.9,tree_method='hist',n_jobs=2,random_state=42,verbosity=0)),
      'CatBoost-200':CatBoostClassifier(iterations=200,depth=8,learning_rate=.05,loss_function='Logloss' if len(np.unique(ys))==2 else 'MultiClass',auto_class_weights='Balanced',verbose=False,thread_count=2,random_seed=42,allow_writing_files=False)
    }
    print(ds,task,'val',len(yv),'features',A.shape[1],flush=True)
    rows=[]
    for name,model in candidates.items():
        try:
            t=time.perf_counter();model.fit(A,ys);fit=time.perf_counter()-t
            p=model.predict(B);p=np.asarray(p).reshape(-1)
            ac,mf=score(yv,p); rows.append((name,ac,mf,fit))
            print(name,'accuracy',round(ac*100,3),'macroF1',round(mf*100,3),'fit_s',round(fit,2),flush=True)
        except Exception as e: print(name,'ERROR',repr(e),flush=True)
    print('TOP',sorted(rows,key=lambda x:(.55*x[1]+.45*x[2]),reverse=True)[:3],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--task',default='native');a=p.parse_args();run(a.dataset,a.task)
