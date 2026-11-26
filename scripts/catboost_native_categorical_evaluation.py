"""Native-categorical CatBoost pilot on official UNSW-NB15 train/test files.
Feature handling and validation are training-only; official test is scoring-only.
"""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
sys.path.insert(0,str(ROOT/'scripts'))
from hybrid_adaptive_evaluation import get_data,metrics,SEED
from app.ml import stratified_sample_indices

ds='UNSW-NB15'; print('load',flush=True)
X,y,Xte,yte,tr,te=get_data(ds)
groups=pd.util.hash_pandas_object(X,index=False).to_numpy(); grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy())
mask=np.isin(groups,gtr); Xfit=X.loc[mask].reset_index(drop=True); Xval=X.loc[~mask].reset_index(drop=True); yfit=y[mask]; yval=y[~mask]
idx=stratified_sample_indices(yfit,20000,SEED+21,min_per_class=10)
Xs=Xfit.iloc[idx].copy(); ys=yfit[idx]
cat=[c for c in X.columns if X[c].dtype=='object']
for df in [Xs,Xval,Xte]:
 for c in cat: df[c]=df[c].fillna('__MISSING__').astype(str)
print('fit',len(Xs),'cats',cat,flush=True)
model=CatBoostClassifier(iterations=100,depth=5,learning_rate=.1,l2_leaf_reg=5.0,
 loss_function='MultiClass',eval_metric='Accuracy',random_seed=SEED,verbose=False,
 allow_writing_files=False,thread_count=1,bootstrap_type='Bernoulli',subsample=.9)
t=time.perf_counter(); model.fit(Xs,ys,cat_features=cat,verbose=False); fitsecs=time.perf_counter()-t
print('fit seconds',round(fitsecs,2),flush=True)
t=time.perf_counter(); pv=np.asarray(model.predict(Xval)).reshape(-1).astype(object); print('val pred seconds',round(time.perf_counter()-t,2),flush=True)
val=metrics(yval,pv,sorted(set(y)|set(yte)))
print('validation',val['accuracy'],val['macro_f1'],flush=True)
t=time.perf_counter(); pt=np.asarray(model.predict(Xte)).reshape(-1).astype(object); print('test pred seconds',round(time.perf_counter()-t,2),flush=True)
test=metrics(yte,pt,sorted(set(y)|set(yte)))
print('official test',test['accuracy'],test['macro_f1'],test['weighted_f1'],flush=True)
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/catboost_native_categorical'
out.mkdir(parents=True,exist_ok=True)
rep={'dataset':ds,'task':'native multiclass','model':'CatBoost native categorical','training_rows':len(yfit),'model_fit_rows':len(idx),'validation_rows':len(yval),'official_test_rows':len(yte),'categorical_features':cat,'iterations':100,'depth':5,'depth':8,'fit_seconds':fitsecs,'validation':val,'official_test':test,'protocol':'Grouped validation from official training file; 40k stratified training-only fit sample; native string categorical columns; no test labels used for training or selection.'}
(out/'results.json').write_text(json.dumps(rep,indent=2,default=float)); model.save_model(str(out/'model.cbm'))
print('saved',out,flush=True)
