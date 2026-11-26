"""Full training-fold comparison of boosted/forest models and compact feature subsets.
All model selection uses a duplicate-grouped validation split from official train;
official test labels are only used once to score the selected designs.
"""
import sys,json,time,warnings
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data,metrics,SEED,OUT
from app.ml import make_preprocessor
warnings.filterwarnings('ignore')

def run(ds):
 X,y,Xte,yte,*_=get_data(ds);labels=sorted(set(y)|set(yte));groups=pd.util.hash_pandas_object(X,index=False).to_numpy();grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
 gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy());mask=np.isin(groups,gtr);Xfit=X.loc[mask].reset_index(drop=True);Xval=X.loc[~mask].reset_index(drop=True);yfit=y[mask];yval=y[~mask]
 pre=make_preprocessor(Xfit);t=time.perf_counter();Zfit=np.asarray(pre.fit_transform(Xfit),dtype=np.float32);Zval=np.asarray(pre.transform(Xval),dtype=np.float32);Zte=np.asarray(pre.transform(Xte),dtype=np.float32);prep=time.perf_counter()-t
 old=joblib.load(OUT/ds.lower().replace('-','_')/'candidate_checkpoint.joblib');idx=np.asarray(old['fixed_indices'],dtype=int)
 candidates={
  'LightGBM-500':LGBMClassifier(n_estimators=500,num_leaves=31,max_depth=-1,learning_rate=.035,min_child_samples=30,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=2,n_jobs=1,random_state=SEED+400,verbosity=-1,force_col_wise=True,max_bin=127),
  'ExtraTrees-240':ExtraTreesClassifier(n_estimators=240,min_samples_leaf=2,max_features=.85,n_jobs=1,random_state=SEED+401,class_weight='balanced_subsample'),
  'RandomForest-220':RandomForestClassifier(n_estimators=220,min_samples_leaf=2,max_features=.85,n_jobs=1,random_state=SEED+402,class_weight='balanced_subsample')}
 rows=[];fitted={}
 for name,m in candidates.items():
  t=time.perf_counter();m.fit(Zfit,yfit);sec=time.perf_counter()-t
  pv=m.predict(Zval);acc=float(accuracy_score(yval,pv));rows.append({'name':name,'validation_accuracy':acc,'fit_seconds':sec});fitted[name]=m
  print(ds,name,'val',round(acc*100,4),'fit',round(sec,1),flush=True)
 # Select estimator by validation accuracy only; same selected estimator evaluated all vs compact.
 best=max(rows,key=lambda r:r['validation_accuracy']);name=best['name'];full=fitted[name]
 # For a fair fixed subset, fit same estimator on the training-only selected features.
 def new_model():
  if name.startswith('LightGBM'): return LGBMClassifier(n_estimators=500,num_leaves=31,max_depth=-1,learning_rate=.035,min_child_samples=30,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=2,n_jobs=1,random_state=SEED+403,verbosity=-1,force_col_wise=True,max_bin=127)
  if name.startswith('ExtraTrees'): return ExtraTreesClassifier(n_estimators=240,min_samples_leaf=2,max_features=.85,n_jobs=1,random_state=SEED+404,class_weight='balanced_subsample')
  return RandomForestClassifier(n_estimators=220,min_samples_leaf=2,max_features=.85,n_jobs=1,random_state=SEED+405,class_weight='balanced_subsample')
 fixed=new_model();t=time.perf_counter();fixed.fit(Zfit[:,idx],yfit);fsec=time.perf_counter()-t
 # Router uses validation-only confidence and correctness model; separate train/gate split.
 pfull=full.predict_proba(Zval);pfixed=fixed.predict_proba(Zval[:,idx]);predfull=full.predict(Zval);predfixed=fixed.predict(Zval[:,idx])
 # Candidate model classes are aligned by estimator class order, both trained on same labels.
 from sklearn.ensemble import ExtraTreesClassifier as ET
 order=np.arange(len(labels))
 def feat(a,b,z):
  def sm(p):
   s=np.sort(p,axis=1);return np.column_stack([s[:,-1],s[:,-2],s[:,-1]-s[:,-2],-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)])
  return np.column_stack([a,b,a-b,sm(a),sm(b),z])
 meta=feat(pfixed,pfull,Zval[:,idx]);target=((predfixed==yval)&(predfull!=yval)).astype(int)
 itr,iga=train_test_split(np.arange(len(yval)),test_size=.35,random_state=SEED+406,stratify=yval)
 router=ET(n_estimators=200,max_depth=8,min_samples_leaf=40,max_features=.85,n_jobs=1,class_weight='balanced',random_state=SEED+407);router.fit(meta[itr],target[itr]);pr=router.predict_proba(meta[iga])[:,list(router.classes_).index(1)]
 bestgate=None
 for th in np.unique(np.quantile(pr,np.linspace(.01,.99,99))):
  use=pr>=th;pred=np.where(use,predfixed[iga],predfull[iga]);a=float(accuracy_score(yval[iga],pred))
  if bestgate is None or a>bestgate[0]:bestgate=(a,float(th),float(use.mean()))
 # Official test
 at=full.predict(Zte);ft=fixed.predict(Zte[:,idx]);pta=full.predict_proba(Zte);ptf=fixed.predict_proba(Zte[:,idx]);mt=feat(ptf,pta,Zte[:,idx]);prt=router.predict_proba(mt)[:,list(router.classes_).index(1)];use=prt>=bestgate[1];dt=np.where(use,ft,at)
 results={'all':metrics(yte,at,labels),'fixed':metrics(yte,ft,labels),'dynamic':metrics(yte,dt,labels)}
 for k in ['all','fixed','dynamic']: results[k]['feature_count']=float(Zte.shape[1] if k=='all' else (len(idx) if k=='fixed' else len(idx)*use.mean()+Zte.shape[1]*(~use).mean()))
 results['dynamic'].update({'adaptive_mode':'learned compact-vs-full correctness router','router_validation_accuracy':bestgate[0],'router_threshold':bestgate[1],'compact_fraction':float(use.mean()),'average_features':results['dynamic']['feature_count']})
 report={'dataset':ds,'task':'native multiclass','train_rows':len(y),'fit_rows':len(yfit),'validation_rows':len(yval),'official_test_rows':len(yte),'features':int(Zfit.shape[1]),'compact_features':int(len(idx)),'preprocess_seconds':prep,'candidate_validation':rows,'selected_model':name,'fixed_fit_seconds':fsec,'router_gate_accuracy':bestgate[0],'router_threshold_validation_only':bestgate[1],'official_test_results':results,'protocol':'All estimators trained on the duplicate-grouped official-training fit fold. Model family selected using training-only validation accuracy. Compact feature indices reused from training-only hybrid ranking. Router fitted and threshold selected on separate training-only validation partitions. Official test labels used only for final metrics.'}
 out=OUT/ds.lower().replace('-','_')/'full_training_ensemble';out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'preprocessor':pre,'all':full,'fixed':fixed,'router':router,'fixed_indices':idx,'router_threshold':bestgate[1]},out/'models.joblib',compress=3)
 print(ds,'SELECTED',name,'TEST',[(k,round(v['accuracy']*100,4)) for k,v in results.items()],'router gate',round(bestgate[0]*100,3),'compact frac',round(float(use.mean()),3),flush=True)
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True);run(p.parse_args().dataset)
