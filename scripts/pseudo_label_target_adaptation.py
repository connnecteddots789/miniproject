"""Transductive self-training using high-confidence unlabeled official-test features.
Test labels are never used for pseudo labels, fitting, or routing; only final scoring.
Results are reported separately from strict inductive benchmark.
"""
import sys,json,time,warnings
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data,metrics,SEED,OUT
from app.ml import make_preprocessor
warnings.filterwarnings('ignore')

def make_model(seed):
 return LGBMClassifier(n_estimators=240,num_leaves=15,max_depth=9,learning_rate=.055,min_child_samples=35,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=3.0,max_bin=63,n_jobs=1,random_state=seed,verbosity=-1,force_col_wise=True)

def run(ds):
 X,y,Xte,yte,*_=get_data(ds);labels=sorted(set(y)|set(yte));groups=pd.util.hash_pandas_object(X,index=False).to_numpy();grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0]);gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy());mask=np.isin(groups,gtr)
 Xfit=X.loc[mask].reset_index(drop=True);Xval=X.loc[~mask].reset_index(drop=True);yfit=y[mask];yval=y[~mask]
 pre=make_preprocessor(Xfit);Zfit=np.asarray(pre.fit_transform(Xfit),dtype=np.float32);Zv=np.asarray(pre.transform(Xval),dtype=np.float32);Zt=np.asarray(pre.transform(Xte),dtype=np.float32)
 base=make_model(SEED+900);t=time.perf_counter();base.fit(Zfit,yfit);basefit=time.perf_counter()-t
 pv=base.predict_proba(Zv);pt=base.predict_proba(Zt);classes=np.asarray(base.classes_,dtype=object);basev=base.classes_[np.argmax(pv,axis=1)];baset=base.classes_[np.argmax(pt,axis=1)]
 reports=[]; candidates=[]
 for th in ([.85] if ds=='UNSW-NB15' else [.75,.85,.92]):
  conf=pt.max(axis=1);sel=conf>=th
  pseudo=classes[np.argmax(pt[sel],axis=1)]
  if sel.sum()<100: continue
  for pw in ([.20] if ds=='UNSW-NB15' else [.15,.30,.50]):
   model=make_model(SEED+901+int(th*100)+int(pw*100))
   Zaug=np.vstack([Zfit,Zt[sel]]);yaug=np.r_[yfit,pseudo];weights=np.r_[np.ones(len(yfit)),np.full(int(sel.sum()),pw)]
   t=time.perf_counter();model.fit(Zaug,yaug,sample_weight=weights);fitsec=time.perf_counter()-t
   pval=model.predict(Zv);acc=float(accuracy_score(yval,pval))
   candidates.append((acc,th,pw,model,sel,fitsec))
   print(ds,'threshold',th,'pseudo',int(sel.sum()),'pseudo_weight',pw,'val',round(acc*100,4),'fit',round(fitsec,1),flush=True)
 best=max(candidates,key=lambda q:q[0]);_,th,pw,model,sel,fitsec=best
 # Learn a router between supervised base and pseudo-adapted expert on validation only.
 qv=model.predict_proba(Zv);predv=model.classes_[np.argmax(qv,axis=1)];
 def feat(a,b,z):
  def sm(p):
   s=np.sort(p,axis=1);return np.column_stack([s[:,-1],s[:,-2],s[:,-1]-s[:,-2],-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)])
  return np.column_stack([a,b,a-b,sm(a),sm(b),z])
 meta=feat(pv,qv,Zv); ywin=((predv==yval)&(basev!=yval)).astype(int)
 ir,ig=train_test_split(np.arange(len(yval)),test_size=.35,random_state=SEED+950,stratify=yval)
 router=ExtraTreesClassifier(n_estimators=200,max_depth=8,min_samples_leaf=40,max_features=.85,n_jobs=1,class_weight='balanced',random_state=SEED+951);router.fit(meta[ir],ywin[ir]);pr=router.predict_proba(meta[ig])[:,list(router.classes_).index(1)]
 bestgate=None
 for cut in np.unique(np.quantile(pr,np.linspace(.01,.99,99))):
  use=pr>=cut;pred=np.where(use,predv[ig],basev[ig]);a=float(accuracy_score(yval[ig],pred))
  if bestgate is None or a>bestgate[0]:bestgate=(a,float(cut),float(use.mean()))
 # final test
 qt=model.predict_proba(Zt);adapt= model.classes_[np.argmax(qt,axis=1)];meta_t=feat(pt,qt,Zt);prt=router.predict_proba(meta_t)[:,list(router.classes_).index(1)];use=prt>=bestgate[1];dyn=np.where(use,adapt,baset)
 result={'all':metrics(yte,baset,labels),'self_trained':metrics(yte,adapt,labels),'dynamic':metrics(yte,dyn,labels)}
 result['all']['feature_count']=float(Zt.shape[1]);result['self_trained']['feature_count']=float(Zt.shape[1]);result['dynamic'].update({'adaptive_mode':'supervised vs unlabeled-target self-trained expert router','threshold':bestgate[1],'self_trained_fraction':float(use.mean()),'average_features':float(Zt.shape[1])})
 report={'dataset':ds,'task':'native multiclass','protocol':'TRANSDUCTIVE self-training: pseudo-labels generated only from unlabeled official-test features. Candidate threshold and pseudo-label weights selected on training-only grouped validation. Router trained and selected on training-only validation partitions. Test labels used only for final metric reporting. Separate from strict inductive benchmark.','train_rows':len(y),'fit_rows':len(yfit),'validation_rows':len(yval),'official_test_rows':len(yte),'feature_count':int(Zfit.shape[1]),'base_fit_seconds':basefit,'selected_pseudo_threshold':th,'selected_pseudo_weight':pw,'pseudo_rows':int(sel.sum()),'selected_validation_accuracy':best[0],'router_gate_accuracy':bestgate[0],'router_threshold_validation_only':bestgate[1],'official_test_results':result,'candidates':[{'threshold':t,'pseudo_weight':w,'val_accuracy':a,'fit_seconds':s,'pseudo_rows':int(se.sum())} for a,t,w,m,se,s in candidates]}
 out=OUT/ds.lower().replace('-','_')/'pseudo_label_adaptation';out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'preprocessor':pre,'base_model':base,'self_trained_model':model,'router':router,'router_threshold':bestgate[1]},out/'models.joblib',compress=3)
 print(ds,'SELECTED',th,pw,'pseudo',int(sel.sum()),'gate',round(bestgate[0]*100,3),'TEST',[(k,round(v['accuracy']*100,4)) for k,v in result.items()],flush=True)
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True);run(p.parse_args().dataset)
