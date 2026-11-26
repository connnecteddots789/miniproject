"""Training-only selected XGBoost confidence cascade; official test is evaluation-only."""
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from xgboost import XGBClassifier
sys.path.insert(0,'backend');sys.path.insert(0,'scripts')
from app.datasets import load_nsl,load_unsw
from app.ml import make_preprocessor,LabelEncodedClassifier,stratified_sample_indices
from optimize_adaptive_research import get_frames,feature_rankings

def metrics(y,p):
 return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(precision_recall_fscore_support(y,p,average='macro',zero_division=0)[2])}
def main(ds):
 X,y,Xte,yte,*_=get_frames(ds,'native'); groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
 gf=pd.DataFrame({'g':groups,'y':y}); gl=gf.groupby('g',sort=False).y.agg(lambda z:z.value_counts().index[0])
 ga,gb=train_test_split(gl.index.to_numpy(),test_size=.15,random_state=42,stratify=gl.to_numpy()); aset=set(ga.tolist())
 tr=np.flatnonzero(np.fromiter((g in aset for g in groups),bool,len(groups))); va=np.flatnonzero(~np.fromiter((g in aset for g in groups),bool,len(groups)))
 Xtr=X.iloc[tr].reset_index(drop=True); Xv=X.iloc[va].reset_index(drop=True); ytr=y[tr];yv=y[va]
 pre=make_preprocessor(Xtr); A=np.asarray(pre.fit_transform(Xtr),np.float32); B=np.asarray(pre.transform(Xv),np.float32); T=np.asarray(pre.transform(Xte),np.float32)
 si=stratified_sample_indices(ytr,min(8000,len(ytr)),51,min_per_class=10); rank,_=feature_rankings(A[si],np.asarray(ytr,dtype=object)[si],52)
 idx=np.asarray(rank['Hybrid (gain ratio + MI + LightGBM)'][:max(3,int(A.shape[1]*.5))],int)
 def model(seed): return LabelEncodedClassifier(XGBClassifier(n_estimators=200,max_depth=7,learning_rate=.05,subsample=.9,colsample_bytree=.9,tree_method='hist',n_jobs=2,random_state=seed,verbosity=0))
 print(ds,'fit',len(ytr),'val',len(yv),'test',len(yte),'features',A.shape[1],flush=True)
 allv=model(42);t=time.perf_counter();allv.fit(A[si],np.asarray(ytr,dtype=object)[si]);print('val all fit',time.perf_counter()-t,flush=True)
 fixedv=model(43);t=time.perf_counter();fixedv.fit(A[si][:,idx],np.asarray(ytr,dtype=object)[si]);print('val fixed fit',time.perf_counter()-t,flush=True)
 pv=allv.predict(B); qv=fixedv.predict_proba(B[:,idx]); conf=qv.max(axis=1); pf=fixedv.predict(B[:,idx])
 rows=[]
 for th in np.unique(np.quantile(conf,[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1])):
  mask=conf<float(th); pred=np.asarray(pf,dtype=object).copy(); pred[mask]=pv[mask]; m=metrics(yv,pred)
  rows.append((.55*m['accuracy']+.45*m['macro_f1'],m['accuracy'],m['macro_f1'],float(th),float(mask.mean())))
 best=max(rows)
 print('val all',metrics(yv,pv),'val fixed',metrics(yv,pf),'val cascade',best,flush=True)
 # Refit candidates on all official train rows. Keep the chosen threshold frozen.
 pre_full=make_preprocessor(X); A_full=np.asarray(pre_full.fit_transform(X),np.float32); T_full=np.asarray(pre_full.transform(Xte),np.float32)
 allm=model(42);fixedm=model(43)
 t=time.perf_counter();allm.fit(A_full,y);print('full all fit',time.perf_counter()-t,flush=True)
 t=time.perf_counter();fixedm.fit(A_full[:,idx],y);print('full fixed fit',time.perf_counter()-t,flush=True)
 pa=allm.predict(T_full); pf=fixedm.predict(T_full[:,idx]); pro=fixedm.predict_proba(T_full[:,idx]); mask=pro.max(axis=1)<best[3]; pdyn=np.asarray(pf,dtype=object).copy();pdyn[mask]=pa[mask]
 out={'dataset':ds,'task':'native','model':'XGBoost-200','feature_count':int(A.shape[1]),'fixed_feature_count':int(len(idx)),'threshold':best[3],'validation_fallback_fraction':best[4], 'validation':{'all':metrics(yv,pv),'fixed':metrics(yv,pf),'dynamic':{'accuracy':best[1],'macro_f1':best[2]}},'official_test':{'all':metrics(yte,pa),'fixed':metrics(yte,pf),'dynamic':metrics(yte,pdyn)},'test_fallback_fraction':float(mask.mean())}
 p=Path('backend/artifacts/research')/f'{ds.lower().replace("-","_")}_xgb_confidence_cascade';p.mkdir(parents=True,exist_ok=True);(p/'metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])
