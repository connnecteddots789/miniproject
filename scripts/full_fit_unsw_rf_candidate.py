"""Full official-train refit of the validation-screen RandomForest candidate."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT/'scripts'))
from optimize_adaptive_research import get_frames,SEED
from app.ml import make_preprocessor
from sklearn.model_selection import train_test_split
import pandas as pd,joblib
X,y,Xte,yte,tr,te,task=get_frames('UNSW-NB15','native')
# This candidate was selected by training-only grouped validation (79.481% among screened families).
pre=make_preprocessor(X); t=time.perf_counter(); A=np.asarray(pre.fit_transform(X),dtype=np.float32); B=np.asarray(pre.transform(Xte),dtype=np.float32); prep=time.perf_counter()-t
m=RandomForestClassifier(n_estimators=180,max_depth=None,min_samples_leaf=1,max_features='sqrt',class_weight='balanced_subsample',n_jobs=1,random_state=42)
t=time.perf_counter();m.fit(A,y);fit=time.perf_counter()-t
t=time.perf_counter();pred=m.predict(B); infer=time.perf_counter()-t
labels=sorted(set(y)|set(yte)); p,r,f,_=precision_recall_fscore_support(yte,pred,labels=labels,average='macro',zero_division=0);pw,rw,fw,_=precision_recall_fscore_support(yte,pred,labels=labels,average='weighted',zero_division=0)
rep={'dataset':'UNSW-NB15','task':'native 10-class','model':'RandomForestClassifier','params':m.get_params(),'train_rows':len(y),'official_test_rows':len(yte),'encoded_features':A.shape[1],'prep_seconds':prep,'fit_seconds':fit,'inference_seconds':infer,'accuracy':float(accuracy_score(yte,pred)),'macro_f1':float(f),'weighted_f1':float(fw),'labels':labels,'confusion_matrix':confusion_matrix(yte,pred,labels=labels).tolist(),'selection_protocol':'Model family selected on training-only grouped validation; this run refits on the complete official training partition and evaluates official test once.'}
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/full_fit_random_forest';out.mkdir(parents=True,exist_ok=True)
(out/'results.json').write_text(json.dumps(rep,indent=2,default=float));joblib.dump({'preprocessor':pre,'model':m,'feature_names':pre.get_feature_names_out().tolist()},out/'model.joblib',compress=3)
print(json.dumps({k:rep[k] for k in ['train_rows','official_test_rows','encoded_features','prep_seconds','fit_seconds','inference_seconds','accuracy','macro_f1','weighted_f1']},indent=2),flush=True)
