"""UNSW-NB15 transductive covariate-shift experiment.

Uses official test FEATURES without labels to estimate source->target density ratios.
Official test labels are read only after models and weights are frozen for reporting.
This is a separate, explicitly transductive research experiment, not the default
inductive benchmark or an application model replacement.
"""
import json,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT/'scripts'))
from optimize_adaptive_research import get_frames
from app.ml import make_preprocessor,stratified_sample_indices
X,y,Xt,yt,tr,te,task=get_frames('UNSW-NB15','native')
pre=make_preprocessor(X);t=time.perf_counter();print('preprocess start',flush=True);A=np.asarray(pre.fit_transform(X),dtype=np.float32);B=np.asarray(pre.transform(Xt),dtype=np.float32);prep_s=time.perf_counter()-t;print('preprocess done',A.shape,B.shape,round(prep_s,2),flush=True)
# Domain discriminator is fitted without any target labels.
ns=min(20000,len(A));nt=min(20000,len(B));si=stratified_sample_indices(y,ns,147,min_per_class=5);rng=np.random.default_rng(148);ti=rng.choice(len(B),size=nt,replace=False)
D=np.vstack([A[si],B[ti]]);dy=np.r_[np.zeros(len(si),dtype=np.int8),np.ones(len(ti),dtype=np.int8)]
domain=LGBMClassifier(n_estimators=90,num_leaves=15,max_depth=-1,learning_rate=.05,min_child_samples=50,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=3,n_jobs=1,random_state=149,verbosity=-1,force_col_wise=True)
t=time.perf_counter();print('domain fit start',D.shape,flush=True);domain.fit(D,dy);domain_s=time.perf_counter()-t;print('domain fit done',round(domain_s,2),flush=True)
pd=domain.predict_proba(D)[:,1];auc=float(roc_auc_score(dy,pd));print('domain auc',auc,'source weight inference start',flush=True);ps=domain.predict_proba(A)[:,1];print('source weight inference done',flush=True)
# Odds ratio with prior correction: p(target|x)/(1-p(target|x)) * n_source/n_target.
raw=(ps/np.clip(1-ps,1e-5,1))*(len(A)/len(B))
raw=np.clip(raw,1e-3,100.0);raw=raw/np.mean(raw)
labels=sorted(set(y)|set(yt));cfg=dict(n_estimators=90,num_leaves=15,max_depth=-1,learning_rate=.05,min_child_samples=35,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=3,max_bin=63,n_jobs=1,random_state=150,verbosity=-1,force_col_wise=True)
rows=[];models={}
for name,sw in [('ratio_clip_5',np.clip(raw,1/5,5))]:
 if sw is not None: sw=sw/np.mean(sw)
 m=LGBMClassifier(**cfg);t=time.perf_counter();print('weighted fit start',name,A.shape,flush=True);m.fit(A,y,sample_weight=sw);fit_s=time.perf_counter()-t;print('weighted fit done',round(fit_s,2),flush=True)
 t=time.perf_counter();print('test prediction start',flush=True);pred=np.asarray(m.predict(B),dtype=object);infer_s=time.perf_counter()-t;print('test prediction done',round(infer_s,2),flush=True)
 p,r,f,_=precision_recall_fscore_support(yt,pred,labels=labels,average='macro',zero_division=0);pw,rw,fw,_=precision_recall_fscore_support(yt,pred,labels=labels,average='weighted',zero_division=0)
 row={'name':name,'accuracy':float(accuracy_score(yt,pred)),'macro_f1':float(f),'weighted_f1':float(fw),'fit_seconds':fit_s,'inference_seconds':infer_s,'weight_stats':None if sw is None else {'min':float(sw.min()),'p50':float(np.median(sw)),'p95':float(np.quantile(sw,.95)),'max':float(sw.max()),'effective_sample_size':float(sw.sum()**2/(np.sum(sw**2)+1e-9))}}
 rows.append(row);models[name]=m;print(name,row,flush=True)
report={'dataset':'UNSW-NB15','task':'native 10-class','experiment':'transductive covariate-shift importance weighting','train_rows':len(y),'test_rows':len(yt),'encoded_features':A.shape[1],'preprocess_seconds':prep_s,'domain_classifier_auc':auc,'domain_classifier_fit_seconds':domain_s,'domain_classifier':'LightGBM 220 trees, 31 leaves; trained on sampled train/test features only','density_ratio':'p(target|x)/(1-p(target|x)) * n_source/n_target; mean-normalized and clipped per variant','official_test_results':rows,'protocol':'Official test FEATURES were used unlabeled for domain discrimination and weight estimation. Official test labels were held out and used only for final metrics. Results are transductive and must not be directly compared as an inductive-only benchmark without this caveat.'}
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/domain_adaptation';out.mkdir(parents=True,exist_ok=True)
(out/'results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'preprocessor':pre,'domain_model':domain,'models':models,'feature_names':pre.get_feature_names_out().tolist()},out/'candidate_models.joblib',compress=3)
print('DOMAIN_AUC',auc,'RESULTS',[(r['name'],round(r['accuracy']*100,3),round(r['macro_f1']*100,3)) for r in rows],flush=True)
