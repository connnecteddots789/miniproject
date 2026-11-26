"""Training-only XGBoost hyperparameter screen, then full official-train refit."""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT/'scripts'))
from optimize_adaptive_research import get_frames
from app.ml import make_preprocessor,stratified_sample_indices
X,y,Xte,yte,tr,te,task=get_frames('UNSW-NB15','native')
# Duplicate-grouped validation using official training data only.
g=pd.util.hash_pandas_object(X,index=False).to_numpy();gl=pd.DataFrame({'g':g,'y':y}).groupby('g',sort=False).y.agg(lambda s:s.value_counts().index[0])
gfit,gval=train_test_split(gl.index.to_numpy(),test_size=.15,random_state=73,stratify=gl.to_numpy());fitset=set(gfit.tolist());itr=np.flatnonzero(np.isin(g,list(fitset)));iva=np.flatnonzero(~np.isin(g,list(fitset)))
Xfit=X.iloc[itr].reset_index(drop=True);Xval=X.iloc[iva].reset_index(drop=True);yfit=y[itr];yval=y[iva]
pre=make_preprocessor(Xfit);Af=np.asarray(pre.fit_transform(Xfit),dtype=np.float32);Av=np.asarray(pre.transform(Xval),dtype=np.float32)
si=stratified_sample_indices(yfit,12000,180,min_per_class=20);A=Af[si];ys=np.asarray(yfit,dtype=object)[si];enc=LabelEncoder().fit(ys);ye=enc.transform(ys);yv=enc.transform(yval)
candidates=[
 ('depth5_300',dict(n_estimators=300,max_depth=5,learning_rate=.05,min_child_weight=3,reg_lambda=3.0,gamma=.05)),
 ('depth7_250',dict(n_estimators=250,max_depth=7,learning_rate=.05,min_child_weight=3,reg_lambda=4.0,gamma=.05)),
 ('depth4_450',dict(n_estimators=450,max_depth=4,learning_rate=.04,min_child_weight=2,reg_lambda=3.0,gamma=.02)),
 ('depth6_350',dict(n_estimators=350,max_depth=6,learning_rate=.04,min_child_weight=4,reg_lambda=5.0,gamma=.1)),
]
base=dict(subsample=.9,colsample_bytree=.9,objective='multi:softprob',eval_metric='mlogloss',tree_method='hist',n_jobs=1,random_state=181,verbosity=0)
rows=[]
for name,params in candidates:
 m=XGBClassifier(**base,**params,num_class=len(enc.classes_));t=time.perf_counter();m.fit(A,ye);ft=time.perf_counter()-t;pred=enc.inverse_transform(m.predict(Av).astype(int));acc=float(accuracy_score(yval,pred));macro=float(precision_recall_fscore_support(yval,pred,labels=enc.classes_,average='macro',zero_division=0)[2]);rows.append({'name':name,'params':params,'validation_accuracy':acc,'validation_macro_f1':macro,'fit_seconds':ft});print(name,'val_acc',round(acc*100,3),'macro',round(macro*100,3),'fit_s',round(ft,2),flush=True)
# Select only from training-side validation accuracy, the user's primary metric.
best=max(rows,key=lambda r:(r['validation_accuracy'],r['validation_macro_f1']));print('selected',best['name'],flush=True)
# Full official training refit with a fresh preprocessor.
pre_full=make_preprocessor(X);t=time.perf_counter();Afull=np.asarray(pre_full.fit_transform(X),dtype=np.float32);Btest=np.asarray(pre_full.transform(Xte),dtype=np.float32);prep_s=time.perf_counter()-t
enc_full=LabelEncoder().fit(y);model=XGBClassifier(**base,**best['params'],num_class=len(enc_full.classes_));t=time.perf_counter();model.fit(Afull,enc_full.transform(y));fit_s=time.perf_counter()-t
pred=enc_full.inverse_transform(model.predict(Btest).astype(int));labels=list(enc_full.classes_);p,r,f,_=precision_recall_fscore_support(yte,pred,labels=labels,average='macro',zero_division=0);pw,rw,fw,_=precision_recall_fscore_support(yte,pred,labels=labels,average='weighted',zero_division=0)
result={'dataset':'UNSW-NB15','task':'native 10-class','train_rows':len(y),'test_rows':len(yte),'encoded_features':Afull.shape[1],'model_fit_rows':len(y),'validation_fit_sample_rows':len(si),'validation_rows':len(yval),'validation_candidates':rows,'selected_candidate':best['name'],'selected_hyperparameters':best['params'],'preprocess_seconds':prep_s,'full_refit_seconds':fit_s,'official_test_accuracy':float(accuracy_score(yte,pred)),'official_test_macro_f1':float(f),'official_test_weighted_f1':float(fw),'labels':labels,'confusion_matrix':confusion_matrix(yte,pred,labels=labels).tolist(),'protocol':'Hyperparameters selected by duplicate-grouped validation within official training data. Final estimator refit on all official training rows. Official test labels used only for final metrics.'}
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/xgb_hyperparameter_search';out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(result,indent=2,default=float));joblib.dump({'preprocessor':pre_full,'label_encoder':enc_full,'model':model,'feature_names':pre_full.get_feature_names_out().tolist()},out/'model.joblib',compress=3)
print('FULL_TEST',result['official_test_accuracy']*100,'macroF1',result['official_test_macro_f1']*100,'weightedF1',result['official_test_weighted_f1']*100,'fit_s',fit_s,flush=True)
