import json,sys,time
from pathlib import Path
import numpy as np,joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT/'scripts'))
from optimize_adaptive_research import get_frames
from app.ml import make_preprocessor
X,y,Xte,yte,*_=get_frames('UNSW-NB15','native');print('data loaded',len(y),len(yte),flush=True)
pre=make_preprocessor(X);t=time.perf_counter();A=np.asarray(pre.fit_transform(X),dtype=np.float32);B=np.asarray(pre.transform(Xte),dtype=np.float32);pt=time.perf_counter()-t;print('prepped',A.shape,B.shape,round(pt,2),flush=True)
enc=LabelEncoder().fit(y);params=dict(n_estimators=250,max_depth=7,learning_rate=.05,min_child_weight=3,reg_lambda=4.0,gamma=.05,subsample=.9,colsample_bytree=.9,objective='multi:softprob',eval_metric='mlogloss',tree_method='hist',n_jobs=2,random_state=181,verbosity=0,num_class=len(enc.classes_))
m=XGBClassifier(**params);t=time.perf_counter();m.fit(A,enc.transform(y));ft=time.perf_counter()-t;print('fitted',round(ft,2),flush=True)
t=time.perf_counter();pred=enc.inverse_transform(m.predict(B).astype(int));it=time.perf_counter()-t
labels=list(enc.classes_);acc=float(accuracy_score(yte,pred));_,_,macro,_=precision_recall_fscore_support(yte,pred,labels=labels,average='macro',zero_division=0);_,_,weighted,_=precision_recall_fscore_support(yte,pred,labels=labels,average='weighted',zero_division=0)
result={'dataset':'UNSW-NB15','task':'native 10-class','train_rows':len(y),'test_rows':len(yte),'feature_count':A.shape[1],'parameters':params,'training_validation_selection':'Selected depth7_250: 81.010% accuracy on duplicate-grouped validation within official training data; model-family/params selected before test scoring.','preprocessing_seconds':pt,'full_fit_seconds':ft,'inference_seconds':it,'official_test_accuracy':acc,'official_test_macro_f1':float(macro),'official_test_weighted_f1':float(weighted),'labels':labels,'confusion_matrix':confusion_matrix(yte,pred,labels=labels).tolist(),'protocol':'Final refit on all official training rows; no official test labels used for training or model selection.'}
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/xgb_hyperparameter_search';out.mkdir(parents=True,exist_ok=True);(out/'results_full_refit.json').write_text(json.dumps(result,indent=2,default=float));joblib.dump({'preprocessor':pre,'encoder':enc,'model':m,'feature_names':pre.get_feature_names_out().tolist()},out/'model_depth7_250.joblib',compress=3)
print('OFFICIAL TEST',round(acc*100,4),'macroF1',round(float(macro)*100,4),'weightedF1',round(float(weighted)*100,4),'inference_s',round(it,2),flush=True)
