"""Full official-training refit of the one-hot-high-cardinality UNSW candidate."""
import sys,time,json
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from xgboost import XGBClassifier
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data
X,y,Xte,yte,*_=get_data('UNSW-NB15');nums=X.select_dtypes(include=['number','bool']).columns.tolist();cats=[c for c in X.columns if c not in nums]
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median'))]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cats)],sparse_threshold=1.0)
t=time.perf_counter();A=pre.fit_transform(X);B=pre.transform(Xte);pt=time.perf_counter()-t;print('prepped',A.shape,B.shape,round(pt,2),flush=True)
enc=LabelEncoder().fit(y);params=dict(n_estimators=200,max_depth=7,learning_rate=.05,subsample=.9,colsample_bytree=.9,min_child_weight=3,reg_lambda=2,objective='multi:softprob',eval_metric='mlogloss',tree_method='hist',n_jobs=2,random_state=42,verbosity=0,num_class=len(enc.classes_))
m=XGBClassifier(**params);t=time.perf_counter();print('full fit start',len(y),flush=True);m.fit(A,enc.transform(y));ft=time.perf_counter()-t;print('full fit done',round(ft,2),flush=True)
t=time.perf_counter();pred=enc.inverse_transform(m.predict(B).astype(int));it=time.perf_counter()-t;labels=list(enc.classes_);acc=float(accuracy_score(yte,pred));_,_,macro,_=precision_recall_fscore_support(yte,pred,labels=labels,average='macro',zero_division=0);_,_,weighted,_=precision_recall_fscore_support(yte,pred,labels=labels,average='weighted',zero_division=0)
report={'dataset':'UNSW-NB15','task':'native 10-class','encoding':'one-hot all categorical columns including proto','train_rows':len(y),'official_test_rows':len(yte),'encoded_features':A.shape[1],'parameters':params,'training_validation_selection':'One-hot high-cardinality validation accuracy 80.8764%, selected on duplicate-grouped validation inside official training partition.','preprocess_seconds':pt,'full_refit_seconds':ft,'inference_seconds':it,'official_test_accuracy':acc,'official_test_macro_f1':float(macro),'official_test_weighted_f1':float(weighted),'labels':labels,'confusion_matrix':confusion_matrix(yte,pred,labels=labels).tolist(),'protocol':'Preprocessor refit on all official training features; XGBoost refit on all official training labels. Official test labels used only for final metrics.'}
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/onehot_high_card';out.mkdir(parents=True,exist_ok=True);(out/'full_refit_results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'preprocessor':pre,'encoder':enc,'model':m,'feature_names':pre.get_feature_names_out().tolist()},out/'full_model.joblib',compress=3)
print('OFFICIAL_TEST',round(acc*100,4),'macroF1',round(float(macro)*100,4),'weightedF1',round(float(weighted)*100,4),'fit_s',round(ft,2),'infer_s',round(it,2),flush=True)
