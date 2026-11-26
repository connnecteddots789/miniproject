"""Full-train UNSW fixed subset XGBoost using filter/wrapper hybrid ranking.
Ranking is computed from a stratified sample of official training data only.
"""
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
from hybrid_adaptive_evaluation import get_data,rank_scores
from app.ml import stratified_sample_indices
X,y,Xte,yte,*_=get_data('UNSW-NB15');nums=X.select_dtypes(include=['number','bool']).columns.tolist();cats=[c for c in X.columns if c not in nums]
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median'))]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cats)],sparse_threshold=1.0)
t=time.perf_counter();A=pre.fit_transform(X);B=pre.transform(Xte);prep_s=time.perf_counter()-t;print('prepped',A.shape,B.shape,round(prep_s,2),flush=True)
si=stratified_sample_indices(y,12000,201,min_per_class=20);Ar=A[si].toarray() if hasattr(A,'toarray') else np.asarray(A[si]);rankings,scores=rank_scores(Ar,np.asarray(y,dtype=object)[si],202);idx=np.asarray(rankings['hybrid_rank_fusion'][:int(A.shape[1]*.50)],dtype=int);print('ranked',len(idx),'of',A.shape[1],'top',pre.get_feature_names_out()[idx[:15]].tolist(),flush=True)
enc=LabelEncoder().fit(y);params=dict(n_estimators=200,max_depth=7,learning_rate=.05,subsample=.9,colsample_bytree=.9,min_child_weight=3,reg_lambda=2,objective='multi:softprob',eval_metric='mlogloss',tree_method='hist',n_jobs=2,random_state=203,verbosity=0,num_class=len(enc.classes_))
m=XGBClassifier(**params);t=time.perf_counter();m.fit(A[:,idx],enc.transform(y));fit_s=time.perf_counter()-t;print('fit done',round(fit_s,2),flush=True)
t=time.perf_counter();pred=enc.inverse_transform(m.predict(B[:,idx]).astype(int));infer=time.perf_counter()-t;labels=list(enc.classes_);acc=float(accuracy_score(yte,pred));_,_,macro,_=precision_recall_fscore_support(yte,pred,labels=labels,average='macro',zero_division=0);_,_,weighted,_=precision_recall_fscore_support(yte,pred,labels=labels,average='weighted',zero_division=0)
report={'dataset':'UNSW-NB15','task':'native 10-class','selector':'hybrid rank fusion: gain ratio + mutual information + correlation eta2 + Decision Tree + ExtraTrees','feature_fraction':.5,'train_rows':len(y),'official_test_rows':len(yte),'encoded_features':A.shape[1],'selected_features':len(idx),'selected_feature_names':pre.get_feature_names_out()[idx].tolist(),'ranking_sample_rows':len(si),'model_parameters':params,'preprocess_seconds':prep_s,'training_seconds':fit_s,'inference_seconds':infer,'official_test_accuracy':acc,'official_test_macro_f1':float(macro),'official_test_weighted_f1':float(weighted),'labels':labels,'confusion_matrix':confusion_matrix(yte,pred,labels=labels).tolist(),'protocol':'Ranking fit on stratified sample of official training records only. Model refit on all official training labels. Official test labels used only for final metrics.'}
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/onehot_high_card';out.mkdir(parents=True,exist_ok=True);(out/'hybrid_subset_results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'preprocessor':pre,'encoder':enc,'model':m,'selected_indices':idx,'selected_features':report['selected_feature_names'],'feature_names':pre.get_feature_names_out().tolist()},out/'hybrid_subset_model.joblib',compress=3)
print('OFFICIAL_TEST',round(acc*100,4),'macroF1',round(float(macro)*100,4),'weightedF1',round(float(weighted)*100,4),flush=True)
