"""Training-only validation of full one-hot categorical expansion for UNSW-NB15."""
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data,SEED
from app.ml import stratified_sample_indices
X,y,Xte,yte,*_=get_data('UNSW-NB15');g=pd.util.hash_pandas_object(X,index=False).to_numpy();gl=pd.DataFrame({'g':g,'y':y}).groupby('g',sort=False).y.agg(lambda s:s.value_counts().index[0]);gfit,gval=train_test_split(gl.index.to_numpy(),test_size=.15,random_state=SEED,stratify=gl.to_numpy());mask=np.isin(g,gfit);Xf=X.loc[mask].reset_index(drop=True);Xv=X.loc[~mask].reset_index(drop=True);yf=y[mask];yv=y[~mask]
nums=Xf.select_dtypes(include=['number','bool']).columns.tolist();cats=[c for c in Xf.columns if c not in nums]
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median'))]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=True))]),cats)],sparse_threshold=1.0)
t=time.perf_counter();A=pre.fit_transform(Xf);B=pre.transform(Xv);print('preprocess',A.shape,B.shape,round(time.perf_counter()-t,2),flush=True)
idx=stratified_sample_indices(yf,12000,SEED+7,min_per_class=20);enc=LabelEncoder().fit(yf);ya=enc.transform(np.asarray(yf,dtype=object)[idx]);yb=enc.transform(yv)
params=dict(n_estimators=200,max_depth=7,learning_rate=.05,subsample=.9,colsample_bytree=.9,min_child_weight=3,reg_lambda=2,objective='multi:softprob',eval_metric='mlogloss',tree_method='hist',n_jobs=2,random_state=42,verbosity=0,num_class=len(enc.classes_))
m=XGBClassifier(**params);t=time.perf_counter();m.fit(A[idx],ya);ft=time.perf_counter()-t;pred=enc.inverse_transform(m.predict(B).astype(int));acc=float(accuracy_score(yv,pred));macro=float(precision_recall_fscore_support(yv,pred,average='macro',zero_division=0)[2]);print('onehot val',round(acc*100,4),'macro',round(macro*100,4),'fit',round(ft,2),flush=True)
# Save fitted validation-only candidate for explicit follow-up/full refit if it wins.
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/onehot_high_card';out.mkdir(parents=True,exist_ok=True)
report={'dataset':'UNSW-NB15','task':'native 10-class','encoding':'one-hot all categoricals (including proto=133 categories)','model':'XGBoost','training_fit_rows':len(yf),'validation_rows':len(yv),'model_fit_sample_rows':len(idx),'encoded_features':A.shape[1],'params':params,'validation_accuracy':acc,'validation_macro_f1':macro,'fit_seconds':ft,'protocol':'Training-only grouped validation; official test was not used.'}
(out/'validation.json').write_text(json.dumps(report,indent=2));
