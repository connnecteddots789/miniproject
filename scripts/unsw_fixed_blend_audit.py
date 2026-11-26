"""Predeclared probability blends of separately trained full-feature UNSW candidates.
Weights are fixed before scoring; the test labels do not select a blend weight."""
import sys,json,time
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT/'scripts'))
from optimize_adaptive_research import get_frames
X,y,Xt,yt,*_=get_frames('UNSW-NB15','native')
xgb=joblib.load(ROOT/'backend/artifacts/research/unsw_nb15_native_xgboost_accuracy/model.joblib')
dom=joblib.load(ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/domain_adaptation/candidate_models.joblib')
Zx=np.asarray(xgb['preprocessor'].transform(Xt),dtype=np.float32);px=xgb['model'].predict_proba(Zx);cx=np.asarray(xgb['model'].classes_,dtype=object)
Zd=np.asarray(dom['preprocessor'].transform(Xt),dtype=np.float32);md=dom['models']['ratio_clip_5'];pd=md.predict_proba(Zd);cd=np.asarray(md.classes_,dtype=object)
# align classes to xgb model ordering
cdmap={str(c):i for i,c in enumerate(cd)};pd=pd[:,[cdmap[str(c)] for c in cx]]
labels=list(cx);rows=[]
for w in [0.25,0.5,0.75]:
 prob=w*px+(1-w)*pd;pred=np.asarray(labels,dtype=object)[np.argmax(prob,axis=1)]
 p,r,f,_=precision_recall_fscore_support(yt,pred,labels=labels,average='macro',zero_division=0);pw,rw,fw,_=precision_recall_fscore_support(yt,pred,labels=labels,average='weighted',zero_division=0)
 rows.append({'xgb_weight':w,'domain_weight':1-w,'accuracy':float(accuracy_score(yt,pred)),'macro_f1':float(f),'weighted_f1':float(fw)})
print('xgb baseline',xgb['metrics']['accuracy'],'domain baseline',json.load(open(ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/domain_adaptation/results.json'))['official_test_results'][0]['accuracy'])
print(rows)
out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition/unsw_nb15/domain_adaptation/blend_results.json';out.write_text(json.dumps({'protocol':'Fixed blend weights 0.25/0.50/0.75 specified before metric calculation; no test labels used to choose blend weight.','rows':rows},indent=2))
