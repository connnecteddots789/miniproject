"""Hierarchical adaptive-feature IDS experiment: normal/attack router + attack-family specialist.
All ranking, fitting and threshold selection use only the official training file.
The official test file is scored once and never used for fitting or tuning.
"""
from __future__ import annotations
import json,time,sys
from pathlib import Path
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import (get_data,rank_scores,make_model,fit_model,predict,metrics,SEED,OUT)
from app.ml import make_preprocessor,stratified_sample_indices
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run(ds,model_rows=40000):
 print('==== hierarchical',ds,flush=True)
 X,y,Xte,yte,tr,te=get_data(ds); labels=sorted(set(y)|set(yte))
 groups=__import__('pandas').util.hash_pandas_object(X,index=False).to_numpy()
 grp=__import__('pandas').DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
 gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy())
 mask=np.isin(groups,gtr); Xfit=X.loc[mask].reset_index(drop=True); Xval=X.loc[~mask].reset_index(drop=True); yfit=y[mask]; yval=y[~mask]
 pre=make_preprocessor(Xfit); Zfit=np.asarray(pre.fit_transform(Xfit),dtype=np.float32); Zval=np.asarray(pre.transform(Xval),dtype=np.float32)
 ridx=stratified_sample_indices(yfit,min(8000,len(yfit)),SEED+11,min_per_class=5)
 # Global hybrid feature ranking and binary-router ranking, computed on training-only rows.
 ranks_global,_=rank_scores(Zfit[ridx],yfit[ridx],SEED+13)
 ybin_fit=np.where(yfit=='Normal','Normal','Attack')
 ranks_router,_=rank_scores(Zfit[ridx],ybin_fit[ridx],SEED+17)
 d=Zfit.shape[1]; k=max(3,int(round(d*.50)))
 fixed_idx=ranks_global['extra_trees'][:k]
 router_idx=ranks_router['gain_ratio_decision_tree'][:k]
 tune=stratified_sample_indices(yfit,min(model_rows,len(yfit)),SEED+21,min_per_class=10)
 Zt=Zfit[tune]; yt=yfit[tune]; ybt=ybin_fit[tune]
 # Rank attack specialist on attack-only rows only; no validation/test labels enter.
 arows=np.flatnonzero(yt!='Normal')
 arank_rows=arows[:min(8000,len(arows))]
 ranks_attack,_=rank_scores(Zt[arank_rows],yt[arank_rows],SEED+19)
 attack_idx=ranks_attack['hybrid_rank_fusion'][:k]
 print('features',d,'fixed',len(fixed_idx),'router',len(router_idx),'attack specialist',len(attack_idx),'fit rows',len(tune),'attack fit rows',len(arows),flush=True)
 # Flat baselines and the two adaptive stages share the same sampled train rows.
 allm,_=fit_model('LightGBM',Zt,yt,SEED+30)
 fixm,_=fit_model('LightGBM',Zt[:,fixed_idx],yt,SEED+31)
 binm,_=fit_model('LightGBM',Zt[:,router_idx],ybt,SEED+32)
 attm,_=fit_model('LightGBM',Zt[arows][:,attack_idx],yt[arows],SEED+33)
 pallval=np.asarray(predict(allm,Zval),dtype=object)
 pfixval=np.asarray(predict(fixm,Zval[:,fixed_idx]),dtype=object)
 pbinval,bcls=predict(binm,Zval[:,router_idx],proba=True)
 batt=np.asarray(predict(attm,Zval[:,attack_idx]),dtype=object)
 attack_col=list(bcls).index('Attack')
 prob_attack=pbinval[:,attack_col]
 # Threshold calibrated on validation only.
 thresholds=np.unique(np.quantile(prob_attack,np.linspace(.02,.98,49)))
 cand=[]
 for th in thresholds:
  pp=np.where(prob_attack<float(th),'Normal',batt).astype(object)
  cand.append((float(accuracy_score(yval,pp)),float(th),pp))
 best=max(cand,key=lambda x:x[0]); dynval=metrics(yval,best[2],labels)
 print('validation all/fixed/dynamic',accuracy_score(yval,pallval),accuracy_score(yval,pfixval),best[0],'threshold',best[1],flush=True)
 # Checkpoint before official test scoring.
 out=OUT/ds.lower().replace('-','_')/'hierarchical'; out.mkdir(parents=True,exist_ok=True)
 ck={'preprocessor':pre,'all':allm,'fixed':fixm,'binary':binm,'attack':attm,'fixed_idx':fixed_idx,'router_idx':router_idx,'attack_idx':attack_idx,
  'threshold':best[1],'labels':labels,'feature_count':d,'model_fit_rows':len(tune),'official_test_X':Xte,'official_test_y':yte}
 joblib.dump(ck,out/'checkpoint.joblib',compress=0); print('checkpoint saved',flush=True)
 Ztest=np.asarray(pre.transform(Xte),dtype=np.float32)
 pall=np.asarray(predict(allm,Ztest),dtype=object); pfix=np.asarray(predict(fixm,Ztest[:,fixed_idx]),dtype=object)
 pbin,bcl=predict(binm,Ztest[:,router_idx],proba=True); patt=np.asarray(predict(attm,Ztest[:,attack_idx]),dtype=object)
 pa=pbin[:,list(bcl).index('Attack')]; use_attack=pa>=best[1]
 pdyn=np.where(use_attack,patt,'Normal').astype(object)
 results={'all':metrics(yte,pall,labels),'fixed':metrics(yte,pfix,labels),'dynamic':metrics(yte,pdyn,labels)}
 avg_features=len(router_idx)+float(use_attack.mean())*len(set(attack_idx)-set(router_idx))
 results['all']['feature_count']=float(d); results['fixed']['feature_count']=float(len(fixed_idx)); results['dynamic'].update({'feature_count':avg_features,'average_features':avg_features,'attack_route_fraction':float(use_attack.mean()),'threshold':best[1],'adaptive_mode':'binary-router-to-attack-specialist'})
 report={'dataset':ds,'task':'native multiclass','protocol':'Duplicate-grouped 85/15 training-only validation; 40k-or-configured shared stratified fit sample; rankings and router threshold fit/tuned without official test labels; official test used only for final scoring.','train_rows':len(y),'training_fit_rows':len(yfit),'model_fit_rows':len(tune),'validation_rows':len(yval),'official_test_rows':len(yte),'feature_count':d,'selector':'Gain Ratio + Decision Tree router rank; ExtraTrees flat baseline; attack-only hybrid specialist rank','selected_features':{'fixed':[str(pre.get_feature_names_out()[i]) for i in fixed_idx],'router':[str(pre.get_feature_names_out()[i]) for i in router_idx],'attack_specialist':[str(pre.get_feature_names_out()[i]) for i in attack_idx]},'validation_dynamic':dynval,'threshold':best[1],'official_test_results':results}
 (out/'results.json').write_text(json.dumps(report,indent=2,default=float)); ck.pop('official_test_X'); ck.pop('official_test_y'); joblib.dump(ck,out/'candidate_bundle.joblib',compress=0)
 for k,v in results.items(): print(k,'acc %.4f macroF1 %.4f weightedF1 %.4f features %.1f'%(v['accuracy'],v['macro_f1'],v['weighted_f1'],v.get('feature_count',0)),flush=True)
 return report
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True); ap.add_argument('--model-rows',type=int,default=40000); a=ap.parse_args(); run(a.dataset,a.model_rows)
