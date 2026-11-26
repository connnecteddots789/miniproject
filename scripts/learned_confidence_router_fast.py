"""Learned correctness router using compact features + calibrated score profile.
Router training/threshold selection use only a dedicated validation gate split.
"""
import sys,json,time,joblib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data,predict,metrics,SEED,OUT

def features(prob,z):
 p=np.asarray(prob,dtype=float); ps=np.sort(p,axis=1)
 maxp=ps[:,-1];second=ps[:,-2] if p.shape[1]>1 else np.zeros(len(p))
 ent=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)
 summary=np.column_stack([maxp,second,maxp-second,ent])
 return np.column_stack([p,summary,np.asarray(z,dtype=float)])

def best_gate(y,allp,fixp,ptrust):
 thresholds=np.unique(np.quantile(ptrust,np.linspace(.01,.99,99)))
 cand=[]
 for th in thresholds:
  use=ptrust>=float(th); pred=allp.copy();pred[use]=fixp[use]
  cand.append((float(accuracy_score(y,pred)),float(th),float(use.mean()),pred))
 return max(cand,key=lambda x:x[0])

def run(ds):
 dn=ds.lower().replace('-','_');base=OUT/dn; b=joblib.load(base/'candidate_checkpoint.joblib')
 X,y,Xte,yte,*_=get_data(ds);groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
 grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
 gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy())
 Xval=X.loc[~np.isin(groups,gtr)].reset_index(drop=True); yval=y[~np.isin(groups,gtr)]
 pre=b['preprocessor'];Zval=np.asarray(pre.transform(Xval),dtype=np.float32);idx=b['fixed_indices']
 pfix,cls=predict(b['fixed'],Zval[:,idx],proba=True);pfix=np.asarray(pfix,dtype=float)
 fpred=np.asarray(predict(b['fixed'],Zval[:,idx]),dtype=object);apred=np.asarray(predict(b['all'],Zval),dtype=object)
 ycorrect=(fpred==yval).astype(int); Xmeta=features(pfix,Zval[:,idx])
 i_train,i_gate=train_test_split(np.arange(len(yval)),test_size=.35,random_state=SEED+44,stratify=ycorrect)
 models=[
  ('ExtraTrees',ExtraTreesClassifier(n_estimators=200,max_depth=7,min_samples_leaf=40,max_features=.85,n_jobs=1,random_state=SEED+51))]
 candidates=[]
 for name,model in models:
  t=time.perf_counter();model.fit(Xmeta[i_train],ycorrect[i_train]); fitsec=time.perf_counter()-t
  pos=list(model.classes_).index(1);ptrust=model.predict_proba(Xmeta[i_gate])[:,pos]
  best=best_gate(yval[i_gate],apred[i_gate],fpred[i_gate],ptrust)
  candidates.append({'name':name,'model':model,'fit_seconds':fitsec,'validation_accuracy':best[0],'threshold':best[1],'fixed_fraction':best[2],'val_predictions':best[3]})
  print(name,'gate validation accuracy',round(best[0],6),'fit',round(fitsec,2),flush=True)
 best=max(candidates,key=lambda z:z['validation_accuracy']);router=best['model'];th=best['threshold']
 labels=sorted(set(y)|set(yte));gate_idx=i_gate
 gate_all=metrics(yval[gate_idx],apred[gate_idx],labels);gate_fix=metrics(yval[gate_idx],fpred[gate_idx],labels);gate_dyn=metrics(yval[gate_idx],best['val_predictions'],labels)
 # Test scoring: router observes only compact feature values + compact probabilities.
 Xtest=b['official_test_X'];Ztest=np.asarray(pre.transform(Xtest),dtype=np.float32);pft,clst=predict(b['fixed'],Ztest[:,idx],proba=True);pft=np.asarray(pft,dtype=float)
 alltest=np.asarray(predict(b['all'],Ztest),dtype=object);fixtest=np.asarray(predict(b['fixed'],Ztest[:,idx]),dtype=object)
 pt=router.predict_proba(features(pft,Ztest[:,idx]))[:,list(router.classes_).index(1)];use=pt>=th;dyn=alltest.copy();dyn[use]=fixtest[use]
 results={'all':metrics(yte,alltest,labels),'fixed':metrics(yte,fixtest,labels),'dynamic':metrics(yte,dyn,labels)}
 fallback=float((~use).mean());avg=len(idx)+fallback*(Ztest.shape[1]-len(idx));results['all']['feature_count']=float(Ztest.shape[1]);results['fixed']['feature_count']=float(len(idx));results['dynamic'].update({'adaptive_mode':'learned compact-feature correctness router','router_model':best['name'],'router_threshold':float(th),'fallback_fraction':fallback,'average_features':avg,'feature_count':avg})
 report={'dataset':ds,'task':'native multiclass','router_model':best['name'],'router_meta_features':'compact class probabilities + confidence profile + selected encoded traffic features','router_train_rows':len(i_train),'router_validation_rows':len(i_gate),'router_fit_seconds':best['fit_seconds'],'router_candidates':[{'name':c['name'],'fit_seconds':c['fit_seconds'],'validation_accuracy':c['validation_accuracy'],'threshold':c['threshold'],'fixed_fraction':c['fixed_fraction']} for c in candidates], 'selected_threshold_validation_only':th,'validation_gate':{'all':gate_all,'fixed':gate_fix,'dynamic':gate_dyn},'official_test_rows':len(yte),'official_test_results':results,'protocol':'Base models trained on official training-side fit fold. Router trained on a separate validation partition to estimate compact-model correctness; threshold selected on held-out validation-gate rows. Test labels used only for final metrics.'}
 out=base/'learned_router_v2';out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'router':router,'threshold':th,'feature_schema':report['router_meta_features']},out/'router.joblib',compress=3)
 print('selected router',best['name'],'val accuracy',best['validation_accuracy'],'threshold',th,flush=True)
 for k,v in results.items():print(k,'accuracy %.4f macroF1 %.4f weightedF1 %.4f features %.1f'%(v['accuracy'],v['macro_f1'],v['weighted_f1'],v.get('feature_count',0)),flush=True)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True);args=a.parse_args();run(args.dataset)
