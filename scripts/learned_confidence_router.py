"""Train a validation-only correctness router for fixed-vs-all feature acquisition.
The router uses only compact-model probabilities at inference time.
"""
import sys,json,time,joblib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data,predict,metrics,SEED,OUT

def meta_features(prob):
 p=np.asarray(prob,dtype=float); ps=np.sort(p,axis=1)
 maxp=ps[:,-1]; second=ps[:,-2] if p.shape[1]>1 else np.zeros(len(p))
 entropy=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)
 return np.column_stack([p,maxp,second,maxp-second,entropy])

def run(ds):
 dname=ds.lower().replace('-','_'); base=OUT/dname; cp=base/'candidate_checkpoint.joblib'
 b=joblib.load(cp)
 X,y,Xte,yte,*_=get_data(ds)
 groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
 grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
 gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy())
 mask=np.isin(groups,gtr); Xval=X.loc[~mask].reset_index(drop=True); yval=y[~mask]
 pre=b['preprocessor'];Zval=np.asarray(pre.transform(Xval),dtype=np.float32);idx=b['fixed_indices'];
 pfix,cls=predict(b['fixed'],Zval[:,idx],proba=True); pfix=np.asarray(pfix,dtype=float)
 pall=np.asarray(predict(b['all'],Zval),dtype=object); pfixpred=np.asarray(predict(b['fixed'],Zval[:,idx]),dtype=object)
 fixed_correct=(pfixpred==yval).astype(int)
 Xmeta=meta_features(pfix)
 ix_train,ix_gate=train_test_split(np.arange(len(yval)),test_size=.35,random_state=SEED+44,stratify=fixed_correct)
 router=ExtraTreesClassifier(n_estimators=160,max_depth=6,min_samples_leaf=50,max_features=.85,
    class_weight=None,n_jobs=1,random_state=SEED+45)
 t=time.perf_counter(); router.fit(Xmeta[ix_train],fixed_correct[ix_train]); trainsecs=time.perf_counter()-t
 ptrust=router.predict_proba(Xmeta[ix_gate])[:,list(router.classes_).index(1)]
 thresholds=np.unique(np.quantile(ptrust,np.linspace(.02,.98,97)))
 cand=[]
 for th in thresholds:
  use=ptrust>=float(th);pp=pall[ix_gate].copy();pp[use]=pfixpred[ix_gate][use]
  cand.append((accuracy_score(yval[ix_gate],pp),float(th),use.mean(),pp))
 best=max(cand,key=lambda q:q[0])
 gate_all=metrics(yval[ix_gate],pall[ix_gate],sorted(set(y)|set(yte)))
 gate_fixed=metrics(yval[ix_gate],pfixpred[ix_gate],sorted(set(y)|set(yte)))
 gate_dyn=metrics(yval[ix_gate],best[3],sorted(set(y)|set(yte)))
 # Score official test. Test labels are only passed to final metrics, never router fitting/thresholding.
 Xtest=b['official_test_X']; Ztest=np.asarray(pre.transform(Xtest),dtype=np.float32)
 pft,clst=predict(b['fixed'],Ztest[:,idx],proba=True);pft=np.asarray(pft,dtype=float)
 alltest=np.asarray(predict(b['all'],Ztest),dtype=object);fixtest=np.asarray(predict(b['fixed'],Ztest[:,idx]),dtype=object)
 ptrust_test=router.predict_proba(meta_features(pft))[:,list(router.classes_).index(1)]
 use=ptrust_test>=best[1]; dyn=alltest.copy();dyn[use]=fixtest[use]
 labels=sorted(set(y)|set(yte))
 results={'all':metrics(yte,alltest,labels),'fixed':metrics(yte,fixtest,labels),'dynamic':metrics(yte,dyn,labels)}
 fallback=float((~use).mean()); avg=len(idx)+fallback*(Ztest.shape[1]-len(idx))
 results['all']['feature_count']=float(Ztest.shape[1]);results['fixed']['feature_count']=float(len(idx))
 results['dynamic'].update({'adaptive_mode':'learned compact-correctness router','router_threshold':float(best[1]),'fallback_fraction':fallback,'average_features':avg,'feature_count':avg})
 report={'dataset':ds,'task':'native multiclass','router':'ExtraTrees correctness meta-router','meta_features':['compact class probabilities','max probability','second-best probability','top-two margin','entropy'],'router_train_rows':len(ix_train),'router_validation_rows':len(ix_gate),'router_fit_seconds':trainsecs,
  'validation_gate_metrics':{'all':gate_all,'fixed':gate_fixed,'dynamic':gate_dyn},'threshold_selected_on_validation_only':best[1], 'official_test_rows':len(yte),'official_test_results':results,
  'protocol':'Base models trained only on the training-side fold/sample. Meta-router trained on validation predictions and their validation correctness; threshold selected on a separate validation-gate holdout; official test labels used only for final metrics.'}
 out=base/'learned_router';out.mkdir(parents=True,exist_ok=True)
 (out/'results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'router':router,'threshold':best[1],'meta_feature_schema':report['meta_features'],'base_checkpoint':str(cp)},out/'router.joblib',compress=3)
 print(ds,'val all/fixed/dyn',gate_all['accuracy'],gate_fixed['accuracy'],gate_dyn['accuracy'],'threshold',best[1],flush=True)
 for k,v in results.items():print(k,'accuracy %.4f macroF1 %.4f weightedF1 %.4f features %.1f'%(v['accuracy'],v['macro_f1'],v['weighted_f1'],v.get('feature_count',0)),flush=True)

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True);args=a.parse_args();run(args.dataset)
