"""Black-box label-shift estimation from validation confusion probabilities.
Only unlabeled official-test features are used to estimate target priors; labels are for final score.
"""
import sys,json,joblib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data,predict,metrics,SEED,OUT

def run(ds):
 dn=ds.lower().replace('-','_');b=joblib.load(OUT/dn/'candidate_checkpoint.joblib')
 X,y,Xte,yte,*_=get_data(ds);g=pd.util.hash_pandas_object(X,index=False).to_numpy();grp=pd.DataFrame({'g':g,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0]);gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy())
 fitmask=np.isin(g,gtr);yfit=y[fitmask];yval=y[~fitmask];Xval=X.loc[~fitmask].reset_index(drop=True)
 pre=b['preprocessor'];Zval=np.asarray(pre.transform(Xval),dtype=np.float32);Zte=np.asarray(pre.transform(Xte),dtype=np.float32)
 pv,cls=predict(b['all'],Zval,proba=True);pt,clst=predict(b['all'],Zte,proba=True)
 order=[list(clst).index(c) for c in cls];pt=pt[:,order]
 labels=list(cls);K=len(labels)
 # Soft confusion/response matrix: expected predicted posterior vector conditional on each true label.
 C=np.zeros((K,K),float);source_prior=[]
 for j,c in enumerate(labels):
  m=np.asarray(yval==c)
  C[:,j]=pv[m].mean(axis=0) if m.any() else np.eye(K)[:,j]
  source_prior.append(float(np.mean(yfit==c)))
 target=pt.mean(axis=0)
 q,res=nnls(C,target);q=q/max(1e-12,q.sum())
 # Balanced model fitting induces a near-uniform effective prior; infer from metadata.
 report_path=OUT/dn/'comparison.json'
 cw=json.load(open(report_path)).get('class_weight') if report_path.exists() else None
 effective=np.ones(K)/K if cw=='balanced' else np.asarray(source_prior,float)
 adjusted=pt*(q/np.clip(effective,1e-8,None))[None,:]
 adjusted=adjusted/np.clip(adjusted.sum(axis=1,keepdims=True),1e-12,None)
 # Also calculate uncorrected probability / predicted output.
 pred_raw=np.asarray(cls,dtype=object)[np.argmax(pt,axis=1)]
 pred_adj=np.asarray(cls,dtype=object)[np.argmax(adjusted,axis=1)]
 labels_sorted=sorted(set(y)|set(yte))
 result={'raw':metrics(yte,pred_raw,labels_sorted),'bbse_corrected':metrics(yte,pred_adj,labels_sorted)}
 # Validation-side result using same correction inferred only from held-out predictions (diagnostic, not selection).
 pv_adj=pv*(q/np.clip(effective,1e-8,None))[None,:];pv_adj/=np.clip(pv_adj.sum(axis=1,keepdims=True),1e-12,None)
 val_raw=np.asarray(cls,dtype=object)[np.argmax(pv,axis=1)];val_adj=np.asarray(cls,dtype=object)[np.argmax(pv_adj,axis=1)]
 rep={'dataset':ds,'method':'soft confusion-matrix Black Box Shift Estimation (BBSE)','source_prior':dict(zip(labels,source_prior)),'effective_model_prior':dict(zip(labels,effective.tolist())),'estimated_target_prior':dict(zip(labels,q.tolist())),'estimated_target_prior_residual':float(res),'soft_confusion_condition_number':float(np.linalg.cond(C)),'validation_accuracy_raw':float(np.mean(val_raw==yval)),'validation_accuracy_corrected':float(np.mean(val_adj==yval)),'official_test':result,'protocol':'Estimated target class prior using only model probabilities on unlabeled official-test features and source validation conditional response matrix. Official test labels are used only for final metrics.'}
 out=OUT/dn/'bbse';out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(rep,indent=2,default=float))
 print(ds,'raw',result['raw']['accuracy'],'bbse',result['bbse_corrected']['accuracy'],'target priors',q,'cond',np.linalg.cond(C),flush=True)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True);x=a.parse_args();run(x.dataset)
