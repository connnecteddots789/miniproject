"""Unlabeled covariate-shift weighting + dynamic expert routing research pass.

Uses official train labels only. Official test features (without labels) are used
for a separate transductive domain-classifier to estimate density ratios. Test
labels are used only for final metrics. This protocol is explicitly labeled
transductive and is not mixed with the strict inductive benchmark.
"""
import sys, json, time, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'backend'))
from hybrid_adaptive_evaluation import get_data, metrics, SEED, OUT
from app.ml import make_preprocessor
warnings.filterwarnings('ignore')

def run(ds):
    X,y,Xte,yte,*_=get_data(ds); labels=sorted(set(y)|set(yte)); droot=OUT/ds.lower().replace('-','_')/'domain_weighting';droot.mkdir(parents=True,exist_ok=True)
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy(); grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
    gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy());mask=np.isin(groups,gtr)
    Xfit=X.loc[mask].reset_index(drop=True);Xval=X.loc[~mask].reset_index(drop=True);yfit=y[mask];yval=y[~mask]
    pre=make_preprocessor(Xfit);Zfit=np.asarray(pre.fit_transform(Xfit),dtype=np.float32);Zval=np.asarray(pre.transform(Xval),dtype=np.float32);Zte=np.asarray(pre.transform(Xte),dtype=np.float32)
    # Reuse the already training-only hybrid ranking frozen in the candidate checkpoint.
    prior=joblib.load(OUT/ds.lower().replace('-','_')/'candidate_checkpoint.joblib'); idx=np.asarray(prior['fixed_indices'],dtype=int)
    # Domain classifier: equal-sized source / target samples; test labels are never read.
    rng=np.random.default_rng(SEED+73); n=min(len(Zfit),len(Zte),50000); isrc=rng.choice(len(Zfit),n,replace=False); itgt=rng.choice(len(Zte),n,replace=False)
    Zd=np.vstack([Zfit[isrc],Zte[itgt]]); yd=np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]
    dc=LGBMClassifier(n_estimators=130,num_leaves=15,max_depth=7,learning_rate=.06,min_child_samples=50,subsample=.9,subsample_freq=1,colsample_bytree=.85,n_jobs=1,random_state=SEED+74,verbosity=-1,force_col_wise=True)
    t=time.perf_counter();dc.fit(Zd,yd);dc_sec=time.perf_counter()-t
    pdom=dc.predict_proba(Zfit)[:,1]; ratio=pdom/np.clip(1-pdom,1e-4,None); weights=np.clip(ratio,.2,5.0);weights/=weights.mean()
    # Keep the source-side training weights stable; record diagnostics.
    def model(): return LGBMClassifier(n_estimators=220,num_leaves=15,max_depth=9,learning_rate=.055,min_child_samples=35,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=3.0,max_bin=63,n_jobs=1,random_state=SEED+80,verbosity=-1,force_col_wise=True)
    base=model();t=time.perf_counter();base.fit(Zfit,yfit);base_sec=time.perf_counter()-t
    compact=model();t=time.perf_counter();compact.fit(Zfit[:,idx],yfit,sample_weight=weights);adapt_sec=time.perf_counter()-t
    bv=base.predict(Zval);cv=compact.predict(Zval[:,idx]);
    # Learn per-row choice between compact shift-weighted and full baseline on a
    # distinct router-train / router-gate split of the official-training validation fold.
    pvb=base.predict_proba(Zval);pvc=compact.predict_proba(Zval[:,idx]);
    train_i,gate_i=train_test_split(np.arange(len(yval)),test_size=.35,random_state=SEED+81,stratify=yval)
    def feats(a,b,z):
        def sm(p):
            s=np.sort(p,axis=1);mx=s[:,-1];sec=s[:,-2];ent=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1);return np.column_stack([mx,sec,mx-sec,ent])
        return np.column_stack([a,b,a-b,sm(a),sm(b),z])
    meta=feats(pvc,pvb,Zval[:,idx]); c_correct=(cv==yval).astype(int); b_correct=(bv==yval).astype(int)
    target=(c_correct>b_correct).astype(int)
    router=ExtraTreesClassifier(n_estimators=220,max_depth=9,min_samples_leaf=35,max_features=.85,n_jobs=1,class_weight='balanced',random_state=SEED+82)
    router.fit(meta[train_i],target[train_i]); pr=router.predict_proba(meta[gate_i])[:,list(router.classes_).index(1)]
    best=None
    for th in np.unique(np.quantile(pr,np.linspace(.01,.99,99))):
        use=pr>=th;pred=np.where(use,cv[gate_i],bv[gate_i]);acc=float(accuracy_score(yval[gate_i],pred))
        if best is None or acc>best[0]:best=(acc,float(th),float(use.mean()))
    th=best[1]
    # Final evaluation, test labels only passed to metrics at this line.
    bt=base.predict(Zte);ct=compact.predict(Zte[:,idx]);pbt=base.predict_proba(Zte);pct=compact.predict_proba(Zte[:,idx]);metat=feats(pct,pbt,Zte[:,idx]);prt=router.predict_proba(metat)[:,list(router.classes_).index(1)];use=prt>=th;dt=np.where(use,ct,bt)
    result={'all':metrics(yte,bt,labels),'fixed_domain_weighted':metrics(yte,ct,labels),'dynamic':metrics(yte,dt,labels)}
    result['all']['feature_count']=float(Zte.shape[1]);result['fixed_domain_weighted']['feature_count']=float(len(idx));avg=len(idx)*float(use.mean())+Zte.shape[1]*float((~use).mean());result['dynamic'].update({'adaptive_mode':'domain-weighted compact expert + learned router','router_threshold_validation_only':th,'compact_fraction':float(use.mean()),'average_features':float(avg),'feature_count':float(avg)})
    report={'dataset':ds,'task':'native multiclass','protocol':'TRANSductive unsupervised covariate-shift weighting: official test FEATURES (labels never used) inform a source-vs-target domain classifier and density-ratio weights. Label ranking, model fitting and router threshold use training-only grouped validation. Final test labels are used only for final metrics. Report separately from strict inductive benchmark.','train_rows':len(y),'fit_rows':len(yfit),'validation_rows':len(yval),'test_rows':len(yte),'feature_count':int(Zfit.shape[1]),'selected_compact_count':int(len(idx)),'domain_classifier_seconds':dc_sec,'base_fit_seconds':base_sec,'adapted_fit_seconds':adapt_sec,'domain_weight_quantiles':{str(q):float(np.quantile(weights,q)) for q in [0,.1,.5,.9,.99,1]},'domain_prob_quantiles':{str(q):float(np.quantile(pdom,q)) for q in [0,.1,.5,.9,.99,1]},'router_gate_accuracy':best[0],'router_compact_fraction_gate':best[2],'official_test_rows':len(yte),'official_test_results':result}
    (droot/'results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'preprocessor':pre,'domain_classifier':dc,'base_model':base,'compact_model':compact,'router':router,'router_threshold':th,'selected_indices':idx,'feature_names':pre.get_feature_names_out().tolist()},droot/'models.joblib',compress=3)
    print(ds,'domain fit',round(dc_sec,1),'base',round(base_sec,1),'adapt',round(adapt_sec,1),'gate',round(best[0]*100,3),'test',[(k,round(v['accuracy']*100,3)) for k,v in result.items()],'avgf',round(avg,1),flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True);run(p.parse_args().dataset)
