"""Paired adaptive expert selector: predict which candidate is more likely correct.

All base estimators are trained on the official training fit fold. Router fitting and
threshold selection use a separate training-only validation partition. Official test
labels are used only for the final report. Original app artifacts are not modified.
"""
import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT/'scripts'))
from hybrid_adaptive_evaluation import get_data,predict,metrics,SEED,OUT

def make_pair_features(pa,pf,Z,which):
    pa=np.asarray(pa,dtype=float);pf=np.asarray(pf,dtype=float);Z=np.asarray(Z,dtype=np.float32)
    this=np.where(which[:,None]==1,pf,pa);other=np.where(which[:,None]==1,pa,pf)
    def summary(p):
        s=np.sort(p,axis=1);mx=s[:,-1];sec=s[:,-2];ent=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)
        return np.column_stack([mx,sec,mx-sec,ent])
    return np.column_stack([this,other,this-other,summary(this),summary(other),Z,which.astype(float)])

def score_router(router,pa,pf,Z):
    n=len(pa);w0=np.zeros(n,dtype=int);w1=np.ones(n,dtype=int)
    X0=make_pair_features(pa,pf,Z,w0);X1=make_pair_features(pa,pf,Z,w1)
    pr0=router.predict_proba(X0)[:,list(router.classes_).index(1)]
    pr1=router.predict_proba(X1)[:,list(router.classes_).index(1)]
    return pr1-pr0

def run(ds):
    dn=ds.lower().replace('-','_');base=OUT/dn;b=joblib.load(base/'candidate_bundle.joblib')
    X,y,Xte,yte,*_=get_data(ds);groups=pd.util.hash_pandas_object(X,index=False).to_numpy();grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
    gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy())
    Xval=X.loc[~np.isin(groups,gtr)].reset_index(drop=True);yval=y[~np.isin(groups,gtr)]
    pre=b['preprocessor'];idx=np.asarray(b['fixed_indices'],dtype=int);labels=sorted(set(y)|set(yte))
    Zv=np.asarray(pre.transform(Xval),dtype=np.float32);pfix,cls=predict(b['fixed'],Zv[:,idx],proba=True);pall,allcls=predict(b['all'],Zv,proba=True)
    # Align probability columns exactly by label.
    ap={str(c):i for i,c in enumerate(allcls)};fp={str(c):i for i,c in enumerate(cls)};order=[ap[str(c)] for c in cls];pall=np.asarray(pall)[:,order];pfix=np.asarray(pfix)
    pall_pred=np.asarray(predict(b['all'],Zv),dtype=object);pfix_pred=np.asarray(predict(b['fixed'],Zv[:,idx]),dtype=object)
    correct_all=(pall_pred==yval);correct_fixed=(pfix_pred==yval)
    itrain,igate=train_test_split(np.arange(len(yval)),test_size=.35,random_state=SEED+99,stratify=yval)
    # Two candidate records per validation sample, with indicator flag and both models' scores.
    ntr=len(itrain);which=np.r_[np.zeros(ntr,dtype=int),np.ones(ntr,dtype=int)]
    Xmeta=np.vstack([make_pair_features(pall[itrain],pfix[itrain],Zv[itrain][:,idx],np.zeros(ntr,dtype=int)),make_pair_features(pall[itrain],pfix[itrain],Zv[itrain][:,idx],np.ones(ntr,dtype=int))])
    ymeta=np.r_[correct_all[itrain].astype(int),correct_fixed[itrain].astype(int)]
    candidates=[('ExtraTrees',ExtraTreesClassifier(n_estimators=180,max_depth=8,min_samples_leaf=30,max_features=.9,n_jobs=1,random_state=SEED+110))]
    rows=[];best=None
    for name,m in candidates:
        t=time.perf_counter();m.fit(Xmeta,ymeta);ft=time.perf_counter()-t
        delta=score_router(m,pall[igate],pfix[igate],Zv[igate][:,idx]);base_all=pall_pred[igate];base_fix=pfix_pred[igate]
        thresholds=np.unique(np.quantile(delta,np.linspace(.01,.99,99)))
        opts=[]
        for th in thresholds:
            use_fix=delta>float(th);pred=np.where(use_fix,base_fix,base_all)
            opts.append((float(accuracy_score(yval[igate],pred)),float(th),float(use_fix.mean()),pred))
        cur=max(opts,key=lambda z:(z[0],-abs(z[2]-.5)))
        rows.append({'name':name,'fit_seconds':ft,'gate_accuracy':cur[0],'threshold':cur[1],'fixed_fraction':cur[2]})
        print(name,'gate acc',round(cur[0],6),'fit_s',round(ft,2),flush=True)
        if best is None or cur[0]>best[0]:best=(cur[0],m,cur[1],cur[2],name)
    _,router,th,fraction,rname=best
    # Gate metrics are on held-out router rows only.
    dgate=score_router(router,pall[igate],pfix[igate],Zv[igate][:,idx]);dyn_gate=np.where(dgate>th,pfix_pred[igate],pall_pred[igate])
    gate={'all':metrics(yval[igate],pall_pred[igate],labels),'fixed':metrics(yval[igate],pfix_pred[igate],labels),'dynamic':metrics(yval[igate],dyn_gate,labels)}
    # Official test features only; router uses no test labels.
    Zt=np.asarray(pre.transform(Xte),dtype=np.float32);pta,clta=predict(b['all'],Zt,proba=True);ptf,cltf=predict(b['fixed'],Zt[:,idx],proba=True)
    order=[list(clta).index(c) for c in cltf];pta=np.asarray(pta)[:,order];ptf=np.asarray(ptf)
    at=np.asarray(predict(b['all'],Zt),dtype=object);ft=np.asarray(predict(b['fixed'],Zt[:,idx]),dtype=object)
    delta=score_router(router,pta,ptf,Zt[:,idx]);usefix=delta>th;dyn=np.where(usefix,ft,at)
    results={'all':metrics(yte,at,labels),'fixed':metrics(yte,ft,labels),'dynamic':metrics(yte,dyn,labels)}
    results['all']['feature_count']=float(Zt.shape[1]);results['fixed']['feature_count']=float(len(idx));avg=len(idx)*float(usefix.mean())+Zt.shape[1]*float((~usefix).mean());results['dynamic'].update({'adaptive_mode':'paired correctness router','router_model':rname,'router_threshold':float(th),'fixed_fraction':float(usefix.mean()),'average_features':float(avg),'feature_count':float(avg)})
    report={'dataset':ds,'task':'native multiclass','base_model':b['all_kind'],'model_fit_rows':b['model_fit_rows'],'validation_rows':len(yval),'router_train_validation_rows':len(itrain),'router_gate_rows':len(igate),'router_candidates':rows,'selected_router':rname,'selected_threshold_validation_only':float(th),'gate_metrics':gate,'official_test_rows':len(yte),'official_test_results':results,'protocol':'Candidate models and feature ranking fit on official training-side fit fold. Paired router fit on validation-train rows; threshold selected on separate validation-gate rows. Test features used for inference only; test labels only for final metrics.'}
    out=base/'pairwise_router';out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(report,indent=2,default=float));joblib.dump({'router':router,'threshold':th,'fixed_indices':idx,'feature_names':b['feature_names']},out/'router.joblib',compress=3)
    for k,v in results.items():print(k,'accuracy',round(v['accuracy']*100,4),'macroF1',round(v['macro_f1']*100,4),'features',v.get('feature_count'),flush=True)
if __name__=='__main__':
 import argparse;a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15'],required=True);args=a.parse_args();run(args.dataset)
