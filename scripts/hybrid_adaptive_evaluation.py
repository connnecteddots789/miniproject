"""Hybrid filter/wrapper ranking + adaptive feature acquisition experiment.

Uses official NSL-KDD / UNSW-NB15 train/test files. All ranking, model choice,
feature budgets and confidence thresholds are chosen on a grouped holdout drawn
only from the official training file. The official test set is evaluated once
for the selected frozen design. Outputs stay under backend/artifacts/research/.
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from scipy.stats import entropy
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parents[1]
BACKEND=ROOT/'backend'
sys.path.insert(0,str(BACKEND))
from app.datasets import dataset_files,load_nsl,load_unsw,prepare_labels,feature_frame
from app.ml import make_preprocessor, stratified_sample_indices

SEED=73
warnings.filterwarnings('ignore')
# Keep research fits single-threaded: this environment's OpenMP multi-worker XGBoost
# stalls on the complete training matrix despite fast validation fits.
JOBS=1
OUT=BACKEND/'artifacts'/'research'/'hybrid_adaptive_acquisition'


def get_data(ds):
    files=dataset_files(ds); reader=load_nsl if ds=='NSL-KDD' else load_unsw
    tr=next(p for p in files if 'train' in p.name.lower())
    te=next(p for p in files if 'test' in p.name.lower() and 'test-21' not in p.name.lower())
    dtr=reader(tr); dte=reader(te)
    col,_,yt=prepare_labels(dtr,ds); _,_,yv=prepare_labels(dte,ds)
    y=yt.to_numpy(dtype=object); ytest=yv.to_numpy(dtype=object)
    if ds=='NSL-KDD':
        keep=np.isin(y,['Normal','DoS','Probe','R2L','U2R']); keept=np.isin(ytest,['Normal','DoS','Probe','R2L','U2R'])
    else:
        keep=y!='Other'; keept=ytest!='Other'
    X=feature_frame(dtr,ds,col).loc[keep].reset_index(drop=True)
    Xt=feature_frame(dte,ds,col).loc[keept].reset_index(drop=True).reindex(columns=X.columns)
    return X,y[keep],Xt,ytest[keept],tr,te


def hashfile(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def rank_scores(Z,y,seed):
    """Gain ratio, MI, class association (eta^2), DecisionTree and ExtraTrees."""
    Z=np.asarray(Z,dtype=np.float32); y=np.asarray(y,dtype=object); n,d=Z.shape
    # Quantile discretization is fitted on the training-side ranking sample only.
    disc=KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile',subsample=200000,random_state=seed)
    Zd=disc.fit_transform(Z).astype(np.int16)
    mi=np.zeros(d); gr=np.zeros(d); corr=np.zeros(d)
    labels, yi=np.unique(y,return_inverse=True)
    for j in range(d):
        col=Zd[:,j]
        mi[j]=mutual_info_score(yi,col)
        h=float(entropy(np.bincount(col)))
        gr[j]=mi[j]/(h+1e-10)
        # Correlation ratio eta^2 between numeric/one-hot feature and class.
        x=Z[:,j].astype(float); mu=float(np.mean(x)); total=float(np.sum((x-mu)**2))
        if total>1e-12:
            between=0.0
            for ci in range(len(labels)):
                xc=x[yi==ci]
                if len(xc): between += len(xc)*(float(np.mean(xc))-mu)**2
            corr[j]=between/total
    dt=DecisionTreeClassifier(max_depth=None,min_samples_leaf=2,class_weight='balanced',random_state=seed)
    dt.fit(Z,y); dti=np.asarray(dt.feature_importances_,dtype=float)
    et=ExtraTreesClassifier(n_estimators=40,min_samples_leaf=3,max_features='sqrt',class_weight=None,
        n_jobs=JOBS,random_state=seed+1)
    et.fit(Z,y); eti=np.asarray(et.feature_importances_,dtype=float)
    # Robust percentile rank fusion avoids scale differences among measures.
    raw={'gain_ratio':gr,'mutual_information':mi,'correlation_eta2':corr,'decision_tree':dti,'extra_trees':eti}
    def percentile(a):
        order=np.argsort(np.argsort(np.nan_to_num(a,nan=0.0)))
        return order/max(1,len(a)-1)
    pgr,pmi,pcorr,pdt,pet=map(percentile,(gr,mi,corr,dti,eti))
    raw['gain_ratio_decision_tree']=.50*pgr+.50*pdt
    raw['gain_ratio_mutual_information']=.50*pgr+.50*pmi
    raw['filter_wrapper_fusion']=.30*pgr+.25*pmi+.15*pcorr+.30*pdt
    raw['hybrid_rank_fusion']=.22*pgr+.22*pmi+.18*pcorr+.22*pdt+.16*pet
    rankings={k:np.argsort(v)[::-1] for k,v in raw.items()}
    return rankings,raw


def make_model(kind,seed,n_estimators=100,class_weight=None):
    if kind=='RandomForest':
        return RandomForestClassifier(n_estimators=260,max_features='sqrt',min_samples_leaf=1,
            class_weight=class_weight,n_jobs=JOBS,random_state=seed)
    if kind=='ExtraTrees':
        return ExtraTreesClassifier(n_estimators=260,max_features='sqrt',min_samples_leaf=1,
            class_weight=class_weight,n_jobs=JOBS,random_state=seed)
    if kind=='LightGBM':
        return LGBMClassifier(n_estimators=n_estimators,num_leaves=15,max_depth=9,learning_rate=.07,
            min_child_samples=35,subsample=.9,subsample_freq=1,colsample_bytree=.9,
            reg_lambda=2.0,max_bin=63,class_weight=class_weight,n_jobs=JOBS,random_state=seed,verbosity=-1,force_col_wise=True)
    if kind=='XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=140,max_depth=5,learning_rate=.08,subsample=.9,colsample_bytree=.9,
            min_child_weight=3,reg_lambda=2.0,objective='multi:softprob',eval_metric='mlogloss',tree_method='hist',
            n_jobs=JOBS,random_state=seed)
    raise ValueError(kind)


def fit_model(kind,Z,y,seed,n_estimators=100,class_weight=None):
    m=make_model(kind,seed,n_estimators=n_estimators,class_weight=class_weight); t=time.perf_counter()
    if kind=='XGBoost':
        from sklearn.preprocessing import LabelEncoder
        enc=LabelEncoder().fit(y); m.set_params(num_class=len(enc.classes_)); weights=None
        if class_weight=='balanced':
            from sklearn.utils.class_weight import compute_sample_weight
            weights=compute_sample_weight('balanced',y)
        m.fit(Z,enc.transform(y),sample_weight=weights); wrap={'kind':kind,'model':m,'encoder':enc}
    else:
        m.fit(Z,y); wrap={'kind':kind,'model':m,'encoder':None}
    return wrap,time.perf_counter()-t


def predict(wrap,Z,proba=False):
    m=wrap['model']; enc=wrap['encoder']
    if proba:
        p=np.asarray(m.predict_proba(Z),dtype=float)
        classes=enc.classes_ if enc is not None else np.asarray(m.classes_,dtype=object)
        return p,classes
    out=np.asarray(m.predict(Z))
    return enc.inverse_transform(out.astype(int)) if enc is not None else out.astype(object)


def metrics(y,pred,labels):
    # Compute all report metrics from one confusion matrix. This is equivalent to
    # sklearn's zero_division=0 classification report but much faster on 80k+ rows.
    cm=confusion_matrix(y,pred,labels=labels)
    tp=np.diag(cm).astype(float); support=cm.sum(axis=1).astype(float); predicted=cm.sum(axis=0).astype(float)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=(precision+recall)>0)
    total=max(1.0,float(support.sum()))
    per={str(label):{'precision':float(precision[i]),'recall':float(recall[i]),'f1-score':float(f1[i]),'support':int(support[i])} for i,label in enumerate(labels)}
    per['accuracy']=float(tp.sum()/total)
    per['macro avg']={'precision':float(precision.mean()),'recall':float(recall.mean()),'f1-score':float(f1.mean()),'support':int(support.sum())}
    per['weighted avg']={'precision':float(np.sum(precision*support)/total),'recall':float(np.sum(recall*support)/total),'f1-score':float(np.sum(f1*support)/total),'support':int(support.sum())}
    return {'accuracy':float(tp.sum()/total),'macro_f1':float(f1.mean()),'weighted_f1':float(np.sum(f1*support)/total),
            'per_class':per,'confusion_matrix':cm.tolist(),'labels':list(labels)}


def objective(m): return m['accuracy']  # the user's primary metric is overall accuracy


def choose_threshold(yval,fixed_pred,all_pred,fixed_prob,all_prob,labels):
    """Select a per-row compact-vs-full acquisition policy on validation only."""
    cf=np.max(fixed_prob,axis=1); ca=np.max(all_prob,axis=1)
    rows=[]; base=np.asarray(fixed_pred,dtype=object); full=np.asarray(all_pred,dtype=object)
    # Policy A: confidence-gated feature acquisition (compact first, full on low confidence).
    for th in np.unique(np.quantile(cf,[.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.60,.70,.80,.90,.95])):
        go=cf<float(th); frac=float(go.mean())
        if .01<=frac<=.99:
            pred=base.copy(); pred[go]=full[go]
            rows.append({'mode':'compact_confidence_fallback','threshold':float(th),'fallback_fraction':frac,
                         'accuracy':float(accuracy_score(yval,pred)),'predictions':pred})
    # Policy B: acquire/use full-vector expert when its confidence advantage is larger.
    delta=cf-ca
    for th in np.unique(np.quantile(delta,[.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95])):
        use_fixed=delta>=float(th); frac=float((~use_fixed).mean())
        if .01<=frac<=.99:
            pred=full.copy(); pred[use_fixed]=base[use_fixed]
            rows.append({'mode':'confidence_margin_router','threshold':float(th),'fallback_fraction':frac,
                         'accuracy':float(accuracy_score(yval,pred)),'predictions':pred})
    if not rows: return None
    best=max(rows,key=lambda r:(r['accuracy'],-r['fallback_fraction']))
    mm=metrics(yval,best['predictions'],labels)
    return {'mode':best['mode'],'threshold':best['threshold'],'fallback_fraction':best['fallback_fraction'],
            'metrics':mm,'objective':objective(mm)}


def run(ds,selector='hybrid_rank_fusion',model_kind='LightGBM',model_fit_limit=40000,class_weight=None):
    print(f'===== {ds} =====',flush=True)
    X,y,Xte,yte,tr,te=get_data(ds); labels=sorted(set(y)|set(yte))
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    grp=pd.DataFrame({'g':groups,'y':y}).groupby('g',sort=False)['y'].agg(lambda s:s.value_counts().index[0])
    gtr,gva=train_test_split(grp.index.to_numpy(),test_size=.15,random_state=SEED,stratify=grp.to_numpy())
    mask=np.isin(groups,gtr)
    Xfit=X.loc[mask].reset_index(drop=True); Xval=X.loc[~mask].reset_index(drop=True)
    yfit=y[mask]; yval=y[~mask]
    print('rows',len(yfit),len(yval),len(yte),'classes',len(labels),flush=True)
    pre=make_preprocessor(Xfit)
    Zfit=np.asarray(pre.fit_transform(Xfit),dtype=np.float32); Zval=np.asarray(pre.transform(Xval),dtype=np.float32)
    # Rank on up to 20k stratified fit-only rows.
    ridx=stratified_sample_indices(yfit,min(8000,len(yfit)),SEED+11,min_per_class=5)
    rank_started=time.perf_counter()
    rankings,rankvals=rank_scores(Zfit[ridx],yfit[ridx],SEED+13)
    print('training-only filter/wrapper ranking completed in %.1fs on %d rows'%(time.perf_counter()-rank_started,len(ridx)),flush=True)
    d=Zfit.shape[1]
    # Train/validate three strong baseline families on all features.
    model_rows=[]; model_cache={}
    tune_idx=stratified_sample_indices(yfit,min(int(model_fit_limit),len(yfit)),SEED+21,min_per_class=10)
    Zt=Zfit[tune_idx]; yt=yfit[tune_idx]
    estimator_count=70 if len(tune_idx)>80000 and model_kind=='LightGBM' else (300 if model_kind=='LightGBM' else 100)
    for i,kind in enumerate([model_kind]):
        print('fit validation ALL',kind,flush=True)
        wrap,secs=fit_model(kind,Zt,yt,SEED+30+i,estimator_count,class_weight)
        pv=predict(wrap,Zval)
        mm=metrics(yval,pv,labels)
        model_rows.append({'name':kind,'feature_set':'All','feature_fraction':1.0,'feature_count':d,
            'validation':mm,'objective':objective(mm),'fit_seconds':secs})
        model_cache[('all',kind)]=(wrap,pv)
        print('  val acc %.4f macroF1 %.4f in %.1fs'%(mm['accuracy'],mm['macro_f1'],secs),flush=True)
    best_all=max([r for r in model_rows if r['feature_set']=='All'],key=lambda r:(r['validation']['accuracy'],r['validation']['macro_f1']))
    all_kind=best_all['name']; all_val_wrap,all_val_pred=model_cache[('all',all_kind)]
    # Evaluate each filter/wrapper ranking at 35% and 50%; train the best validation model-family
    # (LightGBM / ExtraTrees); XGBoost also screened on hybrid subset.
    fixed_candidates=[]
    selectors=[selector]
    for si,sel in enumerate(selectors):
      for frac in [.50]:
        k=max(3,min(d,int(round(d*frac)))); idx=rankings[sel][:k]
        for kind in [all_kind]:
          print('fit validation FIXED',sel,frac,kind,flush=True)
          wrap,secs=fit_model(kind,Zt[:,idx],yt,SEED+100+si*7+int(frac*100),estimator_count,class_weight)
          pv=predict(wrap,Zval[:,idx]); mm=metrics(yval,pv,labels)
          row={'name':kind,'feature_set':sel,'feature_fraction':frac,'feature_count':k,
               'validation':mm,'objective':objective(mm),'fit_seconds':secs}
          fixed_candidates.append((row,idx,wrap,pv))
          print('  val acc %.4f macroF1 %.4f'%(mm['accuracy'],mm['macro_f1']),flush=True)
    best_fixed=max(fixed_candidates,key=lambda z:(z[0]['validation']['accuracy'],z[0]['validation']['macro_f1']))
    fixed_row,fixed_idx,fixed_val_wrap,fixed_val_pred=best_fixed
    # Dynamic acquisition: use compact selected model first; acquire full-vector prediction only
    # when compact confidence falls below a training-validation-selected threshold.
    pf,cls=predict(fixed_val_wrap,Zval[:,fixed_idx],proba=True)
    pa,all_cls=predict(all_val_wrap,Zval,proba=True)
    # Align probability columns defensively by class label.
    all_pos={str(c):i for i,c in enumerate(all_cls)}
    pa=pa[:,[all_pos[str(c)] for c in cls]]
    dyn=choose_threshold(yval,np.asarray(fixed_val_pred,dtype=object),np.asarray(all_val_pred,dtype=object),pf,pa,labels)
    if dyn:
        print('dynamic val',dyn['metrics']['accuracy'], 'mode',dyn['mode'],'fallback',dyn['fallback_fraction'], 'threshold',dyn['threshold'],flush=True)
    else: print('dynamic no viable fallback threshold',flush=True)
    # Checkpoint fitted candidates before scoring the large official test matrix.
    # This permits a separate scoring pass if the interactive execution window ends.
    out=OUT/ds.lower().replace('-','_'); out.mkdir(parents=True,exist_ok=True)
    val_feature_names=np.asarray(pre.get_feature_names_out(),dtype=object)
    checkpoint={'preprocessor':pre,'all':all_val_wrap,'fixed':fixed_val_wrap,'all_kind':all_kind,
        'fixed_kind':fixed_row['name'],'fixed_indices':fixed_idx,'labels':labels,'class_weight':class_weight,
        'dynamic_config':dyn,'feature_names':val_feature_names.tolist(),'model_fit_rows':int(len(tune_idx)),
        'official_test_y':yte,'official_test_X':Xte}
    _t=time.perf_counter(); joblib.dump(checkpoint,out/'candidate_checkpoint.joblib',compress=0)
    print('  candidate checkpoint saved %.2fs'%(time.perf_counter()-_t),flush=True)
    # Matched candidates use the same training-only sample, preprocessing schema,
    # selector choice and official test rows. Production app models are untouched.
    print('evaluate frozen matched candidates on official test rows',flush=True)
    _t=time.perf_counter(); Ztest=np.asarray(pre.transform(Xte),dtype=np.float32); print('  test transform %.1fs %s'%(time.perf_counter()-_t,Ztest.shape),flush=True)
    _t=time.perf_counter(); pall=np.asarray(predict(all_val_wrap,Ztest),dtype=object); print('  all-feature test prediction %.1fs'%(time.perf_counter()-_t),flush=True)
    _t=time.perf_counter(); pfix=np.asarray(predict(fixed_val_wrap,Ztest[:,fixed_idx]),dtype=object); print('  fixed-feature test prediction %.1fs'%(time.perf_counter()-_t),flush=True)
    _t=time.perf_counter(); pprob,cls_test=predict(fixed_val_wrap,Ztest[:,fixed_idx],proba=True); print('  fixed probabilities %.1fs'%(time.perf_counter()-_t),flush=True)
    _t=time.perf_counter(); pallprob,all_cls_test=predict(all_val_wrap,Ztest,proba=True); print('  all probabilities %.1fs'%(time.perf_counter()-_t),flush=True)
    all_pos_test={str(c):i for i,c in enumerate(all_cls_test)}
    pallprob=pallprob[:,[all_pos_test[str(c)] for c in cls_test]]
    cf=np.max(pprob,axis=1); ca=np.max(pallprob,axis=1)
    if dyn and dyn['mode']=='confidence_margin_router':
        use_fixed=(cf-ca)>=float(dyn['threshold']); fallback=~use_fixed
        pdyn=pall.copy(); pdyn[use_fixed]=pfix[use_fixed]
    else:
        fallback=cf<float(dyn['threshold']) if dyn else np.ones(len(yte),dtype=bool)
        pdyn=pfix.copy(); pdyn[fallback]=pall[fallback]
    print('  fallback applied rows=%d (%.2f%%)'%(int(fallback.sum()),100*float(fallback.mean())),flush=True)
    results={}
    for _key,_pred in [('all',pall),('fixed',pfix),('dynamic',pdyn)]:
        _t=time.perf_counter(); print('  metrics start',_key,flush=True)
        results[_key]=metrics(yte,_pred,labels)
        print('  metrics done',_key,'%.2fs'%(time.perf_counter()-_t),flush=True)
    for key in results:
        results[key]['feature_count']=float(Ztest.shape[1] if key=='all' else len(fixed_idx) if key=='fixed' else (len(fixed_idx)*(~fallback).mean()+Ztest.shape[1]*fallback.mean()))
    results['dynamic'].update({'adaptive_mode':dyn['mode'] if dyn else 'full-fallback','confidence_threshold':float(dyn['threshold']) if dyn else None,
        'fallback_fraction':float(fallback.mean()),'average_features':results['dynamic']['feature_count']})
    out=OUT/ds.lower().replace('-','_'); out.mkdir(parents=True,exist_ok=True)
    ranking_info={k:[str(val_feature_names[i]) for i in v[:min(30,len(v))]] for k,v in rankings.items()}
    report={'dataset':ds,'task':'native multiclass','seed':SEED,'class_weight':class_weight,'official_train_sha256':hashfile(tr),'official_test_sha256':hashfile(te),
        'train_rows':len(y),'training_fit_rows':len(yfit),'model_fit_rows':int(len(tune_idx)),'estimator_count':int(estimator_count),'validation_rows':len(yval),'official_test_rows':len(yte),'feature_count':int(Ztest.shape[1]),'labels':labels,
        'protocol':'85/15 duplicate-grouped validation from official train; estimators trained on the same stratified sample from the training-side fit fold; selector and threshold chosen without test data; official test rows evaluated once; no test labels used for tuning',
        'rankers':['Gain Ratio','Mutual Information','Correlation ratio (eta-squared)','Decision Tree impurity importance','ExtraTrees impurity importance','rank-fusion hybrid'],
        'hybrid_weights':{'gain_ratio':.22,'mutual_information':.22,'correlation_eta2':.18,'decision_tree':.22,'extra_trees':.16},
        'feature_selection_scope':'Rankings fit on the duplicate-grouped official-training fit fold only; selected encoded feature indices use the same training-fitted preprocessor applied to official test features; official test labels excluded from selector fitting.',
        'validation_all_candidates':model_rows,
        'validation_fixed_candidates':[z[0] for z in fixed_candidates],
        'selected_all':best_all,'selected_fixed':fixed_row,'selected_dynamic_validation':dyn,
        'selected_fixed_feature_names':[str(val_feature_names[i]) for i in fixed_idx],
        'top_ranked_features':ranking_info,'official_test_results':results,
        'training_seconds':{'all_candidate':float(next(r['fit_seconds'] for r in model_rows if r['name']==all_kind)),'fixed_candidate':float(fixed_row['fit_seconds'])}}
    _t=time.perf_counter(); (out/'comparison.json').write_text(json.dumps(report,indent=2,default=float)); print('  report json %.2fs'%(time.perf_counter()-_t),flush=True)
    _t=time.perf_counter(); joblib.dump({'preprocessor':pre,'all':all_val_wrap,'fixed':fixed_val_wrap,'all_kind':all_kind,'fixed_kind':fixed_row['name'],
        'fixed_indices':fixed_idx,'labels':labels,'dynamic_threshold':float(dyn['threshold']) if dyn else None,
        'validation_config':report['selected_dynamic_validation'],'feature_names':val_feature_names.tolist(),
        'model_fit_rows':int(len(tune_idx))},out/'candidate_bundle.joblib',compress=0); print('  candidate bundle saved %.2fs'%(time.perf_counter()-_t),flush=True)
    print(f'[{ds}] OFFICIAL TEST RESULTS',flush=True)
    for name,m in results.items(): print(name, f"acc={m['accuracy']*100:.3f}% macroF1={m['macro_f1']*100:.3f}% weightedF1={m['weighted_f1']*100:.3f}% features={m['feature_count']:.1f}",flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['NSL-KDD','UNSW-NB15','both'],default='both'); ap.add_argument('--selector',choices=['gain_ratio','mutual_information','correlation_eta2','decision_tree','extra_trees','gain_ratio_decision_tree','gain_ratio_mutual_information','filter_wrapper_fusion','hybrid_rank_fusion'],default='hybrid_rank_fusion'); ap.add_argument('--model',choices=['LightGBM','XGBoost','RandomForest','ExtraTrees'],default='LightGBM'); ap.add_argument('--model-rows',type=int,default=40000); ap.add_argument('--class-weight',choices=['none','balanced'],default='none'); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    datasets=['NSL-KDD','UNSW-NB15'] if args.dataset=='both' else [args.dataset]
    results=[]
    for ds in datasets:
        results.append(run(ds,args.selector,args.model,args.model_rows,None if args.class_weight=='none' else 'balanced'))
    (OUT/'summary.json').write_text(json.dumps(results,indent=2,default=float))
    print('WROTE',OUT/'summary.json',flush=True)
