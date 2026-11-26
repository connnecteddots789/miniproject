"""Validation-selected adaptive feature acquisition experiments for SENTINEL.

Official outer train/test files are respected. Model choice, feature rankings,
feature budgets and the cascade confidence threshold are selected only on a
stratified validation split drawn from the official training file. The official
test file is opened for metrics only after final models are refit on all training
rows. No test labels participate in model/feature selection.

Runs four separately reported tasks:
  * NSL-KDD native five-family
  * UNSW-NB15 native ten-category
  * NSL-KDD binary Normal-vs-Attack (supplementary)
  * UNSW-NB15 binary Normal-vs-Attack (supplementary)
"""
from __future__ import annotations
import json, time, sys, warnings, hashlib, subprocess, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, mutual_info_score)
from sklearn.preprocessing import KBinsDiscretizer
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'backend'
sys.path.insert(0, str(BACKEND))
from app.datasets import dataset_files, load_nsl, load_unsw, prepare_labels, feature_frame
from app.ml import make_preprocessor, stratified_sample_indices

SEED = 42
N_JOBS = 1
OUT = BACKEND / 'artifacts' / 'research' / 'adaptive_feature_selection'


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def get_frames(dataset: str, task: str):
    files=dataset_files(dataset)
    reader=load_nsl if dataset=='NSL-KDD' else load_unsw
    train_path=next((p for p in files if 'train' in p.name.lower()),None)
    test_path=next((p for p in files if 'test' in p.name.lower() and 'test-21' not in p.name.lower()),None)
    if not train_path or not test_path: raise FileNotFoundError(f'Official train/test files not found for {dataset}')
    train_df=reader(train_path); test_df=reader(test_path)
    label_col,raw,y_native=prepare_labels(train_df,dataset)
    _,_,yt_native=prepare_labels(test_df,dataset)
    if task=='native':
        y=y_native.to_numpy(dtype=object); yt=yt_native.to_numpy(dtype=object)
        keep=np.isin(y,['Normal','DoS','Probe','R2L','U2R']) if dataset=='NSL-KDD' else y!='Other'
        keept=np.isin(yt,['Normal','DoS','Probe','R2L','U2R']) if dataset=='NSL-KDD' else yt!='Other'
        X=feature_frame(train_df,dataset,label_col).loc[keep].reset_index(drop=True)
        Xt=feature_frame(test_df,dataset,label_col).loc[keept].reset_index(drop=True)
        y=y[keep]; yt=yt[keept]
        task_name='native five-family' if dataset=='NSL-KDD' else 'native ten-category'
    elif task=='binary':
        if dataset=='UNSW-NB15' and 'label' in train_df and 'label' in test_df:
            y=np.where(pd.to_numeric(train_df['label'],errors='coerce').fillna(0).to_numpy()==0,'Normal','Attack')
            yt=np.where(pd.to_numeric(test_df['label'],errors='coerce').fillna(0).to_numpy()==0,'Normal','Attack')
        else:
            y=np.where(y_native.to_numpy(dtype=object)=='Normal','Normal','Attack')
            yt=np.where(yt_native.to_numpy(dtype=object)=='Normal','Normal','Attack')
        # feature_frame excludes both the multiclass and binary targets.
        X=feature_frame(train_df,dataset,label_col).reset_index(drop=True)
        Xt=feature_frame(test_df,dataset,label_col).reset_index(drop=True)
        task_name='binary Normal-vs-Attack'
    else: raise ValueError(task)
    Xt=Xt.reindex(columns=X.columns)
    return X, y, Xt, yt, train_path, test_path, task_name


def lgbm_params(class_weight, variant=0, seed=SEED, binary=False):
    if variant==0:
        cfg=dict(n_estimators=60,num_leaves=15,learning_rate=.06,min_child_samples=30,
                 subsample=.9,colsample_bytree=.9,reg_lambda=1.0)
    else:
        cfg=dict(n_estimators=80,num_leaves=31,learning_rate=.045,min_child_samples=35,
                 subsample=.9,colsample_bytree=.85,reg_lambda=2.0)
    cfg.update(max_depth=-1,class_weight=class_weight,n_jobs=N_JOBS,random_state=seed,verbosity=-1,force_col_wise=True,max_bin=63)
    return cfg


def train_lgb(X, y, class_weight, variant=0, seed=SEED):
    cfg=lgbm_params(class_weight,variant,seed,binary=len(np.unique(y))==2)
    print('   LGBM fit start',np.shape(X),len(y),cfg['n_estimators'],cfg['num_leaves'],type(y).__name__,flush=True)
    m=LGBMClassifier(**cfg)
    ts=time.perf_counter(); m.fit(X,y); print('   LGBM fit done',round(time.perf_counter()-ts,2),flush=True)
    return m


def metric_pack(y, pred, labels, elapsed=0.0, feature_count=0.0):
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=labels,average='macro',zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y,pred,labels=labels,average='weighted',zero_division=0)
    cm=confusion_matrix(y,pred,labels=labels)
    fp_rates=[]
    for i in range(len(labels)):
        fp=cm[:,i].sum()-cm[i,i]
        tn=cm.sum()-cm[i,:].sum()-cm[:,i].sum()+cm[i,i]
        fp_rates.append(float(fp/(fp+tn)) if fp+tn else 0.)
    report=classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(p),'macro_recall':float(r),
        'macro_f1':float(f),'weighted_precision':float(pw),'weighted_recall':float(rw),
        'weighted_f1':float(fw),'macro_false_positive_rate':float(np.mean(fp_rates)),
        'inference_seconds':float(elapsed),'inference_ms_per_record':float(1000*elapsed/max(1,len(y))),
        'inference_records':int(len(y)),'feature_count':float(feature_count),
        'per_class':report,'confusion_matrix':cm.tolist(),'labels':list(labels)}


def simple_metrics(y, pred, labels):
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=labels,average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f)}

def score_objective(m):
    # Explicitly balance overall accuracy with macro-F1 so rare attack classes
    # cannot be ignored merely to inflate accuracy.
    return .55*m['accuracy'] + .45*m['macro_f1']


def binned_information_scores(X, y, bins=10):
    """Fast deterministic univariate MI after training-fold quantile binning."""
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=object)
    n,d=X.shape
    try:
        disc=KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy='quantile',subsample=200000,random_state=SEED)
        Xd=disc.fit_transform(X).astype(np.int16)
    except Exception:
        Xd=np.zeros((n,d),dtype=np.int16)
        for j in range(d):
            vals=X[:,j]
            if np.unique(vals).size>1:
                Xd[:,j]=pd.qcut(vals,q=min(bins,np.unique(vals).size),labels=False,duplicates='drop').astype(int)
    scores=np.zeros(d,dtype=float)
    for j in range(d):
        scores[j]=mutual_info_score(y,Xd[:,j])
    return scores,Xd

def feature_rankings(X, y, seed=SEED):
    """Return feature indices ordered by training-only gain ratio, MI, and hybrid."""
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=object)
    n,d=X.shape
    # Expensive continuous-feature MI is computed on a deterministic,
    # stratified training-only sample; never touch the validation/test labels.
    if n > 12000:
        take=stratified_sample_indices(y,8000,seed,min_per_class=2)
        X=X[take]; y=y[take]
    # MI handles mixed data after numeric/one-hot training-only preprocessing.
    mi,Xd=binned_information_scores(X,y,bins=10)
    # Gain ratio: mutual information between target and quantile-discretized
    # feature, divided by the feature's split entropy. Bin edges are fit only
    # on this training fold.
    gy=[]; ent=[]
    for j in range(d):
        col=np.asarray(Xd[:,j],dtype=int)
        gy.append(mutual_info_score(y,col))
        ent.append(float(entropy(np.bincount(col))))
    gain_ratio=np.asarray(gy)/(np.asarray(ent)+1e-9)
    # LightGBM gain ranking is fitted using only these training rows.
    selector=LGBMClassifier(n_estimators=180,num_leaves=31,learning_rate=.06,
        class_weight='balanced',n_jobs=N_JOBS,random_state=seed,verbosity=-1)
    selector.fit(X,y)
    gain=selector.booster_.feature_importance(importance_type='gain').astype(float)
    def norm(a):
        a=np.nan_to_num(np.asarray(a,dtype=float),nan=0.,posinf=0.,neginf=0.)
        return a/(a.max()+1e-12)
    hybrid=.35*norm(gain)+.35*norm(mi)+.30*norm(gain_ratio)
    return {
        'Gain ratio':np.argsort(gain_ratio)[::-1],
        'Mutual information':np.argsort(mi)[::-1],
        'LightGBM gain':np.argsort(gain)[::-1],
        'Hybrid (gain ratio + MI + LightGBM)':np.argsort(hybrid)[::-1],
    }, {'mutual_information':mi,'gain_ratio':gain_ratio,'lightgbm_gain':gain,'hybrid':hybrid}


def choose_threshold(conf, y, stage1_pred, fallback_pred, labels):
    # Candidate confidence cutoffs are chosen only on the validation split.
    qs=np.unique(np.quantile(conf,[0,.25,.5,.7,.85,.95,1]))
    rows=[]
    for th in qs:
        use1=conf>=th
        fallback_fraction=float((~use1).mean())
        if fallback_fraction < 0.05:
            continue  # require the second stage to be meaningfully exercised
        pred=np.where(use1,stage1_pred,fallback_pred)
        m=simple_metrics(y,pred,labels)
        rows.append((score_objective(m),m['accuracy'],m['macro_f1'],float(th),int((~use1).sum()),pred))
    best=max(rows,key=lambda x:(x[0],x[1],x[2],-x[4]))
    return {'threshold':best[3],'validation_metrics':{'accuracy':best[1],'macro_f1':best[2]},
            'fallback_records':best[4],'fallback_fraction':best[4]/len(y),'predictions':best[5]}


def run_task(dataset, task, phase="all"):
    cache_dir=OUT/'_fit_cache'
    if cache_dir.exists(): shutil.rmtree(cache_dir)
    Xraw,y,Xtestraw,ytest,train_path,test_path,task_name=get_frames(dataset,task)
    labels=sorted(set(y)|set(ytest))
    # Group by exact feature-row hashes before making the validation fold. This
    # prevents duplicated records from appearing in both fit and validation.
    groups=pd.util.hash_pandas_object(Xraw,index=False).to_numpy()
    group_frame=pd.DataFrame({'group':groups,'label':y})
    group_labels=group_frame.groupby('group',sort=False)['label'].agg(lambda z:z.value_counts().index[0])
    g_train,g_val=train_test_split(group_labels.index.to_numpy(),test_size=.15,random_state=SEED,
                                  stratify=group_labels.to_numpy())
    train_group_set=set(g_train.tolist())
    idx_train=np.flatnonzero(np.fromiter((g in train_group_set for g in groups),dtype=bool,count=len(groups)))
    idx_val=np.flatnonzero(~np.fromiter((g in train_group_set for g in groups),dtype=bool,count=len(groups)))
    if set(groups[idx_train]).intersection(set(groups[idx_val])):
        raise RuntimeError('Duplicate-record groups leaked across validation split')
    Xfit=Xraw.iloc[idx_train].reset_index(drop=True); Xval=Xraw.iloc[idx_val].reset_index(drop=True)
    yfit=y[idx_train]; yval=y[idx_val]
    pre_val=make_preprocessor(Xfit)
    Zfit=np.asarray(pre_val.fit_transform(Xfit),dtype=np.float32)
    Zval=np.asarray(pre_val.transform(Xval),dtype=np.float32)
    feat_names=pre_val.get_feature_names_out().tolist(); d=Zfit.shape[1]
    print(f'[{dataset} / {task}] fit={len(yfit)} val={len(yval)} test={len(ytest)} encoded_features={d} classes={len(labels)}',flush=True)

    # Use a deterministic stratified subset for model selection so the
    # validation search remains laptop-feasible. Final models below use all rows.
    tune_idx=stratified_sample_indices(yfit,min(8000,len(yfit)),SEED+7,min_per_class=10)
    Zt=Zfit[tune_idx]; yt=np.asarray(yfit,dtype=object)[tune_idx]
    print(f' tuning on {len(yt)} stratified fit rows; full refit later uses {len(yfit)}',flush=True)
    # Rank all candidates on the tuning fold only.
    rankings, rank_arrays=feature_rankings(Zt,yt,SEED+11)
    candidate_results=[]
    all_weight='balanced'; all_variant=0
    all_val_model=train_lgb(Zt,yt,all_weight,all_variant,SEED)
    all_val_pred=np.asarray(all_val_model.predict(Zval),dtype=object)
    all_val_metrics=simple_metrics(yval,all_val_pred,labels)
    all_row={'approach':'all','model_variant':all_variant,'class_weight':all_weight,
             'objective':score_objective(all_val_metrics),'metrics':all_val_metrics}
    print(' all selected balanced LightGBM',round(all_val_metrics['accuracy'],4),
          round(all_val_metrics['macro_f1'],4),flush=True)

    fixed_candidates=[]
    fixed_rank_name='Hybrid (gain ratio + MI + LightGBM)'
    rank=rankings[fixed_rank_name]
    for fraction in [.35,.50]:
        k=max(3,min(d,int(round(d*fraction)))); take=np.asarray(rank[:k],dtype=int)
        model=train_lgb(Zt[:,take],yt,all_weight,all_variant,SEED+31+k)
        pred=np.asarray(model.predict(Zval[:,take]),dtype=object)
        mm=simple_metrics(yval,pred,labels)
        row={'approach':'fixed','selector':fixed_rank_name,'feature_fraction':fraction,
             'feature_count':k,'class_weight':all_weight,'model_variant':all_variant,
             'objective':score_objective(mm),'metrics':mm}
        fixed_candidates.append((row,model,take,pred))
        print(' fixed',fraction,round(mm['accuracy'],4),round(mm['macro_f1'],4),flush=True)
    best_fixed=max(fixed_candidates,key=lambda z:(z[0]['objective'],z[0]['metrics']['accuracy'],z[0]['metrics']['macro_f1']))
    fixed_row,fixed_val_model,fixed_idx,fixed_val_pred=best_fixed
    print(' BEST all',round(all_row['metrics']['accuracy'],4),round(all_row['metrics']['macro_f1'],4),
          'fixed',round(fixed_row['metrics']['accuracy'],4),round(fixed_row['metrics']['macro_f1'],4),flush=True)

    # Dynamic multi-expert router: a compact shared router proposes the top-K
    # likely classes, then only those classes' one-vs-rest specialists are run.
    # Every specialist has its own training-only MI-selected feature subset.
    global_rank=rankings['Hybrid (gain ratio + MI + LightGBM)']
    router_fraction=.40
    router_k=max(3,min(d,int(round(d*router_fraction))))
    router_idx=np.asarray(global_rank[:router_k],dtype=int)
    router_val=train_lgb(Zt[:,router_idx],yt,all_weight,all_variant,SEED+101)
    router_prob=np.asarray(router_val.predict_proba(Zval[:,router_idx]))
    router_classes=np.asarray(router_val.classes_,dtype=object)
    expert_val={}; expert_indices_val={}; expert_fit_rows={}
    for class_no,cls in enumerate(router_classes):
        binary_y=np.where(yt==cls,str(cls),'__OTHER__').astype(object)
        esample=stratified_sample_indices(binary_y,min(6000,len(binary_y)),SEED+200+class_no,min_per_class=2)
        Xes=Zt[esample]; yes=binary_y[esample]
        mi,_=binned_information_scores(Xes,yes,bins=10)
        expert_k=max(3,min(d,int(round(d*.25))))
        eidx=np.argsort(mi)[::-1][:expert_k].astype(int)
        emodel=train_lgb(Xes[:,eidx],yes,'balanced',0,SEED+400+class_no)
        pos= int(np.flatnonzero(np.asarray(emodel.classes_,dtype=object)==str(cls))[0])
        eprob=np.asarray(emodel.predict_proba(Zval[:,eidx]))[:,pos]
        expert_val[str(cls)]=eprob; expert_indices_val[str(cls)]=eidx
        expert_fit_rows[str(cls)]=int(len(esample))
        print(' expert',str(cls),'features',expert_k,'fit_rows',len(esample),flush=True)

    print(' all validation experts complete; assembling matrix',flush=True)
    expert_val_matrix=np.column_stack([expert_val[str(cls)] for cls in router_classes])
    print(' expert matrix',expert_val_matrix.shape,flush=True)
    dyn_candidates=[]
    for top_k in [1,2,3]:
        candidate_ids=np.argsort(-router_prob,axis=1)[:,:min(top_k,len(router_classes))]
        candidate_mask=np.zeros_like(router_prob,dtype=bool)
        row_ids=np.arange(len(yval))
        feature_mask=np.zeros((len(yval),d),dtype=bool); feature_mask[:,router_idx]=True
        for pos in range(candidate_ids.shape[1]):
            candidate_mask[row_ids,candidate_ids[:,pos]]=True
        for ci,cls in enumerate(router_classes):
            rows=np.flatnonzero(np.any(candidate_ids==ci,axis=1))
            if len(rows): feature_mask[np.ix_(rows,expert_indices_val[str(cls)])]=True
        feature_counts=feature_mask.sum(axis=1).astype(float)
        print(' dynamic features top-k',top_k,flush=True)
        avg_features=float(feature_counts.mean())
        base_scores=np.log(np.maximum(router_prob,1e-12))
        expert_scores=np.log(np.maximum(expert_val_matrix,1e-12))
        for alpha in [.25,.50,1.0]:
            scores=np.where(candidate_mask,base_scores+alpha*expert_scores,-1e12)
            pred=router_classes[np.argmax(scores,axis=1)]
            mm=simple_metrics(yval,pred,labels)
            row={'approach':'dynamic','router_selector':'Hybrid gain-ratio + MI + LightGBM',
                 'router_feature_fraction':router_fraction,'router_feature_count':router_k,
                 'expert_selector':'class-specific mutual information','expert_feature_fraction':.25,
                 'expert_feature_count':expert_k,'top_k_experts':top_k,'expert_weight_alpha':alpha,
                 'feature_count':avg_features,'objective':score_objective(mm),'metrics':mm}
            dyn_candidates.append((row,router_val,router_idx,expert_val,expert_indices_val,pred,feature_counts))
        print(' dynamic top-k',top_k,'evaluated',flush=True)
    best_dyn=max(dyn_candidates,key=lambda z:(z[0]['objective'],z[0]['metrics']['accuracy'],
                  z[0]['metrics']['macro_f1'],-z[0]['feature_count']))
    dyn_row,dyn_val_model,dyn_idx,dyn_val_probs,dyn_expert_idx,dyn_val_pred,dyn_val_feature_counts=best_dyn
    dyn_row['adaptive_mode']='router-specialists'
    # Second legitimate adaptive design: start with the fixed feature subset,
    # then acquire the complete feature vector only when the compact model is
    # uncertain. The confidence threshold is selected exclusively on validation.
    fixed_val_prob=np.asarray(fixed_val_model.predict_proba(Zval[:,fixed_idx]),dtype=float)
    fixed_conf=np.max(fixed_val_prob,axis=1)
    cascade_candidates=[]
    for th in np.unique(np.quantile(fixed_conf,[0,.10,.20,.30,.40,.50,.60,.70,.80,.90,.95,1.0])):
        fallback=fixed_conf < float(th); fallback_fraction=float(fallback.mean())
        if fallback_fraction < .05 or fallback_fraction > .95: continue
        cascade_pred=np.asarray(fixed_val_pred,dtype=object).copy()
        cascade_pred[fallback]=all_val_pred[fallback]
        mm=simple_metrics(yval,cascade_pred,labels)
        avg_features=float(len(fixed_idx)*(1-fallback_fraction)+d*fallback_fraction)
        row={'approach':'dynamic','adaptive_mode':'confidence-gated fixed-to-all',
             'confidence_source':'maximum class probability from compact model',
             'confidence_threshold':float(th),'fallback_fraction':fallback_fraction,
             'feature_count':avg_features,'objective':score_objective(mm),'metrics':mm}
        cascade_candidates.append((row,cascade_pred))
    if cascade_candidates:
        best_cascade=max(cascade_candidates,key=lambda z:(z[0]['objective'],z[0]['metrics']['accuracy'],
                               z[0]['metrics']['macro_f1'],-z[0]['feature_count']))
        if (best_cascade[0]['objective'],best_cascade[0]['metrics']['accuracy'],best_cascade[0]['metrics']['macro_f1']) > (dyn_row['objective'],dyn_row['metrics']['accuracy'],dyn_row['metrics']['macro_f1']):
            dyn_row=best_cascade[0]
            dyn_row['adaptive_mode']='confidence-gated fixed-to-all'
            dyn_idx=np.asarray(fixed_idx,dtype=int)
            dyn_val_pred=best_cascade[1]
            dyn_val_feature_counts=None
    print(' BEST dynamic',dyn_row.get('adaptive_mode'),round(dyn_row['metrics']['accuracy'],4),round(dyn_row['metrics']['macro_f1'],4),
          'features',round(dyn_row['feature_count'],1),flush=True)

    # Refit the selected preprocess/model configurations on all official training
    # rows. Map feature indices by the stable preprocessor output feature names.
    print(' validation selection complete; fitting full preprocessor',flush=True)
    pre=make_preprocessor(Xraw)
    Ztrain=np.asarray(pre.fit_transform(Xraw),dtype=np.float32)
    Ztest=np.asarray(pre.transform(Xtestraw),dtype=np.float32)
    final_names=pre.get_feature_names_out().tolist(); name_to_idx={n:i for i,n in enumerate(final_names)}
    # The same known features should exist in both preprocessors. Any category
    # present only in the validation partition is absent in the refit schema and
    # thus is simply not selected; record this explicitly.
    def map_features(old_idx):
        names=[feat_names[int(i)] for i in old_idx]
        chosen=[name_to_idx[n] for n in names if n in name_to_idx]
        if not chosen: raise RuntimeError('Selected feature subset did not map into final feature schema')
        return np.asarray(chosen,dtype=int)
    fixed_final_idx=map_features(fixed_idx)
    dynamic_final_idx=map_features(dyn_idx)

    print(' full preprocessor complete; fitting final model bundle in isolated worker',flush=True)
    temp=OUT/'_fit_cache'; temp.mkdir(parents=True,exist_ok=True)
    matrix_path=temp/'matrix.npy'; labels_path=temp/'labels.npy'; bundle_path=temp/'final_bundle.joblib'
    np.save(matrix_path,np.asarray(Ztrain,dtype=np.float32)); np.save(labels_path,np.asarray(y,dtype=object))
    worker_cfg={
      'all':{'indices':list(range(Ztrain.shape[1])),'params':lgbm_params(all_row['class_weight'],all_row['model_variant'],SEED)},
      'fixed':{'indices':fixed_final_idx.tolist(),'params':lgbm_params(fixed_row['class_weight'],fixed_row['model_variant'],SEED+31+len(fixed_final_idx))},
      'router':{'indices':dynamic_final_idx.tolist(),'params':lgbm_params(all_weight,all_variant,SEED+101)},
      'experts':[]}
    dynamic_final_expert_idx={}
    for class_no,cls in enumerate(sorted(set(y))):
        binary_y=np.where(y==cls,str(cls),'__OTHER__').astype(object)
        esample=stratified_sample_indices(binary_y,min(6000,len(binary_y)),SEED+200+class_no,min_per_class=2)
        Xes=Ztrain[esample]; yes=binary_y[esample]
        mi,_=binned_information_scores(Xes,yes,bins=10)
        ek=max(3,min(Ztrain.shape[1],int(round(Ztrain.shape[1]*.25))))
        eidx=np.argsort(mi)[::-1][:ek].astype(int)
        dynamic_final_expert_idx[str(cls)]=eidx.tolist()
        worker_cfg['experts'].append({'family':str(cls),'indices':eidx.tolist(),'rows':esample.tolist(),
            'params':lgbm_params('balanced',0,SEED+400+class_no)})
        print(' selected full-data expert',str(cls),'features',len(eidx),'rows',len(esample),flush=True)
    config_path=temp/'worker_config.json'; config_path.write_text(json.dumps(worker_cfg))
    joblib.dump(pre,temp/'preprocessor.joblib',compress=3)
    global_importance={}
    for j,score in enumerate(rank_arrays['hybrid']):
        name=feat_names[j]
        if name in name_to_idx: global_importance[name]=float(score)
    global_importance=dict(sorted(global_importance.items(),key=lambda kv:kv[1],reverse=True)[:30])
    state={
      'dataset':dataset,'task':task,'task_name':task_name,'labels':list(labels),
      'train_records':int(len(y)),'test_records':int(len(ytest)),
      'validation_records':int(len(yval)),'tuning_records':int(len(yt)),
      'all_candidate':all_row,'fixed_candidate':fixed_row,'dynamic_candidate':dyn_row,
      'fixed_indices':fixed_final_idx.tolist(),'router_indices':dynamic_final_idx.tolist(),
      'feature_names':final_names,'global_importance':global_importance,
      'train_sha256':sha256(train_path),'test_sha256':sha256(test_path),
      'validation_split':'85/15 stratified group holdout; exact duplicate feature rows kept within one side',
      'validation_group_count':int(len(group_labels)),
      'selection_objective':'0.55 * accuracy + 0.45 * macro-F1',
      'lightgbm_parameters':lgbm_params(all_row['class_weight'],all_row['model_variant'],SEED),
      'dynamic_validation':{'adaptive_mode':dyn_row.get('adaptive_mode','router-specialists'),
          'top_k_experts':int(dyn_row.get('top_k_experts',1)),
          'expert_weight_alpha':float(dyn_row.get('expert_weight_alpha',0.5)),
          'confidence_threshold':float(dyn_row.get('confidence_threshold',0.0)),
          'fallback_fraction':float(dyn_row.get('fallback_fraction',0.0)),
          'average_features':float(dyn_row['feature_count'])}}
    (temp/'selection_state.json').write_text(json.dumps(state,indent=2,default=float))
    if phase=='select':
        print(' Prepared selection, preprocessor, matrices, and model configuration.',flush=True)
        return {'prepared':True,'dataset':dataset,'task':task}
    cmd=[sys.executable,str(ROOT/'scripts'/'fit_lgbm_bundle.py'),'--matrix',str(matrix_path),
         '--labels',str(labels_path),'--config-file',str(config_path),'--output',str(bundle_path)]
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    subprocess.run(cmd,check=True,env=env,capture_output=False)
    bundle=joblib.load(bundle_path)
    all_model=bundle['all']['model']; fixed_model=bundle['fixed']['model']; dynamic_model=bundle['router']['model']
    dynamic_experts=bundle['experts']
    all_train_s=bundle['all']['training_seconds']; fixed_train_s=bundle['fixed']['training_seconds']; dynamic_train_s=bundle['router']['training_seconds']
    expert_train_seconds=sum(e['training_seconds'] for e in dynamic_experts.values())
    for family,e in dynamic_experts.items(): print(' installed expert',family,'features',len(e['indices']),flush=True)
    dynamic_final_expert_idx={family:e['indices'] for family,e in dynamic_experts.items()}

    t=time.perf_counter(); p_all=all_model.predict(Ztest); all_inf=time.perf_counter()-t
    t=time.perf_counter(); p_fixed=fixed_model.predict(Ztest[:,fixed_final_idx]); fixed_inf=time.perf_counter()-t
    t=time.perf_counter(); rp=np.asarray(dynamic_model.predict_proba(Ztest[:,dynamic_final_idx])); rclasses=np.asarray(dynamic_model.classes_,dtype=object)
    topk=int(dyn_row['top_k_experts']); alpha=float(dyn_row['expert_weight_alpha'])
    cands=np.argsort(-rp,axis=1)[:,:min(topk,len(rclasses))]
    candidate_mask=np.zeros_like(rp,dtype=bool); row_ids=np.arange(len(Ztest))
    feature_mask=np.zeros((len(Ztest),Ztest.shape[1]),dtype=bool); feature_mask[:,dynamic_final_idx]=True
    for pos in range(cands.shape[1]): candidate_mask[row_ids,cands[:,pos]]=True
    eprob_test=np.full_like(rp,np.nan,dtype=float)
    for ci,cls_value in enumerate(rclasses):
        candidate_rows=np.flatnonzero(np.any(cands==ci,axis=1))
        if len(candidate_rows):
            e=dynamic_experts[str(cls_value)]; ei=np.asarray(e['indices'],dtype=int)
            eprob_test[candidate_rows,ci]=e['model'].predict_proba(Ztest[candidate_rows][:,ei])[:,e['positive_index']]
            feature_mask[np.ix_(candidate_rows,ei)]=True
    score_matrix=np.where(candidate_mask,np.log(np.maximum(rp,1e-12))+alpha*np.log(np.maximum(eprob_test,1e-12)),-1e12)
    p_dyn=rclasses[np.argmax(score_matrix,axis=1)]
    dyn_counts=feature_mask.sum(axis=1).astype(float)
    dyn_inf=time.perf_counter()-t
    d_final=Ztest.shape[1]; k_final=len(dynamic_final_idx)
    dyn_feature_count=float(dyn_counts.mean())
    preprocessing_start=time.perf_counter(); _=pre.transform(Xtestraw); preprocessing_seconds=time.perf_counter()-preprocessing_start
    print(' test predictions complete; computing metrics',flush=True)
    results={
      'all':metric_pack(ytest,p_all,labels,all_inf,d_final),
      'fixed':metric_pack(ytest,p_fixed,labels,fixed_inf,len(fixed_final_idx)),
      'dynamic':metric_pack(ytest,p_dyn,labels,dyn_inf,dyn_feature_count),
    }
    for k,m,ts in [('all',all_model,all_train_s),('fixed',fixed_model,fixed_train_s),('dynamic',dynamic_model,dynamic_train_s)]:
        results[k]['training_seconds']=float(ts)
        results[k]['preprocessing_seconds']=float(preprocessing_seconds)
        results[k]['end_to_end_inference_seconds']=float(results[k]['inference_seconds']+preprocessing_seconds)
        results[k]['end_to_end_inference_ms_per_record']=float(1000*results[k]['end_to_end_inference_seconds']/max(1,len(ytest)))
    results['dynamic'].update({'router_feature_count':k_final,'average_unique_features':float(dyn_counts.mean()),
        'top_k_experts':topk,'expert_weight_alpha':alpha,'expert_features':dynamic_final_expert_idx,})
    # Confusion cell records for the existing inspection UI.
    X_test_records=Xtestraw.reset_index(drop=True)
    predictions={'all':p_all,'fixed':p_fixed,'dynamic':p_dyn}
    confusion_records={}
    for approach,preds in predictions.items():
        cells={}
        for ri,(actual,predicted) in enumerate(zip(ytest,preds)):
            key=f'{actual}|||{predicted}'; bucket=cells.setdefault(key,[])
            if len(bucket)<25:
                bucket.append({'test_record':int(ri),'actual':str(actual),'predicted':str(predicted),
                               'record':X_test_records.iloc[ri].to_dict()})
        confusion_records[approach]=cells

    result_name=f'{dataset.lower().replace("-","_")}_{task}'
    outdir=OUT/result_name; outdir.mkdir(parents=True,exist_ok=True)
    artifact={
      'dataset':dataset,'task':task_name,'kind':'LightGBM Adaptive Specialists','seed':SEED,
      'model_config':{'classifier':'LightGBM','selection_protocol':'20% stratified validation split from official training split; selected by 0.55*accuracy + 0.45*macro-F1',
                      'test_used_for_tuning':False,'all_candidate':all_row,'fixed_candidate':{k:v for k,v in fixed_row.items() if k not in ('metrics',)},
                      'dynamic_candidate':{k:v for k,v in dyn_row.items() if k not in ('metrics','validation_gate')},
                      'lightgbm_parameters':lgbm_params(all_row['class_weight'],all_row['model_variant'],SEED)},
      'feature_columns':Xraw.columns.tolist(),'feature_names':final_names,'preprocessor':pre,'cat_feature_indices':[],
      'models':{'all':all_model,'fixed':fixed_model},'fixed_indices':fixed_final_idx.tolist(),
      'family_features':{cls:[final_names[i] for i in dynamic_experts[cls]['indices']] for cls in dynamic_experts},
      'router':dynamic_model,'router_indices':dynamic_final_idx.tolist(),'experts':dynamic_experts,
      'adaptive_router_mode':'one-vs-rest top-K candidate specialists',
      'dynamic_architecture':'compact multiclass router selects top-K candidates; class-specific one-vs-rest specialists use MI-selected subsets',
      'feature_ranking':'training-only gain ratio, mutual information, LightGBM gain and hybrid ranking',
      'labels':labels,'results':results,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),
      'records':int(len(y)+len(ytest)),'train_records':int(len(y)),'test_records':int(len(ytest)),
      'split':'official train/test files','global_importance':{final_names[i]:float(rank_arrays['hybrid'][i]) for i in np.argsort(rank_arrays['hybrid'])[::-1][:30] if i < len(final_names)},'confusion_records':confusion_records,
      'research_metadata':{'task':task_name,'validation_records':len(yval),'validation_split':'85/15 stratified group holdout; exact duplicate feature rows kept within one side','validation_unique_groups':int(len(np.unique(groups))),'validation_selected_top_k_experts':int(dyn_row['top_k_experts']),
         'validation_selected_feature_names':[feat_names[i] for i in dyn_idx],
         'official_train_sha256':sha256(train_path),'official_test_sha256':sha256(test_path),
         'outer_test_opened_after_selection':True,'objective':'0.55 * accuracy + 0.45 * macro-F1',
         'validation_all':{'model_variant':all_row['model_variant'],'class_weight':all_row['class_weight'],
                           'accuracy':all_row['metrics']['accuracy'],'macro_f1':all_row['metrics']['macro_f1']},
         'validation_fixed':{'selector':fixed_row['selector'],'feature_fraction':fixed_row['feature_fraction'],
                           'accuracy':fixed_row['metrics']['accuracy'],'macro_f1':fixed_row['metrics']['macro_f1']},
         'validation_dynamic':{'router_feature_fraction':dyn_row['router_feature_fraction'],
                           'expert_feature_fraction':dyn_row['expert_feature_fraction'],'top_k_experts':dyn_row['top_k_experts'],
                           'expert_weight_alpha':dyn_row['expert_weight_alpha'],'accuracy':dyn_row['metrics']['accuracy'],
                           'macro_f1':dyn_row['metrics']['macro_f1'],'average_features':dyn_row['feature_count']}}
    }
    joblib.dump(artifact,outdir/'model.joblib',compress=3)
    # If this is the primary native task, install the model into the normal app
    # artifact path so the regular benchmark and prediction UI use these models.
    if task=='native':
        appdir=BACKEND/'artifacts'/dataset.lower().replace('-','_'); appdir.mkdir(parents=True,exist_ok=True)
        joblib.dump(artifact,appdir/'model.joblib',compress=3)
        metadata={k:v for k,v in artifact.items() if k not in ('preprocessor','models','router','experts','confusion_records','adaptive_cascade')}
        (appdir/'metadata.json').write_text(json.dumps(metadata,indent=2,default=float))
    (outdir/'metrics.json').write_text(json.dumps({k:v for k,v in artifact.items() if k not in ('preprocessor','models','router','experts')},indent=2,default=float))
    print(f'[{dataset} / {task}] FINAL TEST',flush=True)
    for name,m in results.items():
        print(name, 'accuracy',round(m['accuracy']*100,3),'macroF1',round(m['macro_f1']*100,3),
              'weightedF1',round(m['weighted_f1']*100,3),'features',round(m['feature_count'],2),
              'latency_ms/row',round(m['end_to_end_inference_ms_per_record'],4),flush=True)
    return artifact


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',default='NSL-KDD,UNSW-NB15')
    parser.add_argument('--tasks',default='native,binary')
    parser.add_argument('--phase',choices=['all','select'],default='all')
    args=parser.parse_args()
    summary=[]
    for ds in args.datasets.split(','):
        for task in args.tasks.split(','):
            try:
                art=run_task(ds.strip(),task.strip(),phase=args.phase)
                if args.phase=='all': summary.append({'dataset':ds.strip(),'task':task.strip(),'results':art['results'],
                                'dynamic_config':art['model_config']['dynamic_candidate']})
            except Exception as exc:
                print(f'ERROR {ds}/{task}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
                raise
    OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='all': (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=float))
    print('WROTE',OUT/'summary.json',flush=True) if args.phase=='all' else print('SELECTION READY',flush=True)

if __name__=='__main__': main()
