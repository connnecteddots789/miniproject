"""Finish official-test evaluation and persist an adaptive SENTINEL model bundle."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

ROOT=Path(__file__).resolve().parents[1]
BACKEND=ROOT/'backend'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(BACKEND))
from optimize_adaptive_research import get_frames, metric_pack

OUT=BACKEND/'artifacts'/'research'/'adaptive_feature_selection'

def finish(dataset,task):
    taskdir=OUT/f"{dataset.lower().replace('-','_')}_{task}"
    cache=OUT/'_fit_cache'
    state=json.loads((cache/'selection_state.json').read_text())
    if state['dataset']!=dataset or state['task']!=task:
        raise RuntimeError('The prepared selection does not match this dataset/task')
    if (cache/'final_bundle.joblib').exists():
        bundle=joblib.load(cache/'final_bundle.joblib')
    else:
        bundle={
          'all':joblib.load(cache/'all.joblib'),
          'fixed':joblib.load(cache/'fixed.joblib'),
          'router':joblib.load(cache/'router.joblib'),
          'experts':joblib.load(cache/'experts.joblib')}
    pre=joblib.load(cache/'preprocessor.joblib')
    Xraw,y,Xtest_raw,ytest,train_path,test_path,task_name=get_frames(dataset,task)
    final_names=pre.get_feature_names_out().tolist()
    Ztest=np.asarray(pre.transform(Xtest_raw),dtype=np.float32)
    ytest=np.asarray(ytest,dtype=object); labels=state['labels']
    fixed_idx=np.asarray(state['fixed_indices'],dtype=int)
    router_idx=np.asarray(state['router_indices'],dtype=int)
    all_model=bundle['all']['model']; fixed_model=bundle['fixed']['model']; router=bundle['router']['model']
    experts=bundle['experts']
    t=time.perf_counter(); p_all=np.asarray(all_model.predict(Ztest),dtype=object); all_inf=time.perf_counter()-t
    t=time.perf_counter(); p_fixed=np.asarray(fixed_model.predict(Ztest[:,fixed_idx]),dtype=object); fixed_inf=time.perf_counter()-t
    adaptive_mode=state['dynamic_validation'].get('adaptive_mode','router-specialists')
    if adaptive_mode=='confidence-gated fixed-to-all':
        threshold=float(state['dynamic_validation']['confidence_threshold'])
        t=time.perf_counter()
        fixed_prob=np.asarray(fixed_model.predict_proba(Ztest[:,fixed_idx]),dtype=float)
        fixed_classes=np.asarray(fixed_model.classes_,dtype=object)
        fixed_stage_pred=fixed_classes[np.argmax(fixed_prob,axis=1)]
        confidence=np.max(fixed_prob,axis=1)
        fallback=confidence < threshold
        p_dynamic=np.asarray(fixed_stage_pred,dtype=object).copy()
        if np.any(fallback):
            p_dynamic[fallback]=all_model.predict(Ztest[fallback])
        fallback_fraction=float(fallback.mean())
        dyn_feature_count=float((~fallback).sum()*len(fixed_idx)+fallback.sum()*Ztest.shape[1])/len(Ztest)
        dyn_inf=time.perf_counter()-t
        topk=0; alpha=0.0
    else:
        topk=int(state['dynamic_validation']['top_k_experts']); alpha=float(state['dynamic_validation']['expert_weight_alpha'])
        t=time.perf_counter()
        router_prob=np.asarray(router.predict_proba(Ztest[:,router_idx]),dtype=float)
        classes=np.asarray(router.classes_,dtype=object)
        candidate_ids=np.argsort(-router_prob,axis=1)[:,:min(topk,len(classes))]
        candidate_mask=np.zeros_like(router_prob,dtype=bool); rows=np.arange(len(Ztest))
        for pos in range(candidate_ids.shape[1]): candidate_mask[rows,candidate_ids[:,pos]]=True
        feature_mask=np.zeros((len(Ztest),Ztest.shape[1]),dtype=bool); feature_mask[:,router_idx]=True
        expert_prob=np.full_like(router_prob,np.nan,dtype=float)
        for ci,cls in enumerate(classes):
            candidate_rows=np.flatnonzero(np.any(candidate_ids==ci,axis=1))
            if len(candidate_rows):
                expert=experts[str(cls)]; idx=np.asarray(expert['indices'],dtype=int)
                expert_prob[candidate_rows,ci]=expert['model'].predict_proba(Ztest[candidate_rows][:,idx])[:,int(expert['positive_index'])]
                feature_mask[np.ix_(candidate_rows,idx)]=True
        scores=np.where(candidate_mask,np.log(np.maximum(router_prob,1e-12))+alpha*np.log(np.maximum(expert_prob,1e-12)),-1e12)
        p_dynamic=classes[np.argmax(scores,axis=1)]
        dynamic_features_per_record=feature_mask.sum(axis=1).astype(float)
        dyn_feature_count=float(dynamic_features_per_record.mean())
        dyn_inf=time.perf_counter()-t
        fallback_fraction=0.0
    t=time.perf_counter(); _=pre.transform(Xtest_raw); pre_seconds=time.perf_counter()-t
    results={
      'all':metric_pack(ytest,p_all,labels,all_inf,Ztest.shape[1]),
      'fixed':metric_pack(ytest,p_fixed,labels,fixed_inf,len(fixed_idx)),
      'dynamic':metric_pack(ytest,p_dynamic,labels,dyn_inf,dyn_feature_count)}
    train_seconds={
      'all':float(bundle['all']['training_seconds']),
      'fixed':float(bundle['fixed']['training_seconds']),
      'dynamic':float(bundle['router']['training_seconds']+sum(e['training_seconds'] for e in experts.values()))}
    for key in results:
        results[key]['training_seconds']=train_seconds[key]
        results[key]['preprocessing_seconds']=float(pre_seconds)
        results[key]['end_to_end_inference_seconds']=float(results[key]['inference_seconds']+pre_seconds)
        results[key]['end_to_end_inference_ms_per_record']=float(1000*results[key]['end_to_end_inference_seconds']/max(1,len(ytest)))
    results['dynamic'].update({'adaptive_mode':adaptive_mode,'router_feature_count':len(router_idx),'average_unique_features':dyn_feature_count,
        'top_k_experts':topk,'expert_weight_alpha':alpha,'fallback_fraction':fallback_fraction,
        'confidence_threshold':float(state['dynamic_validation'].get('confidence_threshold',0.0)),
        'expert_features':{k:v['indices'] for k,v in experts.items()}})
    predictions={'all':p_all,'fixed':p_fixed,'dynamic':p_dynamic}
    confusion_records={}; test_records=Xtest_raw.reset_index(drop=True)
    for approach,preds in predictions.items():
        cells={}
        for ri,(actual,predicted) in enumerate(zip(ytest,preds)):
            key=f'{actual}|||{predicted}'; bucket=cells.setdefault(key,[])
            if len(bucket)<25:
                bucket.append({'test_record':int(ri),'actual':str(actual),'predicted':str(predicted),
                               'record':test_records.iloc[ri].to_dict()})
        confusion_records[approach]=cells
    fixed_candidate=state['fixed_candidate']
    dynamic_candidate=state['dynamic_candidate']
    all_candidate=state['all_candidate']
    model_config={
      'classifier':'LightGBM',
      'selection_protocol':'80/20 stratified group holdout drawn only from the official training split; exact duplicate feature rows stay in one side; final refit uses the complete official training split',
      'test_used_for_tuning':False,
      'all_candidate':all_candidate,
      'fixed_candidate':fixed_candidate,
      'dynamic_candidate':dynamic_candidate,
      'lightgbm_parameters':state['lightgbm_parameters']}
    family_features={cls:[final_names[i] for i in e['indices']] for cls,e in experts.items()}
    artifact={
      'dataset':dataset,'task':task_name,'kind':'LightGBM Adaptive Specialists','seed':42,
      'model_config':model_config,'feature_columns':Xraw.columns.tolist(),'feature_names':final_names,
      'preprocessor':pre,'cat_feature_indices':[],
      'models':{'all':all_model,'fixed':fixed_model},'fixed_indices':fixed_idx.tolist(),
      'family_features':family_features,'expert_training_sample_limit':10000,
      'router':router,'router_indices':router_idx.tolist(),'experts':experts,
      'adaptive_router_mode':('confidence-gated fixed-to-all' if adaptive_mode=='confidence-gated fixed-to-all' else 'one-vs-rest top-K candidate specialists'),
      'dynamic_architecture':('compact model processes every record; full-feature model is invoked only when compact-model confidence falls below the training-validation threshold' if adaptive_mode=='confidence-gated fixed-to-all' else 'compact multiclass router proposes top-K candidate labels; only those class-specific one-vs-rest specialists run, each using its own training-only MI-ranked feature subset'),
      'feature_ranking':'training-only gain ratio + binned mutual information + LightGBM gain hybrid; experts use class-specific binned mutual information',
      'labels':labels,'results':results,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),
      'records':int(len(y)+len(ytest)),'train_records':int(len(y)),'test_records':int(len(ytest)),
      'split':'official train/test files','global_importance':state['global_importance'],
      'confusion_records':confusion_records,
      'research_metadata':{
        'task':task_name,'validation_records':state['validation_records'],'tuning_records':state['tuning_records'],
        'validation_split':state['validation_split'],'validation_unique_groups':state['validation_group_count'],
        'validation_selected_top_k_experts':topk,'validation_selected_feature_names':[final_names[i] for i in router_idx],
        'official_train_sha256':state['train_sha256'],'official_test_sha256':state['test_sha256'],
        'outer_test_opened_after_selection':True,'objective':state['selection_objective'],
        'validation_all':{'accuracy':all_candidate['metrics']['accuracy'],'macro_f1':all_candidate['metrics']['macro_f1']},
        'validation_fixed':{'selector':fixed_candidate['selector'],'feature_fraction':fixed_candidate['feature_fraction'],
          'accuracy':fixed_candidate['metrics']['accuracy'],'macro_f1':fixed_candidate['metrics']['macro_f1']},
        'validation_dynamic':{'top_k_experts':topk,'expert_weight_alpha':alpha,
          'accuracy':dynamic_candidate['metrics']['accuracy'],'macro_f1':dynamic_candidate['metrics']['macro_f1'],
          'average_features':dynamic_candidate['feature_count']}}}
    taskdir.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,taskdir/'model.joblib',compress=3)
    (taskdir/'metrics.json').write_text(json.dumps({k:v for k,v in artifact.items() if k not in ('preprocessor','models','router','experts')},indent=2,default=float))
    if task=='native':
        appdir=BACKEND/'artifacts'/dataset.lower().replace('-','_'); appdir.mkdir(parents=True,exist_ok=True)
        joblib.dump(artifact,appdir/'model.joblib',compress=3)
        metadata={k:v for k,v in artifact.items() if k not in ('preprocessor','models','router','experts','confusion_records')}
        (appdir/'metadata.json').write_text(json.dumps(metadata,indent=2,default=float))
    print(f'[{dataset}/{task}] OFFICIAL TEST',flush=True)
    for k,m in results.items():
        print(k, 'accuracy',round(100*m['accuracy'],3),'macroF1',round(100*m['macro_f1'],3),
              'weightedF1',round(100*m['weighted_f1'],3),'features',round(m['feature_count'],2),
              'e2e_ms_per_record',round(m['end_to_end_inference_ms_per_record'],5),flush=True)
    return artifact

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--task',default='native')
    args=ap.parse_args(); finish(args.dataset,args.task)
