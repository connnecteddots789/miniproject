from pathlib import Path
import json, time, threading, traceback, os
# Avoid loky physical-core probing on Windows installations where WMIC is absent.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 8))
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from .datasets import load_dataset, prepare_labels, feature_frame, ROOT, dataset_files, load_nsl, load_unsw

JOBS={}; LOCK=threading.Lock(); MODEL_CACHE={}; MODEL_CACHE_LOCK=threading.Lock()
class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Adapter for libraries (notably XGBoost) that require integer class IDs."""
    def __init__(self, estimator): self.estimator=estimator
    def fit(self, X, y):
        self.encoder_=LabelEncoder().fit(y)
        encoded=self.encoder_.transform(y)
        if len(self.encoder_.classes_) < 2:
            raise ValueError('Classifier training requires at least two distinct classes; check the training split and family routing.')
        self.estimator_=clone(self.estimator)
        # XGBClassifier needs an objective consistent with the actual target
        # cardinality. Dynamic family experts are binary; the shared router is
        # multiclass. A fixed multi:softprob objective without num_class can
        # produce XGBoost's "num_class must be >= 1" error on binary experts.
        if self.estimator_.__class__.__module__.startswith('xgboost'):
            class_count=len(self.encoder_.classes_)
            params={'objective':'binary:logistic','eval_metric':'logloss'} if class_count == 2 else {
                'objective':'multi:softprob','num_class':class_count,'eval_metric':'mlogloss'}
            self.estimator_.set_params(**params)
        self.estimator_.fit(X,encoded)
        self.classes_=self.encoder_.classes_
        return self
    def predict(self, X):
        return self.encoder_.inverse_transform(np.asarray(self.estimator_.predict(X),dtype=int).reshape(-1))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

def estimator(kind, seed, cat_features=None):
    """Deterministic CPU-safe estimator factory; optional libraries fail clearly."""
    workers=max(1,min(6,os.cpu_count() or 1))
    if kind == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, max_depth=28, random_state=seed, n_jobs=workers, class_weight='balanced_subsample', min_samples_leaf=2, max_features='sqrt')
    if kind == 'Logistic Regression':
        return LogisticRegression(max_iter=1600, class_weight='balanced', C=1.0, random_state=seed)
    if kind == 'Extra Trees':
        return ExtraTreesClassifier(n_estimators=120, max_depth=28, random_state=seed, n_jobs=workers, class_weight='balanced', min_samples_leaf=2, max_features='sqrt')
    if kind in ('SVM', 'SVM (Linear)'):
        return LinearSVC(C=1.0, class_weight='balanced', random_state=seed, dual='auto', max_iter=5000)
    if kind == 'SVM (RBF)':
        return SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', cache_size=1024)
    if kind == 'XGBoost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc: raise RuntimeError('XGBoost is not installed. Run: pip install xgboost') from exc
        return LabelEncodedClassifier(XGBClassifier(n_estimators=180, max_depth=7, learning_rate=.08, subsample=.85, colsample_bytree=.85, tree_method='hist', n_jobs=workers, random_state=seed))
    if kind == 'LightGBM':
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc: raise RuntimeError('LightGBM is not installed. Run: pip install lightgbm') from exc
        return LGBMClassifier(n_estimators=180, max_depth=-1, num_leaves=31, learning_rate=.08, subsample=.85, colsample_bytree=.85, class_weight='balanced', n_jobs=workers, random_state=seed, verbosity=-1)
    if kind == 'CatBoost':
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc: raise RuntimeError('CatBoost is not installed. Run: pip install catboost') from exc
        # The common, training-only one-hot preprocessing keeps feature subsets consistent
        # across all three approaches. CatBoost is therefore trained on encoded columns.
        return CatBoostClassifier(iterations=180, depth=7, learning_rate=.08, loss_function='MultiClass', verbose=False, thread_count=workers, random_seed=seed, allow_writing_files=False, cat_features=cat_features or [])
    if kind == 'Stacking Ensemble':
        try:
            from xgboost import XGBClassifier
            boosted=LabelEncodedClassifier(XGBClassifier(n_estimators=35,max_depth=5,learning_rate=.08,subsample=.85,colsample_bytree=.85,tree_method='hist',n_jobs=2,random_state=seed))
            bases=[('rf',RandomForestClassifier(n_estimators=30,max_depth=20,n_jobs=2,class_weight='balanced_subsample',random_state=seed)),('extra',ExtraTreesClassifier(n_estimators=30,max_depth=20,n_jobs=2,class_weight='balanced',random_state=seed)),('xgb',boosted)]
        except ImportError:
            bases=[('rf',RandomForestClassifier(n_estimators=30,max_depth=20,n_jobs=2,class_weight='balanced_subsample',random_state=seed)),('extra',ExtraTreesClassifier(n_estimators=30,max_depth=20,n_jobs=2,class_weight='balanced',random_state=seed))]
        # sklearn trains the logistic meta-model from out-of-fold base predictions.
        return StackingClassifier(estimators=bases, final_estimator=LogisticRegression(max_iter=1200,class_weight='balanced'), cv=3, stack_method='auto', n_jobs=1, passthrough=False)
    return DecisionTreeClassifier(random_state=seed, class_weight='balanced', min_samples_leaf=3, max_depth=24)

def stratified_sample_indices(y, limit, seed, min_per_class=2):
    """Deterministic proportional sample that preserves a small floor per class."""
    values=np.asarray(y,dtype=object).reshape(-1); labels=np.unique(values)
    if len(values)<=limit: return np.arange(len(values))
    if limit < len(labels)*min_per_class:
        raise ValueError('Training sample limit is too small to preserve every class.')
    rng=np.random.RandomState(seed); groups={label:np.flatnonzero(values==label) for label in labels}
    counts={label:len(idx) for label,idx in groups.items()}
    alloc={label:min(counts[label],max(min_per_class,int(limit*counts[label]/len(values)))) for label in labels}
    while sum(alloc.values())>limit:
        candidates=[k for k in labels if alloc[k]>min_per_class]
        if not candidates: break
        k=max(candidates,key=lambda x:alloc[x]-limit*counts[x]/len(values)); alloc[k]-=1
    while sum(alloc.values())<limit:
        candidates=[k for k in labels if alloc[k]<counts[k]]
        if not candidates: break
        k=max(candidates,key=lambda x:limit*counts[x]/len(values)-alloc[x]); alloc[k]+=1
    picked=np.concatenate([rng.choice(groups[k],size=alloc[k],replace=False) for k in labels])
    rng.shuffle(picked); return picked

def fit_classifier(model, X, y, seed, limit=12000):
    # RBF SVM has quadratic kernel costs; its bounded, stratified sample is explicit.
    # Linear SVM and tree/boosting models use the complete training split.
    if isinstance(model,StackingClassifier):
        min_class=int(pd.Series(y).value_counts().min())
        if min_class<2: raise ValueError('Stacking requires at least two training records in every class for out-of-fold meta-model training.')
        model.cv=StratifiedKFold(n_splits=min(3,min_class),shuffle=True,random_state=seed)
    cap=limit if isinstance(model,SVC) else (8000 if isinstance(model,StackingClassifier) else None)
    if cap and len(y)>cap:
        idx=stratified_sample_indices(y,cap,seed,min_per_class=3 if isinstance(model,StackingClassifier) else 2)
        model.fit(X[idx],np.asarray(y)[idx]); return len(idx)
    model.fit(X, y)
    return len(y)
def make_preprocessor(X, native_cats=False):
    """Training-only preprocessing; ordinal-encode high-cardinality IDs to avoid dense OHE blow-ups."""
    nums=X.select_dtypes(include=['number','bool']).columns.tolist()
    cats=[c for c in X.columns if c not in nums]
    parts=[]
    if nums:
        num_steps=[('impute',SimpleImputer(strategy='median'))]
        if not native_cats: num_steps.append(('scale',StandardScaler()))
        parts.append(('num',Pipeline(num_steps),nums))
    if cats:
        if native_cats:
            high_cats=cats; low_cats=[]
        else:
            high_cats=[c for c in cats if X[c].nunique(dropna=True)>50]
            low_cats=[c for c in cats if c not in high_cats]
        if high_cats:
            high_steps=[('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]
            if not native_cats: high_steps.append(('scale',StandardScaler()))
            parts.append(('cat_id',Pipeline(high_steps),high_cats))
        if low_cats:
            low_pipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            parts.append(('cat',low_pipe,low_cats))
    return ColumnTransformer(parts, remainder='drop', verbose_feature_names_out=False)

def model_input(matrix, indices, cat_indices, kind):
    values=matrix[:,indices]
    if kind=='CatBoost' and len(cat_indices):
        values=np.asarray(values,dtype=object).copy()
        for local_pos, global_idx in enumerate(indices):
            if int(global_idx) in cat_indices:
                values[:,local_pos]=np.asarray(values[:,local_pos]).astype(float).astype(int).astype(str)
    return values

def estimator_for_features(kind, seed, indices, cat_indices):
    local_cats=[pos for pos,global_idx in enumerate(indices) if int(global_idx) in cat_indices]
    return estimator(kind,seed,local_cats)

def metrics(y, pred, labels, elapsed, nfeatures):
    p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
    cm=confusion_matrix(y,pred,labels=labels)
    # Macro one-vs-rest FPR; also retain class-level recall in the report.
    fprs=[]
    for i in range(len(labels)):
        fp=float(cm[:,i].sum()-cm[i,i]); tn=float(cm.sum()-cm[i,:].sum()-cm[:,i].sum()+cm[i,i])
        fprs.append(fp/(fp+tn) if fp+tn else 0.0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f),'weighted_precision':float(pw),'weighted_recall':float(rw),'weighted_f1':float(fw),'macro_false_positive_rate':float(np.mean(fprs)),'inference_seconds':float(elapsed),'inference_ms_per_record':float(1000*elapsed/max(1,len(y))),'inference_records':int(len(y)),'feature_count':float(nfeatures),'per_class':classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':cm.tolist(),'labels':list(labels)}
def model_configuration(kind, seed, split_name):
    workers=max(1,min(6,os.cpu_count() or 1))
    params={
        'Decision Tree':{'max_depth':24,'min_samples_leaf':3,'class_weight':'balanced'},
        'Random Forest':{'n_estimators':100,'max_depth':28,'min_samples_leaf':2,'max_features':'sqrt','class_weight':'balanced_subsample','n_jobs':workers},
        'Logistic Regression':{'C':1.0,'class_weight':'balanced','max_iter':1600,'n_jobs':workers},
        'Extra Trees':{'n_estimators':120,'max_depth':28,'min_samples_leaf':2,'max_features':'sqrt','class_weight':'balanced','n_jobs':workers},
        'SVM (Linear)':{'estimator':'LinearSVC','C':1.0,'class_weight':'balanced','max_iter':5000,'fit_rows':'full training split'},
        'SVM (RBF)':{'estimator':'SVC','kernel':'rbf','C':1.0,'gamma':'scale','class_weight':'balanced','fit_rows_cap':12000},
        'XGBoost':{'n_estimators':180,'max_depth':7,'learning_rate':0.08,'subsample':0.85,'colsample_bytree':0.85,'tree_method':'hist','n_jobs':workers},
        'LightGBM':{'n_estimators':180,'num_leaves':31,'learning_rate':0.08,'subsample':0.85,'colsample_bytree':0.85,'class_weight':'balanced','n_jobs':workers},
        'CatBoost':{'iterations':180,'depth':7,'learning_rate':0.08,'loss_function':'MultiClass','categorical_features':'ordinal-encoded categorical columns with native CatBoost categorical indices','thread_count':workers},
        'Stacking Ensemble':{'base_estimators':['Random Forest','Extra Trees','XGBoost when installed'],'meta_estimator':'LogisticRegression(class_weight=balanced,max_iter=1200)','out_of_fold_cv':'StratifiedKFold(2 or 3 folds based on smallest class)','fit_rows_cap_per_component':8000}
    }
    return {'classifier':kind,'random_seed':int(seed),'evaluation_split':split_name,'test_used_for_tuning':False,'preprocessing_fit':'training split only','parameters':params.get(kind,{})}

def train_pipeline(dataset, kind='Random Forest', seed=42, job_id=None):
    def progress(stage, pct, message, **extra):
        if job_id:
            with LOCK:
                if JOBS[job_id].get('cancel_requested'): raise InterruptedError('Training cancelled by user')
                JOBS[job_id].update({'stage':stage,'progress':pct,'message':message,**extra,'logs':JOBS[job_id]['logs'][-100:] + [{'time':time.strftime('%H:%M:%S'),'message':message}]})
    started=time.time(); progress('loading',3,'Loading and validating dataset')
    files=dataset_files(dataset)
    train_file=next((p for p in files if 'train' in p.name.lower()),None)
    test_file=next((p for p in files if 'test' in p.name.lower() and 'test-21' not in p.name.lower()),None)
    reader=load_nsl if dataset=='NSL-KDD' else load_unsw
    if train_file and test_file:
        train_df=reader(train_file); test_df=reader(test_file)
        lc,_,y_train=prepare_labels(train_df,dataset); _,_,y_test=prepare_labels(test_df,dataset)
        X_train=feature_frame(train_df,dataset,lc); X_test=feature_frame(test_df,dataset,lc)
        X_test=X_test.reindex(columns=X_train.columns)
        valid_train=y_train.ne('Other'); valid_test=y_test.ne('Other')
        X_train=X_train.loc[valid_train].reset_index(drop=True); y_train=y_train.loc[valid_train].reset_index(drop=True)
        X_test=X_test.loc[valid_test].reset_index(drop=True); y_test=y_test.loc[valid_test].reset_index(drop=True)
        split_name='official train/test files'
    else:
        df=load_dataset(dataset); label_col,raw,y=prepare_labels(df,dataset); X=feature_frame(df,dataset,label_col)
        valid=y.ne('Other'); X=X.loc[valid].reset_index(drop=True); y=y.loc[valid].reset_index(drop=True)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=seed,stratify=y if y.value_counts().min()>=2 else None)
        split_name='stratified 75/25 split'
    if len(X_train)<20 or len(X_test)<1: raise ValueError('Not enough labelled records in the train/test data.')
    progress('preprocessing',12,'Preparing independent feature schema',records=len(X_train)+len(X_test),original_features=len(X_train.columns))
    native_cats=(kind=='CatBoost'); pre=make_preprocessor(X_train,native_cats=native_cats); Xt=pre.fit_transform(X_train); preprocess_started=time.perf_counter(); Xv=pre.transform(X_test); preprocessing_seconds=time.perf_counter()-preprocess_started
    nums=X_train.select_dtypes(include=['number','bool']).columns.tolist(); cats=[c for c in X_train.columns if c not in nums]
    cat_indices=set(range(len(nums),len(nums)+len(cats))) if native_cats else set()
    feat_names=pre.get_feature_names_out().tolist(); labels=sorted(set(y_train.unique()).union(set(y_test.unique())))
    progress('importance',23,'Computing training-only feature importance',records=len(X_train)+len(X_test),original_features=len(feat_names))
    # Hybrid, training-only ranking: ExtraTrees captures nonlinear interactions while
    # mutual information captures broader statistical dependence. Both see training rows only.
    selector_sample=stratified_sample_indices(y_train, min(12000, len(y_train)), seed+17, min_per_class=2)
    selector_X=Xt[selector_sample]; selector_y=np.asarray(y_train)[selector_sample]
    selector_tree=ExtraTreesClassifier(n_estimators=180, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', n_jobs=max(1,min(6,os.cpu_count() or 1)),
        random_state=seed+17)
    selector_tree.fit(selector_X, selector_y)
    tree_importance=np.asarray(selector_tree.feature_importances_, dtype=float)
    mi_importance=mutual_info_classif(selector_X, selector_y, discrete_features=False, random_state=seed+23)
    def _unit_scale(values):
        values=np.nan_to_num(np.asarray(values,dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        vmax=float(values.max()) if values.size else 0.0
        return values/vmax if vmax>0 else values
    importance=0.60*_unit_scale(tree_importance)+0.40*_unit_scale(mi_importance)
    ranked=np.argsort(importance)[::-1]
    k=max(1,min(len(ranked),max(5,int(len(ranked)*.35))))
    fixed_idx=ranked[:k]
    models={}; results={}; evaluation_predictions={}
    for i,(key,idx) in enumerate([('all',np.arange(Xt.shape[1])),('fixed',fixed_idx)]):
        progress(key,30+i*18, f'Training {key} feature model',selected_features=len(idx),current_model=kind)
        m=estimator_for_features(kind,seed,idx,cat_indices); train_X=model_input(Xt,idx,cat_indices,kind); test_X=model_input(Xv,idx,cat_indices,kind); t=time.perf_counter(); fitted_rows=fit_classifier(m,train_X,y_train,seed); train_s=time.perf_counter()-t
        t=time.perf_counter(); pred=m.predict(test_X); inf=time.perf_counter()-t
        results[key]=metrics(y_test,pred,labels,inf,len(idx)); results[key]['training_seconds']=train_s; results[key]['fit_records']=fitted_rows; results[key]['preprocessing_seconds']=preprocessing_seconds; results[key]['end_to_end_inference_seconds']=preprocessing_seconds+inf; results[key]['end_to_end_inference_ms_per_record']=1000*(preprocessing_seconds+inf)/max(1,len(y_test))
        evaluation_predictions[key]=np.asarray(pred, dtype=object)
        models[key]=m
    # Dynamic architecture: compact multiclass router followed by a family-vs-normal
    # specialist. A specialist only decides whether the routed family is present;
    # it does not get the test label and cannot switch to another attack family.
    router_idx=ranked[:max(1,min(len(ranked),max(3,int(len(ranked)*.20))))]
    progress('router',68,'Training dynamic family router',selected_features=len(router_idx),current_model=kind)
    router=estimator_for_features(kind,seed,router_idx,cat_indices)
    router_X=model_input(Xt,router_idx,cat_indices,kind)
    router_test_X=model_input(Xv,router_idx,cat_indices,kind)
    router_started=time.perf_counter(); router_fit_rows=fit_classifier(router,router_X,y_train,seed+101)
    router_training_seconds=time.perf_counter()-router_started
    router_inference_started=time.perf_counter()
    routed=np.asarray(router.predict(router_test_X),dtype=object).reshape(-1)
    router_inference_seconds=time.perf_counter()-router_inference_started
    experts={}; family_features={}; expert_train_start=time.perf_counter()
    families=[f for f in sorted(y_train.unique()) if f!='Normal']
    normal_mask=(y_train.to_numpy()=='Normal')
    for family_index, family in enumerate(families, start=1):
        # Train a binary specialist on only Normal + this family. Select its own
        # subset using training-only ExtraTrees/MI ranking, then fit the requested model.
        family_mask=(y_train.to_numpy()==family)
        relevant=np.flatnonzero(family_mask | normal_mask)
        binary_y=np.where(family_mask[relevant], family, 'Normal')
        local_sample=stratified_sample_indices(binary_y, min(12000,len(relevant)), seed+family_index*31, min_per_class=2)
        local_X=Xt[relevant][local_sample]; local_target=binary_y[local_sample]
        local_selector=ExtraTreesClassifier(n_estimators=120, max_depth=None, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced', n_jobs=max(1,min(6,os.cpu_count() or 1)),
            random_state=seed+family_index*31)
        local_selector.fit(local_X,local_target)
        local_tree=_unit_scale(local_selector.feature_importances_)
        local_mi=_unit_scale(mutual_info_classif(local_X,local_target,discrete_features=False,random_state=seed+family_index*37))
        local_rank=np.argsort(.60*local_tree+.40*local_mi)[::-1]
        expert_budget=max(3,int(len(local_rank)*.25))
        ei=local_rank[:min(len(local_rank),expert_budget)]
        # Balanced stratified subset caps specialist cost but preserves both classes.
        expert_idx_local=stratified_sample_indices(binary_y, min(12000,len(binary_y)), seed+family_index*43, min_per_class=2)
        expert=estimator_for_features(kind,seed+family_index,ei,cat_indices)
        expert_X=model_input(Xt[relevant][expert_idx_local],ei,cat_indices,kind)
        expert_fit_rows=fit_classifier(expert,expert_X,binary_y[expert_idx_local],seed+family_index)
        experts[family]={'model':expert,'indices':ei.tolist(),'family':family,
                         'fit_records':expert_fit_rows,'classes':['Normal',family]}
        family_features[family]=[feat_names[j] for j in ei]
        progress('experts',70+int(8*family_index/max(1,len(families))),
                 f'Trained Normal-vs-{family} specialist {family_index}/{len(families)}',
                 experts=family_index,total_experts=len(families),router_features=len(router_idx))
    expert_training_seconds=time.perf_counter()-expert_train_start
    progress('dynamic_inference',80,'Evaluating dynamic router and experts in batches',records=len(routed),experts=len(experts))
    t=time.perf_counter()
    dynamic_pred=np.empty(len(routed),dtype=object)
    dynamic_pred[routed=='Normal']='Normal'
    dynamic_counts=np.full(len(routed),len(router_idx),dtype=float)
    for family_index,(family,expert) in enumerate(experts.items(),start=1):
        family_mask=(routed==family)
        if np.any(family_mask):
            selected=expert['indices']
            dynamic_X=model_input(Xv[family_mask],selected,cat_indices,kind)
            specialist_pred=np.asarray(expert['model'].predict(dynamic_X),dtype=object).reshape(-1)
            dynamic_pred[family_mask]=specialist_pred
            dynamic_counts[family_mask]=len(router_idx)+len(selected)
        progress('dynamic_inference',80+int(7*family_index/max(1,len(experts))),
                 f'Evaluated {family} specialist',evaluated_records=int(np.count_nonzero(family_mask)),total_records=len(routed))
    # Defensive fallback: if router emits an unexpected class, preserve that prediction.
    known=np.isin(routed,np.asarray(['Normal']+families,dtype=object))
    dynamic_pred[~known]=routed[~known]
    dynamic_elapsed=time.perf_counter()-t
    results['dynamic']=metrics(y_test,dynamic_pred,labels,dynamic_elapsed+router_inference_seconds,float(np.mean(dynamic_counts))); results['dynamic']['preprocessing_seconds']=preprocessing_seconds; results['dynamic']['router_inference_seconds']=router_inference_seconds; results['dynamic']['end_to_end_inference_seconds']=preprocessing_seconds+router_inference_seconds+dynamic_elapsed; results['dynamic']['end_to_end_inference_ms_per_record']=1000*results['dynamic']['end_to_end_inference_seconds']/max(1,len(y_test))
    evaluation_predictions['dynamic']=np.asarray(dynamic_pred, dtype=object)
    results['dynamic']['training_seconds']=expert_training_seconds+router_training_seconds
    results['dynamic']['router_feature_count']=len(router_idx); results['dynamic']['experts']=len(experts); results['dynamic']['router_fit_records']=router_fit_rows; results['dynamic']['fit_records']=int(router_fit_rows+sum(e['fit_records'] for e in experts.values()))
    # Persist a small, real sample for every actual/predicted confusion-matrix cell.
    # This powers click-through inspection without storing the entire test set.
    confusion_records={}
    y_actual=np.asarray(y_test, dtype=object)
    for approach_name, predictions in evaluation_predictions.items():
        cells={}
        for row_index, (actual_label, predicted_label) in enumerate(zip(y_actual, predictions)):
            cell_key=f'{actual_label}|||{predicted_label}'
            bucket=cells.setdefault(cell_key, [])
            if len(bucket)<25:
                bucket.append({
                    'test_record': int(row_index),
                    'actual': str(actual_label),
                    'predicted': str(predicted_label),
                    'record': X_test.iloc[row_index].to_dict()
                })
        confusion_records[approach_name]=cells
    progress('evaluation',89,'Generating evaluation reports and confusion matrices')
    out=ROOT/'artifacts'/dataset.lower().replace('-','_'); out.mkdir(parents=True,exist_ok=True)
    artifact={'dataset':dataset,'kind':kind,'seed':seed,'model_config':model_configuration(kind,seed,split_name),'feature_columns':X_train.columns.tolist(),'feature_names':feat_names,'preprocessor':pre,'cat_feature_indices':sorted(cat_indices),'models':models,'fixed_indices':fixed_idx.tolist(),'router':router,'router_indices':router_idx.tolist(),'experts':experts,'family_features':family_features,'expert_training_sample_limit':12000,'dynamic_architecture':'multiclass router + family-vs-normal binary specialists','feature_ranking':'60% ExtraTrees importance + 40% mutual information; training-only','labels':labels,'results':results,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'records':len(X_train)+len(X_test),'train_records':len(X_train),'test_records':len(X_test),'split':split_name,'global_importance':{feat_names[i]:float(importance[i]) for i in ranked[:30]},'confusion_records':confusion_records}
    progress('saving',96,'Persisting fitted pipelines and experiment metadata')
    joblib.dump(artifact,out/'model.joblib',compress=3)
    with MODEL_CACHE_LOCK: MODEL_CACHE.pop(dataset,None)
    metadata={k:v for k,v in artifact.items() if k in ['dataset','kind','seed','model_config','labels','results','created_at','records','train_records','test_records','split','global_importance','family_features','dynamic_architecture','feature_ranking','feature_columns','feature_names','expert_training_sample_limit']}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=float))
    progress('complete',100,'Training complete. Models saved locally.',elapsed_seconds=round(time.time()-started,2),model_ready=True)
    return metadata

def load_model(dataset):
    p=ROOT/'artifacts'/dataset.lower().replace('-','_')/'model.joblib'
    if not p.exists(): return None
    stamp=p.stat().st_mtime_ns
    with MODEL_CACHE_LOCK:
        cached=MODEL_CACHE.get(dataset)
        if cached and cached[0]==stamp: return cached[1]
        model=joblib.load(p)
        MODEL_CACHE[dataset]=(stamp,model)
        return model

def predict(dataset, record, approach='dynamic'):
    a=load_model(dataset)
    if a is None: raise FileNotFoundError(f'No trained {dataset} model. Train it in Model Training Center first.')
    X=pd.DataFrame([{c:record.get(c) for c in a['feature_columns']}]); z=a['preprocessor'].transform(X); cat_indices=set(a.get('cat_feature_indices',[])); kind=a['kind']
    t=time.perf_counter()
    confidence=None
    if approach=='all':
        model=a['models']['all']; values=model_input(z,np.arange(z.shape[1]),cat_indices,kind); pred=model.predict(values)[0]; used=a['feature_names']
        if hasattr(model,'predict_proba'): confidence=float(np.max(model.predict_proba(z)[0]))
    elif approach=='fixed':
        model=a['models']['fixed']; values=model_input(z,a['fixed_indices'],cat_indices,kind); pred=model.predict(values)[0]; used=[a['feature_names'][i] for i in a['fixed_indices']]
        if hasattr(model,'predict_proba'): confidence=float(np.max(model.predict_proba(values)[0]))
    elif a.get('adaptive_router_mode') == 'confidence-gated fixed-to-all':
        indices=np.asarray(a['fixed_indices'],dtype=int)
        compact_model=a['models']['fixed']
        compact_values=model_input(z,indices,cat_indices,kind)
        probabilities=np.asarray(compact_model.predict_proba(compact_values)[0],dtype=float)
        confidence=float(np.max(probabilities))
        threshold=float(a.get('model_config',{}).get('dynamic_candidate',{}).get('confidence_threshold',0.0))
        if confidence >= threshold:
            pred=compact_model.classes_[int(np.argmax(probabilities))]
            used=[a['feature_names'][i] for i in indices]
        else:
            full_model=a['models']['all']
            full_values=model_input(z,np.arange(z.shape[1]),cat_indices,kind)
            pred=full_model.predict(full_values)[0]
            used=a['feature_names']
            if hasattr(full_model,'predict_proba'):
                confidence=float(np.max(full_model.predict_proba(full_values)[0]))
    else:
        rid=a['router_indices']; router_values=model_input(z,rid,cat_indices,kind)
        if a.get('adaptive_router_mode') == 'one-vs-rest top-K candidate specialists':
            router=a['router']; router_prob=np.asarray(router.predict_proba(router_values)[0],dtype=float)
            classes=np.asarray(router.classes_,dtype=object)
            dynamic_cfg=a.get('model_config',{}).get('dynamic_candidate',{})
            top_k=max(1,min(len(classes),int(dynamic_cfg.get('top_k_experts',2))))
            alpha=float(dynamic_cfg.get('expert_weight_alpha',0.5))
            candidate_ids=np.argsort(-router_prob)[:top_k]
            scores=[]; used_set=set(int(i) for i in rid)
            for ci in candidate_ids:
                family=str(classes[ci]); expert=a['experts'].get(family)
                if not expert: continue
                indices=np.asarray(expert['indices'],dtype=int)
                values=model_input(z,indices,cat_indices,kind)
                probabilities=np.asarray(expert['model'].predict_proba(values)[0],dtype=float)
                pos=int(expert.get('positive_index',int(np.flatnonzero(np.asarray(expert['model'].classes_,dtype=object)==family)[0])))
                expert_probability=max(float(probabilities[pos]),1e-12)
                score=float(np.log(max(float(router_prob[ci]),1e-12))+alpha*np.log(expert_probability))
                scores.append((family,score))
                used_set.update(int(i) for i in indices)
            if not scores:
                ci=int(np.argmax(router_prob)); pred=str(classes[ci]); confidence=float(router_prob[ci])
            else:
                pred,winning_score=max(scores,key=lambda item:item[1])
                score_values=np.asarray([item[1] for item in scores],dtype=float)
                shifted=score_values-score_values.max(); softmax=np.exp(shifted)/(np.exp(shifted).sum()+1e-12)
                confidence=float(softmax.max())
            used=[a['feature_names'][i] for i in sorted(used_set)]
        else:
            family= a['router'].predict(router_values)[0]; expert=a['experts'].get(family)
            if expert:
                model=expert['model']; values=model_input(z,expert['indices'],cat_indices,kind); pred=model.predict(values)[0]
                if hasattr(model,'predict_proba'): confidence=float(np.max(model.predict_proba(values)[0]))
                used=[a['feature_names'][i] for i in rid]+[a['feature_names'][i] for i in expert['indices']]
            else:
                pred=family; used=[a['feature_names'][i] for i in rid]
                if hasattr(a['router'],'predict_proba'): confidence=float(np.max(a['router'].predict_proba(router_values)[0]))
    elapsed=(time.perf_counter()-t)*1000
    return {'prediction':str(pred),'verdict':'Normal' if pred=='Normal' else 'Malicious','approach':approach,'selected_features':used,'inference_ms':round(elapsed,3),'confidence':confidence,'confidence_note':'Uncalibrated model probability; not a calibrated risk estimate.' if confidence is not None else None,'dataset':dataset}
