"""Isolated Accuracy Lab: research-only runs, artifacts and reports.
Does not read or write the production model cache or sentinel.db.
"""
from __future__ import annotations
import json, os, sqlite3, threading, time, uuid
from pathlib import Path
from datetime import datetime

# Keep Windows/joblib worker discovery bounded before sklearn/joblib initialize.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', os.environ.get('SENTINEL_ML_WORKERS', '2'))
try:
    MAX_WORKERS = max(1, min(4, int(os.environ.get('SENTINEL_ML_WORKERS', '2'))))
except ValueError:
    MAX_WORKERS = 2
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.base import clone
from .datasets import DATA, dataset_files, load_nsl, load_unsw, prepare_labels, feature_frame

router=APIRouter(prefix='/api/accuracy-lab', tags=['Accuracy Lab'])
ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/'accuracy_lab'
ART=STORE/'runs'
DB=STORE/'accuracy_lab.db'
LOCK=threading.RLock(); TRAINING_SEMAPHORE=threading.Lock(); RUNS={}
ART.mkdir(parents=True,exist_ok=True)

def conn():
    c=sqlite3.connect(DB); c.execute('CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,dataset TEXT,protocol TEXT,status TEXT,created_at TEXT,summary TEXT,artifact_path TEXT)'); c.commit(); return c

def persist(run):
    c=conn(); c.execute('INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?)',(run['id'],run['dataset'],run['protocol'],run['status'],run['created_at'],json.dumps(run.get('summary',{})),run.get('artifact_path'))); c.commit(); c.close()

# Recover abandoned jobs after a backend restart; never leave phantom running experiments.
_c=conn(); _c.execute("UPDATE runs SET status='failed', summary=json_set(COALESCE(summary,'{}'),'$.error','Backend restarted before the run completed') WHERE status IN ('queued','running')"); _c.commit(); _c.close()

class RunRequest(BaseModel):
    dataset:str
    protocol:str='random_holdout'
    test_size:float=Field(default=.2,ge=.1,le=.4)
    seed:int=Field(default=42,ge=0,le=1000000)
    model:str='extra_trees'
    feature_fraction:float=Field(default=.5,ge=.2,le=.9)

def load_protocol_data(dataset,protocol):
    files=dataset_files(dataset)
    if not files: raise ValueError(f'{dataset} files are unavailable')
    if protocol=='random_holdout':
        train_candidates=[p for p in files if 'train' in p.name.lower()]
        source=train_candidates[0] if train_candidates else files[0]
        frame=load_nsl(source) if dataset=='NSL-KDD' else load_unsw(source)
        col,raw,y=prepare_labels(frame,dataset); X=feature_frame(frame,dataset,col)
        return X,y,{'source_files':[source.name],'protocol_note':'Stratified random holdout from the official training file; not the official test benchmark.'}
    if protocol=='official':
        train_file=next((p for p in files if 'train' in p.name.lower()),None)
        test_file=next((p for p in files if 'test' in p.name.lower()),None)
        if not train_file or not test_file: raise ValueError('Official train/test files are required for this protocol')
        loader=load_nsl if dataset=='NSL-KDD' else load_unsw
        tr,te=loader(train_file),loader(test_file)
        c1,_,ytr=prepare_labels(tr,dataset); c2,_,yte=prepare_labels(te,dataset)
        return (feature_frame(tr,dataset,c1),ytr,feature_frame(te,dataset,c2),yte,{'source_files':[train_file.name,test_file.name],'protocol_note':'Official train/test files; test labels are used only for final evaluation.'})
    if protocol=='duplicate_aware':
        train_candidates=[p for p in files if 'train' in p.name.lower()]
        source=train_candidates[0] if train_candidates else files[0]
        frame=load_nsl(source) if dataset=='NSL-KDD' else load_unsw(source)
        col,_,y=prepare_labels(frame,dataset); X=feature_frame(frame,dataset,col)
        # Group exact duplicate feature rows so identical records cannot cross the split.
        groups=pd.util.hash_pandas_object(X.astype(str),index=False).astype(str)
        return X,y,{'source_files':[source.name],'groups':groups,'protocol_note':'Grouped holdout: exact duplicate feature rows are assigned to one side only.'}
    raise ValueError('Unsupported protocol')

def make_preprocessor(X):
    cats=X.select_dtypes(include=['object','category','bool']).columns.tolist(); nums=[c for c in X.columns if c not in cats]
    transformers=[]
    if nums: transformers.append(('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler(with_mean=False))]),nums))
    if cats: transformers.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cats))
    return ColumnTransformer(transformers,remainder='drop'),cats,nums

def metric_pack(y,pred,labels,label_names=None):
    pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=labels,average='macro',zero_division=0)
    wp,wr,wf,_=precision_recall_fscore_support(y,pred,labels=labels,average='weighted',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(pr),'macro_recall':float(re),'macro_f1':float(f1),'weighted_precision':float(wp),'weighted_recall':float(wr),'weighted_f1':float(wf),'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'labels':[str(x) for x in (label_names if label_names is not None else labels)],'per_class':classification_report(y,pred,labels=labels,target_names=[str(x) for x in (label_names if label_names is not None else labels)],output_dict=True,zero_division=0)}

def worker(run_id,req):
    TRAINING_SEMAPHORE.acquire()
    def update(**kw):
        with LOCK: RUNS[run_id].update(kw); persist(RUNS[run_id])
    try:
        update(stage='loading',progress=5,message='Loading isolated research dataset')
        data=load_protocol_data(req.dataset,req.protocol)
        if req.protocol=='official': Xtr,ytr,Xte,yte,meta=data; groups=None
        else: X,y,meta=data; groups=meta.pop('groups',None)
        update(stage='preprocessing',progress=12,message='Preparing training-only feature encodings')
        if req.protocol=='official':
            Xfit,Xval,yfit,yval=train_test_split(Xtr,ytr,test_size=.2,random_state=req.seed,stratify=ytr)
            Xtest,ytest=Xte,yte
        else:
            if groups is not None:
                # Split groups, not rows. A deterministic group split avoids duplicate leakage.
                unique=np.array(pd.unique(groups)); gtr,gte=train_test_split(unique,test_size=req.test_size,random_state=req.seed)
                mask=groups.isin(set(gtr)); Xfit,Xtest=X.loc[mask],X.loc[~mask]; yfit,ytest=y.loc[mask],y.loc[~mask]
            else: Xfit,Xtest,yfit,ytest=train_test_split(X,y,test_size=req.test_size,random_state=req.seed,stratify=y)
            Xfit,Xval,yfit,yval=train_test_split(Xfit,yfit,test_size=.2,random_state=req.seed,stratify=yfit)
        # Bound research compute so interactive lab runs finish on ordinary student hardware.
        # Sampling is training-only and stratified; validation/test partitions remain untouched.
        if len(Xfit)>30000:
            Xfit,_,yfit,_=train_test_split(Xfit,yfit,train_size=30000,random_state=req.seed,stratify=yfit)
        # Quantify exact feature-row overlap across the final train/test boundary.
        train_hash=set(pd.util.hash_pandas_object(Xfit.astype(str),index=False).astype(str).tolist())
        test_hash=pd.util.hash_pandas_object(Xtest.astype(str),index=False).astype(str)
        overlap_count=int(test_hash.isin(train_hash).sum())
        overlap_fraction=float(overlap_count/max(1,len(Xtest)))
        enc=LabelEncoder(); yfit_e=enc.fit_transform(yfit.astype(str)); yval_e=enc.transform(yval.astype(str)); ytest_e=enc.transform(ytest.astype(str))
        pre,cats,nums=make_preprocessor(Xfit)
        Xt=pre.fit_transform(Xfit); Xv=pre.transform(Xval); Xte=pre.transform(Xtest)
        names=pre.get_feature_names_out()
        update(stage='feature_ranking',progress=28,message='Computing mutual information, correlation and tree importance')
        # Train-only rank fusion. MI + ANOVA + chi-square + ExtraTrees importance.
        mi_sample=min(12000, Xt.shape[0])
        if Xt.shape[0]>mi_sample:
            rng=np.random.default_rng(req.seed); mi_idx=rng.choice(Xt.shape[0],size=mi_sample,replace=False)
            mi=mutual_info_classif(Xt[mi_idx],yfit_e[mi_idx],random_state=req.seed,n_neighbors=3)
        else:
            mi=mutual_info_classif(Xt,yfit_e,random_state=req.seed,n_neighbors=3)
        # ANOVA F-scores are undefined for constant columns (and can emit
        # divide-by-zero warnings for degenerate within-class variance).
        # Exclude globally constant columns before calling sklearn, then map
        # valid scores back to the full encoded feature vector. Undefined
        # scores are assigned zero so they cannot influence hybrid ranking.
        anova=np.zeros(Xt.shape[1],dtype=float)
        try:
            anova_mean=np.asarray(Xt.mean(axis=0)).ravel()
            anova_sq=(np.asarray(Xt.power(2).mean(axis=0)).ravel()
                      if sparse.issparse(Xt) else np.mean(np.asarray(Xt)**2,axis=0))
            anova_var=np.maximum(anova_sq-anova_mean*anova_mean,0.0)
            nonconstant=np.flatnonzero(anova_var>1e-12)
            if nonconstant.size:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',category=RuntimeWarning,
                                            message='invalid value encountered in divide')
                    warnings.filterwarnings('ignore',category=RuntimeWarning,
                                            message='divide by zero encountered')
                    scores=f_classif(Xt[:,nonconstant],yfit_e)[0]
                anova[nonconstant]=np.nan_to_num(scores,nan=0.0,posinf=0.0,neginf=0.0)
        except Exception:
            anova=np.zeros(Xt.shape[1],dtype=float)
        try: chi=np.nan_to_num(chi2(abs(Xt),yfit_e)[0],nan=0,posinf=0,neginf=0)
        except Exception: chi=np.zeros(Xt.shape[1])
        # One-vs-rest point-biserial/Pearson correlation for each encoded feature.
        corr=np.zeros(Xt.shape[1]); x_mean=np.asarray(Xt.mean(axis=0)).ravel(); x_sq=np.asarray(Xt.power(2).mean(axis=0)).ravel() if sparse.issparse(Xt) else np.mean(Xt**2,axis=0); x_std=np.sqrt(np.maximum(x_sq-x_mean*x_mean,1e-12))
        for cls in np.unique(yfit_e):
            target=(yfit_e==cls).astype(float); tstd=max(float(target.std()),1e-12)
            cov=np.asarray(Xt.T @ (target-target.mean())).ravel()/Xt.shape[0]
            corr=np.maximum(corr,np.abs(cov/(x_std*tstd)))
        # Gain Ratio approximation: mutual information divided by entropy of binned feature values.
        gain_ratio=np.zeros(Xt.shape[1])
        for j in range(Xt.shape[1]):
            col=np.asarray(Xt[:,j].toarray()).ravel() if sparse.issparse(Xt) else np.asarray(Xt[:,j]).ravel()
            if np.unique(col).size>12:
                edges=np.unique(np.quantile(col,np.linspace(0,1,11))); bins=np.digitize(col,edges[1:-1])
            else: bins=col
            counts=np.unique(bins,return_counts=True)[1].astype(float); probs=counts/counts.sum(); entropy=float(-(probs*np.log2(np.maximum(probs,1e-12))).sum())
            gain_ratio[j]=mi[j]/max(entropy,1e-12)
        tree_model=DecisionTreeClassifier(max_depth=14,min_samples_leaf=3,class_weight='balanced',random_state=req.seed).fit(Xt,yfit_e)
        tree=tree_model.feature_importances_
        ranker=ExtraTreesClassifier(n_estimators=48,min_samples_leaf=2,class_weight='balanced',n_jobs=MAX_WORKERS,random_state=req.seed)
        ranker.fit(Xt,yfit_e); forest=ranker.feature_importances_
        def normrank(a):
            order=np.argsort(np.argsort(-np.nan_to_num(a))); return 1-order/max(1,len(a)-1)
        ranks={'gain_ratio':normrank(gain_ratio),'mutual_information':normrank(mi),'pearson_correlation':normrank(corr),'anova_f_score':normrank(anova),'chi_square':normrank(chi),'decision_tree_importance':normrank(tree),'extra_trees_importance':normrank(forest)}
        hybrid=np.mean(np.vstack(list(ranks.values())),axis=0)
        k=max(2,min(Xt.shape[1],int(np.ceil(Xt.shape[1]*req.feature_fraction))))
        selected=np.argsort(-hybrid)[:k]
        update(stage='training',progress=42,message='Training All Features, Fixed Hybrid and Dynamic selector models')
        def model():
            if req.model=='random_forest': return RandomForestClassifier(n_estimators=220,min_samples_leaf=1,class_weight='balanced_subsample',n_jobs=MAX_WORKERS,random_state=req.seed)
            return ExtraTreesClassifier(n_estimators=120,min_samples_leaf=1,class_weight='balanced',n_jobs=MAX_WORKERS,random_state=req.seed)
        all_model=model(); fixed_model=model()
        all_model.fit(Xt,yfit_e); fixed_model.fit(Xt[:,selected],yfit_e)
        all_val=all_model.predict_proba(Xv); fixed_val=fixed_model.predict_proba(Xv[:,selected])
        # Adaptive acquisition: inspect the compact feature subset first; acquire the
        # remaining features and invoke the full model only when compact confidence is low.
        # Threshold is selected on validation data only, prioritising accuracy and then
        # lower mean feature acquisition when validation accuracy ties.
        conf=fixed_val.max(axis=1)
        thresholds=np.unique(np.r_[-1.0,np.quantile(conf,np.linspace(0,.95,24)),np.inf])
        best=(-1.0,float('inf'),-1.0)
        for threshold_candidate in thresholds:
            use_fixed=conf>=threshold_candidate
            routed_pred=fixed_val.argmax(axis=1).copy()
            routed_pred[~use_fixed]=all_val[~use_fixed].argmax(axis=1)
            score=float(accuracy_score(yval_e,routed_pred))
            mean_features=float(k+(~use_fixed).mean()*(Xt.shape[1]-k))
            if score>best[0]+1e-12 or (abs(score-best[0])<=1e-12 and mean_features<best[1]):
                best=(score,mean_features,float(threshold_candidate))
        validation_dynamic_accuracy,validation_dynamic_features,threshold=best
        update(stage='evaluation',progress=78,message='Measuring adaptive feature acquisition and held-out inference cost')
        import time as _time
        all_wall=_time.perf_counter(); all_cpu=_time.process_time(); all_pred=all_model.predict(Xte); all_cpu_s=_time.process_time()-all_cpu; all_wall_s=_time.perf_counter()-all_wall
        fixed_wall=_time.perf_counter(); fixed_cpu=_time.process_time(); fixed_pred=fixed_model.predict(Xte[:,selected]); fixed_cpu_s=_time.process_time()-fixed_cpu; fixed_wall_s=_time.perf_counter()-fixed_wall
        fixed_wall_all=_time.perf_counter(); fixed_cpu_all=_time.process_time(); fixed_prob=fixed_model.predict_proba(Xte[:,selected]); fixed_acquisition_cpu=_time.process_time()-fixed_cpu_all; fixed_acquisition_wall=_time.perf_counter()-fixed_wall_all
        fixed_conf=fixed_prob.max(axis=1); use_fixed=fixed_conf>=threshold
        dynamic_pred=fixed_prob.argmax(axis=1).copy()
        fallback_idx=np.flatnonzero(~use_fixed)
        dynamic_wall=_time.perf_counter(); dynamic_cpu=_time.process_time()
        if len(fallback_idx):
            fallback_pred=all_model.predict(Xte[fallback_idx])
            dynamic_pred[fallback_idx]=fallback_pred
        dynamic_cpu_s=fixed_acquisition_cpu+(_time.process_time()-dynamic_cpu); dynamic_wall_s=_time.perf_counter()-dynamic_wall+fixed_acquisition_wall
        labels=list(range(len(enc.classes_))); label_names=list(enc.classes_)
        results={'all_features':metric_pack(ytest_e,all_pred,labels,label_names),'fixed_features':metric_pack(ytest_e,fixed_pred,labels,label_names),'dynamic_features':metric_pack(ytest_e,dynamic_pred,labels,label_names)}
        summary={'dataset':req.dataset,'protocol':req.protocol,'seed':req.seed,'model':req.model,'train_records':len(Xfit),'validation_records':len(Xval),'test_records':len(Xtest),'original_feature_count':int(Xfit.shape[1]),'encoded_feature_count':int(Xt.shape[1]),'selected_feature_count':int(k),'selected_feature_fraction':float(k/Xt.shape[1]),'exact_duplicate_overlap_test_rows':overlap_count,'exact_duplicate_overlap_test_fraction':overlap_fraction,'selected_features':[str(names[i]) for i in selected],'feature_rankings':{key:[{'feature':str(names[i]),'score':float(v[i])} for i in np.argsort(-v)[:min(30,len(v))]] for key,v in ranks.items()},'hybrid_top_features':[{'feature':str(names[i]),'score':float(hybrid[i])} for i in np.argsort(-hybrid)[:min(40,len(hybrid))]],'dynamic_confidence_threshold':threshold,'dynamic_meta_router_used':False,'validation_dynamic_accuracy':validation_dynamic_accuracy,'results':results,'efficiency':{'all_features_per_record':int(Xt.shape[1]),'fixed_features_per_record':int(k),'dynamic_initial_features_per_record':int(k),'dynamic_fallback_rate':float((~use_fixed).mean()),'dynamic_average_features_per_record':float(k+(~use_fixed).mean()*(Xt.shape[1]-k)),'dynamic_feature_reduction_vs_all_pct':float((1-(k+(~use_fixed).mean()*(Xt.shape[1]-k))/max(1,Xt.shape[1]))*100),'dynamic_feature_reduction_vs_fixed_pct':float((1-(k+(~use_fixed).mean()*(Xt.shape[1]-k))/max(1,k))*100),'all_inference_wall_seconds':float(all_wall_s),'fixed_inference_wall_seconds':float(fixed_wall_s),'dynamic_inference_wall_seconds':float(dynamic_wall_s),'all_inference_cpu_seconds':float(all_cpu_s),'fixed_inference_cpu_seconds':float(fixed_cpu_s),'dynamic_inference_cpu_seconds':float(dynamic_cpu_s),'dynamic_fallback_records':int(len(fallback_idx)),'test_records_timed':int(len(Xtest)),'timing_scope':'Model inference only; excludes dataset loading, preprocessing and feature ranking.'},'protocol_note':meta['protocol_note'],'source_files':meta['source_files'],'training_cap':30000,'dynamic_policy':'Adaptive feature acquisition: first evaluate the compact selected subset; if confidence is below the validation-tuned threshold, acquire remaining features and invoke the full-feature model. The threshold is selected on validation data, prioritising accuracy and then lower average feature use. Test labels are used only for final metrics.'}
        folder=ART/run_id; folder.mkdir(parents=True,exist_ok=True)
        import joblib
        joblib.dump({'preprocessor':pre,'all_model':all_model,'fixed_model':fixed_model,'selected_indices':selected,'label_encoder':enc,'threshold':threshold},folder/'models.joblib',compress=3)
        (folder/'results.json').write_text(json.dumps(summary,indent=2,default=float))
        update(status='completed',stage='completed',progress=100,message='Research experiment completed',summary=summary,artifact_path=str(folder),finished_at=time.time())
    except Exception as exc:
        update(status='failed',stage='failed',progress=100,message=str(exc),error=str(exc),finished_at=time.time())
    finally:
        TRAINING_SEMAPHORE.release()

@router.post('/runs')
def start_run(req:RunRequest,background:BackgroundTasks):
    if req.dataset not in ('NSL-KDD','UNSW-NB15'): raise HTTPException(400,'Unsupported dataset')
    if req.protocol not in ('random_holdout','duplicate_aware','official'): raise HTTPException(400,'Unsupported protocol')
    rid=str(uuid.uuid4()); now=datetime.now().isoformat(timespec='seconds')
    run={'id':rid,'dataset':req.dataset,'protocol':req.protocol,'status':'queued','stage':'queued','progress':0,'message':'Queued for isolated research execution','created_at':now,'summary':{}}
    with LOCK: RUNS[rid]=run; persist(run)
    background.add_task(worker,rid,req)
    return {'run_id':rid,'status':'queued'}

@router.get('/runs')
def list_runs():
    c=conn(); rows=c.execute('SELECT id,dataset,protocol,status,created_at,summary FROM runs ORDER BY created_at DESC LIMIT 100').fetchall(); c.close()
    return [{'id':r[0],'dataset':r[1],'protocol':r[2],'status':r[3],'created_at':r[4],'summary':json.loads(r[5] or '{}')} for r in rows]

@router.get('/runs/{run_id}')
def get_run(run_id:str):
    with LOCK:
        if run_id in RUNS: return RUNS[run_id]
    c=conn(); row=c.execute('SELECT id,dataset,protocol,status,created_at,summary,artifact_path FROM runs WHERE id=?',(run_id,)).fetchone(); c.close()
    if not row: raise HTTPException(404,'Accuracy Lab run not found')
    return {'id':row[0],'dataset':row[1],'protocol':row[2],'status':row[3],'created_at':row[4],'summary':json.loads(row[5] or '{}'),'artifact_path':row[6]}

@router.get('/runs/{run_id}/report.json')
def report_json(run_id:str):
    r=get_run(run_id)
    if r['status']!='completed': raise HTTPException(409,'Research run is not complete')
    return r['summary']

@router.get('/runs/{run_id}/report.csv')
def report_csv(run_id:str):
    r=get_run(run_id)
    if r['status']!='completed': raise HTTPException(409,'Research run is not complete')
    s=r['summary']; lines=['approach,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1,avg_features_per_record,inference_wall_seconds,inference_cpu_seconds']
    for key,title in [('all_features','All Features'),('fixed_features','Fixed Features'),('dynamic_features','Dynamic Features')]:
        m=s['results'][key]; n=s['encoded_feature_count'] if key=='all_features' else (s['selected_feature_count'] if key=='fixed_features' else 'adaptive')
        e=s.get('efficiency',{}); ek={'All Features':'all','Fixed Features':'fixed','Dynamic Features':'dynamic'}[title]; fc={'all':e.get('all_features_per_record'),'fixed':e.get('fixed_features_per_record'),'dynamic':e.get('dynamic_average_features_per_record')}[ek]; wall=e.get(f'{ek}_inference_wall_seconds',''); cpu=e.get(f'{ek}_inference_cpu_seconds',''); lines.append(','.join(map(str,[title,m['accuracy'],m['macro_precision'],m['macro_recall'],m['macro_f1'],m['weighted_f1'],fc,wall,cpu])))
    return Response('\n'.join(lines),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename=sentinel-accuracy-lab-{run_id[:8]}.csv'})

@router.get('/runs/{run_id}/report.pdf')
def report_pdf(run_id:str):
    r=get_run(run_id)
    if r['status']!='completed': raise HTTPException(409,'Research run is not complete')
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
    s=r['summary']; buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=landscape(A4),rightMargin=17*mm,leftMargin=17*mm,topMargin=17*mm,bottomMargin=17*mm,title='SENTINEL Accuracy Lab Research Report')
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='SentinelTitle',parent=styles['Title'],textColor=colors.HexColor('#d9652b'),fontSize=21,leading=26,spaceAfter=8)); styles.add(ParagraphStyle(name='SmallNote',parent=styles['BodyText'],fontSize=8,leading=11,textColor=colors.HexColor('#555555')))
    story=[Paragraph('SENTINEL · ACCURACY LAB',styles['SentinelTitle']),Paragraph('Research Experiment Report',styles['Heading2']),Paragraph(f"Dataset: {s['dataset']} &nbsp;&nbsp; Protocol: {s['protocol']} &nbsp;&nbsp; Seed: {s['seed']}",styles['BodyText']),Spacer(1,8),Paragraph(s['protocol_note'],styles['SmallNote']),Spacer(1,12),Paragraph('Experiment configuration',styles['Heading2'])]
    config=[['Model',s['model']],['Training rows',f"{s['train_records']:,}"],['Validation rows',f"{s['validation_records']:,}"],['Test rows',f"{s['test_records']:,}"],['Original features',str(s['original_feature_count'])],['Encoded features',str(s['encoded_feature_count'])],['Fixed subset size',str(s['selected_feature_count'])]]
    t=Table(config,colWidths=[48*mm,110*mm],repeatRows=0); t.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#f4eee9')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d8d0c9')),('FONTSIZE',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); story += [t,Spacer(1,12),Paragraph('All vs Fixed vs Dynamic',styles['Heading2'])]
    rows=[['Approach','Accuracy','Macro-F1','Weighted-F1','Precision','Recall'],]
    for key,title in [('all_features','All Features'),('fixed_features','Fixed Features'),('dynamic_features','Dynamic Features')]:
        m=s['results'][key]; rows.append([title,*[f"{m[k]*100:.2f}%" for k in ['accuracy','macro_f1','weighted_f1','macro_precision','macro_recall']]])
    t=Table(rows,colWidths=[39*mm,23*mm,23*mm,27*mm,25*mm,22*mm],repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#28231f')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d8d0c9')),('FONTSIZE',(0,0),(-1,-1),7.5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8f5f2')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story += [t,Spacer(1,12),Paragraph('Computational efficiency and feature acquisition',styles['Heading2'])]
    e=s.get('efficiency',{})
    erows=[['Measure','All Features','Fixed Features','Dynamic Features'],['Average features / record',str(e.get('all_features_per_record','—')),str(e.get('fixed_features_per_record','—')),f"{e.get('dynamic_average_features_per_record',0):.2f}"],['Inference wall time (s)',f"{e.get('all_inference_wall_seconds',0):.4f}",f"{e.get('fixed_inference_wall_seconds',0):.4f}",f"{e.get('dynamic_inference_wall_seconds',0):.4f}"],['Inference CPU time (s)',f"{e.get('all_inference_cpu_seconds',0):.4f}",f"{e.get('fixed_inference_cpu_seconds',0):.4f}",f"{e.get('dynamic_inference_cpu_seconds',0):.4f}"]]
    et=Table(erows,colWidths=[55*mm,42*mm,42*mm,42*mm],repeatRows=1); et.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#28231f')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d8d0c9')),('FONTSIZE',(0,0),(-1,-1),7.5),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story += [et,Spacer(1,5),Paragraph(f"Dynamic fallback rate: {e.get('dynamic_fallback_rate',0)*100:.2f}% · Feature reduction vs All: {e.get('dynamic_feature_reduction_vs_all_pct',0):.2f}% · Fallback records: {e.get('dynamic_fallback_records',0):,}. CPU and wall times are measured for this run and depend on hardware and system load. Feature counts refer to encoded model inputs; feature extraction and routing overhead are not fully represented by feature count alone. Timings cover model inference only, excluding data loading, preprocessing and feature ranking.",styles['SmallNote']),Spacer(1,10),Paragraph('Dynamic routing policy',styles['Heading2']),Paragraph(s['dynamic_policy'],styles['BodyText']),Spacer(1,8),Paragraph('Selected fixed features',styles['Heading2'])]
    story.append(Paragraph(', '.join(s['selected_features']),styles['SmallNote']))
    story += [Spacer(1,8),Paragraph('Top hybrid-ranked features',styles['Heading2']),Paragraph(', '.join(x['feature'] for x in s['hybrid_top_features'][:20]),styles['SmallNote'])]
    story += [PageBreak(),Paragraph('Confusion matrices',styles['Heading1'])]
    for matrix_index,(key,title) in enumerate([('all_features','All Features'),('fixed_features','Fixed Features'),('dynamic_features','Dynamic Features')]):
        if matrix_index: story.append(PageBreak())
        m=s['results'][key]; labels=m['labels']; matrix=m['confusion_matrix']; cells=[['Actual \\ Predicted']+labels]
        for i,label in enumerate(labels): cells.append([label]+[str(v) for v in matrix[i]])
        story += [Paragraph(title,styles['Heading2'])]
        mt=Table(cells,repeatRows=1,hAlign='LEFT'); mt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#28231f')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d8d0c9')),('FONTSIZE',(0,0),(-1,-1),7),('ALIGN',(1,1),(-1,-1),'RIGHT'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)])); story += [mt,Spacer(1,10)]
    story += [Paragraph('Methodology and limitations',styles['Heading2']),Paragraph(f"Feature ranking and preprocessing were fitted using training rows only. The confidence threshold was selected using validation labels. The held-out test labels were used only for final evaluation. Exact duplicate feature-row overlap in the test partition: {s.get('exact_duplicate_overlap_test_rows',0):,} rows ({s.get('exact_duplicate_overlap_test_fraction',0)*100:.2f}%). Random holdout results are not equivalent to official benchmark results. Duplicate-aware protocol groups exact duplicate feature rows. Dynamic routing uses a validation-tuned confidence gate and may fall back to the all-feature model.",styles['SmallNote'])]
    doc.build(story); return Response(buf.getvalue(),media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename=sentinel-accuracy-lab-{run_id[:8]}-report.pdf'})
