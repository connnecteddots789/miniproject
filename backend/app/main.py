from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import Response
from pathlib import Path
import uuid, json, time, shutil, sqlite3, pandas as pd, threading, subprocess, sys
from .datasets import DATA, dataset_files, status, load_dataset, prepare_labels, feature_frame, load_nsl, load_unsw, NSL_ATTACKS, UNSW_ATTACKS
from .ml import JOBS, LOCK, train_pipeline, load_model, predict, MODEL_CACHE, MODEL_CACHE_LOCK
from .accuracy_lab import router as accuracy_lab_router, TRAINING_SEMAPHORE

app=FastAPI(title='SENTINEL API',version='1.0.0',description='Adaptive Network Intelligence research prototype')
app.include_router(accuracy_lab_router)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_origin_regex=r'https?://(localhost|127\.0\.0\.1)(:\d+)?',allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
DB=Path(__file__).resolve().parents[1]/'sentinel.db'
SIMULATIONS={}; SIM_LOCK=threading.Lock()
def db():
    c=sqlite3.connect(DB); c.execute('CREATE TABLE IF NOT EXISTS detections(id INTEGER PRIMARY KEY, dataset TEXT, prediction TEXT, verdict TEXT, approach TEXT, inference_ms REAL, created_at TEXT, record TEXT)');
    columns={r[1] for r in c.execute('PRAGMA table_info(detections)').fetchall()}
    if 'selected_features' not in columns: c.execute('ALTER TABLE detections ADD COLUMN selected_features TEXT')
    if 'model_type' not in columns: c.execute('ALTER TABLE detections ADD COLUMN model_type TEXT')
    if 'confidence' not in columns: c.execute('ALTER TABLE detections ADD COLUMN confidence REAL')
    c.execute('CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY, dataset TEXT, model_type TEXT, created_at TEXT, metadata TEXT)'); c.commit(); return c
class TrainRequest(BaseModel): dataset:str; model_type:str='Random Forest'; seed:int=42
class PredictRequest(BaseModel): dataset:str; record:dict; approach:str='dynamic'

@app.on_event('startup')
def startup(): db().close()
@app.get('/api/health')
def health(): return {'status':'ok','service':'SENTINEL API','timestamp':time.time()}
@app.get('/api/datasets')
def datasets():
    result=[]
    for n in ['NSL-KDD','UNSW-NB15']:
        s=status(n); m=load_model(n); result.append({**s,'trained':m is not None,'labels':m['labels'] if m else []})
    return result
@app.get('/api/inspect/{dataset}')
def inspect_dataset(dataset:str):
    if dataset not in ['NSL-KDD','UNSW-NB15']: raise HTTPException(400,'Invalid dataset')
    if not dataset_files(dataset): raise HTTPException(404,'Dataset files are unavailable')
    try:
        from .datasets import target_column, prepare_labels, feature_frame
        frame=load_dataset(dataset); label_col,raw,labels=prepare_labels(frame,dataset); features=feature_frame(frame,dataset,label_col)
        missing=frame.isna().sum().sort_values(ascending=False).head(12)
        file_counts=[]
        for source_file in dataset_files(dataset):
            source_frame=load_nsl(source_file) if dataset=='NSL-KDD' else load_unsw(source_file)
            lower=source_file.name.lower()
            role='Training' if 'train' in lower else ('Testing' if 'test' in lower else 'Source file')
            file_counts.append({'file':source_file.name,'role':role,'records':len(source_frame)})
        numeric=frame.select_dtypes(include='number').describe().to_dict()
        numeric_summary={str(col):{str(stat):(None if pd.isna(value) else float(value)) for stat,value in stats.items()} for col,stats in numeric.items()}
        return {'dataset':dataset,'records':len(frame),'feature_count':len(features.columns),'columns':[{'name':c,'dtype':str(frame[c].dtype),'missing':int(frame[c].isna().sum())} for c in frame.columns],'missing_top':{str(k):int(v) for k,v in missing.items()},'class_distribution':labels.value_counts().to_dict(),'numeric_summary':numeric_summary,'file_record_counts':file_counts,'preview':frame.head(5).fillna('').to_dict(orient='records')}
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/api/inspect/{dataset}/records')
def inspect_dataset_records(dataset:str, page:int=1, page_size:int=25, q:str=''):
    """Paginated inspection of original bundled dataset rows; labels remain ground truth."""
    if dataset not in ['NSL-KDD','UNSW-NB15']:
        raise HTTPException(400,'Invalid dataset')
    if not dataset_files(dataset):
        raise HTTPException(404,'Dataset files are unavailable')
    page=max(1,page); page_size=max(5,min(page_size,50))
    try:
        frame=load_dataset(dataset)
        label_col, raw_labels, mapped_labels=prepare_labels(frame,dataset)
        total=len(frame)
        filtered=frame
        query=q.strip()
        if query:
            mask=pd.Series(False,index=frame.index)
            for column in frame.columns:
                mask=mask | frame[column].astype(str).str.contains(query,case=False,regex=False,na=False)
            filtered=frame.loc[mask]
        filtered_total=len(filtered)
        start=(page-1)*page_size
        display_columns=[c for c in frame.columns if c not in (label_col,'difficulty')][:8]
        rows=[]
        for idx,row in filtered.iloc[start:start+page_size].iterrows():
            values={str(c):(None if pd.isna(row[c]) else (row[c].item() if hasattr(row[c],'item') else row[c])) for c in display_columns}
            rows.append({'record_number':int(idx)+1,'source':'Bundled dataset','ground_truth':str(mapped_labels.loc[idx]),'original_label':str(raw_labels.loc[idx]),'values':values})
        return {'dataset':dataset,'page':page,'page_size':page_size,'total_records':total,'filtered_records':filtered_total,'total_pages':max(1,(filtered_total+page_size-1)//page_size),'columns':display_columns,'rows':rows,'search':query,'label_taxonomy':'NSL-KDD five-family mapping' if dataset=='NSL-KDD' else ('UNSW-NB15 binary label mapping' if label_col=='label' else 'UNSW-NB15 attack_cat taxonomy')}
    except Exception as exc:
        raise HTTPException(400,f'Dataset record inspection failed: {exc}')

@app.post('/api/train')
def train(req:TrainRequest, background:BackgroundTasks):
    if req.dataset not in ['NSL-KDD','UNSW-NB15']: raise HTTPException(400,'Unsupported dataset')
    if req.model_type not in ['Decision Tree','Random Forest','Extra Trees','Logistic Regression','SVM','SVM (Linear)','SVM (RBF)','XGBoost','LightGBM','CatBoost','Stacking Ensemble']: raise HTTPException(400,'Unsupported model')
    if not dataset_files(req.dataset): raise HTTPException(400,f'{req.dataset} dataset unavailable. Run scripts/setup_datasets.py and review its output.')
    jid=str(uuid.uuid4())
    with LOCK: JOBS[jid]={'id':jid,'status':'running','stage':'queued','progress':0,'message':'Queued','logs':[],'started':time.time()}
    def worker():
        try:
            with TRAINING_SEMAPHORE:
                metadata=train_pipeline(req.dataset,req.model_type,req.seed,jid)
            c=db(); c.execute('INSERT INTO experiments(dataset,model_type,created_at,metadata) VALUES(?,?,?,?)',(req.dataset,req.model_type,metadata.get('created_at'),json.dumps(metadata,default=float))); c.commit(); c.close()
            with LOCK: JOBS[jid]['status']='completed'
        except Exception as e:
            with LOCK:
                cancelled=JOBS[jid].get('cancel_requested',False)
                JOBS[jid].update({'status':'cancelled' if cancelled else 'failed','error':str(e),'message':str(e),'stage':'cancelled' if cancelled else 'failed','logs':JOBS[jid]['logs']+[{'time':time.strftime('%H:%M:%S'),'message':('CANCELLED: ' if cancelled else 'ERROR: ')+str(e)}]})
    background.add_task(worker)
    return {'job_id':jid,'status':'running'}
@app.get('/api/train/{job_id}')
def job(job_id:str):
    with LOCK:
        if job_id not in JOBS: raise HTTPException(404,'Training job not found')
        return dict(JOBS[job_id])
@app.post('/api/train/{job_id}/cancel')
def cancel_training(job_id:str):
    with LOCK:
        if job_id not in JOBS: raise HTTPException(404,'Training job not found')
        if JOBS[job_id]['status']!='running': raise HTTPException(409,'Training job is not running')
        JOBS[job_id]['cancel_requested']=True
        return {'job_id':job_id,'cancel_requested':True,'message':'Cancellation will be applied at the next training stage boundary.'}

@app.post('/api/load/{dataset}')
def load_saved_model(dataset:str):
    if dataset not in ['NSL-KDD','UNSW-NB15']: raise HTTPException(400,'Invalid dataset')
    model=load_model(dataset)
    if model is None: raise HTTPException(404,f'No saved model for {dataset}')
    return {'dataset':dataset,'loaded':True,'model_type':model['kind'],'created_at':model['created_at'],'labels':model['labels']}

@app.get('/api/experiments/{dataset}')
def experiments(dataset:str):
    c=db(); rows=c.execute('SELECT id,dataset,model_type,created_at,metadata FROM experiments WHERE dataset=? ORDER BY id DESC LIMIT 50',(dataset,)).fetchall(); c.close()
    return [{'id':r[0],'dataset':r[1],'model_type':r[2],'created_at':r[3],'metadata':json.loads(r[4])} for r in rows]

@app.get('/api/experiments')
def all_experiments():
    c=db(); rows=c.execute('SELECT id,dataset,model_type,created_at,metadata FROM experiments ORDER BY id DESC LIMIT 100').fetchall(); c.close()
    return [{'id':r[0],'dataset':r[1],'model_type':r[2],'created_at':r[3],'metadata':json.loads(r[4])} for r in rows]

@app.delete('/api/experiments/{experiment_id}')
def delete_experiment(experiment_id:int):
    c=db(); cur=c.execute('DELETE FROM experiments WHERE id=?',(experiment_id,)); c.commit(); deleted=cur.rowcount; c.close()
    if not deleted: raise HTTPException(404,'Experiment not found')
    return {'deleted':True,'experiment_id':experiment_id}

@app.get('/api/research/binary/unsw')
def binary_research_validation():
    report_path=Path(__file__).resolve().parents[1]/'artifacts'/'research'/'unsw_nb15_binary_lightgbm'/'metrics.json'
    if not report_path.exists(): raise HTTPException(404,'Binary research validation has not been run. Execute scripts/validate_unsw_binary.py.')
    return json.loads(report_path.read_text())

@app.post('/api/research/binary/unsw/run')
def run_binary_research(background:BackgroundTasks):
    """Run the separate UNSW-NB15 binary validation only on explicit user request."""
    jid='binary-'+str(uuid.uuid4())
    with LOCK:
        JOBS[jid]={'id':jid,'status':'running','stage':'preprocessing','progress':None,
                   'message':'Running the UNSW-NB15 binary research protocol','logs':[],
                   'started':time.time(),'research_task':True}
    def worker():
        project_root=Path(__file__).resolve().parents[2]
        script=project_root/'scripts'/'validate_unsw_binary.py'
        try:
            process=subprocess.Popen([sys.executable,str(script)],cwd=str(project_root),
                                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            while process.poll() is None:
                time.sleep(.5)
                with LOCK:
                    cancel=JOBS.get(jid,{}).get('cancel_requested',False)
                if cancel:
                    process.terminate()
                    try: process.wait(timeout=8)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    with LOCK:
                        JOBS[jid].update({'status':'cancelled','stage':'cancelled','message':'Binary training cancelled by user','finished':time.time()})
                    return
            stdout,stderr=process.communicate()
            if process.returncode!=0:
                detail=(stderr or stdout or 'Binary validation failed')[-5000:]
                raise RuntimeError(detail)
            report_path=Path(__file__).resolve().parents[1]/'artifacts'/'research'/'unsw_nb15_binary_lightgbm'/'metrics.json'
            report=json.loads(report_path.read_text())
            with LOCK:
                JOBS[jid].update({'status':'completed','stage':'completed','progress':100,
                                  'message':'Official held-out binary evaluation completed',
                                  'finished':time.time(),'test_accuracy':report['test_metrics']['accuracy']})
        except Exception as exc:
            with LOCK:
                JOBS[jid].update({'status':'failed','stage':'failed','message':str(exc),
                                  'error':str(exc),'finished':time.time()})
    background.add_task(worker)
    return {'job_id':jid,'status':'running','task':'UNSW-NB15 binary validation'}

@app.get('/api/benchmark/{dataset}')
def benchmark(dataset:str):
    m=load_model(dataset)
    if not m: raise HTTPException(404,'No saved model for this dataset')
    return {'dataset':dataset,'results':m['results'],'created_at':m['created_at'],'model_type':m['kind'],'records':m['records'],'train_records':m['train_records'],'test_records':m['test_records']}
@app.get('/api/benchmark/{dataset}/records')
def benchmark_cell_records(dataset:str, approach:str, actual:str, predicted:str, limit:int=25):
    if dataset not in ['NSL-KDD','UNSW-NB15']: raise HTTPException(400,'Invalid dataset')
    if approach not in ['all','fixed','dynamic']: raise HTTPException(400,'Invalid approach')
    model=load_model(dataset)
    if not model: raise HTTPException(404,'No saved model for this dataset')
    cells=model.get('confusion_records',{}).get(approach,{})
    key=f'{actual}|||{predicted}'
    rows=cells.get(key,[])
    return {'dataset':dataset,'approach':approach,'actual':actual,'predicted':predicted,'count':len(rows),'samples_available':bool(cells),'records':rows[:max(1,min(limit,100))]}

@app.get('/api/report/{dataset}')
def experiment_report(dataset:str):
    m=load_model(dataset)
    if not m: raise HTTPException(404,'Train a model first')
    return {'project':'SENTINEL — Adaptive Network Intelligence','dataset':dataset,'model_type':m['kind'],'seed':m['seed'],'created_at':m['created_at'],'split':m.get('split'),'records':m['records'],'train_records':m['train_records'],'test_records':m['test_records'],'labels':m['labels'],'results':m['results'],'global_importance':m['global_importance'],'family_features':m['family_features'],'limitations':['Research prototype; not live packet capture','Metrics are specific to this dataset and test distribution','Dynamic selection is not assumed to outperform fixed or all-feature baselines']}

@app.get('/api/report/{dataset}/pdf')
def experiment_report_pdf(dataset:str, task:str=Query('native')):
    if dataset not in ['NSL-KDD','UNSW-NB15']: raise HTTPException(400,'Invalid dataset')
    try:
        if task=='binary':
            if dataset!='UNSW-NB15': raise HTTPException(400,'Binary report is available for UNSW-NB15 only')
            report_path=Path(__file__).resolve().parents[1]/'artifacts'/'research'/'unsw_nb15_binary_lightgbm'/'metrics.json'
            if not report_path.exists(): raise HTTPException(404,'Run UNSW-NB15 binary training before exporting its report')
            from .reporting import build_binary_pdf_report
            content=build_binary_pdf_report(json.loads(report_path.read_text()))
            filename='sentinel-unsw-nb15-binary-model-test-report.pdf'
        elif task=='native':
            model=load_model(dataset)
            if not model: raise HTTPException(404,'Train a model before exporting a PDF report')
            from .reporting import build_pdf_report
            content=build_pdf_report(model)
            filename=f"sentinel-{dataset.lower()}-model-test-report.pdf"
        else:
            raise HTTPException(400,'task must be native or binary')
        return Response(content=content,media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename="{filename}"'})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500,f'PDF report generation failed: {exc}')

@app.get('/api/features/{dataset}')
def features(dataset:str):
    m=load_model(dataset)
    if not m: raise HTTPException(404,'Train a model first')
    fixed=set(m['fixed_indices']); router=set(m['router_indices']); names=m['feature_names']; return {'dataset':dataset,'global_importance':m['global_importance'],'family_features':m['family_features'],'feature_count':len(names),'fixed_count':len(fixed),'router_count':len(router),'all_features':names,'fixed_features':[names[i] for i in sorted(fixed)],'removed_features':[name for i,name in enumerate(names) if i not in fixed],'router_features':[names[i] for i in sorted(router)]}
@app.get('/api/schema/{dataset}')
def schema(dataset:str):
    m=load_model(dataset)
    if not m: raise HTTPException(404,'Train a model first')
    numeric_columns=[]; categorical_columns=[]; categorical_options={}
    for name, transformer, columns in m['preprocessor'].transformers_:
        cols=list(columns)
        if name=='num': numeric_columns.extend(cols)
        elif name=='cat':
            categorical_columns.extend(cols)
            try:
                encoder=transformer.named_steps['encode']
                cats=encoder.categories_
                categorical_options={col:[str(v) for v in values] for col,values in zip(cols,cats)}
            except Exception:
                categorical_options={col:[] for col in cols}
    return {'dataset':dataset,'columns':m['feature_columns'],'labels':m['labels'],
            'numeric_columns':numeric_columns,'categorical_columns':categorical_columns,
            'categorical_options':categorical_options}

@app.get('/api/schema/{dataset}/examples')
def schema_examples(dataset:str, limit:int=6):
    """Return representative input-only examples from the official training file."""
    if dataset not in ['NSL-KDD','UNSW-NB15']: raise HTTPException(400,'Invalid dataset')
    model=load_model(dataset)
    if model is None: raise HTTPException(409,'Train a model before loading example inputs')
    files=dataset_files(dataset)
    train_file=next((f for f in files if 'train' in f.name.lower()),None)
    if train_file is None: raise HTTPException(404,'Official training file is unavailable')
    frame=load_nsl(train_file) if dataset=='NSL-KDD' else load_unsw(train_file)
    label_col,raw_labels,mapped_labels=prepare_labels(frame,dataset)
    feature_cols=list(model['feature_columns'])
    examples=[]; seen=set()
    for idx in frame.index:
        label=str(mapped_labels.loc[idx])
        if label in seen: continue
        row=frame.loc[idx]
        values={}
        for col in feature_cols:
            value=row[col]
            values[col]=None if pd.isna(value) else (value.item() if hasattr(value,'item') else value)
        examples.append({'label':label,'source':'Official training record','record_number':int(idx)+1,'values':values})
        seen.add(label)
        if len(examples)>=max(1,min(limit,12)): break
    return {'dataset':dataset,'source_file':train_file.name,'examples':examples}

class SimulationRequest(BaseModel):
    dataset:str
    speed:float=Field(default=5.0,ge=0.2,le=20)
    limit:int=Field(default=300,ge=1,le=5000)
    scenario:str='sequential'

@app.post('/api/simulation/start')
def simulation_start(req:SimulationRequest):
    if load_model(req.dataset) is None: raise HTTPException(409,'Train a model before simulation')
    files=dataset_files(req.dataset)
    test_file=next((p for p in files if 'test' in p.name.lower() and 'test-21' not in p.name.lower()),None)
    source=test_file or (files[0] if files else None)
    if source is None: raise HTTPException(404,'No dataset file available for replay')
    if req.scenario not in ('sequential','random','attacks'): raise HTTPException(400,'Unsupported replay scenario')
    try:
        frame=(load_nsl(source) if req.dataset=='NSL-KDD' else load_unsw(source))
        if req.dataset=='NSL-KDD' and len(frame.columns) in (42,43):
            from .datasets import NSL_FEATURES
            frame.columns=NSL_FEATURES+(['label','difficulty'] if len(frame.columns)==43 else ['label'])
        if req.scenario=='attacks':
            target='label' if 'label' in frame.columns else ('attack_cat' if 'attack_cat' in frame.columns else None)
            if target is None: raise ValueError('The selected test file has no attack label column for attack-focused replay.')
            frame=frame[frame[target].astype(str).str.strip().str.rstrip('.').str.lower().ne('normal')]
        if req.scenario=='random': frame=frame.sample(frac=1,random_state=42)
        rows=frame.head(req.limit).to_dict(orient='records')
    except Exception as e: raise HTTPException(400,f'Could not read replay data: {e}')
    sid=str(uuid.uuid4())
    with SIM_LOCK: SIMULATIONS[sid]={'id':sid,'status':'running','processed':0,'total':len(rows),'events':[],'dataset':req.dataset,'source':source.name,'scenario':req.scenario,'started_at':time.time(),'finished_at':None}
    def worker():
        for row in rows:
            while True:
                with SIM_LOCK:
                    state=SIMULATIONS[sid]; status_now=state['status']
                if status_now!='paused': break
                time.sleep(.15)
            if status_now=='stopped': break
            # Remove target/leakage fields before passing input; predict() uses only saved schema.
            try:
                from .datasets import prepare_labels
                clean={k:v for k,v in row.items() if k not in ('label','attack_cat','attack_category','difficulty','class')}
                result=predict(req.dataset,clean,'dynamic')
                # Keep the source label separate from the model prediction. Only
                # mark a row comparable when the dataset's native label taxonomy
                # matches the classifier's output taxonomy.
                ground_truth=None; ground_truth_field=None; ground_truth_comparable=False
                if req.dataset=='NSL-KDD' and 'label' in row:
                    raw_label=str(row.get('label','')).strip().rstrip('.')
                    ground_truth=NSL_ATTACKS.get(raw_label.lower(), 'Normal' if raw_label.lower() in ('normal','0') else 'Other')
                    ground_truth_field='label'; ground_truth_comparable=True
                elif req.dataset=='UNSW-NB15' and 'attack_cat' in row:
                    raw_label=str(row.get('attack_cat','')).strip()
                    ground_truth=UNSW_ATTACKS.get(raw_label, raw_label)
                    ground_truth_field='attack_cat'; ground_truth_comparable=True
                elif req.dataset=='UNSW-NB15' and 'label' in row:
                    raw_label=str(row.get('label','')).strip()
                    ground_truth='Normal' if raw_label.lower() in ('normal','0') else 'Attack'
                    ground_truth_field='label (binary)'
                safe_record=json.loads(pd.Series(clean).to_json())
                event={'index':SIMULATIONS[sid]['processed']+1,'prediction':result['prediction'],'verdict':result['verdict'],'inference_ms':result['inference_ms'],'confidence':result.get('confidence'),'confidence_note':result.get('confidence_note'),'timestamp':time.strftime('%H:%M:%S'),'record':safe_record,'selected_features':result.get('selected_features',[]),'ground_truth':ground_truth,'ground_truth_field':ground_truth_field,'correct':(ground_truth==result['prediction']) if ground_truth_comparable else None}
                c=db(); c.execute('INSERT INTO detections(dataset,prediction,verdict,approach,inference_ms,created_at,record,selected_features,model_type,confidence) VALUES(?,?,?,?,?,?,?,?,?,?)',(req.dataset,result['prediction'],result['verdict'],'dynamic',result['inference_ms'],time.strftime('%Y-%m-%d %H:%M:%S'),json.dumps(clean),json.dumps(result.get('selected_features',[])),load_model(req.dataset)['kind'],result.get('confidence'))); c.commit(); c.close()
                with SIM_LOCK:
                    state=SIMULATIONS[sid]; state['processed']+=1; state['events']=(state['events']+[event])[-30:]
            except Exception as e:
                with SIM_LOCK: SIMULATIONS[sid].update({'status':'failed','error':str(e)})
                return
            time.sleep(0.2/req.speed)
        with SIM_LOCK:
            if SIMULATIONS[sid]['status']!='stopped': SIMULATIONS[sid]['status']='completed'
            SIMULATIONS[sid]['finished_at']=time.time()
    threading.Thread(target=worker,daemon=True).start()
    return {'simulation_id':sid,'status':'running','total':len(rows),'source':source.name}

@app.get('/api/simulation/{simulation_id}')
def simulation_status(simulation_id:str):
    with SIM_LOCK:
        if simulation_id not in SIMULATIONS: raise HTTPException(404,'Simulation not found')
        state=dict(SIMULATIONS[simulation_id])
    elapsed=max(0.001,(state.get('finished_at') or time.time())-state.get('started_at',time.time()))
    state['elapsed_seconds']=round(elapsed,2)
    state['records_per_second']=round(state.get('processed',0)/elapsed,2)
    state.pop('started_at',None); state.pop('finished_at',None)
    return state

@app.post('/api/simulation/{simulation_id}/{action}')
def simulation_control(simulation_id:str,action:str):
    if action not in ('pause','resume','stop'): raise HTTPException(400,'Unsupported simulation action')
    with SIM_LOCK:
        if simulation_id not in SIMULATIONS: raise HTTPException(404,'Simulation not found')
        SIMULATIONS[simulation_id]['status']={'pause':'paused','resume':'running','stop':'stopped'}[action]
        if action=='stop': SIMULATIONS[simulation_id]['finished_at']=time.time()
        return dict(SIMULATIONS[simulation_id])

@app.post('/api/predict')
def single(req:PredictRequest):
    model=load_model(req.dataset)
    if model is None: raise HTTPException(409,f'No trained {req.dataset} model. Train it in Model Training Center first.')
    required=list(model['feature_columns']); missing=[k for k in required if k not in req.record or req.record[k] is None or (isinstance(req.record[k],str) and not req.record[k].strip())]
    if missing: raise HTTPException(422,'Complete all required feature fields before generating a prediction. Missing: '+', '.join(missing))
    numeric=set()
    categorical=set()
    for name, transformer, columns in model['preprocessor'].transformers_:
        if name=='num': numeric.update(columns)
        elif name=='cat': categorical.update(columns)
    invalid=[]
    for key in numeric:
        try:
            value=float(req.record[key])
            if not pd.notna(value) or value in (float('inf'),float('-inf')): raise ValueError()
        except (TypeError,ValueError): invalid.append(key)
    category_map={}
    if categorical:
        try:
            cat_transformer=next(t for n,t,c in model['preprocessor'].transformers_ if n=='cat')
            categories=cat_transformer.named_steps['encode'].categories_
            cols=next(c for n,t,c in model['preprocessor'].transformers_ if n=='cat')
            category_map={k:{str(v) for v in values} for k,values in zip(cols,categories)}
        except Exception: category_map={}
    invalid_categories=[k for k in categorical if k in category_map and str(req.record[k]) not in category_map[k]]
    if invalid or invalid_categories:
        raise HTTPException(422,'Invalid dataset values. Numeric fields: '+', '.join(sorted(invalid))+'; categorical fields: '+', '.join(sorted(invalid_categories)))
    clean=dict(req.record)
    for key in numeric: clean[key]=float(clean[key])
    try: result=predict(req.dataset,clean,req.approach)
    except FileNotFoundError as e: raise HTTPException(409,str(e))
    c=db(); c.execute('INSERT INTO detections(dataset,prediction,verdict,approach,inference_ms,created_at,record,selected_features,model_type,confidence) VALUES(?,?,?,?,?,?,?,?,?,?)',(req.dataset,result['prediction'],result['verdict'],req.approach,result['inference_ms'],time.strftime('%Y-%m-%d %H:%M:%S'),json.dumps(req.record),json.dumps(result.get('selected_features',[])),load_model(req.dataset)['kind'],result.get('confidence'))); c.commit(); c.close()
    return result
@app.post('/api/predict/compare')
def compare_prediction_strategies(req:PredictRequest):
    """Run one validated input through all three persisted strategies without logging three detections."""
    model=load_model(req.dataset)
    if model is None: raise HTTPException(409,f'No trained {req.dataset} model. Train it in Model Training Center first.')
    required=list(model['feature_columns'])
    missing=[k for k in required if k not in req.record or req.record[k] is None or (isinstance(req.record[k],str) and not req.record[k].strip())]
    if missing: raise HTTPException(422,'Complete all required feature fields before comparison. Missing: '+', '.join(missing))
    numeric=set(); categorical=set()
    for name, transformer, columns in model['preprocessor'].transformers_:
        if name=='num': numeric.update(columns)
        elif name=='cat': categorical.update(columns)
    invalid=[]
    for key in numeric:
        try:
            value=float(req.record[key])
            if not pd.notna(value) or value in (float('inf'),float('-inf')): raise ValueError()
        except (TypeError,ValueError): invalid.append(key)
    category_map={}
    if categorical:
        try:
            cat_transformer=next(t for n,t,c in model['preprocessor'].transformers_ if n=='cat')
            categories=cat_transformer.named_steps['encode'].categories_
            cols=next(c for n,t,c in model['preprocessor'].transformers_ if n=='cat')
            category_map={k:{str(v) for v in values} for k,values in zip(cols,categories)}
        except Exception: category_map={}
    invalid_categories=[k for k in categorical if k in category_map and str(req.record[k]) not in category_map[k]]
    if invalid or invalid_categories:
        raise HTTPException(422,'Invalid dataset values. Numeric fields: '+', '.join(sorted(invalid))+'; categorical fields: '+', '.join(sorted(invalid_categories)))
    clean=dict(req.record)
    for key in numeric: clean[key]=float(clean[key])
    try:
        results={approach:predict(req.dataset,clean,approach) for approach in ('all','fixed','dynamic')}
    except FileNotFoundError as exc:
        raise HTTPException(409,str(exc))
    return {'dataset':req.dataset,'results':results,'agreement':len({r['prediction'] for r in results.values()})==1,
            'note':'What-if comparison only; these three evaluations are not added to the detection history.'}

@app.post('/api/predict/batch')
async def batch(dataset:str, file:UploadFile=File(...), approach:str='dynamic'):
    if not file.filename.lower().endswith('.csv'): raise HTTPException(400,'Upload a CSV file')
    dest=Path('/tmp')/(str(uuid.uuid4())+'.csv')
    with dest.open('wb') as f: shutil.copyfileobj(file.file,f)
    try:
        if dataset=='NSL-KDD':
            from .datasets import NSL_FEATURES
            probe=pd.read_csv(dest,nrows=1,header=None)
            if len(probe.columns) in (41,42,43):
                df=pd.read_csv(dest,header=None)
                if len(df.columns)==41: df.columns=NSL_FEATURES
                elif len(df.columns)==42: df.columns=NSL_FEATURES+['label']
                else: df.columns=NSL_FEATURES+['label','difficulty']
            else: df=pd.read_csv(dest)
        else: df=pd.read_csv(dest,low_memory=False)
        records=df.to_dict(orient='records'); out=[predict(dataset,r,approach) for r in records]
        return {'count':len(out),'results':out}
    except Exception as e: raise HTTPException(400,str(e))
    finally: dest.unlink(missing_ok=True)
@app.get('/api/history')
def history(dataset:str|None=None, limit:int=100):
    c=db(); q='SELECT id,dataset,prediction,verdict,approach,inference_ms,created_at,record,selected_features,model_type,confidence FROM detections'; args=[]
    if dataset: q+=' WHERE dataset=?'; args.append(dataset)
    q+=' ORDER BY id DESC LIMIT ?'; args.append(min(limit,500)); rows=c.execute(q,args).fetchall(); c.close()
    return [{'id':r[0],'dataset':r[1],'prediction':r[2],'verdict':r[3],'approach':r[4],'inference_ms':r[5],'created_at':r[6],'record':json.loads(r[7]),'selected_features':json.loads(r[8] or '[]'),'model_type':r[9],'confidence':r[10]} for r in rows]
@app.post('/api/upload/{dataset}')
async def upload(dataset:str,file:UploadFile=File(...)):
    if dataset not in ['NSL-KDD','UNSW-NB15'] or not file.filename.lower().endswith(('.csv','.txt')): raise HTTPException(400,'Choose a supported dataset and CSV/TXT file')
    folder=DATA/('nsl_kdd' if dataset=='NSL-KDD' else 'unsw_nb15'); folder.mkdir(parents=True,exist_ok=True); safe=Path(file.filename).name; dest=folder/safe
    with dest.open('wb') as f: shutil.copyfileobj(file.file,f)
    try:
        df=pd.read_csv(dest,header=None if dataset=='NSL-KDD' and len(pd.read_csv(dest,nrows=1,header=None).columns) in (42,43) else 'infer') if dataset=='NSL-KDD' else pd.read_csv(dest,nrows=5)
        return {'filename':safe,'saved':True,'preview':df.head(5).fillna('').to_dict(orient='records'),'columns':list(df.columns)}
    except Exception as e:
        dest.unlink(missing_ok=True); raise HTTPException(400,f'File could not be parsed: {e}')
@app.get('/api/overview')
def overview():
    c=db(); rows=c.execute('SELECT COUNT(*),SUM(CASE WHEN verdict="Normal" THEN 1 ELSE 0 END),SUM(CASE WHEN verdict!="Normal" THEN 1 ELSE 0 END) FROM detections').fetchone(); c.close()
    return {'records_analyzed':rows[0] or 0,'normal':rows[1] or 0,'malicious':rows[2] or 0,'datasets':[{'dataset':n,**status(n)} for n in ['NSL-KDD','UNSW-NB15']]}
@app.delete('/api/reset/{dataset}')
def reset(dataset:str):
    if dataset not in ['NSL-KDD','UNSW-NB15']: raise HTTPException(400,'Invalid dataset')
    p=Path(__file__).resolve().parents[1]/'artifacts'/dataset.lower().replace('-','_')
    if p.exists(): shutil.rmtree(p)
    # Binary UNSW research is a separate persisted task; reset UNSW-NB15 must
    # remove it too, otherwise stale binary results reappear after reset.
    if dataset == 'UNSW-NB15':
        binary_dir=Path(__file__).resolve().parents[1]/'artifacts'/'research'/'unsw_nb15_binary_lightgbm'
        if binary_dir.exists(): shutil.rmtree(binary_dir)
    with MODEL_CACHE_LOCK: MODEL_CACHE.pop(dataset,None)
    c=db(); c.execute('DELETE FROM detections WHERE dataset=?',(dataset,)); c.execute('DELETE FROM experiments WHERE dataset=?',(dataset,)); c.commit(); c.close(); return {'reset':True,'dataset':dataset}
