"""Score a saved hybrid research checkpoint on its official test features."""
import sys,json,time
from pathlib import Path
import numpy as np,joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT/'scripts'))
from hybrid_adaptive_evaluation import predict,metrics
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();out=ROOT/'backend/artifacts/research/hybrid_adaptive_acquisition'/a.dataset.lower().replace('-','_');b=joblib.load(out/'candidate_checkpoint.joblib');Xte=b['official_test_X'];yte=b['official_test_y'];pre=b['preprocessor'];idx=np.asarray(b['fixed_indices'],dtype=int);labels=b['labels']
t=time.perf_counter();Z=np.asarray(pre.transform(Xte),dtype=np.float32);print('transform',Z.shape,round(time.perf_counter()-t,2),flush=True)
t=time.perf_counter();pa=np.asarray(predict(b['all'],Z),dtype=object);print('all pred',round(time.perf_counter()-t,2),flush=True)
t=time.perf_counter();pf=np.asarray(predict(b['fixed'],Z[:,idx]),dtype=object);print('fixed pred',round(time.perf_counter()-t,2),flush=True)
t=time.perf_counter();pfix,cls=predict(b['fixed'],Z[:,idx],proba=True);print('fixed proba',round(time.perf_counter()-t,2),flush=True)
t=time.perf_counter();pall,acls=predict(b['all'],Z,proba=True);print('all proba',round(time.perf_counter()-t,2),flush=True)
pos={str(c):i for i,c in enumerate(acls)};pall=np.asarray(pall)[:,[pos[str(c)] for c in cls]];pfix=np.asarray(pfix)
conf=np.max(pfix,axis=1);ca=np.max(pall,axis=1);dyn=b['dynamic_config'];th=float(dyn['threshold'])
if dyn['mode']=='confidence_margin_router':
 use=(conf-ca)>=th;pdyn=pa.copy();pdyn[use]=pf[use];fallback=~use
else:
 fallback=conf<th;pdyn=pf.copy();pdyn[fallback]=pa[fallback]
res={}
for k,p in [('all',pa),('fixed',pf),('dynamic',pdyn)]:
 res[k]=metrics(yte,p,labels);res[k]['feature_count']=float(Z.shape[1] if k=='all' else len(idx) if k=='fixed' else len(idx)*(~fallback).mean()+Z.shape[1]*fallback.mean())
res['dynamic'].update({'adaptive_mode':dyn['mode'],'threshold':th,'fallback_fraction':float(fallback.mean()),'average_features':res['dynamic']['feature_count']})
rep={'dataset':a.dataset,'task':'native multiclass','model_fit_rows':b['model_fit_rows'],'estimator_all':str(type(b['all']['model'])),'estimator_fixed':str(type(b['fixed']['model'])),'dynamic_validation_config':dyn,'official_test_rows':len(yte),'official_test_results':res,'protocol':'Saved candidates fit on official training-side grouped fit fold. Dynamic threshold selected on training-only validation. Official test labels used only for metrics.'}
(out/'checkpoint_score.json').write_text(json.dumps(rep,indent=2,default=float));print('RESULTS',[(k,round(v['accuracy']*100,4),round(v['macro_f1']*100,4),v['feature_count']) for k,v in res.items()],flush=True)
