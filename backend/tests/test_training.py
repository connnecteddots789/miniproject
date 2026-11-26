import numpy as np, pandas as pd
from app import ml

def test_full_training_persists_three_approaches(tmp_path, monkeypatch):
    rng=np.random.default_rng(4); n=180
    df=pd.DataFrame({'duration':rng.random(n),'src_bytes':rng.integers(0,500,n),'dst_bytes':rng.integers(0,500,n),'protocol_type':rng.choice(['tcp','udp','icmp'],n),'label':np.where(np.arange(n)%3==0,'normal',np.where(np.arange(n)%3==1,'neptune','ipsweep'))})
    monkeypatch.setattr(ml,'load_dataset',lambda dataset:df)
    monkeypatch.setattr(ml,'dataset_files',lambda dataset:[])
    monkeypatch.setattr(ml,'ROOT',tmp_path)
    ml.JOBS['test-run']={'id':'test-run','status':'running','stage':'queued','progress':0,'logs':[]}
    result=ml.train_pipeline('NSL-KDD','Decision Tree',11,job_id='test-run')
    stages=[entry['message'] for entry in ml.JOBS['test-run']['logs']]
    assert any('Evaluating dynamic router and experts in batches' in message for message in stages)
    assert set(result['results'])=={'all','fixed','dynamic'}
    artifact=ml.load_model('NSL-KDD')
    assert set(artifact['confusion_records'])=={'all','fixed','dynamic'}
    assert sum(len(v) for v in artifact['confusion_records']['dynamic'].values())>0
    assert (tmp_path/'artifacts/nsl_kdd/model.joblib').exists()
    assert (tmp_path/'artifacts/nsl_kdd/metadata.json').exists()
    assert ml.load_model('NSL-KDD') is not None
    prediction=ml.predict('NSL-KDD',{'duration':1,'src_bytes':100,'dst_bytes':50,'protocol_type':'tcp'})
    assert prediction['verdict'] in ('Normal','Malicious')

def test_official_named_train_test_split_is_used(tmp_path, monkeypatch):
    rng=np.random.default_rng(9)
    train=pd.DataFrame({'id':range(90),'dur':rng.random(90),'proto':rng.choice(['tcp','udp'],90),'label':np.where(np.arange(90)%3==0,0,1),'attack_cat':np.where(np.arange(90)%3==0,'Normal',np.where(np.arange(90)%3==1,'Fuzzers','Generic'))})
    test=pd.DataFrame({'id':range(1000,1030),'dur':rng.random(30),'proto':rng.choice(['tcp','udp'],30),'label':np.where(np.arange(30)%3==0,0,1),'attack_cat':np.where(np.arange(30)%3==0,'Normal',np.where(np.arange(30)%3==1,'Fuzzers','Generic'))})
    train_path=tmp_path/'UNSW_NB15_training-set.csv'; test_path=tmp_path/'UNSW_NB15_testing-set.csv'
    train.to_csv(train_path,index=False); test.to_csv(test_path,index=False)
    monkeypatch.setattr(ml,'dataset_files',lambda dataset:[train_path,test_path])
    monkeypatch.setattr(ml,'ROOT',tmp_path/'output')
    result=ml.train_pipeline('UNSW-NB15','Decision Tree',7)
    assert result['split']=='official train/test files'
    assert result['train_records']==90 and result['test_records']==30

def test_svm_classifier_trains_and_persists_dynamic_path(tmp_path, monkeypatch):
    rng=np.random.default_rng(21); n=120
    df=pd.DataFrame({'duration':rng.random(n),'src_bytes':rng.integers(0,500,n),'protocol_type':rng.choice(['tcp','udp'],n),'label':np.where(np.arange(n)%2==0,'normal','neptune')})
    monkeypatch.setattr(ml,'load_dataset',lambda dataset:df)
    monkeypatch.setattr(ml,'dataset_files',lambda dataset:[])
    monkeypatch.setattr(ml,'ROOT',tmp_path)
    ml.JOBS['svm-test']={'id':'svm-test','status':'running','stage':'queued','progress':0,'logs':[]}
    result=ml.train_pipeline('NSL-KDD','SVM',21,job_id='svm-test')
    assert set(result['results'])=={'all','fixed','dynamic'}
    artifact=ml.load_model('NSL-KDD')
    assert artifact['kind']=='SVM'
    prediction=ml.predict('NSL-KDD',{'duration':.2,'src_bytes':120,'protocol_type':'tcp'})
    assert prediction['verdict'] in ('Normal','Malicious')
