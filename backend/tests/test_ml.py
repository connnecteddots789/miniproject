import numpy as np
import pandas as pd
from app.datasets import prepare_labels, feature_frame

def test_nsl_kdd_family_mapping_and_leakage_exclusion():
    df=pd.DataFrame({'duration':[0,1,2,3],'protocol_type':['tcp']*4,'src_bytes':[1,2,3,4],'label':['normal','neptune','ipsweep','buffer_overflow'],'difficulty':[1,2,3,4]})
    col,raw,y=prepare_labels(df,'NSL-KDD')
    assert y.tolist()==['Normal','DoS','Probe','U2R']
    X=feature_frame(df,'NSL-KDD',col)
    assert 'label' not in X and 'difficulty' not in X

def test_unsw_taxonomy_preserved():
    df=pd.DataFrame({'attack_cat':['Normal','Fuzzers','Backdoors','Generic'],'label':[0,1,1,1],'dur':[1,2,3,4]})
    col,raw,y=prepare_labels(df,'UNSW-NB15')
    assert y.tolist()==['Normal','Fuzzers','Backdoors','Generic']
    X=feature_frame(df,'UNSW-NB15',col)
    assert list(X.columns)==['dur']


def test_additional_classifier_choices_are_available():
    from app.ml import estimator
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import LinearSVC, SVC
    assert isinstance(estimator('Extra Trees', 17), ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression
    assert isinstance(estimator('Logistic Regression', 17), LogisticRegression)
    assert isinstance(estimator('SVM', 17), LinearSVC)
    assert isinstance(estimator('SVM (RBF)', 17), SVC)

def test_extended_estimators_fit_and_predict_small_multiclass_data():
    from app.ml import estimator, fit_classifier, make_preprocessor, model_input, estimator_for_features
    from sklearn.datasets import make_classification
    Xn, y = make_classification(n_samples=180, n_features=8, n_informative=6,
                                n_classes=3, n_clusters_per_class=1, random_state=19)
    X = pd.DataFrame(Xn, columns=[f'n{i}' for i in range(8)])
    X['protocol'] = np.where(np.arange(len(X)) % 3 == 0, 'tcp',
                             np.where(np.arange(len(X)) % 3 == 1, 'udp', 'icmp'))
    X['service'] = np.where(np.arange(len(X)) % 2 == 0, 'http', 'dns')
    pre = make_preprocessor(X, native_cats=True)
    Z = pre.fit_transform(X)
    cat_indices = {8, 9}
    indices = np.arange(Z.shape[1])
    cat_values = model_input(Z, indices, cat_indices, 'CatBoost')
    for name in ['Decision Tree', 'Random Forest', 'Extra Trees', 'Logistic Regression', 'SVM (Linear)', 'SVM (RBF)', 'XGBoost', 'LightGBM', 'CatBoost', 'Stacking Ensemble']:
        model = estimator_for_features(name, 23, indices, cat_indices)
        values = cat_values if name == 'CatBoost' else Z
        fitted = fit_classifier(model, values, y, 23)
        assert fitted == len(y)
        assert len(model.predict(values[:12])) == 12

def test_high_cardinality_categories_are_encoded_compactly():
    from app.ml import make_preprocessor
    frame = pd.DataFrame({'bytes': np.arange(120),
                          'srcip': [f'10.0.0.{i}' for i in range(120)],
                          'protocol': ['tcp', 'udp', 'icmp'] * 40})
    prep = make_preprocessor(frame)
    transformed = prep.fit_transform(frame)
    # 120 unique IP-like values stay one encoded column instead of 120 dummy columns.
    assert transformed.shape[1] < 20

def test_stratified_sampler_preserves_rare_classes():
    from app.ml import stratified_sample_indices
    y = np.array(['common'] * 9990 + ['rare-a'] * 5 + ['rare-b'] * 5, dtype=object)
    idx = stratified_sample_indices(y, 100, 31, min_per_class=3)
    assert len(idx) == 100
    assert set(np.unique(y[idx])) == {'common', 'rare-a', 'rare-b'}
    assert (y[idx] == 'rare-a').sum() >= 3
    assert (y[idx] == 'rare-b').sum() >= 3


def test_xgboost_adapter_selects_objective_from_binary_or_multiclass_targets():
    pytest = __import__('pytest')
    pytest.importorskip('xgboost')
    import numpy as np
    from app.ml import estimator

    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]], dtype=float)
    multiclass_y = np.asarray(['Normal', 'Normal', 'DoS', 'DoS', 'Probe', 'Probe', 'R2L', 'R2L'])
    multi = estimator('XGBoost', 42).fit(X, multiclass_y)
    assert multi.estimator_.get_xgb_params()['objective'] == 'multi:softprob'
    assert multi.estimator_.get_xgb_params()['num_class'] == 4
    assert multi.predict_proba(X).shape == (8, 4)

    binary_y = np.asarray(['Normal', 'Normal', 'DoS', 'DoS', 'Normal', 'Normal', 'DoS', 'DoS'])
    binary = estimator('XGBoost', 42).fit(X, binary_y)
    assert binary.estimator_.get_xgb_params()['objective'] == 'binary:logistic'
    assert binary.predict_proba(X).shape == (8, 2)
