import numpy as np
from src.models.run_xgb_fiveclass_transfer import split_labels, calibrate, metrics
from src.models.baselines import make_baseline_model

def test_nested_disjoint_deterministic_budgets():
    y=np.repeat(np.arange(5),20)
    previous=set()
    for k in [1,2,5,10]:
        a,e=split_labels(y,k,42)
        assert len(a)==5*k and not set(a)&set(e)
        assert set(a)|set(e)==set(range(100))
        assert previous<=set(a)
        assert np.array_equal(a,split_labels(y,k,42)[0])
        assert np.all(np.bincount(y[a])==k)
        previous=set(a)

def test_calibration_is_finite_and_deterministic():
    rng=np.random.default_rng(5); p=rng.dirichlet(np.ones(5),size=100); y=np.repeat(np.arange(5),20)
    a,e=split_labels(y,2,42)
    q,v=calibrate(p[a],y[a],p[e]); q2,v2=calibrate(p[a],y[a],p[e])
    assert q.shape==(90,5) and np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(axis=1),1)
    np.testing.assert_allclose(q,q2); np.testing.assert_allclose(v,v2)

def test_benchmark_and_train_only_imputation():
    m=make_baseline_model('xgboost',20260909)
    assert m.named_steps['classifier'].get_params()['n_estimators']==300
    imp=m.named_steps['imputer']; train=np.array([[1.,np.nan],[3.,4.]])
    imp.fit(train); before=imp.statistics_.copy(); imp.transform([[100.,1000.]])
    np.testing.assert_array_equal(before,imp.statistics_)
    assert metrics(np.arange(5),np.eye(5))['macro_f1']==1
