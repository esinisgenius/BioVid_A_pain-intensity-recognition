import numpy as np
import pandas as pd
from src.analysis.analyze_xgb_subject_paired_f1 import score_trials,summarize,METRICS

def test_class_f1_includes_false_positives_from_other_classes():
    d=pd.DataFrame({'class_id':range(5),**{f'p{i}':np.full(5,int(i==0)) for i in range(5)}})
    score=score_trials(d)
    assert np.isclose(score[1],1/3)
    assert np.isclose(score[0],1/15)
    assert np.all(score[2:]==0)

def test_bootstrap_preserves_pairs_and_counts_ties():
    rows=[]
    for metric in METRICS:
        for sid,delta in enumerate([-.1,0,.1]): rows.append(dict(subject_id=sid,metric=metric,cross_f1=.3,within_f1=.3+delta,delta_f1=delta))
    p=pd.DataFrame(rows); a=summarize(p,1000); b=summarize(p,1000)
    pd.testing.assert_frame_equal(a,b)
    assert (a.n_improved==1).all() and (a.n_worsened==1).all() and (a.n_tied==1).all()
    assert (a.mean_delta_ci_low<0).all() and (a.mean_delta_ci_high>0).all()
