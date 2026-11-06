"""Fixed benchmark replication, five-class within/cross and labelled calibration.
No tuning. k denotes labels per class. Outputs are exploratory on the existing cohort.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.baselines import make_baseline_model
from src.models.run_phase3b_robustness import load_phase3_folds, make_fold_indices
from src.models.run_subject_independent_baselines import feature_sets, validate_feature_inputs
from src.models.run_xgboost_diagnostic import validate_protocol

SEED = 20260909

def split_labels(y, k, seed):
    rng = np.random.default_rng(seed)
    a = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:k] for c in range(5)])
    e = np.setdiff1d(np.arange(len(y)), a)
    assert len(a) == 5*k and len(set(a) & set(e)) == 0
    return a, e

def calibrate(p_train, y_train, p_test, penalty=10.):
    # Identity-centred scalar temperature plus five class offsets: six parameters.
    # Fixed penalty, summed likelihood, no test-label access or hyperparameter choice.
    z = np.log(np.clip(p_train, 1e-12, 1))
    def loss(v):
        s = np.exp(v[0])*z + v[1:]
        residual = softmax(s, axis=1) - np.eye(5)[y_train]
        val = np.sum(logsumexp(s, axis=1)-s[np.arange(len(y_train)), y_train]) + .5*penalty*np.dot(v,v)
        grad = np.r_[np.sum(residual*z)*np.exp(v[0]), residual.sum(axis=0)] + penalty*v
        return val, grad
    fit = minimize(loss, np.zeros(6), jac=True, method='L-BFGS-B', bounds=[(-3,3)]+[(None,None)]*5)
    if not fit.success:
        raise RuntimeError(f'Calibration failed: {fit.message}')
    return softmax(np.exp(fit.x[0])*np.log(np.clip(p_test,1e-12,1))+fit.x[1:], axis=1), fit.x

def metrics(y, p):
    pred = p.argmax(axis=1)
    prec, rec, f, _ = precision_recall_fscore_support(y,pred,labels=range(5),zero_division=0)
    return dict(macro_f1=float(f1_score(y,pred,average='macro',labels=range(5),zero_division=0)), balanced_accuracy=float(balanced_accuracy_score(y,pred)), **{f'{n}_{c}':float(arr[c]) for n,arr in [('precision',prec),('recall',rec),('f1',f)] for c in range(5)})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); ap.add_argument('--output',default='experiments/xgb_fiveclass_transfer'); ap.add_argument('--repeats',type=int,default=10)
    args=ap.parse_args(); root=args.root.resolve(); out=root/args.output; out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    table=pd.read_csv(root/'artifacts/feature_table_filtered.csv'); meta=pd.read_json(root/'artifacts/feature_metadata.json')
    sets=feature_sets(meta); validate_feature_inputs(table,meta,sets); cols=sets['combined_unique']; assert len(cols)==59
    folds=make_fold_indices(table,load_phase3_folds(root/'experiments/baselines/fold_assignments.csv'))
    audit=validate_protocol(table,folds,cols)
    assert (table.groupby(['subject_id','class_id']).size()==20).all()
    assert sorted(np.concatenate([s['test_index'] for s in folds]))==list(range(len(table)))
    x=table[cols].to_numpy(float); y=table.class_id.to_numpy(int)
    model=make_baseline_model('xgboost',SEED)
    config=dict(seed=SEED,features=cols,label_map=audit['observed_label_map'],model=model.named_steps['classifier'].get_params(),k_per_class=[1,2,5,10],k_total=[5,10,25,50],repeats=args.repeats,within='StratifiedKFold 5 folds per repeat; same fixed benchmark model; 80 train/20 test',cross='Original five subject-grouped folds',adaptation='Identity-centred log-temperature plus five class offsets; L2 penalty 10; fitted only on target adaptation labels',audit=audit,input_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['artifacts/feature_table_filtered.csv','artifacts/feature_metadata.json','experiments/baselines/fold_assignments.csv','src/models/baselines.py']})
    import importlib.metadata as im
    config['versions']={p:im.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost']}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    cross=np.zeros((len(y),5)); foldids=np.zeros(len(y),int)
    for s in folds:
        tr,te=s['train_index'],s['test_index']; m=make_baseline_model('xgboost',SEED); m.fit(x[tr],y[tr]); cross[te]=m.predict_proba(x[te]); foldids[te]=s['fold']
        print('cross fold',s['fold'],'seconds',round(time.monotonic()-start,1),flush=True)
    original=pd.read_csv(root/'experiments/model_strengthening_benchmark/oof_predictions_5class.csv'); original=original[original.model=='xgboost'].set_index('sample_name').loc[table.sample_name]
    mismatches=int(np.sum(original.pred_class_id.to_numpy()!=cross.argmax(axis=1)))
    (out/'replication.json').write_text(json.dumps(dict(prediction_mismatches=mismatches,**metrics(y,cross)),indent=2))
    if mismatches: raise RuntimeError('Benchmark reproduction mismatch; stop before downstream analysis')
    rows=[]; splitrows=[]; coefrows=[]; predictions=[]
    def savepred(idx,p,method,k,repeat):
        for i,pr in zip(idx,p):
            predictions.append(dict(sample_name=table.iloc[i].sample_name,subject_id=int(table.iloc[i].subject_id),class_id=int(y[i]),outer_fold=int(foldids[i]),method=method,k_per_class=k,repeat=repeat,**{f'p{c}':float(pr[c]) for c in range(5)}))
    savepred(np.arange(len(y)),cross,'cross',0,-1)
    within=np.zeros_like(cross)
    for pos,(sid,sub) in enumerate(table.groupby('subject_id',sort=True)):
        ix=sub.index.to_numpy(); yy=y[ix]; avg=np.zeros((len(ix),5))
        for r in range(args.repeats):
            wp=np.zeros_like(avg)
            for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED+r).split(x[ix],yy)):
                m=make_baseline_model('xgboost',SEED); m.fit(x[ix[tr]],yy[tr]); wp[te]=m.predict_proba(x[ix[te]])
                for j in te: splitrows.append(dict(subject_id=int(sid),sample_name=table.iloc[ix[j]].sample_name,protocol='within',repeat=r,k_per_class=0,role='test',fold=f))
            avg+=wp/args.repeats
            savepred(ix,wp,'within',0,r)
            rows.append(dict(subject_id=int(sid),method='within',k_per_class=0,repeat=r,**metrics(yy,wp)))
        within[ix]=avg
        rows.append(dict(subject_id=int(sid),method='cross',k_per_class=0,repeat=-1,**metrics(yy,cross[ix])))
        rows.append(dict(subject_id=int(sid),method='within_averaged_probability',k_per_class=0,repeat=-1,**metrics(yy,avg)))
        for k in [1,2,5,10]:
            for r in range(args.repeats):
                a,e=split_labels(yy,k,SEED+int(sid)*1000+r)
                pp,v=calibrate(cross[ix[a]],yy[a],cross[ix[e]])
                coefrows.append(dict(subject_id=int(sid),k_per_class=k,repeat=r,**{f'v{i}':float(z) for i,z in enumerate(v)}))
                for role,jj in [('adaptation',a),('evaluation',e)]:
                    for j in jj: splitrows.append(dict(subject_id=int(sid),sample_name=table.iloc[ix[j]].sample_name,protocol='adaptation',repeat=r,k_per_class=k,role=role,fold=-1))
                for method,p in [('population_matched',cross[ix[e]]),('adapted',pp)]:
                    savepred(ix[e],p,method,k,r)
                    rows.append(dict(subject_id=int(sid),method=method,k_per_class=k,repeat=r,**metrics(yy[e],p)))
        pd.DataFrame(rows).to_csv(out/'subject_repeat_metrics.csv',index=False)
        print('subject',pos+1,'of',table.subject_id.nunique(),'elapsed',round(time.monotonic()-start,1),flush=True)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv.gz',index=False)
    pd.DataFrame(splitrows).to_csv(out/'split_manifest.csv.gz',index=False)
    pd.DataFrame(coefrows).to_csv(out/'calibration_coefficients.csv',index=False)
    df=pd.DataFrame(rows); ms=[c for c in df if c not in ['subject_id','method','k_per_class','repeat']]
    subj=df.groupby(['subject_id','method','k_per_class'])[ms].mean().reset_index(); subj.to_csv(out/'subject_metrics.csv',index=False)
    rng=np.random.default_rng(SEED); summary=[]
    for (method,k),g in subj.groupby(['method','k_per_class']):
        for metric in ms:
            vals=g[metric].to_numpy(); b=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(axis=1)
            summary.append(dict(method=method,k_per_class=int(k),metric=metric,mean=float(vals.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975))))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    contrasts=[]
    for metric in ms:
        for k in [0,1,2,5,10]:
            a,b=('within','cross') if k==0 else ('adapted','population_matched')
            pivot=subj[subj.k_per_class==k].pivot(index='subject_id',columns='method',values=metric); delta=(pivot[a]-pivot[b]).to_numpy()
            boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(axis=1)
            contrasts.append(dict(contrast=a+' minus '+b,k_per_class=k,metric=metric,mean=float(delta.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    pooled={name:metrics(y,p) for name,p in [('cross',cross),('within_averaged_probability',within)]}
    for name,p in [('cross',cross),('within_averaged_probability',within)]:
        np.savetxt(out/f'confusion_{name}.csv',confusion_matrix(y,p.argmax(axis=1)),delimiter=',',fmt='%d')
    (out/'pooled_metrics.json').write_text(json.dumps(pooled,indent=2))
    report=['# Five-class fixed XGBoost transfer and labelled adaptation','',f'87 subjects; 8,700 trials; 59 combined_unique features; labels BL1/PA1/PA2/PA3/PA4. Seed {SEED}.', '', 'Cross reproduces the original five subject-grouped folds and fixed benchmark configuration. No tuning or model selection was performed. All median imputers were fit on training samples only.',f'Original cross prediction mismatches: {mismatches}. Pooled cross macro F1: {pooled["cross"]["macro_f1"]:.6f}.','', 'Within uses 5-fold stratified CV repeated 10 times, 80 training trials per subject per fit. Primary comparisons average repeat metrics within subject, then average subjects. Probability-averaged within OOF metrics are secondary and use an ensemble across repeats. Cross training uses thousands of trials: the gap is not a pure subject-shift effect.', '', 'Adaptation uses only the held-out subject calibration labels to fit six strongly regularized score-calibration parameters on frozen population XGBoost outputs. k is PER CLASS (total 5k). This is score calibration, not retraining or fine-tuning the trees. Evaluation excludes calibration trials; population and adapted predictions use identical evaluation samples. Budgets are nested per repeat. No target evaluation labels enter fitting.', '', '## Subject-mean macro F1 (95% subject bootstrap CI)','']
    for s in summary:
        if s['metric']=='macro_f1': report.append(f'- {s["method"]}, k/class={s["k_per_class"]}: {s["mean"]:.4f} [{s["ci_low"]:.4f}, {s["ci_high"]:.4f}]')
    report+=['','## Paired macro F1 differences','']
    for s in contrasts:
        if s['metric']=='macro_f1': report.append(f'- {s["contrast"]}, k/class={s["k_per_class"]}: {s["mean"]:+.4f} [{s["ci_low"]:+.4f}, {s["ci_high"]:+.4f}]')
    report+=['','These exploratory results are conditional on the existing features, fixed model, folds and cohort. Bootstrap intervals resample subjects after repeat averaging; they do not capture training-cohort resampling uncertainty. No significance tests or selective reporting of budgets are used. Fixed XGBoost may be unsuitable for small within-subject training sets; lack of improvement is not an information ceiling. Known BL1 recovery-period and experimental-heat-pain limitations apply. No clinical or causal interpretation is justified.',f'Elapsed seconds: {time.monotonic()-start:.1f}.']
    (root/'reports/xgb_fiveclass_transfer.md').write_text('\n'.join(report)+'\n')
    print('COMPLETE',time.monotonic()-start,flush=True)

if __name__=='__main__': main()
