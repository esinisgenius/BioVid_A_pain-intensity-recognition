"""Recompute paired subject F1 directly from saved OOF probabilities; no fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

LABELS=['BL1','PA1','PA2','PA3','PA4']; METRICS=['macro_F1']+LABELS
SEED=20261106

def score_trials(g):
    y=g.class_id.to_numpy(int); p=g[[f'p{i}' for i in range(5)]].to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-6):
        raise ValueError('Invalid probabilities')
    f=f1_score(y,p.argmax(axis=1),labels=range(5),average=None,zero_division=0)
    return np.r_[f.mean(),f]

def paired_scores(d):
    d=d[d.method.isin(['cross','within'])].copy()
    if d.subject_id.nunique()!=87: raise ValueError('Expected exactly 87 subjects')
    if d.duplicated(['method','subject_id','repeat','sample_name']).any(): raise ValueError('Duplicate predictions')
    rows=[]; repeats=[]
    for sid,g in d.groupby('subject_id',sort=True):
        cross=g[g.method=='cross']; within=g[g.method=='within']
        if len(cross)!=100 or not (cross.groupby('class_id').size().reindex(range(5),fill_value=0)==20).all(): raise ValueError('Cross trial/label mismatch')
        if set(within['repeat'])!=set(range(10)): raise ValueError('Expected ten repeats')
        ref=cross.set_index('sample_name').class_id.sort_index()
        scores=[]
        for rep,w in within.groupby('repeat',sort=True):
            if len(w)!=100 or not w.set_index('sample_name').class_id.sort_index().equals(ref): raise ValueError('Within/cross trial-label mismatch')
            s=score_trials(w); scores.append(s)
            repeats.append(dict(subject_id=int(sid),repeat=int(rep),**dict(zip(METRICS,s))))
        cs=score_trials(cross); ws=np.mean(scores,axis=0)
        for m,c,w in zip(METRICS,cs,ws): rows.append(dict(subject_id=int(sid),metric=m,cross_f1=c,within_f1=w,delta_f1=w-c))
    return pd.DataFrame(rows),pd.DataFrame(repeats)

def summarize(paired,n_boot=10000):
    subjects=sorted(paired.subject_id.unique()); rng=np.random.default_rng(SEED)
    draws=rng.integers(len(subjects),size=(n_boot,len(subjects)))
    rows=[]
    for metric in METRICS:
        g=paired[paired.metric==metric].set_index('subject_id').loc[subjects]
        delta=g.delta_f1.to_numpy(); positive=delta>1e-12; negative=delta < -1e-12
        row=dict(metric=metric,n_subjects=len(g),cross_mean=g.cross_f1.mean(),within_mean=g.within_f1.mean(),mean_delta=delta.mean(),median_delta=np.median(delta),sd_delta=np.std(delta,ddof=1),q25_delta=np.quantile(delta,.25),q75_delta=np.quantile(delta,.75),min_delta=delta.min(),max_delta=delta.max(),n_improved=int(positive.sum()),n_worsened=int(negative.sum()),n_tied=int((~(positive|negative)).sum()),improved_fraction=positive.mean())
        for name,vals in [('mean_delta',delta),('median_delta',delta),('improved_fraction',positive.astype(float))]:
            boot=np.median(vals[draws],axis=1) if name=='median_delta' else vals[draws].mean(axis=1)
            row[name+'_ci_low'],row[name+'_ci_high']=np.quantile(boot,[.025,.975])
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    root=Path(__file__).resolve().parents[2]; out=root/'experiments/xgb_subject_paired_f1'; out.mkdir(exist_ok=True)
    source=root/'experiments/xgb_fiveclass_transfer/predictions.csv.gz'
    d=pd.read_csv(source); paired,repeats=paired_scores(d); summary=summarize(paired)
    # Independent consistency check against the prior run, without using its scores as input.
    old=pd.read_csv(root/'experiments/xgb_fiveclass_transfer/subject_repeat_metrics.csv')
    for r in paired.itertuples():
        field='macro_f1' if r.metric=='macro_F1' else 'f1_'+str(LABELS.index(r.metric))
        for method,val in [('cross',r.cross_f1),('within',r.within_f1)]:
            prior=old[(old.subject_id==r.subject_id)&(old.method==method)][field].mean()
            if not np.isclose(prior,val,atol=1e-12,rtol=0): raise ValueError('Prior metric disagreement')
    paired.to_csv(out/'subject_paired_f1_long.csv',index=False)
    paired.pivot(index='subject_id',columns='metric',values=['cross_f1','within_f1','delta_f1']).to_csv(out/'subject_paired_f1_wide.csv')
    repeats.to_csv(out/'within_repeat_f1.csv',index=False); summary.to_csv(out/'paired_summary.csv',index=False)
    font=font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'text.color':'#50545A','axes.labelcolor':'#50545A','xtick.color':'#50545A','ytick.color':'#50545A','axes.edgecolor':'#D9DADD','pdf.fonttype':42,'svg.fonttype':'none'})
    rng=np.random.default_rng(SEED); fig,axs=plt.subplots(2,3,figsize=(10,5.8),sharey=True)
    for ax,metric in zip(axs.flat,METRICS):
        g=paired[paired.metric==metric]; s=summary[summary.metric==metric].iloc[0]; v=g.delta_f1.to_numpy()*100
        vp=ax.violinplot(v,positions=[0],widths=.65,showextrema=False)
        for b in vp['bodies']: b.set_facecolor('#607F99'); b.set_edgecolor('#607F99'); b.set_alpha(.27)
        jitter=rng.uniform(-.2,.2,len(v))
        ax.scatter(jitter,v,s=10,c=np.where(v>=0,'#607F99','#B86F78'),alpha=.85,linewidths=0)
        mean=s.mean_delta*100; lo=s.mean_delta_ci_low*100; hi=s.mean_delta_ci_high*100
        ax.errorbar(.5,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',color='#50545A',capsize=3,markersize=4,lw=1)
        ax.axhline(0,color='#D9DADD',lw=.8); ax.set_xlim(-.45,.75); ax.set_xticks([0,.5],['Subjects','Mean / 95% CI'])
        ax.set_title(metric.replace('_',' '),fontweight='normal'); ax.text(.03,.97,f'Improved: {int(s.n_improved)}/87',transform=ax.transAxes,va='top',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    for ax in axs[:,0]: ax.set_ylabel('Within − Cross F1 (percentage points)')
    fig.tight_layout()
    for ext in ['png','pdf','svg']: fig.savefig(out/f'paired_f1_distribution.{ext}',dpi=320,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    caption='Five-class fixed benchmark XGBoost; 59 features; 87 subjects. Each point is one subject: within F1 is computed over 100 OOF trials separately for each of ten 5-fold repeats, then averaged; cross F1 uses 100 subject-held-out predictions. Blue positive differences favour within; rose negative differences favour cross. Diamonds and bars show mean paired differences and pointwise 95% percentile bootstrap CIs from 10,000 resamples of 87 subjects, preserving pairing; violin widths describe the empirical distribution. Repeats and trials are not independent bootstrap units. Within uses 80 training trials per fit; cross uses thousands from other subjects. No retraining.'
    (out/'figure_caption.txt').write_text(caption+'\n')
    config=dict(source=str(source.relative_to(root)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=SEED,bootstrap_resamples=10000,bootstrap_unit='subject, paired; common resamples across metrics',interval='pointwise percentile 95%; not simultaneous; no hypothesis tests',within_aggregation='mean of ten repeat-level F1 scores; no probability averaging',zero_division=0,tie_tolerance=1e-12,font=font,n_subjects=87,n_cross_predictions=8700,n_within_predictions=87000,prior_metric_check='all subject/metric values agree within 1e-12',no_training=True)
    (out/'provenance.json').write_text(json.dumps(config,indent=2))
    lines=['# 五分类 XGBoost：受试者级配对 F1 分析','','## 方法','','直接从保存的预测概率取 argmax，不训练模型。Cross 对每人 100 条试次计算一次 F1；Within 在每次重复的完整 100 条折外预测上计算 F1，再对 10 次重复取均值。每类 F1 使用五分类 one-vs-rest 混淆计数（包括来自其他真实类别的假阳性），不是仅截取该类试次。Macro F1 为五类 F1 的算术平均。','','受试者配对差值 ΔF1 = Within − Cross。10,000 次受试者级配对 bootstrap，每次有放回抽取 87 人，同一组重采样用于所有指标；95% CI 为百分位区间。重复先在受试者内平均，不视为独立样本。无受试者排除；新计算的全部受试者级分数与原指标文件一致。','','## 配对结果','','差值及区间以百分点表示；Cross/Within 为受试者平均 F1。','','| 指标 | Cross | Within | 平均 Δ（95% CI） | 改善 | 下降 | 持平 |','|---|---:|---:|---:|---:|---:|---:|']
    for s in summary.itertuples(): lines.append(f'| {s.metric} | {s.cross_mean:.4f} | {s.within_mean:.4f} | {s.mean_delta*100:+.2f} [{s.mean_delta_ci_low*100:+.2f}, {s.mean_delta_ci_high*100:+.2f}] | {s.n_improved}/87 ({s.improved_fraction*100:.1f}%) | {s.n_worsened} | {s.n_tied} |')
    lines+=['','## 差异分布','','| 指标 | 中位数 Δ | 第25–75百分位 | 最小–最大 | SD |','|---|---:|---:|---:|---:|']
    for s in summary.itertuples(): lines.append(f'| {s.metric} | {s.median_delta*100:+.2f} | {s.q25_delta*100:+.2f} 至 {s.q75_delta*100:+.2f} | {s.min_delta*100:+.2f} 至 {s.max_delta*100:+.2f} | {s.sd_delta*100:.2f} |')
    lines += ['', '## 主要发现', '\n- 总体 macro F1 平均提高 7.60 个百分点，95% CI [5.58, 9.75]；70/87 人（80.5%）改善，17 人下降。改善比例的 bootstrap 95% CI 为 [72.4%, 88.5%]。差值中位数为 +6.57 个百分点，四分位范围为 +1.07 至 +13.13，范围为 −12.39 至 +41.72，提示存在明显个体差异。\n- PA2 的平均 F1 从 0.1131 升至 0.2915（+17.84 个百分点），73/87 人改善；PA3 从 0.2119 升至 0.3477（+13.58 个百分点），64/87 人改善。两者是本次观察到的最大逐类均值收益；未正式检验类别间收益差异。\n- PA1 的均值收益为 +5.05 个百分点，但只有 48/87 人改善，中位数收益仅 +1.71 个百分点；其改善并不普遍。PA4 均值提高 8.27 个百分点，56/87 人改善。\n- BL1 平均下降 6.72 个百分点，66/87 人下降。因此总体提升并非五个等级一致提升，而伴随明显等级间差异。PA2/PA3 的改善支持当前设置下中间等级的个体内可预测性高于跨个体预测，但绝对 F1 仍有限。\n- 五个逐类均值差值的平均恰好等于总体 macro F1 差值；这来自指标定义，不应被解释为各类之间独立的生理贡献分解。\n\n']
    lines+=['','## 解释边界','','逐类结果可以回答 PA1–PA3 是否在当前个体模型下获得改善；PA2 为五等级中央类别。必须同时考察平均差值、区间和改善人数，不能用总体均值代表所有人。正差值是该固定模型与协议下的预测证据，不直接证明生理机制或特征重要性改变。','','Within 与 Cross 训练量和训练人群构成不同，同一受试者随机试次分折还可能受记录条件、试次依赖及时间漂移影响。Bootstrap 区间以现有拟合模型和队列为条件，不包含重训或新队列不确定性。六个区间为逐项区间，不是同时置信区间；未进行显著性检验或挑选等级。实验热刺激等级及 BL1 恢复期的解释限制仍适用。','','## 图注','',caption,'','输出：`experiments/xgb_subject_paired_f1/`；复现：`python -m src.analysis.analyze_xgb_subject_paired_f1`。']
    (root/'reports/xgb_subject_paired_f1.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))
if __name__=='__main__': main()
