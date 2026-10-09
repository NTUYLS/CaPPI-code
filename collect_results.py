"""Verify every saved prediction against its immutable fold, then export figures.

This script never changes training configurations or selects model checkpoints.
Fold SD describes variation across folds; it is not a confidence interval.
"""
from pathlib import Path
import argparse
import hashlib
import json
import torch  # Initialize Windows torch DLLs before scipy/sklearn imports.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from cappi_data import load_fold
from run_experiments import MATRICES,metrics,summarize

HERE=Path(__file__).resolve().parent
LABELS={
 'pair__random__drw__residual':'Residual fusion',
 'pair__random__drw__concat_matched':'Capacity-matched MLP',
 'pair__random__drw__concat_linear':'Linear concatenation',
 'pair__random__drw__sequence_only':'Sequence only',
 'pair__random__drw__topology_only':'Topology only',
 'pair__random__deepwalk__residual':'Uniform walks (DeepWalk-style)',
 'pair__random__node2vec__residual':'node2vec walks',
 'pair__degree_matched__drw__residual':'Approx. degree-matched negatives',
 'protein__random__drw__residual':'Protein-disjoint: residual',
 'protein__random__drw__sequence_only':'Protein-disjoint: sequence only',
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def collect(campaign,prepared,output):
    output.mkdir(parents=True,exist_ok=True)
    expected={f'{p}__{n}__{g}__{v}__fold{f}' for v,n,p,g in MATRICES['reviewer'] for f in range(5)}
    completed={p.parent.name:p for p in campaign.glob('*/completed.json')}
    if set(completed)-expected:raise ValueError('Unexpected runs in campaign; do not mix unverified experiments')
    missing=sorted(expected-completed.keys())
    if missing:raise RuntimeError(f'Campaign incomplete ({len(missing)} missing runs): {missing[:4]}')
    rows=[];groups={};verified=[]
    for name in sorted(expected):
        record=json.loads(completed[name].read_text(encoding='utf-8'));cfg=record['config'];folder=completed[name].parent
        canonical=f"{cfg['protocol']}__{cfg['negative_sampling']}__{cfg['graph']['method']}__{cfg['variant']}__fold{cfg['fold']}"
        if name!=record['name'] or name!=canonical:raise ValueError('Run name/configuration mismatch')
        if not record['test_evaluated']:raise ValueError('Training-only run cannot be reported as an evaluation')
        if sha(folder/'best.pt')!=record['checkpoint_sha256']:raise ValueError('Checkpoint content changed')
        if sha(folder/'test_predictions.csv')!=record['predictions_sha256']:raise ValueError('Prediction content changed')
        data=load_fold(prepared,cfg['negative_sampling'],cfg['protocol'],cfg['fold'])
        pred=pd.read_csv(folder/'test_predictions.csv')
        ids=data['node_ids'][data['test_pairs']]
        if not np.array_equal(pred[['protein_u','protein_v']].values,ids):raise ValueError('Test pair mismatch')
        if not np.array_equal(pred.label.values,data['test_labels']):raise ValueError('Test label mismatch')
        if set(pred.split_hash)!={data['split_hash']}:raise ValueError('Split provenance mismatch')
        if set(pred.checkpoint_sha256)!={record['checkpoint_sha256']}:raise ValueError('Checkpoint provenance mismatch')
        if set(pred.fold)!={cfg['fold']} or set(pred.seed)!={cfg['seed']}:raise ValueError('Fold/seed metadata mismatch')
        if not np.isfinite(pred.probability).all() or not pred.probability.between(0,1).all():raise ValueError('Invalid probabilities')
        if not np.array_equal(pred.prediction.values,(pred.probability.values>=.5).astype(int)):raise ValueError('Threshold mismatch')
        recalculated=metrics(pred.label.values,pred.probability.values)
        for metric,value in recalculated.items():
            if value is not None and not np.isclose(value,record['test_metrics'][metric],rtol=0,atol=1e-10):raise ValueError(f'Metric mismatch: {name} {metric}')
        group=name.rsplit('__fold',1)[0]
        groups.setdefault(group,[]).append(pred)
        param=json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
        rows.append({'experiment':group,'label':LABELS[group],'fold':cfg['fold'],'best_epoch':record['best_epoch'],
                     'epochs_completed':record['epochs_completed'],'best_validation_bce':record['best_validation_bce'],
                     'checkpoint_sha256':record['checkpoint_sha256'],'split_hash':data['split_hash'],**recalculated})
        verified.append({'run':name,'prediction_sha256':record['predictions_sha256'],'parameters':param})
    prepared_manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    for group,frames in groups.items():
        all_pairs=pd.concat(frames,ignore_index=True)
        if all_pairs.duplicated(['protein_u','protein_v']).any():raise ValueError(f'Outer test overlap: {group}')
        if group.startswith('pair__') and len(all_pairs)!=2*prepared_manifest['n_positives']:raise ValueError('Pair outer tests do not cover benchmark')
    summarize(campaign)
    detail=pd.DataFrame(rows);detail.to_csv(output/'verified_fold_metrics.csv',index=False,encoding='utf-8-sig')
    summary=pd.read_csv(campaign/'summary.csv');summary.insert(1,'label',summary.experiment.map(LABELS))
    summary.to_csv(output/'verified_summary.csv',index=False,encoding='utf-8-sig')
    # Paired descriptive differences only: folds overlap in training proteins, so
    # no independent-fold p-values or edge-level confidence intervals are claimed.
    ref='pair__random__drw__residual';paired=[]
    for group in LABELS:
        if group==ref or not group.startswith('pair__random__'):continue
        a=detail[detail.experiment==ref].sort_values('fold');b=detail[detail.experiment==group].sort_values('fold')
        for metric in ['auroc','average_precision','mcc']:
            delta=a[metric].values-b[metric].values
            paired.append({'reference':ref,'comparator':group,'metric':metric,'mean_delta':float(delta.mean()),
                           'fold_delta_sd':float(delta.std(ddof=1)),'fold_deltas':delta.tolist()})
    save_json(output/'paired_descriptive_differences.json',paired)
    save_json(output/'verification.json',{'status':'passed','verified_runs':len(verified),'all_outer_tests_disjoint':True,
       'no_new_selection_or_tuning':True,'fold_sd_is_not_a_confidence_interval':True,
       'campaign_config_sha256':sha(campaign/'campaign_config.json'),'runs':verified})
    plot(summary,groups,output)
    return summary

def plot(summary,groups,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    ordered=list(LABELS)
    s=summary.set_index('experiment').loc[ordered]
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.0),sharey=True)
    y=np.arange(len(s));colors=['#235789']*7+['#B7791F']+['#34755D']*2
    for ax,metric,title in zip(axes,['auroc','average_precision'],['AUROC','Average precision (AP)']):
        ax.barh(y,s[metric+'_mean'],xerr=s[metric+'_sd'],color=colors,alpha=.88,capsize=3)
        ax.set_xlim(0,1.03);ax.set_xlabel(title);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        for i,(mean,sd) in enumerate(zip(s[metric+'_mean'],s[metric+'_sd'])):
            ax.text(.02,i,f'{mean:.3f} +/- {sd:.3f}',va='center',color='white',fontsize=9)
    axes[0].set_yticks(y,s.label);axes[0].invert_yaxis()
    fig.suptitle('Five-fold held-out performance; error bars show fold SD',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ['png','pdf','svg']:fig.savefig(output/f'figure_performance.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    panels=[('Fusion controls',['pair__random__drw__'+x for x in ['residual','concat_matched','concat_linear','sequence_only','topology_only']]),
            ('Walk controls',['pair__random__'+g+'__residual' for g in ['drw','deepwalk','node2vec']]),
            ('Protein-disjoint evaluation',['protein__random__drw__'+x for x in ['residual','sequence_only']])]
    for name,keys in panels:
        fig,axes=plt.subplots(1,2,figsize=(8,3.9))
        for key in keys:
            frame=pd.concat(groups[key],ignore_index=True);ytrue=frame.label.values;p=frame.probability.values
            fpr,tpr,_=roc_curve(ytrue,p);precision,recall,_=precision_recall_curve(ytrue,p)
            label=LABELS[key].replace('Protein-disjoint: ','')
            axes[0].plot(fpr,tpr,label=label,linewidth=1.5);axes[1].plot(recall,precision,label=label,linewidth=1.5)
        axes[0].plot([0,1],[0,1],ls='--',color='.6',lw=.8)
        axes[0].set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.02))
        axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
        for ax in axes:ax.legend(loc='lower left',fontsize=10);ax.grid(alpha=.15)
        fig.suptitle(name+' | pooled held-out scores (descriptive)',fontsize=12);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(output/('figure_'+name.lower().replace(' ','_')+'.'+ext),dpi=200,bbox_inches='tight')
        plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,default=HERE/'runs/reviewer_v1')
    p.add_argument('--prepared',type=Path,default=HERE/'prepared');p.add_argument('--output',type=Path,default=HERE/'results')
    args=p.parse_args();print(collect(args.campaign,args.prepared,args.output).to_string(index=False))
