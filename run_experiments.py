"""Reproducible CaPPI training. Original files and historical results are read-only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from sklearn.metrics import (accuracy_score,average_precision_score,confusion_matrix,f1_score,
    matthews_corrcoef,precision_score,recall_score,roc_auc_score,precision_recall_curve,auc)
from cappi_data import prepare_dataset,load_fold
from cappi_graph import GraphConfig,train_topology
from cappi_model import ModelConfig,CaPPI,tokenize_sequences,get_parameter_report

HERE=Path(__file__).resolve().parent
def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8');tmp.replace(path)

def save_checkpoint(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)
def code_hash():
    h=hashlib.sha256()
    for name in ['cappi_data.py','cappi_graph.py','cappi_model.py','legacy_skipgram.py','run_experiments.py']:
        p=HERE/name;h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def metrics(labels,probabilities):
    y=np.asarray(labels,dtype=int);p=np.asarray(probabilities,dtype=float);pred=(p>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    pr,re,_=precision_recall_curve(y,p)
    return {'n':len(y),'positive_n':int(y.sum()),'accuracy':accuracy_score(y,pred),
      'sensitivity':recall_score(y,pred,zero_division=0),'specificity':float(tn/(tn+fp)) if tn+fp else None,
      'precision':precision_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),
      'mcc':matthews_corrcoef(y,pred),'auroc':roc_auc_score(y,p) if len(set(y))==2 else None,
      'average_precision':average_precision_score(y,p),'pr_trapezoid_area':auc(re,pr),'threshold':0.5}

def _load(path,device='cpu'):return torch.load(path,map_location=device,weights_only=False)

def run_fold(args,variant,negative,protocol,graph_method,fold):
    name=f'{protocol}__{negative}__{graph_method}__{variant}__fold{fold}'
    out=Path(args.output)/name
    data=load_fold(Path(args.prepared),negative,protocol,fold)
    model_config=ModelConfig(fusion=variant,max_length=args.max_length,dropout=0.3)
    graph_config=GraphConfig(method=graph_method,epochs=args.graph_epochs)
    run_seed=args.seed+fold
    identity={'code_sha256':code_hash(),'split_hash':data['split_hash'],'seed':run_seed,
      'model':asdict(model_config),'graph':asdict(graph_config),'max_epochs':args.epochs,
      'learning_rate':args.lr,'patience':args.patience,'weight_decay':args.weight_decay,
      'pair_chunk':args.pair_chunk,'device':args.device,'torch':str(torch.__version__),'numpy':np.__version__,
      'protocol':protocol,'negative_sampling':negative,'variant':variant,'fold':fold,
      'optimizer':'AdamW; explicit weight decay replaces old activity regularization','selection':'minimum validation BCE; held-out metadata audited; test labels/scores never used for training or selection',
      'hyperparameters':'fixed before campaign; no search or optimality claim',
      'epoch_definition':'one full training-set gradient update with loss-weighted pair chunks',
      'node_encoder_training':'GCNN, topology projection, fusion and pair head jointly optimized',
      'changes_from_legacy':['one-hot plus unknown channel and masked pooling','joint upstream supervised training','symmetric endpoint prediction','explicit AdamW weight decay','full training loss per optimizer step','true outer folds and train-only topology'],
      'topology_training':'fold training positive graph only; frozen SGNS embeddings',
      'inductive_policy':'zero topology for unseen/isolated nodes; sequence available; no test edges',
      'evaluation_enabled':not args.no_test}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
    if (out/'completed.json').exists():
        previous=json.loads((out/'completed.json').read_text(encoding='utf-8'))
        if previous['run_signature']!=signature:raise ValueError(f'Incompatible previous output: {out}')
        if sha(out/'best.pt')!=previous['checkpoint_sha256']:raise ValueError('Completed checkpoint content changed')
        if previous.get('test_evaluated') and sha(out/'test_predictions.csv')!=previous['predictions_sha256']:
            raise ValueError('Completed predictions content changed')
        print('SKIP VERIFIED COMPLETE',name,flush=True);return previous
    if out.exists() and not args.resume:raise FileExistsError(f'Use new output or --resume: {out}')
    out.mkdir(parents=True,exist_ok=True)
    existing=out/'run_config.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))['run_signature']!=signature:
        raise ValueError('Cannot resume incompatible run')
    write_json(existing,{**identity,'run_signature':signature,'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'device':args.device})
    device=torch.device(args.device);seed_all(run_seed)
    graph_edges=data['graph_edges']
    # Defense in depth: graph must be exactly the training positives.
    observed={tuple(map(int,x)) for x in graph_edges}
    expected={tuple(map(int,x)) for x,y in zip(data['train_pairs'],data['train_labels']) if y==1}
    if observed!=expected:raise ValueError('Graph is not exactly the training positive edge set')
    for key in ['val_pairs','test_pairs']:
        if observed & {tuple(map(int,x)) for x in data[key]}:raise ValueError('Held-out pair entered graph')
    embedding,present,graph_hash=train_topology(graph_edges,data['n_nodes'],graph_config,run_seed,str(device),Path(args.output)/'graph_cache')
    # Train-only centering/scaling; retain literal zero for absent node topology.
    train_node_ids=np.asarray(data['train_nodes'],dtype=int)
    fitting=train_node_ids[present[train_node_ids]]
    mean=embedding[fitting].mean(0) if len(fitting) else np.zeros(model_config.topo_dim)
    std=embedding[fitting].std(0) if len(fitting) else np.ones(model_config.topo_dim)
    embedding=(embedding-mean)/np.maximum(std,1e-6);embedding[~present]=0
    np.savez_compressed(out/'topology.npz',embedding=embedding,present=present,mean=mean,std=std,graph_hash=graph_hash)
    tokens=tokenize_sequences(data['sequences'],args.max_length).to(device)
    topology=torch.as_tensor(embedding,dtype=torch.float32,device=device)
    topo_present=torch.as_tensor(present,dtype=torch.bool,device=device)
    train_pairs=torch.as_tensor(data['train_pairs'],dtype=torch.long,device=device)
    train_y=torch.as_tensor(data['train_labels'],dtype=torch.float32,device=device)
    val_pairs=torch.as_tensor(data['val_pairs'],dtype=torch.long,device=device)
    val_y=torch.as_tensor(data['val_labels'],dtype=torch.float32,device=device)
    if len(train_y)==0 or len(val_y)==0 or len(set(data['val_labels']))<2 or len(set(data['train_labels']))<2:raise ValueError('Empty or one-class train/validation split')
    seed_all(run_seed)
    model=CaPPI(model_config).to(device)
    write_json(out/'parameters.json',get_parameter_report(model))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=10)
    history=[];best=float('inf');stale=0;start_epoch=0;best_epoch=None;start_time=time.monotonic()
    last=out/'last.pt'
    if args.resume and last.exists():
        ck=_load(last,device)
        if ck['run_signature']!=signature or ck['graph_hash']!=graph_hash:raise ValueError('Checkpoint provenance mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];best=ck['best'];stale=ck['stale'];start_epoch=ck['epoch']+1;best_epoch=ck['best_epoch']
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in ck['rng_cuda']])
    print(f'START {name} train={len(train_y)} val={len(val_y)} nodes={data["n_nodes"]}',flush=True)
    for epoch in range(start_epoch,args.epochs if stale<args.patience else start_epoch):
        model.train();optimizer.zero_grad(set_to_none=True)
        node_vectors=model.encode_nodes(tokens,topology,topo_present)
        # One encoder pass; accumulate loss-weighted chunks before a single update.
        order=torch.randperm(len(train_pairs),device=device)
        loss_terms=[];loss_sum=0.0
        for start in range(0,len(order),args.pair_chunk):
            indices=order[start:start+args.pair_chunk]
            logits=model.score_pairs(node_vectors,train_pairs[indices])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,train_y[indices],reduction='sum')/len(train_y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss_terms.append(loss);loss_sum+=float(loss.detach())
        torch.stack(loss_terms).sum().backward()
        del loss_terms
        if epoch==start_epoch:
            grads={n:float(p.grad.norm().detach()) if p.grad is not None else None for n,p in model.named_parameters()}
            write_json(out/'first_step_gradient_norms.json',grads)
            if any(g is not None and not np.isfinite(g) for g in grads.values()):raise FloatingPointError('Nonfinite gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
        del node_vectors
        model.eval()
        with torch.no_grad():
            nv=model.encode_nodes(tokens,topology,topo_present)
            val_logits=torch.cat([model.score_pairs(nv,val_pairs[i:i+args.pair_chunk]) for i in range(0,len(val_pairs),args.pair_chunk)])
            val_loss=float(torch.nn.functional.binary_cross_entropy_with_logits(val_logits,val_y))
            val_ap=average_precision_score(data['val_labels'],torch.sigmoid(val_logits).cpu().numpy())
        if not np.isfinite(val_loss):raise FloatingPointError('Nonfinite validation loss')
        scheduler.step(val_loss)
        improved=val_loss<best-1e-6
        if improved:best=val_loss;best_epoch=epoch+1;stale=0
        else:stale+=1
        history.append({'epoch':epoch+1,'train_bce':loss_sum,'validation_bce':val_loss,'validation_ap':val_ap,'lr':optimizer.param_groups[0]['lr']})
        ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
          'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'config':identity,
          'run_signature':signature,'graph_hash':graph_hash,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
          'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        if improved:save_checkpoint(ck,out/'best.pt')
        if (epoch+1)%5==0 or stale>=args.patience or epoch+1==args.epochs:save_checkpoint(ck,last)
        write_json(out/'history.json',history)
        if epoch==start_epoch or (epoch+1)%10==0 or stale>=args.patience:
            print(f'{name} epoch={epoch+1} train={loss_sum:.5f} val={val_loss:.5f} valAP={val_ap:.5f} stale={stale} elapsed={time.monotonic()-start_time:.1f}s',flush=True)
        if stale>=args.patience:break
    best_path=out/'best.pt';best_ck=_load(best_path,device)
    if best_ck['run_signature']!=signature or best_ck['graph_hash']!=graph_hash:raise ValueError('Best checkpoint provenance mismatch')
    model.load_state_dict(best_ck['model']);model.eval()
    checkpoint_hash=sha(best_path)
    result={'run_signature':signature,'name':name,'best_epoch':best_epoch,'epochs_completed':len(history),
      'best_validation_bce':best,'elapsed_seconds':time.monotonic()-start_time,'graph_hash':graph_hash,
      'split_hash':data['split_hash'],'checkpoint_sha256':checkpoint_hash,'test_evaluated':False,'config':identity}
    if not args.no_test:
        test_pairs=torch.as_tensor(data['test_pairs'],dtype=torch.long,device=device)
        if len(test_pairs)==0 or len(set(data['test_labels']))<2:raise ValueError('Empty or one-class test set')
        with torch.no_grad():
            nv=model.encode_nodes(tokens,topology,topo_present)
            probs=torch.cat([torch.sigmoid(model.score_pairs(nv,test_pairs[i:i+args.pair_chunk])) for i in range(0,len(test_pairs),args.pair_chunk)]).cpu().numpy()
        result['test_metrics']=metrics(data['test_labels'],probs);result['test_evaluated']=True
        with (out/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['protein_u','protein_v','label','probability','prediction','fold','seed','split_hash','checkpoint_sha256'])
            for (u,v),y,p in zip(data['test_pairs'],data['test_labels'],probs):
                writer.writerow([data['node_ids'][u],data['node_ids'][v],int(y),float(p),int(p>=.5),fold,run_seed,data['split_hash'],checkpoint_hash])
        result['predictions_sha256']=sha(out/'test_predictions.csv')
    write_json(out/'completed.json',result)
    print('COMPLETE',name,json.dumps(result.get('test_metrics',{'test':'not evaluated'})),flush=True)
    del model,optimizer,best_ck,tokens,topology,train_pairs,train_y,val_pairs,val_y
    if device.type=='cuda':torch.cuda.empty_cache()
    return result

MATRICES={
 'main':[('residual','random','pair','drw')],
 'reviewer':[(x,'random','pair','drw') for x in ['residual','concat_matched','concat_linear','sequence_only','topology_only']]
    +[('residual','random','pair',g) for g in ['deepwalk','node2vec']]
    +[('residual','degree_matched','pair','drw')]
    +[(v,'random','protein','drw') for v in ['residual','sequence_only']],
}
def summarize(output):
    groups={}
    for p in Path(output).glob('*/completed.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        if not r.get('test_evaluated'):continue
        key=r['name'].rsplit('__fold',1)[0];groups.setdefault(key,[]).append(r)
    summary=[]
    for key,rs in sorted(groups.items()):
        policies=[{k:v for k,v in r['config'].items() if k not in {'seed','fold','split_hash'}} for r in rs]
        if any(x!=policies[0] for x in policies[1:]):raise ValueError(f'Mixed training policies in summary: {key}')
        if len({r['config']['fold'] for r in rs})!=len(rs):raise ValueError('Duplicate fold in summary')
        row={'experiment':key,'completed_folds':len(rs)}
        for metric in ['accuracy','sensitivity','specificity','precision','f1','mcc','auroc','average_precision','pr_trapezoid_area']:
            values=np.array([r['test_metrics'][metric] for r in rs],float)
            row[metric+'_mean']=float(values.mean());row[metric+'_sd']=float(values.std(ddof=1)) if len(values)>1 else None
        summary.append(row)
    write_json(Path(output)/'summary.json',summary)
    if summary:
        with (Path(output)/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(HERE.parent));ap.add_argument('--prepared',default=str(HERE/'prepared'))
    ap.add_argument('--output',default=str(HERE/'runs/reviewer_v1'))
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--matrix',choices=list(MATRICES),default='main');ap.add_argument('--folds',default='0,1,2,3,4')
    ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--graph-epochs',type=int,default=5);ap.add_argument('--lr',type=float,default=.0005)
    ap.add_argument('--weight-decay',type=float,default=.0001);ap.add_argument('--max-length',type=int,default=1500)
    ap.add_argument('--pair-chunk',type=int,default=4096);ap.add_argument('--seed',type=int,default=20260930)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--no-test',action='store_true')
    args=ap.parse_args();torch.set_num_threads(min(8,os.cpu_count() or 1))
    if min(args.epochs,args.patience,args.pair_chunk,args.graph_epochs,args.max_length)<=0:raise ValueError('Budgets must be positive')
    if args.lr<=0 or args.weight_decay<0:raise ValueError('Invalid optimization settings')
    folds=list(map(int,args.folds.split(',')))
    if len(folds)!=len(set(folds)) or any(x not in range(5) for x in folds):raise ValueError('Folds must be distinct indices 0..4')
    if args.prepare or args.prepare_only:prepare_dataset(Path(args.root),Path(args.prepared),seed=args.seed,n_folds=5)
    if args.prepare_only:return
    Path(args.output).mkdir(parents=True,exist_ok=True)
    # A campaign may be extended to more folds/conditions, but never mixed with
    # another numerical training policy, source revision or prepared benchmark.
    campaign={k:v for k,v in vars(args).items() if k not in {'folds','matrix','resume','prepare','prepare_only','root','output'}}
    campaign['code_sha256']=code_hash();campaign['prepared_manifest_sha256']=sha(Path(args.prepared)/'manifest.json')
    campaign['torch']=str(torch.__version__);campaign['numpy']=np.__version__
    lock=Path(args.output)/'campaign_config.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))!=campaign:raise ValueError('Incompatible campaign: choose a fresh output directory')
    write_json(lock,campaign)
    for variant,negative,protocol,graph in MATRICES[args.matrix]:
        for fold in folds:
            run_fold(args,variant,negative,protocol,graph,fold);summarize(args.output)
if __name__=='__main__':main()
