"""Fold-specific walks and train-graph-only skip-gram with negative sampling."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import random
import subprocess
import tempfile
import numpy as np

@dataclass
class GraphConfig:
    method: str = 'drw'
    dim: int = 64
    walks_per_node: int = 10
    walk_length: int = 20
    window: int = 5
    epochs: int = 5
    negative_samples: int = 5
    p: float = 1.0
    q: float = 0.5

def fingerprint(edges, n_nodes, config, seed):
    payload=np.ascontiguousarray(edges,dtype=np.int64).tobytes()
    payload+=json.dumps({'config':asdict(config),'seed':seed,'n_nodes':n_nodes},sort_keys=True).encode()
    payload+=Path(__file__).read_bytes()
    payload+=Path(__file__).with_name('legacy_skipgram.py').read_bytes()
    return hashlib.sha256(payload).hexdigest()

def make_walks(edges, n_nodes, config, seed):
    if config.method not in {'drw','deepwalk','node2vec'}:
        raise ValueError(f'Unknown graph method {config.method}')
    if config.p<=0 or config.q<=0: raise ValueError('node2vec p and q must be positive')
    if n_nodes<=0 or config.walks_per_node<=0 or config.walk_length<2:
        raise ValueError('Invalid node count or walk budget')
    adj=[set() for _ in range(n_nodes)]
    for u,v in edges:
        if not (0<=u<n_nodes and 0<=v<n_nodes):raise ValueError('Node outside graph')
        if u==v:raise ValueError('Self edge')
        adj[int(u)].add(int(v));adj[int(v)].add(int(u))
    neighbors=[sorted(x) for x in adj]
    active=[i for i,x in enumerate(neighbors) if x]
    rng=random.Random(seed); walks=[]
    for _ in range(config.walks_per_node):
        order=active.copy();rng.shuffle(order)
        for start in order:
            walk=[start];visited={start}
            for __ in range(config.walk_length-1):
                current=walk[-1];candidates=neighbors[current]
                if not candidates:break
                if config.method=='drw':
                    fresh=[x for x in candidates if x not in visited]
                    nxt=rng.choice(fresh or candidates)
                elif config.method=='node2vec' and len(walk)>1:
                    previous=walk[-2]
                    weights=[1/config.p if x==previous else (1.0 if x in adj[previous] else 1/config.q) for x in candidates]
                    nxt=rng.choices(candidates,weights=weights,k=1)[0]
                else:nxt=rng.choice(candidates)
                walk.append(nxt);visited.add(nxt)
            walks.append(walk)
    return walks,np.array([bool(x) for x in neighbors],dtype=bool)

def train_topology(edges,n_nodes,config,seed,device='cpu',cache_dir=None):
    edges=np.asarray(edges,dtype=np.int64)
    identity=fingerprint(edges,n_nodes,config,seed)
    cache=Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True,exist_ok=True)
        path=cache/(identity+'.npz')
        if path.exists():
            metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if metadata['npz_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Graph cache content changed')
            with np.load(path,allow_pickle=False) as z:
                if str(z['fingerprint'])!=identity:raise ValueError('Cache identity mismatch')
                if z['embedding'].shape!=(n_nodes,config.dim):raise ValueError('Cache shape mismatch')
                return z['embedding'].copy(),z['present'].copy(),identity
    walks,present=make_walks(edges,n_nodes,config,seed)
    helper=Path(__file__).with_name('legacy_skipgram.py')
    interpreter=os.environ.get('CAPPI_GENSIM_PYTHON',r'D:\Anaconda\envs\tf\python.exe')
    print(f'GRAPH {config.method}: {len(walks)} walks; gensim 3.1 Word2Vec, workers=1',flush=True)
    with tempfile.TemporaryDirectory(prefix='cappi_walks_') as folder:
        tmp_dir=Path(folder);walk_path=tmp_dir/'walks.json';embedding_path=tmp_dir/'embedding.npz'
        walk_path.write_text(json.dumps(walks),encoding='utf-8')
        command=[interpreter,str(helper),'--walks',str(walk_path),'--output',str(embedding_path),
                 '--n-nodes',str(n_nodes),'--dim',str(config.dim),'--window',str(config.window),
                 '--epochs',str(config.epochs),'--negative',str(config.negative_samples),'--seed',str(seed)]
        completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if completed.returncode:
            raise RuntimeError(f'Word2Vec failed: {completed.stdout}\n{completed.stderr}')
        with np.load(embedding_path,allow_pickle=False) as z:
            embedding=z['embedding'].copy()
            if not np.array_equal(present,z['present']):raise ValueError('Walk coverage mismatch')
        metadata=json.loads(embedding_path.with_suffix('.json').read_text(encoding='utf-8'))
    if not np.isfinite(embedding).all() or np.any(embedding[~present]!=0):
        raise ValueError('Invalid topology embedding')
    if cache:
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,embedding=embedding,present=present,fingerprint=identity)
        tmp.replace(path)
        path.with_suffix('.json').write_text(json.dumps({'fingerprint':identity,'config':asdict(config),
          'seed':seed,'n_nodes':n_nodes,'n_train_positive_edges':len(edges),'n_walks':len(walks),
          'source':'training positive edges only','skipgram':metadata,
          'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'unseen_or_isolated':'all-zero topology; not assigned a random embedding'},indent=2),encoding='utf-8')
    return embedding,present,identity
