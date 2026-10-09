"""Controlled missing-topology training and a separately gated test evaluation.

Original core, prepared folds, historical runs and source graph caches are read-only.
Training loads only train/validation labels; held-out pair IDs are used for audits.
The test evaluator requires a frozen, checkpoint-specific selection declaration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch  # Import first in this Windows runtime.

from cappi_data import NEGATIVE_MODES, PROTOCOLS, SCHEMA_VERSION, _fold_signature, _json_hash
from cappi_graph import GraphConfig, fingerprint, train_topology
from cappi_model import ModelConfig, get_parameter_report, tokenize_sequences
from cappi_missing_topology import MissingTopologyCaPPI, apply_topology_dropout
from run_experiments import _load, code_hash, metrics, save_checkpoint, seed_all, sha, write_json

HERE = Path(__file__).resolve().parent
CORE_SHA = '76593480cfb156c8817d6bd12b522f09b50f4219c28720af2f2e93d89f381e91'
PREPARED_SHA = 'ebef279a77d6c172559b96eb29388d3a41b98751867f3126e1a7db7c07a13d69'
PAIR_CHUNK = 4096
WEIGHT_DECAY = .0001
MASK_SEED_OFFSET = 7919
FAMILIES = ('residual', 'topology_dropout', 'topology_dropout_bypass',
            'sequence_only', 'topology_only', 'concat_matched')


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def repair_code_hash():
    h = hashlib.sha256()
    for name in ('run_missing_topology.py', 'cappi_missing_topology.py'):
        h.update(name.encode()); h.update((HERE / name).read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class RepairConfig:
    family: str = 'residual'
    learning_rate: float = .0005
    topology_dropout: float = 0.
    missing_policy: str = 'original'

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f'Unsupported repair family: {self.family}')
        if self.learning_rate not in (.0005, .001):
            raise ValueError('The controlled learning-rate grid is .0005 or .001')
        dropout_family = self.family in ('topology_dropout', 'topology_dropout_bypass')
        # Make the family defaults explicit in serialized settings.
        if dropout_family and self.topology_dropout == 0:
            object.__setattr__(self, 'topology_dropout', .5)
        if self.family == 'topology_dropout_bypass' and self.missing_policy == 'original':
            object.__setattr__(self, 'missing_policy', 'bypass')
        expected_probability = .5 if dropout_family else 0.
        expected_policy = 'bypass' if self.family == 'topology_dropout_bypass' else 'original'
        if (isinstance(self.topology_dropout, bool) or self.topology_dropout != expected_probability
                or self.missing_policy != expected_policy):
            raise ValueError('Family requires its predeclared dropout (0 or .5) and missing policy')

    @property
    def name(self):
        return f'{self.family}__lr{self.learning_rate:.4f}'.replace('.', 'p')

    @property
    def model(self):
        fusion = 'residual' if self.family.startswith('topology_dropout') else self.family
        return ModelConfig(fusion=fusion, max_length=1500, dropout=.3)

    @property
    def graph(self):
        return GraphConfig(method='drw', dim=64, walk_length=20, epochs=5)


def load_validation_only(prepared: Path, protocol: str, negative: str, fold: int):
    """Verify an immutable fold without decompressing the test label member."""
    prepared = Path(prepared).resolve()
    if protocol not in PROTOCOLS or negative not in NEGATIVE_MODES:
        raise ValueError('Unknown protocol or negative sampling')
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Prepared schema mismatch')
    request = {k: manifest.get(k) for k in ('schema_version', 'seed', 'n_folds', 'source_sha256')}
    if (_json_hash(request) != manifest.get('request_hash')
            or _json_hash({'request': request, 'files': manifest['files']}) != manifest.get('dataset_hash')):
        raise ValueError('Prepared manifest identity mismatch')
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(manifest['n_folds']):
        raise ValueError('Fold index out of range')
    relative = f'{negative}/{protocol}/fold_{fold}'
    folder = prepared / relative
    names = ['nodes.csv', 'positives.npz'] + [f'{relative}/{x}.npz'
             for x in ('train', 'validation', 'test', 'graph')]
    for name in names:
        if sha(prepared / name) != manifest['files'][name]:
            raise ValueError(f'Prepared artifact changed: {name}')
    split_hash = manifest['folds'][relative]['split_hash']
    if _fold_signature(prepared, relative, manifest['files']) != split_hash:
        raise ValueError('Fold signature mismatch')
    with (prepared / 'nodes.csv').open(encoding='utf-8', newline='') as handle:
        nodes = list(csv.DictReader(handle))
    if (len(nodes) != manifest['n_nodes']
            or [int(x['index']) for x in nodes] != list(range(len(nodes)))):
        raise ValueError('Node mapping mismatch')
    result = {'node_ids': np.asarray([int(x['node_id']) for x in nodes], dtype=np.int64),
              'sequences': [x['sequence'] for x in nodes], 'n_nodes': len(nodes),
              'split_hash': split_hash, 'manifest': manifest}
    if len(set(result['node_ids'].tolist())) != len(nodes):
        raise ValueError('Duplicate original node IDs')
    for name, prefix in (('train', 'train'), ('validation', 'val')):
        with np.load(folder / f'{name}.npz', allow_pickle=False) as z:
            result[f'{prefix}_pairs'] = z['pairs'].copy()
            result[f'{prefix}_labels'] = z['labels'].copy()
    with np.load(folder / 'test.npz', allow_pickle=False) as z:
        # NPZ is lazy. Explicitly access only the pair IDs, never z['labels'].
        result['test_pairs_ids'] = z['pairs'].copy()
    with np.load(folder / 'graph.npz', allow_pickle=False) as z:
        result['graph_edges'] = z['graph_edges'].copy()
        for key in ('train_nodes', 'val_nodes', 'test_nodes'):
            result[key] = z[key].copy()
    with np.load(prepared / 'positives.npz', allow_pickle=False) as z:
        known = set(map(tuple, z['pairs'].tolist()))
    if len(known) != manifest['n_positives']:
        raise ValueError('Positive pool count mismatch')
    sets = {}
    for prefix in ('train', 'val', 'test'):
        pairs = result[f'{prefix}_pairs' if prefix != 'test' else 'test_pairs_ids']
        if (pairs.dtype != np.int64 or pairs.ndim != 2 or pairs.shape[1] != 2
                or np.any(pairs[:, 0] >= pairs[:, 1]) or np.any(pairs < 0)
                or np.any(pairs >= len(nodes))):
            raise ValueError(f'Invalid canonical {prefix} pairs')
        sets[prefix] = set(map(tuple, pairs.tolist()))
        if len(sets[prefix]) != len(pairs):
            raise ValueError(f'Duplicate {prefix} pair')
        if prefix != 'test':
            labels = result[f'{prefix}_labels']
            if (labels.dtype != np.float32 or labels.shape != (len(pairs),)
                    or set(labels.tolist()) != {0., 1.}):
                raise ValueError(f'Empty, malformed or one-class {prefix} labels')
            if any((tuple(p) in known) != bool(y) for p, y in zip(pairs, labels)):
                raise ValueError('Training/validation labels contradict known positives')
        node_set = result[f'{prefix}_nodes']
        if (not np.issubdtype(node_set.dtype, np.integer) or node_set.ndim != 1
                or len(set(node_set.tolist())) != len(node_set)
                or np.any(node_set < 0) or np.any(node_set >= len(nodes))):
            raise ValueError('Invalid node partition')
        if protocol == 'protein' and not set(pairs.reshape(-1)).issubset(set(node_set)):
            raise ValueError('Protein membership violation')
    if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Pair leakage across folds')
    if protocol == 'protein' and any(set(result[f'{a}_nodes']) & set(result[f'{b}_nodes'])
                                     for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Protein leakage across partitions')
    edges = result['graph_edges']
    expected = {tuple(p) for p, y in zip(result['train_pairs'], result['train_labels']) if y == 1}
    if (edges.dtype != np.int64 or edges.shape != (len(expected), 2)
            or set(map(tuple, edges.tolist())) != expected
            or expected & (sets['val'] | sets['test'])):
        raise ValueError('Graph is not exactly training positives or includes held-out pair IDs')
    return result


def environment(device):
    device = torch.device(device)
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'torch': str(torch.__version__), 'numpy': np.__version__,
            'sklearn': importlib.metadata.version('scikit-learn'), 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'torch_threads': torch.get_num_threads(), 'cudnn_deterministic': True,
            'cudnn_benchmark': False}


def _safe_output(prepared, output):
    out = Path(output).resolve()
    forbidden = [Path(prepared).resolve(), HERE / 'runs' / 'reviewer_v1',
                 HERE / 'runs' / 'supplement_20261001', HERE / 'results']
    for path in forbidden:
        if out == path or path in out.parents:
            raise ValueError(f'Output would modify protected original artifacts: {out}')
    return out


def _copy_graph_cache(source, destination, graph_hash):
    """Copy only a content-verified cache pair; never create or modify source."""
    destination.mkdir(parents=True, exist_ok=True)
    if source is None:
        return None
    source = Path(source).resolve()
    if source == destination.resolve():
        raise ValueError('Source graph cache must be separate and read-only')
    src_npz = source / f'{graph_hash}.npz'
    src_json = source / f'{graph_hash}.json'
    if not src_npz.exists() and not src_json.exists():
        return None
    if not src_npz.is_file() or not src_json.is_file():
        raise ValueError('Incomplete source graph cache')
    meta = json.loads(src_json.read_text(encoding='utf-8'))
    if meta.get('fingerprint') != graph_hash or meta.get('npz_sha256') != sha(src_npz):
        raise ValueError('Source graph cache identity or content mismatch')
    provenance = {}
    for src in (src_npz, src_json):
        dst = destination / src.name
        if dst.exists() and sha(dst) != sha(src):
            raise ValueError('Existing copied graph cache differs from source')
        if not dst.exists():
            temp = dst.with_suffix(dst.suffix + '.copy.tmp')
            shutil.copyfile(src, temp)
            if sha(temp) != sha(src):
                raise ValueError('Graph cache copy checksum mismatch')
            temp.replace(dst)
        provenance[src.name] = {'source_path': str(src), 'source_sha256': sha(src),
                                'destination_path': str(dst), 'destination_sha256': sha(dst)}
    return provenance


def _topology(data, config, graph_seed, device, out, source_graph_cache):
    graph_hash = fingerprint(data['graph_edges'], data['n_nodes'], config.graph, graph_seed)
    cache = out.parent / 'graph_cache'
    provenance = _copy_graph_cache(source_graph_cache, cache, graph_hash)
    raw, present, returned_hash = train_topology(data['graph_edges'], data['n_nodes'],
                                                config.graph, graph_seed, str(device), cache)
    if (returned_hash != graph_hash or raw.shape != (data['n_nodes'], config.model.topo_dim)
            or present.dtype != bool or present.shape != (data['n_nodes'],)
            or not np.isfinite(raw).all() or np.any(raw[~present] != 0)):
        raise ValueError('Invalid raw topology or graph identity')
    fitting = data['train_nodes'][present[data['train_nodes']]]
    mean = raw[fitting].mean(0) if len(fitting) else np.zeros(config.model.topo_dim)
    std = raw[fitting].std(0) if len(fitting) else np.ones(config.model.topo_dim)
    embedding = (raw - mean) / np.maximum(std, 1e-6)
    embedding[~present] = 0
    topology = {'raw_embedding': raw, 'embedding': embedding, 'present': present,
                'mean': mean, 'std': std, 'fitting_nodes': fitting,
                'graph_hash': np.asarray(graph_hash)}
    path = out / 'topology.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(topology) or any(not np.array_equal(z[k], v) for k, v in topology.items()):
                raise ValueError('Resumed normalized topology differs from immutable train graph')
    else:
        np.savez_compressed(path, **topology)
    write_json(out / 'graph_cache_provenance.json', {'graph_hash': graph_hash,
               'source_cache_reused': bool(provenance), 'copied_artifacts': provenance,
               'local_cache': str(cache.resolve()), 'source_cache_modified': False})
    return topology, graph_hash


def mask_coverage(base_present, effective_present, pairs, labels, epoch, probability):
    """A descriptive audit computed after label-independent node masking."""
    base = np.asarray(base_present, dtype=bool)
    effective = np.asarray(effective_present, dtype=bool)
    if effective.shape != base.shape or np.any(effective & ~base):
        raise ValueError('Mask cannot create unavailable topology')
    result = {'epoch': int(epoch), 'probability': probability, 'n_nodes': len(base),
              'original_present_nodes': int(base.sum()), 'effective_present_nodes': int(effective.sum()),
              'artificially_masked_nodes': int((base & ~effective).sum()), 'by_label': {}}
    for label, name in ((1., 'positive'), (0., 'negative')):
        selected = pairs[labels == label]
        original = base[selected].sum(axis=1)
        current = effective[selected].sum(axis=1)
        result['by_label'][name] = {'n_pairs': len(selected),
            'original_present_endpoints': int(original.sum()),
            'effective_present_endpoints': int(current.sum()),
            'artificially_masked_endpoints': int((original - current).sum()),
            **{f'original_{count}_present': int((original == count).sum()) for count in (0, 1, 2)},
            **{f'effective_{count}_present': int((current == count).sum()) for count in (0, 1, 2)}}
    return result


def _write_masks(out, records, traces, base):
    write_json(out / 'mask_audit.json', records)
    np.savez_compressed(out / 'mask_trace.npz', epochs=np.asarray([r['epoch'] for r in records], dtype=np.int64),
                        effective_present=np.asarray(traces, dtype=bool), base_present=base)


def _infer(model, tokens, topology, present, pairs, labels=None):
    model.eval()
    with torch.no_grad():
        vectors = model.encode_nodes(tokens, topology, present)
        logits = torch.cat([model.score_pairs(vectors, pairs[i:i + PAIR_CHUNK])
                            for i in range(0, len(pairs), PAIR_CHUNK)])
        loss = None if labels is None else float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    if (loss is not None and not np.isfinite(loss)) or not np.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite inference output')
    return probabilities, loss


def _write_predictions(path, data, pairs, labels, probabilities, identity, checkpoint_hash, partition):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['protein_u', 'protein_v', 'label', 'probability', 'prediction', 'fold',
                         'model_seed', 'graph_seed', 'split_hash', 'checkpoint_sha256', 'partition'])
        for (u, v), y, p in zip(pairs, labels, probabilities):
            writer.writerow([data['node_ids'][u], data['node_ids'][v], int(y), float(p), int(p >= .5),
                identity['fold'], identity['model_seed'], identity['graph_seed'], data['split_hash'],
                checkpoint_hash, partition])


def verify_completed(out, signature=None):
    out = Path(out)
    result = json.loads((out / 'completed.json').read_text(encoding='utf-8'))
    if signature is not None and result['run_signature'] != signature:
        raise ValueError('Completed run identity mismatch')
    if result['test_evaluated'] or result['evaluation_partition'] != 'validation' or result['test_labels_or_scores_loaded']:
        raise ValueError('Training completion must be validation-only')
    for name, expected in result['artifact_sha256'].items():
        if sha(out / name) != expected:
            raise ValueError(f'Completed artifact changed: {name}')
    config = json.loads((out / 'run_config.json').read_text(encoding='utf-8'))
    identity = {k: v for k, v in config.items() if k != 'run_signature'}
    if digest(identity) != result['run_signature'] or config['run_signature'] != result['run_signature']:
        raise ValueError('Completed run_config signature mismatch')
    return result


def run_repair_fold(prepared: Path, output: Path, config: RepairConfig, protocol: str,
                    negative: str, fold: int, model_seed: int, graph_seed: int,
                    device='cuda', epochs=300, patience=20, resume=True,
                    source_graph_cache: Path | None = None, no_test=True):
    """Train one exact run directory. Selection and all test labels stay external."""
    if not isinstance(config, RepairConfig):
        raise TypeError('config must be RepairConfig')
    if not no_test:
        raise ValueError('Use the separately gated evaluate_selected_run after frozen selection')
    if (isinstance(epochs, bool) or isinstance(patience, bool) or not isinstance(epochs, int)
            or not isinstance(patience, int) or min(epochs, patience) <= 0):
        raise ValueError('Positive integer training budgets required')
    for seed in (model_seed, graph_seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32 - MASK_SEED_OFFSET:
            raise ValueError('Integer seeds must be nonnegative and within the RNG range')
    if code_hash() != CORE_SHA or sha(Path(prepared) / 'manifest.json') != PREPARED_SHA:
        raise ValueError('Original core or prepared manifest changed; refuse mixed protocols')
    started = time.monotonic()
    data = load_validation_only(prepared, protocol, negative, fold)
    out = _safe_output(prepared, output)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    identity = {'schema': 'cappi-missing-topology-run-v1', 'core_code_sha256': code_hash(),
        'repair_code_sha256': repair_code_hash(), 'prepared_manifest_sha256': sha(Path(prepared) / 'manifest.json'),
        'settings': asdict(config), 'model': asdict(config.model), 'graph': asdict(config.graph),
        'prepared': str(Path(prepared).resolve()), 'split_hash': data['split_hash'],
        'protocol': protocol, 'negative_sampling': negative, 'fold': fold,
        'model_seed': model_seed, 'graph_seed': graph_seed, 'mask_seed': model_seed + MASK_SEED_OFFSET,
        'max_epochs': epochs, 'patience': patience, 'learning_rate': config.learning_rate,
        'weight_decay': WEIGHT_DECAY, 'pair_chunk': PAIR_CHUNK, 'gradient_clip_norm': 5.,
        'optimizer': 'AdamW', 'scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': .5, 'patience': 10},
        'selection': 'minimum validation BCE with improvement > 1e-6', 'threshold': .5,
        'epoch_definition': 'one encoder training pass and one loss-weighted full-training-set optimizer update',
        'masking': 'node-wise independent Bernoulli retention on originally present topology only, independent of labels; separate torch.Generator',
        'evaluation_mask': 'original availability; no stochastic evaluation masking',
        'topology_training': 'frozen SGNS on training-positive graph; centering/scaling uses present training nodes only',
        'test_labels_or_scores_loaded': False, 'evaluation_partition': 'validation',
        'device': str(device), 'environment': environment(device)}
    signature = digest(identity)
    if (out / 'completed.json').exists():
        return verify_completed(out, signature)
    if out.exists() and not resume:
        raise FileExistsError('Choose a new output or enable resume')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_config.json').exists():
        saved = json.loads((out / 'run_config.json').read_text(encoding='utf-8'))
        if saved.get('run_signature') != signature:
            raise ValueError('Cannot resume with changed settings, code, data or environment')
    write_json(out / 'run_config.json', {**identity, 'run_signature': signature})
    write_json(out / 'data_audit.json', {'split_hash': data['split_hash'], 'n_nodes': data['n_nodes'],
        'training_pairs': len(data['train_pairs']), 'validation_pairs': len(data['val_pairs']),
        'heldout_pair_ids_audited': len(data['test_pairs_ids']), 'test_labels_or_scores_loaded': False,
        'graph_edges': len(data['graph_edges']), 'graph_is_exactly_training_positives': True,
        'protein_disjointness_checked': protocol == 'protein'})
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    topology, graph_hash = _topology(data, config, graph_seed, device, out, source_graph_cache)
    tokens = tokenize_sequences(data['sequences'], config.model.max_length).to(device)
    topo = torch.as_tensor(topology['embedding'], dtype=torch.float32, device=device)
    present = torch.as_tensor(topology['present'], dtype=torch.bool, device=device)
    train_pairs = torch.as_tensor(data['train_pairs'], dtype=torch.long, device=device)
    train_y = torch.as_tensor(data['train_labels'], dtype=torch.float32, device=device)
    val_pairs = torch.as_tensor(data['val_pairs'], dtype=torch.long, device=device)
    val_y = torch.as_tensor(data['val_labels'], dtype=torch.float32, device=device)
    seed_all(model_seed)
    mask_generator = torch.Generator(device=device).manual_seed(model_seed + MASK_SEED_OFFSET)
    model = MissingTopologyCaPPI(config.model, missing_policy=config.missing_policy).to(device)
    write_json(out / 'parameters.json', get_parameter_report(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=10)
    history, records, traces = [], [], []
    best, stale, start_epoch, best_epoch, previous_elapsed = float('inf'), 0, 0, None, 0.
    last_path = out / 'last.pt'
    if resume and last_path.exists():
        ck = _load(last_path, device)
        if ck['run_signature'] != signature or ck['graph_hash'] != graph_hash:
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        history, records, traces = ck['history'], ck['mask_records'], ck['mask_trace']
        best, stale, best_epoch, start_epoch = ck['best'], ck['stale'], ck['best_epoch'], ck['epoch'] + 1
        previous_elapsed = ck.get('elapsed_seconds', 0.)
        random.setstate(ck['rng_python']); np.random.set_state(ck['rng_numpy'])
        torch.set_rng_state(ck['rng_torch'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in ck['rng_cuda']])
        mask_generator.set_state(ck['mask_generator_state'].cpu())
    training_started = time.monotonic()
    print(f'START {out.name} family={config.family} train={len(train_y)} val={len(val_y)} seed={model_seed}', flush=True)
    for epoch in range(start_epoch, epochs if stale < patience else start_epoch):
        model.train(); optimizer.zero_grad(set_to_none=True)
        effective = apply_topology_dropout(present, config.topology_dropout, generator=mask_generator)
        trace = effective.detach().cpu().numpy().copy()
        records.append(mask_coverage(topology['present'], trace, data['train_pairs'], data['train_labels'],
                                     epoch + 1, config.topology_dropout))
        traces.append(trace)
        vectors = model.encode_nodes(tokens, topo, effective)
        order = torch.randperm(len(train_pairs), device=device)
        loss_terms, train_loss = [], 0.
        for start in range(0, len(order), PAIR_CHUNK):
            indices = order[start:start + PAIR_CHUNK]
            logits = model.score_pairs(vectors, train_pairs[indices])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y[indices],
                                                                        reduction='sum') / len(train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss_terms.append(loss); train_loss += float(loss.detach())
        torch.stack(loss_terms).sum().backward()
        del loss_terms
        if epoch == 0:
            grads = {n: float(p.grad.norm().detach()) if p.grad is not None else None
                     for n, p in model.named_parameters()}
            write_json(out / 'first_step_gradient_norms.json', grads)
            if any(g is None or not np.isfinite(g) for g in grads.values()):
                raise FloatingPointError('Missing or nonfinite trainable parameter gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step(); del vectors
        probabilities, val_loss = _infer(model, tokens, topo, present, val_pairs, val_y)
        from sklearn.metrics import average_precision_score
        val_ap = average_precision_score(data['val_labels'], probabilities)
        scheduler.step(val_loss)
        improved = val_loss < best - 1e-6
        if improved:
            best, best_epoch, stale = val_loss, epoch + 1, 0
        else:
            stale += 1
        history.append({'epoch': epoch + 1, 'train_bce': train_loss, 'validation_bce': val_loss,
                        'validation_ap': val_ap, 'lr': optimizer.param_groups[0]['lr']})
        ck = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'stale': stale, 'history': history,
            'config': identity, 'run_signature': signature, 'graph_hash': graph_hash,
            'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'mask_generator_state': mask_generator.get_state(), 'mask_records': records, 'mask_trace': traces,
            'elapsed_seconds': previous_elapsed + time.monotonic() - training_started}
        if improved:
            save_checkpoint(ck, out / 'best.pt')
        if (epoch + 1) % 5 == 0 or stale >= patience or epoch + 1 == epochs:
            save_checkpoint(ck, last_path)
            _write_masks(out, records, traces, topology['present'])
        write_json(out / 'history.json', history)
        if epoch == start_epoch or (epoch + 1) % 25 == 0 or stale >= patience:
            print(f'{out.name} epoch={epoch+1} valBCE={val_loss:.6f} stale={stale} elapsed={time.monotonic()-started:.1f}s', flush=True)
        if stale >= patience:
            break
    checkpoint = _load(out / 'best.pt', device)
    if checkpoint['run_signature'] != signature or checkpoint['graph_hash'] != graph_hash:
        raise ValueError('Best checkpoint provenance mismatch')
    model.load_state_dict(checkpoint['model'])
    probabilities, validation_bce = _infer(model, tokens, topo, present, val_pairs, val_y)
    if abs(validation_bce - checkpoint['best']) > 1e-6:
        raise ValueError('Reloaded best checkpoint does not reproduce validation BCE')
    checkpoint_hash = sha(out / 'best.pt')
    _write_predictions(out / 'validation_predictions.csv', data, data['val_pairs'], data['val_labels'],
                       probabilities, identity, checkpoint_hash, 'validation')
    validation_metrics = metrics(data['val_labels'], probabilities)
    write_json(out / 'validation_metrics.json', validation_metrics)
    _write_masks(out, records, traces, topology['present'])
    artifact_names = ['best.pt', 'last.pt', 'run_config.json', 'topology.npz', 'parameters.json',
        'history.json', 'first_step_gradient_norms.json', 'graph_cache_provenance.json', 'data_audit.json',
        'mask_audit.json', 'mask_trace.npz', 'validation_predictions.csv', 'validation_metrics.json']
    result = {'schema': 'cappi-missing-topology-training-completed-v1', 'name': out.name,
        'settings': asdict(config), 'config': identity, 'fold': fold, 'model_seed': model_seed,
        'graph_seed': graph_seed, 'run_signature': signature, 'split_hash': data['split_hash'],
        'graph_hash': graph_hash, 'checkpoint_sha256': checkpoint_hash,
        'best_epoch': checkpoint['best_epoch'], 'epochs_completed': len(history),
        'best_validation_bce': validation_bce, 'validation_metrics': validation_metrics,
        'test_evaluated': False, 'test_labels_or_scores_loaded': False, 'evaluation_partition': 'validation',
        'elapsed_seconds': previous_elapsed + time.monotonic() - training_started,
        'this_invocation_elapsed_seconds': time.monotonic() - started,
        'peak_cuda_memory_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'peak_cuda_memory_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        'artifact_sha256': {name: sha(out / name) for name in artifact_names}}
    write_json(out / 'completed.json', result)
    print('COMPLETE', out.name, json.dumps(validation_metrics), flush=True)
    del model, optimizer, scheduler, checkpoint, ck, tokens, topo, present, train_pairs, train_y, val_pairs, val_y
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def evaluate_selected_run(out: Path, prepared: Path, device='cuda'):
    """Evaluate only the frozen checkpoint and original graph availability."""
    out = _safe_output(prepared, out)
    complete = verify_completed(out)
    identity = complete['config']
    if (code_hash() != identity['core_code_sha256'] or repair_code_hash() != identity['repair_code_sha256']
            or sha(Path(prepared) / 'manifest.json') != identity['prepared_manifest_sha256']):
        raise ValueError('Code/prepared provenance changed before evaluation')
    declaration_path = out / 'selection_frozen.json'
    declaration = json.loads(declaration_path.read_text(encoding='utf-8'))
    if (declaration.get('selected_before_test') is not True
            or declaration.get('run_signature') != complete['run_signature']
            or declaration.get('checkpoint_sha256') != complete['checkpoint_sha256']
            or not isinstance(declaration.get('selection_rule'), str) or not declaration['selection_rule'].strip()):
        raise ValueError('Missing or incompatible frozen selection declaration')
    declaration_hash = sha(declaration_path)
    previous = out / 'test_completed.json'
    if previous.exists():
        result = json.loads(previous.read_text(encoding='utf-8'))
        if (result['selection_declaration_sha256'] != declaration_hash
                or result['run_signature'] != complete['run_signature']):
            raise ValueError('Test evaluation already exists for another selection')
        for name, expected in result['artifact_sha256'].items():
            if sha(out / name) != expected:
                raise ValueError('Test artifact changed')
        return result
    data = load_validation_only(prepared, identity['protocol'], identity['negative_sampling'], identity['fold'])
    if data['split_hash'] != complete['split_hash']:
        raise ValueError('Evaluation fold changed')
    # This is the sole test-label decompression point in this module.
    folder = Path(prepared) / f"{identity['negative_sampling']}/{identity['protocol']}/fold_{identity['fold']}"
    with np.load(folder / 'test.npz', allow_pickle=False) as z:
        test_pairs, test_labels = z['pairs'].copy(), z['labels'].copy()
    if (not np.array_equal(test_pairs, data['test_pairs_ids']) or test_labels.dtype != np.float32
            or test_labels.shape != (len(test_pairs),) or set(test_labels.tolist()) != {0., 1.}):
        raise ValueError('Invalid selected test labels or pair IDs')
    expected_graph = fingerprint(data['graph_edges'], data['n_nodes'], GraphConfig(**identity['graph']), identity['graph_seed'])
    with np.load(out / 'topology.npz', allow_pickle=False) as z:
        topology = {k: z[k].copy() for k in z.files}
    if str(topology['graph_hash']) != expected_graph or expected_graph != complete['graph_hash']:
        raise ValueError('Test-time train graph identity mismatch')
    dev = torch.device(device)
    cfg = RepairConfig(**identity['settings'])
    model = MissingTopologyCaPPI(cfg.model, missing_policy=cfg.missing_policy).to(dev)
    checkpoint = _load(out / 'best.pt', dev)
    if checkpoint['run_signature'] != complete['run_signature'] or checkpoint['graph_hash'] != expected_graph:
        raise ValueError('Selected checkpoint identity mismatch')
    model.load_state_dict(checkpoint['model'])
    tokens = tokenize_sequences(data['sequences'], cfg.model.max_length).to(dev)
    topo = torch.as_tensor(topology['embedding'], dtype=torch.float32, device=dev)
    present = torch.as_tensor(topology['present'], dtype=torch.bool, device=dev)
    pairs = torch.as_tensor(test_pairs, dtype=torch.long, device=dev)
    started = time.monotonic()
    probabilities, _ = _infer(model, tokens, topo, present, pairs)
    _write_predictions(out / 'test_predictions.csv', data, test_pairs, test_labels,
                       probabilities, identity, complete['checkpoint_sha256'], 'test')
    test_metrics = metrics(test_labels, probabilities)
    write_json(out / 'test_metrics.json', test_metrics)
    result = {'schema': 'cappi-missing-topology-selected-test-v1', 'name': out.name,
        'run_signature': complete['run_signature'], 'split_hash': complete['split_hash'],
        'checkpoint_sha256': complete['checkpoint_sha256'], 'graph_hash': expected_graph,
        'selection_declaration_sha256': declaration_hash, 'selection_rule': declaration['selection_rule'],
        'evaluation_partition': 'test', 'test_evaluated': True, 'test_metrics': test_metrics,
        'elapsed_seconds': time.monotonic() - started, 'device': str(dev),
        'artifact_sha256': {name: sha(out / name) for name in
                          ('test_predictions.csv', 'test_metrics.json', 'selection_frozen.json')}}
    write_json(previous, result)
    print('TEST COMPLETE', out.name, json.dumps(test_metrics), flush=True)
    del model, checkpoint, tokens, topo, present, pairs
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--single-json', help='One JSON object with run arguments and nested config')
    source.add_argument('--config-file', type=Path, help='UTF-8 JSON object with run arguments and nested config')
    parser.add_argument('--evaluate-selected', action='store_true')
    args = parser.parse_args()
    settings = json.loads(args.single_json if args.single_json else args.config_file.read_text(encoding='utf-8-sig'))
    if args.evaluate_selected:
        result = evaluate_selected_run(**settings)
    else:
        settings['config'] = RepairConfig(**settings['config'])
        result = run_repair_fold(**settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
