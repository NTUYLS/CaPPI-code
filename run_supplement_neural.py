"""Validation-only parameter sensitivity, preserving reviewer_v1 and its core code.

The same validation partition chooses the checkpoint and describes sensitivity;
these are not independent test estimates or a nested hyperparameter search.
No test labels, scores or metrics are accessed. Test pairs are read only to audit
disjointness, and the immutable test artifact is hashed for split integrity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
import torch  # Import before sklearn on this Windows runtime.

from cappi_data import _fold_signature
from cappi_graph import GraphConfig, fingerprint, train_topology
from cappi_model import CaPPI, ModelConfig, get_parameter_report, tokenize_sequences
from run_experiments import _load, code_hash, metrics, save_checkpoint, seed_all, sha, write_json

HERE = Path(__file__).resolve().parent
CORE_SHA = '76593480cfb156c8817d6bd12b522f09b50f4219c28720af2f2e93d89f381e91'
PREPARED_SHA = 'ebef279a77d6c172559b96eb29388d3a41b98751867f3126e1a7db7c07a13d69'
METRICS = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc',
           'auroc', 'average_precision', 'pr_trapezoid_area')
SEED = 20260930
EPOCHS = 300
PATIENCE = 20
PAIR_CHUNK = 4096
WEIGHT_DECAY = .0001


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class SensitivityConfig:
    dim: int = 64
    walk_length: int = 20
    kernel: int = 7
    max_length: int = 1500
    learning_rate: float = .0005

    @property
    def name(self):
        lr = format(self.learning_rate, '.4f').replace('.', 'p')
        return f'dim{self.dim}_walk{self.walk_length}_kernel{self.kernel}_length{self.max_length}_lr{lr}'

    @property
    def model(self):
        return ModelConfig(topo_dim=self.dim, kernel=self.kernel,
                           max_length=self.max_length, fusion='residual', dropout=.3)

    @property
    def graph(self):
        return GraphConfig(method='drw', dim=self.dim, walk_length=self.walk_length, epochs=5)


BASE = SensitivityConfig()


def build_grid():
    groups = {
        'walk_dimension': [replace(BASE, dim=d, walk_length=w)
                           for d in (32, 64, 128, 256) for w in (10, 20, 30, 40)],
        'kernel_length': [replace(BASE, kernel=k, max_length=n)
                          for k in (3, 5, 7) for n in (1000, 1500, 2000)],
        'initial_learning_rate': [replace(BASE, learning_rate=lr)
                                  for lr in (.0001, .0005, .001, .005, .01)],
    }
    # Base first: validate source compatibility before any expensive training.
    configurations = {BASE.name: BASE}
    membership = {}
    for group, members in groups.items():
        membership[group] = [c.name for c in members]
        configurations.update((c.name, c) for c in members)
    assert len(configurations) == 28
    return list(configurations.values()), membership


def verify_contract(prepared, source):
    if code_hash() != CORE_SHA:
        raise ValueError('Original core code has changed; refuse reuse or mixed protocol')
    if sha(Path(prepared) / 'manifest.json') != PREPARED_SHA:
        raise ValueError('Prepared data manifest differs from reviewer_v1')
    source_campaign = json.loads((Path(source) / 'campaign_config.json').read_text(encoding='utf-8'))
    expected = {'epochs': EPOCHS, 'patience': PATIENCE, 'graph_epochs': 5,
                'lr': BASE.learning_rate, 'weight_decay': WEIGHT_DECAY,
                'max_length': BASE.max_length, 'pair_chunk': PAIR_CHUNK, 'seed': SEED,
                'device': 'cuda', 'code_sha256': CORE_SHA,
                'prepared_manifest_sha256': PREPARED_SHA,
                'torch': str(torch.__version__), 'numpy': np.__version__}
    for key, value in expected.items():
        if source_campaign.get(key) != value:
            raise ValueError(f'Original campaign mismatch: {key}')


def load_validation_fold(prepared, fold):
    """Immutable training/validation loader; never decompress test labels."""
    prepared = Path(prepared)
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    if fold not in range(5) or manifest['seed'] != SEED:
        raise ValueError('Invalid fold or preparation seed')
    relative = f'random/pair/fold_{fold}'
    folder = prepared / relative
    required = ['nodes.csv', 'positives.npz'] + [f'{relative}/{x}.npz'
                for x in ('train', 'validation', 'test', 'graph')]
    for name in required:
        if sha(prepared / name) != manifest['files'][name]:
            raise ValueError(f'Prepared artifact changed: {name}')
    split_hash = manifest['folds'][relative]['split_hash']
    if _fold_signature(prepared, relative, manifest['files']) != split_hash:
        raise ValueError('Fold identity mismatch')
    with (prepared / 'nodes.csv').open(encoding='utf-8', newline='') as handle:
        nodes = list(csv.DictReader(handle))
    if [int(x['index']) for x in nodes] != list(range(len(nodes))):
        raise ValueError('Node indices are not canonical')
    result = {'node_ids': np.asarray([int(x['node_id']) for x in nodes]),
              'sequences': [x['sequence'] for x in nodes], 'n_nodes': len(nodes),
              'split_hash': split_hash}
    if len(nodes) != manifest['n_nodes']:
        raise ValueError('Node count changed')
    for file_name, prefix in (('train', 'train'), ('validation', 'val')):
        with np.load(folder / f'{file_name}.npz', allow_pickle=False) as z:
            result[f'{prefix}_pairs'] = z['pairs'].copy()
            result[f'{prefix}_labels'] = z['labels'].copy()
    with np.load(folder / 'graph.npz', allow_pickle=False) as z:
        result['graph_edges'] = z['graph_edges'].copy()
        result['train_nodes'] = z['train_nodes'].copy()
    with np.load(folder / 'test.npz', allow_pickle=False) as z:
        test_pairs = z['pairs']  # membership audit only; do not read z['labels'].
    with np.load(prepared / 'positives.npz', allow_pickle=False) as z:
        known = set(map(tuple, z['pairs'].tolist()))
    sets = {}
    for prefix in ('train', 'val'):
        pairs, labels = result[f'{prefix}_pairs'], result[f'{prefix}_labels']
        if pairs.shape != (len(labels), 2) or set(labels.tolist()) != {0., 1.}:
            raise ValueError('Invalid training/validation shape or labels')
        if np.any(pairs[:, 0] >= pairs[:, 1]) or np.any(pairs < 0) or np.any(pairs >= len(nodes)):
            raise ValueError('Noncanonical or out-of-range pairs')
        sets[prefix] = set(map(tuple, pairs.tolist()))
        if len(sets[prefix]) != len(pairs):
            raise ValueError('Duplicate pair')
        if any((tuple(p) in known) != bool(y) for p, y in zip(pairs, labels)):
            raise ValueError('Labels contradict fixed benchmark positive pool')
    sets['test'] = set(map(tuple, test_pairs.tolist()))
    if sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']:
        raise ValueError('Split pair leakage')
    observed = set(map(tuple, result['graph_edges'].tolist()))
    expected = {tuple(p) for p, y in zip(result['train_pairs'], result['train_labels']) if y == 1}
    if observed != expected or observed & sets['val'] or observed & sets['test']:
        raise ValueError('Graph contains held-out pairs or differs from training positives')
    return result


def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'torch': str(torch.__version__), 'numpy': np.__version__,
            'sklearn': importlib.metadata.version('scikit-learn'),
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'torch_threads': torch.get_num_threads(), 'cudnn_deterministic': True,
            'cudnn_benchmark': False}


def campaign_contract(args):
    configurations, membership = build_grid()
    return {'schema': 'cappi-supplement-validation-v1', 'core_code_sha256': code_hash(),
            'supplement_code_sha256': sha(__file__), 'prepared_manifest_sha256': PREPARED_SHA,
            'source_campaign_sha256': sha(Path(args.source) / 'campaign_config.json'),
            'prepared': str(Path(args.prepared).resolve()), 'source': str(Path(args.source).resolve()),
            'environment': environment(), 'configurations': [asdict(c) for c in configurations],
            'groups': membership, 'unique_configs': 28, 'folds_per_config': 5,
            'new_training_runs': 135, 'reused_baseline_runs': 5,
            'seed_rule': '20260930 + fold', 'epochs': EPOCHS, 'patience': PATIENCE,
            'weight_decay': WEIGHT_DECAY, 'optimizer': 'AdamW', 'pair_chunk': PAIR_CHUNK,
            'scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': .5, 'patience': 10},
            'checkpoint_selection': 'minimum validation BCE with improvement > 1e-6',
            'threshold': .5, 'protocol': 'pair', 'negative_sampling': 'random', 'graph': 'drw',
            'test_evaluation': False, 'test_labels_or_scores_loaded': False,
            'interpretation': 'Validation sensitivity only. The same validation set selects the checkpoint and describes performance. No independent test estimate, significance test, or optimality claim; no replacement of reviewer_v1 main results.'}


def run_identity(config, fold, data, campaign):
    return {'campaign_signature': digest(campaign), 'settings': asdict(config),
            'model': asdict(config.model), 'graph': asdict(config.graph),
            'seed': SEED + fold, 'fold': fold, 'split_hash': data['split_hash'],
            'epochs': EPOCHS, 'patience': PATIENCE, 'pair_chunk': PAIR_CHUNK,
            'weight_decay': WEIGHT_DECAY, 'device': 'cuda', 'evaluation_partition': 'validation',
            'baseline_reuse': config == BASE}


def verify_completed(out, signature):
    result = json.loads((out / 'completed.json').read_text(encoding='utf-8'))
    if result['run_signature'] != signature:
        raise ValueError(f'Incompatible completed run: {out}')
    for name, value in result['artifact_sha256'].items():
        if sha(out / name) != value:
            raise ValueError(f'Completed artifact content changed: {out / name}')
    if result['test_evaluated'] or result['evaluation_partition'] != 'validation':
        raise ValueError('Nonvalidation result in sensitivity campaign')
    return result


def load_source_baseline(args, fold, data):
    source = Path(args.source) / f'pair__random__drw__residual__fold{fold}'
    run_config = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    # Only provenance fields of completed.json are used, never test_metrics.
    completion = json.loads((source / 'completed.json').read_text(encoding='utf-8'))
    checkpoint = _load(source / 'best.pt')
    identity = checkpoint['config']
    expected = {'model': normalized(asdict(BASE.model)), 'graph': asdict(BASE.graph),
                'seed': SEED + fold, 'split_hash': data['split_hash'], 'code_sha256': CORE_SHA,
                'max_epochs': EPOCHS, 'learning_rate': BASE.learning_rate, 'patience': PATIENCE,
                'weight_decay': WEIGHT_DECAY, 'pair_chunk': PAIR_CHUNK,
                'protocol': 'pair', 'negative_sampling': 'random', 'variant': 'residual', 'fold': fold}
    for key, value in expected.items():
        if normalized(identity.get(key)) != normalized(value) or normalized(run_config.get(key)) != normalized(value):
            raise ValueError(f'Baseline config mismatch: {key}')
    if digest(identity) != checkpoint['run_signature'] or run_config['run_signature'] != checkpoint['run_signature']:
        raise ValueError('Baseline checkpoint/run_config signature mismatch')
    if completion['run_signature'] != checkpoint['run_signature'] or sha(source / 'best.pt') != completion['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint hash mismatch')
    expected_graph = fingerprint(data['graph_edges'], data['n_nodes'], BASE.graph, SEED + fold)
    if checkpoint['graph_hash'] != expected_graph or completion['graph_hash'] != expected_graph:
        raise ValueError('Baseline graph identity mismatch')
    source_cache = Path(args.source) / 'graph_cache'
    if not (source_cache / f'{expected_graph}.npz').is_file() or not (source_cache / f'{expected_graph}.json').is_file():
        raise ValueError('Original graph cache is missing; refuse regeneration in read-only source')
    raw, raw_present, _ = train_topology(data['graph_edges'], data['n_nodes'], BASE.graph,
                                      SEED + fold, 'cuda', source_cache)
    with np.load(source / 'topology.npz', allow_pickle=False) as z:
        topology = {key: z[key].copy() for key in z.files}
    fitting = data['train_nodes'][raw_present[data['train_nodes']]]
    mean, std = raw[fitting].mean(0), raw[fitting].std(0)
    scaled = (raw - mean) / np.maximum(std, 1e-6)
    scaled[~raw_present] = 0
    if (str(topology['graph_hash']) != expected_graph or not np.array_equal(topology['embedding'], scaled)
            or not np.array_equal(topology['present'], raw_present)
            or not np.array_equal(topology['mean'], mean) or not np.array_equal(topology['std'], std)):
        raise ValueError('Baseline normalized topology differs from verified train-only cache')
    provenance = {name: {'path': str(source / name), 'sha256': sha(source / name)}
                  for name in ('best.pt', 'topology.npz', 'run_config.json', 'completed.json', 'history.json')}
    return checkpoint, topology, provenance


def tensors(data, config, topology):
    return (tokenize_sequences(data['sequences'], config.max_length).to('cuda'),
            torch.as_tensor(topology['embedding'], dtype=torch.float32, device='cuda'),
            torch.as_tensor(topology['present'], dtype=torch.bool, device='cuda'),
            torch.as_tensor(data['val_pairs'], dtype=torch.long, device='cuda'),
            torch.as_tensor(data['val_labels'], dtype=torch.float32, device='cuda'))


def infer_validation(model, tokens, topology, present, pairs, labels):
    model.eval()
    with torch.no_grad():
        node_vectors = model.encode_nodes(tokens, topology, present)
        logits = torch.cat([model.score_pairs(node_vectors, pairs[i:i + PAIR_CHUNK])
                            for i in range(0, len(pairs), PAIR_CHUNK)])
        loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(loss) or not np.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite validation result')
    return probabilities, loss


def baseline_preflight(args):
    data = load_validation_fold(args.prepared, 0)
    checkpoint, topology, provenance = load_source_baseline(args, 0, data)
    model = CaPPI(BASE.model).to('cuda')
    model.load_state_dict(checkpoint['model']); model.eval()
    tokens, topo, present, pairs, labels = tensors(data, BASE, topology)
    probabilities, loss = infer_validation(model, tokens, topo, present, pairs, labels)
    with torch.no_grad():
        # Two public inference interfaces must agree on an actual saved checkpoint.
        direct = torch.sigmoid(model(tokens, topo, present, pairs[:137])).cpu().numpy()
    error = float(np.max(np.abs(probabilities[:137] - direct)))
    bce_error = abs(loss - checkpoint['best'])
    if error > 1e-6 or bce_error > 1e-6:
        raise ValueError(f'Base checkpoint forward/BCE mismatch: {error}, {bce_error}')
    result = {'status': 'passed', 'fold': 0, 'validation_n': len(probabilities),
              'max_forward_absolute_difference': error, 'validation_bce_difference': bce_error,
              'validation_bce': loss, 'saved_best_bce': checkpoint['best'],
              'source_artifacts': provenance, 'test_evaluated': False,
              'test_labels_or_scores_loaded': False, 'environment': environment()}
    del model, checkpoint, tokens, topo, present, pairs, labels
    torch.cuda.empty_cache()
    return result


def export_validation(out, model, data, config, fold, tensor_values, signature, graph_hash,
                      checkpoint_hash, best_bce, best_epoch, history, elapsed, source=None):
    probabilities, bce = infer_validation(model, *tensor_values)
    if abs(bce - best_bce) > 1e-6:
        raise ValueError('Reloaded best checkpoint does not reproduce saved validation BCE')
    with (out / 'validation_predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['protein_u', 'protein_v', 'label', 'probability', 'prediction', 'fold',
                         'seed', 'split_hash', 'checkpoint_sha256', 'partition'])
        for (u, v), y, p in zip(data['val_pairs'], data['val_labels'], probabilities):
            writer.writerow([data['node_ids'][u], data['node_ids'][v], int(y), float(p), int(p >= .5),
                             fold, SEED + fold, data['split_hash'], checkpoint_hash, 'validation'])
    validation_metrics = metrics(data['val_labels'], probabilities)
    write_json(out / 'validation_metrics.json', validation_metrics)
    artifacts = ['best.pt', 'topology.npz', 'run_config.json', 'history.json',
                 'validation_predictions.csv', 'validation_metrics.json', 'parameters.json']
    if source:
        artifacts.append('source_provenance.json')
    else:
        artifacts += ['last.pt', 'first_step_gradient_norms.json']
    result = {'name': out.name, 'config_name': config.name, 'settings': asdict(config), 'fold': fold,
              'run_signature': signature, 'split_hash': data['split_hash'], 'graph_hash': graph_hash,
              'checkpoint_sha256': checkpoint_hash, 'evaluation_partition': 'validation',
              'test_evaluated': False, 'test_labels_or_scores_loaded': False,
              'baseline_reused': bool(source), 'best_epoch': best_epoch,
              'epochs_in_history': len(history), 'best_validation_bce': bce,
              'elapsed_seconds': elapsed, 'validation_metrics': validation_metrics,
              'artifact_sha256': {name: sha(out / name) for name in artifacts}}
    write_json(out / 'completed.json', result)
    print('COMPLETE', out.name, json.dumps(validation_metrics), flush=True)
    return result


def run_fold(args, config, fold, campaign):
    started = time.monotonic()
    data = load_validation_fold(args.prepared, fold)
    identity = run_identity(config, fold, data, campaign)
    signature = digest(identity)
    out = Path(args.output) / f'{config.name}__fold{fold}'
    if (out / 'completed.json').exists():
        result = verify_completed(out, signature)
        print('SKIP VERIFIED COMPLETE', out.name, flush=True)
        return result
    if out.exists() and not args.resume:
        raise FileExistsError(f'Use --resume or a fresh output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / 'run_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding='utf-8'))['run_signature'] != signature:
        raise ValueError('Cannot resume incompatible run')
    write_json(cfg_path, {**identity, 'run_signature': signature})
    seed_all(SEED + fold)
    if config == BASE:
        checkpoint, topology, provenance = load_source_baseline(args, fold, data)
        for name in ('best.pt', 'topology.npz', 'history.json'):
            shutil.copyfile(provenance[name]['path'], out / name)
        write_json(out / 'source_provenance.json', provenance)
        graph_hash = checkpoint['graph_hash']
        model = CaPPI(config.model).to('cuda')
        model.load_state_dict(checkpoint['model'])
        write_json(out / 'parameters.json', get_parameter_report(model))
        source_history = json.loads((out / 'history.json').read_text(encoding='utf-8'))
        result = export_validation(out, model, data, config, fold, tensors(data, config, topology),
                                   signature, graph_hash, sha(out / 'best.pt'), checkpoint['best'],
                                   checkpoint['best_epoch'], source_history, time.monotonic()-started,
                                   source=provenance)
        del model, checkpoint
        torch.cuda.empty_cache()
        return result
    # Existing default graph cache is strictly read-only; novel settings use a new cache.
    graph_hash = fingerprint(data['graph_edges'], data['n_nodes'], config.graph, SEED + fold)
    old_cache = Path(args.source) / 'graph_cache'
    if (old_cache / f'{graph_hash}.npz').exists():
        cache = old_cache
    else:
        cache = Path(args.output) / 'graph_cache'
    embedding, present, returned_hash = train_topology(data['graph_edges'], data['n_nodes'],
                                                      config.graph, SEED + fold, 'cuda', cache)
    if returned_hash != graph_hash:
        raise ValueError('Topology fingerprint mismatch')
    fitting = data['train_nodes'][present[data['train_nodes']]]
    mean = embedding[fitting].mean(0) if len(fitting) else np.zeros(config.dim)
    std = embedding[fitting].std(0) if len(fitting) else np.ones(config.dim)
    embedding = (embedding - mean) / np.maximum(std, 1e-6)
    embedding[~present] = 0
    topology = {'embedding': embedding, 'present': present, 'mean': mean, 'std': std,
                'graph_hash': np.asarray(graph_hash)}
    topo_path = out / 'topology.npz'
    if topo_path.exists():
        with np.load(topo_path, allow_pickle=False) as z:
            if any(not np.array_equal(z[key], value) for key, value in topology.items()):
                raise ValueError('Resumed normalized topology mismatch')
    else:
        np.savez_compressed(topo_path, **topology)
    tensor_values = tensors(data, config, topology)
    tokens, topo, present_tensor, val_pairs, val_y = tensor_values
    train_pairs = torch.as_tensor(data['train_pairs'], dtype=torch.long, device='cuda')
    train_y = torch.as_tensor(data['train_labels'], dtype=torch.float32, device='cuda')
    seed_all(SEED + fold)
    model = CaPPI(config.model).to('cuda')
    write_json(out / 'parameters.json', get_parameter_report(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=10)
    history, best, stale, start_epoch, best_epoch = [], float('inf'), 0, 0, None
    last = out / 'last.pt'
    if args.resume and last.exists():
        checkpoint = _load(last, 'cuda')
        if checkpoint['run_signature'] != signature or checkpoint['graph_hash'] != graph_hash:
            raise ValueError('Resumed checkpoint provenance mismatch')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        history, best, stale = checkpoint['history'], checkpoint['best'], checkpoint['stale']
        start_epoch, best_epoch = checkpoint['epoch'] + 1, checkpoint['best_epoch']
        random.setstate(checkpoint['rng_python']); np.random.set_state(checkpoint['rng_numpy'])
        torch.set_rng_state(checkpoint['rng_torch'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['rng_cuda']])
    print(f'START {out.name} train={len(train_y)} val={len(val_y)}', flush=True)
    for epoch in range(start_epoch, EPOCHS if stale < PATIENCE else start_epoch):
        model.train(); optimizer.zero_grad(set_to_none=True)
        node_vectors = model.encode_nodes(tokens, topo, present_tensor)
        order = torch.randperm(len(train_pairs), device='cuda')
        loss_terms, loss_sum = [], 0.
        for start in range(0, len(order), PAIR_CHUNK):
            indices = order[start:start + PAIR_CHUNK]
            logits = model.score_pairs(node_vectors, train_pairs[indices])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y[indices],
                                                                        reduction='sum') / len(train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss_terms.append(loss); loss_sum += float(loss.detach())
        torch.stack(loss_terms).sum().backward()
        del loss_terms
        if epoch == 0:
            grads = {n: float(p.grad.norm().detach()) if p.grad is not None else None
                     for n, p in model.named_parameters()}
            write_json(out / 'first_step_gradient_norms.json', grads)
            if any(g is None or not np.isfinite(g) for g in grads.values()):
                raise FloatingPointError('Missing or nonfinite trainable parameter gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step(); del node_vectors
        probabilities, val_loss = infer_validation(model, *tensor_values)
        # Preserve the old diagnostic AP log; checkpoint selection uses BCE only.
        from sklearn.metrics import average_precision_score
        val_ap = average_precision_score(data['val_labels'], probabilities)
        scheduler.step(val_loss)
        if val_loss < best - 1e-6:
            best, best_epoch, stale, improved = val_loss, epoch + 1, 0, True
        else:
            stale += 1; improved = False
        history.append({'epoch': epoch + 1, 'train_bce': loss_sum,
                        'validation_bce': val_loss, 'validation_ap': val_ap,
                        'lr': optimizer.param_groups[0]['lr']})
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best,
                      'best_epoch': best_epoch, 'stale': stale, 'history': history, 'config': identity,
                      'run_signature': signature, 'graph_hash': graph_hash,
                      'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                      'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
        if improved:
            save_checkpoint(checkpoint, out / 'best.pt')
        if (epoch + 1) % 5 == 0 or stale >= PATIENCE or epoch + 1 == EPOCHS:
            save_checkpoint(checkpoint, last)
        write_json(out / 'history.json', history)
        if epoch == start_epoch or (epoch + 1) % 50 == 0 or stale >= PATIENCE:
            print(f'{out.name} epoch={epoch+1} valBCE={val_loss:.6f} stale={stale} elapsed={time.monotonic()-started:.1f}s', flush=True)
        if stale >= PATIENCE:
            break
    best_checkpoint = _load(out / 'best.pt', 'cuda')
    if best_checkpoint['run_signature'] != signature or best_checkpoint['graph_hash'] != graph_hash:
        raise ValueError('Best checkpoint provenance mismatch')
    model.load_state_dict(best_checkpoint['model'])
    result = export_validation(out, model, data, config, fold, tensor_values, signature, graph_hash,
                               sha(out / 'best.pt'), best, best_epoch, history, time.monotonic()-started)
    del model, optimizer, scheduler, checkpoint, best_checkpoint, tensor_values
    del tokens, topo, present_tensor, val_pairs, val_y, train_pairs, train_y
    torch.cuda.empty_cache()
    return result


def summarize(output, campaign):
    output = Path(output)
    grouped = {}
    for path in output.glob('*__fold*/completed.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['test_evaluated'] or result['evaluation_partition'] != 'validation':
            raise ValueError('Invalid sensitivity partition')
        grouped.setdefault(result['config_name'], []).append(result)
    rows = []
    configs, groups = build_grid()
    for config in configs:
        members = grouped.get(config.name, [])
        if not members:
            continue
        if len({r['fold'] for r in members}) != len(members):
            raise ValueError('Duplicate fold in sensitivity summary')
        row = {'config_name': config.name, **asdict(config), 'completed_folds': len(members),
               'groups': ';'.join(g for g, names in groups.items() if config.name in names),
               'evaluation_partition': 'validation', 'baseline_reused': config == BASE}
        for metric in METRICS:
            values = np.asarray([r['validation_metrics'][metric] for r in members], float)
            row[metric + '_mean'] = float(values.mean())
            row[metric + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    write_json(output / 'validation_summary.json', rows)
    if rows:
        with (output / 'validation_summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    done = sum(len(x) for x in grouped.values())
    reused = sum(r['baseline_reused'] for rs in grouped.values() for r in rs)
    write_json(output / 'status.json', {'status': 'complete' if done == 140 else 'in_progress',
                                      'completed_folds': done, 'expected_folds': 140,
                                      'new_training_completed': done - reused,
                                      'baseline_reuse_completed': reused,
                                      'campaign_signature': digest(campaign), 'test_evaluated': False})


def verify_all(args, campaign):
    """Recompute every reported validation metric from saved pair predictions."""
    configs, _ = build_grid()
    data_by_fold = {fold: load_validation_fold(args.prepared, fold) for fold in range(5)}
    records = []
    for config in configs:
        for fold in range(5):
            data = data_by_fold[fold]
            out = Path(args.output) / f'{config.name}__fold{fold}'
            signature = digest(run_identity(config, fold, data, campaign))
            result = verify_completed(out, signature)
            with (out / 'validation_predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != len(data['val_labels']):
                raise ValueError('Prediction row count differs from validation split')
            actual_pairs = np.asarray([[int(row['protein_u']), int(row['protein_v'])] for row in rows])
            expected_pairs = data['node_ids'][data['val_pairs']]
            labels = np.asarray([int(row['label']) for row in rows])
            probabilities = np.asarray([float(row['probability']) for row in rows])
            if not np.array_equal(actual_pairs, expected_pairs) or not np.array_equal(labels, data['val_labels']):
                raise ValueError('Prediction IDs or labels differ from immutable validation split')
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
                raise ValueError('Prediction probabilities invalid')
            for row in rows:
                if (int(row['fold']) != fold or int(row['seed']) != SEED + fold
                        or row['split_hash'] != data['split_hash'] or row['partition'] != 'validation'
                        or row['checkpoint_sha256'] != result['checkpoint_sha256']
                        or int(row['prediction']) != int(float(row['probability']) >= .5)):
                    raise ValueError('Prediction provenance or threshold mismatch')
            recomputed = metrics(labels, probabilities)
            saved_metrics = json.loads((out / 'validation_metrics.json').read_text(encoding='utf-8'))
            for key in METRICS + ('n', 'positive_n', 'threshold'):
                if not np.isclose(recomputed[key], result['validation_metrics'][key], rtol=0, atol=1e-12):
                    raise ValueError(f'Recomputed metric differs: {out.name}/{key}')
                if not np.isclose(recomputed[key], saved_metrics[key], rtol=0, atol=1e-12):
                    raise ValueError(f'Saved metric file differs: {out.name}/{key}')
            records.append({'name': out.name, 'status': 'passed', 'validation_n': len(rows),
                            'baseline_reused': config == BASE,
                            'checkpoint_sha256': result['checkpoint_sha256']})
    report = {'status': 'passed', 'verified_runs': len(records), 'expected_runs': 140,
              'new_training_runs': 135, 'reused_baseline_runs': 5, 'metrics_per_run': len(METRICS),
              'campaign_signature': digest(campaign), 'test_evaluated': False,
              'test_labels_or_scores_loaded': False, 'checks': ['artifact SHA256',
              'run and campaign signature', 'exact validation pair IDs and labels',
              'seed/fold/split/checkpoint provenance', 'fixed threshold predictions',
              'all nine metrics recomputed from CSV'], 'records': records}
    write_json(Path(args.output) / 'verification.json', report)
    print('VERIFIED 140 VALIDATION RUNS; 135 NEW TRAININGS + 5 REUSED BASELINES', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default=str(HERE / 'prepared'))
    parser.add_argument('--source', default=str(HERE / 'runs/reviewer_v1'))
    parser.add_argument('--output', default=str(HERE / 'runs/supplement_20261001/parameter_sensitivity'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='Only verify actual base checkpoint inference')
    parser.add_argument('--verify-only', action='store_true', help='Recompute all 140 saved validation results')
    parser.add_argument('--describe', action='store_true', help='Print the fixed grid without training')
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--configs', default='', help='Optional comma-separated exact configuration names')
    args = parser.parse_args()
    torch.set_num_threads(2)
    configurations, groups = build_grid()
    if args.describe:
        print(json.dumps({'configs': [asdict(c) for c in configurations], 'groups': groups,
                          'unique_configs': 28, 'folds': 140, 'new_training_runs': 135,
                          'baseline_reused': 5, 'evaluation': 'validation only'}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('This campaign contract requires the original CUDA runtime')
    verify_contract(args.prepared, args.source)
    output = Path(args.output).resolve()
    if output == Path(args.source).resolve() or Path(args.source).resolve() in output.parents:
        raise ValueError('Output cannot overwrite original campaign')
    output.mkdir(parents=True, exist_ok=True)
    campaign = campaign_contract(args)
    if args.preflight:
        result = baseline_preflight(args)
        result['supplement_code_sha256'] = sha(__file__)
        write_json(output / 'base_forward_preflight.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    lock = output / 'campaign_config.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) != normalized(campaign):
        raise ValueError('Campaign contract changed; refusing mixed output')
    if args.verify_only:
        if not lock.exists():
            raise ValueError('No campaign to verify')
        verify_all(args, campaign)
        return
    write_json(lock, campaign)
    folds = list(map(int, args.folds.split(',')))
    if len(set(folds)) != len(folds) or not folds or any(f not in range(5) for f in folds):
        raise ValueError('Folds must be distinct indices 0 through 4')
    if args.configs:
        requested = args.configs.split(',')
        if set(requested) - {c.name for c in configurations}:
            raise ValueError('Unknown config')
        configurations = [c for c in configurations if c.name in requested]
    for config in configurations:
        for fold in folds:
            run_fold(args, config, fold, campaign)
            summarize(output, campaign)
    if len(configurations) == 28 and len(folds) == 5:
        verify_all(args, campaign)
    print('REQUESTED RUNS FINISHED', output, flush=True)


if __name__ == '__main__':
    main()
