"""Auditable feature-based controls, not reproductions of published deep models.

Use the same prepared random-negative pair/protein folds as run_experiments.py.
All hyperparameters are fixed. Validation/test labels never select a model or
threshold. Graph heuristics report ranking metrics only, not classification.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, auc,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cappi_data import load_fold
from cappi_graph import GraphConfig, fingerprint, train_topology

HERE = Path(__file__).resolve().parent
AA20 = 'ACDEFGHIKLMNPQRSTVWY'
METHODS = ('degree_product', 'common_neighbor', 'aa20_drw_lr', 'aa20_drw_rf')
LEARNED = {'aa20_drw_lr', 'aa20_drw_rf'}
METRIC_NAMES = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1',
                'mcc', 'auroc', 'average_precision', 'pr_trapezoid_area')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(json.dumps(list(array.shape)).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def code_hash():
    digest = hashlib.sha256()
    for name in ('run_classical_baselines.py', 'cappi_data.py',
                 'cappi_graph.py', 'legacy_skipgram.py'):
        digest.update(name.encode('ascii'))
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def amino_acid_composition(sequences):
    """Canonical-AA counts divided by full sequence length; unknowns have no bin."""
    features = np.zeros((len(sequences), 20), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        sequence = ''.join(sequence.split()).upper()
        if sequence:
            features[row] = [sequence.count(aa) / len(sequence) for aa in AA20]
    return features


def symmetric_pair_features(node_features, pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    nodes = np.asarray(node_features, dtype=np.float32)
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(nodes)):
        raise ValueError('Pair endpoint outside node feature matrix')
    left, right = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
    return np.concatenate([np.abs(left - right), left * right], axis=1)


def graph_scores(edges, n_nodes, pairs, method):
    """Compute scores from training adjacency only, including zero for unseen nodes."""
    if method not in ('degree_product', 'common_neighbor'):
        raise ValueError('Unsupported graph heuristic')
    adjacency = [set() for _ in range(n_nodes)]
    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        if not (0 <= u < n_nodes and 0 <= v < n_nodes) or u == v:
            raise ValueError('Invalid training edge')
        adjacency[u].add(int(v))
        adjacency[v].add(int(u))
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= n_nodes):
        raise ValueError('Invalid scoring endpoint')
    if method == 'degree_product':
        degree = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.float64)
        return degree[pairs[:, 0]] * degree[pairs[:, 1]]
    return np.asarray([len(adjacency[u] & adjacency[v]) for u, v in pairs], dtype=np.float64)


def fixed_estimator(method, seed):
    if method == 'aa20_drw_lr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                max_iter=2000, tol=1e-4, class_weight=None, random_state=seed)),
        ])
    if method == 'aa20_drw_rf':
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
            max_features='sqrt', n_jobs=4, random_state=seed, criterion='gini',
            max_depth=None, min_samples_split=2, bootstrap=True, class_weight=None)
    raise ValueError('Method has no learned classifier')


def fixed_model_config(method):
    if method == 'aa20_drw_lr':
        return {'algorithm': 'LogisticRegression', 'C': 1.0, 'penalty': 'l2',
                'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-4,
                'class_weight': None, 'standardization': 'StandardScaler fitted only on training pairs'}
    if method == 'aa20_drw_rf':
        return {'algorithm': 'RandomForestClassifier', 'n_estimators': 200,
                'min_samples_leaf': 2, 'max_features': 'sqrt', 'n_jobs': 4,
                'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2,
                'bootstrap': True, 'class_weight': None, 'standardization': 'none'}
    return {'algorithm': method, 'calibration': 'none', 'decision_threshold': None}


def compute_metrics(labels, scores, is_probability):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or not len(labels) or not np.isfinite(scores).all():
        raise ValueError('Invalid labels or nonfinite scores')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must be binary')
    result = {name: None for name in METRIC_NAMES}
    result.update(n=len(labels), positive_n=int(labels.sum()), threshold=None,
                  metric_scope='ranking_only' if not is_probability else 'classification_and_ranking')
    if len(np.unique(labels)) == 2:
        result['auroc'] = float(roc_auc_score(labels, scores))
        result['average_precision'] = float(average_precision_score(labels, scores))
    if not is_probability:
        return result
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError('Probabilities must be in [0, 1]')
    prediction = (scores >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, scores)
    result.update(accuracy=float(accuracy_score(labels, prediction)),
        sensitivity=float(recall_score(labels, prediction, zero_division=0)),
        specificity=float(tn / (tn + fp)) if tn + fp else None,
        precision=float(precision_score(labels, prediction, zero_division=0)),
        f1=float(f1_score(labels, prediction, zero_division=0)),
        mcc=float(matthews_corrcoef(labels, prediction)),
        pr_trapezoid_area=float(auc(recall, precision)), threshold=.5)
    return result


def validate_graph_contract(data):
    edge_set = {tuple(map(int, pair)) for pair in data['graph_edges']}
    train_positive = {tuple(map(int, pair)) for pair, y in
                      zip(data['train_pairs'], data['train_labels']) if y == 1}
    if edge_set != train_positive:
        raise ValueError('Topology must be exactly the training positives')
    for split in ('val', 'test'):
        if edge_set & {tuple(map(int, pair)) for pair in data[split + '_pairs']}:
            raise ValueError('Held-out pair entered topology')


def _verify_completed(folder, signature):
    completed = folder / 'completed.json'
    if not completed.exists():
        if folder.exists():
            raise FileExistsError('Incomplete existing output; use a fresh directory: {}'.format(folder))
        return None
    result = json.loads(completed.read_text(encoding='utf-8'))
    if result['run_signature'] != signature:
        raise ValueError('Incompatible completed control: {}'.format(folder))
    for filename, expected in result['artifacts_sha256'].items():
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder.resolve()) or file_hash(path) != expected:
            raise ValueError('Control artifact changed: {}'.format(filename))
    return result


def run_fold(prepared, output, protocol, fold, seed=20260930,
             graph_epochs=5, graph_cache=None, methods=METHODS):
    if protocol not in ('pair', 'protein') or not methods or not set(methods).issubset(METHODS):
        raise ValueError('Unsupported protocol or method')
    data = load_fold(Path(prepared), 'random', protocol, fold)
    validate_graph_contract(data)
    if len(np.unique(data['train_labels'])) != 2:
        raise ValueError('Training split must contain both classes')
    output = Path(output)
    run_seed = seed + fold
    graph_config = GraphConfig(method='drw', dim=64, epochs=graph_epochs)
    graph_identity = fingerprint(data['graph_edges'], data['n_nodes'], graph_config, run_seed)
    split_hashes = {split: array_hash(data[split + '_pairs'], data[split + '_labels'])
                    for split in ('train', 'val', 'test')}
    prepared_relative = 'random/{}/fold_{}'.format(protocol, fold)
    prepared_hashes = {split: data['manifest']['files'][prepared_relative + '/' + filename + '.npz']
                       for split, filename in [('train', 'train'), ('val', 'validation'), ('test', 'test'), ('graph', 'graph')]}
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'sklearn': sklearn.__version__, 'joblib': joblib.__version__}
    common = {'code_sha256': code_hash(), 'dataset_hash': data['manifest']['dataset_hash'],
        'protocol': protocol, 'negative_sampling': 'random', 'base_seed': seed,
        'graph': asdict(graph_config), 'environment': environment,
        'control_category': 'feature-based controls; not published deep-model reproductions',
        'selection': 'Fixed hyperparameters; training pairs only; no validation/test model or threshold tuning.',
        'feature_definition': 'Full-sequence canonical AA20 frequencies plus raw 64D DRW; '
                              'pair = concat(abs(u-v), u*v), 168 dimensions.',
        'aa20_unknown_policy': 'Unknown residues count toward full sequence length but have no AA20 bin.',
        'unseen_topology': 'All-zero for nodes absent from the training graph.',
        'heuristic_metrics': 'Raw degree-product/common-neighbor ranking scores; only AUROC/AP; no classifier threshold.'}
    todo, results = [], []
    for method in methods:
        campaign = dict(common, method=method, model=fixed_model_config(method))
        if method not in LEARNED:
            campaign['feature_definition'] = ('Product of two endpoint degrees in training adjacency.'
                if method == 'degree_product' else 'Number of shared neighbors in training adjacency.')
        identity = dict(campaign, seed=run_seed, fold=fold, split_hash=data['split_hash'],
                        split_array_hashes=split_hashes, prepared_artifact_hashes=prepared_hashes,
                        graph_hash=graph_identity if method in LEARNED else array_hash(data['graph_edges']))
        signature = json_hash(identity)
        name = '{}__random__{}__fold{}'.format(protocol, method, fold)
        folder = output / name
        previous = _verify_completed(folder, signature)
        if previous is not None:
            results.append(previous)
        else:
            todo.append((method, name, folder, identity, signature, json_hash(campaign)))
    if not todo:
        return results
    features = None
    if any(item[0] in LEARNED for item in todo):
        cache = Path(graph_cache) if graph_cache else output / 'graph_cache'
        embedding, present, actual_hash = train_topology(data['graph_edges'], data['n_nodes'],
            graph_config, run_seed, device='cpu', cache_dir=cache)
        if actual_hash != graph_identity or embedding.shape != (data['n_nodes'], 64):
            raise ValueError('Unexpected graph embedding identity or shape')
        expected_present = np.zeros(data['n_nodes'], dtype=bool)
        expected_present[data['graph_edges'].reshape(-1)] = True
        if not np.array_equal(present, expected_present) or np.any(embedding[~present] != 0):
            raise ValueError('Training-graph presence or zero-topology contract violated')
        if not np.isfinite(embedding).all():
            raise ValueError('Nonfinite topology')
        if protocol == 'protein' and (present[data['val_nodes']].any() or present[data['test_nodes']].any()):
            raise ValueError('Unseen protein received training topology')
        nodes = np.concatenate([amino_acid_composition(data['sequences']), embedding], axis=1)
        features = {split: symmetric_pair_features(nodes, data[split + '_pairs'])
                    for split in ('train', 'val', 'test')}
    for method, name, folder, identity, signature, campaign_hash in todo:
        started = time.monotonic()
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / 'run_config.json', dict(identity, run_signature=signature,
            campaign_signature=campaign_hash, python_executable=sys.executable))
        model_path = folder / ('model.joblib' if method in LEARNED else 'heuristic.json')
        if method in LEARNED:
            model = fixed_estimator(method, run_seed)
            # A fixed iteration budget that fails to converge must be visible.
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(features['train'], data['train_labels'].astype(int))
            temp = model_path.with_suffix('.tmp.joblib')
            joblib.dump(model, temp)
            temp.replace(model_path)
            scores = {split: model.predict_proba(features[split])[:, list(model.classes_).index(1)]
                      for split in ('train', 'val', 'test')}
        else:
            write_json(model_path, dict(method=method, graph_edges=data['graph_edges'].tolist(),
                n_nodes=int(data['n_nodes']), source='training positive adjacency only',
                score_interpretation='uncalibrated ranking score; no classification threshold'))
            scores = {split: graph_scores(data['graph_edges'], data['n_nodes'],
                        data[split + '_pairs'], method) for split in ('train', 'val', 'test')}
        model_hash = file_hash(model_path)
        all_metrics = {split: compute_metrics(data[split + '_labels'], scores[split], method in LEARNED)
                       for split in ('train', 'val', 'test')}
        prediction_path = folder / 'test_predictions.csv'
        columns = ['protein_u', 'protein_v', 'label', 'raw_score', 'probability', 'prediction',
                   'method', 'protocol', 'fold', 'seed', 'split_hash', 'train_array_hash',
                   'val_array_hash', 'test_array_hash', 'checkpoint_sha256']
        with prediction_path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for (u, v), label, score in zip(data['test_pairs'], data['test_labels'], scores['test']):
                writer.writerow([int(data['node_ids'][u]), int(data['node_ids'][v]), int(label),
                    float(score) if method not in LEARNED else '',
                    float(score) if method in LEARNED else '',
                    int(score >= .5) if method in LEARNED else '',
                    method, protocol, fold, run_seed, data['split_hash'],
                    split_hashes['train'], split_hashes['val'], split_hashes['test'], model_hash])
        write_json(folder / 'metrics.json', all_metrics)
        result = {'name': name, 'method': method, 'protocol': protocol, 'fold': fold,
            'run_signature': signature, 'campaign_signature': campaign_hash, 'config': identity,
            'split_hash': data['split_hash'], 'split_array_hashes': split_hashes,
            'checkpoint_sha256': model_hash, 'test_evaluated': True,
            'test_metrics': all_metrics['test'], 'elapsed_seconds': time.monotonic() - started,
            'artifacts_sha256': {path.name: file_hash(path) for path in
                (folder / 'run_config.json', model_path, prediction_path, folder / 'metrics.json')}}
        write_json(folder / 'completed.json', result)
        results.append(result)
        print('CONTROL COMPLETE {} {}'.format(name, json.dumps(all_metrics['test'])), flush=True)
    return results


def summarize(output):
    output = Path(output)
    grouped = {}
    for path in sorted(output.glob('*/completed.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        _verify_completed(path.parent, result['run_signature'])
        key = result['protocol'] + '__random__' + result['method']
        grouped.setdefault(key, []).append(result)
    rows = []
    for name, group in sorted(grouped.items()):
        if len({result['campaign_signature'] for result in group}) != 1:
            raise ValueError('Refusing to aggregate incompatible control configurations: ' + name)
        if len({result['fold'] for result in group}) != len(group):
            raise ValueError('Duplicate fold in control summary')
        row = {'experiment': name, 'category': 'feature-based control',
               'completed_folds': len(group), 'fold_ids': ','.join(str(result['fold']) for result in group)}
        for metric in METRIC_NAMES:
            values = [result['test_metrics'][metric] for result in group]
            if any(value is None for value in values):
                row[metric + '_mean'] = row[metric + '_sd'] = None
            else:
                row[metric + '_mean'] = float(np.mean(values))
                row[metric + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', rows)
    if rows:
        with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=HERE / 'prepared')
    parser.add_argument('--output', type=Path, default=HERE / 'runs' / 'classical_controls_v1')
    parser.add_argument('--graph-cache', type=Path,
                        help='Existing deep campaign graph_cache for exact shared fold embeddings.')
    parser.add_argument('--protocols', default='pair,protein')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--graph-epochs', type=int, default=5)
    args = parser.parse_args()
    methods = tuple(args.methods.split(','))
    protocols = tuple(args.protocols.split(','))
    folds = tuple(int(value) for value in args.folds.split(','))
    if len(methods) != len(set(methods)) or len(protocols) != len(set(protocols)) or len(folds) != len(set(folds)):
        parser.error('Duplicate method, protocol, or fold')
    for protocol in protocols:
        for fold in folds:
            run_fold(args.prepared, args.output, protocol, fold, seed=args.seed,
                     graph_epochs=args.graph_epochs, graph_cache=args.graph_cache, methods=methods)
            summarize(args.output)


if __name__ == '__main__':
    main()
