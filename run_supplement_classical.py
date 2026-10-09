"""Fixed CPU classical controls on the immutable September 30 folds and features.

This additive campaign never modifies prepared data, existing controls, or graph
caches. XGBoost 2.1.1 lives in a task-local --no-deps target, not the ML environment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

# This environment requires torch DLLs to load before sklearn's runtime.
import torch
torch.set_num_threads(2)
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from cappi_data import load_fold
from cappi_graph import GraphConfig, fingerprint
import run_classical_baselines as base
from verify_classical_results import calculate_metrics, check_metrics

HERE = Path(__file__).resolve().parent
DEPS = HERE / '.supplement_deps'
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
import xgboost
from xgboost import XGBClassifier

METHODS = ('aa20_drw_adaboost', 'aa20_drw_knn', 'aa20_drw_svm', 'aa20_drw_xgboost')
SPLITS = ('train', 'val', 'test')
DEFAULT_OUTPUT = HERE / 'runs' / 'supplement_20261001' / 'classical'


def code_hash():
    return base.json_hash({name: base.file_hash(HERE / name) for name in (
        'run_supplement_classical.py', 'run_classical_baselines.py', 'cappi_data.py',
        'cappi_graph.py', 'legacy_skipgram.py', 'verify_classical_results.py')})


def model_config(method):
    configs = {
        'aa20_drw_adaboost': dict(algorithm='AdaBoostClassifier', n_estimators=200,
            learning_rate=1.0, algorithm_variant='SAMME.R', weak_learner='DecisionTreeClassifier(max_depth=1)',
            standardization='none'),
        'aa20_drw_knn': dict(algorithm='KNeighborsClassifier', n_neighbors=5,
            weights='uniform', metric='minkowski', p=2, algorithm_variant='brute', n_jobs=2,
            standardization='StandardScaler fit on training pairs only'),
        'aa20_drw_svm': dict(algorithm='SVC', C=1.0, kernel='rbf', gamma='scale',
            probability=True, class_weight=None, cache_size=1024, tol=0.001,
            max_iter=-1, shrinking=True, calibration='libsvm internal five-fold probability calibration on training pairs only',
            standardization='StandardScaler fit on training pairs only'),
        'aa20_drw_xgboost': dict(algorithm='XGBClassifier', n_estimators=200,
            max_depth=6, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0,
            min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0, objective='binary:logistic',
            eval_metric='logloss', tree_method='hist', device='cpu', n_jobs=2,
            standardization='none', early_stopping='none'),
    }
    if method not in configs:
        raise ValueError('Unknown supplement method')
    return configs[method]


def fixed_estimator(method, seed):
    if method == 'aa20_drw_adaboost':
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
            n_estimators=200, learning_rate=1.0, algorithm='SAMME.R', random_state=seed)
    if method == 'aa20_drw_knn':
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute',
            metric='minkowski', p=2, n_jobs=2)
    elif method == 'aa20_drw_svm':
        clf = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True,
            class_weight=None, cache_size=1024, tol=.001, max_iter=-1,
            shrinking=True, random_state=seed)
    elif method == 'aa20_drw_xgboost':
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=.1,
            subsample=1., colsample_bytree=1., min_child_weight=1,
            reg_alpha=0., reg_lambda=1., objective='binary:logistic', eval_metric='logloss',
            tree_method='hist', device='cpu', n_jobs=2, random_state=seed)
    else:
        raise ValueError('Unknown supplement method')
    return Pipeline([('scaler', StandardScaler()), ('classifier', clf)])


def load_existing_topology(data, graph_cache, seed):
    config = GraphConfig()
    identity = fingerprint(data['graph_edges'], data['n_nodes'], config, seed)
    path = Path(graph_cache) / (identity + '.npz')
    metadata_path = path.with_suffix('.json')
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError('Required original graph cache is missing; refusing to regenerate: ' + str(path))
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    if meta['npz_sha256'] != base.file_hash(path) or meta['fingerprint'] != identity:
        raise ValueError('Original graph cache fingerprint or content changed')
    if meta['config'] != asdict(config) or meta['seed'] != seed:
        raise ValueError('Original graph configuration changed')
    with np.load(path, allow_pickle=False) as stored:
        embedding, present = stored['embedding'].copy(), stored['present'].copy()
        if str(stored['fingerprint']) != identity:
            raise ValueError('Embedded graph fingerprint changed')
    expected = np.zeros(data['n_nodes'], dtype=bool)
    expected[data['graph_edges'].reshape(-1)] = True
    if (embedding.shape != (data['n_nodes'], 64) or not np.array_equal(present, expected)
            or not np.isfinite(embedding).all() or np.any(embedding[~present] != 0)):
        raise ValueError('Invalid shared graph embedding or zero-topology mask')
    nodes = np.concatenate([base.amino_acid_composition(data['sequences']), embedding], axis=1)
    features = {split: base.symmetric_pair_features(nodes, data[split + '_pairs']) for split in SPLITS}
    return features, dict(graph_hash=identity, graph_cache_sha256=base.file_hash(path),
        graph_metadata_sha256=base.file_hash(metadata_path), graph_feature_hash=base.array_hash(embedding, present),
        node_feature_hash=base.array_hash(nodes), feature_hashes={key: base.array_hash(value) for key, value in features.items()})


def check_reference_controls(reference, data, features, proof, protocol, fold):
    """Prove same raw features by replaying the saved LR/RF prediction arrays."""
    checks = []
    split_hashes = {split: base.array_hash(data[split + '_pairs'], data[split + '_labels']) for split in SPLITS}
    for method in ('aa20_drw_lr', 'aa20_drw_rf'):
        folder = Path(reference) / f'{protocol}__random__{method}__fold{fold}'
        record = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
        base._verify_completed(folder, record['run_signature'])
        cfg = record['config']
        if (cfg['code_sha256'] != base.code_hash() or cfg['split_hash'] != data['split_hash']
                or cfg['split_array_hashes'] != split_hashes or cfg['graph_hash'] != proof['graph_hash']
                or cfg['dataset_hash'] != data['manifest']['dataset_hash']):
            raise ValueError('Original LR/RF source, graph, dataset, or split mismatch')
        model = joblib.load(folder / 'model.joblib')
        # Thread limit changes execution only, never model content or predictions.
        if hasattr(model, 'n_jobs'):
            model.set_params(n_jobs=2)
        scores = model.predict_proba(features['test'])[:, list(model.classes_).index(1)]
        with (folder / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        old_pairs = np.array([(int(row['protein_u']), int(row['protein_v'])) for row in rows])
        old_y = np.array([int(row['label']) for row in rows])
        old_scores = np.array([float(row['probability']) for row in rows])
        if (not np.array_equal(old_pairs, data['node_ids'][data['test_pairs']])
                or not np.array_equal(old_y, data['test_labels'])
                or not np.allclose(scores, old_scores, rtol=0, atol=1e-12)):
            raise ValueError('Shared feature replay disagrees with original LR/RF predictions')
        checks.append(dict(method=method, completed_sha256=base.file_hash(folder / 'completed.json'),
            prediction_sha256=base.file_hash(folder / 'test_predictions.csv'),
            checkpoint_sha256=base.file_hash(folder / 'model.joblib'), n=len(rows),
            max_abs_prediction_difference=float(np.max(np.abs(scores-old_scores)))))
    return checks


def write_predictions(path, data, scores, method, protocol, fold, model_hash, splits=SPLITS):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['protein_u', 'protein_v', 'label', 'probability', 'prediction',
                         'split', 'method', 'protocol', 'fold', 'checkpoint_sha256'])
        for split in splits:
            for (u, v), y, score in zip(data[split + '_pairs'], data[split + '_labels'], scores[split]):
                writer.writerow([int(data['node_ids'][u]), int(data['node_ids'][v]), int(y), float(score),
                    int(score >= .5), split, method, protocol, fold, model_hash])


def verify_completed(folder, data, expected_identity=None):
    record = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
    cfg = record['config']
    if record['run_signature'] != base.json_hash(cfg) or cfg['code_sha256'] != code_hash():
        raise ValueError('Changed supplement code or invalid signature')
    if expected_identity is not None and cfg != expected_identity:
        raise ValueError('Incompatible supplement resume identity')
    base._verify_completed(folder, record['run_signature'])
    if (cfg['split_hash'] != data['split_hash'] or cfg['dataset_hash'] != data['manifest']['dataset_hash']
            or cfg['split_array_hashes'] != {s: base.array_hash(data[s+'_pairs'], data[s+'_labels']) for s in SPLITS}):
        raise ValueError('Supplement split membership changed')
    if record['checkpoint_sha256'] != base.file_hash(folder / 'model.joblib'):
        raise ValueError('Changed model checkpoint')
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    with (folder / 'predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sum(len(data[s + '_labels']) for s in SPLITS):
        raise ValueError('Prediction row count mismatch')
    for split in SPLITS:
        selected = [row for row in rows if row['split'] == split]
        ids = np.array([(int(row['protein_u']), int(row['protein_v'])) for row in selected])
        labels = np.array([int(row['label']) for row in selected])
        scores = np.array([float(row['probability']) for row in selected])
        predicted = np.array([int(row['prediction']) for row in selected])
        if (not np.array_equal(ids, data['node_ids'][data[split + '_pairs']])
                or not np.array_equal(labels, data[split + '_labels'])
                or not np.array_equal(predicted, (scores >= .5).astype(int))):
            raise ValueError('Prediction IDs, labels, or fixed threshold mismatch')
        for key, value in dict(method=cfg['method'], protocol=cfg['protocol'], fold=str(cfg['fold']),
                               checkpoint_sha256=record['checkpoint_sha256']).items():
            if {row[key] for row in selected} != {value}:
                raise ValueError('Prediction provenance mismatch: ' + key)
        check_metrics(calculate_metrics(labels, scores, True), metrics[split], folder.name + '/' + split)
    check_metrics(metrics['test'], record['test_metrics'], folder.name)
    return record


def run_fold(prepared, output, graph_cache, reference, protocol, fold, methods=METHODS):
    if protocol not in ('pair', 'protein') or not set(methods).issubset(METHODS):
        raise ValueError('Unsupported supplement protocol or method')
    data = load_fold(prepared, 'random', protocol, fold)
    base.validate_graph_contract(data)
    seed = data['manifest']['seed'] + fold
    if data['manifest']['seed'] != 20260930:
        raise ValueError('This campaign requires the original frozen seed')
    features, proof = load_existing_topology(data, graph_cache, seed)
    references = check_reference_controls(reference, data, features, proof, protocol, fold)
    common = dict(code_sha256=code_hash(), original_classical_code_sha256=base.code_hash(),
        dataset_hash=data['manifest']['dataset_hash'], base_seed=20260930, negative_sampling='random',
        protocol=protocol, graph=asdict(GraphConfig()),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            sklearn=sklearn.__version__, joblib=joblib.__version__, xgboost=xgboost.__version__, torch=torch.__version__),
        feature_definition='Full-sequence canonical AA20 plus raw DRW64; concat(abs(u-v),u*v); 168D.',
        selection='Fixed prespecified settings; train-only fit/calibration; no validation/test tuning or early stopping.',
        threshold=.5, cpu_threads=2, category='feature-based classical controls; not published model reproductions')
    results = []
    for method in methods:
        campaign = dict(common, method=method, model=model_config(method))
        cfg = dict(campaign, seed=seed, fold=fold, split_hash=data['split_hash'],
            split_array_hashes={s: base.array_hash(data[s+'_pairs'], data[s+'_labels']) for s in SPLITS},
            prepared_manifest_sha256=base.file_hash(Path(prepared)/'manifest.json'),
            feature_proof=proof, original_control_replay=references)
        name = f'{protocol}__random__{method}__fold{fold}'
        folder = Path(output) / name
        if folder.exists():
            if not (folder / 'completed.json').exists():
                raise FileExistsError('Incomplete run retained; use a new output directory: ' + str(folder))
            results.append(verify_completed(folder, data, cfg))
            print('VERIFIED REUSE ' + name, flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        signature = base.json_hash(cfg)
        base.write_json(folder / 'run_config.json', dict(cfg, run_signature=signature,
            campaign_signature=base.json_hash(campaign), python_executable=sys.executable))
        print('START ' + name, flush=True)
        model = fixed_estimator(method, seed)
        model.fit(features['train'], data['train_labels'].astype(int))
        if isinstance(model, Pipeline):
            if not np.allclose(model.named_steps['scaler'].mean_, features['train'].mean(axis=0, dtype=np.float64), atol=1e-7):
                raise ValueError('Scaler did not fit training features')
        model_path = folder / 'model.joblib'
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)
        scores = {split: loaded.predict_proba(features[split])[:, list(loaded.classes_).index(1)] for split in SPLITS}
        model_hash = base.file_hash(model_path)
        metrics = {split: base.compute_metrics(data[split + '_labels'], scores[split], True) for split in SPLITS}
        base.write_json(folder / 'metrics.json', metrics)
        write_predictions(folder / 'predictions.csv', data, scores, method, protocol, fold, model_hash)
        write_predictions(folder / 'test_predictions.csv', data, scores, method, protocol, fold, model_hash, splits=('test',))
        record = dict(name=name, method=method, protocol=protocol, fold=fold, config=cfg,
            run_signature=signature, campaign_signature=base.json_hash(campaign),
            split_hash=data['split_hash'], split_array_hashes=cfg['split_array_hashes'],
            checkpoint_sha256=model_hash, test_evaluated=True, test_metrics=metrics['test'],
            elapsed_seconds=time.monotonic()-started,
            artifacts_sha256={p.name: base.file_hash(p) for p in (folder/'run_config.json', model_path,
                folder/'predictions.csv', folder/'test_predictions.csv', folder/'metrics.json')})
        base.write_json(folder / 'completed.json', record)
        verify_completed(folder, data, cfg)
        results.append(record)
        print('COMPLETE ' + name + ' ' + json.dumps(metrics['test']) + f' elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=HERE/'prepared')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--graph-cache', type=Path, default=HERE/'runs'/'reviewer_v1'/'graph_cache')
    parser.add_argument('--reference', type=Path, default=HERE/'runs'/'classical_controls_v1')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--protocols', default='pair,protein')
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    methods, protocols = tuple(args.methods.split(',')), tuple(args.protocols.split(','))
    folds = tuple(int(x) for x in args.folds.split(','))
    if (len(set(methods)) != len(methods) or len(set(protocols)) != len(protocols)
            or len(set(folds)) != len(folds) or not set(methods).issubset(METHODS)):
        parser.error('Duplicate or unknown campaign item')
    if xgboost.__version__ != '2.1.1':
        raise RuntimeError('Pinned supplement requires XGBoost 2.1.1')
    records = []
    with threadpool_limits(limits=2):
        for protocol in protocols:
            for fold in folds:
                if args.verify_only:
                    data = load_fold(args.prepared, 'random', protocol, fold)
                    features, proof = load_existing_topology(data, args.graph_cache, 20260930+fold)
                    references = check_reference_controls(args.reference, data, features, proof, protocol, fold)
                    for method in methods:
                        folder = args.output/f'{protocol}__random__{method}__fold{fold}'
                        record = verify_completed(folder, data)
                        if record['config']['feature_proof'] != proof or record['config']['original_control_replay'] != references:
                            raise ValueError('Original feature replay proof changed')
                        records.append(record)
                else:
                    records.extend(run_fold(args.prepared,args.output,args.graph_cache,args.reference,protocol,fold,methods))
                    base.summarize(args.output)
    base.summarize(args.output)
    base.write_json(args.output/'verification.json', dict(status='passed', completed_runs=len(records),
        requested_runs=len(methods)*len(protocols)*len(folds), methods=methods, protocols=protocols, folds=folds,
        code_sha256=code_hash(), original_lr_rf_replay=f'{2*len(protocols)*len(folds)} original saved models matched shared frozen features',
        metrics='All train/validation/test metrics independently recomputed from ID-checked CSV predictions',
        completed_sha256={r['name']:base.file_hash(args.output/r['name']/'completed.json') for r in records}))
    print(f'SUPPLEMENT VERIFIED {len(records)} runs', flush=True)


if __name__ == '__main__':
    main()
