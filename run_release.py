"""Portable, additive entry points for the frozen CaPPI revision release.

Existing training sources and prepared manifests are never rewritten. Archived
prediction verification does not assert checkpoint replay or a fresh blind test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


METRICS = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc',
           'auroc', 'average_precision', 'aupr')
FAMILIES = ('residual', 'topology_dropout', 'topology_dropout_bypass',
            'sequence_only', 'topology_only', 'concat_matched')
SELECTION_RULE = 'per condition-family-fold: minimum best validation BCE, ties choose lower LR'
BENCHMARK_SCOPE = 'adaptive_same_benchmark_no_fresh_independent_test'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def report(value, destination=None):
    result = json.dumps(value, ensure_ascii=False, indent=2)
    if destination:
        path = Path(destination).resolve()
        require(not path.exists(), 'Choose a new report path; existing evidence is never overwritten')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result + '\n', encoding='utf-8')
        compact = {key: item for key, item in value.items() if key not in ('runs', 'folds')}
        compact['report_path'] = str(path)
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    else:
        print(result)


def setup_scientific(root):
    # Preserve the verified Windows DLL import order before NumPy/SciPy/sklearn.
    import torch
    torch.set_num_threads(2)
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_option(root, value, default):
    path = Path(value) if value else root / default
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def require_new_run(root, path):
    protected = ('prepared', 'data', 'runs', 'results', 'historical_baselines', 'legacy')
    require(not any(path.is_relative_to(root / name) for name in protected),
            'Use a fresh run directory outside archived data/results (for example runs_new/)')


def archived_run(campaign, original):
    """Interpret original Windows paths as metadata, never access their roots."""
    parts = str(original).replace('\\', '/').split('/')
    positions = [i for i, part in enumerate(parts) if part in ('trials', 'selected_repeats')]
    require(len(positions) == 1 and len(parts) == positions[0] + 2,
            'Invalid archived run path: ' + str(original))
    relative = parts[positions[0]:]
    require(all(part not in ('', '.', '..') for part in relative), 'Unsafe archive-relative path')
    path = (campaign / relative[0] / relative[1]).resolve()
    require(path.is_relative_to(campaign.resolve()), 'Archived run resolves outside campaign')
    return path


def inspect_data(args, root):
    from cappi_data import _verify_artifacts, load_fold
    prepared = resolve_option(root, args.prepared, 'prepared')
    manifest = read(prepared / 'manifest.json')
    _verify_artifacts(prepared, manifest)
    rows = []
    for negative in ('random', 'degree_matched'):
        for protocol in ('pair', 'protein'):
            for fold in range(manifest['n_folds']):
                data = load_fold(prepared, negative, protocol, fold)
                rows.append({'negative': negative, 'protocol': protocol, 'fold': fold,
                    'split_hash': data['split_hash'], 'training_pairs': len(data['train_pairs']),
                    'validation_pairs': len(data['val_pairs']), 'test_pairs': len(data['test_pairs']),
                    'train_positive_graph_edges': len(data['graph_edges'])})
    report({'status': 'passed', 'scope': 'prepared-data integrity and all saved fold contracts',
            'n_proteins': manifest['n_nodes'], 'n_positive_associations': manifest['n_positives'],
            'dataset_hash': manifest['dataset_hash'], 'manifest_sha256': sha(prepared / 'manifest.json'),
            'verified_folds': len(rows), 'folds': rows,
            'raw_source_paths': 'Informational original paths retained; no manifest rewriting'}, args.report)


def calculate(labels, probabilities):
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score, auc, confusion_matrix,
        f1_score, matthews_corrcoef, precision_recall_curve, precision_score, roc_auc_score)
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, p)
    return {'accuracy': float(accuracy_score(y, pred)),
        'sensitivity': float(tp / (tp + fn)), 'specificity': float(tn / (tn + fp)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y, pred)), 'auroc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)), 'aupr': float(auc(recall, precision))}


def numerical_match(actual, expected, message, tolerance=1e-10):
    require(abs(float(actual) - float(expected)) <= tolerance, message)


def verify_predictions(args, root):
    import numpy as np
    from cappi_data import load_fold
    prepared = resolve_option(root, args.prepared, 'prepared')
    campaign = resolve_option(root, args.campaign, 'runs/repair_20261007')
    results = resolve_option(root, args.results, 'results/repair_20261007')
    manifest = read(prepared / 'manifest.json')
    index = read(campaign / 'selected_index.json')
    contract = read(campaign / 'campaign_contract.json')
    jobs = index['runs']
    expected_keys = {(condition['protocol'], condition['negative'], family, fold, repeat)
        for condition in contract['conditions'] for family in condition['families']
        for fold in contract['folds'] for repeat in contract['repeat_indices']}
    actual_keys = [(j['protocol'], j['negative'], j['family'], j['fold'], j['repeat']) for j in jobs]
    require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys,
            'Selected index does not cover the frozen condition/fold/repeat matrix exactly')
    require(len(jobs) == contract['counts']['selected_test_evaluations'], 'Wrong selected test count')
    folds, detail = {}, []
    for job in jobs:
        key = job['negative'], job['protocol'], job['fold']
        if key not in folds:
            folds[key] = load_fold(prepared, *key)
        data = folds[key]
        folder = archived_run(campaign, job['out'])
        complete = read(folder / 'test_completed.json')
        cfg = read(folder / 'run_config.json')
        declaration = read(folder / 'selection_frozen.json')
        require(complete['evaluation_partition'] == 'test' and complete['test_evaluated'] is True,
                'Run has not completed its selected test evaluation')
        require(cfg['split_hash'] == complete['split_hash'] == data['split_hash'], 'Fold provenance differs')
        require(cfg['run_signature'] == complete['run_signature'] == declaration['run_signature'],
                'Run signatures differ')
        require(declaration['selected_before_test'] is True, 'No pre-test frozen selection')
        require(complete['checkpoint_sha256'] == declaration['checkpoint_sha256'], 'Checkpoint identity differs')
        require(sha(folder / 'selection_frozen.json') == complete['selection_declaration_sha256'],
                'Frozen declaration bytes changed')
        require(cfg['prepared_manifest_sha256'] == sha(prepared / 'manifest.json'), 'Prepared manifest differs')
        for config_key, job_key in (('protocol', 'protocol'), ('negative_sampling', 'negative'),
                                   ('fold', 'fold'), ('model_seed', 'model_seed'), ('graph_seed', 'graph_seed')):
            require(cfg[config_key] == job[job_key], 'Selected job and saved configuration differ')
        require(cfg['settings']['family'] == job['family']
                and cfg['learning_rate'] == job['learning_rate'], 'Selected family or LR differs')
        row = {'protocol': job['protocol'], 'negative': job['negative'], 'family': job['family'],
               'fold': job['fold'], 'repeat': job['repeat'], 'archive_relative_run': folder.relative_to(campaign).as_posix()}
        for partition, pair_key, label_key in (('validation', 'val_pairs', 'val_labels'),
                                             ('test', 'test_pairs', 'test_labels')):
            path = folder / (partition + '_predictions.csv')
            expected_hash = (complete['artifact_sha256'][path.name] if partition == 'test'
                             else read(folder / 'completed.json')['artifact_sha256'][path.name])
            require(sha(path) == expected_hash, 'Prediction bytes differ from saved provenance: ' + str(path))
            with path.open(encoding='utf-8-sig', newline='') as handle:
                samples = list(csv.DictReader(handle))
            pairs = np.asarray([(int(s['protein_u']), int(s['protein_v'])) for s in samples], dtype=np.int64)
            labels = np.asarray([int(s['label']) for s in samples], dtype=int)
            probabilities = np.asarray([float(s['probability']) for s in samples], dtype=float)
            expected_pairs = data['node_ids'][data[pair_key]]
            require(np.array_equal(pairs, expected_pairs) and np.array_equal(labels, data[label_key]),
                    'Saved prediction membership/labels differ from prepared fold')
            require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
                    'Non-finite or invalid probability')
            require(all(int(s['prediction']) == int(p >= .5) for s, p in zip(samples, probabilities)),
                    'Saved classes do not use fixed threshold 0.5')
            require(all(s['partition'] == partition and s['split_hash'] == data['split_hash']
                and s['checkpoint_sha256'] == complete['checkpoint_sha256']
                and int(s['fold']) == job['fold'] and int(s['model_seed']) == job['model_seed']
                and int(s['graph_seed']) == job['graph_seed'] for s in samples),
                'Per-pair metadata differs from selected run')
            values = calculate(labels, probabilities)
            saved_metrics = read(folder / (partition + '_metrics.json'))
            for metric, actual in values.items():
                saved_key = 'pr_trapezoid_area' if metric == 'aupr' else metric
                numerical_match(actual, saved_metrics[saved_key], 'Saved metric mismatch: ' + metric)
                row[partition + '_' + metric] = actual
        detail.append(row)
    with (results / 'summary.csv').open(encoding='utf-8-sig', newline='') as handle:
        summary = list(csv.DictReader(handle))
    groups = {(r['protocol'], r['negative'], r['family']) for r in detail}
    require(groups == {(r['protocol'], r['negative'], r['family']) for r in summary},
            'Published summary group coverage differs')
    for published in summary:
        group = [r for r in detail if (r['protocol'], r['negative'], r['family']) ==
                 (published['protocol'], published['negative'], published['family'])]
        for partition in ('validation', 'test'):
            for metric in METRICS:
                name = partition + '_' + metric
                repeat_values = [np.asarray([r[name] for r in group if r['repeat'] == repeat])
                                 for repeat in contract['repeat_indices']]
                require(all(len(v) == len(contract['folds']) for v in repeat_values), 'Incomplete fold group')
                means = np.asarray([v.mean() for v in repeat_values])
                quantities = {'mean': means.mean(), 'seed_sd': means.std(ddof=1),
                              'fold_sd_mean': np.mean([v.std(ddof=1) for v in repeat_values])}
                for suffix, value in quantities.items():
                    numerical_match(value, published[name + '_' + suffix], 'Summary hierarchy mismatch: ' + name)
    report({'status': 'passed', 'scope': 'archived prediction integrity, prepared membership/labels and numerical metrics',
            'verified_selected_runs': len(detail), 'verified_prediction_files': len(detail) * 2,
            'verified_summary_groups': len(summary), 'metric_names': list(METRICS),
            'aupr_definition': 'Native precision-recall curve trapezoidal area; average_precision stored separately',
            'uncertainty': 'Each model repeat first averages five folds; sample SD over three repeat means',
            'dataset_hash': manifest['dataset_hash'], 'selected_index_sha256': sha(campaign / 'selected_index.json'),
            'evaluation_scope': BENCHMARK_SCOPE, 'fresh_independent_test': False,
            'checkpoint_replay_performed': False,
            'excluded_scope': 'Original 7.33 GB optimizer/checkpoint states are not required for this numerical verification',
            'runs': detail}, args.report)


def train(args, root):
    prepared = resolve_option(root, args.prepared, 'prepared')
    output = resolve_option(root, args.output, '')
    cache = resolve_option(root, args.graph_cache, 'runs/reviewer_v1/graph_cache')
    settings = {'prepared': str(prepared), 'output': str(output),
        'config': {'family': args.family, 'learning_rate': args.learning_rate},
        'protocol': args.protocol, 'negative': args.negative, 'fold': args.fold,
        'model_seed': 20260930 + args.repeat * 1000 + args.fold,
        'graph_seed': 20260930 + args.fold, 'device': args.device,
        'epochs': args.epochs, 'patience': args.patience, 'resume': args.resume,
        'source_graph_cache': str(cache) if cache.is_dir() else None, 'no_test': True}
    environment = os.environ.copy()
    if args.sgns_python:
        interpreter = Path(args.sgns_python).resolve()
        require(interpreter.is_file(), 'SGNS Python executable not found')
        environment['CAPPI_GENSIM_PYTHON'] = str(interpreter)
    command = [sys.executable, str(root / 'run_missing_topology.py'),
               '--single-json', json.dumps(settings, ensure_ascii=False)]
    if args.show_config:
        report({'training_settings': settings, 'training_started': False,
                'sgns_python': environment.get('CAPPI_GENSIM_PYTHON', 'original helper default')})
        return
    require_new_run(root, output)
    require(not output.is_relative_to(prepared), 'Training output must be outside prepared data')
    subprocess.run(command, check=True, cwd=root, env=environment)


def select(args, root):
    from run_missing_topology import verify_completed
    paths = [resolve_option(root, value, '') for value in args.candidates]
    require(len(paths) == 2 and len(set(paths)) == 2, 'Supply exactly two distinct LR candidate runs')
    for path in paths:
        require_new_run(root, path)
    runs = [verify_completed(path) for path in paths]
    require({r['config']['learning_rate'] for r in runs} == {.0005, .001}, 'Supply both declared learning rates')
    keys = ('prepared_manifest_sha256', 'split_hash', 'protocol', 'negative_sampling', 'fold',
            'model_seed', 'graph_seed', 'max_epochs', 'patience', 'core_code_sha256', 'repair_code_sha256')
    for key in keys:
        require(runs[0]['config'][key] == runs[1]['config'][key], 'Candidate configuration differs beyond LR: ' + key)
    for key in ('family', 'topology_dropout', 'missing_policy'):
        require(runs[0]['settings'][key] == runs[1]['settings'][key], 'Candidate model family settings differ')
    require(all(r['config']['model_seed'] == r['config']['graph_seed'] for r in runs),
            'The declared learning-rate search is limited to model repeat 0')
    require(all(not (p / 'test_completed.json').exists() for p in paths),
            'Refuse selecting candidates whose test outcomes already exist')
    chosen = min(zip(paths, runs), key=lambda item: (item[1]['best_validation_bce'], item[1]['config']['learning_rate']))
    declaration = {'selected_before_test': True, 'run_signature': chosen[1]['run_signature'],
        'checkpoint_sha256': chosen[1]['checkpoint_sha256'], 'selection_rule': SELECTION_RULE,
        'learning_rate': chosen[1]['config']['learning_rate'],
        'test_status': 'Previously inspected benchmark; adaptive comparison, not fresh blind test.',
        'candidates': [{'relative_or_absolute_run': str(path), 'learning_rate': run['config']['learning_rate'],
                       'best_validation_bce': run['best_validation_bce'], 'checkpoint_sha256': run['checkpoint_sha256']}
                       for path, run in zip(paths, runs)]}
    destination = chosen[0] / 'selection_frozen.json'
    require(not destination.exists(), 'A frozen selection already exists; do not overwrite it')
    destination.write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report({'selected_run': str(chosen[0]), 'declaration': declaration, 'test_evaluation_started': False})


def freeze_repeat(args, root):
    from run_missing_topology import verify_completed
    output = resolve_option(root, args.run, '')
    selected = resolve_option(root, args.selected, '')
    require_new_run(root, output)
    require_new_run(root, selected)
    current, reference = verify_completed(output), verify_completed(selected)
    declaration = read(selected / 'selection_frozen.json')
    require(declaration.get('selected_before_test') is True
            and declaration['run_signature'] == reference['run_signature']
            and declaration['checkpoint_sha256'] == reference['checkpoint_sha256'],
            'Reference repeat-0 selection is not frozen for its actual checkpoint')
    require(reference['config']['model_seed'] == reference['config']['graph_seed'],
            'Reference must be the chosen repeat-0 checkpoint')
    keys = ('prepared_manifest_sha256', 'split_hash', 'protocol', 'negative_sampling', 'fold',
            'graph_seed', 'max_epochs', 'patience', 'core_code_sha256', 'repair_code_sha256', 'learning_rate')
    for key in keys:
        require(current['config'][key] == reference['config'][key], 'Repeat differs from selected settings: ' + key)
    require(current['settings'] == reference['settings'] and current['graph_hash'] == reference['graph_hash'],
            'Repeat model/graph differs from selected repeat-0 configuration')
    offset = current['config']['model_seed'] - reference['config']['model_seed']
    require(offset in (1000, 2000), 'The declared additional model repeats use seed offsets 1000/2000')
    require(not (output / 'test_completed.json').exists(), 'Refuse freezing after this repeat was evaluated')
    destination = output / 'selection_frozen.json'
    require(not destination.exists(), 'A frozen declaration already exists; do not overwrite it')
    record = {'selected_before_test': True, 'run_signature': current['run_signature'],
        'checkpoint_sha256': current['checkpoint_sha256'],
        'selection_rule': 'Freeze repeat-0 validation-selected LR; this repeat selects checkpoint by validation BCE only',
        'learning_rate': current['config']['learning_rate'], 'repeat': offset // 1000,
        'repeat0_declaration_sha256': sha(selected / 'selection_frozen.json'),
        'repeat0_run_signature': reference['run_signature'],
        'test_status': 'Previously inspected benchmark; adaptive comparison, not fresh blind test.'}
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report({'frozen_repeat_run': str(output), 'declaration': record, 'test_evaluation_started': False})


def evaluate(args, root):
    out = resolve_option(root, args.run, '')
    require_new_run(root, out)
    prepared = resolve_option(root, args.prepared, 'prepared')
    settings = {'out': str(out), 'prepared': str(prepared), 'device': args.device}
    subprocess.run([sys.executable, str(root / 'run_missing_topology.py'), '--evaluate-selected',
                    '--single-json', json.dumps(settings, ensure_ascii=False)], check=True, cwd=root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Release directory containing the unchanged scientific scripts')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'verify'):
        part = sub.add_parser(command, help='Read-only data audit' if command == 'inspect' else 'Recompute archived prediction metrics')
        part.add_argument('--prepared')
        part.add_argument('--report', type=Path, help='Optional new JSON report path; existing files never overwritten')
        if command == 'verify':
            part.add_argument('--campaign')
            part.add_argument('--results')
    part = sub.add_parser('train', help='Run one new train/validation fit; never evaluate test outcomes')
    part.add_argument('--prepared')
    part.add_argument('--output', required=True)
    part.add_argument('--graph-cache')
    part.add_argument('--family', choices=FAMILIES, default='residual')
    part.add_argument('--protocol', choices=('pair', 'protein'), default='pair')
    part.add_argument('--negative', choices=('random', 'degree_matched'), default='random')
    part.add_argument('--fold', type=int, choices=range(5), default=0)
    part.add_argument('--repeat', type=int, choices=(0, 1, 2), default=0)
    part.add_argument('--learning-rate', type=float, choices=(.0005, .001), default=.0005)
    part.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    part.add_argument('--epochs', type=int, default=300)
    part.add_argument('--patience', type=int, default=20)
    part.add_argument('--resume', action='store_true')
    part.add_argument('--sgns-python', help='Explicit separate gensim 3.1.0 Python executable')
    part.add_argument('--show-config', action='store_true', help='Print resolved settings without training')
    part = sub.add_parser('select', help='Freeze the lower-validation-BCE result of two declared LR runs')
    part.add_argument('--candidates', nargs=2, required=True)
    part = sub.add_parser('freeze-repeat', help='Freeze an additional repeat using the pre-test selected repeat-0 LR')
    part.add_argument('--run', required=True)
    part.add_argument('--selected', required=True, help='Selected repeat-0 run with frozen declaration')
    part = sub.add_parser('evaluate', help='Evaluate a newly fitted checkpoint only after pre-test selection is frozen')
    part.add_argument('--run', required=True)
    part.add_argument('--prepared')
    part.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    root = setup_scientific(args.root)
    {'inspect': inspect_data, 'verify': verify_predictions, 'train': train,
     'select': select, 'freeze-repeat': freeze_repeat, 'evaluate': evaluate}[args.command](args, root)


if __name__ == '__main__':
    main()
