"""Verify saved supplementary predictions and produce publication figure sources.

This script performs no training, model selection, or test-set tuning. Original
artifacts are read-only. Parameter plots use validation predictions only.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch  # Required Windows DLL import order in the existing environment.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef

from cappi_data import load_fold

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / 'runs' / 'supplement_20261001'
OUTPUT = HERE / 'results' / 'supplement_20261001'
METRICS = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc',
           'auroc', 'average_precision', 'pr_trapezoid_area')
METRIC_LABELS = ('Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1',
                 'MCC', 'AUROC', 'Average precision (AP)', 'PR trapezoid area')
COLORS = ['#222222', '#1764AB', '#D27A20', '#537D3B', '#936BB0', '#BA527C', '#369AAB', '#7C6C50']
LINESTYLES = ['-', '-', '--', '-.', ':', '--', '-.', ':']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(path).with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def calculate(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.shape != p.shape or not len(y) or set(y) != {0, 1}:
        raise ValueError('Expected nonempty binary prediction arrays with both classes')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probability outside [0,1] or nonfinite')
    pred = p >= .5
    tp = int(np.sum((y == 1) & pred)); tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred))
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(n=len(y), positive_n=int(y.sum()), accuracy=(tp + tn) / len(y),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                precision=tp / (tp + fp) if tp + fp else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                mcc=float(matthews_corrcoef(y, pred)), auroc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                pr_trapezoid_area=float(auc(recall, precision)))


def assert_metrics(actual, saved, source):
    for metric in METRICS:
        if metric not in saved or not np.isclose(actual[metric], saved[metric], rtol=0, atol=1e-9):
            raise ValueError(f'{source}: inconsistent {metric}: {actual[metric]} / {saved.get(metric)}')


def verify_frame(path, data, split):
    # Preserve exported float64 values exactly; rounding can merge ranking ties.
    frame = pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip')
    required = {'protein_u', 'protein_v', 'label', 'probability'}
    if not required.issubset(frame):
        raise ValueError(f'{path}: missing columns {required - set(frame)}')
    actual = list(zip(frame.protein_u.astype(int), frame.protein_v.astype(int), frame.label.astype(int)))
    expected = [(int(data['node_ids'][u]), int(data['node_ids'][v]), int(y))
                for (u, v), y in zip(data[split + '_pairs'], data[split + '_labels'])]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(f'{path}: prediction membership or labels differ from immutable {split} split')
    if 'split_hash' in frame and set(frame.split_hash) != {data['split_hash']}:
        raise ValueError(f'{path}: split fingerprint differs')
    if 'prediction' in frame and not np.array_equal(frame.prediction.astype(int), (frame.probability >= .5).astype(int)):
        raise ValueError(f'{path}: fixed classification threshold differs')
    result = calculate(frame.label.values, frame.probability.values)
    return frame, result


def verify_preserved():
    manifest = json.loads((CAMPAIGN / 'supplement_manifest.json').read_text(encoding='utf-8'))
    records = []
    for name, expected in manifest['preserved_files_sha256'].items():
        actual = sha(HERE / name)
        if actual != expected:
            raise ValueError(f'An original protected artifact changed: {name}')
        records.append({'path': name, 'sha256': actual})
    return records


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def save_figure(fig, stem):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT / (stem + '.' + ext), dpi=300, bbox_inches='tight')
    plt.close(fig)


def summary_row(name, frames, scope, **extra):
    result = {'condition': name, 'scope': scope, 'completed_folds': len(frames), **extra}
    values = [calculate(f.label.values, f.probability.values) for f in frames]
    for metric in METRICS:
        array = np.array([v[metric] for v in values])
        result[metric + '_mean'] = float(array.mean())
        result[metric + '_sd'] = float(array.std(ddof=1)) if len(array) > 1 else None
    return result


def precision_steps(labels, probabilities, grid):
    """Right-endpoint precision steps consistent with sklearn average precision.

    Linear interpolation can inflate the displayed area for tied scores, notably
    five-neighbor KNN. Keep recall duplicates: the first ascending match supplies
    the precision of the threshold that first attained this recall.
    """
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    locations = np.searchsorted(recall[::-1], grid, side='left')
    return precision[::-1][locations]


def curves(groups, title, stem):
    """Mean fold curves; legend numbers are mean fold metrics, not pooled metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    grid = np.linspace(0, 1, 1001)
    for i, (name, frames) in enumerate(groups.items()):
        rocs, prs, aucs, aps = [], [], [], []
        for frame in frames:
            y, p = frame.label.values, frame.probability.values
            fpr, tpr, _ = roc_curve(y, p)
            roc_y = np.interp(grid, fpr, tpr); roc_y[0] = 0.; roc_y[-1] = 1.
            rocs.append(roc_y)
            prs.append(precision_steps(y, p, grid))
            aucs.append(roc_auc_score(y, p)); aps.append(average_precision_score(y, p))
        kwargs = dict(color=COLORS[i % len(COLORS)], linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=1.8)
        axes[0].plot(grid, np.mean(rocs, axis=0), label=f'{name} ({np.mean(aucs):.4f})', **kwargs)
        # The final duplicate recall=1 point displays the all-positive endpoint
        # without giving its vertical segment nonzero integration weight.
        axes[1].step(np.r_[grid, 1.], np.r_[np.mean(prs, axis=0), np.mean([f.label.mean() for f in frames])],
                     where='pre', label=f'{name} ({np.mean(aps):.4f})', **kwargs)
    axes[0].plot([0, 1], [0, 1], color='.65', ls='--', lw=.9)
    prevalences = [float(f.label.mean()) for frames in groups.values() for f in frames]
    if max(prevalences) - min(prevalences) < 1e-4:
        axes[1].axhline(np.mean(prevalences), color='.65', ls='--', lw=.9)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC | mean fold AUROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall | mean fold AP')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02)); ax.grid(alpha=.16)
        ax.legend(loc='lower left', frameon=False)
    if stem == 'classical_protein_roc_pr':
        axes[0].legend(loc='lower right', frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.text(.5, -.005, 'Curves averaged over five held-out folds; legend values are fold metric means. AP is average precision.',
             ha='center', fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def old_neural_group(key):
    frames = []
    for fold in range(5):
        folder = HERE / 'runs' / 'reviewer_v1' / f'{key}__fold{fold}'
        record = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
        cfg = record['config']
        data = load_fold(HERE / 'prepared', cfg['negative_sampling'], cfg['protocol'], fold)
        frame, result = verify_frame(folder / 'test_predictions.csv', data, 'test')
        if sha(folder / 'test_predictions.csv') != record['predictions_sha256']:
            raise ValueError('Original neural predictions changed')
        assert_metrics(result, record['test_metrics'], folder)
        frames.append(frame)
    return frames


def performance_figures():
    base = old_neural_group('pair__random__drw__residual')
    degree = old_neural_group('pair__degree_matched__drw__residual')
    groups = {'Random negatives': base, 'Approx. degree-matched negatives': degree}
    curves(groups, 'Negative-sampling sensitivity | outer test folds', 'negative_sampling_roc_pr')
    rows = [summary_row(k, v, 'outer_test') for k, v in groups.items()]
    pd.DataFrame(rows).to_csv(OUTPUT / 'negative_sampling_summary.csv', index=False, encoding='utf-8-sig')
    fold_rows = []
    for name, frames in groups.items():
        for fold, frame in enumerate(frames):
            fold_rows.append(dict(condition=name, fold=fold, **calculate(frame.label, frame.probability)))
    pd.DataFrame(fold_rows).to_csv(OUTPUT / 'negative_sampling_fold_metrics.csv', index=False, encoding='utf-8-sig')
    groups = {'CaPPI residual fusion': base}
    for variant, name in [('concat_matched', 'Capacity-matched MLP'), ('concat_linear', 'Linear concatenation'),
                          ('sequence_only', 'Sequence only'), ('topology_only', 'Topology only')]:
        groups[name] = old_neural_group('pair__random__drw__' + variant)
    curves(groups, 'Fusion and modality controls | outer test folds', 'ablation_roc_pr')
    pd.DataFrame([summary_row(k, v, 'outer_test') for k, v in groups.items()]).to_csv(
        OUTPUT / 'ablation_summary.csv', index=False, encoding='utf-8-sig')
    walks = {'DRW': base, 'Uniform walks': old_neural_group('pair__random__deepwalk__residual'),
             'node2vec': old_neural_group('pair__random__node2vec__residual')}
    curves(walks, 'Random-walk controls | outer test folds', 'walk_controls_roc_pr')
    return {'reused_neural_conditions': 8, 'scope': 'Existing verified neural results redrawn; no retraining'}


def classical_figures():
    names = [('aa20_drw_lr', 'Logistic regression'), ('aa20_drw_rf', 'Random forest'),
             ('aa20_drw_adaboost', 'AdaBoost'), ('aa20_drw_knn', 'KNN'),
             ('aa20_drw_svm', 'SVM'), ('aa20_drw_xgboost', 'XGBoost')]
    rows, verified = [], []
    for protocol in ('pair', 'protein'):
        neural = 'pair__random__drw__residual' if protocol == 'pair' else 'protein__random__drw__residual'
        groups = {'CaPPI': old_neural_group(neural)}
        for method, label in names:
            frames = []
            base = HERE / 'runs' / 'classical_controls_v1' if method in ('aa20_drw_lr', 'aa20_drw_rf') else CAMPAIGN / 'classical'
            for fold in range(5):
                folder = base / f'{protocol}__random__{method}__fold{fold}'
                record = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
                for filename, expected in record['artifacts_sha256'].items():
                    if sha(folder / filename) != expected:
                        raise ValueError(f'{folder}: changed {filename}')
                data = load_fold(HERE / 'prepared', 'random', protocol, fold)
                frame, result = verify_frame(folder / 'test_predictions.csv', data, 'test')
                assert_metrics(result, record['test_metrics'], folder)
                if set(frame.checkpoint_sha256) != {record['checkpoint_sha256']}:
                    raise ValueError('Classical prediction checkpoint identity differs')
                if sha(folder / 'model.joblib') != record['checkpoint_sha256']:
                    raise ValueError('Classical model differs from prediction record')
                frames.append(frame)
                verified.append({'run': folder.name, 'original_result_reused': method in ('aa20_drw_lr', 'aa20_drw_rf'),
                                 'prediction_sha256': sha(folder / 'test_predictions.csv'), 'metrics': result})
            groups[label] = frames
        # Models in a protocol must use exactly the same outer test examples.
        for fold in range(5):
            expected = set(zip(groups['CaPPI'][fold].protein_u, groups['CaPPI'][fold].protein_v, groups['CaPPI'][fold].label))
            for frames in groups.values():
                if set(zip(frames[fold].protein_u, frames[fold].protein_v, frames[fold].label)) != expected:
                    raise ValueError('Different comparison test populations')
        title = 'Pair-level evaluation' if protocol == 'pair' else 'Both-protein holdout evaluation'
        curves(groups, title + ' | six feature classifiers and CaPPI', 'classical_' + protocol + '_roc_pr')
        rows.extend(summary_row(k, v, 'outer_test', protocol=protocol) for k, v in groups.items())
    pd.DataFrame(rows).to_csv(OUTPUT / 'classical_comparison_summary.csv', index=False, encoding='utf-8-sig')
    write_json(OUTPUT / 'classical_independent_verification.json', {'status': 'passed', 'new_evaluations': 40,
               'reused_lr_rf_evaluations': 20, 'runs': verified})
    return {'new_evaluations_verified': 40, 'reused_lr_rf_evaluations_verified': 20,
            'feature_scope': 'Six classifiers share AA20/DRW pair features; CaPPI learns sequence/topology representations'}


def parameter_figures():
    # Use the validation-only loader, including its check that test labels are not read.
    from run_supplement_neural import build_grid, load_validation_fold
    base = CAMPAIGN / 'parameter_sensitivity'
    configurations, membership = build_grid()
    rows, fold_rows, verified = [], [], []
    grouped = {}
    data_cache = {fold: load_validation_fold(HERE / 'prepared', fold) for fold in range(5)}
    for config in configurations:
        frames = []
        for fold in range(5):
            folder = base / f'{config.name}__fold{fold}'
            record = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
            if record['evaluation_partition'] != 'validation' or record['test_evaluated']:
                raise ValueError('Test results found in validation sensitivity figures')
            if record['fold'] != fold or record['config_name'] != config.name:
                raise ValueError('Parameter configuration/fold mismatch')
            if record['settings'] != asdict(config):
                raise ValueError('Saved numerical settings do not match the declared grid configuration')
            for filename, expected in record['artifact_sha256'].items():
                if sha(folder / filename) != expected:
                    raise ValueError(f'{folder}: changed {filename}')
            frame, result = verify_frame(folder / 'validation_predictions.csv', data_cache[fold], 'val')
            if set(frame.partition) != {'validation'}:
                raise ValueError('Invalid sensitivity prediction partition')
            assert_metrics(result, record['validation_metrics'], folder)
            if sha(folder / 'best.pt') != record['checkpoint_sha256'] or set(frame.checkpoint_sha256) != {record['checkpoint_sha256']}:
                raise ValueError('Sensitivity model/prediction identity differs')
            history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
            best_loss = float('inf'); best_epoch = None
            for entry in history:
                if entry['validation_bce'] < best_loss - 1e-6:
                    best_loss, best_epoch = entry['validation_bce'], entry['epoch']
            if best_epoch != record['best_epoch'] or abs(best_loss - record['best_validation_bce']) > 1e-6:
                raise ValueError('Checkpoint was not selected by the declared validation BCE rule')
            frames.append(frame)
            fold_rows.append({'config_name': config.name, **record['settings'], 'fold': fold,
                              'baseline_reused': record['baseline_reused'], 'best_epoch': best_epoch, **result})
            verified.append({'run': folder.name, 'prediction_sha256': sha(folder / 'validation_predictions.csv'),
                             'checkpoint_sha256': record['checkpoint_sha256'], 'baseline_reused': record['baseline_reused']})
        grouped[config.name] = frames
        row = summary_row(config.name, frames, 'validation_sensitivity', **record['settings'])
        row['groups'] = ';'.join(name for name, members in membership.items() if config.name in members)
        rows.append(row)
    if len(verified) != 140 or sum(v['baseline_reused'] for v in verified) != 5:
        raise ValueError('Incomplete parameter-sensitivity matrix')
    summary = pd.DataFrame(rows); folds = pd.DataFrame(fold_rows)
    summary.to_csv(OUTPUT / 'parameter_validation_summary.csv', index=False, encoding='utf-8-sig')
    folds.to_csv(OUTPUT / 'parameter_validation_fold_metrics.csv', index=False, encoding='utf-8-sig')

    # Figure 1: annotate every grid cell; no interpolation or invented combinations.
    selected = summary[summary.condition.isin(membership['walk_dimension'])]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    for ax, metric, label in zip(axes, ('accuracy', 'auroc', 'average_precision'), ('Accuracy', 'AUROC', 'AP')):
        table = selected.pivot(index='walk_length', columns='dim', values=metric + '_mean').sort_index()
        if table.shape != (4, 4) or table.isna().any().any():
            raise ValueError('Incomplete walk/dimension grid')
        values = table.to_numpy(); im = ax.imshow(values, origin='lower', cmap='Blues', aspect='auto')
        ax.set_xticks(range(4), table.columns); ax.set_yticks(range(4), table.index)
        ax.set(xlabel='Embedding dimension', ylabel='Walk length', title=label)
        midpoint = (values.min() + values.max()) / 2
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f'{value:.3f}', ha='center', va='center', color='white' if value > midpoint else '#222222', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle('Validation sensitivity | five-fold means', fontsize=13)
    save_figure(fig, 'walk_dimension_validation_heatmap')

    # Figure 2: show all five fold values over each box; the folds are not five independent datasets.
    selected_names = membership['kernel_length']
    configs_by_name = {c.name: c for c in configurations}
    labels = [f'{configs_by_name[name].kernel}×{configs_by_name[name].max_length}' for name in selected_names]
    plotting_metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'mcc', 'f1', 'average_precision', 'auroc']
    plotting_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'MCC', 'F1', 'AP', 'AUROC']
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7), constrained_layout=True)
    for ax, metric, label in zip(axes.flat, plotting_metrics, plotting_labels):
        values = [folds.loc[folds.config_name == name, metric].values for name in selected_names]
        boxes = ax.boxplot(values, patch_artist=True, widths=.62, showfliers=False,
                           medianprops={'color': '#202020', 'linewidth': 1.1})
        for patch in boxes['boxes']:
            patch.set(facecolor='#C5DDED', edgecolor='#325B76', alpha=.85)
        for i, v in enumerate(values, 1):
            ax.scatter(i + np.linspace(-.12, .12, len(v)), v, s=11, c='#174B6D', zorder=3)
        ax.set_xticks(range(1, 10), labels, rotation=55, ha='right')
        ax.set(title=label, ylabel=label); ax.grid(axis='y', alpha=.17)
    fig.suptitle('Validation sensitivity | kernel × maximum sequence length | five folds', fontsize=13)
    save_figure(fig, 'kernel_length_validation_boxplots')

    selected = summary[summary.condition.isin(membership['initial_learning_rate'])].sort_values('learning_rate')
    selected.to_csv(OUTPUT / 'learning_rate_validation_summary.csv', index=False, encoding='utf-8-sig')
    headers = ['Initial LR', 'ACC', 'Sen', 'Spec', 'Prec', 'MCC', 'F1', 'AUROC', 'AP']
    ms = ['accuracy', 'sensitivity', 'specificity', 'precision', 'mcc', 'f1', 'auroc', 'average_precision']
    text_rows = [[f'{row.learning_rate:g}'] + [f'{row[m+"_mean"]:.4f} ± {row[m+"_sd"]:.4f}' for m in ms]
                 for _, row in selected.iterrows()]
    fig, ax = plt.subplots(figsize=(15, 2.5)); ax.axis('off')
    table = ax.table(cellText=text_rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#CDCDCD'); cell.set_linewidth(.5)
        if r == 0:
            cell.set_facecolor('#E9F0F5'); cell.set_text_props(weight='bold')
    ax.set_title('Initial learning-rate sensitivity | validation mean ± fold SD', pad=12)
    fig.text(.5, .015, 'Validation selects the checkpoint and supplies these descriptive metrics. No outer-test hyperparameter selection.',
             ha='center', fontsize=9)
    save_figure(fig, 'learning_rate_validation_table')
    write_json(OUTPUT / 'parameter_independent_verification.json', {'status': 'passed', 'verified_folds': 140,
               'new_training_runs': 135, 'base_checkpoints_reused': 5, 'test_evaluated': False,
               'scope': 'Descriptive validation sensitivity; same validation partition used for checkpoint selection', 'runs': verified})
    return {'verified_folds': 140, 'new_training_runs': 135, 'base_checkpoints_reused': 5, 'scope': 'validation_only'}


def case_results():
    base = load_fold(HERE / 'prepared', 'random', 'pair', 0)
    folder = CAMPAIGN / 'cases'
    mapping = json.loads((folder / 'source_mapping_verified.json').read_text(encoding='utf-8'))
    for path, expected in mapping['sources'].items():
        if sha(path) != expected:
            raise ValueError('Target source-mapping artifact changed')
    with np.load(HERE / 'prepared' / 'positives.npz', allow_pickle=False) as z:
        known = set(map(tuple, z['pairs'].tolist()))
    indices = {int(v): i for i, v in enumerate(base['node_ids'])}
    rows = []
    for target_name, target in mapping['targets'].items():
        out = folder / ('target_' + target_name)
        training = json.loads((out / 'training_completed.json').read_text(encoding='utf-8'))
        ranking = json.loads((out / 'ranking_completed.json').read_text(encoding='utf-8'))
        for name, expected in training['artifacts'].items():
            if sha(out / name) != expected:
                raise ValueError('Target training artifact changed')
        target_index = target['index']; target_id = int(target['node_id'])
        with np.load(out / 'split_arrays.npz', allow_pickle=False) as stored:
            for split in ('train', 'val'):
                mask = ~np.any(base[split + '_pairs'] == target_index, axis=1)
                for suffix in ('pairs', 'labels'):
                    if not np.array_equal(stored[split + '_' + suffix], base[split + '_' + suffix][mask]):
                        raise ValueError('Target exclusion altered unrelated split membership')
                if np.any(stored[split + '_pairs'] == target_index):
                    raise ValueError('Target present in optimization or validation')
            graph = set(map(tuple, stored['graph_edges'].tolist()))
            train_positive = set(map(tuple, stored['train_pairs'][stored['train_labels'] == 1].tolist()))
            if graph != train_positive or any(target_index in edge for edge in graph):
                raise ValueError('Target incident graph edge or nontraining edge')
        with np.load(out / 'topology.npz', allow_pickle=False) as topology:
            if bool(topology['present'][target_index]) or np.any(topology['embedding'][target_index] != 0):
                raise ValueError('Target received an unavailable topology vector')
        candidates = pd.read_csv(out / 'all_candidates.csv', encoding='utf-8-sig')
        top20 = pd.read_csv(out / 'top20_candidates.csv', encoding='utf-8-sig')
        if len(candidates) != 581 or set(candidates.partner_node_id) != set(base['node_ids']) - {target_id}:
            raise ValueError('Target ranking does not cover all 581 other proteins')
        if set(candidates.target_node_id) != {target_id} or not top20.equals(candidates.iloc[:20].reset_index(drop=True)):
            raise ValueError('Target identity or top20 rows differ')
        expected_labels = [int(tuple(sorted((target_index, indices[int(partner)]))) in known)
                           for partner in candidates.partner_node_id]
        if not np.array_equal(candidates.known_association, expected_labels):
            raise ValueError('Target labels differ from supplied database association records')
        p = candidates.model_score.to_numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or np.any(np.diff(p) > 0):
            raise ValueError('Target scores not finite, probabilistic, and descending')
        for metric, actual in {'auroc': roc_auc_score(expected_labels, p),
                               'average_precision': average_precision_score(expected_labels, p),
                               'hits_at_20_count': int(sum(expected_labels[:20]))}.items():
            if not np.isclose(actual, ranking[metric], atol=1e-9, rtol=0):
                raise ValueError('Target ranking metric differs: ' + metric)
        if (sha(out / 'best.pt') != ranking['checkpoint_sha256'] or
                sha(out / 'all_candidates.csv') != ranking['all_candidates_sha256'] or
                sha(out / 'top20_candidates.csv') != ranking['top20_sha256']):
            raise ValueError('Target ranking artifact identity differs')
        rows.append({'target': target_name, 'n_candidates': 581, 'known_associations': sum(expected_labels),
                     'auroc': ranking['auroc'], 'average_precision': ranking['average_precision'],
                     'hits_at_20_count': ranking['hits_at_20_count'], 'precision_at_20': ranking['precision_at_20'],
                     'scope': 'Single-target retrospective holdout; supplied associations versus unobserved pairs',
                     'all_candidates_file': str(out / 'all_candidates.csv'),
                     'top20_file': str(out / 'top20_candidates.csv')})
    if len(rows) != 2:
        raise ValueError('Missing target')
    trajectory = folder / 'trajectory_fold0'
    result = json.loads((trajectory / 'training_completed.json').read_text(encoding='utf-8'))
    if result['epochs_completed'] != 600 or not result['trajectory']:
        raise ValueError('Missing 600-epoch descriptive trajectory')
    for filename, expected in result['artifacts'].items():
        if sha(trajectory / filename) != expected:
            raise ValueError('Trajectory artifact changed')
    for epoch in (50, 150, 300, 600):
        with np.load(trajectory / f'validation_hidden_epoch_{epoch}.npz', allow_pickle=False) as stored:
            if (not np.array_equal(stored['pairs'], base['val_pairs']) or
                    not np.array_equal(stored['labels'], base['val_labels']) or
                    stored['features'].shape != (len(base['val_pairs']), 32) or
                    not np.isfinite(stored['features']).all() or int(stored['epoch']) != epoch):
                raise ValueError('Invalid trajectory snapshot or validation population')
    pd.DataFrame(rows).to_csv(OUTPUT / 'target_holdout_summary.csv', index=False, encoding='utf-8-sig')
    report = {'status': 'passed', 'targets': rows, 'trajectory_epochs': [50, 150, 300, 600],
              'trajectory_validation_pairs': len(base['val_pairs']), 'docking': 'excluded_by_user',
              'projection_status': 'Check separate UMAP completion and visual QA records'}
    write_json(OUTPUT / 'cases_independent_verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=['performance', 'classical', 'parameters', 'cases', 'all', 'preserved'], default='preserved')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True); style()
    protected = verify_preserved()
    record = {'status': 'passed', 'protected_artifacts': protected}
    if args.section in ('performance', 'all'):
        record['performance'] = performance_figures()
    if args.section in ('classical', 'all'):
        record['classical'] = classical_figures()
    if args.section in ('parameters', 'all'):
        record['parameters'] = parameter_figures()
    if args.section in ('cases', 'all'):
        record['cases'] = case_results()
    write_json(OUTPUT / ('verification_' + args.section + '.json'), record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
