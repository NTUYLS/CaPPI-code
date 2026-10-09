"""Render only complete, independently verified CaPPI repair test predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch  # Established Windows import order, before sklearn.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
CAMPAIGN = RESULTS.parents[1] / 'runs' / 'repair_20261007'
STYLE = {
    'residual': ('CaPPI', '#111111', '-'),
    'topology_dropout': ('Topology dropout', '#0072B2', '--'),
    'topology_dropout_bypass': ('Dropout + bypass', '#CC79A7', '-.'),
    'sequence_only': ('Sequence only', '#0072B2', ':'),
    'topology_only': ('Topology only', '#CC79A7', '--'),
}
FIGURES = (
    ('pair_random_repair', 'pair', 'random', ('residual', 'topology_dropout', 'topology_dropout_bypass')),
    ('protein_random_repair', 'protein', 'random', ('residual', 'topology_dropout', 'topology_dropout_bypass', 'sequence_only')),
    ('pair_random_modalities', 'pair', 'random', ('residual', 'sequence_only', 'topology_only')),
    ('pair_degree_modalities', 'pair', 'degree_matched', ('residual', 'sequence_only', 'topology_only')),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_checks():
    verification = read(RESULTS / 'verification.json')
    collection = read(RESULTS / 'collection_status.json')
    require(verification.get('status') == 'passed' and verification.get('scope') == 'selected_test',
            'A complete independent selected-test verification is required.')
    require(collection.get('status') == 'complete_verified', 'A verified completed collection is required.')
    require(verification.get('verified_training_runs') == 300 and verification.get('verified_test_evaluations') == 225,
            'This script requires the full frozen campaign.')
    require(collection['verification_report_sha256'] == sha(RESULTS / 'verification.json'),
            'Verification changed after collection.')
    require(collection['selected_index_sha256'] == sha(CAMPAIGN / 'selected_index.json'),
            'Selected runs changed after collection.')
    for name in ('run_metrics.csv', 'summary.csv'):
        require(collection['files'][name] == sha(RESULTS / name), 'Verified result file changed: ' + name)
    detail, summary = pd.read_csv(RESULTS / 'run_metrics.csv'), pd.read_csv(RESULTS / 'summary.csv')
    require(len(detail) == 225 and len(summary) == 15, 'Result coverage differs from the contract.')
    return detail, summary


def average_curves(rows):
    require(len(rows) == 15 and set(rows['repeat']) == {0, 1, 2}, 'Expected three repeats of five folds.')
    x = np.linspace(0, 1, 1001)
    seed_rocs, seed_prs, seed_aucs, seed_auprs, prevalence, provenance = [], [], [], [], [], []
    for repeat, selected in rows.groupby('repeat', sort=True):
        require(len(selected) == 5 and set(selected.fold) == set(range(5)), 'Repeat fold coverage differs.')
        rocs, prs, aucs, auprs = [], [], [], []
        for row in selected.sort_values('fold').itertuples(index=False):
            path = Path(row.run_dir) / 'test_predictions.csv'
            require(sha(path) == row.test_prediction_sha256, 'Prediction changed after verification.')
            frame = pd.read_csv(path)
            y, probability = frame['label'].to_numpy(), frame['probability'].to_numpy()
            require(set(np.unique(y)) == {0, 1} and np.isfinite(probability).all(), 'Invalid verified scores.')
            fpr, tpr, _ = roc_curve(y, probability)
            precision, recall, _ = precision_recall_curve(y, probability)
            auc_value, aupr_value = roc_auc_score(y, probability), auc(recall, precision)
            require(abs(auc_value - row.test_auroc) < 1e-10 and abs(aupr_value - row.test_aupr) < 1e-10,
                    'Figure metric disagrees with the verified prediction.')
            roc = np.interp(x, fpr, tpr)
            roc[0], roc[-1] = 0, 1
            pr = np.interp(x, recall[::-1], precision[::-1])
            pr[0] = precision[-1]
            # At recall=1, the empirical precision is the class prevalence.
            # Preserve that endpoint; do not invent a fall to precision=0.
            pr[-1] = precision[0]
            rocs.append(roc); prs.append(pr); aucs.append(auc_value); auprs.append(aupr_value)
            prevalence.append(float(np.mean(y)))
            provenance.append({'run_dir': str(path.parent), 'prediction_sha256': sha(path),
                               'fold': int(row.fold), 'repeat': int(repeat),
                               'checkpoint_sha256': row.checkpoint_sha256,
                               'auc': float(auc_value), 'aupr': float(aupr_value),
                               'positive_prevalence': float(np.mean(y))})
        seed_rocs.append(np.mean(rocs, axis=0)); seed_prs.append(np.mean(prs, axis=0))
        seed_aucs.append(np.mean(aucs)); seed_auprs.append(np.mean(auprs))
    return {'x': x, 'roc': np.mean(seed_rocs, axis=0), 'pr': np.mean(seed_prs, axis=0),
            'auc': float(np.mean(seed_aucs)), 'aupr': float(np.mean(seed_auprs)),
            'prevalence': float(np.mean(prevalence)), 'source_runs': provenance}


def main():
    detail, summary = source_checks()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'legend.fontsize': 7.5, 'lines.linewidth': 1.5,
                         'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'savefig.dpi': 300})
    records = []
    for name, protocol, negative, families in FIGURES:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
        fig.subplots_adjust(left=.073, right=.986, bottom=.155, top=.85, wspace=.25)
        curves = {}
        for family in families:
            rows = detail[(detail.protocol == protocol) & (detail.negative == negative) & (detail.family == family)]
            values = average_curves(rows)
            expected = summary[(summary.protocol == protocol) & (summary.negative == negative) & (summary.family == family)]
            require(len(expected) == 1, 'Summary group not found.')
            require(abs(values['auc'] - expected.iloc[0].test_auroc_mean) < 1e-10
                    and abs(values['aupr'] - expected.iloc[0].test_aupr_mean) < 1e-10, 'Mean legend metric differs.')
            label, color, linestyle = STYLE[family]
            axes[0].plot(values['x'], values['roc'], color=color, linestyle=linestyle,
                         label=f"{label} (AUC = {values['auc']:.4f})")
            axes[1].plot(values['x'], values['pr'], color=color, linestyle=linestyle,
                         label=f"{label} (AUPR = {values['aupr']:.4f})")
            curves[family] = {key: value for key, value in values.items() if key not in ('x', 'roc', 'pr')}
        axes[0].plot([0, 1], [0, 1], '--', color='#666666', linewidth=.85, zorder=0)
        titles = ('Receiver operating characteristic curves', 'Precision–Recall curves')
        for index, ax in enumerate(axes):
            ax.set(xlim=(0, 1), ylim=(0, 1.01), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
            ax.set_xlabel(('False Positive Rate', 'Recall')[index])
            ax.set_ylabel(('True Positive Rate', 'Precision')[index])
            ax.set_title(titles[index], pad=5)
            ax.grid(True, linestyle='--', linewidth=.6, color='#BBBBBB')
            ax.set_axisbelow(True)
            ax.text(-.065, 1.085, ('a', 'b')[index], transform=ax.transAxes, fontsize=14)
            legend = ax.legend(loc=('lower right', 'lower left')[index], frameon=True, fancybox=False,
                               framealpha=1, borderpad=.5, handlelength=2.6)
            legend.get_frame().set_edgecolor('#BBBBBB'); legend.get_frame().set_linewidth(.65)
        fig.canvas.draw()
        outputs = {}
        for ext in ('png', 'pdf', 'svg'):
            output = HERE / f'{name}.{ext}'
            fig.savefig(output)
            outputs[ext] = {'path': str(output), 'sha256': sha(output)}
        plt.close(fig)
        records.append({'name': name, 'protocol': protocol, 'negative': negative,
                        'curves': curves, 'outputs': outputs})
    manifest = {'status': 'verified_predictions_rendered', 'figures': records,
                'script_sha256': sha(Path(__file__)), 'verification_sha256': sha(RESULTS / 'verification.json'),
                'collection_status_sha256': sha(RESULTS / 'collection_status.json'),
                'selected_index_sha256': sha(CAMPAIGN / 'selected_index.json'),
                'summary_sha256': sha(RESULTS / 'summary.csv'), 'run_metrics_sha256': sha(RESULTS / 'run_metrics.csv'),
                'legend_definition': 'Mean of each repeat five-fold metric mean across three model repeats; AUPR is trapezoid PR area.',
                'curve_definition': '1001-point linear interpolation per fold, mean within repeat, then mean across three repeats.',
                'curve_area_equals_legend': False,
                'endpoint_definition': 'ROC origin fixed0/0; PR recall0/precision1 follows the source curve terminal convention; recall1 endpoint is empirical positive prevalence; no artificial recall1/precision0 segment.',
                'benchmark_scope': 'Adaptive reuse of the previously inspected benchmark; not a fresh blind external evaluation.',
                'capacity_control': 'Retained in numerical summaries; omitted from these figures as requested.'}
    (HERE / 'figure_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Rendered four verified ROC/PR figure groups.')


if __name__ == '__main__':
    main()
