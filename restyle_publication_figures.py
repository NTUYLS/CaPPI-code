"""Restyle verified CaPPI displays without fitting models or changing scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

import collect_supplement_results as source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'results/supplement_20261001'
OUTPUT = ROOT / 'results/publication_style_20261001'
DEFINITION = ('AUPR is computed as non-interpolated, recall-weighted average precision '
              '(sklearn average_precision_score); values are arithmetic means across '
              'five held-out folds, not trapezoidal areas or the area of the mean curve.')
BLUE = '#2469A0'
PINK = '#BD477B'
STYLES = {
    'CaPPI residual fusion': ('#111111', '-'), 'CaPPI': ('#111111', '-'),
    'Random negatives': ('#111111', '-'), 'DRW': ('#111111', '-'),
    'Approx. degree-matched negatives': (BLUE, '--'),
    'Capacity-matched MLP': (BLUE, '--'), 'Linear concatenation': (PINK, '-.'),
    'Sequence only': (BLUE, ':'), 'Topology only': (PINK, '--'),
    'Uniform walks': (BLUE, '--'), 'node2vec': (PINK, '-.'),
    'Logistic regression': (BLUE, '--'), 'Random forest': (PINK, '-.'),
    'AdaBoost': (BLUE, ':'), 'KNN': (PINK, '--'),
    'SVM': (BLUE, '-.'), 'XGBoost': (PINK, ':'),
}
DISPLAY = {'CaPPI residual fusion': 'CaPPI', 'Logistic regression': 'LR',
           'Random forest': 'RF', 'Approx. degree-matched negatives': 'Approx. degree-matched',
           'Capacity-matched MLP': 'Matched MLP'}
EXCLUSIONS = {'none': None, 'sequence_only': 'Sequence only',
              'topology_only': 'Topology only', 'concat_matched': 'Capacity-matched MLP',
              'concat_linear': 'Linear concatenation'}
RECORDS = []
TEXTS = {}
EXCLUDE = None


def save(fig, stem):
    text = [item.get_text() for item in fig.findobj(plt.Text)]
    if any(re.search(r'\bAP\b', value) for value in text):
        raise ValueError('Unconverted AP label in ' + stem)
    TEXTS[stem] = [value for value in text if value]
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT / f'{stem}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def draw_curves(groups, unused_title, stem):
    if stem == 'ablation_roc_pr' and EXCLUDE is not None:
        # Exclude from the current display bundle; original runs remain intact.
        groups = {name: frames for name, frames in groups.items() if name != EXCLUDE}
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.85))
    fig.subplots_adjust(left=.074, right=.985, bottom=.155, top=.86, wspace=.235)
    grid = np.linspace(0, 1, 1001)
    for name, frames in groups.items():
        rocs, prs, aucs, areas = [], [], [], []
        for frame in frames:
            labels, probabilities = frame.label.values, frame.probability.values
            fpr, tpr, _ = roc_curve(labels, probabilities)
            tpr_grid = np.interp(grid, fpr, tpr)
            tpr_grid[0], tpr_grid[-1] = 0., 1.
            rocs.append(tpr_grid)
            prs.append(source.precision_steps(labels, probabilities, grid))
            aucs.append(roc_auc_score(labels, probabilities))
            areas.append(average_precision_score(labels, probabilities))
        color, linestyle = STYLES[name]
        style = dict(color=color, linestyle=linestyle, linewidth=1.5)
        label = DISPLAY.get(name, name)
        axes[0].plot(grid, np.mean(rocs, axis=0),
                     label=f'{label} (AUC = {np.mean(aucs):.4f})', **style)
        prevalence = float(np.mean([f.label.mean() for f in frames]))
        axes[1].step(np.r_[grid, 1.], np.r_[np.mean(prs, axis=0), prevalence],
                     where='pre', label=f'{label} (AUPR = {np.mean(areas):.4f})', **style)
        RECORDS.append({'figure': stem, 'condition': name, 'folds': len(frames),
                        'AUC_mean': float(np.mean(aucs)), 'AUC_sd': float(np.std(aucs, ddof=1)),
                        'AUPR_mean': float(np.mean(areas)), 'AUPR_sd': float(np.std(areas, ddof=1)),
                        'PR_endpoint_precision': prevalence})
    axes[0].plot([0, 1], [0, 1], color='.3', linestyle='--', linewidth=.9, zorder=1)
    prevalences = [float(f.label.mean()) for frames in groups.values() for f in frames]
    if max(prevalences) - min(prevalences) < 1e-4:
        axes[1].axhline(np.mean(prevalences), color='.5', linestyle='--', linewidth=.9, zorder=1)
    axes[0].set(title='Receiver operating characteristic curves',
                xlabel='False Positive Rate', ylabel='True Positive Rate')
    axes[1].set(title='Precision-Recall curves', xlabel='Recall', ylabel='Precision')
    for panel, ax in zip(('a', 'b'), axes):
        # Leave a small margin so the recall=1 endpoint is not hidden by a spine.
        ax.set(xlim=(-.008, 1.008), ylim=(-.008, 1.018),
               xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
        ax.grid(color='.7', linestyle='--', linewidth=.65, alpha=.75)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', length=3, width=.7, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('.25'); spine.set_linewidth(.85)
        ax.text(-.07, 1.095, panel, transform=ax.transAxes, fontsize=16, va='bottom')
    for ax, location in zip(axes, ('lower right', 'lower left')):
        ax.legend(loc=location, fontsize=7.7, frameon=True, facecolor='white',
                  edgecolor='.8', framealpha=.97, fancybox=False, borderpad=.5,
                  handlelength=2.4, labelspacing=.35)
    save(fig, stem)


def parameter_figures():
    summary = pd.read_csv(PREVIOUS / 'parameter_validation_summary.csv', float_precision='round_trip')
    folds = pd.read_csv(PREVIOUS / 'parameter_validation_fold_metrics.csv', float_precision='round_trip')
    selected = summary[summary.groups.str.contains('walk_dimension')]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), constrained_layout=True)
    for panel, ax, metric, name in zip('abc', axes, ('accuracy', 'auroc', 'average_precision'), ('ACC', 'AUC', 'AUPR')):
        table = selected.pivot(index='walk_length', columns='dim', values=metric+'_mean').sort_index()
        assert table.shape == (4, 4) and not table.isna().any().any()
        values = table.to_numpy()
        im = ax.imshow(values, origin='lower', cmap='Blues', aspect='auto')
        ax.set(xticks=range(4), yticks=range(4), xlabel='Embedding dimension', ylabel='Walk length', title=f'({panel}) {name}')
        ax.set_xticklabels(table.columns); ax.set_yticklabels(table.index)
        middle = (values.min()+values.max())/2
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f'{value:.3f}', ha='center', va='center',
                    color='white' if value > middle else '#222222', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.047, pad=.04)
    save(fig, 'walk_dimension_validation_heatmap')

    selected = summary[summary.groups.str.contains('kernel_length')].sort_values(['kernel', 'max_length'])
    labels = [f'{r.kernel}×{r.max_length}' for r in selected.itertuples()]
    names = ['ACC', 'Sen', 'Spec', 'Prec', 'MCC', 'F1-score', 'AUPR', 'AUC']
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'mcc', 'f1', 'average_precision', 'auroc']
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 6.1), constrained_layout=True)
    for ax, metric, name in zip(axes.flat, metrics, names):
        values = [folds.loc[folds.config_name == config, metric].values for config in selected.condition]
        assert all(len(v) == 5 for v in values)
        boxes = ax.boxplot(values, patch_artist=True, widths=.6, showfliers=False,
                           medianprops={'color': '#111111', 'linewidth': 1.})
        for patch in boxes['boxes']:
            patch.set(facecolor='#D7E6F1', edgecolor=BLUE, linewidth=.9)
        for index, value in enumerate(values, 1):
            ax.scatter(index+np.linspace(-.1, .1, 5), value, s=10, color=BLUE, zorder=3)
        ax.set(title=name, ylabel=name, xticks=range(1, 10))
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle=':', linewidth=.6, color='.7')
    save(fig, 'kernel_length_validation_boxplots')

    selected = summary[summary.groups.str.contains('initial_learning_rate')].sort_values('learning_rate')
    names = ['LR', 'ACC', 'Sen', 'Spec', 'Prec', 'MCC', 'F1-score', 'AUC', 'AUPR']
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'mcc', 'f1', 'auroc', 'average_precision']
    values = [[f'{r.learning_rate:g}']+[f'{r[m+"_mean"]:.4f} ± {r[m+"_sd"]:.4f}' for m in metrics]
              for _, r in selected.iterrows()]
    fig, ax = plt.subplots(figsize=(13.6, 2.1)); ax.axis('off')
    fig.subplots_adjust(left=.015, right=.985, bottom=.035, top=.965)
    table = ax.table(cellText=values, colLabels=names, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.7)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('.75'); cell.set_linewidth(.5)
        if row == 0: cell.set_facecolor('#EDF2F6'); cell.set_text_props(weight='bold')
    save(fig, 'learning_rate_validation_table')


def main():
    global EXCLUDE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exclude-ablation', '--exclude-main-ablation',
                        dest='exclude_main_ablation', choices=EXCLUSIONS, default='concat_matched')
    args = parser.parse_args(); EXCLUDE = EXCLUSIONS[args.exclude_main_ablation]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous_manifest = json.loads((PREVIOUS/'supplement_delivery_manifest.json').read_text(encoding='utf-8'))
    for path, expected in previous_manifest['artifacts'].items():
        if source.sha(ROOT/path) != expected: raise ValueError('Previous delivered source changed: '+path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'axes.spines.top': True, 'axes.spines.right': True,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    source.OUTPUT = OUTPUT
    source.curves = draw_curves
    source.verify_preserved()
    source.performance_figures()
    source.classical_figures()
    draw_curves({'CaPPI': source.old_neural_group('protein__random__drw__residual'),
                 'Sequence only': source.old_neural_group('protein__random__drw__sequence_only')},
                '', 'protein_disjoint_modalities_roc_pr')
    full = pd.read_csv(OUTPUT/'ablation_summary.csv', float_precision='round_trip')
    if EXCLUDE:
        full[full.condition != EXCLUDE].to_csv(OUTPUT/'ablation_summary.csv', index=False, encoding='utf-8-sig')
    parameter_figures()
    pd.DataFrame(RECORDS).to_csv(OUTPUT/'curve_metrics.csv', index=False, encoding='utf-8-sig')
    notes = {'AUPR_definition': DEFINITION,
             'curves': 'Mean of five held-out folds. ROC uses linear interpolation; PR uses recall-weighted precision steps. No curve smoothing or score modification.',
             'parameter_displays': 'Descriptive validation sensitivity; the same validation partition selects checkpoints and supplies metrics.',
             'excluded_from_displays': EXCLUDE,
             'ablation_table': 'ablation_summary.csv',
             'ablation_figure': 'ablation_roc_pr.png',
             'scope': 'New display exports only. No original document or training artifact changed.'}
    source.write_json(OUTPUT/'figure_notes.json', notes)
    source.write_json(OUTPUT/'figure_text.json', TEXTS)
    caption = 'AUC and AUPR are arithmetic means of five held-out fold scores. '+DEFINITION
    titles = {'negative_sampling_roc_pr':'负采样比较', 'ablation_roc_pr':'融合与模态消融',
              'walk_controls_roc_pr':'随机游走对照', 'classical_pair_roc_pr':'蛋白对划分：分类器比较',
              'classical_protein_roc_pr':'双蛋白留出：分类器比较', 'protein_disjoint_modalities_roc_pr':'双蛋白留出：模态比较',
              'walk_dimension_validation_heatmap':'游走长度与嵌入维度', 'kernel_length_validation_boxplots':'卷积核与序列长度',
              'learning_rate_validation_table':'学习率'}
    cards = []
    for stem in TEXTS:
        links = ' · '.join(f'<a href="{stem}.{ext}">{ext.upper()}</a>' for ext in ('png','pdf','svg'))
        cards.append(f'<section><h2>{html.escape(titles[stem])}</h2><img src="{stem}.png" alt="{html.escape(titles[stem])}"><p>{links}</p></section>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CaPPI新版图表</title><style>body{max-width:1100px;margin:30px auto;padding:0 24px;font:15px Arial,sans-serif;color:#222}h1{font-size:26px}h2{font-size:19px;margin-top:36px}img{width:100%;height:auto}a{color:#2469a0}p{line-height:1.6}</style><h1>CaPPI 新版图表</h1>'+''.join(cards)+'<details><summary>计算口径与来源</summary><p>'+html.escape(caption)+'</p><p>参数图来自验证集描述性分析。</p><p><a href="figure_notes.json">图注与范围</a> · <a href="curve_metrics.csv">曲线指标</a> · <a href="ablation_summary.csv">消融表</a></p></details></html>'
    (OUTPUT/'figures.html').write_text(page, encoding='utf-8')
    source.write_json(OUTPUT/'restyle_manifest.json', {'status':'generated_pending_visual_qa',
        'created_at':datetime.now(timezone.utc).isoformat(), 'source_bundle':str(PREVIOUS),
        'source_bundle_sha256':source.sha(PREVIOUS/'supplement_delivery_manifest.json'),
        'generator_sha256':source.sha(Path(__file__)), 'new_training_runs':0,
        'excluded_from_displays':EXCLUDE, 'AUPR_definition':DEFINITION,
        'figures':list(TEXTS), 'artifacts':{p.name:source.sha(p) for p in OUTPUT.iterdir()
           if p.is_file() and p.name not in ('restyle_manifest.json','visual_qa.json')}})
    print(json.dumps({'status':'generated','figures':len(TEXTS),'path':str(OUTPUT)},ensure_ascii=False))


if __name__ == '__main__':
    main()
