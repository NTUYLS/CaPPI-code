"""Frozen validation-first CaPPI missing-topology campaign; previous work is read-only."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import torch  # Must precede sklearn on this Windows environment.

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'repair_20261007'
RESULTS = HERE / 'results' / 'repair_20261007'
CORE = ('cappi_data.py', 'cappi_graph.py', 'cappi_model.py', 'legacy_skipgram.py', 'run_experiments.py')
TRAIN_CODE = CORE + ('cappi_missing_topology.py', 'run_missing_topology.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def digest(value):
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True).encode()).hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(p)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def setup():
    contract = read(ROOT / 'campaign_contract.json')
    protected = {str(HERE / f): sha(HERE / f) for f in CORE}
    for relative in ('prepared/manifest.json', 'results/verified_summary.csv',
                     'results/verified_fold_metrics.csv', 'results/verified_classical_summary.csv',
                     'results/publication_style_20261001/ablation_summary.csv'):
        protected[str(HERE / relative)] = sha(HERE / relative)
    source_manifest = read(HERE / 'prepared' / 'manifest.json')
    for record in source_manifest['sources'].values():
        p = Path(record['path'])
        if sha(p) != record['sha256']:
            raise ValueError('Original source data changed: ' + str(p))
        protected[str(p)] = record['sha256']
    identity = {'contract_sha256': sha(ROOT / 'campaign_contract.json'),
                'training_sources_sha256': {f: sha(HERE / f) for f in TRAIN_CODE},
                'protected_files_sha256': protected,
                'torch': str(torch.__version__), 'python': sys.version,
                'executable': sys.executable, 'device': torch.cuda.get_device_name(0)}
    p = ROOT / 'frozen_campaign_identity.json'
    if p.exists() and normalized(read(p)) != normalized(identity):
        raise ValueError('Frozen campaign inputs differ; refuse mixed experiment.')
    write(p, identity)
    (ROOT / 'source_snapshot').mkdir(exist_ok=True)
    for f in TRAIN_CODE:
        snapshot = ROOT / 'source_snapshot' / f
        if snapshot.exists() and sha(snapshot) != identity['training_sources_sha256'][f]:
            raise ValueError('Frozen source snapshot differs.')
        if not snapshot.exists():
            snapshot.write_bytes((HERE / f).read_bytes())
    return contract, identity


def assert_inputs(identity):
    for f, expected in identity['training_sources_sha256'].items():
        if sha(HERE / f) != expected:
            raise ValueError('Training source changed during campaign: ' + f)
    for f, expected in identity['protected_files_sha256'].items():
        if sha(f) != expected:
            raise ValueError('Protected original file changed: ' + f)


def build_jobs(contract):
    jobs = []
    for condition in contract['conditions']:
        for family in condition['families']:
            for fold in contract['folds']:
                for lr in contract['validation_grid']['learning_rate']:
                    settings = contract['families'][family]
                    tag = f'{condition["protocol"]}__{condition["negative"]}__{family}__lr{lr:.4f}'.replace('.', 'p')
                    out = ROOT / 'trials' / f'{tag}__repeat0__fold{fold}'
                    jobs.append({'protocol': condition['protocol'], 'negative': condition['negative'],
                                 'family': family, 'fold': fold, 'learning_rate': lr,
                                 'repeat': 0, 'model_seed': 20260930 + fold,
                                 'graph_seed': 20260930 + fold, 'out': str(out),
                                 'topology_dropout': settings['topology_dropout'],
                                 'missing_policy': settings['missing_policy']})
    if len(jobs) != contract['counts']['validation_training_runs']:
        raise ValueError('Trial count does not match frozen contract.')
    return jobs


def update_progress(stage, jobs, status='running', active=None, started_at=None):
    complete = sum((Path(j['out']) / 'completed.json').exists() for j in jobs)
    all_trial = sum(1 for p in (ROOT / 'trials').glob('*/completed.json'))
    all_repeat = sum(1 for p in (ROOT / 'selected_repeats').glob('*/completed.json'))
    all_test = sum(1 for base in ('trials', 'selected_repeats')
                   for p in (ROOT / base).glob('*/test_completed.json'))
    active_runs = active if isinstance(active, list) else ([active] if active else [])
    value = {'status': status, 'stage': stage, 'stage_completed': complete,
             'stage_total': len(jobs), 'trials_completed': all_trial,
             'additional_seeds_completed': all_repeat, 'test_evaluations_completed': all_test,
             'active_run': active_runs[0] if active_runs else None, 'active_runs': active_runs,
             'updated_at': timestamp(), 'started_at': started_at}
    write(ROOT / 'progress.json', value)
    return value


def warm_graph_caches(identity):
    """Verify and copy source pairs serially before GPU workers share a cache."""
    from run_missing_topology import _copy_graph_cache
    assert_inputs(identity)
    source = HERE / 'runs' / 'reviewer_v1' / 'graph_cache'
    pairs = sorted(source.glob('*.npz'))
    if not pairs:
        raise ValueError('Established read-only graph cache is unavailable.')
    records = []
    for base in ('trials', 'selected_repeats'):
        destination = ROOT / base / 'graph_cache'
        for pair in pairs:
            provenance = _copy_graph_cache(source, destination, pair.stem)
            if not provenance:
                raise ValueError('Graph cache warm-up did not verify a source pair.')
            records.append({'graph_hash': pair.stem, 'destination': str(destination),
                            'copied_artifacts': provenance})
    write(ROOT / 'graph_cache_warmup.json', {'source_modified': False,
          'verified_source_pairs': len(pairs), 'copies': records, 'verified_at': timestamp()})
    assert_inputs(identity)


def initialize_train_worker():
    # Windows spawn creates an isolated RNG and CUDA context for each worker.
    torch.set_num_threads(4)


def train_worker(job, identity):
    from run_missing_topology import RepairConfig, run_repair_fold
    assert_inputs(identity)
    config = RepairConfig(family=job['family'], learning_rate=job['learning_rate'],
                          topology_dropout=job['topology_dropout'], missing_policy=job['missing_policy'])
    completed = run_repair_fold(
        prepared=HERE / 'prepared', output=Path(job['out']), config=config,
        protocol=job['protocol'], negative=job['negative'], fold=job['fold'],
        model_seed=job['model_seed'], graph_seed=job['graph_seed'], device='cuda',
        epochs=300, patience=20, resume=True,
        source_graph_cache=HERE / 'runs' / 'reviewer_v1' / 'graph_cache', no_test=True)
    if completed.get('test_evaluated', False):
        raise ValueError('Training unexpectedly accessed test outcomes.')
    return {'out': job['out'], 'run_signature': completed['run_signature']}


def train_jobs(jobs, identity, stage, started_at, max_jobs=None, workers=2):
    if workers not in (1, 2):
        raise ValueError('Only one or two GPU workers are supported.')
    if max_jobs is not None and max_jobs < 1:
        raise ValueError('max_jobs must be positive.')
    warm_graph_caches(identity)
    bounded_jobs = jobs[:max_jobs] if max_jobs is not None else jobs
    if workers == 1:
        for job in bounded_jobs:
            update_progress(stage, jobs, active=job['out'], started_at=started_at)
            train_worker(job, identity)
            progress = update_progress(stage, jobs, started_at=started_at)
            print('CAMPAIGN_PROGRESS', json.dumps(progress, ensure_ascii=False), flush=True)
    else:
        executor = ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn'), initializer=initialize_train_worker)
        remaining = iter(bounded_jobs)
        active = {}
        def submit_next():
            job = next(remaining, None)
            if job is None:
                return False
            assert_inputs(identity)
            active[executor.submit(train_worker, job, identity)] = job
            return True
        try:
            for _ in range(workers):
                submit_next()
            while active:
                update_progress(stage, jobs, active=[j['out'] for j in active.values()], started_at=started_at)
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                # Check all returned jobs before submitting any more work.
                for future in done:
                    future.result()
                for future in done:
                    del active[future]
                    submit_next()
                progress = update_progress(stage, jobs, active=[j['out'] for j in active.values()], started_at=started_at)
                print('CAMPAIGN_PROGRESS', json.dumps(progress, ensure_ascii=False), flush=True)
        except BaseException:
            for future in active:
                future.cancel()
            update_progress(stage, jobs, status='failed', active=[j['out'] for j in active.values()], started_at=started_at)
            # Submit no more work; let already running writes finish safely.
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    if len(bounded_jobs) < len(jobs):
        update_progress(stage, jobs, status='bounded_stop', started_at=started_at)
        return False
    return True


def choose(trials, contract):
    from verify_missing_topology import verify_run
    groups = {}
    for j in trials:
        key = (j['protocol'], j['negative'], j['family'], j['fold'])
        p = Path(j['out']) / 'completed.json'
        if not p.exists():
            raise ValueError('All validation trials must finish before selection: ' + str(p))
        audit = verify_run(Path(j['out']), HERE / 'prepared')
        accepted_statuses = ('validation_only', 'test_evaluated') if (ROOT / 'selection_registry.json').exists() else ('validation_only',)
        if audit.get('status') not in accepted_statuses:
            raise ValueError('Independent validation audit failed before LR selection.')
        write(Path(j['out']) / 'validation_independent_verification.json', audit)
        result = read(p)
        if result.get('test_evaluated', False) or (Path(j['out']) / 'test_completed.json').exists():
            # An unchanged registry on resume is allowed; new selection after tests is not.
            if not (ROOT / 'selection_registry.json').exists():
                raise ValueError('Cannot make a new selection after test evaluation.')
        groups.setdefault(key, []).append((float(result['best_validation_bce']), j['learning_rate'], j, result))
    choices = []
    for key, candidates in sorted(groups.items()):
        score, lr, job, completed = min(candidates, key=lambda x: (x[0], x[1]))
        choices.append({'job': job, 'best_validation_bce': score,
                        'run_signature': completed['run_signature'],
                        'checkpoint_sha256': completed['checkpoint_sha256'],
                        'candidates': [{'out': c[2]['out'], 'learning_rate': c[1],
                                        'best_validation_bce': c[0],
                                        'checkpoint_sha256': c[3]['checkpoint_sha256']} for c in candidates]})
    stable = {'rule': contract['validation_grid']['selection'], 'choices': choices,
              'candidate_count': len(trials), 'choice_count': len(choices),
              'contract_sha256': sha(ROOT / 'campaign_contract.json')}
    p = ROOT / 'selection_registry.json'
    if p.exists():
        previous = read(p)
        if previous['selection_signature'] != digest(stable):
            raise ValueError('Frozen selection changed; refuse adaptive re-selection.')
        return previous
    registry = {**stable, 'selection_signature': digest(stable), 'frozen_at': timestamp(),
                'test_outcomes_used': False}
    write(p, registry)
    return registry


def repeat_jobs(registry):
    jobs = []
    for choice in registry['choices']:
        for repeat in (1, 2):
            j = dict(choice['job']); j['repeat'] = repeat
            j['model_seed'] = 20260930 + 1000 * repeat + j['fold']
            name = Path(j['out']).name.replace('__repeat0__', f'__repeat{repeat}__')
            j['out'] = str(ROOT / 'selected_repeats' / name)
            jobs.append(j)
    return jobs


def freeze_selected(registry, repeats):
    selected = [dict(c['job']) for c in registry['choices']] + repeats
    for j in selected:
        out = Path(j['out']); r = read(out / 'completed.json')
        freeze = {'selected_before_test': True, 'run_signature': r['run_signature'],
                  'checkpoint_sha256': r['checkpoint_sha256'],
                  'selection_rule': registry['rule'] if j['repeat'] == 0 else 'Repeat-0 validation-selected learning rate; no retuning.',
                  'selection_registry_sha256': sha(ROOT / 'selection_registry.json'),
                  'selection_signature': registry['selection_signature'],
                  'repeat': j['repeat'], 'learning_rate': j['learning_rate'],
                  'test_status': 'Previously inspected benchmark; adaptive comparison, not fresh blind test.'}
        p = out / 'selection_frozen.json'
        if p.exists() and normalized(read(p)) != normalized(freeze):
            raise ValueError('Selected model identity changed.')
        if not p.exists():
            if (out / 'test_completed.json').exists():
                raise ValueError('Test artifact predates its freeze gate.')
            write(p, freeze)
    index = {'selection_signature': registry['selection_signature'], 'runs': selected,
             'selected_run_count': len(selected), 'expected_count': 225,
             'all_training_complete_before_evaluation': True}
    if len(selected) != 225:
        raise ValueError('Selected count mismatch.')
    write(ROOT / 'selected_index.json', index)
    return selected


def evaluate_jobs(selected, identity, started_at):
    from run_missing_topology import evaluate_selected_run
    for j in selected:
        assert_inputs(identity)
        update_progress('evaluation', selected, active=j['out'], started_at=started_at)
        evaluate_selected_run(Path(j['out']), HERE / 'prepared', device='cuda')
    update_progress('evaluation', selected, status='evaluated_not_yet_verified', started_at=started_at)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('pilot', 'grid', 'selected', 'evaluate', 'finish', 'all'), default='all')
    ap.add_argument('--max-jobs', type=int)
    ap.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Established CUDA hardware is required; no silent CPU fallback.')
    torch.set_num_threads(4)
    contract, identity = setup(); started_at = timestamp()
    trials = build_jobs(contract); write(ROOT / 'trial_index.json', {'runs': trials, 'count': len(trials)})
    if args.stage == 'pilot':
        pilot = [j for j in trials if j['protocol'] == 'protein' and j['fold'] == 0
                 and j['learning_rate'] == .0005 and j['family'] in ('residual', 'topology_dropout', 'topology_dropout_bypass')]
        train_jobs(pilot, identity, 'pilot', started_at, args.max_jobs, args.workers)
        update_progress('pilot', pilot, status='pilot_training_complete_not_tested', started_at=started_at)
        return
    if args.stage in ('grid', 'all'):
        if not train_jobs(trials, identity, 'validation_grid', started_at, args.max_jobs, args.workers):
            return
        if args.stage == 'grid':
            choose(trials, contract); return
    registry = choose(trials, contract); repeats = repeat_jobs(registry)
    if args.stage in ('selected', 'all'):
        if not train_jobs(repeats, identity, 'selected_additional_seeds', started_at, args.max_jobs, args.workers):
            return
        if args.stage == 'selected':
            freeze_selected(registry, repeats); return
    selected = freeze_selected(registry, repeats)
    if args.stage in ('evaluate', 'all'):
        evaluate_jobs(selected, identity, started_at)
        if args.stage == 'evaluate':
            return
    if args.stage in ('finish', 'all'):
        # The independent verifier and collector are imported only after all
        # training and frozen-model test evaluation have completed.
        from verify_missing_topology import verify_campaign
        from collect_missing_topology import collect_results
        RESULTS.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS / 'verification.json'
        report = verify_campaign(ROOT, HERE / 'prepared')
        write(report_path, report)
        if report.get('status') != 'passed':
            raise ValueError('Independent numerical/provenance verification failed.')
        collect_results(ROOT, RESULTS, report_path)
        assert_inputs(identity)
        update_progress('complete', selected, status='completed_and_verified', started_at=started_at)
        write(ROOT / 'campaign_completed.json', {'status': 'completed_and_verified',
              'training_runs': len(trials) + len(repeats), 'test_evaluations': len(selected),
              'verification': str(report_path), 'protected_originals_unchanged': True,
              'completed_at': timestamp(), 'selection_signature': registry['selection_signature']})
    print('CAMPAIGN_STAGE_FINISHED', args.stage, flush=True)


if __name__ == '__main__':
    main()
