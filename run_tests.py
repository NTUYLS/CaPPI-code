"""Run synthetic contract tests with the verified Windows import order.

PyTorch must be imported before scikit-learn/SciPy in this Windows environment.
This entrypoint changes no installed packages or scientific experiment code.
Use --cpu-only to keep all tests off the GPU while another experiment is active.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import unittest


HERE = Path(__file__).resolve().parent


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.starts = {}

    def startTest(self, test):
        self.starts[test.id()] = time.monotonic()
        super().startTest(test)

    def record(self, test, status, reason=None):
        entry = {'test': test.id(), 'status': status,
                 'elapsed_seconds': time.monotonic() - self.starts.get(test.id(), time.monotonic())}
        if reason is not None:
            entry['reason'] = reason
        self.records.append(entry)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, 'passed')

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.record(test, 'failed', self._exc_info_to_string(error, test))

    def addError(self, test, error):
        super().addError(test, error)
        self.record(test, 'error', self._exc_info_to_string(error, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, 'skipped', reason)


def import_order_diagnostics():
    results = []
    for order in [('torch', 'numpy', 'scipy', 'sklearn', 'joblib'),
                  ('numpy', 'scipy', 'sklearn', 'joblib', 'torch')]:
        code = ('import importlib,json; order=' + repr(list(order)) + '; '
                '[(importlib.import_module(name),print(name,flush=True)) for name in order]; '
                'print(json.dumps({"ok":True,"order":order}))')
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        run = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             timeout=60, env=environment)
        results.append({'order': list(order), 'returncode': run.returncode,
                        'stdout': run.stdout.decode('utf-8', errors='replace'),
                        'stderr': run.stderr.decode('utf-8', errors='replace')})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--pattern', default='test_*.py')
    parser.add_argument('--expected-tests', type=int)
    parser.add_argument('--diagnose-import-order', action='store_true')
    parser.add_argument('--report', type=Path, default=HERE / 'test_validation.json')
    parser.add_argument('--log', type=Path, default=HERE / 'test_validation.log')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    started = time.monotonic()
    report = {
        'started_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'python_executable': sys.executable, 'python_version': sys.version,
        'scope': 'Synthetic contract tests only; no manuscript-data experiment training.',
        'entrypoint': str(Path(__file__).resolve()),
        'entrypoint_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'import_policy': 'Import torch before unittest discovery and before NumPy/SciPy/sklearn.',
        'cpu_only_requested': args.cpu_only, 'torch_cpu_threads': args.threads,
        'test_source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((HERE / 'tests').glob(args.pattern))},
        'success': False,
    }
    with args.log.open('w', encoding='utf-8', newline='') as handle:
        tee = Tee(sys.stdout, handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                if args.diagnose_import_order:
                    report['import_order_diagnostics'] = import_order_diagnostics()
                    for item in report['import_order_diagnostics']:
                        print('IMPORT ORDER {}: exit={}'.format(' -> '.join(item['order']), item['returncode']))
                # Keep this the first third-party import in the test process.
                import torch
                torch.set_num_threads(args.threads)
                import numpy
                import scipy
                import sklearn
                report['runtime'] = {'torch': str(torch.__version__), 'numpy': numpy.__version__,
                    'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                    'cuda_available': bool(torch.cuda.is_available()), 'compiled_cuda': torch.version.cuda}
                print('TEST RUNTIME ' + json.dumps(report['runtime']))
                suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'), pattern=args.pattern)
                report['discovered_test_count'] = suite.countTestCases()
                result = unittest.TextTestRunner(stream=tee, verbosity=2,
                                                resultclass=RecordingResult).run(suite)
                count_ok = args.expected_tests is None or result.testsRun == args.expected_tests
                report.update(tests_run=result.testsRun, failures=len(result.failures),
                    errors=len(result.errors), skipped=len(result.skipped),
                    unexpected_successes=len(result.unexpectedSuccesses),
                    expected_test_count=args.expected_tests, expected_count_matched=count_ok,
                    test_records=result.records, success=result.wasSuccessful() and count_ok)
                report['failure_details'] = [{'test': test.id(), 'traceback': reason}
                                             for test, reason in result.failures + result.errors]
                if torch.cuda.is_initialized():
                    report['cuda_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
                if not count_ok:
                    print('ERROR: Expected {} tests, ran {}'.format(args.expected_tests, result.testsRun))
            except Exception:
                report['entrypoint_error'] = traceback.format_exc()
                traceback.print_exc()
            report['elapsed_seconds'] = time.monotonic() - started
            print('VALIDATION ' + json.dumps({'success': report['success'],
                'tests_run': report.get('tests_run'), 'report': str(args.report), 'log': str(args.log)},
                ensure_ascii=True))
    report['log_sha256'] = hashlib.sha256(args.log.read_bytes()).hexdigest()
    temporary = args.report.with_suffix(args.report.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(args.report)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
