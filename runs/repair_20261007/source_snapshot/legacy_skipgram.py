"""Train the original gensim 3.1 Word2Vec implementation on supplied walks.

Run with D:\\Anaconda\\envs\\tf\\python.exe. Importing this module performs no
training or package imports. Only the caller-provided walk corpus is consumed;
the caller must construct walks exclusively from the fold's training graph.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walks', type=Path, required=True,
                        help='UTF-8 JSON list of lists of zero-based integer node IDs.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output .npz; a sibling .json records provenance.')
    parser.add_argument('--n-nodes', type=int, required=True)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--negative', type=int, default=5)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args(argv)
    for name in ['n_nodes', 'dim', 'window', 'epochs', 'negative']:
        if getattr(args, name) <= 0:
            parser.error('--{} must be positive'.format(name.replace('_', '-')))
    if not 0 <= args.seed <= 4294967295:
        parser.error('--seed must be in [0, 4294967295]')
    if args.output.suffix.lower() != '.npz':
        parser.error('--output must end in .npz')
    return args


def read_walks(path, n_nodes):
    raw = path.read_bytes()
    walks = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(walks, list):
        raise ValueError('Walk corpus must be a JSON list of lists')
    active = set()
    for walk_index, walk in enumerate(walks):
        if not isinstance(walk, list):
            raise ValueError('Walk {} is not a list'.format(walk_index))
        for offset, node in enumerate(walk):
            if type(node) is not int or not 0 <= node < n_nodes:
                raise ValueError('Invalid node at walk {} offset {}: {!r}; '
                                 'expected an integer in [0, {})'.format(
                                     walk_index, offset, node, n_nodes))
            active.add(node)
    return walks, sorted(active), hashlib.sha256(raw).hexdigest()


def train_and_save(args):
    import gensim
    import numpy as np
    from gensim.models import Word2Vec

    if not gensim.__version__.startswith('3.1.'):
        raise RuntimeError('Expected gensim 3.1.x, found {} in {}'.format(
            gensim.__version__, sys.executable))
    output = args.output.resolve()
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Refusing to overwrite an existing embedding or '
                              'provenance file: {}'.format(output))
    walks, active, corpus_hash = read_walks(args.walks, args.n_nodes)
    sentences = [[str(node) for node in walk] for walk in walks if walk]
    config = {
        'size': args.dim, 'window': args.window, 'min_count': 0,
        'sg': 1, 'workers': 1, 'seed': args.seed, 'iter': args.epochs,
        'negative': args.negative, 'hs': 0, 'alpha': 0.025,
        'min_alpha': 0.0001, 'sample': 0.001, 'max_vocab_size': None,
        'cbow_mean': 1, 'null_word': 0, 'trim_rule': None,
        'sorted_vocab': 1, 'batch_words': 10000, 'compute_loss': False,
    }
    embedding = np.zeros((args.n_nodes, args.dim), dtype=np.float32)
    present = np.zeros(args.n_nodes, dtype=np.bool_)
    status = 'empty_corpus_zero_embeddings'
    if sentences:
        # Preserve gensim's default built-in hash, dynamic window, subsampling,
        # negative distribution, and learning-rate scheduling. main() fixes the
        # interpreter hash seed before this function imports gensim.
        model = Word2Vec(sentences=sentences, hashfxn=hash, **config)
        for node in active:
            embedding[node] = model.wv[str(node)]
            present[node] = True
        status = 'gensim_training_completed'
    if not np.isfinite(embedding).all():
        raise FloatingPointError('Word2Vec produced non-finite embeddings')
    if np.any(embedding[~present] != 0):
        raise AssertionError('Nodes absent from walks must have zero embeddings')
    metadata = {
        'implementation': 'gensim.models.Word2Vec',
        'gensim_version': gensim.__version__, 'numpy_version': np.__version__,
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'word2vec_config': dict(config, hashfxn='builtins.hash'),
        'walks_file': str(args.walks.resolve()), 'walks_sha256': corpus_hash,
        'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'n_nodes': args.n_nodes, 'n_walks': len(walks),
        'n_nonempty_walks': len(sentences),
        'n_tokens': sum(len(walk) for walk in walks),
        'n_present_nodes': len(active), 'embedding_shape': list(embedding.shape),
        'embedding_dtype': str(embedding.dtype), 'present_dtype': str(present.dtype),
        'status': status,
        'source_contract': 'Caller must supply walks from training positive edges only; '
                           'this helper receives no validation/test pairs or labels.',
        'absent_node_policy': 'All-zero embeddings; present=False.',
        'present_semantics': 'True iff the node occurs in the supplied walk corpus; '
                             'this does not assert that its vector received an update.',
        'output_npz': str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(output.parent), suffix='.npz',
                                     delete=False) as handle:
        temp_npz = Path(handle.name)
    try:
        np.savez_compressed(str(temp_npz), embedding=embedding, present=present)
        temp_npz.replace(output)
    finally:
        if temp_npz.exists():
            temp_npz.unlink()
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                       encoding='utf-8')
    print(json.dumps({'output': str(output), 'metadata': str(sidecar),
                      'status': status, 'n_present_nodes': len(active)},
                     ensure_ascii=True))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    # Changing PYTHONHASHSEED inside a running interpreter is ineffective.
    # Re-execute before importing gensim to retain its original hash function
    # while making seeded, single-worker runs reproducible across processes.
    if os.environ.get('PYTHONHASHSEED') != str(args.seed):
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(args.seed)
        return subprocess.call([sys.executable, str(Path(__file__).resolve())] + argv,
                               env=env)
    return train_and_save(args)


if __name__ == '__main__':
    sys.exit(main())
