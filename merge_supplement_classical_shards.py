"""Merge finished independent supplement workers without changing run contents.

Call only after the worker process exited successfully. Directory rename is atomic
on the same volume; target existence is always an error, never an overwrite.
"""
import argparse
import json
from pathlib import Path

import run_supplement_classical as supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard', type=Path)
    args = parser.parse_args()
    root = (supplement.HERE/'runs'/'supplement_20261001').resolve()
    shard = args.shard.resolve()
    destination = (root/'classical').resolve()
    if not shard.is_relative_to(root/'classical_shards') or not destination.is_relative_to(root):
        raise ValueError('Merge paths outside the explicitly named supplement workspace')
    verification = json.loads((shard/'verification.json').read_text(encoding='utf-8'))
    if verification['status'] != 'passed':
        raise ValueError('Worker did not finish verification')
    moves = []
    for name, expected in verification['completed_sha256'].items():
        source = (shard/name).resolve()
        target = (destination/name).resolve()
        if not source.is_relative_to(shard) or not target.is_relative_to(destination):
            raise ValueError('Run paths escaped merge roots')
        if target.exists():
            raise FileExistsError('Refusing to merge over an existing run: '+str(target))
        if supplement.base.file_hash(source/'completed.json') != expected:
            raise ValueError('Worker record changed after verification')
        record = json.loads((source/'completed.json').read_text(encoding='utf-8'))
        data = supplement.load_fold(supplement.HERE/'prepared','random',record['protocol'],record['fold'])
        supplement.verify_completed(source,data)
        moves.append((source,target,expected))
    for source,target,expected in moves:
        if target.exists():
            raise FileExistsError('Target appeared during merge; refusing overwrite')
        source.rename(target)
    supplement.base.write_json(shard/'merge_record.json',dict(status='merged',
        entries=[dict(source=str(source),destination=str(target),completed_sha256=expected)
                 for source,target,expected in moves]))
    print('MERGED',len(moves),'completed runs from',shard)


if __name__=='__main__':
    main()
