"""Reserve and publish a finished independent fold without permitting duplicate fits.

Reservations make the original runner stop safely if it reaches a pending shard.
All model files are copied first and completed.json is published atomically last.
No original scientific run file or runner source is altered.
"""
import argparse
import json
from pathlib import Path
import shutil

import run_supplement_classical as supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('reserve','publish'))
    parser.add_argument('--protocol',choices=('pair','protein'),required=True)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--shard',type=Path,required=True)
    args=parser.parse_args()
    root=(supplement.HERE/'runs'/'supplement_20261001').resolve()
    shard=args.shard.resolve()
    destination=(root/'classical').resolve()
    if not shard.is_relative_to(root/'classical_shards') or args.fold not in range(5):
        raise ValueError('Shard or fold outside this task')
    names=[f'{args.protocol}__random__{method}__fold{args.fold}' for method in supplement.METHODS]
    marker=dict(owner='supplement_20261001',source=str(shard),protocol=args.protocol,fold=args.fold,
                purpose='Prevent duplicate fitting while a separate CPU worker finishes this exact frozen fold')
    if args.mode=='reserve':
        if any((destination/name).exists() for name in names):
            raise FileExistsError('A target run already exists; refusing reservation')
        for name in names:
            folder=destination/name
            folder.mkdir(exist_ok=False)
            supplement.base.write_json(folder/'shard_reservation.json',marker)
        print('RESERVED',names)
        return
    verification=json.loads((shard/'verification.json').read_text(encoding='utf-8'))
    if verification['status']!='passed' or set(verification['completed_sha256'])!=set(names):
        raise ValueError('Expected exactly this verified four-method fold')
    data=supplement.load_fold(supplement.HERE/'prepared','random',args.protocol,args.fold)
    records=[]
    for name in names:
        source=(shard/name).resolve()
        target=(destination/name).resolve()
        if not source.is_relative_to(shard) or not target.is_relative_to(destination):
            raise ValueError('Run path escaped task roots')
        if {p.name for p in target.iterdir()}!={'shard_reservation.json'}:
            raise FileExistsError('Reservation acquired run contents; refusing overwrite')
        if json.loads((target/'shard_reservation.json').read_text(encoding='utf-8'))!=marker:
            raise ValueError('Foreign reservation')
        record=supplement.verify_completed(source,data)
        if supplement.base.file_hash(source/'completed.json')!=verification['completed_sha256'][name]:
            raise ValueError('Completed source changed')
        records.append((source,target,record))
    for source,target,record in records:
        for filename in record['artifacts_sha256']:
            shutil.copy2(source/filename,target/filename)
    for source,target,record in records:
        shutil.copy2(source/'completed.json',target/'completed.pending')
        (target/'completed.pending').replace(target/'completed.json')
        # Delete only this task's checked one-file reservation; no directory removal.
        (target/'shard_reservation.json').unlink()
        supplement.verify_completed(target,data)
    supplement.base.write_json(shard/'publication_record.json',dict(status='published_without_refit',
        entries=[dict(source=str(source),destination=str(target),completed_sha256=supplement.base.file_hash(target/'completed.json'))
                 for source,target,_ in records]))
    print('PUBLISHED',len(records),'completed runs')


if __name__=='__main__':
    main()
