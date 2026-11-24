"""Build disjoint full/sampled corpora with exact public dice identities."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path

from agent_apis.dataset import (_generate_game, encode, get_feature_ordering,
                               MAX_BYTES, action_vocabulary)


def exact_numbers(public_map):
    tiles = {t['id']: t for t in public_map['tiles'] if t['type'] == 'LAND'}
    return [tiles[i]['number'] or 0 for i in sorted(tiles)]


def create_numbered_sample(game, color):
    """Inference/training encoder matching schema 2, using public map numbers."""
    from catanatron.features import create_sample_vector
    from catanatron.serialization import map_to_json
    return create_sample_vector(game, color) + exact_numbers(map_to_json(game.state.board.map))


def retained_steps(rows):
    """Every tenth action per actor, all setup, and last ten per actor."""
    by_actor = {}
    keep = set()
    for row in rows:
        actor_rows = by_actor.setdefault(row['actor'], [])
        if len(actor_rows) % 10 == 0 or row['observation']['is_initial_build_phase']:
            keep.add(row['step'])
        actor_rows.append(row['step'])
    for steps in by_actor.values():
        keep.update(steps[-10:])
    return keep


def convert(payload, sampled):
    lines = gzip.decompress(payload).splitlines()
    header = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]
    numbers = exact_numbers(header['initial_state']['map'])
    keep = retained_steps(rows) if sampled else {r['step'] for r in rows}
    header.update(schema_version=2, sampling='per_actor_1_in_10_setup_tail10' if sampled else 'full',
                  saved_steps=len(keep), tile_numbers=numbers)
    selected = []
    for row in rows:
        if row['step'] in keep:
            row['features'].extend(numbers)
            selected.append(row)
    raw = encode(header) + b''.join(encode(r) for r in selected)
    return gzip.compress(raw, mtime=0), len(raw), header, selected


def decision_action(record):
    action = list(record[0])
    if action[1] in ("ROLL", "BUY_DEVELOPMENT_CARD"):
        action[2] = None
    return action


def replay_saved(path):
    """Replay unsaved actions too; yield only retained pre-action states."""
    from catanatron.models.player import SimplePlayer, Color
    from catanatron.serialization import state_from_json, action_from_json, action_record_to_json, state_to_json
    with gzip.open(path, 'rt') as stream:
        header = json.loads(next(stream))
        initial = header['initial_state']
        game = state_from_json(initial, [SimplePlayer(Color[c]) for c in initial['colors']])
        records = header['final_state']['action_records']
        cursor = 0
        for line in stream:
            row = json.loads(line)
            while cursor < row['step']:
                result = game.execute(action_from_json(decision_action(records[cursor])))
                assert action_record_to_json(result) == records[cursor]
                cursor += 1
            yield game, row
        while cursor < len(records):
            result = game.execute(action_from_json(decision_action(records[cursor])))
            assert action_record_to_json(result) == records[cursor]
            cursor += 1
        assert state_to_json(game) == header['final_state']


def build(source, output, full_games=250, sampled_games=10000, workers=4, max_bytes=MAX_BYTES):
    if full_games != 250 or not 1 <= sampled_games <= 10000 or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError('Require 250 full games, 1..10000 sampled games and <=1 GB')
    source, output = Path(source), Path(output)
    resume = (output/'full-250'/'manifest.json').exists()
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Output must be empty or a resumable bundle')
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source/'manifest.json').read_text())
    manifests = {}
    for name in ('full-250', 'downsampled'):
        (output/name).mkdir(exist_ok=True)
        if resume:
            manifests[name]=json.loads((output/name/'manifest.json').read_text())
            continue
        m = {k:v for k,v in original.items() if k not in ('games','failures','num_games','num_samples','shard_bytes','uncompressed_shard_bytes')}
        m.update(schema_version=2, feature_names=get_feature_ordering(2,'BASE') + [f'TILE{i}_DICE_NUMBER' for i in range(19)],
                 games=[], failures=[], num_games=0, num_samples=0, stop_reason='running')
        m['limits'] = dict(m['limits'], games=250 if name=='full-250' else sampled_games,
                           bytes=max_bytes, byte_basis='combined stored AND uncompressed JSON')
        m['sampling'] = 'full' if name=='full-250' else 'every tenth action PER ACTOR (ordinal 0,10,...), all setup, last ten per actor'
        manifests[name] = m
    stored = sum(e['bytes'] for m in manifests.values() for e in m['games'])
    raw_total = sum(e['uncompressed_bytes'] for m in manifests.values() for e in m['games'])
    existing_ids = {e['game_id'] for m in manifests.values() for e in m['games']}
    def save():
        for name,m in manifests.items():
            m['num_games']=len(m['games'])
            m['num_samples']=sum(g['saved_steps'] for g in m['games'])
            (output/name/'manifest.json').write_bytes(encode(m))
    def accept(payload, entry):
        nonlocal stored, raw_total
        name = 'full-250' if len(manifests['full-250']['games']) < full_games else 'downsampled'
        packed, raw_size, header, rows = convert(payload, name=='downsampled')
        if max(stored+len(packed), raw_total+raw_size) > max_bytes-4_000_000:
            return False
        entry = {k:v for k,v in entry.items() if k!='payload'}
        entry.update(bytes=len(packed), uncompressed_bytes=raw_size, sha256=hashlib.sha256(packed).hexdigest(),
                     saved_steps=len(rows), action_counts=dict(Counter(r['action'][1] for r in rows)))
        (output/name/entry['file']).write_bytes(packed)
        manifests[name]['games'].append(entry)
        stored += len(packed)
        raw_total += raw_size
        save()
        count=sum(len(m['games']) for m in manifests.values())
        if count % 25 == 0:
            print(f'{count} total games; {len(manifests["downsampled"]["games"])} sampled games; {raw_total/1e6:.1f} MB raw',flush=True)
        return True
    exhausted=False
    for entry in original['games']:
        if len(manifests['full-250']['games']) == full_games and len(manifests['downsampled']['games']) >= sampled_games:
            break
        if entry['game_id'] in existing_ids:
            continue
        if not accept((source/entry['file']).read_bytes(),entry):
            exhausted=True
            break
    next_index=max(g['index'] for g in original['games'] + [e for m in manifests.values() for e in m['games']])+1
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        pending=set()
        while not exhausted and (pending or len(manifests['downsampled']['games']) < sampled_games):
            while (len(pending)<workers and
                   len(manifests['downsampled']['games'])+len(pending)<sampled_games):
                pending.add(pool.submit(_generate_game,next_index,original['base_seed']+next_index,1000,10000,MAX_BYTES))
                next_index+=1
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for task in done:
                result=task.result()
                if 'error' in result:
                    manifests['downsampled']['failures'].append(result)
                    save()
                    print('Skipped',result,flush=True)
                elif not exhausted and not accept(result['payload'],result):
                    exhausted=True
            if exhausted:
                for task in pending: task.cancel()
                break
    if len(manifests['full-250']['games']) != 250:
        raise RuntimeError('Budget could not fit 250 full games')
    manifests['full-250']['stop_reason']='game_limit'
    manifests['downsampled']['stop_reason']='byte_limit' if exhausted else 'game_limit'
    save()
    summary={'full_games':250,'sampled_games':len(manifests['downsampled']['games']),
             'stored_bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
             'uncompressed_bytes':raw_total+sum((output/n/'manifest.json').stat().st_size for n in manifests),
             'max_bytes':max_bytes}
    assert summary['stored_bytes']<max_bytes and summary['uncompressed_bytes']<max_bytes
    summary_size=len(encode(summary))
    summary['stored_bytes']+=summary_size
    summary['uncompressed_bytes']+=summary_size
    assert max(summary['stored_bytes'],summary['uncompressed_bytes'])<=max_bytes
    (output/'summary.json').write_bytes(encode(summary))
    print(json.dumps(summary),flush=True)
    return summary


def verify_bundle(output):
    from catanatron.features import create_sample_vector
    from agent_apis.dataset import observation, COLORS, to_action_space, action_from_json
    output=Path(output)
    ids=set()
    seeds=set()
    total_raw=0
    report={}
    for name in ('full-250','downsampled'):
        directory=output/name
        m=json.loads((directory/'manifest.json').read_text())
        total_raw+=(directory/'manifest.json').stat().st_size
        assert m['feature_names']==get_feature_ordering(2,'BASE')+[f'TILE{i}_DICE_NUMBER' for i in range(19)]
        count=0
        for e in m['games']:
            assert e['game_id'] not in ids
            ids.add(e['game_id'])
            assert e['seed'] not in seeds
            seeds.add(e['seed'])
            payload=(directory/e['file']).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==e['sha256']
            raw=gzip.decompress(payload)
            assert len(raw)==e['uncompressed_bytes']
            total_raw+=len(raw)
            lines=raw.splitlines()
            h=json.loads(lines[0]); rows=[json.loads(l) for l in lines[1:]]
            assert h['winner']==e['winner']
            numbers=exact_numbers(h['initial_state']['map'])
            assert h['tile_numbers']==numbers
            # Determine required samples independently from full action history.
            records=h['final_state']['action_records']
            by_actor={c:[] for c in h['initial_state']['colors']}
            required=set(range(8))
            for i,(a,_) in enumerate(records):
                if len(by_actor[a[0]])%10==0: required.add(i)
                by_actor[a[0]].append(i)
            for steps in by_actor.values(): required.update(steps[-10:])
            assert {r['step'] for r in rows} == (required if name=='downsampled' else set(range(len(records))))
            for game,row in replay_saved(directory/e['file']):
                assert row['features']==create_sample_vector(game,game.state.current_color())+numbers
                assert len(row['features'])==633
                assert row['observation']==observation(game)
                assert row['num_turns']==game.state.num_turns
                assert row['actor']==game.state.current_color().value
                assert row['record']==records[row['step']]
                assert row['action']==decision_action(row['record'])
                assert row['action_id']==to_action_space(action_from_json(row['action']),COLORS,'BASE')
                assert row['legal_action_ids']==sorted({to_action_space(a,COLORS,'BASE') for a in game.playable_actions})
                assert row['value_target']==(1 if row['actor']==e['winner'] else -1)
                terminal=row['step']==len(records)-1
                assert row['terminated']==terminal and not row['truncated']
                assert row['reward']==(row['value_target'] if terminal else 0)
                expected_actor=records[row['step']+1][0][0] if not terminal else h['final_state']['colors'][h['final_state']['current_player_index']]
                assert row['next_actor']==expected_actor
                count+=1
            assert game.winning_color().value==e['winner']
            assert game.state.num_turns==e['num_turns']
            assert dict(Counter(r['action'][1] for r in rows))==e['action_counts']
            assert len(rows)==e['saved_steps']==h['saved_steps']
        assert count==m['num_samples']
        report[name]={'games':len(m['games']),'samples':count}
    assert report['full-250']['games']==250 and report['downsampled']['games']<=10000
    total_stored=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    total_raw+=(output/'summary.json').stat().st_size
    assert max(total_raw,total_stored)<=MAX_BYTES
    report.update(stored_bytes=total_stored,uncompressed_bytes=total_raw,verified=True)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','verify'])
    parser.add_argument('--source',default='data/jsettlers-1v1')
    parser.add_argument('--output',default='data/jsettlers-corpora')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sampled-games',type=int,default=10000)
    args=parser.parse_args()
    if args.command=='verify': verify_bundle(args.output)
    else: build(args.source,args.output,sampled_games=args.sampled_games,workers=args.workers)
