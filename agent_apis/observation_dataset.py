"""Privileged replay/audit boundary for schema-3 observation-only data views.

python -m agent_apis.observation_dataset audit [--limit-games N]
The default audit covers every recorded action in both corpora.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color, SimplePlayer
from catanatron.gym.envs.action_space import to_action_space
from catanatron.serialization import (action_from_json, action_record_to_json,
                                     action_to_json, client_view, state_from_json,
                                     state_to_json)

from agent_apis.dataset_bundle import decision_action
from agent_apis.dataset import action_vocabulary
from agent_apis.experiment import (ROOT, EXPERIMENTS, corpus_snapshot, digest,
                                   file_digest, source_snapshot, stream_seeds,
                                   validate_record)
from agent_apis.observation import (COLORS, SCHEMA_VERSION, ResourceLedger,
                                   base_names, development_knowledge,
                                   encode_observation, feature_names,
                                   from_web_view, require, scales)


def permitted_view(document, color, legal):
    """Audit/live adapter: redact BEFORE passing anything to the encoder."""
    view = client_view(document, Color[color])
    view['current_playable_actions'] = legal
    view['state_index'] = len(view['action_records'])
    return from_web_view(view, color)


def _load(path):
    with gzip.open(path, 'rt') as stream:
        header = json.loads(next(stream))
        require(header['schema_version'] == 2, 'Only schema-2 corpora are supported')
        rows = [json.loads(line) for line in stream]
    retained = {r['step']: r for r in rows}
    require(len(retained) == len(rows) == header['saved_steps'], 'Duplicate/missing retained rows')
    initial = header['initial_state']
    game = state_from_json(initial, [SimplePlayer(Color[c]) for c in initial['colors']])
    return header, retained, game


def _frames(path):
    """Internal privileged layer. Consumers must not export game/document."""
    header, retained, game = _load(path)
    records = header['final_state']['action_records']
    require(len(records) == header['steps'], 'Incomplete recorded action log')
    for step, recorded in enumerate(records):
        legal = [action_to_json(a) for a in game.playable_actions]
        document = state_to_json(game)
        actor = game.state.current_color().value
        action = action_from_json(decision_action(recorded))
        legal_ids = sorted({to_action_space(a, (Color.BLUE, Color.RED), 'BASE')
                            for a in game.playable_actions})
        action_id = to_action_space(action, (Color.BLUE, Color.RED), 'BASE')
        require(action_id in legal_ids and action in game.playable_actions,
                f'Illegal demonstration at {path}:{step}')
        view = permitted_view(document, actor, legal)
        features = encode_observation(view)
        # Outcomes are attached only AFTER features have been encoded.
        row = {'game_id': game.id, 'step': step, 'num_turns': game.state.num_turns,
               'actor': actor, 'features': features, 'action_id': action_id,
               'legal_action_ids': legal_ids,
               'value_target': int(actor == header['winner']),
               'terminated': step == len(records)-1, 'truncated': False,
               'development_knowledge': development_knowledge(view)['events']}
        yield game, document, view, row, retained.get(step)
        result = game.execute(action)
        require(action_record_to_json(result) == recorded,
                f'Chance/replay mismatch at {path}:{step}')
    require(state_to_json(game) == header['final_state'], f'Final replay mismatch at {path}')


def iter_observed_samples(directory, split=None, view='complete'):
    """Stream schema 3 in retained or complete-decision view, without hidden state.

    Replays omitted decisions; retained is never treated as consecutive states.
    No importance multiplier is applied. Caller can explicitly normalize via
    normalized_features. Game order here is stable; training must shuffle games
    and use a bounded state shuffle buffer. Outcomes are [0,1].
    """
    require(view in ('retained', 'complete'), 'Unknown data view')
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for entry in manifest['games']:
        if split is not None and entry['split'] != split:
            continue
        path = directory / entry['file']
        require(file_digest(path) == entry['sha256'], f'Checksum mismatch: {path}')
        for _, _, _, row, saved in _frames(path):
            if view == 'complete' or saved is not None:
                yield row | {'split': entry['split'], 'source_sha256': entry['sha256'],
                             'feature_schema_version': SCHEMA_VERSION}


def phase(view):
    if view['is_initial_build_phase']:
        return 'setup'
    if max(view['player_state'][f'P{i}_VICTORY_POINTS'] for i in range(2)) >= 12:
        return 'endgame'
    return 'middle'


def _check_ledger(ledger, view, document):
    ledger.update(view)
    # Authoritative hands are read here ONLY to audit an already derived ledger.
    for color in COLORS:
        seat = document['colors'].index(color)
        expected = [document['player_state'][f'P{seat}_{r}_IN_HAND'] for r in RESOURCES]
        require(ledger.hands[color] == expected, f'Ledger mismatch for {color}')
    recovered = ResourceLedger(view['perspective'])
    recovered.update(view)
    require(recovered.hands == ledger.hands and recovered.bank == ledger.bank,
            'Restart ledger mismatch')


def audit_game(path):
    counts = {'complete': Counter(), 'retained': Counter()}
    ledgers = {c: ResourceLedger(c) for c in COLORS}
    checked = retained_count = 0
    last_game = None
    vocabulary = action_vocabulary()
    for game, document, view, row, saved in _frames(path):
        last_game = game
        for color in COLORS:
            seat_view = view if color == row['actor'] else permitted_view(
                document, color, view['current_playable_actions'])
            _check_ledger(ledgers[color], seat_view, document)
            checked += 1
        encoded = encode_observation(view, ledgers[row['actor']])
        require(encoded == row['features'], 'Continuous/live encoder differs from restart encoder')
        # Use the common vocabulary for the actual demonstration family.
        family = vocabulary[row['action_id']][0]
        tags = ['decisions', 'phase:' + phase(view), 'family:' + family,
                'prompt:' + view['current_prompt']]
        counts['complete'].update(tags)
        if saved is not None:
            retained_count += 1
            counts['retained'].update(tags)
            require(encoded[:633] == saved['features'], 'Original 633 features differ')
            require(row['legal_action_ids'] == saved['legal_action_ids'], 'Saved legal mask differs')
            require(row['action_id'] == saved['action_id'], 'Saved target differs')
            require(row['actor'] == saved['actor'] and row['step'] == saved['step']
                    and row['num_turns'] == saved['num_turns'], 'Saved decision metadata differs')
            require(row['value_target'] == (saved['value_target'] + 1)/2,
                    'Saved outcome label differs')
            # Reconstruct omitted static/history context using ONLY pre-action data.
            restored = dict(saved['observation'])
            restored.update(map=view['map'], game=view['game'],
                            action_records=view['action_records'],
                            state_index=view['state_index'],
                            current_playable_actions=view['current_playable_actions'])
            restored = from_web_view(restored, row['actor'])
            require(encode_observation(restored) == encoded, 'Live/retained replay encoder differs')
            # Check every stored observation field, including fields not in features.
            redacted = client_view(document, Color[row['actor']])
            for key, value in saved['observation'].items():
                if key in ('development_listdeck', 'perspective'):
                    expected = view[key]
                else:
                    expected = redacted[key]
                require(value == expected, f'Saved observation field differs: {key}')
    require(last_game is not None, 'Empty game')
    # The generator has executed the final action and verified the exact final state.
    final = state_to_json(last_game)
    for color in COLORS:
        _check_ledger(ledgers[color], permitted_view(final, color, []), final)
        checked += 1
    return {'counts': {k: dict(v) for k, v in counts.items()},
            'ledger_snapshots': checked, 'restart_checks': checked,
            'retained_encoder_checks': retained_count}


def audit(config_path, output, limit_games=None):
    start = time.perf_counter()
    source = source_snapshot()
    config = json.loads(config_path.read_text())
    require(config['data']['model_schema'] == SCHEMA_VERSION, 'Expected model schema 3')
    corpora = corpus_snapshot(config)
    require(list(base_names()) == corpora[0]['feature_names'],
            'Frozen schema base names disagree with corpus manifests')
    seen_ids, seen_seeds = set(), set()
    for corpus in corpora:
        for game in corpus['games']:
            require(game['game_id'] not in seen_ids, 'Overlapping game IDs across corpus/splits')
            require(game['seed'] not in seen_seeds, 'Overlapping seeds across corpus/splits')
            seen_ids.add(game['game_id'])
            seen_seeds.add(game['seed'])
            bucket = int(hashlib.sha256(str(game['seed']).encode()).hexdigest()[:8], 16) % 100
            require(game['split'] == ('train' if bucket < 90 else 'validation' if bucket < 95 else 'test'),
                    'Split assignment differs from fixed seed hash')
    schema = json.loads((EXPERIMENTS / 'feature-schema-v3.json').read_text())
    require(schema['feature_names'] == list(feature_names()) and schema['scales'] == scales(),
            'Versioned schema differs from implementation')
    totals, failures = {}, []
    for corpus in corpora:
        by_split = {}
        selected = corpus['games'][:limit_games] if limit_games else corpus['games']
        for index, entry in enumerate(selected):
            try:
                result = audit_game(ROOT / corpus['path'] / entry['file'])
                split = by_split.setdefault(entry['split'], {'games': 0, 'complete': Counter(),
                    'retained': Counter(), 'ledger_snapshots': 0, 'restart_checks': 0,
                    'retained_encoder_checks': 0})
                split['games'] += 1
                for name in ('complete', 'retained'):
                    split[name].update(result['counts'][name])
                for name in ('ledger_snapshots', 'restart_checks', 'retained_encoder_checks'):
                    split[name] += result[name]
            except Exception as error:
                failures.append({'corpus': corpus['path'], 'file': entry['file'],
                                 'error': f'{type(error).__name__}: {error}'})
                print(f'FAILED {entry["file"]}: {error}', flush=True)
            if (index+1) % 25 == 0 or index+1 == len(selected):
                print(f'{corpus["path"]}: {index+1}/{len(selected)}, failures={len(failures)}', flush=True)
        totals[corpus['path']] = by_split
    seed = config['randomness']['screening_seed']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f'phase2-audit-{digest(config)[:12]}-seed{seed}-{stamp}'
    record = {'schema_version': 1, 'run_id': run_id,
              'status': 'failed' if failures else 'completed',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'configuration': {'content': config, 'sha256': digest(config)},
              'seed': seed, 'rng_seeds': stream_seeds(seed), 'source': source,
              'datasets': corpora, 'checkpoint': None,
              'software': {'python': sys.version, 'executable': sys.executable,
                           'packages': dict(sorted((d.metadata['Name'], d.version)
                                                  for d in importlib.metadata.distributions()))},
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                           'available_devices': ['cpu']},
              'metrics': {'audit': {'scope': 'partial' if limit_games else 'all',
                          'gate_passed': not failures and limit_games is None,
                          'cross_split_game_overlap': 0, 'cross_split_seed_overlap': 0,
                          'feature_schema_sha256': file_digest(EXPERIMENTS / 'feature-schema-v3.json'),
                          'counts': totals, 'failures': failures}},
              'elapsed_seconds': time.perf_counter() - start}
    validate_record(record)
    folder = output / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(folder / 'run.json', flush=True)
    require(not failures, 'Audit failed; see run record. Training remains blocked.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('audit')
    p.add_argument('--config', type=Path, default=EXPERIMENTS / 'phase2.json')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/catan-ai')
    p.add_argument('--limit-games', type=int)
    args = parser.parse_args()
    require(args.limit_games is None or args.limit_games > 0, 'Limit must be positive')
    audit(args.config, args.output, args.limit_games)


if __name__ == '__main__':
    main()
