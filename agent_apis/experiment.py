"""Phase-one experiment provenance and device preflight (no training).

Run with the project-local Python: python -m agent_apis.experiment prepare.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments/catan-ai'
STREAMS = ('board', 'simulator', 'search', 'training', 'evaluation')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stream_seeds(seed):
    """Stable named seeds; adding a stream never changes an existing seed."""
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return {name: int.from_bytes(hashlib.sha256(
        f'catan-ai-rng-v1:{seed}:{name}'.encode()).digest()[:8], 'big')
            for name in STREAMS}


def rng_streams(seed):
    """Independent Python RNGs. Never seed or inspect the live game's RNG here."""
    return {name: random.Random(value) for name, value in stream_seeds(seed).items()}


def git(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_snapshot():
    # rg respects repository ignore rules, including local environments and data.
    result = subprocess.run(['rg', '--files', '--hidden', '-g', '!.git',
                             '-g', '!.ipynb_checkpoints'], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    suffixes = {'.py', '.java', '.json', '.toml', '.md', '.sh', '.yml', '.yaml',
                '.js', '.jsx', '.ts', '.tsx', '.css', '.html', '.txt'}
    paths = sorted(p for p in result.stdout.splitlines()
                   if Path(p).suffix in suffixes or p == '.gitignore')
    files = {p: file_digest(ROOT / p) for p in paths}
    return {'revision': git('rev-parse', '--verify', 'HEAD'),
            'status': git('status', '--porcelain'), 'files': files,
            'sha256': digest(files)}


def corpus_snapshot(config):
    corpora = []
    for directory in config['data']['corpora']:
        folder = ROOT / directory
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['schema_version'] != config['data']['source_schema']:
            raise ValueError(f'{directory}: incompatible source schema')
        if len(manifest['feature_names']) != config['data']['source_features']:
            raise ValueError(f'{directory}: incompatible features')
        if len(manifest['action_vocabulary']) != config['actions']['count']:
            raise ValueError(f'{directory}: incompatible actions')
        if manifest['num_games'] != len(manifest['games']):
            raise ValueError(f'{directory}: inconsistent game count')
        if manifest['num_samples'] != sum(g['saved_steps'] for g in manifest['games']):
            raise ValueError(f'{directory}: inconsistent sample count')
        games = []
        for entry in manifest['games']:
            actual = file_digest(folder / entry['file'])
            if actual != entry['sha256']:
                raise ValueError(f'{directory}/{entry["file"]}: checksum mismatch')
            games.append({k: entry[k] for k in ('file', 'game_id', 'seed', 'split', 'sha256')})
        corpora.append({'path': directory, 'manifest_sha256': file_digest(manifest_path),
                        'schema_version': manifest['schema_version'],
                        'feature_names': manifest['feature_names'],
                        'action_vocabulary': manifest['action_vocabulary'],
                        'games': games, 'num_games': manifest['num_games'],
                        'num_samples': manifest['num_samples'],
                        'provider': manifest['provider'], 'strategy': manifest['strategy'],
                        'jsettlers_revision': manifest['jsettlers_revision']})
    if any(c['feature_names'] != corpora[0]['feature_names'] or
           c['action_vocabulary'] != corpora[0]['action_vocabulary'] for c in corpora):
        raise ValueError('Corpora disagree on features/actions')
    return corpora


def benchmark(config, seed):
    import torch
    if torch.__version__ != config['runtime']['torch_version']:
        raise ValueError(f'Expected torch {config["runtime"]["torch_version"]}, got {torch.__version__}')
    torch.manual_seed(stream_seeds(seed)['training'])
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')
    spec = config['runtime']['benchmark']
    layers = []
    width = config['data']['source_features']
    for _ in range(3):
        layers.extend([torch.nn.Linear(width, 256), torch.nn.GELU()])
        width = 256
    # Combined output only measures representative compute, not real policy/value.
    model = torch.nn.Sequential(*layers, torch.nn.Linear(256, 333)).eval()
    weights = model.state_dict()
    results = []
    for device in devices:
        model = model.to(device)
        model.load_state_dict(weights)
        def synchronize():
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
        for batch in spec['batch_sizes']:
            x = torch.zeros(batch, config['data']['source_features'], device=device)
            with torch.inference_mode():
                synchronize()
                start = time.perf_counter()
                model(x)
                synchronize()
                cold_ms = (time.perf_counter() - start) * 1000
                for _ in range(spec['warmup']):
                    model(x)
                synchronize()
                samples = []
                for _ in range(spec['repeats']):
                    start = time.perf_counter()
                    model(x)
                    synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
            results.append({'device': device, 'batch_size': batch,
                            'first_forward_ms': cold_ms, 'samples_ms': samples,
                            'p50_ms': statistics.median(samples),
                            'p95_ms': sorted(samples)[int(.95 * (len(samples)-1))]})
    return {'workload': 'synthetic 633-256-256-256-333 GELU, zero input; no encoding/search',
            'evaluation_device': None, 'results': results,
            'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                         'processor': platform.processor(),
                         'cpu': subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                               capture_output=True, text=True).stdout.strip()
                         if sys.platform == 'darwin' else platform.processor(),
                         'torch_threads': torch.get_num_threads(),
                         'cuda_devices': [torch.cuda.get_device_name(i)
                                          for i in range(torch.cuda.device_count())],
                         'mps_built': torch.backends.mps.is_built(),
                         'available_devices': devices}}


def validate_record(record):
    from jsonschema import Draft202012Validator
    schema = json.loads((EXPERIMENTS / 'run-record.schema.json').read_text())
    Draft202012Validator(schema).validate(record)
    if record['configuration']['sha256'] != digest(record['configuration']['content']):
        raise ValueError('Configuration hash mismatch')
    if record['source']['sha256'] != digest(record['source']['files']):
        raise ValueError('Source manifest hash mismatch')
    if record['rng_seeds'] != stream_seeds(record['seed']):
        raise ValueError('RNG stream seeds mismatch')


def prepare(config_path, seed, output):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    if config['stage'] != 'preflight':
        raise ValueError('This command only prepares preflight runs; training gates are not implemented')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    identity = f'{config["experiment"]}-{digest(config)[:12]}-seed{seed}-{timestamp}'
    record = {'schema_version': 1, 'run_id': identity, 'status': 'preflight-complete',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'configuration': {'content': config, 'sha256': digest(config)},
              'seed': seed, 'rng_seeds': stream_seeds(seed),
              'source': source_snapshot(), 'datasets': corpus_snapshot(config),
              'software': {'python': sys.version, 'executable': sys.executable,
                           'packages': dict(sorted((d.metadata['Name'], d.version)
                                                  for d in importlib.metadata.distributions()))},
              'checkpoint': None, 'metrics': {}, 'elapsed_seconds': 0}
    measured = benchmark(config, seed)
    record['hardware'] = measured.pop('hardware')
    record['metrics']['device_preflight'] = measured
    record['elapsed_seconds'] = time.perf_counter() - started
    validate_record(record)
    destination = output / identity
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'run.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(destination / 'run.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', type=Path, default=EXPERIMENTS / 'phase1.json')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/catan-ai')
    p = sub.add_parser('validate')
    p.add_argument('record', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.config, args.seed, args.output)
    else:
        validate_record(json.loads(args.record.read_text()))
        print('Valid run record')


if __name__ == '__main__':
    main()
