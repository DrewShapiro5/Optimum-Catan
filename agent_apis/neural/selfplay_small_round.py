"""One bounded self-play round, preserving the earlier collection provenance."""
import argparse
import json

from agent_apis import benchmark
from agent_apis.experiment import ROOT, EXPERIMENTS, digest, file_digest
from agent_apis.neural import selfplay
from agent_apis.observation import require

PROTOCOL = EXPERIMENTS / 'phase9-selfplay-small-round.json'
OUTPUT = ROOT / 'runs/catan-ai/selfplay-small-round'
STATE = ROOT / 'runs/catan-ai/selfplay-small-round-progress.json'


def seed_records(manifest):
    records = []
    for entry in manifest['games']:
        path = ROOT / entry['record']['path']
        require(file_digest(path) == entry['record']['sha256'], 'Imported game record changed')
        record = json.loads(path.read_text())
        require(record['status'] == 'completed' and record['parent'] == manifest['parent'],
                'Imported game is incomplete or has a different parent')
        require(record['source_sha256'] == digest(manifest['original_source_sha256']), 'Imported source changed')
        require(record['dataset'] == entry['dataset'] and
                file_digest(ROOT / record['dataset']['path']) == record['dataset']['sha256'],
                'Imported training shard changed')
        require(entry['simulations_per_decision'] == manifest['original_protocol']['simulations'],
                'Imported search budget mislabeled')
        records.append(record)
    require(len(records) == manifest['completed_games'], 'Imported game count changed')
    require(len({r['pair']['pair_id'] for r in records}) == len(records), 'Duplicate imported games')
    return records


class SmallRound(selfplay.SelfPlay):
    def __init__(self):
        protocol = json.loads(PROTOCOL.read_text())
        self.seed_manifest = json.loads((ROOT / protocol['seed_manifest']).read_text())
        self.imported = seed_records(self.seed_manifest)
        self.offset = len(self.imported)
        super().__init__()
        require(self.parent == self.seed_manifest['parent'], 'Seed collection parent changed')
        self.state['reused_collection'] = {
            'manifest': protocol['seed_manifest'], 'games': self.offset,
            'rows': self.seed_manifest['training_rows'], 'simulations_per_decision': 512,
            'unfinished_game_excluded': True}
        self.persist()

    def game_spec(self, iteration, attempt, completed_slot=None):
        # Continue the original completed-game mixture sequence on fresh slots.
        slot = attempt if completed_slot is None else completed_slot
        return super().game_spec(iteration, attempt + self.offset, slot + self.offset)

    def collect(self, iteration, parent, history, record):
        require(iteration == 1, 'This pilot authorizes exactly one round')
        imported = seed_records(self.seed_manifest)
        target = self.protocol['completed_games_per_iteration']
        require(0 < len(imported) < target, 'Unexpected imported cohort size')
        record.update(reused_completed_games=len(imported),
            reused_training_rows=self.seed_manifest['training_rows'],
            new_completed_games_target=target-len(imported))
        original = self.protocol
        # The saved protocol remains the 100-game total. The existing collector
        # is asked only for the remainder; its records retain their new hashes.
        self.protocol = {**original, 'completed_games_per_iteration': target-len(imported)}
        try:
            new = super().collect(iteration, parent, history, record)
        finally:
            self.protocol = original
        combined = imported + new
        record.update(new_completed_games=sum(r['status'] == 'completed' for r in new),
            completed_games=sum(r['status'] == 'completed' for r in combined),
            attempted_games=len(combined),
            generation_seconds=sum(r['elapsed_seconds'] for r in combined),
            searched_training_rows=sum(r['dataset']['rows'] for r in combined if r.get('dataset')))
        selfplay.write(OUTPUT / 'iteration-1' / 'combined-collection.json', combined)
        self.persist()
        return combined

    def execute(self):
        record = self.state['iterations'].setdefault('1', {})
        records = self.collect(1, self.parent, [self.parent], record)
        record['training'] = self.train(1, self.parent, records)
        self.persist()
        record['evaluation'] = self.validate(1, record['training']['checkpoint'], self.parent, [self.parent], record)
        record['status'] = 'completed'
        self.state.update(status='completed', stage='completed-small-first-round', pid=None)
        self.persist()
        selfplay.write(OUTPUT / 'results.json', self.state)


def configure():
    selfplay.PROTOCOL, selfplay.OUTPUT, selfplay.STATE = PROTOCOL, OUTPUT, STATE
    manifest = json.loads(PROTOCOL.read_text())['seed_manifest']
    selfplay.CORE = (*selfplay.CORE, 'agent_apis/neural/selfplay_small_round.py',
        'experiments/catan-ai/phase9-selfplay-small-round.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    configure()
    with benchmark.exclusive_benchmark():
        work = SmallRound()
        try:
            if args.preflight:
                work.state.update(status='prepared', stage='ready-not-started', pid=None)
                work.persist()
                print(json.dumps({'status': 'prepared', 'reused_games': work.offset,
                    'new_games': work.protocol['completed_games_per_iteration']-work.offset,
                    'new_simulations': work.protocol['simulations'],
                    'training_updates': work.protocol['updates_per_iteration']}), flush=True)
            else:
                work.execute()
        except BaseException as error:
            work.state.update(status='needs-investigation', error=f'{type(error).__name__}: {error}', pid=None)
            work.persist()
            raise


if __name__ == '__main__':
    main()
