"""Twelve single-thread CPU game collectors for the pinned fresh W4 lineage.

The original serial engine/trainer stays pinned. A separate execution manifest
archives this scheduler and records its provenance on every newly written game.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import argparse
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import time
import zipfile

import torch
from catanatron.models.player import Color
from agent_apis import benchmark
from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.observation import require
from agent_apis.neural import fresh_cohorts as fresh, compact_iterations as compact, friendly_w4 as w4, selfplay
from agent_apis.neural.pipeline import write

WORKER = None
MODELS = None


class CohortLedger:
    """Stable completed-game slots; retries retain a slot's opponent family."""
    def __init__(self, total, max_attempts, data=None):
        self.total, self.max_attempts = total, max_attempts
        self.data = data if data is not None else {'next_slot': 0, 'next_attempt': 0, 'assignments': {}}

    @property
    def assignments(self):
        return self.data['assignments']

    @property
    def completed_slots(self):
        return {a['slot'] for a in self.assignments.values() if a['status'] == 'completed'}

    def reserve(self, active, make_assignment):
        busy = {self.assignments[str(i)]['slot'] for i in active}
        for key in sorted(self.assignments, key=int):
            a = self.assignments[key]
            if int(key) not in active and a['slot'] not in busy and a['status'] not in ('completed', 'cutoff'):
                a['status'] = 'queued'
                return a
        if self.data['next_attempt'] >= self.max_attempts:
            return None
        incomplete = sorted({a['slot'] for a in self.assignments.values()} - self.completed_slots - busy)
        if incomplete:
            slot = incomplete[0]
        elif self.data['next_slot'] < self.total:
            slot = self.data['next_slot']
            self.data['next_slot'] += 1
        else:
            return None
        attempt = self.data['next_attempt']
        self.data['next_attempt'] += 1
        a = {**make_assignment(attempt, slot), 'attempt': attempt, 'slot': slot, 'status': 'queued'}
        self.assignments[str(attempt)] = a
        return a

    def finish(self, attempt, status, summary=None):
        require(status in ('completed', 'cutoff', 'interrupted'), 'Invalid worker result status')
        a = self.assignments[str(attempt)]
        if status == 'completed':
            require(not any(other['slot'] == a['slot'] and other['status'] == 'completed'
                            for key, other in self.assignments.items() if key != str(attempt)),
                    'Duplicate completed cohort slot')
        a['status'] = status
        if summary is not None:
            a['result'] = summary


class GameWorker(compact.Controller):
    def __init__(self, protocol, core, deadline, stop, parent_pid, execution, backend="python"):
        self.protocol, self.core, self.deadline = protocol, core, deadline
        self.stop_event, self.parent_pid, self.execution = stop, parent_pid, execution
        self.stop_requested = False
        self.frozen_ref = json.loads((ROOT/'experiments/catan-ai/phase5-frozen-opponent.json').read_text())['checkpoint']
        self.native = None
        if backend == "rust":
            from agent_apis.neural.native_selfplay import NativeClient
            self.native = NativeClient()
        self.frozen = None if self.native else load_model(self.frozen_ref)
        self.state, self.progress = {}, None

    def budget(self):
        if self.stop_requested or self.stop_event.is_set() or time.time() >= self.deadline or os.getppid() != self.parent_pid:
            raise compact.ReviewDue()

    def persist(self):
        if self.progress:
            write(self.progress, {**self.state, 'pid': os.getpid(), 'cpu_threads': torch.get_num_threads(),
                                  'inference_device': 'cpu', 'updated_at': self.now()})


def load_model(ref):
    key = ref['sha256']
    if key not in MODELS:
        model = selfplay.load_checkpoint(ref)
        require(str(model.device) == 'cpu', 'Parallel collection requires CPU inference')
        MODELS[key] = model
    torch.set_num_threads(1)
    return MODELS[key]


def initialize(protocol, core, deadline, stop, parent_pid, execution, parent, backend="python"):
    global WORKER, MODELS
    MODELS = {}
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    WORKER = GameWorker(protocol, core, deadline, stop, parent_pid, execution, backend)
    signal.signal(signal.SIGTERM, WORKER.request_stop)
    torch.set_num_threads(1)
    if backend == "python":load_model(parent)


def run_game(assignment, folder, parent, history):
    worker = WORKER
    attempt = assignment['attempt']
    path = ROOT/folder/f'game-{attempt:05d}.json.gz'
    partial = path.with_suffix('.partial.gz')
    worker.progress = ROOT/folder/'worker-progress'/f'game-{attempt:05d}.json'
    worker.state = {'attempt': attempt, 'slot': assignment['slot'], 'opponent_kind': assignment['kind']}
    if partial.exists() and not path.exists():
        # Keep interrupted prefixes even when the pinned engine restarts that seed.
        archive = path.parent/'partial-history'/f'game-{attempt:05d}-{file_digest(partial)}.json.gz'
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            shutil.copyfile(partial, archive)
    require(torch.get_num_threads() == 1, 'Game worker CPU thread count changed')
    try:
        if worker.native and assignment['kind'] != 'JSETTLERS':
            if path.exists():
                from agent_apis.neural.native_selfplay import load_journal
                record = load_journal(path)
                require(record['pair']==assignment['pair'] and record['candidate_checkpoint']==parent and
                        record['opponent_checkpoint']==assignment['enemy'] and record['source_sha256']==digest(worker.core),
                        'Committed native game lineage changed')
            else:
                record = worker.native.play(path,assignment['pair'],assignment['candidate_color'],parent,assignment['enemy'],
                    worker.frozen_ref,assignment['kind'],worker.protocol,worker.core,worker.deadline,
                    budget=worker.budget,execution=worker.execution)
        else:
            if worker.frozen is None:worker.frozen=load_model(worker.frozen_ref)
            model, enemy = load_model(parent), load_model(assignment['enemy'])
            record = worker.play(path, assignment['pair'], Color[assignment['candidate_color']], model, enemy,
                                 assignment['kind'], history, 'search')
        target = partial if record['status']=='interrupted' else path
    except compact.ReviewDue:
        if not partial.exists():
            return {'attempt': attempt, 'status': 'interrupted', 'pid': os.getpid(), 'cpu_threads': 1, 'inference_device': 'cpu'}
        with gzip.open(partial, 'rt') as stream:
            record = json.load(stream)
        target = partial
    record['collection_execution'] = worker.execution
    compact.save_journal(target, record)
    worker.state['status'] = record['status']
    worker.persist()
    return {'attempt': attempt, 'status': record['status'], 'journal': selfplay.reference(target),
            'actions': len(record['events']), 'pid': os.getpid(), 'cpu_threads': 1, 'inference_device': 'cpu',
            'elapsed_seconds': record.get('elapsed_seconds'), 'replay_verified': True}


def validate_plan_identity(plan, identity, core):
    """Permit execution-only migration without rewriting old assignments/journals."""
    saved=plan['identity']
    require({k:v for k,v in saved.items() if k!='execution'}=={k:v for k,v in identity.items() if k!='execution'},
            'Parallel cohort plan changed')
    if saved['execution']!=identity['execution']:
        ref=saved['execution']
        require(file_digest(ROOT/ref['path'])==ref['sha256'],'Historical execution manifest changed')
        old=json.loads((ROOT/ref['path']).read_text())
        require(old['original_source_sha256']==core,'Pinned engine/trainer changed during backend migration')


class ParallelFresh(fresh.Fresh):
    def __init__(self, hours, workers=12, backend="python"):
        require(backend in ("python","rust"), "Unknown collection backend")
        self.backend=backend
        super().__init__(hours)
        self.workers = workers
        self.extension = {str(Path(__file__).relative_to(ROOT)): file_digest(Path(__file__))}
        if backend == 'rust':
            from agent_apis.neural import native_assets,native_selfplay,native_codec
            native_assets.ensure_binary()
            files=[native_assets.ensure_binary(),Path(native_assets.__file__),Path(native_selfplay.__file__),Path(native_codec.__file__),*native_assets.source_files(),*sorted((native_assets.NATIVE/'runtime').glob('*'))]
            self.extension.update({str(p.relative_to(ROOT)):file_digest(p) for p in files})
        identity = {'collector_backend':backend,'original_source_sha256': self.core, 'extension_source_sha256': self.extension,
                    'workers': workers, 'cpu_threads_per_worker': 1, 'inference_device': 'cpu',
                    'scheduler': 'completed-game slots; retry same slot with fresh attempt seed; preserve exact opponent mixture',
                    'trainer': 'unchanged pinned fresh W4 trainer; existing GPU optimizer configuration'}
        folder = fresh.OUTPUT/'execution'
        folder.mkdir(exist_ok=True)
        path = folder/f'parallel-cpu-{digest(identity)}.json'
        if path.exists():
            require(json.loads(path.read_text()) == identity, 'Parallel execution manifest changed')
        else:
            write(path, identity)
            with zipfile.ZipFile(path.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name in self.extension:
                    archive.write(ROOT/name, name)
        self.execution_ref = selfplay.reference(path)
        self.state['collection_execution'] = self.execution_ref
        self.state['collection_runtime'] = {'workers': workers, 'cpu_threads_per_worker': 1, 'inference_device': 'cpu','backend':backend}
        self.persist()

    def guard(self):
        super().guard()
        require(all(file_digest(ROOT/name) == sha for name, sha in self.extension.items()), 'Parallel collector source changed')

    def assignment(self, iteration, attempt, slot, parent, history):
        pair, kind, color = self.game_spec(iteration, attempt, slot)
        enemy = history[benchmark.named_seed(pair['agent_seed'], 'history') % len(history)] if kind == 'historical' else parent
        return {'pair': pair, 'kind': kind, 'candidate_color': color.value, 'enemy': enemy}

    def validate_record(self, path, assignment, parent):
        with gzip.open(path, 'rt') as stream:
            record = json.load(stream)
        if record.get('format') == 'catan-compact-native-v1':
            require(record['native']['events_sha256']==digest(record['events']), 'Native journal targets/events changed')
        require(record['pair'] == assignment['pair'] and record['candidate_color'] == assignment['candidate_color'] and
                record['opponent_kind'] == assignment['kind'] and record['source_sha256'] == digest(self.core) and
                record['candidate_checkpoint'] == parent and record['opponent_checkpoint'] == assignment['enemy'],
                'Committed collection lineage or slot changed')
        compact.replay_journal(record)
        return record

    def collect_parallel(self, folder, iteration, parent, history, item):
        if self.backend == 'rust':
            from agent_apis.neural.native_assets import export_checkpoint
            frozen=json.loads((ROOT/'experiments/catan-ai/phase5-frozen-opponent.json').read_text())['checkpoint']
            for ref in [parent,frozen,*history]:export_checkpoint(ref)
        plan_path = folder/'parallel-plan.json'
        total = self.protocol['completed_games_per_iteration']
        make = lambda attempt, slot: self.assignment(iteration, attempt, slot, parent, history)
        identity = {'round': iteration, 'parent': parent, 'history': history, 'target_games': total,
                    'execution': self.execution_ref}
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            validate_plan_identity(plan,identity,self.core)
            revisions=plan.setdefault('execution_revisions',[plan['identity']['execution']])
            if self.execution_ref not in revisions:revisions.append(self.execution_ref)
            ledger = CohortLedger(total, self.protocol['max_attempts_per_iteration'], plan['ledger'])
        else:
            ledger = CohortLedger(total, self.protocol['max_attempts_per_iteration'])
            # Bootstrap only the contiguous committed serial history, preserving
            # its completed-slot choices and every original journal byte.
            paths = sorted((folder/'collection').glob('*.json.gz'))
            for attempt, path in enumerate(paths):
                require(path.name == f'game-{attempt:05d}.json.gz', 'Unplanned collection journal gap')
                slot = len(ledger.completed_slots)
                assignment = {**make(attempt, slot), 'attempt': attempt, 'slot': slot, 'status': 'queued'}
                ledger.assignments[str(attempt)] = assignment
                record = self.validate_record(path, assignment, parent)
                ledger.finish(attempt, record['status'], {'journal': selfplay.reference(path), 'replay_verified': True})
                ledger.data.update(next_attempt=attempt+1, next_slot=len(ledger.completed_slots))
            plan = {'identity': identity, 'ledger': ledger.data}
        known = set()
        for key, assignment in ledger.assignments.items():
            expected = make(int(key), assignment['slot'])
            require(all(assignment[name] == value for name, value in expected.items()), 'Assignment seeds/opponent changed')
            path = folder/'collection'/f'game-{int(key):05d}.json.gz'
            known.add(path.name)
            if path.exists():
                committed=assignment.get('result',{}).get('journal')
                if committed:require(file_digest(path)==committed['sha256'],'Committed parallel journal checksum changed')
                record = self.validate_record(path, assignment, parent)
                ledger.finish(int(key), record['status'], {'journal': selfplay.reference(path), 'replay_verified': True})
            else:
                require(assignment['status'] not in ('completed', 'cutoff'), 'Committed parallel journal missing')
        require(all(path.name in known for path in (folder/'collection').glob('*.json.gz')), 'Unplanned parallel game journal')
        write(plan_path, plan)
        self.state.pop('active_game', None)
        ctx = mp.get_context('spawn')
        stop = ctx.Event()
        active = {}
        last_progress = 0.
        self.state['stage'] = f'round-{iteration}:parallel-large-cohort-collection'
        try:
            with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=initialize,
                    initargs=(self.protocol, self.core, self.deadline, stop, os.getpid(), self.execution_ref, parent, self.backend)) as pool:
                try:
                    while len(ledger.completed_slots) < total:
                        if self.stop_requested or time.time() >= self.deadline:
                            stop.set()
                        if not stop.is_set():
                            self.guard()
                            while len(active) < self.workers:
                                assignment = ledger.reserve(set(active.values()), make)
                                if assignment is None:
                                    break
                                write(plan_path, plan)  # durable before submitting any work
                                future = pool.submit(run_game, assignment, str((folder/'collection').relative_to(ROOT)), parent, history)
                                active[future] = assignment['attempt']
                        if not active:
                            if stop.is_set():
                                raise compact.ReviewDue()
                            require(len(ledger.completed_slots) == total, 'Cohort attempt budget exhausted')
                            break
                        done, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED)
                        for future in done:
                            attempt = active.pop(future)
                            result = future.result()
                            require(result['cpu_threads'] == 1 and result['inference_device'] == 'cpu', 'Unexpected worker runtime')
                            ledger.finish(attempt, result['status'], result)
                            write(plan_path, plan)
                            print(f'PARALLEL ROUND {iteration} completed {len(ledger.completed_slots)}/{total}; attempt {attempt} {result["status"]}', flush=True)
                        if done or time.monotonic() - last_progress >= 5:
                            item.update(completed_games=len(ledger.completed_slots),
                                        attempted_games=sum(a['status'] in ('completed', 'cutoff') for a in ledger.assignments.values()),
                                        dispatched_attempts=ledger.data['next_attempt'],
                                        completed_opponents=dict(Counter(a['kind'] for a in ledger.assignments.values() if a['status']=='completed')))
                            progress = []
                            for attempt in active.values():
                                path = folder/'collection'/'worker-progress'/f'game-{attempt:05d}.json'
                                progress.append(json.loads(path.read_text()) if path.exists() else {'attempt': attempt, 'status': 'starting'})
                            self.state['active_games'] = progress
                            self.persist()
                            last_progress = time.monotonic()
                finally:
                    stop.set()
        finally:
            write(plan_path, plan)
            self.state['active_games'] = []
            self.persist()
        require(len(ledger.completed_slots) == total, 'Incomplete parallel cohort')

    def execute(self):
        groups = self.collect_expert()
        self.expert, self.cache = self.load_samples(groups['train']), self.load_samples(groups['validation'])
        self.parent = self.state['initial_parent']
        w4.OUTPUT, w4.CORPUS = fresh.OUTPUT/'pretraining', fresh.OUTPUT/'expert'
        self.state['stage'] = 'fresh-XD-CNN-initial-training'
        self.persist()
        pretrain = self.train_w4(self.expert, self.expert)
        self.state['pretraining'], self.state['initial_parent'] = pretrain, pretrain['checkpoint']
        self.persist()
        parent, history, iteration = pretrain['checkpoint'], [pretrain['checkpoint']], 1
        while True:
            self.guard()
            self.current_round = iteration
            folder = fresh.OUTPUT/f'round-{iteration:03d}'
            (folder/'collection').mkdir(parents=True, exist_ok=True)
            item = self.state['iterations'].setdefault(str(iteration), {'status': 'collecting', 'parent': parent})
            require(item['parent'] == parent, 'Fresh cohort lineage changed')
            if item['status'] == 'completed':
                parent = item['training']['checkpoint']; history.append(parent); iteration += 1
                continue
            self.collect_parallel(folder, iteration, parent, history, item)
            data = self.replay_data()
            self.parent, w4.OUTPUT = parent, folder
            item['training'] = self.train_w4(self.expert, data)
            del data
            self.persist()
            item['evaluation'] = self.evaluate_policy(folder, parent, item['training']['checkpoint'])
            item['status'] = 'completed'
            self.persist()
            parent = item['training']['checkpoint']; history.append(parent); iteration += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=3)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--engine',choices=('python','rust'),default='python',help='Rust neural self-play; JSettlers retains Python collection')
    args = parser.parse_args()
    require(0 < args.hours <= 3 and args.workers == 12, 'Use bounded windows and the selected twelve CPU workers')
    compact.OUTPUT = fresh.OUTPUT; compact.STATE = fresh.STATE
    selfplay.OUTPUT = fresh.OUTPUT; selfplay.STATE = fresh.STATE
    with benchmark.exclusive_benchmark():
        work = ParallelFresh(args.hours, args.workers, args.engine)
        try:
            work.execute()
        except compact.ReviewDue:
            work.state.update(status='paused', stage='window-ended-or-user-paused', pid=None, active_games=[])
            work.persist()
        except BaseException as error:
            work.state.update(status='needs-investigation', error=f'{type(error).__name__}: {error}', pid=None, active_games=[])
            work.persist()
            raise


if __name__ == '__main__':
    main()
