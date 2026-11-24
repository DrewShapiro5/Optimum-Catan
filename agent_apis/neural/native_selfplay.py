"""Production Rust self-play with replay-verified, Python-trainable journals.

CLI collection is separate from training. The cohort scheduler can select this
backend while retaining its pinned trainer, whole-game splits and update gate.
"""
import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import queue
import shutil
import threading
import time

from catanatron.serialization import action_record_to_json,map_to_json
from agent_apis import benchmark
from agent_apis.experiment import ROOT,digest,file_digest
from agent_apis.observation import require
from agent_apis.neural import compact_iterations as compact,selfplay
from agent_apis.neural.native_assets import NATIVE,ensure_binary,export_checkpoint,runtime_environment,source_identity
from agent_apis.neural.native_codec import Codec,canonical,job_meta
from agent_apis.neural.pipeline import write

PROTOCOL_VERSION=1

def validate_protocol(protocol):
    fixed={'friendly_robber':True,'reuse_subtree':True,'max_turns':1000,'max_game_actions':10000,'c_puct':1.5,'max_search_actions':128,'root_noise_fraction':.25,'dirichlet_total_concentration':10,'early_temperature':1.,'late_temperature':.25,'early_turns':20}
    require(all(protocol.get(k)==v for k,v in fixed.items()),'Native search does not support changed protocol settings')
    require(type(protocol['simulations']) is int and protocol['simulations']>0,'Invalid native simulation count')


def execution_identity(binary=None):
    binary=binary or ensure_binary()
    library=Path(runtime_environment()['ORT_DYLIB_PATH'])
    return {'backend':'rust-onnx-cpu','protocol_version':PROTOCOL_VERSION,'binary_sha256':file_digest(binary),
            'runtime_sha256':file_digest(library),'native_source_sha256':source_identity(),
            'python_source_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in [Path(__file__),Path(__file__).with_name('native_assets.py'),Path(__file__).with_name('native_codec.py')]},
            'cpu_threads_per_game':1}


class JournalRecorder:
    """Verify streaming native events and write the existing compact game format."""
    def __init__(self,game,record,simulations,setup_simulations=None):
        self.game,self.record,self.simulations,self.setup_simulations=game,record,simulations,setup_simulations
        self.codec=Codec(game);self.total_simulations=0;self.decisions=0
        self.record.update(initial_sha256=compact.snapshot_digest(game),events=[],
            board_sha256=digest(map_to_json(game.state.board.map)),simulator_seed=record['pair']['simulator_seed'])

    def append(self,events):
        require(isinstance(events,list) and 0<len(events)<=10,'Invalid native event batch')
        for e in events:
            require(self.game.winning_color() is None,'Events after terminal game')
            legal={selfplay.action_id(a):a for a in self.game.playable_actions}
            require(type(e['action_id']) is int and e['action_id'] in legal,'Illegal native action')
            actor=self.game.state.current_color()
            require(type(e['actor']) is int and e['actor']==self.game.state.colors.index(actor),'Wrong native actor')
            visits=e['visits'];require(isinstance(visits,list),'Invalid native visits')
            ids=[p[0] for p in visits]
            require(len(ids)==len(set(ids)) and all(type(i) is int and i in legal and type(n) is int and n>0 for i,n in visits),'Invalid native search targets')
            metrics=e['metrics']
            if len(legal)>1:
                require(isinstance(metrics,dict) and type(metrics['simulations']) is int,'Missing native search metrics')
                expected_sims=self.setup_simulations if (self.game.state.is_initial_build_phase and self.setup_simulations is not None) else self.simulations
                require(expected_sims is None or metrics['simulations']==expected_sims,'Short native search')
                if expected_sims is not None:require(visits,'Missing native visit target')
                self.total_simulations+=metrics['simulations'];self.decisions+=1
            else:require(not visits and metrics is None,'Searched forced native action')
            a=legal[e['action_id']];result=self.game.execute(a)
            require(self.codec.chance(a,result)==e['chance'],'Native chance replay mismatch')
            if a.action_type.value=='ROLL':require(list(result.result)==e['dice'],'Native dice replay mismatch')
            event={'a':e['action_id'],'event':action_record_to_json(result)}
            if visits and (actor.value==self.record['candidate_color'] or self.record['opponent_kind']=='current'):event['v']=visits
            self.record['events'].append(event)

    def partial(self):
        return {**self.record,'native':{**self.record['native'],'events_sha256':digest(self.record['events'])},'status':'in-progress','final_sha256':compact.snapshot_digest(self.game)}

    def finish(self,message):
        require(message['type']=='done' and message['version']==PROTOCOL_VERSION,'Invalid native completion')
        require(message['actions']==len(self.record['events']) and message['searched_decisions']==self.decisions and message['simulations']==self.total_simulations,'Native game counters mismatch')
        require(canonical(self.codec.snapshot(self.game))['state']==canonical({'state':message['final_state']})['state'],'Native final state mismatch')
        require(list(self.game.state.random.getstate()[1])==message['rng_state'],'Native final RNG mismatch')
        winner=self.game.winning_color();index=self.game.state.colors.index(winner) if winner else -1
        require(index==message['winner'],'Native outcome mismatch')
        status=message['status'];require(status in ('completed','cutoff','interrupted') and (status=='completed')==(winner is not None),'Invalid native completion status')
        self.record.update(status=status,winner=winner.value if winner else None,turns=self.game.state.num_turns,
            final_sha256=compact.snapshot_digest(self.game),elapsed_seconds=message['elapsed_seconds'],simulations=self.total_simulations,
            searched_rows=sum(bool(e.get('v')) for e in self.record['events']),match_score=float(winner.value==self.record['candidate_color']) if winner else .5)
        self.record['native']['events_sha256']=digest(self.record['events'])
        compact.replay_journal(self.record)
        return self.record


class NativeClient:
    """One long-lived Rust process; never more than one game at a time."""
    def __init__(self):
        import subprocess
        self.binary=ensure_binary();self.identity=execution_identity(self.binary)
        self.log=(NATIVE/'target/worker-stderr.log').open('a')
        self.process=subprocess.Popen([str(self.binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,bufsize=1,env=runtime_environment())
        self.messages=queue.Queue()
        def read():
            try:
                for line in self.process.stdout:self.messages.put(json.loads(line))
            except BaseException as e:self.messages.put(e)
            finally:self.messages.put(None)
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start();self.closed=False
        ready=self._receive(30);require(ready.get('type')=='ready' and ready['version']==PROTOCOL_VERSION and ready['threads']==1,'Unsupported native worker')
        atexit.register(self.close)

    def _receive(self,timeout=.25):
        message=self.messages.get(timeout=timeout)
        if isinstance(message,BaseException):raise RuntimeError('Invalid native worker output') from message
        require(message is not None,'Native worker exited; see native_worker/target/worker-stderr.log')
        require(message.get('type')!='error',f'Native worker rejected request: {message.get("error")}')
        return message

    def close(self):
        if self.closed:return
        self.closed=True
        if self.process.poll() is None:
            try:self.process.stdin.close();self.process.wait(timeout=3)
            except BaseException:self.process.kill();self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdout.close()
        if not self.process.stdin.closed:self.process.stdin.close()
        self.log.close()
        atexit.unregister(self.close)

    def play(self,path,pair,candidate_color,candidate,enemy,frozen,kind,protocol,core,deadline,budget=None,execution=None,max_turns=1000,max_actions=10000,seconds=None):
        require(kind in ('current','historical'),'Java opponents use the Python collector')
        validate_protocol(protocol)
        require(seconds is None or math.isfinite(seconds) and seconds>0,'Invalid native time budget')
        require(type(max_turns) is int and 0<max_turns<=1000 and type(max_actions) is int and 0<max_actions<=10000,'Invalid native game limits')
        pair=dict(pair);pair.setdefault('pair_id',pair.get('id',digest(pair)))
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        require(not path.exists(),'Refusing to overwrite committed native game')
        old_partial=path.with_suffix('.partial.gz')
        if old_partial.exists():
            archive=path.parent/'partial-history'/f'{path.stem}-{file_digest(old_partial)}.json.gz'
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():shutil.copyfile(old_partial,archive)
        candidate_path,enemy_path,frozen_path=[export_checkpoint(r) for r in [candidate,enemy,frozen]]
        game=benchmark.make_game(pair,friendly_robber=True);meta=job_meta(game);initial=meta['initial'];state=list(game.state.random.getstate()[1]);colors=[c.value for c in game.state.colors]
        color=getattr(candidate_color,'value',candidate_color);require(color in colors,'Invalid learner color')
        stop=path.with_suffix('.stop');stop.unlink(missing_ok=True)
        task={'job':meta,'start':initial,'start_rng':state,'logs':[],'id':path.name,'prefix':0,'seat_seeds':[benchmark.named_seed(pair['agent_seed'],c) for c in colors],'rng_state':state}
        paths=[str(candidate_path if c==color or kind=='current' else enemy_path) for c in colors]
        record={'format':'catan-compact-native-v1','pair':pair,'candidate_color':color,'opponent_kind':kind,'mode':'search',
            'candidate_checkpoint':candidate,'opponent_checkpoint':enemy,'frozen_search_opponent':frozen,'source_sha256':digest(core),
            'collection_execution':execution,'native':{'execution':self.identity,'candidate_export_sha256':file_digest(candidate_path),
                'enemy_export_sha256':file_digest(enemy_path),'frozen_export_sha256':file_digest(frozen_path),'search_protocol':protocol,'budget_seconds':seconds,'limits':{'max_turns':max_turns,'max_actions':max_actions}}}
        recorder=JournalRecorder(game,record,None if seconds is not None else protocol['simulations']);partial=path.with_suffix('.partial.gz')
        request={'version':PROTOCOL_VERSION,'task':task,'models':paths,'frozen':str(frozen_path),'simulations':None if seconds is not None else protocol['simulations'],'seconds':seconds,
            'deadline':deadline,'stop_file':str(stop.resolve()),'max_turns':max_turns,'max_actions':max_actions}
        self.process.stdin.write(json.dumps(request,separators=(',',':'),allow_nan=False)+'\n');self.process.stdin.flush()
        stopping=False
        try:
            while True:
                try:
                    if budget and not stopping:budget()
                except compact.ReviewDue:stop.touch();stopping=True
                try:message=self._receive()
                except queue.Empty:continue
                if message['type']=='events':recorder.append(message['events']);compact.save_journal(partial,recorder.partial())
                elif message['type']=='done':
                    record=recorder.finish(message);target=partial if record['status']=='interrupted' else path
                    compact.save_journal(target,record)
                    if target==path:partial.unlink(missing_ok=True)
                    return record
                else:raise ValueError('Unexpected native message')
        except BaseException:
            self.close();raise
        finally:stop.unlink(missing_ok=True)


def load_journal(path,derive=False,limit=64,seed=11):
    with gzip.open(path,'rt') as f:record=json.load(f)
    require(record.get('format')=='catan-compact-native-v1','Not a native training journal')
    require(record['native']['events_sha256']==digest(record['events']),'Native journal targets/events changed')
    compact.replay_journal(record)
    if not derive:return record
    require(record['status']=='completed' and record['winner'] is not None,'Only completed games can supply training outcomes')
    from agent_apis.neural.fresh_cohorts import derive as derive_positions
    return derive_positions(record,limit,seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--opponent',type=Path)
    parser.add_argument('--games',type=int,default=1);parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=2026111901)
    budget=parser.add_mutually_exclusive_group();budget.add_argument('--simulations',type=int,default=None);budget.add_argument('--time-ms',type=float)
    parser.add_argument('--max-turns',type=int,default=1000);parser.add_argument('--max-actions',type=int,default=10000)
    args=parser.parse_args();require(args.games>0 and args.workers>0 and (args.simulations is None or args.simulations>0),'Invalid collection arguments')
    protocol=json.loads((ROOT/'experiments/catan-ai/fresh-xdcnn-visible.json').read_text())
    if args.simulations is not None:protocol['simulations']=args.simulations
    candidate=selfplay.reference(args.checkpoint.resolve());enemy=selfplay.reference(args.opponent.resolve()) if args.opponent else candidate
    frozen=json.loads((ROOT/'experiments/catan-ai/phase5-frozen-opponent.json').read_text())['checkpoint']
    for ref in [candidate,enemy,frozen]:export_checkpoint(ref)
    ensure_binary();args.output.mkdir(parents=True,exist_ok=True);results=[];local=threading.local();clients=[];clients_lock=threading.Lock()
    def collect(i):
        pair={'pair_id':f'native-{args.seed}-{i}','board_seed':benchmark.named_seed(args.seed,f'board-{i}'),'simulator_seed':benchmark.named_seed(args.seed,f'simulator-{i}'),'agent_seed':benchmark.named_seed(args.seed,f'agent-{i}')}
        if not hasattr(local,'client'):
            local.client=NativeClient()
            with clients_lock:clients.append(local.client)
        client=local.client
        record=client.play(args.output/f'game-{i:05d}.json.gz',pair,'BLUE',candidate,enemy,frozen,'historical' if args.opponent else 'current',protocol,{},time.time()+3*3600,
                max_turns=args.max_turns,max_actions=args.max_actions,seconds=args.time_ms/1000 if args.time_ms is not None else None)
        return {'path':str((args.output/f'game-{i:05d}.json.gz').resolve()),'status':record['status'],'actions':len(record['events']),'searched_rows':record['searched_rows']}
    try:
        with benchmark.exclusive_benchmark():
            with ThreadPoolExecutor(max_workers=min(args.workers,args.games)) as pool:
                for result in pool.map(collect,range(args.games)):results.append(result);print(json.dumps(result),flush=True)
    finally:
        for client in clients:client.close()
    write(args.output/'collection.json',{'backend':'rust-onnx-cpu','workers':args.workers,'protocol':protocol,'games':results,'training_started':False})
if __name__=='__main__':main()
