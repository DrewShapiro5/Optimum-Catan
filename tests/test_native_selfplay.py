"""Unit checks and opt-in real Rust -> compact journal -> W4 training test."""
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from agent_apis import benchmark
from agent_apis.experiment import ROOT,digest,file_digest
from agent_apis.neural import compact_iterations as compact,friendly_w4 as w4,selfplay
from agent_apis.neural.native_codec import Codec,job_meta
from agent_apis.neural.native_selfplay import JournalRecorder,load_journal,validate_protocol
from agent_apis.neural.parallel_cohorts import validate_plan_identity

PAIR={'pair_id':'native-unit','board_seed':1123,'simulator_seed':2134,'agent_seed':3145}

def record_base():
    return {'pair':PAIR,'candidate_color':'BLUE','opponent_kind':'current','native':{},'format':'catan-compact-native-v1'}

def trace(count=40):
    game=benchmark.make_game(PAIR);codec=Codec(game);events=[];rng=random.Random(8)
    for _ in range(count):
        actor=game.state.current_player_index;actions=game.playable_actions;a=rng.choice(actions);i=selfplay.action_id(a);searched=len(actions)>1
        result=game.execute(a);events.append({'actor':actor,'action_id':i,'chance':codec.chance(a,result),'dice':list(result.result) if a.action_type.value=='ROLL' else None,'visits':[[i,4]] if searched else [],'metrics':{'simulations':4} if searched else None})
    return game,events


class RecordingTests(unittest.TestCase):
    def test_codec_does_not_mutate_initial_digest(self):
        game=benchmark.make_game(PAIR);before=compact.snapshot_digest(game);job_meta(game)
        self.assertEqual(before,compact.snapshot_digest(game))

    def test_streamed_record_is_accepted_by_unmodified_replayer(self):
        game,events=trace();rec=JournalRecorder(benchmark.make_game(PAIR),record_base(),4)
        for i in range(0,len(events),10):rec.append(events[i:i+10])
        result=rec.finish({'type':'done','version':1,'actions':len(events),'searched_decisions':rec.decisions,'simulations':rec.total_simulations,
            'final_state':Codec(game).snapshot(game)['state'],'rng_state':list(game.state.random.getstate()[1]),'winner':-1,'status':'cutoff','elapsed_seconds':1})
        self.assertEqual(compact.snapshot_digest(compact.replay_journal(result)),result['final_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'game.json.gz';compact.save_journal(path,result)
            with self.assertRaisesRegex(ValueError,'Only completed'):load_journal(path,derive=True)
            result['events'][0]['v'][0][1]+=1;compact.save_journal(path,result)
            with self.assertRaisesRegex(ValueError,'targets/events changed'):load_journal(path)

    def test_duplicate_illegal_visits_short_search_and_wrong_chance_fail(self):
        _,events=trace();first=events[0]
        bad=deepcopy(first);bad['visits']*=2
        with self.assertRaisesRegex(ValueError,'targets'):JournalRecorder(benchmark.make_game(PAIR),record_base(),4).append([bad])
        bad=deepcopy(first);bad['metrics']['simulations']=3
        with self.assertRaisesRegex(ValueError,'Short'):JournalRecorder(benchmark.make_game(PAIR),record_base(),4).append([bad])
        bad=deepcopy(first);bad['action_id']=999
        with self.assertRaisesRegex(ValueError,'Illegal'):JournalRecorder(benchmark.make_game(PAIR),record_base(),4).append([bad])
        bad=deepcopy(first);bad['chance']=999
        with self.assertRaisesRegex(ValueError,'chance'):JournalRecorder(benchmark.make_game(PAIR),record_base(),4).append([bad])

    def test_changed_search_protocol_is_rejected(self):
        protocol=json.loads((ROOT/'experiments/catan-ai/fresh-xdcnn-visible.json').read_text());validate_protocol(protocol)
        for field,value in [('c_puct',2),('max_search_actions',64),('early_temperature',.5),('friendly_robber',False),('reuse_subtree',False)]:
            changed={**protocol,field:value}
            with self.assertRaisesRegex(ValueError,'changed protocol'):validate_protocol(changed)

    def test_execution_migration_keeps_original_plan_and_blocks_lineage_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            p=Path(directory)/'old-execution.json';p.write_text(json.dumps({'original_source_sha256':{'engine':'abc'}}));ref=selfplay.reference(p)
            original={'round':1,'parent':{'sha256':'model'},'history':[],'target_games':2000,'execution':ref};plan={'identity':deepcopy(original),'ledger':{'old':'retained'}}
            new={**original,'execution':{'path':'new.json','sha256':'new'}}
            validate_plan_identity(plan,new,{'engine':'abc'});self.assertEqual(plan['identity'],original)
            for changed in [{**new,'target_games':8},{**new,'parent':{'sha256':'other'}}]:
                with self.assertRaises(ValueError):validate_plan_identity(plan,changed,{'engine':'abc'})
            with self.assertRaises(ValueError):validate_plan_identity(plan,new,{'engine':'changed'})


class BackendDispatchTests(unittest.TestCase):
    def test_rust_neural_dispatch_java_fallback_and_interrupted_target(self):
        from agent_apis.neural import parallel_cohorts as p
        for kind,status in [('current','cutoff'),('historical','interrupted'),('JSETTLERS','cutoff')]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT) as directory:
                record={'status':status,'events':[]}
                native=Mock();native.play.return_value=record
                worker=SimpleNamespace(native=native,execution={'sha256':'execution'},frozen=object(),frozen_ref={'sha256':'frozen'},protocol={},core={},deadline=time.time()+60,budget=Mock(),persist=Mock(),play=Mock(return_value=record))
                assignment={'attempt':0,'slot':0,'kind':kind,'candidate_color':'BLUE','pair':PAIR,'enemy':{'sha256':'enemy'}}
                with patch.object(p,'WORKER',worker),patch.object(p,'load_model',return_value=object()) as load,patch.object(p.torch,'get_num_threads',return_value=1):
                    result=p.run_game(assignment,str(Path(directory).relative_to(ROOT)),{'sha256':'parent'},[])
                if kind=='JSETTLERS':worker.play.assert_called_once();native.play.assert_not_called();self.assertEqual(load.call_count,2)
                else:native.play.assert_called_once();worker.play.assert_not_called();load.assert_not_called()
                target=Path(directory)/('game-00000.json.partial.gz' if status=='interrupted' else 'game-00000.json.gz')
                self.assertTrue(target.exists());self.assertEqual(result['status'],status)
                if status=='interrupted':self.assertFalse((Path(directory)/'game-00000.json.gz').exists())


@unittest.skipUnless(os.environ.get('CATAN_NATIVE_INTEGRATION')=='1','Set CATAN_NATIVE_INTEGRATION=1 for real native collection/training')
class NativeIntegrationTests(unittest.TestCase):
    def test_full_native_games_are_trainable_with_whole_game_splits(self):
        import torch
        from agent_apis.neural.native_selfplay import NativeClient
        from agent_apis.neural.value_ablation_models import TypedExphormer
        torch.set_num_threads(1)
        candidate=json.loads((ROOT/'runs/catan-ai/sparse-transformer-expert-catchup-visible/progress.json').read_text())['latest_checkpoint']
        frozen=json.loads((ROOT/'experiments/catan-ai/phase5-frozen-opponent.json').read_text())['checkpoint']
        protocol=json.loads((ROOT/'experiments/catan-ai/fresh-xdcnn-visible.json').read_text());protocol['simulations']=8
        # Fresh independently seeded games; never derive validation from the train game.
        with tempfile.TemporaryDirectory(dir=ROOT/'runs/catan-ai') as directory:
            client=NativeClient();datasets=[]
            try:
                for i in range(2):
                    seed=2026111901;pair={'pair_id':f'integration-{i}','board_seed':benchmark.named_seed(seed,f'board-{i}'),'simulator_seed':benchmark.named_seed(seed,f'simulator-{i}'),'agent_seed':benchmark.named_seed(seed,f'agent-{i}')}
                    path=Path(directory)/f'game-{i}.json.gz'
                    record=client.play(path,pair,'BLUE',candidate,candidate,frozen,'current',protocol,{},time.time()+180)
                    self.assertEqual(record['status'],'completed');
                    from agent_apis.neural.fresh_cohorts import Fresh
                    loader=object.__new__(Fresh);loader.budget=lambda:None;loader.protocol={'positions_per_game':64,'seed':11}
                    direct=loader.load_samples([path]);self.assertEqual(direct.summary['rows'],64)
                    data=load_journal(path,derive=True,limit=64,seed=11+i);w4.check_arrays(data);datasets.append(data)
                    again=load_journal(path,derive=True,limit=64,seed=11+i)
                    for key in data:np.testing.assert_array_equal(data[key],again[key])
                    with self.assertRaises(ValueError):client.play(path,pair,'BLUE',candidate,candidate,frozen,'current',protocol,{},time.time()+180)
                train=compact.MemoryData([datasets[0]],['train-game']);batch=train.sample(np.random.default_rng(3),16)
                self.assertAlmostEqual(train.summary['value_weight_expectation'],1.)
                model=TypedExphormer();model.load_state_dict(torch.load(ROOT/candidate['path'],map_location='cpu',weights_only=True)['state_dict']);model.train()
                before=model.heads.value.weight.detach().clone();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
                logits,values=model(torch.from_numpy(batch['features']));pl,vl=w4.losses(logits,values,*[torch.from_numpy(batch[k]) for k in ['legal','policy','value','value_weight']]);loss=pl+vl
                self.assertTrue(torch.isfinite(loss));optimizer.zero_grad();loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()));torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();self.assertFalse(torch.equal(before,model.heads.value.weight))
                with torch.inference_mode():
                    validation=datasets[1];logits,values=model(torch.from_numpy(validation['features']));p,v=w4.losses(logits,values,*[torch.from_numpy(validation[k]) for k in ['legal','policy','value']],torch.ones(len(validation['value'])));self.assertTrue(torch.isfinite(p+v))
                # Stop before any new event; interrupted games remain partial and are never trainable.
                path=Path(directory)/'interrupted.json.gz'
                record=client.play(path,PAIR,'RED',candidate,frozen,frozen,'historical',protocol,{},time.time()-1)
                self.assertEqual(record['status'],'interrupted');self.assertFalse(path.exists());self.assertTrue(path.with_suffix('.partial.gz').exists())
                with self.assertRaises(ValueError):load_journal(path.with_suffix('.partial.gz'),derive=True)
            finally:client.close()

    def test_historical_uses_distinct_seat_models_and_python_search_seeds(self):
        import torch
        from agent_apis.neural.native_selfplay import NativeClient
        from agent_apis.neural.provider import NeuralCheckpoint
        from agent_apis.observation import from_web_view
        from catanatron.serialization import web_view
        torch.set_num_threads(1)
        candidate=json.loads((ROOT/'runs/catan-ai/fresh-xdcnn-visible-progress.json').read_text())['initial_parent']
        frozen=json.loads((ROOT/'experiments/catan-ai/phase5-frozen-opponent.json').read_text())['checkpoint']
        protocol=json.loads((ROOT/'experiments/catan-ai/fresh-xdcnn-visible.json').read_text());protocol['simulations']=8
        with tempfile.TemporaryDirectory(dir=ROOT/'runs/catan-ai') as directory:
            client=NativeClient()
            try:
                record=client.play(Path(directory)/'historical.json.gz',PAIR,'RED',candidate,frozen,frozen,'historical',protocol,{},time.time()+120,max_actions=40)
            finally:client.close()
            game=benchmark.make_game(PAIR);model=NeuralCheckpoint(ROOT/candidate['path'],device='cpu');enemy=NeuralCheckpoint(ROOT/frozen['path'],device='cpu');torch.set_num_threads(1)
            seats={c:selfplay.SearchSeat(model if c.value=='RED' else enemy,enemy,benchmark.named_seed(PAIR['agent_seed'],c.value),protocol,game.id) for c in game.state.colors}
            for event in record['events']:
                actor=game.state.current_color()
                if len(game.playable_actions)>1:
                    selected,_,_=seats[actor].choose(from_web_view(web_view(game,actor),actor.value));self.assertEqual(event['a'],selfplay.action_id(selected))
                    if actor.value=='RED':
                        visits=[[i,e.visits] for i,e in sorted(seats[actor].tree.nodes[()].edges.items()) if e.visits];self.assertEqual(event['v'],visits)
                    else:self.assertNotIn('v',event)
                action=next(a for a in game.playable_actions if selfplay.action_id(a)==event['a']);game.execute(action)

if __name__=='__main__':unittest.main()
