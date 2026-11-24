import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch
from agent_apis import benchmark as b
from agent_apis.experiment import EXPERIMENTS,file_digest
from tests.test_benchmark import result


class QuickScheduleTests(unittest.TestCase):
    def config(self):
        c=json.loads(b.CONFIG.read_text());c['candidate']['key']=c['opponent']['key']='RANDOM'
        c['evaluation'].update(schedule='quick-v1',schedule_sha256=file_digest(EXPERIMENTS/'evaluation-quick-v1.json'))
        return c
    def fixture(self,pair,color,config,stream):
        event={'role':'candidate','elapsed_ms':1.,'cold':False,'forced':False,'overrun':False}
        stream.write(json.dumps(event)+'\n')
        return result(pair['pair_id'],color.value,float(color.value=='BLUE')),[event]
    def test_ten_smoke_fifty_total_and_prefix_reuse(self):
        c=self.config()
        with TemporaryDirectory() as temp,patch.object(b,'play_game',side_effect=self.fixture) as play,redirect_stdout(io.StringIO()):
            smoke=b.run(c,Path(temp));self.assertEqual(play.call_count,10)
            c['evaluation']['tier']='screening';full=b.run(c,Path(temp),smoke)
            self.assertEqual(play.call_count,50)
            r=json.loads(full.read_text());self.assertEqual(r['metrics']['benchmark']['games'],50)
            self.assertEqual(r['metrics']['continuation']['reused_pairs'],5)
            self.assertEqual(len(r['metrics']['evaluation_pairs']),25)
            before=[json.loads(l) for l in (smoke.parent/'games.jsonl').read_text().splitlines()]
            after=[json.loads(l) for l in (full.parent/'games.jsonl').read_text().splitlines()]
            self.assertEqual(before,after[:10])
    def test_interrupted_run_keeps_completed_odd_seat_without_replaying_it(self):
        c=self.config();completed=[];attempts=0
        def interrupt(pair,color,config,stream):
            nonlocal attempts
            attempts+=1
            if attempts==18:raise KeyboardInterrupt()
            completed.append((pair['pair_id'],color.value));return self.fixture(pair,color,config,stream)
        with TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            with patch.object(b,'play_game',side_effect=self.fixture):smoke=b.run(c,Path(temp))
            c['evaluation']['tier']='screening'
            with patch.object(b,'play_game',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):b.run(c,Path(temp),smoke)
            def resume(pair,color,config,stream):
                completed.append((pair['pair_id'],color.value));return self.fixture(pair,color,config,stream)
            with patch.object(b,'play_game',side_effect=resume) as play:full=b.run(c,Path(temp),smoke)
            self.assertEqual(play.call_count,23)
            self.assertEqual(len(completed),40);self.assertEqual(len(set(completed)),40)
            self.assertEqual(json.loads(full.read_text())['metrics']['benchmark']['games'],50)
    def test_unknown_or_unpinned_schedule_rejected(self):
        c=self.config();c['evaluation']['schedule_sha256']='0'*64
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'schedule checksum'):b.run(c,Path(temp))
        c['evaluation']['schedule']='invented'
        with self.assertRaisesRegex(ValueError,'Unknown evaluation schedule'):b.schedule_sizes(c['evaluation'])


if __name__=='__main__':unittest.main()

class QuickWorkflowTests(unittest.TestCase):
    def test_stopped_policy_is_available_but_failed_policy_is_excluded(self):
        from agent_apis.neural import quick_evaluation as q
        from agent_apis.experiment import ROOT
        with TemporaryDirectory(dir=ROOT/'runs/catan-ai') as tmp:
            folder=Path(tmp);path=folder/'phase3-NEURAL_POLICY-fixture'/'run.json';path.parent.mkdir()
            log=path.parent/'games.jsonl';log.write_text(json.dumps(result('0','BLUE',0.))+'\n')
            record={'status':'stopped-early','configuration':{'content':{'candidate':{'checkpoint':'fixture'}}},
                    'source':{'files':{p:file_digest(ROOT/p) for p in
                        ('agent_apis/neural/models.py','agent_apis/neural/topology.py','agent_apis/observation.py')}},
                    'metrics':{'providers':{'opponent':{'key':'fixture'}},'artifacts':{'games.jsonl':file_digest(log)},
                               'benchmark':{'futility':{'stopped':True}}}}
            work=q.QuickEvaluation.__new__(q.QuickEvaluation)
            with patch.object(q,'OUTPUT',folder),patch.object(work,'configuration',return_value={'candidate':{'checkpoint':'fixture'}}),\
                 patch.object(q.benchmark,'provider_identity',return_value={'key':'fixture'}):
                path.write_text(json.dumps(record));reference,rows=work.policy_reference('TRANSFORMER')
                self.assertEqual(reference,path);self.assertEqual(len(rows),1)
                log.write_text(log.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'logs changed'):work.policy_reference('TRANSFORMER')
                record['status']='failed';path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'No preserved policy baseline'):work.policy_reference('TRANSFORMER')

    def test_report_only_uses_saved_games_and_rejects_incomplete_state(self):
        from agent_apis.neural import quick_evaluation as q
        from agent_apis.experiment import ROOT,digest
        with TemporaryDirectory(dir=ROOT/'runs/catan-ai') as tmp:
            folder=Path(tmp);state=folder/'state.json';saved={};configs={};records={}
            for model in q.ORDER:
                for tier in ('smoke','screening'):
                    n=10 if tier=='smoke' else 40
                    stopped=tier=='screening'
                    rows=[result(str(i),c,float(c=='BLUE')) for i in range(n//2) for c in ('BLUE','RED')]
                    path=folder/f'{model}-{tier}'/'run.json';path.parent.mkdir()
                    log=path.parent/'games.jsonl';log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    config={'model':model,'tier':tier};configs[model,tier]=config
                    m={'games':n,'wins':n//2,'losses':n//2,'cutoffs':0,'errors':0,'overall_match_score':.5,
                       'futility':{'stopped':stopped},'failure_gate_passed':True,'cutoff_gate_passed':True,'timing_gate_passed':True}
                    record={'status':'stopped-early' if stopped else 'completed',
                            'configuration':{'sha256':digest(config),'content':config},
                            'metrics':{'benchmark':m,'artifacts':{'games.jsonl':file_digest(log)}}}
                    path.write_text(json.dumps(record));records[model,tier]=path
                    saved[f'{model}:{tier}']={'run_record':str(path.relative_to(ROOT)),'metrics':m,'status':record['status']}
            original={'status':'needs-investigation','error':'missing baseline','scope':'fixture',
                      'schedule':{'smoke_games':10,'total_games':50},'runs':saved,
                      'source_sha256':{'agent_apis/neural/quick_evaluation.py':'original-reporter'}}
            state.write_text(json.dumps(original))
            with patch.object(q,'STATE',state),patch.object(q,'EXPERIMENTS',folder),\
                 patch.object(q.QuickEvaluation,'configuration',side_effect=lambda m,t:configs[m,t]),\
                 patch.object(q.QuickEvaluation,'policy_reference',side_effect=lambda m:(records[m,'screening'],q.games(records[m,'screening']))),\
                 patch.object(q.benchmark,'run',side_effect=AssertionError('Must never play games')),redirect_stdout(io.StringIO()):
                work=q.QuickEvaluation(report_only=True);work.report_saved()
                output=json.loads((folder/'PHASE5_QUICK_RESULTS.json').read_text())
                self.assertEqual(output['reporting_recovery']['games_replayed'],0)
                self.assertEqual(output['source_sha256'],original['source_sha256'])
                self.assertTrue(all(r['search_minus_policy']['ci95']==[None,None] for r in output['architectures'].values()))
                self.assertEqual(json.loads(state.read_text())['status'],'completed')
                original['runs'].pop('MLP:screening');state.write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError,'Missing saved MLP screening'):
                    q.QuickEvaluation(report_only=True).report_saved()

    def test_all_smokes_precede_extensions_and_no_training_is_launched(self):
        from agent_apis.neural import quick_evaluation as q
        from agent_apis.experiment import ROOT
        with TemporaryDirectory(dir=ROOT/'runs/catan-ai') as tmp:
            folder=Path(tmp);calls=[]
            class Fixture(q.QuickEvaluation):
                def __init__(self):self.core={};self.state={'scope':'fixture','schedule':{'smoke_games':10,'total_games':50}}
                def persist(self):pass
                def measure(self,model,tier,prior=None):
                    calls.append((model,tier,prior is not None));path=folder/f'{model}-{tier}'/'run.json';path.parent.mkdir()
                    rows=[{**result(str(i),c,float(c=='BLUE'))} for i in range(5 if tier=='smoke' else 25) for c in ('BLUE','RED')]
                    (path.parent/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    m={'games':len(rows),'wins':len(rows)//2,'losses':len(rows)//2,'cutoffs':0,'errors':0,
                       'overall_match_score':.5,'futility':{'stopped':False},'failure_gate_passed':True,
                       'cutoff_gate_passed':True,'timing_gate_passed':True}
                    record={'status':'completed','metrics':{'benchmark':m}};path.write_text(json.dumps(record));return path,record
                def policy_reference(self,model):
                    path=folder/f'{model}-screening'/'run.json';return path,q.games(path)
            with patch.object(q,'EXPERIMENTS',folder),redirect_stdout(io.StringIO()):Fixture().finish()
            self.assertEqual(calls[:4],[(m,'smoke',False) for m in q.ORDER])
            self.assertEqual(calls[4:],[(m,'screening',True) for m in q.ORDER])
            r=json.loads((folder/'PHASE5_QUICK_RESULTS.json').read_text())
            self.assertEqual(r['training_seeds'],[11]);self.assertFalse(r['final_seeds_used'])
            self.assertTrue(all(v['metrics']['games']==50 for v in r['architectures'].values()))
