"""Serial resumable training queue; consumes only explicitly listed configurations."""
import argparse
import json
from pathlib import Path
import time
import torch
from agent_apis.benchmark import exclusive_benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest
from agent_apis.neural.train import train


def completed(config):
    for path in sorted((ROOT/'runs/catan-ai').glob('phase5-*/run.json')):
        try:record=json.loads(path.read_text())
        except json.JSONDecodeError:continue
        if record['configuration']['sha256']==digest(config) and record['status']=='completed':
            if file_digest(ROOT/record['checkpoint']['path'])!=record['checkpoint']['sha256']:
                raise ValueError('Completed checkpoint changed')
            return path,record
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('configs',nargs='+',type=Path);args=parser.parse_args()
    for config_path in args.configs:
        config=json.loads(config_path.read_text())
        while True:
            if completed(config):break
            try:
                with exclusive_benchmark():
                    if completed(config):break
                    resumable=[]
                    for path in (ROOT/'runs/catan-ai').glob('phase5-*'):
                        if (path/'configuration.json').exists() and (path/'resume.pt').exists() and not (path/'run.json').exists():
                            if digest(json.loads((path/'configuration.json').read_text()))==digest(config):resumable.append(path)
                    train(config,args.cache,ROOT/'runs/catan-ai',max(resumable) if resumable else None)
                break
            except RuntimeError as error:
                if 'Another benchmark is running' not in str(error):raise
                time.sleep(5)
        index={}
        for path in sorted((ROOT/'runs/catan-ai').glob('phase5-*/run.json')):
            record=json.loads(path.read_text());content=record['configuration']['content']
            if content['stage']=='supervised-training' and record['status']=='completed':
                index[f'{content["model"]["id"]}:{content["seed"]}']={
                    'run_record':str(path.relative_to(ROOT)),'checkpoint':record['checkpoint'],
                    'parameters':record['metrics']['training']['parameters'],
                    'best_step':record['metrics']['training']['best_step'],
                    'best_validation_loss':record['metrics']['training']['best_validation_loss'],
                    'test':record['metrics']['training']['test'],'training_seconds':record['elapsed_seconds']}
        (EXPERIMENTS/'phase5-training-index.json').write_text(json.dumps(index,indent=2)+'\n')
        print(f'Queue completed {config["model"]["id"]} seed {config["seed"]}',flush=True)
        if torch.backends.mps.is_available():torch.mps.empty_cache()


if __name__=='__main__':main()
