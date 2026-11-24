"""Content-addressed exports and the production native worker build."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import torch
from agent_apis.experiment import ROOT,digest,file_digest
from agent_apis.observation import require
from agent_apis.neural.pipeline import write

NATIVE=Path(__file__).with_name('native_worker')
CACHE=ROOT/'runs/catan-ai/native-assets'
_EXPORT_LOCK=threading.RLock()

def source_files():
    return [NATIVE/'Cargo.toml',NATIVE/'Cargo.lock',NATIVE/'build.rs',NATIVE/'build.sh',*sorted((NATIVE/'src').glob('*.rs')),*sorted((NATIVE/'native').glob('*'))]

def source_identity():
    return {str(p.relative_to(ROOT)):file_digest(p) for p in source_files()}

def ensure_binary():
    binary=NATIVE/'target/release/catan-native-worker'
    manifest=NATIVE/'target/build-manifest.json'
    (NATIVE/'target').mkdir(exist_ok=True)
    with (NATIVE/'target/build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        sources=source_identity()
        if binary.exists() and manifest.exists():
            saved=json.loads(manifest.read_text())
            if saved['sources']==sources and saved['binary_sha256']==file_digest(binary):return binary
        subprocess.run([str(NATIVE/'build.sh')],check=True)
        write(manifest,{'sources':source_identity(),'binary_sha256':file_digest(binary)})
    return binary


def export_checkpoint(ref):
    with _EXPORT_LOCK,torch.random.fork_rng(devices=[]):
        return _export_checkpoint(ref)


def _export_checkpoint(ref):
    require(file_digest(ROOT/ref['path'])==ref['sha256'],'Native checkpoint changed')
    sources={str(p.relative_to(ROOT)):file_digest(p) for p in [Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('topology.py'),Path(__file__).with_name('value_ablation_models.py')]}
    identity={'checkpoint':ref,'sources':sources,'torch':torch.__version__,'opset':18,'batch':1,'dtype':'float32','features':681,'actions':332}
    folder=CACHE/digest(identity);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        output=folder/'model.onnx';manifest=folder/'manifest.json'
        if manifest.exists():
            saved=json.loads(manifest.read_text());require(saved['identity']==identity and output.exists() and file_digest(output)==saved['onnx_sha256'],'Native model export changed');return output
        saved=torch.load(ROOT/ref['path'],map_location='cpu',weights_only=True)
        if saved['metadata'].get('model_id')=='TYPED-EXPHORMER' or saved.get('variant')=='typed-exphormer':
            from agent_apis.neural.value_ablation_models import TypedExphormer
            expected=saved['metadata'].get('sparse_implementation_sha256')
            if expected:require(expected==sources[str(Path(__file__).with_name('value_ablation_models.py').relative_to(ROOT))],'Sparse architecture changed')
            from agent_apis.neural.topology import topology
            meta=saved['metadata']
            require(meta['feature_schema_version']==3 and meta['action_count']==332 and meta['topology_sha256']==topology()['sha256'],'Sparse tensor/feature schema changed')
            require(meta['feature_schema_sha256']==file_digest(ROOT/'experiments/catan-ai/feature-schema-v3.json') and meta['model_implementation_sha256']==file_digest(Path(__file__).with_name('models.py')),'Sparse feature/model implementation changed')
            model=TypedExphormer();model.load_state_dict(saved['state_dict'],strict=True);model.eval()
        else:
            from agent_apis.neural.provider import NeuralCheckpoint
            previous_threads=torch.get_num_threads()
            try:model=NeuralCheckpoint(ROOT/ref['path'],device='cpu').model.eval()
            finally:torch.set_num_threads(previous_threads)
        import onnx
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            temporary=folder/'model.tmp.onnx'
            with torch.inference_mode():
                torch.onnx.export(model,(torch.zeros(1,681),),str(temporary),input_names=['features'],output_names=['logits','value_logit'],opset_version=18,dynamo=False)
            onnx.checker.check_model(str(temporary));temporary.replace(output)
            write(manifest,{'identity':identity,'onnx_sha256':file_digest(output)})
        finally:torch.set_num_threads(previous)
    return output


def runtime_environment():
    env=dict(os.environ)
    library=NATIVE/'runtime/libonnxruntime.1.32.0.dylib'
    require('ORT_DYLIB_PATH' in env or library.exists(),'Install ONNX Runtime and set ORT_DYLIB_PATH')
    env.setdefault('ORT_DYLIB_PATH',str(library))
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env
