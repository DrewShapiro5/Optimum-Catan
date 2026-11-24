"""Content-addressed exports and the production native worker build."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import torch
from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.observation import require
from agent_apis.neural.device_utils import file_lock
from agent_apis.neural.pipeline import write

NATIVE = Path(__file__).with_name('native_worker')
CACHE = ROOT / 'runs/catan-ai/native-assets'
_EXPORT_LOCK = threading.RLock()


def source_files():
    return [
        NATIVE / 'Cargo.toml',
        NATIVE / 'Cargo.lock',
        NATIVE / 'build.rs',
        NATIVE / 'build.sh',
        *sorted((NATIVE / 'src').glob('*.rs')),
        *sorted((NATIVE / 'native').glob('*')),
    ]


def source_identity():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): file_digest(p) for p in source_files()}


def _cargo_executable():
    found = shutil.which('cargo')
    if found:
        return found
    cargo_home = Path.home() / '.cargo' / 'bin' / ('cargo.exe' if sys.platform == 'win32' else 'cargo')
    if cargo_home.exists():
        return str(cargo_home)
    raise RuntimeError('Rust `cargo` executable not found on PATH or in ~/.cargo/bin')


def ensure_binary():
    exe_name = 'catan-native-worker.exe' if sys.platform == 'win32' else 'catan-native-worker'
    binary = NATIVE / 'target/release' / exe_name
    manifest = NATIVE / 'target/build-manifest.json'
    (NATIVE / 'target').mkdir(exist_ok=True)
    with (NATIVE / 'target/build.lock').open('a') as lock:
        with file_lock(lock):
            sources = source_identity()
            if binary.exists() and manifest.exists():
                saved = json.loads(manifest.read_text())
                if saved.get('sources') == sources and saved.get('binary_sha256') == file_digest(binary):
                    return binary
            env = dict(os.environ)
            cargo_bin = str(Path.home() / '.cargo' / 'bin')
            if cargo_bin not in env.get('PATH', ''):
                env['PATH'] = cargo_bin + os.pathsep + env.get('PATH', '')
            subprocess.run(
                [_cargo_executable(), 'build', '--release', '--manifest-path', str(NATIVE / 'Cargo.toml')],
                check=True,
                env=env,
            )
            write(manifest, {'sources': source_identity(), 'binary_sha256': file_digest(binary)})
    return binary


def export_checkpoint(ref):
    with _EXPORT_LOCK, torch.random.fork_rng(devices=[]):
        return _export_checkpoint(ref)


def _export_checkpoint(ref):
    require(file_digest(ROOT / ref['path']) == ref['sha256'], 'Native checkpoint changed')
    sources = {
        str(p.relative_to(ROOT)).replace('\\', '/'): file_digest(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name('models.py'),
            Path(__file__).with_name('topology.py'),
            Path(__file__).with_name('value_ablation_models.py'),
        ]
    }
    identity = {
        'checkpoint': ref,
        'sources': sources,
        'torch': torch.__version__,
        'opset': 18,
        'batch': 1,
        'dtype': 'float32',
        'features': 681,
        'actions': 332,
    }
    folder = CACHE / digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'export.lock').open('a') as lock:
        with file_lock(lock):
            output = folder / 'model.onnx'
            manifest = folder / 'manifest.json'
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                require(
                    saved['identity'] == identity
                    and output.exists()
                    and file_digest(output) == saved['onnx_sha256'],
                    'Native model export changed',
                )
                return output
            saved = torch.load(ROOT / ref['path'], map_location='cpu', weights_only=True)
            if saved['metadata'].get('model_id') == 'TYPED-EXPHORMER' or saved.get('variant') == 'typed-exphormer':
                from agent_apis.neural.value_ablation_models import TypedExphormer
                expected = saved['metadata'].get('sparse_implementation_sha256')
                if expected:
                    require(
                        expected == sources[str(Path(__file__).with_name('value_ablation_models.py').relative_to(ROOT)).replace('\\', '/')],
                        'Sparse architecture changed',
                    )
                from agent_apis.neural.topology import topology
                meta = saved['metadata']
                require(
                    meta['feature_schema_version'] == 3
                    and meta['action_count'] == 332
                    and meta['topology_sha256'] == topology()['sha256'],
                    'Sparse tensor/feature schema changed',
                )
                require(
                    meta['feature_schema_sha256'] == file_digest(ROOT / 'experiments/catan-ai/feature-schema-v3.json')
                    and meta['model_implementation_sha256'] == file_digest(Path(__file__).with_name('models.py')),
                    'Sparse feature/model implementation changed',
                )
                model = TypedExphormer()
                model.load_state_dict(saved['state_dict'], strict=True)
                model.eval()
            else:
                from agent_apis.neural.provider import NeuralCheckpoint
                previous_threads = torch.get_num_threads()
                try:
                    model = NeuralCheckpoint(ROOT / ref['path'], device='cpu').model.eval()
                finally:
                    torch.set_num_threads(previous_threads)
            import onnx
            previous = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                temporary = folder / 'model.tmp.onnx'
                with torch.inference_mode():
                    torch.onnx.export(
                        model,
                        (torch.zeros(1, 681),),
                        str(temporary),
                        input_names=['features'],
                        output_names=['logits', 'value_logit'],
                        opset_version=18,
                        dynamo=False,
                    )
                onnx.checker.check_model(str(temporary))
                temporary.replace(output)
                write(manifest, {'identity': identity, 'onnx_sha256': file_digest(output)})
            finally:
                torch.set_num_threads(previous)
    return output


def _discover_onnxruntime_library():
    """Find the platform-appropriate ONNX Runtime shared library."""
    runtime_dir = NATIVE / 'runtime'
    if sys.platform == 'darwin':
        pinned = runtime_dir / 'libonnxruntime.1.32.0.dylib'
        if pinned.exists():
            return pinned
        if runtime_dir.is_dir():
            matches = sorted(runtime_dir.glob('libonnxruntime*.dylib'))
            if matches:
                return matches[-1]
    elif sys.platform == 'win32':
        local_dll = runtime_dir / 'onnxruntime.dll'
        if local_dll.exists():
            return local_dll
        try:
            import onnxruntime as ort_pkg
            capi = Path(ort_pkg.__file__).resolve().parent / 'capi' / 'onnxruntime.dll'
            if capi.exists():
                return capi
        except ImportError:
            pass
    else:
        if runtime_dir.is_dir():
            matches = sorted(runtime_dir.glob('libonnxruntime.so*'))
            if matches:
                return matches[-1]
        try:
            import onnxruntime as ort_pkg
            capi_dir = Path(ort_pkg.__file__).resolve().parent / 'capi'
            matches = sorted(capi_dir.glob('libonnxruntime.so*'))
            if matches:
                return matches[-1]
        except ImportError:
            pass
    return None


def runtime_environment():
    env = dict(os.environ)
    if 'ORT_DYLIB_PATH' not in env:
        library = _discover_onnxruntime_library()
        require(
            library is not None and library.exists(),
            'ONNX Runtime shared library not found. Install `onnxruntime-gpu` (Windows/CUDA) '
            'or place the library in native_worker/runtime/ or set ORT_DYLIB_PATH.',
        )
        env['ORT_DYLIB_PATH'] = str(library)
    if sys.platform == 'win32':
        # Prepend ORT capi dir and PyTorch CUDA DLL dirs so onnxruntime_providers_cuda.dll resolves cuBLAS/cuDNN.
        extra_dirs = []
        ort_dir = str(Path(env['ORT_DYLIB_PATH']).resolve().parent)
        extra_dirs.append(ort_dir)
        torch_lib = Path(torch.__file__).resolve().parent / 'lib'
        if torch_lib.is_dir():
            extra_dirs.append(str(torch_lib))
        site_packages = Path(torch.__file__).resolve().parent.parent
        nvidia_dir = site_packages / 'nvidia'
        if nvidia_dir.is_dir():
            for bin_dir in sorted(nvidia_dir.glob('*/bin')):
                extra_dirs.append(str(bin_dir))
        current_path = env.get('PATH', '')
        prefix = os.pathsep.join(d for d in extra_dirs if d not in current_path)
        if prefix:
            env['PATH'] = prefix + os.pathsep + current_path
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env
