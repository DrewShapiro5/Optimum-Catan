"""Cross-platform PyTorch device selection, cache/RNG helpers, and file locking."""
from contextlib import contextmanager
import os
import sys
import torch

from agent_apis.observation import require


@contextmanager
def file_lock(file_obj, nonblocking=False):
    """Cross-platform exclusive file lock (fcntl on Unix, msvcrt on Windows)."""
    if sys.platform == 'win32':
        import msvcrt
        mode = msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK
        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), mode, 1)
        except OSError as exc:
            if nonblocking:
                raise BlockingIOError('File is already locked') from exc
            raise
        try:
            yield
        finally:
            try:
                file_obj.seek(0)
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    else:
        import fcntl
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(file_obj, flags)
        try:
            yield
        finally:
            fcntl.flock(file_obj, fcntl.LOCK_UN)


def resolve_torch_device(device='auto'):
    """Resolve 'auto', 'cuda', 'mps', or 'cpu' into a validated PyTorch device string."""
    req = (device or 'auto').lower()
    if req == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if req == 'cuda' or req.startswith('cuda:'):
        require(torch.cuda.is_available(), f'Requested PyTorch CUDA device "{req}" is not available')
        return req
    if req == 'mps':
        require(
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'Requested PyTorch MPS device is not available',
        )
        return 'mps'
    if req == 'cpu':
        return 'cpu'
    raise ValueError(f'Unsupported PyTorch device: {device}')


def configure_torch_device(device):
    """Apply backend-specific performance settings (e.g., TF32 on Ampere+ NVIDIA GPUs)."""
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def empty_device_cache(device):
    """Release cached allocator memory for the active accelerator."""
    dev = str(device)
    if dev.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_device_rng_state(device):
    """Return accelerator RNG state tensor (or None on CPU)."""
    dev = str(device)
    if dev.startswith('cuda') and torch.cuda.is_available():
        return torch.cuda.get_rng_state()
    if dev == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.mps.get_rng_state()
    return None


def set_device_rng_state(device, state):
    """Restore accelerator RNG state tensor if compatible."""
    if state is None:
        return
    dev = str(device)
    if dev.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state(state)
    elif dev == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state)


def default_gpu_backend():
    """Return the default in-process Rust ONNX Runtime execution provider for this host."""
    if torch.cuda.is_available():
        return 'cuda'
    if sys.platform == 'darwin':
        return 'coreml'
    return 'cpu-batch'


def default_socket_device():
    """Return the default PyTorch GPU device for socket-batched inference."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def nonexistent_stop_file():
    """Return a cross-platform path that will never exist (for eval-mode stop_file)."""
    return os.path.join(os.path.abspath(os.sep), 'nonexistent-catan-stop-file')
