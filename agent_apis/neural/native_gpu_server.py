"""High-throughput binary Unix-domain socket GPU batching server for native Rust workers.

Mirrors the batching contract of `agent_apis.neural.shared_inference`:
- Bounded batch size (`max_batch_rows`, default 64) and collection window (`wait_seconds`, default 0.5 ms).
- Checkpoint-separated batching via `model_id` in each binary request header.
- Zero Python game-state projection: Rust workers send raw `[f32; 681]` feature vectors
  and receive raw `[f32; 332]` policy logits + `f32` value logit.
"""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import selectors
import socket
import struct
import threading
import time
import traceback

import numpy as np
import torch

from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.observation import require
from agent_apis.neural.native_assets import CACHE, _EXPORT_LOCK
from agent_apis.neural.pipeline import write
from agent_apis.neural.provider import NeuralCheckpoint

FEATURES_DIM = 681
ACTIONS_DIM = 332
MAGIC = 0x4341544E  # "CATN"
REQ_HEADER = struct.Struct('<IIII')
RESP_HEADER = struct.Struct('<IIf')
REQ_BYTES = REQ_HEADER.size + FEATURES_DIM * 4
RESP_BYTES = RESP_HEADER.size + ACTIONS_DIM * 4


def load_inference_model(checkpoint_path, device='cpu'):
    """Load either a standard NeuralCheckpoint (XD-CNN/MLP/H-GNN/Transformer) or TypedExphormer."""
    checkpoint_path = Path(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    meta = saved.get('metadata', {})
    if meta.get('model_id') == 'TYPED-EXPHORMER' or saved.get('variant') == 'typed-exphormer':
        from agent_apis.neural.value_ablation_models import TypedExphormer
        model = TypedExphormer()
        model.load_state_dict(saved['state_dict'], strict=True)
        return model.to(device).eval()
    prev_threads = torch.get_num_threads()
    try:
        return NeuralCheckpoint(checkpoint_path, device=device).model.eval()
    finally:
        torch.set_num_threads(prev_threads)


def _dynamic_export_wrapper(model):
    """Wrap XDCNN with symbolic `raw.shape[0]` so legacy ONNX export preserves dynamic batch size.

    Leaves `agent_apis/neural/models.py` untouched so `model_implementation_sha256` stays valid.
    """
    from agent_apis.neural.models import XDCNN
    if not isinstance(model, XDCNN):
        return model

    import torch.nn.functional as F

    class _DynamicXDCNN(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, raw):
            inp = self.base.inputs
            batch = raw.shape[0]
            symmetry = torch.zeros((batch,), dtype=torch.long, device=raw.device)
            transformed = raw.gather(1, inp.feature_inverse[symmetry])
            ports = raw[:, inp.port_indices]
            edge_ports = torch.zeros((batch, 72, 6), dtype=raw.dtype, device=raw.device)
            edge_ports[:, inp.port_edges] = ports
            edge_ports = edge_ports.gather(1, inp.edge_inverse[symmetry, :, None].expand(-1, -1, 6))
            x = transformed / inp.scales
            dice = F.one_hot(transformed[:, inp.dice_indices].long(), 13).to(raw.dtype)
            tile = torch.cat((x[:, inp.tile_indices], dice), -1)
            node = torch.cat((x[:, inp.node_indices], torch.matmul(inp.node_road, edge_ports)), -1)
            road = torch.cat((x[:, inp.edge_indices], edge_ports), -1)
            ids = inp.player_indices
            player = torch.cat(
                (
                    x[:, ids.clamp_min(0)] * (ids >= 0),
                    (ids >= 0).to(raw.dtype)[None].expand(batch, -1, -1),
                ),
                -1,
            )
            glob = x[:, inp.global_indices]

            planes = torch.zeros((batch, self.base.h * self.base.w, 60), dtype=raw.dtype, device=raw.device)
            planes[:, inp.tile_cell, :21] = tile
            planes[:, inp.tile_cell, 55] = 1
            planes[:, inp.node_cell, 21:31] = node
            planes[:, inp.node_cell, 56] = 1
            for ori in range(3):
                selected = getattr(inp, f'road_group{ori}')
                cells = inp.road_cell[selected]
                planes[:, cells, 31 + 8 * ori : 39 + 8 * ori] = road[:, selected]
                planes[:, cells, 57 + ori] = 1
            spatial = self.base.initial(planes.transpose(1, 2).reshape(batch, 60, self.base.h, self.base.w)) * self.base.valid
            scalar = self.base.scalar(torch.cat((glob, player.flatten(1)), 1))
            for block in self.base.blocks:
                spatial, scalar = block(spatial, scalar, self.base.valid)
            flat = spatial.flatten(2).transpose(1, 2)
            tile_tok = self.base.entity[0](flat[:, inp.tile_cell])
            node_tok = self.base.entity[1](flat[:, inp.node_cell])
            road_tok = torch.zeros((batch, 72, 128), dtype=flat.dtype, device=flat.device)
            for ori in range(3):
                selected = getattr(inp, f'road_group{ori}')
                road_tok[:, selected] = self.base.entity[2 + ori](flat[:, inp.road_cell[selected]])
            return self.base.heads((tile_tok, node_tok, road_tok, self.base.players(player), scalar))

    return _DynamicXDCNN(model).eval()


def _coreml_static_wrapper(model, batch_size):
    """Scatter-free static-batch wrapper for XDCNN and TypedExphormer so CoreML EP fuses cleanly."""
    from agent_apis.neural.models import XDCNN
    from agent_apis.neural.value_ablation_models import TypedExphormer
    if isinstance(model, TypedExphormer):
        return _coreml_static_exphormer(model, batch_size)
    if not isinstance(model, XDCNN):
        return model

    class _CoreMLStaticXDCNN(torch.nn.Module):
        def __init__(self, base, b):
            super().__init__()
            self.base = base
            self.b = b
            inp = base.inputs
            hw = base.h * base.w

            port_scatter = torch.zeros((72, len(inp.port_edges)), dtype=torch.float32)
            for j, edge_idx in enumerate(inp.port_edges.tolist()):
                port_scatter[edge_idx, j] = 1.0
            self.register_buffer('port_scatter', port_scatter, persistent=False)

            self.register_buffer('dice_bins', torch.arange(13, dtype=torch.float32), persistent=False)

            ids = inp.player_indices
            self.register_buffer('player_clamped', ids.clamp_min(0), persistent=False)
            pmask = (ids >= 0).to(torch.float32)
            self.register_buffer('player_mask', pmask, persistent=False)
            self.register_buffer('player_mask_b', pmask[None].expand(b, -1, -1).clone(), persistent=False)

            tile_scatter = torch.zeros((hw, len(inp.tile_cell)), dtype=torch.float32)
            for j, cell in enumerate(inp.tile_cell.tolist()):
                tile_scatter[cell, j] = 1.0
            self.register_buffer('tile_scatter', tile_scatter, persistent=False)

            node_scatter = torch.zeros((hw, len(inp.node_cell)), dtype=torch.float32)
            for j, cell in enumerate(inp.node_cell.tolist()):
                node_scatter[cell, j] = 1.0
            self.register_buffer('node_scatter', node_scatter, persistent=False)

            flags = torch.zeros((hw, 5), dtype=torch.float32)
            flags[inp.tile_cell, 0] = 1.0
            flags[inp.node_cell, 1] = 1.0
            for ori in range(3):
                selected = getattr(inp, f'road_group{ori}')
                cells = inp.road_cell[selected]
                flags[cells, 2 + ori] = 1.0
                rs = torch.zeros((hw, len(selected)), dtype=torch.float32)
                for j, cell in enumerate(cells.tolist()):
                    rs[cell, j] = 1.0
                self.register_buffer(f'road_scatter{ori}', rs, persistent=False)
                ru = torch.zeros((72, len(selected)), dtype=torch.float32)
                for j, ridx in enumerate(selected.tolist()):
                    ru[ridx, j] = 1.0
                self.register_buffer(f'road_unscatter{ori}', ru, persistent=False)
            self.register_buffer('flags_b', flags[None].expand(b, -1, -1).clone(), persistent=False)

        def forward(self, raw):
            inp = self.base.inputs
            ports = raw[:, inp.port_indices]
            edge_ports = torch.matmul(self.port_scatter, ports)
            x = raw / inp.scales
            dice = (raw[:, inp.dice_indices].unsqueeze(-1) == self.dice_bins).to(raw.dtype)
            tile = torch.cat((x[:, inp.tile_indices], dice), -1)
            node = torch.cat((x[:, inp.node_indices], torch.matmul(inp.node_road, edge_ports)), -1)
            road = torch.cat((x[:, inp.edge_indices], edge_ports), -1)
            player = torch.cat((x[:, self.player_clamped] * self.player_mask, self.player_mask_b), -1)
            glob = x[:, inp.global_indices]

            tile_p = torch.matmul(self.tile_scatter, tile)
            node_p = torch.matmul(self.node_scatter, node)
            r0 = torch.matmul(self.road_scatter0, road[:, inp.road_group0])
            r1 = torch.matmul(self.road_scatter1, road[:, inp.road_group1])
            r2 = torch.matmul(self.road_scatter2, road[:, inp.road_group2])
            planes = torch.cat((tile_p, node_p, r0, r1, r2, self.flags_b), -1)

            spatial = (
                self.base.initial(planes.transpose(1, 2).reshape(self.b, 60, self.base.h, self.base.w))
                * self.base.valid
            )
            scalar = self.base.scalar(torch.cat((glob, player.flatten(1)), 1))
            for block in self.base.blocks:
                spatial, scalar = block(spatial, scalar, self.base.valid)
            flat = spatial.flatten(2).transpose(1, 2)
            tile_tok = self.base.entity[0](flat[:, inp.tile_cell])
            node_tok = self.base.entity[1](flat[:, inp.node_cell])
            road_tok = (
                torch.matmul(self.road_unscatter0, self.base.entity[2](flat[:, inp.road_cell[inp.road_group0]]))
                + torch.matmul(self.road_unscatter1, self.base.entity[3](flat[:, inp.road_cell[inp.road_group1]]))
                + torch.matmul(self.road_unscatter2, self.base.entity[4](flat[:, inp.road_cell[inp.road_group2]]))
            )
            return self.base.heads((tile_tok, node_tok, road_tok, self.base.players(player), scalar))

    return _CoreMLStaticXDCNN(model, batch_size).eval()


def _coreml_static_exphormer(model, batch_size):
    """Scatter-free static-batch TypedExphormer wrapper (identical dense 152x152 attention)."""
    import torch.nn.functional as F

    class _CoreMLStaticTypedExphormer(torch.nn.Module):
        def __init__(self, base, b):
            super().__init__()
            self.base = base
            self.b = b
            inp = base.inputs

            port_scatter = torch.zeros((72, len(inp.port_edges)), dtype=torch.float32)
            for j, edge_idx in enumerate(inp.port_edges.tolist()):
                port_scatter[edge_idx, j] = 1.0
            self.register_buffer('port_scatter', port_scatter, persistent=False)
            self.register_buffer('dice_bins', torch.arange(13, dtype=torch.float32), persistent=False)

            ids = inp.player_indices
            self.register_buffer('player_clamped', ids.clamp_min(0), persistent=False)
            pmask = (ids >= 0).to(torch.float32)
            self.register_buffer('player_mask', pmask, persistent=False)
            self.register_buffer('player_mask_b', pmask[None].expand(b, -1, -1).clone(), persistent=False)
            self.register_buffer('connectors_b', base.connectors.detach()[None].expand(b, -1, -1).clone(), persistent=False)

            with torch.no_grad():
                eb = base.edge_bias.weight.detach()  # [15, 4]
                base_edge_b = eb[base.edge_types].permute(2, 0, 1)[None]  # [1, 4, 152, 152]
                dr = (base.distance(inp.distance) + base.relation(inp.relation)).permute(2, 0, 1)[None]  # [1, 4, 148, 148]
                base_bias = base_edge_b + F.pad(dr, (0, 4, 0, 4))
                self.register_buffer('base_bias', base_bias, persistent=False)
                self.register_buffer('base_blocked', (base.edge_types == 0)[None, None], persistent=False)
                self.register_buffer('neg_inf', torch.tensor(-1e9, dtype=torch.float32), persistent=False)

                self.register_buffer('delta_on', eb[10].view(1, 4, 1, 1) - base_edge_b[:, :, 145:147, 19:73], persistent=False)
                self.register_buffer('delta_on_t', eb[11].view(1, 4, 1, 1) - base_edge_b[:, :, 19:73, 145:147], persistent=False)
                self.register_buffer('delta_or', eb[12].view(1, 4, 1, 1) - base_edge_b[:, :, 145:147, 73:145], persistent=False)
                self.register_buffer('delta_or_t', eb[13].view(1, 4, 1, 1) - base_edge_b[:, :, 73:145, 145:147], persistent=False)
                self.register_buffer('delta_port', eb[14].view(1, 4, 1, 1) - base_edge_b[:, :, 19:73, 19:73], persistent=False)
                self.register_buffer('own_bias', base.ownership_bias.detach().view(1, 4, 1, 1), persistent=False)
                self.register_buffer('p_bias', base.port_bias.detach().view(1, 4, 1, 1), persistent=False)

        def forward(self, raw):
            inp = self.base.inputs
            ports = raw[:, inp.port_indices]
            edge_ports = torch.matmul(self.port_scatter, ports)
            x = raw / inp.scales
            dice = (raw[:, inp.dice_indices].unsqueeze(-1) == self.dice_bins).to(raw.dtype)
            tile = torch.cat((x[:, inp.tile_indices], dice), -1)
            node = torch.cat((x[:, inp.node_indices], torch.matmul(inp.node_road, edge_ports)), -1)
            road = torch.cat((x[:, inp.edge_indices], edge_ports), -1)
            player = torch.cat((x[:, self.player_clamped] * self.player_mask, self.player_mask_b), -1)
            glob = x[:, inp.global_indices]

            owner_node = torch.stack((node[:, :, 0] + node[:, :, 1], node[:, :, 2] + node[:, :, 3]), 1)
            owner_road = road[:, :, :2].transpose(1, 2)
            flags = edge_ports.sum(-1)
            port_adj = torch.matmul(inp.edge_a.T, inp.edge_b[None] * flags[:, :, None])
            port_adj = port_adj + port_adj.transpose(1, 2)

            states = self.base.embed({'tile': tile, 'node': node, 'road': road, 'player': player, 'global': glob})
            entities = torch.cat((*states[:4], states[4][:, None], self.connectors_b), 1)

            on_m = (owner_node != 0).to(raw.dtype)[:, None]
            on_mt = on_m.transpose(2, 3)
            or_m = (owner_road != 0).to(raw.dtype)[:, None]
            or_mt = or_m.transpose(2, 3)
            pa_m = (port_adj != 0).to(raw.dtype)[:, None]

            on_b = on_m * self.delta_on + owner_node[:, None] * self.own_bias
            on_bt = on_mt * self.delta_on_t + owner_node.transpose(1, 2)[:, None] * self.own_bias
            or_b = or_m * self.delta_or + owner_road[:, None] * self.own_bias
            or_bt = or_mt * self.delta_or_t + owner_road.transpose(1, 2)[:, None] * self.own_bias
            pa_b = pa_m * self.delta_port + port_adj[:, None] * self.p_bias

            bias = (
                self.base_bias
                + F.pad(on_b, (19, 152 - 73, 145, 152 - 147))
                + F.pad(on_bt, (145, 152 - 147, 19, 152 - 73))
                + F.pad(or_b, (73, 152 - 145, 145, 152 - 147))
                + F.pad(or_bt, (145, 152 - 147, 73, 152 - 145))
                + F.pad(pa_b, (19, 152 - 73, 19, 152 - 73))
            )
            dyn_unblock = (
                F.pad(on_m, (19, 152 - 73, 145, 152 - 147))
                + F.pad(on_mt, (145, 152 - 147, 19, 152 - 73))
                + F.pad(or_m, (73, 152 - 145, 145, 152 - 147))
                + F.pad(or_mt, (145, 152 - 147, 73, 152 - 145))
                + F.pad(pa_m, (19, 152 - 73, 19, 152 - 73))
            )
            blocked = self.base_blocked & (dyn_unblock == 0)
            bias = torch.where(blocked, self.neg_inf, bias)

            for block in self.base.blocks:
                entities = block(entities, bias)
            return self.base.heads((entities[:, :19], entities[:, 19:73], entities[:, 73:145], entities[:, 145:147], entities[:, 147]))

    return _CoreMLStaticTypedExphormer(model, batch_size).eval()


def export_checkpoint_coreml_static(ref, batch_size=32):
    """Export a Scatter-free static-batch ONNX model optimized for Apple CoreML EP."""
    with _EXPORT_LOCK, torch.random.fork_rng(devices=[]):
        require(file_digest(ROOT / ref['path']) == ref['sha256'], 'Native checkpoint changed')
        assets_dir = Path(__file__).parent
        sources = {
            str(p.relative_to(ROOT)): file_digest(p)
            for p in [
                assets_dir / 'native_assets.py',
                assets_dir / 'models.py',
                assets_dir / 'topology.py',
                assets_dir / 'value_ablation_models.py',
            ]
        }
        identity = {
            'checkpoint': ref,
            'sources': sources,
            'torch': torch.__version__,
            'opset': 18,
            'batch': f'coreml-static-{batch_size}-v3',
            'dtype': 'float32',
            'features': FEATURES_DIM,
            'actions': ACTIONS_DIM,
        }
        folder = CACHE / digest(identity)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            output = folder / 'model.onnx'
            manifest = folder / 'manifest.json'
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                require(
                    saved['identity'] == identity
                    and output.exists()
                    and file_digest(output) == saved['onnx_sha256'],
                    'CoreML static model export changed',
                )
                return output
            model = load_inference_model(ROOT / ref['path'], device='cpu')
            export_model = _coreml_static_wrapper(model, batch_size)
            import onnx
            previous = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                temporary = folder / 'model.tmp.onnx'
                with torch.inference_mode():
                    torch.onnx.export(
                        export_model,
                        (torch.zeros(batch_size, FEATURES_DIM),),
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


def export_checkpoint_dynamic(ref):
    """Export a dynamic-batch ONNX model for pure-Rust CoreML/ONNX batched inference."""
    with _EXPORT_LOCK, torch.random.fork_rng(devices=[]):
        require(file_digest(ROOT / ref['path']) == ref['sha256'], 'Native checkpoint changed')
        assets_dir = Path(__file__).parent
        sources = {
            str(p.relative_to(ROOT)): file_digest(p)
            for p in [
                assets_dir / 'native_assets.py',
                assets_dir / 'models.py',
                assets_dir / 'topology.py',
                assets_dir / 'value_ablation_models.py',
            ]
        }
        identity = {
            'checkpoint': ref,
            'sources': sources,
            'torch': torch.__version__,
            'opset': 18,
            'batch': 'dynamic-v2',
            'dtype': 'float32',
            'features': FEATURES_DIM,
            'actions': ACTIONS_DIM,
        }
        folder = CACHE / digest(identity)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            output = folder / 'model.onnx'
            manifest = folder / 'manifest.json'
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                require(
                    saved['identity'] == identity
                    and output.exists()
                    and file_digest(output) == saved['onnx_sha256'],
                    'Dynamic native model export changed',
                )
                return output
            saved = torch.load(ROOT / ref['path'], map_location='cpu', weights_only=True)
            if saved['metadata'].get('model_id') == 'TYPED-EXPHORMER' or saved.get('variant') == 'typed-exphormer':
                from agent_apis.neural.value_ablation_models import TypedExphormer
                model = TypedExphormer()
                model.load_state_dict(saved['state_dict'], strict=True)
                model.eval()
            else:
                prev_threads = torch.get_num_threads()
                try:
                    model = NeuralCheckpoint(ROOT / ref['path'], device='cpu').model.eval()
                finally:
                    torch.set_num_threads(prev_threads)
            export_model = _dynamic_export_wrapper(model)
            import onnx
            previous = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                temporary = folder / 'model.tmp.onnx'
                with torch.inference_mode():
                    torch.onnx.export(
                        export_model,
                        (torch.zeros(2, FEATURES_DIM),),
                        str(temporary),
                        input_names=['features'],
                        output_names=['logits', 'value_logit'],
                        dynamic_axes={
                            'features': {0: 'batch'},
                            'logits': {0: 'batch'},
                            'value_logit': {0: 'batch'},
                        },
                        opset_version=18,
                        dynamo=False,
                    )
                onnx.checker.check_model(str(temporary))
                temporary.replace(output)
                write(manifest, {'identity': identity, 'onnx_sha256': file_digest(output)})
            finally:
                torch.set_num_threads(previous)
        return output


class BinaryGpuBatchServer:
    """Single-thread kqueue/select Unix-socket server batching Rust worker inference on MPS/CPU."""

    def __init__(self, socket_path, model_refs, device='mps', max_batch_rows=64, wait_seconds=0.0005):
        self.socket_path = Path(socket_path)
        self.model_refs = dict(model_refs)  # {int model_id: ref_dict}
        self.device = device
        self.max_batch_rows = max_batch_rows
        self.wait_seconds = wait_seconds
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._error = None
        self._thread = None
        self.metrics = {}

    def start(self):
        self._thread = threading.Thread(target=self._serve_loop, name='native-gpu-server', daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=120):
            raise RuntimeError('Timed out starting native GPU batch server')
        if self._error is not None:
            raise RuntimeError(f'Native GPU batch server failed to start:\n{self._error}')
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        if self._error is not None:
            raise RuntimeError(f'Native GPU batch server error:\n{self._error}')
        return self.metrics

    def _serve_loop(self):
        server = None
        sel = selectors.DefaultSelector()
        buffers = {}
        try:
            if self.device == 'mps':
                require(torch.backends.mps.is_available(), 'PyTorch MPS GPU device is not available')
            torch.set_num_threads(1)
            models = {}
            for model_id, ref in self.model_refs.items():
                require(file_digest(ROOT / ref['path']) == ref['sha256'], 'GPU server checkpoint changed')
                models[int(model_id)] = load_inference_model(ROOT / ref['path'], device=self.device)
            torch.set_num_threads(1)

            # Warmup MPS kernels before accepting workers.
            with torch.inference_mode():
                dummy = torch.zeros((4, FEATURES_DIM), dtype=torch.float32, device=self.device)
                for m in models.values():
                    logits, val = m(dummy)
                    _ = logits.cpu().numpy()
                    _ = val.cpu().numpy()

            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self.socket_path.exists():
                self.socket_path.unlink()
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(self.socket_path))
            server.listen(128)
            server.setblocking(False)
            sel.register(server, selectors.EVENT_READ, data='accept')

            self.metrics = {
                int(k): {
                    'requests': 0,
                    'rows': 0,
                    'batches': 0,
                    'forward_seconds': 0.0,
                    'queue_window_seconds': 0.0,
                    'batch_rows_histogram': Counter(),
                }
                for k in models
            }
            self._ready_event.set()

            pending = []
            while not self._stop_event.is_set() or buffers or pending:
                timeout = 0.05 if not pending else 0.0
                events = sel.select(timeout=timeout)
                self._drain_events(events, sel, server, buffers, pending)
                if not pending:
                    if self._stop_event.is_set() and not buffers:
                        break
                    continue

                window_start = time.perf_counter()
                deadline = window_start + self.wait_seconds
                while len(pending) < self.max_batch_rows:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    more = sel.select(timeout=remaining)
                    if not more:
                        break
                    self._drain_events(more, sel, server, buffers, pending)
                window_secs = time.perf_counter() - window_start

                batch_items = pending[: self.max_batch_rows]
                del pending[: self.max_batch_rows]

                grouped = {}
                for item in batch_items:
                    grouped.setdefault(item[3], []).append(item)

                for model_id, group in grouped.items():
                    model = models.get(model_id)
                    if model is None:
                        for conn, _, seq, _, _ in group:
                            self._send_error(conn, seq, status=1)
                        continue
                    arr = np.empty((len(group), FEATURES_DIM), dtype=np.float32)
                    for idx, item in enumerate(group):
                        arr[idx] = np.frombuffer(item[4], dtype='<f4', count=FEATURES_DIM)

                    fwd_start = time.perf_counter()
                    with torch.inference_mode():
                        raw = torch.from_numpy(arr).to(self.device)
                        logits_t, vals_t = model(raw)
                        logits_np = np.ascontiguousarray(logits_t.cpu().numpy(), dtype='<f4')
                        vals_np = np.ascontiguousarray(vals_t.cpu().numpy(), dtype='<f4')
                    fwd_secs = time.perf_counter() - fwd_start

                    m = self.metrics[model_id]
                    m['requests'] += len(group)
                    m['rows'] += len(group)
                    m['batches'] += 1
                    m['forward_seconds'] += fwd_secs
                    m['queue_window_seconds'] += window_secs
                    m['batch_rows_histogram'][len(group)] += 1

                    for idx, (conn, _, seq, _, _) in enumerate(group):
                        header = RESP_HEADER.pack(seq, 0, float(vals_np[idx]))
                        payload = logits_np[idx].tobytes()
                        try:
                            conn.sendall(header + payload)
                        except OSError:
                            pass
        except BaseException:
            self._error = traceback.format_exc()
            self._ready_event.set()
        finally:
            for conn in list(buffers):
                try:
                    sel.unregister(conn)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            if server is not None:
                try:
                    sel.unregister(server)
                except Exception:
                    pass
                try:
                    server.close()
                except Exception:
                    pass
            sel.close()
            if self.socket_path.exists():
                try:
                    self.socket_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _send_error(conn, seq, status=1):
        try:
            conn.sendall(RESP_HEADER.pack(seq, status, 0.0) + b'\x00' * (ACTIONS_DIM * 4))
        except OSError:
            pass

    @staticmethod
    def _drain_events(events, sel, server, buffers, pending):
        for key, _ in events:
            if key.data == 'accept':
                while True:
                    try:
                        conn, _ = server.accept()
                    except BlockingIOError:
                        break
                    conn.setblocking(False)
                    buffers[conn] = bytearray()
                    sel.register(conn, selectors.EVENT_READ, data='client')
            else:
                conn = key.fileobj
                buf = buffers.get(conn)
                if buf is None:
                    continue
                try:
                    chunk = conn.recv(65536)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b''
                if not chunk:
                    try:
                        sel.unregister(conn)
                    except Exception:
                        pass
                    buffers.pop(conn, None)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    continue
                buf.extend(chunk)
                while len(buf) >= REQ_BYTES:
                    frame = bytes(buf[:REQ_BYTES])
                    del buf[:REQ_BYTES]
                    magic, worker_id, seq, model_id = REQ_HEADER.unpack_from(frame, 0)
                    if magic != MAGIC:
                        BinaryGpuBatchServer._send_error(conn, seq, status=2)
                        continue
                    pending.append((conn, worker_id, seq, model_id, frame[REQ_HEADER.size:]))
