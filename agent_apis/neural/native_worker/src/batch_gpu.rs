//! Batched GPU inference backends for native Rust self-play workers.
//!
//! Supports two GPU-batched modes alongside the default single-threaded CPU ONNX path:
//! 1. `RemoteSocket`: Binary Unix-domain socket client sending raw `[f32; 681]` feature
//!    vectors (single or multi-leaf batches) to a shared batched PyTorch MPS GPU server.
//! 2. `InProcessQueue`: In-process bounded Rust queue drained by `streams` concurrent GPU
//!    owner threads running batched ONNX Runtime with Apple CoreML EP (or batched CPU).
use super::*;
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat, CoreMLSpecializationStrategy,
};
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write as IoWrite};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

pub const FEATURES_DIM: usize = 681;
pub const ACTIONS_DIM: usize = 332;
pub const MAGIC: u32 = 0x4341_544E; // "CATN"
const REQ_BYTES: usize = 16 + FEATURES_DIM * 4;
const RESP_BYTES: usize = 12 + ACTIONS_DIM * 4;

pub enum CandidateModel {
    LocalCpu(Session),
    RemoteSocket {
        client: Arc<Mutex<SocketClient>>,
        model_id: u32,
    },
    InProcessQueue {
        client: QueueClient,
        model_path: String,
    },
}

impl CandidateModel {
    pub fn predict(&mut self, features: Vec<f32>) -> Result<(Vec<f32>, f32), String> {
        match self {
            CandidateModel::LocalCpu(s) => predict(s, features).map_err(|e| e.to_string()),
            CandidateModel::RemoteSocket { client, model_id } => client
                .lock()
                .map_err(|_| "GPU socket client mutex poisoned".to_string())?
                .predict(*model_id, &features),
            CandidateModel::InProcessQueue { client, model_path } => {
                client.predict(model_path, features)
            }
        }
    }

    pub fn predict_batch(
        &mut self,
        features_batch: Vec<Vec<f32>>,
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
        if features_batch.is_empty() {
            return Ok(vec![]);
        }
        if features_batch.len() == 1 {
            let one = features_batch.into_iter().next().unwrap();
            return Ok(vec![self.predict(one)?]);
        }
        match self {
            CandidateModel::LocalCpu(s) => features_batch
                .into_iter()
                .map(|f| predict(s, f).map_err(|e| e.to_string()))
                .collect(),
            CandidateModel::RemoteSocket { client, model_id } => client
                .lock()
                .map_err(|_| "GPU socket client mutex poisoned".to_string())?
                .predict_batch(*model_id, &features_batch),
            CandidateModel::InProcessQueue { client, model_path } => {
                client.predict_batch(model_path, features_batch)
            }
        }
    }
}

/// Persistent binary Unix-domain socket client for the shared MPS GPU inference queue.
pub struct SocketClient {
    stream: UnixStream,
    worker_id: u32,
    sequence: u32,
    req_buf: Vec<u8>,
    resp_buf: Vec<u8>,
}

impl SocketClient {
    pub fn connect(socket_path: &str, worker_id: u32) -> Result<Self, String> {
        let stream = UnixStream::connect(socket_path)
            .map_err(|e| format!("Failed to connect to GPU socket {socket_path}: {e}"))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(90)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(90)))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            worker_id,
            sequence: 0,
            req_buf: vec![0u8; REQ_BYTES],
            resp_buf: vec![0u8; RESP_BYTES],
        })
    }

    pub fn predict(&mut self, model_id: u32, features: &[f32]) -> Result<(Vec<f32>, f32), String> {
        if features.len() != FEATURES_DIM {
            return Err(format!(
                "Expected {FEATURES_DIM} features, got {}",
                features.len()
            ));
        }
        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);

        self.req_buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        self.req_buf[4..8].copy_from_slice(&self.worker_id.to_le_bytes());
        self.req_buf[8..12].copy_from_slice(&seq.to_le_bytes());
        self.req_buf[12..16].copy_from_slice(&model_id.to_le_bytes());
        for (i, &val) in features.iter().enumerate() {
            let offset = 16 + i * 4;
            self.req_buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }

        self.stream
            .write_all(&self.req_buf)
            .map_err(|e| format!("GPU socket write failed: {e}"))?;
        self.stream
            .read_exact(&mut self.resp_buf)
            .map_err(|e| format!("GPU socket read failed: {e}"))?;

        Self::parse_response(&self.resp_buf, seq)
    }

    pub fn predict_batch(
        &mut self,
        model_id: u32,
        batch: &[Vec<f32>],
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
        let k = batch.len();
        let mut out_buf = vec![0u8; k * REQ_BYTES];
        let mut seqs = Vec::with_capacity(k);
        for (idx, features) in batch.iter().enumerate() {
            if features.len() != FEATURES_DIM {
                return Err(format!(
                    "Expected {FEATURES_DIM} features, got {}",
                    features.len()
                ));
            }
            let seq = self.sequence;
            self.sequence = self.sequence.wrapping_add(1);
            seqs.push(seq);
            let base = idx * REQ_BYTES;
            out_buf[base..base + 4].copy_from_slice(&MAGIC.to_le_bytes());
            out_buf[base + 4..base + 8].copy_from_slice(&self.worker_id.to_le_bytes());
            out_buf[base + 8..base + 12].copy_from_slice(&seq.to_le_bytes());
            out_buf[base + 12..base + 16].copy_from_slice(&model_id.to_le_bytes());
            for (i, &val) in features.iter().enumerate() {
                let offset = base + 16 + i * 4;
                out_buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        self.stream
            .write_all(&out_buf)
            .map_err(|e| format!("GPU socket batch write failed: {e}"))?;
        let mut in_buf = vec![0u8; k * RESP_BYTES];
        self.stream
            .read_exact(&mut in_buf)
            .map_err(|e| format!("GPU socket batch read failed: {e}"))?;

        let mut results = Vec::with_capacity(k);
        for (idx, seq) in seqs.into_iter().enumerate() {
            let base = idx * RESP_BYTES;
            results.push(Self::parse_response(&in_buf[base..base + RESP_BYTES], seq)?);
        }
        Ok(results)
    }

    fn parse_response(buf: &[u8], expected_seq: u32) -> Result<(Vec<f32>, f32), String> {
        let recv_seq = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let status = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let value_logit = f32::from_le_bytes(buf[8..12].try_into().unwrap());
        if recv_seq != expected_seq {
            return Err(format!(
                "GPU socket sequence mismatch: expected {expected_seq}, got {recv_seq}"
            ));
        }
        if status != 0 {
            return Err(format!(
                "GPU inference server returned error status {status}"
            ));
        }
        let mut scores = Vec::with_capacity(ACTIONS_DIM);
        for i in 0..ACTIONS_DIM {
            let offset = 12 + i * 4;
            scores.push(f32::from_le_bytes(
                buf[offset..offset + 4].try_into().unwrap(),
            ));
        }
        Ok((scores, value_logit))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InProcessBackend {
    CoreML,
    CpuBatch,
}

impl InProcessBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            InProcessBackend::CoreML => "rust-onnx-coreml-batch",
            InProcessBackend::CpuBatch => "rust-onnx-cpu-batch",
        }
    }
}

#[derive(Default, Clone, Serialize)]
pub struct BatchMetrics {
    pub requests: usize,
    pub rows: usize,
    pub batches: usize,
    pub forward_seconds: f64,
    pub queue_window_seconds: f64,
    pub streams: usize,
    pub batch_rows_histogram: BTreeMap<usize, usize>,
}

struct InferRequest {
    model_path: String,
    features: Vec<f32>,
    reply: SyncSender<Result<(Vec<f32>, f32), String>>,
}

struct SharedQueueState {
    items: VecDeque<InferRequest>,
    closed: bool,
}

#[derive(Clone)]
pub struct QueueClient {
    shared: Arc<(Mutex<SharedQueueState>, Condvar)>,
}

impl QueueClient {
    pub fn predict(&self, model_path: &str, features: Vec<f32>) -> Result<(Vec<f32>, f32), String> {
        let (rtx, rrx) = mpsc::sync_channel(1);
        {
            let (lock, cvar) = &*self.shared;
            let mut guard = lock.lock().unwrap();
            if guard.closed {
                return Err("In-process GPU inference queue closed".to_string());
            }
            guard.items.push_back(InferRequest {
                model_path: model_path.to_string(),
                features,
                reply: rtx,
            });
            cvar.notify_one();
        }
        rrx.recv()
            .map_err(|_| "In-process GPU inference thread dropped reply".to_string())?
    }

    pub fn predict_batch(
        &self,
        model_path: &str,
        batch: Vec<Vec<f32>>,
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
        let mut receivers = Vec::with_capacity(batch.len());
        {
            let (lock, cvar) = &*self.shared;
            let mut guard = lock.lock().unwrap();
            if guard.closed {
                return Err("In-process GPU inference queue closed".to_string());
            }
            for features in batch {
                let (rtx, rrx) = mpsc::sync_channel(1);
                guard.items.push_back(InferRequest {
                    model_path: model_path.to_string(),
                    features,
                    reply: rtx,
                });
                receivers.push(rrx);
            }
            cvar.notify_all();
        }
        let mut results = Vec::with_capacity(receivers.len());
        for rrx in receivers {
            results.push(
                rrx.recv()
                    .map_err(|_| "In-process GPU inference thread dropped reply".to_string())??,
            );
        }
        Ok(results)
    }
}

static SESSION_INIT_LOCK: Mutex<()> = Mutex::new(());

fn batched_session(
    path: &Path,
    backend: InProcessBackend,
    static_shapes: bool,
) -> ort::Result<Session> {
    // Serialize session creation so multiple CoreML streams never race writing
    // into the same `coreml-cache` directory on a cold cache.
    let _init_guard = SESSION_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let builder = Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?;
    match backend {
        InProcessBackend::CoreML => {
            let mut ep = CoreMLExecutionProvider::default()
                .with_model_format(CoreMLModelFormat::MLProgram)
                .with_compute_units(CoreMLComputeUnits::CPUAndGPU)
                .with_static_input_shapes(static_shapes)
                .with_specialization_strategy(CoreMLSpecializationStrategy::FastPrediction);
            if let Some(parent) = path.parent() {
                let cache_dir = parent.join("coreml-cache");
                let _ = std::fs::create_dir_all(&cache_dir);
                ep = ep.with_model_cache_dir(cache_dir.to_string_lossy());
            }
            builder
                .with_execution_providers([ep.build().error_on_failure()])?
                .commit_from_file(path)
        }
        InProcessBackend::CpuBatch => builder.commit_from_file(path),
    }
}

fn run_batch(
    session: &mut Session,
    batch_rows: &[Vec<f32>],
    fixed_batch: usize,
) -> Result<Vec<(Vec<f32>, f32)>, String> {
    let actual = batch_rows.len();
    let b = if fixed_batch > 0 { fixed_batch } else { actual };
    if actual == 0 || actual > b {
        return Err(format!("Invalid batch size {actual} for target {b}"));
    }
    let mut flat = Vec::with_capacity(b * FEATURES_DIM);
    for row in batch_rows {
        if row.len() != FEATURES_DIM {
            return Err(format!(
                "Expected {FEATURES_DIM} features, got {}",
                row.len()
            ));
        }
        flat.extend_from_slice(row);
    }
    if b > actual {
        let pad_row = &batch_rows[actual - 1];
        for _ in actual..b {
            flat.extend_from_slice(pad_row);
        }
    }
    let input = Tensor::from_array(([b, FEATURES_DIM], flat)).map_err(|e| e.to_string())?;
    let out = session
        .run(ort::inputs![input])
        .map_err(|e| e.to_string())?;
    let logits = out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?
        .1;
    let values = out[1]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?
        .1;
    if logits.len() != b * ACTIONS_DIM || values.len() != b {
        return Err(format!(
            "Unexpected batched ONNX output lengths: logits={}, values={}, batch={}",
            logits.len(),
            values.len(),
            b
        ));
    }
    let mut results = Vec::with_capacity(actual);
    for i in 0..actual {
        let start = i * ACTIONS_DIM;
        results.push((logits[start..start + ACTIONS_DIM].to_vec(), values[i]));
    }
    Ok(results)
}

/// Multi-stream in-process batched GPU inference owner pool.
pub struct BatchQueue {
    shared: Arc<(Mutex<SharedQueueState>, Condvar)>,
    metrics: Arc<Mutex<BatchMetrics>>,
    handles: Vec<JoinHandle<()>>,
}

impl BatchQueue {
    pub fn start(
        backend: InProcessBackend,
        max_batch_rows: usize,
        wait_us: u64,
        fixed_batch: usize,
        streams: usize,
    ) -> Self {
        let num_streams = streams.max(1);
        let effective_max = if fixed_batch > 0 {
            max_batch_rows.min(fixed_batch)
        } else {
            max_batch_rows
        };
        let shared = Arc::new((
            Mutex::new(SharedQueueState {
                items: VecDeque::with_capacity(effective_max * num_streams * 2),
                closed: false,
            }),
            Condvar::new(),
        ));
        let metrics = Arc::new(Mutex::new(BatchMetrics {
            streams: num_streams,
            ..BatchMetrics::default()
        }));
        let wait_duration = Duration::from_micros(wait_us);

        let mut handles = Vec::with_capacity(num_streams);
        for stream_id in 0..num_streams {
            let shared_bg = Arc::clone(&shared);
            let metrics_bg = Arc::clone(&metrics);
            let handle = std::thread::Builder::new()
                .name(format!("gpu-batch-owner-{stream_id}"))
                .stack_size(16 * 1024 * 1024)
                .spawn(move || {
                    let mut sessions: HashMap<String, Session> = HashMap::new();
                    let mut pending: Vec<InferRequest> = Vec::with_capacity(effective_max);

                    loop {
                        let window_secs = {
                            let (lock, cvar) = &*shared_bg;
                            let mut guard = lock.lock().unwrap();
                            while guard.items.is_empty() && !guard.closed {
                                guard = cvar.wait(guard).unwrap();
                            }
                            if guard.items.is_empty() && guard.closed {
                                break;
                            }
                            while pending.len() < effective_max {
                                if let Some(req) = guard.items.pop_front() {
                                    pending.push(req);
                                } else {
                                    break;
                                }
                            }
                            let window_start = Instant::now();
                            while pending.len() < effective_max && !guard.closed {
                                let elapsed = window_start.elapsed();
                                if elapsed >= wait_duration {
                                    break;
                                }
                                let (g, timeout_res) =
                                    cvar.wait_timeout(guard, wait_duration - elapsed).unwrap();
                                guard = g;
                                while pending.len() < effective_max {
                                    if let Some(req) = guard.items.pop_front() {
                                        pending.push(req);
                                    } else {
                                        break;
                                    }
                                }
                                if timeout_res.timed_out() {
                                    break;
                                }
                            }
                            // If items remain for another stream, wake one up before releasing lock.
                            if !guard.items.is_empty() {
                                cvar.notify_one();
                            }
                            window_start.elapsed().as_secs_f64()
                        };

                        if pending.is_empty() {
                            continue;
                        }

                        // Group drained requests by checkpoint path.
                        let mut grouped: HashMap<String, Vec<InferRequest>> = HashMap::new();
                        for req in pending.drain(..) {
                            grouped
                                .entry(req.model_path.clone())
                                .or_default()
                                .push(req);
                        }

                        for (model_path, mut group) in grouped {
                            if !sessions.contains_key(&model_path) {
                                match batched_session(
                                    Path::new(&model_path),
                                    backend,
                                    fixed_batch > 0,
                                ) {
                                    Ok(s) => {
                                        sessions.insert(model_path.clone(), s);
                                    }
                                    Err(e) => {
                                        let msg = format!(
                                            "Failed to initialize {:?} session for {}: {}",
                                            backend, model_path, e
                                        );
                                        for req in group {
                                            let _ = req.reply.send(Err(msg.clone()));
                                        }
                                        continue;
                                    }
                                }
                            }
                            let session = sessions.get_mut(&model_path).unwrap();
                            let features: Vec<Vec<f32>> = group
                                .iter_mut()
                                .map(|r| std::mem::take(&mut r.features))
                                .collect();
                            let n = features.len();
                            let fwd_start = Instant::now();
                            let batch_res = run_batch(session, &features, fixed_batch);
                            let fwd_secs = fwd_start.elapsed().as_secs_f64();

                            if let Ok(mut m) = metrics_bg.lock() {
                                m.requests += n;
                                m.rows += n;
                                m.batches += 1;
                                m.forward_seconds += fwd_secs;
                                m.queue_window_seconds += window_secs;
                                *m.batch_rows_histogram.entry(n).or_insert(0) += 1;
                            }

                            match batch_res {
                                Ok(outputs) => {
                                    for (req, out) in group.into_iter().zip(outputs) {
                                        let _ = req.reply.send(Ok(out));
                                    }
                                }
                                Err(err) => {
                                    for req in group {
                                        let _ = req.reply.send(Err(err.clone()));
                                    }
                                }
                            }
                        }
                    }
                })
                .expect("Failed to spawn GPU batch owner thread");
            handles.push(handle);
        }

        Self {
            shared,
            metrics,
            handles,
        }
    }

    pub fn client(&self) -> QueueClient {
        QueueClient {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn shutdown(mut self) -> BatchMetrics {
        {
            let (lock, cvar) = &*self.shared;
            let mut guard = lock.lock().unwrap();
            guard.closed = true;
            cvar.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
        self.metrics.lock().unwrap().clone()
    }
}
