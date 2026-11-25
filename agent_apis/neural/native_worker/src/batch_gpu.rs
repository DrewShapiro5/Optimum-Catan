//! Batched GPU inference backends for native Rust self-play workers.
//!
//! Supports two GPU-batched modes alongside the default single-threaded CPU ONNX path:
//! 1. `RemoteSocket`: Binary Unix-domain or TCP (`tcp://host:port`) socket client sending raw
//!    `[f32; 681]` feature vectors to a shared batched PyTorch (MPS/CUDA) GPU server.
//! 2. `InProcessQueue`: In-process bounded Rust queue drained by `streams` concurrent GPU
//!    owner threads running batched ONNX Runtime with CoreML EP, CUDA EP, or batched CPU.
use super::*;
use ort::execution_providers::cuda::{CuDNNConvAlgorithmSearch, CUDAExecutionProvider};
#[cfg(target_os = "macos")]
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat, CoreMLSpecializationStrategy,
};
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write as IoWrite};
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

enum GpuStream {
    #[cfg(unix)]
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl GpuStream {
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            #[cfg(unix)]
            GpuStream::Unix(s) => s.write_all(buf),
            GpuStream::Tcp(s) => s.write_all(buf),
        }
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        match self {
            #[cfg(unix)]
            GpuStream::Unix(s) => s.read_exact(buf),
            GpuStream::Tcp(s) => s.read_exact(buf),
        }
    }
}

/// Persistent binary socket client (Unix-domain or TCP loopback) for the shared GPU inference queue.
pub struct SocketClient {
    stream: GpuStream,
    worker_id: u32,
    sequence: u32,
    req_buf: Vec<u8>,
    resp_buf: Vec<u8>,
}

impl SocketClient {
    pub fn connect(endpoint: &str, worker_id: u32) -> Result<Self, String> {
        let stream = if let Some(addr) = endpoint.strip_prefix("tcp://") {
            let s = TcpStream::connect(addr)
                .map_err(|e| format!("Failed to connect to GPU TCP endpoint {addr}: {e}"))?;
            let _ = s.set_nodelay(true);
            s.set_read_timeout(Some(Duration::from_secs(90)))
                .map_err(|e| e.to_string())?;
            s.set_write_timeout(Some(Duration::from_secs(90)))
                .map_err(|e| e.to_string())?;
            GpuStream::Tcp(s)
        } else {
            #[cfg(unix)]
            {
                let s = UnixStream::connect(endpoint)
                    .map_err(|e| format!("Failed to connect to GPU socket {endpoint}: {e}"))?;
                s.set_read_timeout(Some(Duration::from_secs(90)))
                    .map_err(|e| e.to_string())?;
                s.set_write_timeout(Some(Duration::from_secs(90)))
                    .map_err(|e| e.to_string())?;
                GpuStream::Unix(s)
            }
            #[cfg(not(unix))]
            {
                let s = TcpStream::connect(endpoint).map_err(|e| {
                    format!("Failed to connect to GPU TCP endpoint {endpoint} (use tcp://host:port on Windows): {e}")
                })?;
                let _ = s.set_nodelay(true);
                s.set_read_timeout(Some(Duration::from_secs(90)))
                    .map_err(|e| e.to_string())?;
                s.set_write_timeout(Some(Duration::from_secs(90)))
                    .map_err(|e| e.to_string())?;
                GpuStream::Tcp(s)
            }
        };
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
    Cuda,
    CpuBatch,
}

impl InProcessBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            InProcessBackend::CoreML => "rust-onnx-coreml-batch",
            InProcessBackend::Cuda => "rust-onnx-cuda-batch",
            InProcessBackend::CpuBatch => "rust-onnx-cpu-batch",
        }
    }
}

#[derive(Default, Clone, Serialize)]
pub struct BatchMetrics {
    pub requests: usize,
    pub rows: usize,
    pub batches: usize,
    pub dynamic_batches: usize,
    pub dynamic_rows: usize,
    pub forward_seconds: f64,
    pub queue_window_seconds: f64,
    pub streams: usize,
    pub batch_rows_histogram: BTreeMap<usize, usize>,
}

struct InferRequest {
    model_path: String,
    fallback_model_path: Option<String>,
    features: Vec<f32>,
    slot: usize,
    reply: SyncSender<(usize, Result<(Vec<f32>, f32), String>)>,
}

struct SharedQueueState {
    items: VecDeque<InferRequest>,
    closed: bool,
}

struct QueueCounters {
    queued_items: AtomicUsize,
    active_games: AtomicUsize,
    tracks_active: AtomicBool,
}

/// RAII guard tracking an active concurrent game in `BatchQueue`.
pub struct ActiveGameGuard {
    counters: Arc<QueueCounters>,
}

impl Drop for ActiveGameGuard {
    fn drop(&mut self) {
        self.counters.active_games.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone)]
pub struct QueueClient {
    shared: Arc<(Mutex<SharedQueueState>, Condvar)>,
    counters: Arc<QueueCounters>,
}

impl QueueClient {
    pub fn predict(&self, model_path: &str, features: Vec<f32>) -> Result<(Vec<f32>, f32), String> {
        self.predict_with_fallback(model_path, None, features)
    }

    pub fn predict_with_fallback(
        &self,
        model_path: &str,
        fallback_model_path: Option<&str>,
        features: Vec<f32>,
    ) -> Result<(Vec<f32>, f32), String> {
        let (rtx, rrx) = mpsc::sync_channel(1);
        {
            let (lock, cvar) = &*self.shared;
            let mut guard = lock.lock().unwrap();
            if guard.closed {
                return Err("In-process GPU inference queue closed".to_string());
            }
            guard.items.push_back(InferRequest {
                model_path: model_path.to_string(),
                fallback_model_path: fallback_model_path.map(str::to_string),
                features,
                slot: 0,
                reply: rtx,
            });
            self.counters
                .queued_items
                .store(guard.items.len(), Ordering::Release);
            cvar.notify_one();
        }
        let (_, res) = rrx
            .recv()
            .map_err(|_| "In-process GPU inference thread dropped reply".to_string())?;
        res
    }

    pub fn predict_batch(
        &self,
        model_path: &str,
        batch: Vec<Vec<f32>>,
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
        self.predict_batch_with_fallback(model_path, None, batch)
    }

    pub fn predict_batch_with_fallback(
        &self,
        model_path: &str,
        fallback_model_path: Option<&str>,
        batch: Vec<Vec<f32>>,
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
        let k = batch.len();
        if k == 0 {
            return Ok(vec![]);
        }
        if k == 1 {
            let one = batch.into_iter().next().unwrap();
            return Ok(vec![self.predict_with_fallback(
                model_path,
                fallback_model_path,
                one,
            )?]);
        }
        let (rtx, rrx) = mpsc::sync_channel(k);
        {
            let (lock, cvar) = &*self.shared;
            let mut guard = lock.lock().unwrap();
            if guard.closed {
                return Err("In-process GPU inference queue closed".to_string());
            }
            let fallback_owned = fallback_model_path.map(str::to_string);
            for (slot, features) in batch.into_iter().enumerate() {
                guard.items.push_back(InferRequest {
                    model_path: model_path.to_string(),
                    fallback_model_path: fallback_owned.clone(),
                    features,
                    slot,
                    reply: rtx.clone(),
                });
            }
            self.counters
                .queued_items
                .store(guard.items.len(), Ordering::Release);
            cvar.notify_all();
        }
        drop(rtx);
        let mut ordered: Vec<Option<(Vec<f32>, f32)>> = vec![None; k];
        for _ in 0..k {
            let (slot, res) = rrx
                .recv()
                .map_err(|_| "In-process GPU inference thread dropped reply".to_string())?;
            ordered[slot] = Some(res?);
        }
        Ok(ordered.into_iter().map(|v| v.unwrap()).collect())
    }
}

static SESSION_INIT_LOCK: Mutex<()> = Mutex::new(());

fn batched_session(
    path: &Path,
    backend: InProcessBackend,
    static_shapes: bool,
) -> ort::Result<Session> {
    // Serialize session creation so multiple GPU streams never race initializing
    // execution providers or writing into the same cache directory.
    let _init_guard = SESSION_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let builder = Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?;
    match backend {
        InProcessBackend::CoreML => {
            #[cfg(target_os = "macos")]
            {
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
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (path, static_shapes, builder);
                Err(ort::Error::new(
                    "CoreML execution provider is only available on macOS; use --gpu-backend cuda or cpu-batch",
                ))
            }
        }
        InProcessBackend::Cuda => {
            let ep = CUDAExecutionProvider::default()
                .with_device_id(0)
                .with_conv_algorithm_search(if static_shapes {
                    CuDNNConvAlgorithmSearch::Exhaustive
                } else {
                    CuDNNConvAlgorithmSearch::Heuristic
                })
                .with_tf32(true);
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

fn drain_pending(
    guard: &mut SharedQueueState,
    pending: &mut Vec<InferRequest>,
    effective_max: usize,
    queued_items: &AtomicUsize,
) {
    while pending.len() < effective_max {
        if let Some(req) = guard.items.pop_front() {
            pending.push(req);
        } else {
            break;
        }
    }
    queued_items.store(guard.items.len(), Ordering::Release);
}

fn should_use_dynamic_fallback(
    backend: InProcessBackend,
    fixed_batch: usize,
    actual_rows: usize,
    tracked_active: bool,
    active_games: usize,
) -> bool {
    if fixed_batch == 0 || backend == InProcessBackend::CoreML || actual_rows >= fixed_batch {
        return false;
    }
    let small_game_limit = (fixed_batch / 2).max(1);
    !tracked_active || active_games <= small_game_limit
}

fn should_skip_coalesce_wait(
    wait_duration: Duration,
    pending_len: usize,
    effective_max: usize,
    closed: bool,
    tracked_active: bool,
    active_games: usize,
) -> bool {
    wait_duration.is_zero()
        || pending_len >= effective_max
        || closed
        || (tracked_active && active_games <= 1)
}

fn wait_for_coalesce_window<'a>(
    lock: &'a Mutex<SharedQueueState>,
    cvar: &Condvar,
    mut guard: std::sync::MutexGuard<'a, SharedQueueState>,
    pending: &mut Vec<InferRequest>,
    effective_max: usize,
    wait_duration: Duration,
    counters: &QueueCounters,
) -> (std::sync::MutexGuard<'a, SharedQueueState>, f64, usize, bool) {
    let tracked = counters.tracks_active.load(Ordering::Acquire);
    let active_now = counters.active_games.load(Ordering::Acquire);
    let window_start = Instant::now();

    if !should_skip_coalesce_wait(
        wait_duration,
        pending.len(),
        effective_max,
        guard.closed,
        tracked,
        active_now,
    ) {
        // Sub-millisecond Condvar::wait_timeout sleeps until the next
        // OS timer tick (~1-15.6 ms on Windows); spin-wait atomically
        // for short windows instead so single/tail batches aren't stalled.
        if wait_duration <= Duration::from_millis(2) {
            drop(guard);
            loop {
                if counters.queued_items.load(Ordering::Acquire) > 0 {
                    guard = lock.lock().unwrap();
                    drain_pending(&mut guard, pending, effective_max, &counters.queued_items);
                    let still_active = counters.active_games.load(Ordering::Acquire);
                    if should_skip_coalesce_wait(
                        wait_duration,
                        pending.len(),
                        effective_max,
                        guard.closed,
                        tracked,
                        still_active,
                    ) || window_start.elapsed() >= wait_duration
                    {
                        break;
                    }
                    drop(guard);
                } else if window_start.elapsed() >= wait_duration {
                    guard = lock.lock().unwrap();
                    drain_pending(&mut guard, pending, effective_max, &counters.queued_items);
                    break;
                } else {
                    std::hint::spin_loop();
                }
            }
        } else {
            while pending.len() < effective_max && !guard.closed {
                let elapsed = window_start.elapsed();
                if elapsed >= wait_duration {
                    break;
                }
                let (g, timeout_res) = cvar.wait_timeout(guard, wait_duration - elapsed).unwrap();
                guard = g;
                drain_pending(&mut guard, pending, effective_max, &counters.queued_items);
                let still_active = counters.active_games.load(Ordering::Acquire);
                if timeout_res.timed_out() || (tracked && still_active <= 1) {
                    break;
                }
            }
        }
    }

    let observed = counters.active_games.load(Ordering::Acquire);
    (guard, window_start.elapsed().as_secs_f64(), observed, tracked)
}

/// Multi-stream in-process batched GPU inference owner pool.
pub struct BatchQueue {
    shared: Arc<(Mutex<SharedQueueState>, Condvar)>,
    counters: Arc<QueueCounters>,
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
        let counters = Arc::new(QueueCounters {
            queued_items: AtomicUsize::new(0),
            active_games: AtomicUsize::new(0),
            tracks_active: AtomicBool::new(false),
        });
        let metrics = Arc::new(Mutex::new(BatchMetrics {
            streams: num_streams,
            ..BatchMetrics::default()
        }));
        let wait_duration = Duration::from_micros(wait_us);

        let mut handles = Vec::with_capacity(num_streams);
        for stream_id in 0..num_streams {
            let shared_bg = Arc::clone(&shared);
            let counters_bg = Arc::clone(&counters);
            let metrics_bg = Arc::clone(&metrics);
            let handle = std::thread::Builder::new()
                .name(format!("gpu-batch-owner-{stream_id}"))
                .stack_size(16 * 1024 * 1024)
                .spawn(move || {
                    let mut sessions: HashMap<String, Session> = HashMap::new();
                    let mut pending: Vec<InferRequest> = Vec::with_capacity(effective_max);

                    loop {
                        let (window_secs, observed_active, tracked_active) = {
                            let (lock, cvar) = &*shared_bg;
                            let mut guard = lock.lock().unwrap();
                            while guard.items.is_empty() && !guard.closed {
                                guard = cvar.wait(guard).unwrap();
                            }
                            if guard.items.is_empty() && guard.closed {
                                break;
                            }
                            drain_pending(
                                &mut guard,
                                &mut pending,
                                effective_max,
                                &counters_bg.queued_items,
                            );
                            let (g, secs, observed, tracked) = wait_for_coalesce_window(
                                lock,
                                cvar,
                                guard,
                                &mut pending,
                                effective_max,
                                wait_duration,
                                &counters_bg,
                            );
                            guard = g;
                            // If items remain for another stream, wake one up before releasing lock.
                            if !guard.items.is_empty() {
                                cvar.notify_one();
                            }
                            (secs, observed, tracked)
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
                            let n = group.len();
                            let fallback_path = if should_use_dynamic_fallback(
                                backend,
                                fixed_batch,
                                n,
                                tracked_active,
                                observed_active,
                            ) {
                                group.iter().find_map(|r| r.fallback_model_path.as_ref()).cloned()
                            } else {
                                None
                            };
                            let used_dynamic = fallback_path.is_some();
                            let (run_path, run_fixed_batch) = match fallback_path {
                                Some(dyn_path) => (dyn_path, 0),
                                None => (model_path, fixed_batch),
                            };
                            if !sessions.contains_key(&run_path) {
                                let init_res = match std::panic::catch_unwind(
                                    std::panic::AssertUnwindSafe(|| {
                                        batched_session(
                                            Path::new(&run_path),
                                            backend,
                                            run_fixed_batch > 0,
                                        )
                                    }),
                                ) {
                                    Ok(res) => res.map_err(|e| e.to_string()),
                                    Err(payload) => {
                                        let msg = payload
                                            .downcast_ref::<&str>()
                                            .copied()
                                            .or_else(|| {
                                                payload.downcast_ref::<String>().map(String::as_str)
                                            })
                                            .unwrap_or("GPU session init panicked");
                                        Err(msg.to_string())
                                    }
                                };
                                match init_res {
                                    Ok(s) => {
                                        sessions.insert(run_path.clone(), s);
                                    }
                                    Err(e) => {
                                        let msg = format!(
                                            "Failed to initialize {:?} session for {}: {}",
                                            backend, run_path, e
                                        );
                                        for req in group {
                                            let _ = req.reply.send((req.slot, Err(msg.clone())));
                                        }
                                        continue;
                                    }
                                }
                            }
                            let session = sessions.get_mut(&run_path).unwrap();
                            let features: Vec<Vec<f32>> = group
                                .iter_mut()
                                .map(|r| std::mem::take(&mut r.features))
                                .collect();
                            let fwd_start = Instant::now();
                            let batch_res = match std::panic::catch_unwind(
                                std::panic::AssertUnwindSafe(|| {
                                    run_batch(session, &features, run_fixed_batch)
                                }),
                            ) {
                                Ok(res) => res,
                                Err(payload) => {
                                    let msg = payload
                                        .downcast_ref::<&str>()
                                        .copied()
                                        .or_else(|| {
                                            payload.downcast_ref::<String>().map(String::as_str)
                                        })
                                        .unwrap_or("GPU batch inference panicked");
                                    Err(format!("GPU batch inference panicked: {msg}"))
                                }
                            };
                            let fwd_secs = fwd_start.elapsed().as_secs_f64();

                            if let Ok(mut m) = metrics_bg.lock() {
                                m.requests += n;
                                m.rows += n;
                                m.batches += 1;
                                if used_dynamic {
                                    m.dynamic_batches += 1;
                                    m.dynamic_rows += n;
                                }
                                m.forward_seconds += fwd_secs;
                                m.queue_window_seconds += window_secs;
                                *m.batch_rows_histogram.entry(n).or_insert(0) += 1;
                            }

                            match batch_res {
                                Ok(outputs) => {
                                    for (req, out) in group.into_iter().zip(outputs) {
                                        let _ = req.reply.send((req.slot, Ok(out)));
                                    }
                                }
                                Err(err) => {
                                    for req in group {
                                        let _ = req.reply.send((req.slot, Err(err.clone())));
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
            counters,
            metrics,
            handles,
        }
    }

    pub fn client(&self) -> QueueClient {
        QueueClient {
            shared: Arc::clone(&self.shared),
            counters: Arc::clone(&self.counters),
        }
    }

    pub fn active_guard(&self) -> ActiveGameGuard {
        self.counters.tracks_active.store(true, Ordering::Release);
        self.counters.active_games.fetch_add(1, Ordering::AcqRel);
        ActiveGameGuard {
            counters: Arc::clone(&self.counters),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_coalesce_wait_when_single_active_game() {
        let shared = Mutex::new(SharedQueueState {
            items: VecDeque::new(),
            closed: false,
        });
        let cvar = Condvar::new();
        let counters = QueueCounters {
            queued_items: AtomicUsize::new(0),
            active_games: AtomicUsize::new(1),
            tracks_active: AtomicBool::new(true),
        };
        let mut pending = Vec::new();
        let guard = shared.lock().unwrap();
        let (_g, secs, observed, tracked) = wait_for_coalesce_window(
            &shared,
            &cvar,
            guard,
            &mut pending,
            128,
            Duration::from_millis(50),
            &counters,
        );
        assert!(tracked);
        assert_eq!(observed, 1);
        assert!(secs < 0.005, "Solo active game waited {secs}s");
    }

    #[test]
    fn active_guard_tracks_and_drops_count() {
        let q = BatchQueue::start(InProcessBackend::CpuBatch, 64, 50, 0, 1);
        assert!(!q.counters.tracks_active.load(Ordering::Acquire));
        let g1 = q.active_guard();
        assert!(q.counters.tracks_active.load(Ordering::Acquire));
        assert_eq!(q.counters.active_games.load(Ordering::Acquire), 1);
        {
            let _g2 = q.active_guard();
            assert_eq!(q.counters.active_games.load(Ordering::Acquire), 2);
        }
        assert_eq!(q.counters.active_games.load(Ordering::Acquire), 1);
        drop(g1);
        assert_eq!(q.counters.active_games.load(Ordering::Acquire), 0);
        let _ = q.shutdown();
    }

    #[test]
    fn dynamic_fallback_policy_matches_small_game_counts() {
        assert!(!should_use_dynamic_fallback(
            InProcessBackend::CoreML,
            128,
            1,
            true,
            1
        ));
        assert!(!should_use_dynamic_fallback(
            InProcessBackend::Cuda,
            0,
            1,
            true,
            1
        ));
        assert!(!should_use_dynamic_fallback(
            InProcessBackend::Cuda,
            128,
            128,
            true,
            1
        ));
        assert!(!should_use_dynamic_fallback(
            InProcessBackend::Cuda,
            128,
            16,
            true,
            256
        ));
        assert!(should_use_dynamic_fallback(
            InProcessBackend::Cuda,
            128,
            16,
            true,
            64
        ));
        assert!(should_use_dynamic_fallback(
            InProcessBackend::Cuda,
            128,
            1,
            true,
            1
        ));
        assert!(should_use_dynamic_fallback(
            InProcessBackend::Cuda,
            128,
            8,
            false,
            0
        ));
    }

    #[test]
    fn sub_ms_spin_wait_completes_quickly() {
        let shared = Mutex::new(SharedQueueState {
            items: VecDeque::new(),
            closed: false,
        });
        let cvar = Condvar::new();
        let counters = QueueCounters {
            queued_items: AtomicUsize::new(0),
            active_games: AtomicUsize::new(2),
            tracks_active: AtomicBool::new(true),
        };
        let mut pending = Vec::new();
        let guard = shared.lock().unwrap();
        let started = Instant::now();
        let (_g, secs, observed, tracked) = wait_for_coalesce_window(
            &shared,
            &cvar,
            guard,
            &mut pending,
            128,
            Duration::from_micros(50),
            &counters,
        );
        let elapsed = started.elapsed();
        assert!(tracked);
        assert_eq!(observed, 2);
        assert!(
            elapsed < Duration::from_millis(2) && secs < 0.002,
            "50us spin wait window took {:?} (secs={})",
            elapsed,
            secs
        );
    }
}
