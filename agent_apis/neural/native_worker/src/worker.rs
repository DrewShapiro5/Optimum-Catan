//! Versioned JSON-lines bridge. Search, inference and actual game RNG stay native.
//!
//! Supports:
//! - Default (`ServeMode::SingleCpu`): original single-game-per-process ONNX CPU worker.
//! - `--gpu-socket <PATH> [--worker-id <ID>]`: single-game worker sending candidate CNN
//!   feature vectors over a binary Unix-domain socket to a shared batched GPU queue.
//! - `--pool --workers <N> [--gpu-backend coreml|cpu-batch] [--max-batch <B>] [--wait-us <US>]`:
//!   multi-threaded Rust worker pool submitting candidate CNN requests to an in-process
//!   batched ONNX Runtime (CoreML EP or CPU) owner thread.
use super::*;
use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::sync::{Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub enum ServeMode {
    SingleCpu,
    SingleSocket {
        socket_path: String,
        worker_id: u32,
    },
    Pool {
        workers: usize,
        backend: batch_gpu::InProcessBackend,
        max_batch: usize,
        wait_us: u64,
        fixed_batch: usize,
        streams: usize,
    },
    BridgePool {
        workers: usize,
        backend: batch_gpu::InProcessBackend,
        max_batch: usize,
        wait_us: u64,
        fixed_batch: usize,
        streams: usize,
    },
}

pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<ServeMode, String> {
    let mut iter = args.into_iter();
    let mut gpu_socket: Option<String> = None;
    let mut worker_id: u32 = 0;
    let mut pool = false;
    let mut bridge_pool = false;
    let mut workers: usize = 12;
    #[cfg(target_os = "macos")]
    let mut backend = batch_gpu::InProcessBackend::CoreML;
    #[cfg(not(target_os = "macos"))]
    let mut backend = batch_gpu::InProcessBackend::Cuda;
    let mut max_batch: usize = 64;
    let mut wait_us: u64 = 500;
    let mut fixed_batch: usize = 0;
    let mut streams: usize = 1;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--gpu-socket" => {
                gpu_socket = Some(
                    iter.next()
                        .ok_or_else(|| "--gpu-socket requires a path".to_string())?,
                );
            }
            "--worker-id" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--worker-id requires an integer".to_string())?;
                worker_id = v
                    .parse::<u32>()
                    .map_err(|e| format!("Invalid --worker-id '{v}': {e}"))?;
            }
            "--pool" => pool = true,
            "--bridge-pool" => bridge_pool = true,
            "--workers" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--workers requires an integer".to_string())?;
                workers = v
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid --workers '{v}': {e}"))?;
                if workers == 0 {
                    return Err("--workers must be >= 1".into());
                }
            }
            "--gpu-backend" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--gpu-backend requires coreml, cuda, or cpu-batch".to_string())?;
                backend = match v.as_str() {
                    "coreml" => batch_gpu::InProcessBackend::CoreML,
                    "cuda" => batch_gpu::InProcessBackend::Cuda,
                    "cpu-batch" => batch_gpu::InProcessBackend::CpuBatch,
                    _ => return Err(format!("Unsupported --gpu-backend '{v}'")),
                };
            }
            "--max-batch" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--max-batch requires an integer".to_string())?;
                max_batch = v
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid --max-batch '{v}': {e}"))?;
                if max_batch == 0 {
                    return Err("--max-batch must be >= 1".into());
                }
            }
            "--wait-us" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--wait-us requires an integer".to_string())?;
                wait_us = v
                    .parse::<u64>()
                    .map_err(|e| format!("Invalid --wait-us '{v}': {e}"))?;
            }
            "--fixed-batch" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--fixed-batch requires an integer".to_string())?;
                fixed_batch = v
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid --fixed-batch '{v}': {e}"))?;
            }
            "--gpu-streams" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--gpu-streams requires an integer".to_string())?;
                streams = v
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid --gpu-streams '{v}': {e}"))?;
                if streams == 0 {
                    return Err("--gpu-streams must be >= 1".into());
                }
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
    }

    if (pool || bridge_pool) && gpu_socket.is_some() {
        return Err("--pool/--bridge-pool and --gpu-socket are mutually exclusive".into());
    }
    if pool && bridge_pool {
        return Err("--pool and --bridge-pool are mutually exclusive".into());
    }
    if bridge_pool {
        Ok(ServeMode::BridgePool {
            workers,
            backend,
            max_batch,
            wait_us,
            fixed_batch,
            streams,
        })
    } else if pool {
        Ok(ServeMode::Pool {
            workers,
            backend,
            max_batch,
            wait_us,
            fixed_batch,
            streams,
        })
    } else if let Some(socket_path) = gpu_socket {
        Ok(ServeMode::SingleSocket {
            socket_path,
            worker_id,
        })
    } else {
        Ok(ServeMode::SingleCpu)
    }
}

#[derive(Deserialize)]
struct Request {
    version: u32,
    task: Task,
    models: [String; 2],
    frozen: String,
    simulations: Option<usize>,
    seconds: Option<f64>,
    deadline: f64,
    stop_file: String,
    max_turns: usize,
    max_actions: usize,
    #[serde(default)]
    game_id: Option<String>,
    #[serde(default)]
    model_ids: Option<[u32; 2]>,
    #[serde(default)]
    lanes: Option<usize>,
    #[serde(default)]
    virtual_loss: Option<f64>,
    #[serde(default)]
    eval_mode: bool,
    #[serde(default)]
    seat_simulations: Option<[usize; 2]>,
    #[serde(default)]
    adversarial_puct: bool,
    #[serde(default)]
    seat_adversarial_puct: Option<[bool; 2]>,
    #[serde(default)]
    mcts_opponent: Option<MctsOpponent>,
    #[serde(default)]
    seat_mcts_opponent: Option<[MctsOpponent; 2]>,
    #[serde(default)]
    setup_simulations: Option<usize>,
    #[serde(default)]
    setup_temperature: Option<f64>,
    #[serde(default)]
    vp_margin_alpha: Option<f64>,
    #[serde(default)]
    seat_vp_margin_alpha: Option<[f64; 2]>,
    #[serde(default)]
    dynamic_models: Option<[String; 2]>,
}

fn emit(mut value: serde_json::Value, game_id: Option<&str>) -> Result<(), String> {
    if let (Some(id), Some(obj)) = (game_id, value.as_object_mut()) {
        obj.insert("game_id".into(), serde_json::Value::String(id.into()));
    }
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &value).map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

enum CandidateProvider {
    Cpu,
    Socket(Arc<Mutex<batch_gpu::SocketClient>>),
    Queue(batch_gpu::QueueClient),
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("Worker thread panicked")
        .to_string()
}

fn pool_thread_count(workers: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    // Keep several games per CPU thread so tiny/head-to-head pools batch more rows per submit wave.
    workers.div_ceil(4).max(1).min(cores.max(1))
}

fn ensure_models(
    cache: &mut HashMap<(String, String, u32), search::Models>,
    keys: &[(String, String, u32); 2],
    provider: &CandidateProvider,
    need_frozen: [bool; 2],
) -> Result<(), String> {
    if keys.iter().any(|key| !cache.contains_key(key)) && cache.len() + 2 > 4 {
        cache.clear();
    }
    for (idx, key) in keys.iter().enumerate() {
        if !cache.contains_key(key) {
            let candidate = match provider {
                CandidateProvider::Cpu => batch_gpu::CandidateModel::LocalCpu(
                    session(Path::new(&key.0)).map_err(|e| e.to_string())?,
                ),
                CandidateProvider::Socket(client) => batch_gpu::CandidateModel::RemoteSocket {
                    client: Arc::clone(client),
                    model_id: key.2,
                },
                CandidateProvider::Queue(client) => batch_gpu::CandidateModel::InProcessQueue {
                    client: client.clone(),
                    model_path: key.0.clone(),
                },
            };
            let opponent = if need_frozen[idx] {
                Some(session(Path::new(&key.1)).map_err(|e| e.to_string())?)
            } else {
                None
            };
            cache.insert(
                key.clone(),
                search::Models {
                    candidate,
                    opponent,
                },
            );
        } else if need_frozen[idx] {
            let entry = cache.get_mut(key).unwrap();
            if entry.opponent.is_none() {
                entry.opponent = Some(session(Path::new(&key.1)).map_err(|e| e.to_string())?);
            }
        }
    }
    Ok(())
}

fn parse_seat_opts(r: &Request) -> [Options; 2] {
    let default_opp = r.mcts_opponent.unwrap_or(if r.adversarial_puct {
        MctsOpponent::AdversarialPuct
    } else {
        MctsOpponent::FrozenMlp
    });
    std::array::from_fn(|i| {
        let mcts_opponent = if let Some(s) = r.seat_mcts_opponent {
            s[i]
        } else if let Some(s) = r.seat_adversarial_puct {
            if s[i] {
                MctsOpponent::AdversarialPuct
            } else {
                MctsOpponent::FrozenMlp
            }
        } else {
            default_opp
        };
        Options {
            simulations: r.seat_simulations.map(|s| s[i]).or(r.simulations),
            seconds: r.seconds,
            noise: !r.eval_mode,
            lanes: r.lanes.unwrap_or(1).max(1),
            virtual_loss: r.virtual_loss.unwrap_or(1.0),
            mcts_opponent,
            vp_margin_alpha: r
                .seat_vp_margin_alpha
                .map(|s| s[i])
                .or(r.vp_margin_alpha)
                .unwrap_or(0.03),
        }
    })
}

fn validate_request(r: &Request) -> Result<(), String> {
    if r.version != 1
        || r.simulations.is_some() == r.seconds.is_some()
        || r.simulations == Some(0)
        || r.setup_simulations == Some(0)
        || r.setup_temperature.is_some_and(|t| !t.is_finite() || t <= 0.)
        || r.vp_margin_alpha
            .is_some_and(|a| !a.is_finite() || !(0.0..=0.5).contains(&a))
        || r.seat_vp_margin_alpha.is_some_and(|s| {
            !s[0].is_finite()
                || !s[1].is_finite()
                || !(0.0..=0.5).contains(&s[0])
                || !(0.0..=0.5).contains(&s[1])
        })
        || r.seat_simulations.is_some_and(|s| s[0] == 0 || s[1] == 0 || r.seconds.is_some())
        || r.seconds.is_some_and(|s| !s.is_finite() || s <= 0.)
        || r.max_turns == 0
        || r.max_turns > 1000
        || r.max_actions == 0
        || r.max_actions > 10000
    {
        return Err("Invalid native protocol/budget/limits".into());
    }
    let t = &r.task;
    if t.start_rng != t.rng_state {
        return Err("Initial RNG mismatch".into());
    }
    if !t.logs.is_empty() || t.prefix != 0 {
        return Err("Production games must start at the initial state".into());
    }
    Ok(())
}

fn play(
    r: Request,
    cache: &mut HashMap<(String, String, u32), search::Models>,
    provider: &CandidateProvider,
) -> Result<(), String> {
    validate_request(&r)?;
    let game_id = r.game_id.clone();
    let t = &r.task;
    let model_ids = r.model_ids.unwrap_or([0, 0]);
    let keys = [
        (r.models[0].clone(), r.frozen.clone(), model_ids[0]),
        (r.models[1].clone(), r.frozen.clone(), model_ids[1]),
    ];
    let seat_opts = parse_seat_opts(&r);
    let need_frozen = [
        seat_opts[0].mcts_opponent == MctsOpponent::FrozenMlp,
        seat_opts[1].mcts_opponent == MctsOpponent::FrozenMlp,
    ];
    ensure_models(cache, &keys, provider, need_frozen)?;
    let enc = Encoder::new(Arc::new(t.job.clone()));
    let mut w = Arc::new(task_world(t));
    if !w.e.state.friendly
        || w.e.state.target != 15
        || w.e.state.discard_limit != 9
        || w.e.state.players.len() != 2
        || !w.e.state.initial
    {
        return Err("Unsupported production rules/start state".into());
    }
    let mut actual = rng::Rng::from_state(&t.rng_state);
    let mut seeds =
        std::array::from_fn::<_, 2, _>(|i| rng::Rng::new(rng::named(t.seat_seeds[i], "search")));
    let mut action_rng = std::array::from_fn::<_, 2, _>(|i| {
        pcg::Pcg::new(rng::named(t.seat_seeds[i], "action-sampling"))
    });
    let mut seats: [Option<search::Search>; 2] = [None, None];
    let mut batch = Vec::new();
    let mut actions = 0;
    let mut decisions = 0;
    let mut sims = 0;
    let started = Instant::now();
    let mut interrupted = false;
    while w.e.winner() < 0 && w.e.state.turns < r.max_turns && actions < r.max_actions {
        if now() >= r.deadline || Path::new(&r.stop_file).exists() {
            interrupted = true;
            break;
        }
        let actor = w.e.state.current;
        let (a, visits, metrics) = if w.e.actions.len() == 1 {
            (w.e.actions[0], vec![], None)
        } else {
            let seed = seeds[actor].u64();
            let mut tree = if let Some(old) = seats[actor].take() {
                old.advance(w.clone(), &enc)
            } else {
                search::Search::new(w.clone(), seed)
            };
            let base_opt = &seat_opts[actor];
            let opt = Options {
                simulations: if w.e.state.initial {
                    r.setup_simulations.or(base_opt.simulations)
                } else {
                    base_opt.simulations
                },
                seconds: base_opt.seconds,
                noise: base_opt.noise,
                lanes: base_opt.lanes,
                virtual_loss: base_opt.virtual_loss,
                mcts_opponent: base_opt.mcts_opponent,
                vp_margin_alpha: base_opt.vp_margin_alpha,
            };
            let result = tree.run(
                &enc,
                cache.get_mut(&keys[actor]).unwrap(),
                seed,
                &opt,
                None,
                false,
            )?;
            assert!(
                opt.simulations.is_none() || result.metrics.simulations == opt.simulations.unwrap()
            );
            let selected = if r.eval_mode {
                result.action_id
            } else {
                let temperature = if w.e.state.initial {
                    r.setup_temperature.unwrap_or(1.)
                } else if w.e.state.turns < 20 {
                    1.
                } else {
                    0.25
                };
                if result.statistics.iter().any(|x| x.visits > 0) {
                    search::choose_temperature(&result.statistics, temperature, &mut action_rng[actor])
                } else {
                    result.action_id
                }
            };
            tree.selected = Some(selected);
            let a =
                *w.e.actions
                    .iter()
                    .find(|&&a| enc.id(&w.e, a) == selected)
                    .unwrap();
            let visits = result
                .statistics
                .iter()
                .filter(|x| x.visits > 0)
                .map(|x| (x.action_id, x.visits))
                .collect::<Vec<_>>();
            sims += result.metrics.simulations;
            decisions += 1;
            seats[actor] = Some(tree);
            (a, visits, Some(result.metrics))
        };
        let id = enc.id(&w.e, a);
        let (chance, dice) = search::actual_chance(&w, a, &mut actual);
        let mut next_w = w.step(a, chance);
        next_w.logs.compact();
        w = Arc::new(next_w);
        actions += 1;
        batch.push(serde_json::json!({"actor":actor,"action_id":id,"chance":chance,"dice":dice,"visits":visits,"metrics":metrics}));
        if batch.len() >= 10 {
            emit(
                serde_json::json!({"type":"events","events":batch}),
                game_id.as_deref(),
            )?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        emit(
            serde_json::json!({"type":"events","events":batch}),
            game_id.as_deref(),
        )?;
    }
    let winner = w.e.winner();
    emit(
        serde_json::json!({"type":"done","version":1,"id":t.id,"status":if winner>=0{"completed"}else if interrupted{"interrupted"}else{"cutoff"},"winner":winner,"turns":w.e.state.turns,"actions":actions,"searched_decisions":decisions,"simulations":sims,"elapsed_seconds":started.elapsed().as_secs_f64(),"final_state":w.e.state,"rng_state":actual.state()}),
        game_id.as_deref(),
    )
}

pub fn serve(mode: ServeMode) -> Result<(), String> {
    match mode {
        ServeMode::SingleCpu => {
            emit(
                serde_json::json!({"type":"ready","version":1,"backend":"rust-onnx-cpu","threads":1}),
                None,
            )?;
            let mut models = HashMap::new();
            let provider = CandidateProvider::Cpu;
            for line in io::stdin().lock().lines() {
                let line = line.map_err(|e| e.to_string())?;
                let result = serde_json::from_str::<Request>(&line)
                    .map_err(|e| e.to_string())
                    .and_then(|r| play(r, &mut models, &provider));
                if let Err(error) = result {
                    emit(serde_json::json!({"type":"error","error":error}), None)?;
                }
            }
            Ok(())
        }
        ServeMode::SingleSocket {
            socket_path,
            worker_id,
        } => {
            let client = Arc::new(Mutex::new(batch_gpu::SocketClient::connect(
                &socket_path,
                worker_id,
            )?));
            emit(
                serde_json::json!({"type":"ready","version":1,"backend":"rust-gpu-socket","threads":1,"worker_id":worker_id}),
                None,
            )?;
            let mut models = HashMap::new();
            let provider = CandidateProvider::Socket(client);
            for line in io::stdin().lock().lines() {
                let line = line.map_err(|e| e.to_string())?;
                let result = serde_json::from_str::<Request>(&line)
                    .map_err(|e| e.to_string())
                    .and_then(|r| play(r, &mut models, &provider));
                if let Err(error) = result {
                    emit(serde_json::json!({"type":"error","error":error}), None)?;
                }
            }
            Ok(())
        }
        ServeMode::Pool {
            workers,
            backend,
            max_batch,
            wait_us,
            fixed_batch,
            streams,
        } => serve_pool(workers, backend, max_batch, wait_us, fixed_batch, streams),
        ServeMode::BridgePool {
            workers,
            backend,
            max_batch,
            wait_us,
            fixed_batch,
            streams,
        } => serve_bridge_pool(workers, backend, max_batch, wait_us, fixed_batch, streams),
    }
}

struct PoolGame {
    game_id: Option<String>,
    task_id: String,
    keys: [(String, String, u32); 2],
    dynamic_models: [Option<String>; 2],
    need_frozen: [bool; 2],
    seat_opts: [Options; 2],
    enc: Encoder,
    w: Arc<World>,
    actual: rng::Rng,
    seeds: [rng::Rng; 2],
    action_rng: [pcg::Pcg; 2],
    seats: [Option<search::Search>; 2],
    batch: Vec<serde_json::Value>,
    actions: usize,
    decisions: usize,
    sims: usize,
    started: Instant,
    interrupted: bool,
    deadline: f64,
    stop_file: String,
    max_turns: usize,
    max_actions: usize,
    eval_mode: bool,
    setup_simulations: Option<usize>,
    setup_temperature: Option<f64>,
    active_decision: Option<ActiveDecision>,
    _active_guard: Option<batch_gpu::ActiveGameGuard>,
}

struct ActiveDecision {
    actor: usize,
    opt_sims: Option<usize>,
    stepper: search::SearchStepper,
    infer_start: Option<Instant>,
    reply: Option<Vec<(Vec<f32>, f32)>>,
}

enum PoolStep {
    NeedEval {
        model_path: String,
        fallback_model: Option<String>,
        features: Vec<Vec<f32>>,
    },
    Finished,
}

impl PoolGame {
    fn new(
        r: Request,
        cache: &mut HashMap<(String, String, u32), search::Models>,
        provider: &CandidateProvider,
    ) -> Result<Self, String> {
        validate_request(&r)?;
        let game_id = r.game_id.clone();
        let t = &r.task;
        let model_ids = r.model_ids.unwrap_or([0, 0]);
        let keys = [
            (r.models[0].clone(), r.frozen.clone(), model_ids[0]),
            (r.models[1].clone(), r.frozen.clone(), model_ids[1]),
        ];
        let dynamic_models = match &r.dynamic_models {
            Some([d0, d1]) => [Some(d0.clone()), Some(d1.clone())],
            None => [None, None],
        };
        let seat_opts = parse_seat_opts(&r);
        let need_frozen = [
            seat_opts[0].mcts_opponent == MctsOpponent::FrozenMlp,
            seat_opts[1].mcts_opponent == MctsOpponent::FrozenMlp,
        ];
        ensure_models(cache, &keys, provider, need_frozen)?;
        let enc = Encoder::new(Arc::new(t.job.clone()));
        let w = Arc::new(task_world(t));
        if !w.e.state.friendly
            || w.e.state.target != 15
            || w.e.state.discard_limit != 9
            || w.e.state.players.len() != 2
            || !w.e.state.initial
        {
            return Err("Unsupported production rules/start state".into());
        }
        let actual = rng::Rng::from_state(&t.rng_state);
        let seeds =
            std::array::from_fn::<_, 2, _>(|i| rng::Rng::new(rng::named(t.seat_seeds[i], "search")));
        let action_rng = std::array::from_fn::<_, 2, _>(|i| {
            pcg::Pcg::new(rng::named(t.seat_seeds[i], "action-sampling"))
        });
        Ok(Self {
            game_id,
            task_id: t.id.clone(),
            keys,
            dynamic_models,
            need_frozen,
            seat_opts,
            enc,
            w,
            actual,
            seeds,
            action_rng,
            seats: [None, None],
            batch: Vec::new(),
            actions: 0,
            decisions: 0,
            sims: 0,
            started: Instant::now(),
            interrupted: false,
            deadline: r.deadline,
            stop_file: r.stop_file,
            max_turns: r.max_turns,
            max_actions: r.max_actions,
            eval_mode: r.eval_mode,
            setup_simulations: r.setup_simulations,
            setup_temperature: r.setup_temperature,
            active_decision: None,
            _active_guard: None,
        })
    }

    fn apply_action(
        &mut self,
        actor: usize,
        a: Act,
        visits: Vec<(usize, usize)>,
        metrics: Option<search::Metrics>,
    ) -> Result<(), String> {
        let id = self.enc.id(&self.w.e, a);
        let (chance, dice) = search::actual_chance(&self.w, a, &mut self.actual);
        let mut next_w = self.w.step(a, chance);
        next_w.logs.compact();
        self.w = Arc::new(next_w);
        self.actions += 1;
        self.batch.push(serde_json::json!({
            "actor": actor,
            "action_id": id,
            "chance": chance,
            "dice": dice,
            "visits": visits,
            "metrics": metrics
        }));
        if self.batch.len() >= 10 {
            emit(
                serde_json::json!({"type":"events","events":self.batch}),
                self.game_id.as_deref(),
            )?;
            self.batch.clear();
        }
        Ok(())
    }

    fn finish_decision(
        &mut self,
        actor: usize,
        opt_sims: Option<usize>,
        mut search: search::Search,
        result: search::ResultData,
    ) -> Result<(), String> {
        assert!(opt_sims.is_none() || result.metrics.simulations == opt_sims.unwrap());
        let selected = if self.eval_mode {
            result.action_id
        } else {
            let temperature = if self.w.e.state.initial {
                self.setup_temperature.unwrap_or(1.)
            } else if self.w.e.state.turns < 20 {
                1.
            } else {
                0.25
            };
            if result.statistics.iter().any(|x| x.visits > 0) {
                search::choose_temperature(&result.statistics, temperature, &mut self.action_rng[actor])
            } else {
                result.action_id
            }
        };
        search.selected = Some(selected);
        let a = *self
            .w
            .e
            .actions
            .iter()
            .find(|&&a| self.enc.id(&self.w.e, a) == selected)
            .unwrap();
        let visits = result
            .statistics
            .iter()
            .filter(|x| x.visits > 0)
            .map(|x| (x.action_id, x.visits))
            .collect::<Vec<_>>();
        self.sims += result.metrics.simulations;
        self.decisions += 1;
        self.seats[actor] = Some(search);
        self.apply_action(actor, a, visits, Some(result.metrics))
    }

    fn advance_until_eval(
        &mut self,
        cache: &mut HashMap<(String, String, u32), search::Models>,
        provider: &CandidateProvider,
    ) -> Result<PoolStep, String> {
        ensure_models(cache, &self.keys, provider, self.need_frozen)?;
        loop {
            if let Some(mut dec) = self.active_decision.take() {
                if let Some(t) = dec.infer_start.take() {
                    dec.stepper.add_inference_seconds(t.elapsed().as_secs_f64());
                }
                let models = cache.get_mut(&self.keys[dec.actor]).unwrap();
                match dec.stepper.step(&self.enc, models, dec.reply.take())? {
                    search::StepOutcome::NeedCandidate { stepper, features } => {
                        let model_path = self.keys[dec.actor].0.clone();
                        let fallback_model = self.dynamic_models[dec.actor].clone();
                        self.active_decision = Some(ActiveDecision {
                            actor: dec.actor,
                            opt_sims: dec.opt_sims,
                            stepper,
                            infer_start: Some(Instant::now()),
                            reply: None,
                        });
                        return Ok(PoolStep::NeedEval {
                            model_path,
                            fallback_model,
                            features,
                        });
                    }
                    search::StepOutcome::Done { search, result } => {
                        self.finish_decision(dec.actor, dec.opt_sims, search, result)?;
                    }
                }
                continue;
            }

            if self.w.e.winner() >= 0
                || self.w.e.state.turns >= self.max_turns
                || self.actions >= self.max_actions
            {
                break;
            }
            if now() >= self.deadline || Path::new(&self.stop_file).exists() {
                self.interrupted = true;
                break;
            }
            let actor = self.w.e.state.current;
            if self.w.e.actions.len() == 1 {
                let a = self.w.e.actions[0];
                self.apply_action(actor, a, vec![], None)?;
                continue;
            }
            let seed = self.seeds[actor].u64();
            let tree = if let Some(old) = self.seats[actor].take() {
                old.advance(self.w.clone(), &self.enc)
            } else {
                search::Search::new(self.w.clone(), seed)
            };
            let base_opt = &self.seat_opts[actor];
            let opt = Options {
                simulations: if self.w.e.state.initial {
                    self.setup_simulations.or(base_opt.simulations)
                } else {
                    base_opt.simulations
                },
                seconds: base_opt.seconds,
                noise: base_opt.noise,
                lanes: base_opt.lanes,
                virtual_loss: base_opt.virtual_loss,
                mcts_opponent: base_opt.mcts_opponent,
                vp_margin_alpha: base_opt.vp_margin_alpha,
            };
            let stepper = search::SearchStepper::new(tree, seed, &opt, None, false);
            self.active_decision = Some(ActiveDecision {
                actor,
                opt_sims: opt.simulations,
                stepper,
                infer_start: None,
                reply: None,
            });
        }

        if !self.batch.is_empty() {
            emit(
                serde_json::json!({"type":"events","events":self.batch}),
                self.game_id.as_deref(),
            )?;
            self.batch.clear();
        }
        let winner = self.w.e.winner();
        emit(
            serde_json::json!({
                "type":"done",
                "version":1,
                "id":self.task_id,
                "status":if winner>=0{"completed"}else if self.interrupted{"interrupted"}else{"cutoff"},
                "winner":winner,
                "turns":self.w.e.state.turns,
                "actions":self.actions,
                "searched_decisions":self.decisions,
                "simulations":self.sims,
                "elapsed_seconds":self.started.elapsed().as_secs_f64(),
                "final_state":self.w.e.state,
                "rng_state":self.actual.state()
            }),
            self.game_id.as_deref(),
        )?;
        Ok(PoolStep::Finished)
    }

    fn set_reply(&mut self, outputs: Vec<(Vec<f32>, f32)>) {
        if let Some(dec) = self.active_decision.as_mut() {
            dec.reply = Some(outputs);
        }
    }
}

struct PoolSlot {
    token: u64,
    game: PoolGame,
    pending_outputs: Vec<Option<(Vec<f32>, f32)>>,
    pending_left: usize,
    ready_to_step: bool,
}

fn remove_slot_by_token_with_error(active: &mut Vec<PoolSlot>, token: u64, err: &str) {
    if let Some(pos) = active.iter().position(|s| s.token == token) {
        let st = active.swap_remove(pos);
        let _ = emit(
            serde_json::json!({"type":"error","error":err}),
            st.game.game_id.as_deref(),
        );
    }
}

fn apply_tagged_pool_reply(
    active: &mut Vec<PoolSlot>,
    tag: u64,
    slot_idx: usize,
    res: Result<(Vec<f32>, f32), String>,
) {
    let Some(pos) = active.iter().position(|s| s.token == tag) else {
        return;
    };
    match res {
        Ok(val) => {
            let st = &mut active[pos];
            if slot_idx < st.pending_outputs.len() && st.pending_outputs[slot_idx].is_none() {
                st.pending_outputs[slot_idx] = Some(val);
                st.pending_left = st.pending_left.saturating_sub(1);
                if st.pending_left == 0 {
                    let outputs = st
                        .pending_outputs
                        .drain(..)
                        .map(|v| v.expect("missing GPU reply slot"))
                        .collect();
                    st.game.set_reply(outputs);
                    st.ready_to_step = true;
                }
            }
        }
        Err(err) => {
            let st = active.swap_remove(pos);
            let _ = emit(
                serde_json::json!({"type":"error","error":err}),
                st.game.game_id.as_deref(),
            );
        }
    }
}

fn serve_pool(
    workers: usize,
    backend: batch_gpu::InProcessBackend,
    max_batch: usize,
    wait_us: u64,
    fixed_batch: usize,
    streams: usize,
) -> Result<(), String> {
    let batch_queue = batch_gpu::BatchQueue::start(backend, max_batch, wait_us, fixed_batch, streams);
    let work_queue = Arc::new((
        Mutex::new((VecDeque::<(Request, batch_gpu::ActiveGameGuard)>::new(), false)),
        Condvar::new(),
    ));
    let cpu_threads = pool_thread_count(workers);

    let mut handles = Vec::with_capacity(cpu_threads);
    for i in 0..cpu_threads {
        let wq = Arc::clone(&work_queue);
        let client = batch_queue.client();
        let slot_cap = (workers / cpu_threads) + usize::from(i < (workers % cpu_threads));
        let handle = std::thread::Builder::new()
            .name(format!("catan-worker-{i}"))
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let mut models_cache = HashMap::new();
                let provider = CandidateProvider::Queue(client.clone());
                let (reply_tx, reply_rx) =
                    std::sync::mpsc::channel::<(u64, usize, Result<(Vec<f32>, f32), String>)>();
                let mut active: Vec<PoolSlot> = Vec::with_capacity(slot_cap);
                let mut next_token: u64 = 1;
                let mut closed = false;
                loop {
                    {
                        let (lock, cvar) = &*wq;
                        let mut guard = lock.lock().unwrap();
                        while active.is_empty() && guard.0.is_empty() && !guard.1 {
                            guard = cvar.wait(guard).unwrap();
                        }
                        while active.len() < slot_cap {
                            if let Some((req, active_guard)) = guard.0.pop_front() {
                                let gid = req.game_id.clone();
                                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    PoolGame::new(req, &mut models_cache, &provider)
                                })) {
                                    Ok(Ok(mut g)) => {
                                        g._active_guard = Some(active_guard);
                                        let token = next_token;
                                        next_token = next_token.wrapping_add(1).max(1);
                                        active.push(PoolSlot {
                                            token,
                                            game: g,
                                            pending_outputs: Vec::new(),
                                            pending_left: 0,
                                            ready_to_step: true,
                                        });
                                    }
                                    Ok(Err(error)) => {
                                        let _ = emit(
                                            serde_json::json!({"type":"error","error":error}),
                                            gid.as_deref(),
                                        );
                                    }
                                    Err(payload) => {
                                        let _ = emit(
                                            serde_json::json!({"type":"error","error":format!("Worker thread panicked: {}", panic_message(payload))}),
                                            gid.as_deref(),
                                        );
                                    }
                                }
                            } else {
                                break;
                            }
                        }
                        if guard.1 && guard.0.is_empty() {
                            closed = true;
                        }
                        if !guard.0.is_empty() {
                            cvar.notify_one();
                        }
                    }

                    if active.is_empty() && closed {
                        break;
                    }

                    let mut to_submit: Vec<(u64, String, Option<String>, Vec<Vec<f32>>)> = Vec::new();
                    let mut idx = 0;
                    while idx < active.len() {
                        if !active[idx].ready_to_step {
                            idx += 1;
                            continue;
                        }
                        active[idx].ready_to_step = false;
                        let gid = active[idx].game.game_id.clone();
                        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            active[idx].game.advance_until_eval(&mut models_cache, &provider)
                        }));
                        match res {
                            Ok(Ok(PoolStep::NeedEval {
                                model_path,
                                fallback_model,
                                features,
                            })) => {
                                let count = features.len();
                                if count == 0 {
                                    active[idx].game.set_reply(Vec::new());
                                    active[idx].ready_to_step = true;
                                } else {
                                    active[idx].pending_outputs = vec![None; count];
                                    active[idx].pending_left = count;
                                    to_submit.push((
                                        active[idx].token,
                                        model_path,
                                        fallback_model,
                                        features,
                                    ));
                                }
                                idx += 1;
                            }
                            Ok(Ok(PoolStep::Finished)) => {
                                active.swap_remove(idx);
                            }
                            Ok(Err(error)) => {
                                let _ = emit(
                                    serde_json::json!({"type":"error","error":error}),
                                    gid.as_deref(),
                                );
                                active.swap_remove(idx);
                            }
                            Err(payload) => {
                                let _ = emit(
                                    serde_json::json!({"type":"error","error":format!("Worker thread panicked: {}", panic_message(payload))}),
                                    gid.as_deref(),
                                );
                                active.swap_remove(idx);
                            }
                        }
                    }

                    for (token, model_path, fallback_model, features) in to_submit {
                        if let Err(err) = client.submit_tagged_with_fallback(
                            &model_path,
                            fallback_model.as_deref(),
                            features,
                            token,
                            &reply_tx,
                        ) {
                            remove_slot_by_token_with_error(&mut active, token, &err);
                        }
                    }

                    while let Ok((tag, slot_idx, res)) = reply_rx.try_recv() {
                        apply_tagged_pool_reply(&mut active, tag, slot_idx, res);
                    }

                    if !active.is_empty() && !active.iter().any(|s| s.ready_to_step) {
                        if !closed && active.len() < slot_cap {
                            let (lock, _) = &*wq;
                            let guard = lock.lock().unwrap();
                            if !guard.0.is_empty() {
                                continue;
                            }
                            if guard.1 {
                                closed = true;
                            }
                        }
                        match reply_rx.recv() {
                            Ok((tag, slot_idx, res)) => {
                                apply_tagged_pool_reply(&mut active, tag, slot_idx, res);
                                while let Ok((tag, slot_idx, res)) = reply_rx.try_recv() {
                                    apply_tagged_pool_reply(&mut active, tag, slot_idx, res);
                                }
                            }
                            Err(_) => {
                                for st in active.drain(..) {
                                    let _ = emit(
                                        serde_json::json!({"type":"error","error":"GPU reply channel closed unexpectedly"}),
                                        st.game.game_id.as_deref(),
                                    );
                                }
                                break;
                            }
                        }
                    }
                }
            })
            .map_err(|e| format!("Failed to spawn pool worker {i}: {e}"))?;
        handles.push(handle);
    }

    emit(
        serde_json::json!({
            "type":"ready",
            "version":1,
            "backend":backend.as_str(),
            "threads":cpu_threads,
            "workers":workers,
            "max_batch":max_batch,
            "wait_us":wait_us,
            "streams":streams
        }),
        None,
    )?;

    for line in io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Request>(&line) {
            Ok(req) => {
                let guard_item = batch_queue.active_guard();
                let (lock, cvar) = &*work_queue;
                let mut guard = lock.lock().unwrap();
                guard.0.push_back((req, guard_item));
                cvar.notify_one();
            }
            Err(e) => {
                emit(
                    serde_json::json!({"type":"error","error":e.to_string()}),
                    None,
                )?;
            }
        }
    }

    {
        let (lock, cvar) = &*work_queue;
        let mut guard = lock.lock().unwrap();
        guard.1 = true;
        cvar.notify_all();
    }
    for h in handles {
        let _ = h.join();
    }
    let metrics = batch_queue.shutdown();
    emit(
        serde_json::json!({"type":"pool_metrics","metrics":metrics}),
        None,
    )?;
    Ok(())
}

#[derive(Deserialize)]
struct BridgeCommand {
    op: String,
    request_id: u64,
    game_id: String,
    #[serde(default)]
    task: Option<Task>,
    #[serde(default)]
    learner_seat: Option<usize>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    dynamic_model: Option<String>,
    #[serde(default)]
    frozen: Option<String>,
    #[serde(default)]
    simulations: Option<usize>,
    #[serde(default)]
    lanes: Option<usize>,
    #[serde(default)]
    virtual_loss: Option<f64>,
    #[serde(default)]
    eval_mode: bool,
    #[serde(default)]
    adversarial_puct: bool,
    #[serde(default)]
    mcts_opponent: Option<MctsOpponent>,
    #[serde(default)]
    setup_simulations: Option<usize>,
    #[serde(default)]
    setup_temperature: Option<f64>,
    #[serde(default)]
    vp_margin_alpha: Option<f64>,
    #[serde(default)]
    action_id: Option<usize>,
    #[serde(default)]
    chance: Option<i32>,
}

struct BridgeGame {
    enc: Encoder,
    world: Arc<World>,
    learner_seat: usize,
    search_rng: rng::Rng,
    action_rng: pcg::Pcg,
    tree: Option<search::Search>,
    model: String,
    dynamic_model: Option<String>,
    frozen: String,
    options: Options,
    setup_simulations: Option<usize>,
    setup_temperature: Option<f64>,
    eval_mode: bool,
    _active_guard: Option<batch_gpu::ActiveGameGuard>,
}

struct BridgeTask {
    req_id: u64,
    game_arc: Arc<Mutex<BridgeGame>>,
    key: (String, String, u32),
    dynamic_model: Option<String>,
    need_frozen: bool,
    enc: Encoder,
    stepper: Option<search::SearchStepper>,
    infer_start: Option<Instant>,
    reply: Option<Vec<(Vec<f32>, f32)>>,
}

fn emit_bridge_reply(req_id: u64, res: Result<(usize, Vec<(usize, usize)>), String>) {
    match res {
        Ok((action_id, visits)) => {
            let _ = emit(
                serde_json::json!({"type":"reply","request_id":req_id,"ok":true,"action_id":action_id,"visits":visits}),
                None,
            );
        }
        Err(err) => {
            let _ = emit(
                serde_json::json!({"type":"reply","request_id":req_id,"ok":false,"error":err}),
                None,
            );
        }
    }
}

fn init_bridge_task(
    req_id: u64,
    game_arc: Arc<Mutex<BridgeGame>>,
    cache: &mut HashMap<(String, String, u32), search::Models>,
    provider: &CandidateProvider,
) -> Result<Result<BridgeTask, (usize, Vec<(usize, usize)>)>, String> {
    let mut g = game_arc.lock().unwrap();
    if g.world.e.state.current != g.learner_seat {
        return Err("Not learner turn in bridge choose".into());
    }
    if g.world.e.actions.len() == 1 {
        return Ok(Err((g.enc.id(&g.world.e, g.world.e.actions[0]), vec![])));
    }
    let key = (g.model.clone(), g.frozen.clone(), 0);
    let dynamic_model = g.dynamic_model.clone();
    let need_frozen = g.options.mcts_opponent == MctsOpponent::FrozenMlp;
    ensure_models(cache, &[key.clone(), key.clone()], provider, [need_frozen, need_frozen])?;
    let seed = g.search_rng.u64();
    let tree = if let Some(old) = g.tree.take() {
        old.advance(g.world.clone(), &g.enc)
    } else {
        search::Search::new(g.world.clone(), seed)
    };
    let opt = Options {
        simulations: if g.world.e.state.initial {
            g.setup_simulations.or(g.options.simulations)
        } else {
            g.options.simulations
        },
        seconds: g.options.seconds,
        noise: g.options.noise,
        lanes: g.options.lanes,
        virtual_loss: g.options.virtual_loss,
        mcts_opponent: g.options.mcts_opponent,
        vp_margin_alpha: g.options.vp_margin_alpha,
    };
    let enc = Encoder::new(Arc::clone(&g.enc.job));
    drop(g);
    let stepper = search::SearchStepper::new(tree, seed, &opt, None, false);
    Ok(Ok(BridgeTask {
        req_id,
        game_arc,
        key,
        dynamic_model,
        need_frozen,
        enc,
        stepper: Some(stepper),
        infer_start: None,
        reply: None,
    }))
}

fn serve_bridge_pool(
    workers: usize,
    backend: batch_gpu::InProcessBackend,
    max_batch: usize,
    wait_us: u64,
    fixed_batch: usize,
    streams: usize,
) -> Result<(), String> {
    let batch_queue = batch_gpu::BatchQueue::start(backend, max_batch, wait_us, fixed_batch, streams);
    let sessions: Arc<Mutex<HashMap<String, Arc<Mutex<BridgeGame>>>>> = Arc::new(Mutex::new(HashMap::new()));
    let choose_queue = Arc::new((
        Mutex::new((VecDeque::<(u64, Arc<Mutex<BridgeGame>>)>::new(), false)),
        Condvar::new(),
    ));
    let cpu_threads = pool_thread_count(workers);

    let mut handles = Vec::with_capacity(cpu_threads);
    for i in 0..cpu_threads {
        let cq = Arc::clone(&choose_queue);
        let client = batch_queue.client();
        let slot_cap = (workers / cpu_threads) + usize::from(i < (workers % cpu_threads));
        let handle = std::thread::Builder::new()
            .name(format!("catan-bridge-{i}"))
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let mut cache: HashMap<(String, String, u32), search::Models> = HashMap::new();
                let provider = CandidateProvider::Queue(client.clone());
                let mut active: Vec<BridgeTask> = Vec::with_capacity(slot_cap);
                let mut closed = false;
                loop {
                    {
                        let (lock, cvar) = &*cq;
                        let mut guard = lock.lock().unwrap();
                        while active.is_empty() && guard.0.is_empty() && !guard.1 {
                            guard = cvar.wait(guard).unwrap();
                        }
                        while active.len() < slot_cap {
                            if let Some((req_id, game_arc)) = guard.0.pop_front() {
                                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    init_bridge_task(req_id, game_arc, &mut cache, &provider)
                                })) {
                                    Ok(Ok(Ok(task))) => active.push(task),
                                    Ok(Ok(Err(immediate))) => emit_bridge_reply(req_id, Ok(immediate)),
                                    Ok(Err(err)) => emit_bridge_reply(req_id, Err(err)),
                                    Err(payload) => emit_bridge_reply(
                                        req_id,
                                        Err(format!("Bridge worker panicked: {}", panic_message(payload))),
                                    ),
                                }
                            } else {
                                break;
                            }
                        }
                        if guard.1 && guard.0.is_empty() {
                            closed = true;
                        }
                        if !guard.0.is_empty() {
                            cvar.notify_one();
                        }
                    }

                    if active.is_empty() && closed {
                        break;
                    }

                    let mut pending_by_model: HashMap<(String, Option<String>), Vec<(usize, Vec<Vec<f32>>)>> =
                        HashMap::new();
                    let mut idx = 0;
                    while idx < active.len() {
                        let req_id = active[idx].req_id;
                        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Option<(String, Option<String>, Vec<Vec<f32>>)>, String> {
                            let t = &mut active[idx];
                            ensure_models(
                                &mut cache,
                                &[t.key.clone(), t.key.clone()],
                                &provider,
                                [t.need_frozen, t.need_frozen],
                            )?;
                            if let Some(started) = t.infer_start.take() {
                                if let Some(s) = t.stepper.as_mut() {
                                    s.add_inference_seconds(started.elapsed().as_secs_f64());
                                }
                            }
                            let stepper = t.stepper.take().unwrap();
                            let models = cache.get_mut(&t.key).unwrap();
                            match stepper.step(&t.enc, models, t.reply.take())? {
                                search::StepOutcome::NeedCandidate { stepper, features } => {
                                    t.stepper = Some(stepper);
                                    t.infer_start = Some(Instant::now());
                                    Ok(Some((t.key.0.clone(), t.dynamic_model.clone(), features)))
                                }
                                search::StepOutcome::Done { mut search, result } => {
                                    let mut g = t.game_arc.lock().unwrap();
                                    let selected = if g.eval_mode {
                                        result.action_id
                                    } else {
                                        let temp = if g.world.e.state.initial {
                                            g.setup_temperature.unwrap_or(1.)
                                        } else if g.world.e.state.turns < 20 {
                                            1.
                                        } else {
                                            0.25
                                        };
                                        if result.statistics.iter().any(|x| x.visits > 0) {
                                            search::choose_temperature(&result.statistics, temp, &mut g.action_rng)
                                        } else {
                                            result.action_id
                                        }
                                    };
                                    let visits = result
                                        .statistics
                                        .iter()
                                        .filter(|x| x.visits > 0)
                                        .map(|x| (x.action_id, x.visits))
                                        .collect::<Vec<_>>();
                                    search.selected = Some(selected);
                                    g.tree = Some(search);
                                    drop(g);
                                    emit_bridge_reply(req_id, Ok((selected, visits)));
                                    Ok(None)
                                }
                            }
                        }));
                        match res {
                            Ok(Ok(Some((model_path, fallback_model, features)))) => {
                                pending_by_model.entry((model_path, fallback_model)).or_default().push((idx, features));
                                idx += 1;
                            }
                            Ok(Ok(None)) => {
                                active.swap_remove(idx);
                            }
                            Ok(Err(err)) => {
                                emit_bridge_reply(req_id, Err(err));
                                active.swap_remove(idx);
                            }
                            Err(payload) => {
                                emit_bridge_reply(
                                    req_id,
                                    Err(format!("Bridge worker panicked: {}", panic_message(payload))),
                                );
                                active.swap_remove(idx);
                            }
                        }
                    }

                    if !pending_by_model.is_empty() {
                        let mut submitted = Vec::with_capacity(pending_by_model.len());
                        let mut failed_indices: Vec<(usize, String)> = Vec::new();
                        for ((model_path, fallback_model), entries) in pending_by_model {
                            let mut flat = Vec::new();
                            let mut counts = Vec::with_capacity(entries.len());
                            for (_, feats) in &entries {
                                counts.push(feats.len());
                                flat.extend(feats.iter().cloned());
                            }
                            match client.submit_batch_with_fallback(
                                &model_path,
                                fallback_model.as_deref(),
                                flat,
                            ) {
                                Ok(handle) => submitted.push((entries, counts, handle)),
                                Err(err) => {
                                    for (task_idx, _) in entries {
                                        failed_indices.push((task_idx, err.clone()));
                                    }
                                }
                            }
                        }
                        for (entries, counts, handle) in submitted {
                            match handle.wait() {
                                Ok(outputs) => {
                                    let mut iter = outputs.into_iter();
                                    for ((task_idx, _), count) in entries.into_iter().zip(counts) {
                                        let chunk: Vec<_> = iter.by_ref().take(count).collect();
                                        active[task_idx].reply = Some(chunk);
                                    }
                                }
                                Err(err) => {
                                    for (task_idx, _) in entries {
                                        failed_indices.push((task_idx, err.clone()));
                                    }
                                }
                            }
                        }
                        if !failed_indices.is_empty() {
                            failed_indices.sort_by(|a, b| b.0.cmp(&a.0));
                            for (task_idx, err) in failed_indices {
                                let req_id = active[task_idx].req_id;
                                emit_bridge_reply(req_id, Err(err));
                                active.swap_remove(task_idx);
                            }
                        }
                    }
                }
            })
            .map_err(|e| format!("Failed to spawn bridge worker {i}: {e}"))?;
        handles.push(handle);
    }

    emit(
        serde_json::json!({"type":"ready","version":1,"backend":backend.as_str(),"mode":"bridge-pool","workers":workers,"max_batch":max_batch,"wait_us":wait_us,"streams":streams}),
        None,
    )?;

    for line in io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let cmd: BridgeCommand = match serde_json::from_str(&line) {
            Ok(c) => c,
            Err(e) => {
                emit(serde_json::json!({"type":"error","error":e.to_string()}), None)?;
                continue;
            }
        };
        match cmd.op.as_str() {
            "init" => {
                let res: Result<(), String> = (|| {
                    let t = cmd.task.ok_or("Missing task in init")?;
                    let seat = cmd.learner_seat.ok_or("Missing learner_seat in init")?;
                    let model = cmd.model.ok_or("Missing model in init")?;
                    let frozen = cmd.frozen.ok_or("Missing frozen in init")?;
                    let sims = cmd.simulations.ok_or("Missing simulations in init")?;
                    if cmd.setup_simulations == Some(0) {
                        return Err("setup_simulations must be >= 1".into());
                    }
                    if cmd.setup_temperature.is_some_and(|t| !t.is_finite() || t <= 0.) {
                        return Err("setup_temperature must be finite and > 0".into());
                    }
                    if cmd
                        .vp_margin_alpha
                        .is_some_and(|a| !a.is_finite() || !(0.0..=0.5).contains(&a))
                    {
                        return Err("vp_margin_alpha must be finite and between 0.0 and 0.5".into());
                    }
                    let enc = Encoder::new(Arc::new(t.job.clone()));
                    let world = Arc::new(task_world(&t));
                    let game = BridgeGame {
                        enc,
                        world,
                        learner_seat: seat,
                        search_rng: rng::Rng::new(rng::named(t.seat_seeds[seat], "search")),
                        action_rng: pcg::Pcg::new(rng::named(t.seat_seeds[seat], "action-sampling")),
                        tree: None,
                        model,
                        dynamic_model: cmd.dynamic_model,
                        frozen,
                        options: Options {
                            simulations: Some(sims),
                            seconds: None,
                            noise: !cmd.eval_mode,
                            lanes: cmd.lanes.unwrap_or(1).max(1),
                            virtual_loss: cmd.virtual_loss.unwrap_or(1.0),
                            mcts_opponent: cmd.mcts_opponent.unwrap_or(if cmd.adversarial_puct {
                                MctsOpponent::AdversarialPuct
                            } else {
                                MctsOpponent::FrozenMlp
                            }),
                            vp_margin_alpha: cmd.vp_margin_alpha.unwrap_or(0.03),
                        },
                        setup_simulations: cmd.setup_simulations,
                        setup_temperature: cmd.setup_temperature,
                        eval_mode: cmd.eval_mode,
                        _active_guard: Some(batch_queue.active_guard()),
                    };
                    sessions
                        .lock()
                        .unwrap()
                        .insert(cmd.game_id, Arc::new(Mutex::new(game)));
                    Ok(())
                })();
                match res {
                    Ok(()) => emit(serde_json::json!({"type":"reply","request_id":cmd.request_id,"ok":true}), None)?,
                    Err(e) => emit(serde_json::json!({"type":"reply","request_id":cmd.request_id,"ok":false,"error":e}), None)?,
                }
            }
            "choose" => {
                let maybe = sessions.lock().unwrap().get(&cmd.game_id).cloned();
                if let Some(game_arc) = maybe {
                    let (lock, cvar) = &*choose_queue;
                    let mut guard = lock.lock().unwrap();
                    guard.0.push_back((cmd.request_id, game_arc));
                    cvar.notify_one();
                } else {
                    emit(
                        serde_json::json!({"type":"reply","request_id":cmd.request_id,"ok":false,"error":"Unknown game_id"}),
                        None,
                    )?;
                }
            }
            "step" => {
                let res: Result<(), String> = (|| {
                    let action_id = cmd.action_id.ok_or("Missing action_id in step")?;
                    let chance = cmd.chance.ok_or("Missing chance in step")?;
                    let game_arc = sessions
                        .lock()
                        .unwrap()
                        .get(&cmd.game_id)
                        .cloned()
                        .ok_or("Unknown game_id in step")?;
                    let mut g = game_arc.lock().unwrap();
                    let a = *g
                        .world
                        .e
                        .actions
                        .iter()
                        .find(|&&a| g.enc.id(&g.world.e, a) == action_id)
                        .ok_or_else(|| format!("Illegal step action_id {action_id}"))?;
                    let mut step_w = g.world.step(a, chance);
                    step_w.logs.compact();
                    g.world = Arc::new(step_w);
                    Ok(())
                })();
                match res {
                    Ok(()) => emit(serde_json::json!({"type":"reply","request_id":cmd.request_id,"ok":true}), None)?,
                    Err(e) => emit(serde_json::json!({"type":"reply","request_id":cmd.request_id,"ok":false,"error":e}), None)?,
                }
            }
            "close" => {
                sessions.lock().unwrap().remove(&cmd.game_id);
                emit(serde_json::json!({"type":"reply","request_id":cmd.request_id,"ok":true}), None)?;
            }
            other => {
                emit(
                    serde_json::json!({"type":"reply","request_id":cmd.request_id,"ok":false,"error":format!("Unknown bridge op {other}")}),
                    None,
                )?;
            }
        }
    }

    {
        let (lock, cvar) = &*choose_queue;
        let mut guard = lock.lock().unwrap();
        guard.1 = true;
        cvar.notify_all();
    }
    for h in handles {
        let _ = h.join();
    }
    let metrics = batch_queue.shutdown();
    emit(
        serde_json::json!({"type":"pool_metrics","metrics":metrics}),
        None,
    )?;
    Ok(())
}
