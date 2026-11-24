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
    let mut backend = batch_gpu::InProcessBackend::CoreML;
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
                    .ok_or_else(|| "--gpu-backend requires coreml or cpu-batch".to_string())?;
                backend = match v.as_str() {
                    "coreml" => batch_gpu::InProcessBackend::CoreML,
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

fn play(
    r: Request,
    cache: &mut HashMap<(String, String, u32), search::Models>,
    provider: &CandidateProvider,
) -> Result<(), String> {
    let game_id = r.game_id.clone();
    if r.version != 1
        || r.simulations.is_some() == r.seconds.is_some()
        || r.simulations == Some(0)
        || r.setup_simulations == Some(0)
        || r.setup_temperature.is_some_and(|t| !t.is_finite() || t <= 0.)
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
    let model_ids = r.model_ids.unwrap_or([0, 0]);
    let keys = [
        (r.models[0].clone(), r.frozen.clone(), model_ids[0]),
        (r.models[1].clone(), r.frozen.clone(), model_ids[1]),
    ];
    if keys.iter().any(|key| !cache.contains_key(key)) && cache.len() + 2 > 4 {
        cache.clear();
    }
    for key in &keys {
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
            let opponent = session(Path::new(&key.1)).map_err(|e| e.to_string())?;
            cache.insert(
                key.clone(),
                search::Models {
                    candidate,
                    opponent,
                },
            );
        }
    }
    let default_opp = r.mcts_opponent.unwrap_or(if r.adversarial_puct {
        MctsOpponent::AdversarialPuct
    } else {
        MctsOpponent::FrozenMlp
    });
    let seat_opts: [Options; 2] = std::array::from_fn(|i| {
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
        }
    });
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
            };
            let result = tree.run(
                &enc,
                cache.get_mut(&keys[actor]).unwrap(),
                seed,
                &opt,
                None,
                false,
            );
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
        w = Arc::new(w.step(a, chance));
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
        } => {
            let batch_queue =
                batch_gpu::BatchQueue::start(backend, max_batch, wait_us, fixed_batch, streams);
            let work_queue = Arc::new((Mutex::new((VecDeque::<Request>::new(), false)), Condvar::new()));

            let mut handles = Vec::with_capacity(workers);
            for i in 0..workers {
                let wq = Arc::clone(&work_queue);
                let client = batch_queue.client();
                let handle = std::thread::Builder::new()
                    .name(format!("catan-worker-{i}"))
                    .stack_size(32 * 1024 * 1024)
                    .spawn(move || {
                        let mut models = HashMap::new();
                        let provider = CandidateProvider::Queue(client);
                        loop {
                            let req = {
                                let (lock, cvar) = &*wq;
                                let mut guard = lock.lock().unwrap();
                                while guard.0.is_empty() && !guard.1 {
                                    guard = cvar.wait(guard).unwrap();
                                }
                                match guard.0.pop_front() {
                                    Some(r) => r,
                                    None => break,
                                }
                            };
                            let gid = req.game_id.clone();
                            if let Err(error) = play(req, &mut models, &provider) {
                                let _ = emit(
                                    serde_json::json!({"type":"error","error":error}),
                                    gid.as_deref(),
                                );
                            }
                        }
                    })
                    .map_err(|e| format!("Failed to spawn pool worker {i}: {e}"))?;
                handles.push(handle);
            }

            emit(
                serde_json::json!({"type":"ready","version":1,"backend":backend.as_str(),"threads":1,"workers":workers,"max_batch":max_batch,"wait_us":wait_us,"streams":streams}),
                None,
            )?;

            for line in io::stdin().lock().lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Request>(&line) {
                    Ok(req) => {
                        let (lock, cvar) = &*work_queue;
                        let mut guard = lock.lock().unwrap();
                        guard.0.push_back(req);
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
    frozen: String,
    options: Options,
    setup_simulations: Option<usize>,
    setup_temperature: Option<f64>,
    eval_mode: bool,
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

    let mut handles = Vec::with_capacity(workers);
    for i in 0..workers {
        let cq = Arc::clone(&choose_queue);
        let client = batch_queue.client();
        let handle = std::thread::Builder::new()
            .name(format!("catan-bridge-{i}"))
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let mut cache: HashMap<(String, String), search::Models> = HashMap::new();
                loop {
                    let (req_id, game_arc) = {
                        let (lock, cvar) = &*cq;
                        let mut guard = lock.lock().unwrap();
                        while guard.0.is_empty() && !guard.1 {
                            guard = cvar.wait(guard).unwrap();
                        }
                        match guard.0.pop_front() {
                            Some(item) => item,
                            None => break,
                        }
                    };
                    let res: Result<(usize, Vec<(usize, usize)>), String> = (|| {
                        let mut g = game_arc.lock().unwrap();
                        if g.world.e.state.current != g.learner_seat {
                            return Err("Not learner turn in bridge choose".into());
                        }
                        if g.world.e.actions.len() == 1 {
                            return Ok((g.enc.id(&g.world.e, g.world.e.actions[0]), vec![]));
                        }
                        let key = (g.model.clone(), g.frozen.clone());
                        if !cache.contains_key(&key) {
                            let opponent = session(Path::new(&key.1)).map_err(|e| e.to_string())?;
                            cache.insert(
                                key.clone(),
                                search::Models {
                                    candidate: batch_gpu::CandidateModel::InProcessQueue {
                                        client: client.clone(),
                                        model_path: key.0.clone(),
                                    },
                                    opponent,
                                },
                            );
                        }
                        let seed = g.search_rng.u64();
                        let mut tree = if let Some(old) = g.tree.take() {
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
                        };
                        let result = tree.run(
                            &g.enc,
                            cache.get_mut(&key).unwrap(),
                            seed,
                            &opt,
                            None,
                            false,
                        );
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
                        tree.selected = Some(selected);
                        g.tree = Some(tree);
                        Ok((selected, visits))
                    })();
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
                        },
                        setup_simulations: cmd.setup_simulations,
                        setup_temperature: cmd.setup_temperature,
                        eval_mode: cmd.eval_mode,
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
                    g.world = Arc::new(g.world.step(a, chance));
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
