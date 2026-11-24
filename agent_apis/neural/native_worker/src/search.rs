use super::*;
use std::collections::HashSet;
type Key = [u8; 32];
#[derive(Clone, Hash, Eq, PartialEq, Serialize, Deserialize, Debug, Ord, PartialOrd)]
struct Step {
    id: usize,
    visible: Key,
    denied: u8,
}
type TreePath = Vec<Step>;
#[derive(Clone)]
struct Class {
    rep: i32,
    p: f64,
    denied: u8,
    key: Key,
}
struct Child {
    w: Arc<World>,
    n: usize,
    total: f64,
}
struct CW {
    classes: Vec<Class>,
    children: HashMap<Key, Child>,
}
struct Edge {
    a: Act,
    prior: f64,
    n: usize,
    avail: usize,
    total: f64,
    vl_n: usize,
    vl_loss: f64,
    worlds: HashMap<Key, CW>,
}
struct Node {
    obs: Key,
    n: usize,
    vl_n: usize,
    edges: Vec<usize>,
}
impl Edge {
    fn mean(&self) -> f64 {
        if self.n == 0 {
            0.
        } else {
            self.total / self.n as f64
        }
    }
}
#[derive(Default, Serialize, Clone)]
pub struct Metrics {
    pub simulations: usize,
    pub evaluations: usize,
    pub opponent_evaluations: usize,
    pub capped_simulations: usize,
    pub max_action_depth: usize,
    pub max_root_depth: usize,
    pub max_completed_turn_depth: usize,
    pub chance_distributions: usize,
    pub raw_outcomes: usize,
    pub merged_classes: usize,
    pub retained_chance_children: usize,
    pub reused_nodes: usize,
    pub reused_root_visits: usize,
    pub reuse_statistics_reset: bool,
    pub reset_reason: String,
    pub belief_evidence: f64,
    pub seconds: f64,
    pub feature_seconds: f64,
    pub inference_seconds: f64,
    pub chance_enumeration_seconds: f64,
    pub posterior_seconds: f64,
    pub root_noise: Vec<f64>,
}
#[derive(Serialize, Clone)]
pub struct Row {
    pub action_id: usize,
    pub visits: usize,
    pub mean_value: f64,
    pub prior: f64,
}
#[derive(Serialize)]
pub struct ResultData {
    pub action_id: usize,
    pub sampled_action: usize,
    pub statistics: Vec<Row>,
    pub metrics: Metrics,
    pub allocations: Vec<belief::Allocation>,
}
pub struct Models {
    pub candidate: batch_gpu::CandidateModel,
    pub opponent: Session,
}
impl Models {
    pub fn predict_candidate(&mut self, features: Vec<f32>) -> Result<(Vec<f32>, f32), String> {
        self.candidate.predict(features)
    }
    pub fn predict_candidate_batch(
        &mut self,
        features_batch: Vec<Vec<f32>>,
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
        self.candidate.predict_batch(features_batch)
    }
    pub fn predict_opponent(&mut self, features: Vec<f32>) -> Result<(Vec<f32>, f32), String> {
        predict(&mut self.opponent, features).map_err(|e| e.to_string())
    }
}
pub struct Search {
    own: usize,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    paths: HashMap<TreePath, usize>,
    op_paths: HashMap<TreePath, usize>,
    op_edges: HashMap<(TreePath, usize), usize>,
    positions: HashMap<TreePath, Arc<World>>,
    root: Arc<World>,
    pub selected: Option<usize>,
    m: Metrics,
    chance: rng::Rng,
    opponent: rng::Rng,
    world: rng::Rng,
    templates: HashMap<String, Arc<World>>,
    allocations: Vec<belief::Allocation>,
}
impl Search {
    pub fn new(root: Arc<World>, seed: u64) -> Self {
        Self {
            own: root.e.state.current,
            nodes: vec![],
            edges: vec![],
            paths: HashMap::new(),
            op_paths: HashMap::new(),
            op_edges: HashMap::new(),
            positions: HashMap::new(),
            root,
            selected: None,
            m: Metrics::default(),
            chance: rng::Rng::new(rng::named(seed, "chance")),
            opponent: rng::Rng::new(rng::named(seed, "opponent")),
            world: rng::Rng::new(rng::named(seed, "worlds")),
            templates: HashMap::new(),
            allocations: vec![],
        }
    }
    fn prediction(
        &mut self,
        w: &World,
        enc: &Encoder,
        models: &mut Models,
        opponent: bool,
    ) -> (Vec<(Act, usize, f64)>, f64) {
        let actor = w.e.state.current;
        let mut actions: Vec<_> = w.e.actions.iter().map(|&a| (a, enc.id(&w.e, a))).collect();
        actions.sort_by_key(|x| x.1);
        if opponent && actions.len() == 1 {
            return (vec![(actions[0].0, actions[0].1, 1.)], 0.5);
        }
        if opponent {
            self.m.opponent_evaluations += 1;
        } else {
            self.m.evaluations += 1;
        }
        let feature_start = Instant::now();
        let features = w.features(enc, actor).clone();
        self.m.feature_seconds += feature_start.elapsed().as_secs_f64();
        let inference_start = Instant::now();
        let (scores, value) = if opponent {
            models.predict_opponent(features)
        } else {
            models.predict_candidate(features)
        }
        .unwrap();
        self.m.inference_seconds += inference_start.elapsed().as_secs_f64();
        assert!(scores.iter().all(|x| x.is_finite()) && value.is_finite());
        let ids: Vec<_> = actions.iter().map(|x| x.1).collect();
        let priors = masked(&scores, &ids);
        let value = (1. / (1. + (-(value as f64)).exp())) as f32 as f64;
        (
            actions
                .into_iter()
                .zip(priors)
                .map(|((a, i), p)| (a, i, p))
                .collect(),
            if actor == self.own { value } else { 1. - value },
        )
    }
    fn edge(&mut self, a: Act, p: f64) -> usize {
        let i = self.edges.len();
        self.edges.push(Edge {
            a,
            prior: p,
            n: 0,
            avail: 0,
            total: 0.,
            vl_n: 0,
            vl_loss: 0.,
            worlds: HashMap::new(),
        });
        i
    }
    fn node(&mut self, w: &World, enc: &Encoder, models: &mut Models) -> (usize, f64) {
        let (pred, value) = self.prediction(w, enc, models, false);
        let edges = pred.into_iter().map(|(a, _, p)| self.edge(a, p)).collect();
        let i = self.nodes.len();
        self.nodes.push(Node {
            obs: w.key(enc, self.own),
            n: 0,
            vl_n: 0,
            edges,
        });
        (i, value)
    }
    fn backup_trace(&mut self, trace: &[(Option<usize>, usize, Key, Key)], v: f64, vl_loss: f64) {
        for &(node, edge, wk, ck) in trace.iter().rev() {
            if let Some(i) = node {
                self.edges[edge].vl_n -= 1;
                self.edges[edge].vl_loss -= vl_loss;
                self.nodes[i].vl_n -= 1;
                self.nodes[i].n += 1;
            }
            self.edges[edge].n += 1;
            self.edges[edge].total += v;
            let c = self.edges[edge]
                .worlds
                .get_mut(&wk)
                .unwrap()
                .children
                .get_mut(&ck)
                .unwrap();
            c.n += 1;
            c.total += v;
        }
    }
    fn select_leaf(
        &mut self,
        mut w: Arc<World>,
        enc: &Encoder,
        models: &mut Models,
        vl_loss: f64,
        mcts_opponent: MctsOpponent,
    ) -> Option<(Arc<World>, TreePath, Vec<(Option<usize>, usize, Key, Key)>, Vec<f32>)> {
        let mut path: TreePath = vec![];
        let mut trace: Vec<(Option<usize>, usize, Key, Key)> = vec![];
        let mut rd = 0usize;
        let mut depth = 0usize;
        loop {
            if w.e.state.current == self.own && w.e.winner() < 0 {
                self.positions
                    .entry(path.clone())
                    .or_insert_with(|| w.clone());
            }
            self.m.max_action_depth = self.m.max_action_depth.max(depth);
            self.m.max_root_depth = self.m.max_root_depth.max(rd);
            self.m.max_completed_turn_depth = self
                .m
                .max_completed_turn_depth
                .max(w.e.state.turns - self.root.e.state.turns);
            let winner = w.e.winner();
            if winner >= 0 {
                let v = (winner as usize == self.own) as u8 as f64;
                self.backup_trace(&trace, v, vl_loss);
                return None;
            }
            if depth >= 128 {
                self.m.capped_simulations += 1;
                self.backup_trace(&trace, 0.5, vl_loss);
                return None;
            }
            let (edge, node, next_rd) = if w.e.state.current == self.own {
                let ni = if let Some(&ni) = self.paths.get(&path) {
                    ni
                } else {
                    let feature_start = Instant::now();
                    let features = w.features(enc, w.e.state.current).clone();
                    self.m.feature_seconds += feature_start.elapsed().as_secs_f64();
                    return Some((w, path, trace, features));
                };
                assert_eq!(
                    self.nodes[ni].obs,
                    w.key(enc, self.own),
                    "Root path observation alias"
                );
                let parent_n = (self.nodes[ni].n + self.nodes[ni].vl_n).max(1) as f64;
                let score = |i: usize| {
                    let e = &self.edges[i];
                    let n_eff = e.n + e.vl_n;
                    let q = if n_eff == 0 {
                        0.5
                    } else {
                        (e.total - e.vl_loss) / n_eff as f64
                    };
                    q + 1.5 * e.prior * parent_n.sqrt() / (1 + n_eff) as f64
                };
                let ei = *self.nodes[ni]
                    .edges
                    .iter()
                    .max_by(|&&a, &&b| {
                        score(a).total_cmp(&score(b)).then_with(|| {
                            enc.id(&w.e, self.edges[b].a)
                                .cmp(&enc.id(&w.e, self.edges[a].a))
                        })
                    })
                    .unwrap();
                self.edges[ei].vl_n += 1;
                self.edges[ei].vl_loss += vl_loss;
                self.nodes[ni].vl_n += 1;
                (ei, Some(ni), rd + 1)
            } else {
                let (selected_act, selected_id, selected_p) = if w.e.actions.len() == 1 {
                    let a = w.e.actions[0];
                    (a, enc.id(&w.e, a), 1.0)
                } else {
                    let use_frozen = mcts_opponent == MctsOpponent::FrozenMlp;
                    let (pred, _) = self.prediction(&w, enc, models, use_frozen);
                    let mut threshold = self.opponent.uniform();
                    let mut selected = *pred.last().unwrap();
                    for t in pred {
                        threshold -= t.2;
                        if threshold < 0. {
                            selected = t;
                            break;
                        }
                    }
                    selected
                };
                let k = (path.clone(), selected_id);
                let ei = if let Some(&i) = self.op_edges.get(&k) {
                    i
                } else {
                    let i = self.edge(selected_act, selected_p);
                    self.op_edges.insert(k, i);
                    i
                };
                (ei, None, rd)
            };
            let (child, wk, ck, denied) = self.step(&w, edge);
            path.push(Step {
                id: enc.id(&w.e, self.edges[edge].a),
                visible: visible(&child, enc, self.own),
                denied,
            });
            trace.push((node, edge, wk, ck));
            w = child;
            rd = next_rd;
            depth += 1;
        }
    }
    fn shortages(w: &World, sum: i32) -> u8 {
        if sum == 7 {
            return 0;
        }
        let mut totals = [0; 5];
        for (i, t) in w.e.map.tiles.iter().enumerate() {
            if i == w.e.state.robber || t.number != sum || t.resource < 0 {
                continue;
            }
            for &n in &t.nodes {
                let b = w.e.state.buildings[n];
                if b[0] >= 0 {
                    totals[t.resource as usize] += b[1];
                }
            }
        }
        let mut bits = 0;
        for r in 0..5 {
            if totals[r] > w.e.state.bank[r] {
                bits |= 1 << r;
            }
        }
        bits
    }
    fn classes(w: &World, a: Act) -> (Vec<Class>, usize) {
        let mut raw = vec![];
        match a.k {
            1 => {
                for d1 in 1..=6 {
                    for d2 in 1..=6 {
                        raw.push((d1 + d2, 1. / 36.));
                    }
                }
            }
            5 => {
                for r in [0, 2, 1, 3, 4] {
                    let n = w.e.state.deck.iter().filter(|&&d| d == r).count();
                    if n > 0 {
                        raw.push((r as i32, n as f64 / w.e.state.deck.len() as f64));
                    }
                }
            }
            7 if a.b >= 0 => {
                let q = &w.e.state.players[a.b as usize];
                let total = q.hand.iter().sum::<i32>() as f64;
                for r in 0..5 {
                    if q.hand[r] > 0 {
                        raw.push((r as i32, q.hand[r] as f64 / total));
                    }
                }
            }
            _ => raw.push((-1, 1.)),
        }
        let count = raw.len();
        let mut classes: Vec<Class> = vec![];
        for (rep, p) in raw {
            let denied = if a.k == 1 { Self::shortages(w, rep) } else { 0 };
            let child = w.step(a, rep);
            let key = hash(&(child.canonical(), denied));
            if let Some(c) = classes.iter_mut().find(|c| c.key == key) {
                c.p += p;
            } else {
                classes.push(Class {
                    rep,
                    p,
                    denied,
                    key,
                });
            }
        }
        assert!((classes.iter().map(|x| x.p).sum::<f64>() - 1.).abs() < 1e-12);
        (classes, count)
    }
    fn step(&mut self, w: &World, edge: usize) -> (Arc<World>, Key, Key, u8) {
        let key = w.canonical();
        if !self.edges[edge].worlds.contains_key(&key) {
            let class_start = Instant::now();
            let (classes, raw) = Self::classes(w, self.edges[edge].a);
            self.m.chance_enumeration_seconds += class_start.elapsed().as_secs_f64();
            self.m.chance_distributions += 1;
            self.m.raw_outcomes += raw;
            self.m.merged_classes += classes.len();
            self.edges[edge].worlds.insert(
                key,
                CW {
                    classes,
                    children: HashMap::new(),
                },
            );
        }
        let action = self.edges[edge].a;
        let cw = self.edges[edge].worlds.get_mut(&key).unwrap();
        let mut threshold = self.chance.uniform();
        let mut selected = cw.classes.last().unwrap().clone();
        for c in &cw.classes {
            threshold -= c.p;
            if threshold < 0. {
                selected = c.clone();
                break;
            }
        }
        if !cw.children.contains_key(&selected.key) {
            cw.children.insert(
                selected.key,
                Child {
                    w: Arc::new(w.step(action, selected.rep)),
                    n: 0,
                    total: 0.,
                },
            );
            self.m.retained_chance_children += 1;
        }
        let child = self.edges[edge].worlds[&key].children[&selected.key]
            .w
            .clone();
        (child, key, selected.key, selected.denied)
    }
    fn simulate(
        &mut self,
        w: Arc<World>,
        path: TreePath,
        rd: usize,
        depth: usize,
        enc: &Encoder,
        models: &mut Models,
        mcts_opponent: MctsOpponent,
    ) -> f64 {
        if w.e.state.current == self.own && w.e.winner() < 0 {
            self.positions
                .entry(path.clone())
                .or_insert_with(|| w.clone());
        }
        self.m.max_action_depth = self.m.max_action_depth.max(depth);
        self.m.max_root_depth = self.m.max_root_depth.max(rd);
        self.m.max_completed_turn_depth = self
            .m
            .max_completed_turn_depth
            .max(w.e.state.turns - self.root.e.state.turns);
        let winner = w.e.winner();
        if winner >= 0 {
            return (winner as usize == self.own) as u8 as f64;
        }
        if depth >= 128 {
            self.m.capped_simulations += 1;
            return 0.5;
        }
        let (edge, node, next_rd) = if w.e.state.current == self.own {
            let ni = if let Some(&ni) = self.paths.get(&path) {
                ni
            } else {
                let (ni, v) = self.node(&w, enc, models);
                self.paths.insert(path, ni);
                self.nodes[ni].n += 1;
                return v;
            };
            assert_eq!(
                self.nodes[ni].obs,
                w.key(enc, self.own),
                "Root path observation alias"
            );
            let actual: HashSet<_> = w.e.actions.iter().map(|&a| enc.id(&w.e, a)).collect();
            let expected: HashSet<_> = self.nodes[ni]
                .edges
                .iter()
                .map(|&i| enc.id(&w.e, self.edges[i].a))
                .collect();
            assert_eq!(actual, expected, "Hidden world changed legal mask");
            let score = |i: usize| {
                let e = &self.edges[i];
                (if e.n == 0 { 0.5 } else { e.mean() })
                    + 1.5 * e.prior * (self.nodes[ni].n.max(1) as f64).sqrt() / (1 + e.n) as f64
            };
            let ei = *self.nodes[ni]
                .edges
                .iter()
                .max_by(|&&a, &&b| {
                    score(a).total_cmp(&score(b)).then_with(|| {
                        enc.id(&w.e, self.edges[b].a)
                            .cmp(&enc.id(&w.e, self.edges[a].a))
                    })
                })
                .unwrap();
            (ei, Some(ni), rd + 1)
        } else if mcts_opponent == MctsOpponent::AdversarialPuct {
            if w.e.actions.len() == 1 {
                let a = w.e.actions[0];
                let aid = enc.id(&w.e, a);
                let k = (path.clone(), aid);
                let ei = if let Some(&i) = self.op_edges.get(&k) {
                    i
                } else {
                    let i = self.edge(a, 1.0);
                    self.op_edges.insert(k, i);
                    i
                };
                (ei, None, rd)
            } else {
                // Cached-prior no-stop adversarial IS-MCTS PUCT: evaluate candidate policy
                // once to cache priors on the opponent node, then continue descending to
                // own next leaf so opponent turns do not cut lookahead depth in half.
                let ni = if let Some(&ni) = self.op_paths.get(&path) {
                    let existing_acts: Vec<Act> = self.nodes[ni]
                        .edges
                        .iter()
                        .map(|&i| self.edges[i].a)
                        .collect();
                    if w.e.actions.iter().any(|a| !existing_acts.contains(a)) {
                        let (pred, _) = self.prediction(&w, enc, models, false);
                        for (a, _, p) in pred {
                            if !existing_acts.contains(&a) {
                                let ei = self.edge(a, p);
                                self.nodes[ni].edges.push(ei);
                            }
                        }
                    }
                    ni
                } else {
                    let (pred, _) = self.prediction(&w, enc, models, false);
                    let edges = pred.into_iter().map(|(a, _, p)| self.edge(a, p)).collect();
                    let ni = self.nodes.len();
                    self.nodes.push(Node {
                        obs: w.key(enc, self.own),
                        n: 0,
                        vl_n: 0,
                        edges,
                    });
                    self.op_paths.insert(path.clone(), ni);
                    ni
                };
                let legal_edges: Vec<usize> = self.nodes[ni]
                    .edges
                    .iter()
                    .copied()
                    .filter(|&i| w.e.actions.contains(&self.edges[i].a))
                    .collect();
                for &i in &legal_edges {
                    self.edges[i].avail += 1;
                }
                let score = |i: usize| {
                    let e = &self.edges[i];
                    let q_opp = if e.n == 0 { 0.5 } else { 1.0 - e.mean() };
                    q_opp + 1.5 * e.prior * (e.avail.max(1) as f64).sqrt() / (1 + e.n) as f64
                };
                let ei = *legal_edges
                    .iter()
                    .max_by(|&&a, &&b| {
                        score(a).total_cmp(&score(b)).then_with(|| {
                            enc.id(&w.e, self.edges[b].a)
                                .cmp(&enc.id(&w.e, self.edges[a].a))
                        })
                    })
                    .unwrap();
                (ei, Some(ni), rd)
            }
        } else {
            let (selected_act, selected_id, selected_p) = if w.e.actions.len() == 1 {
                let a = w.e.actions[0];
                (a, enc.id(&w.e, a), 1.0)
            } else {
                let use_frozen = mcts_opponent == MctsOpponent::FrozenMlp;
                let (pred, _) = self.prediction(&w, enc, models, use_frozen);
                let mut threshold = self.opponent.uniform();
                let mut selected = *pred.last().unwrap();
                for t in pred {
                    threshold -= t.2;
                    if threshold < 0. {
                        selected = t;
                        break;
                    }
                }
                selected
            };
            let k = (path.clone(), selected_id);
            let ei = if let Some(&i) = self.op_edges.get(&k) {
                i
            } else {
                let i = self.edge(selected_act, selected_p);
                self.op_edges.insert(k, i);
                i
            };
            (ei, None, rd)
        };
        let (child, wk, ck, denied) = self.step(&w, edge);
        let mut next = path;
        next.push(Step {
            id: enc.id(&w.e, self.edges[edge].a),
            visible: visible(&child, enc, self.own),
            denied,
        });
        let v = self.simulate(child, next, next_rd, depth + 1, enc, models, mcts_opponent);
        self.edges[edge].n += 1;
        self.edges[edge].total += v;
        let c = self.edges[edge]
            .worlds
            .get_mut(&wk)
            .unwrap()
            .children
            .get_mut(&ck)
            .unwrap();
        c.n += 1;
        c.total += v;
        if let Some(i) = node {
            self.nodes[i].n += 1;
        }
        v
    }
    pub fn run(
        &mut self,
        enc: &Encoder,
        models: &mut Models,
        seed: u64,
        o: &Options,
        noise_override: Option<&[f64]>,
        audit_allocations: bool,
    ) -> ResultData {
        let start = Instant::now();
        let previous = self.m.clone();
        self.m = Metrics::default();
        self.m.reused_nodes = previous.reused_nodes;
        self.m.reused_root_visits = previous.reused_root_visits;
        self.m.reuse_statistics_reset = previous.reuse_statistics_reset;
        self.m.reset_reason = previous.reset_reason;
        self.chance = rng::Rng::new(rng::named(seed, "chance"));
        self.world = rng::Rng::new(rng::named(seed, "worlds"));
        self.opponent = rng::Rng::new(rng::named(seed, "opponent"));
        self.templates.clear();
        self.allocations.clear();
        let root = self.root.clone();
        if root.e.actions.len() == 1 {
            let id = enc.id(&root.e, root.e.actions[0]);
            return ResultData {
                action_id: id,
                sampled_action: id,
                statistics: vec![],
                metrics: self.m.clone(),
                allocations: vec![],
            };
        }
        let ni = if let Some(&n) = self.paths.get(&vec![]) {
            n
        } else {
            let (n, _) = self.node(&root, enc, models);
            self.paths.insert(vec![], n);
            n
        };
        // Same two evaluations on a fresh tree as ExploratorySearch: create node, refresh priors.
        let (pred, _) = self.prediction(&root, enc, models, false);
        let noise = if !o.noise {
            vec![0.; pred.len()]
        } else if let Some(x) = noise_override {
            assert_eq!(x.len(), pred.len());
            x.to_vec()
        } else {
            pcg::Pcg::new(rng::named(seed, "root-noise")).dirichlet(pred.len())
        };
        self.m.root_noise = noise.clone();
        for ((_, id, p), n) in pred.iter().zip(noise) {
            let ei = *self.nodes[ni]
                .edges
                .iter()
                .find(|&&i| enc.id(&root.e, self.edges[i].a) == *id)
                .unwrap();
            self.edges[ei].prior = if o.noise { 0.75 * p + 0.25 * n } else { *p };
        }
        let posterior_start = Instant::now();
        let mut belief = belief::Belief::new(&root, self.own);
        self.m.posterior_seconds += posterior_start.elapsed().as_secs_f64();
        self.m.belief_evidence = belief.evidence;
        let mut longest = 0f64;
        let lanes = if o.mcts_opponent == MctsOpponent::AdversarialPuct {
            1
        } else {
            o.lanes.max(1)
        };
        if lanes == 1 {
            loop {
                if o.simulations.is_some_and(|n| self.m.simulations >= n) {
                    break;
                }
                if let Some(seconds) = o.seconds {
                    if start.elapsed().as_secs_f64() + 1.25 * longest >= seconds {
                        break;
                    }
                }
                let begin = Instant::now();
                let posterior_start = Instant::now();
                let allocation = belief.sample(&mut self.world);
                self.m.posterior_seconds += posterior_start.elapsed().as_secs_f64();
                let key = serde_json::to_string(&allocation).unwrap();
                let w = if let Some(w) = self.templates.get(&key) {
                    w.clone()
                } else {
                    let w = Arc::new(root.sampled(&allocation, self.own));
                    if self.templates.len() < 128 {
                        self.templates.insert(key, w.clone());
                    }
                    w
                };
                if audit_allocations {
                    self.allocations.push(allocation);
                }
                self.simulate(w, vec![], 0, 0, enc, models, o.mcts_opponent);
                self.m.simulations += 1;
                longest = longest.max(begin.elapsed().as_secs_f64());
            }
        } else {
            loop {
                if o.simulations.is_some_and(|n| self.m.simulations >= n) {
                    break;
                }
                if let Some(seconds) = o.seconds {
                    if start.elapsed().as_secs_f64() + 1.25 * longest >= seconds {
                        break;
                    }
                }
                let begin = Instant::now();
                let remaining = o
                    .simulations
                    .map(|n| n.saturating_sub(self.m.simulations))
                    .unwrap_or(lanes);
                let wave_target = lanes.min(remaining);
                let mut pending = Vec::with_capacity(wave_target);
                let mut features_batch = Vec::with_capacity(wave_target);
                for _ in 0..wave_target {
                    let posterior_start = Instant::now();
                    let allocation = belief.sample(&mut self.world);
                    self.m.posterior_seconds += posterior_start.elapsed().as_secs_f64();
                    let key = serde_json::to_string(&allocation).unwrap();
                    let w = if let Some(w) = self.templates.get(&key) {
                        w.clone()
                    } else {
                        let w = Arc::new(root.sampled(&allocation, self.own));
                        if self.templates.len() < 128 {
                            self.templates.insert(key, w.clone());
                        }
                        w
                    };
                    if audit_allocations {
                        self.allocations.push(allocation);
                    }
                    if let Some((lw, path, trace, feat)) =
                        self.select_leaf(w, enc, models, o.virtual_loss, o.mcts_opponent)
                    {
                        pending.push((lw, path, trace));
                        features_batch.push(feat);
                    } else {
                        self.m.simulations += 1;
                    }
                }
                if !pending.is_empty() {
                    self.m.evaluations += pending.len();
                    let inference_start = Instant::now();
                    let outputs = models.predict_candidate_batch(features_batch).unwrap();
                    self.m.inference_seconds += inference_start.elapsed().as_secs_f64();
                    for ((lw, path, trace), (scores, raw_val)) in
                        pending.into_iter().zip(outputs.into_iter())
                    {
                        assert!(scores.iter().all(|x| x.is_finite()) && raw_val.is_finite());
                        let actor = lw.e.state.current;
                        let val = (1. / (1. + (-(raw_val as f64)).exp())) as f32 as f64;
                        let v = if actor == self.own { val } else { 1. - val };
                        if let Some(&existing_ni) = self.paths.get(&path) {
                            self.nodes[existing_ni].n += 1;
                        } else {
                            let mut actions: Vec<_> = lw
                                .e
                                .actions
                                .iter()
                                .map(|&a| (a, enc.id(&lw.e, a)))
                                .collect();
                            actions.sort_by_key(|x| x.1);
                            let ids: Vec<_> = actions.iter().map(|x| x.1).collect();
                            let priors = masked(&scores, &ids);
                            let edges = actions
                                .into_iter()
                                .zip(priors)
                                .map(|((a, _), p)| self.edge(a, p))
                                .collect();
                            let new_ni = self.nodes.len();
                            self.nodes.push(Node {
                                obs: lw.key(enc, self.own),
                                n: 1,
                                vl_n: 0,
                                edges,
                            });
                            self.paths.insert(path, new_ni);
                        }
                        self.backup_trace(&trace, v, o.virtual_loss);
                        self.m.simulations += 1;
                    }
                }
                longest = longest.max(begin.elapsed().as_secs_f64());
            }
            assert!(self.edges.iter().all(|e| e.vl_n == 0));
            assert!(self.nodes.iter().all(|n| n.vl_n == 0));
        }
        let mut rows: Vec<_> = self.nodes[ni]
            .edges
            .iter()
            .map(|&i| {
                let e = &self.edges[i];
                Row {
                    action_id: enc.id(&root.e, e.a),
                    visits: e.n,
                    mean_value: e.mean(),
                    prior: e.prior,
                }
            })
            .collect();
        rows.sort_by_key(|r| r.action_id);
        let selected = rows
            .iter()
            .max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| a.mean_value.total_cmp(&b.mean_value))
                    .then_with(|| b.action_id.cmp(&a.action_id))
            })
            .unwrap()
            .action_id;
        self.selected = Some(selected);
        self.m.seconds = start.elapsed().as_secs_f64();
        ResultData {
            action_id: selected,
            sampled_action: selected,
            statistics: rows,
            metrics: self.m.clone(),
            allocations: self.allocations.clone(),
        }
    }
    pub fn advance(mut self, root: Arc<World>, enc: &Encoder) -> Self {
        let mut reason = "compatible-subtree";
        let delta = root.logs.len().saturating_sub(self.root.logs.len());
        let matches_history = delta > 0
            && root.logs.len() >= self.root.logs.len()
            && root
                .logs
                .iter()
                .zip(self.root.logs.iter())
                .all(|(a, b)| log_same(a, b, self.own));
        if !matches_history {
            reason = "history-rewind-or-divergence";
        }
        // Rates can change after the first new action; derive its canonical ID on the old root.
        let matches_history = if matches_history {
            let matched =
                self.selected == Some(enc.id(&self.root.e, root.logs[self.root.logs.len()].a));
            if !matched {
                reason = "different-played-action";
            }
            matched
        } else {
            false
        };
        let mut candidates: Vec<_> = if matches_history {
            self.positions
                .iter()
                .filter(|(p, w)| {
                    p.len() == delta
                        && w.key(enc, self.own) == root.key(enc, self.own)
                        && same_logs(&w.logs, &root.logs, self.own)
                })
                .map(|(p, _)| p.clone())
                .collect()
        } else {
            vec![]
        };
        candidates.retain(|p| self.paths.contains_key(p));
        candidates.sort_by(|a, b| {
            self.nodes[self.paths[a]]
                .n
                .cmp(&self.nodes[self.paths[b]].n)
                .then_with(|| repr_path(a).cmp(&repr_path(b)))
        });
        if let Some(prefix) = candidates.last() {
            let candidate_roots: Vec<_> = candidates
                .iter()
                .map(|p| self.positions[p].clone())
                .collect();
            let prefix = prefix.clone();
            let mut paths = HashMap::new();
            for (p, &n) in &self.paths {
                if p.starts_with(&prefix) {
                    paths.insert(p[prefix.len()..].to_vec(), n);
                }
            }
            let mut op_paths = HashMap::new();
            for (p, &n) in &self.op_paths {
                if p.starts_with(&prefix) {
                    op_paths.insert(p[prefix.len()..].to_vec(), n);
                }
            }
            if paths.len() + op_paths.len() <= 4096 {
                self.paths = paths;
                self.op_paths = op_paths;
                self.op_edges = self
                    .op_edges
                    .into_iter()
                    .filter(|((p, _), _)| p.starts_with(&prefix))
                    .map(|((p, a), e)| ((p[prefix.len()..].to_vec(), a), e))
                    .collect();
                self.positions = self
                    .positions
                    .into_iter()
                    .filter(|(p, _)| p.starts_with(&prefix))
                    .map(|(p, w)| (p[prefix.len()..].to_vec(), w))
                    .collect();
                let reset = root.logs.iter().any(|l| l.p != self.own && l.a.k == 5)
                    || self.m.capped_simulations > 0;
                let mut used = HashSet::new();
                for &n in self.paths.values().chain(self.op_paths.values()) {
                    used.extend(self.nodes[n].edges.iter().copied());
                }
                used.extend(self.op_edges.values().copied());
                let mut live = HashSet::new();
                for w in &candidate_roots {
                    live.insert(Arc::as_ptr(w));
                }
                for w in self.positions.values() {
                    live.insert(Arc::as_ptr(w));
                }
                for &e in &used {
                    for cw in self.edges[e].worlds.values() {
                        for c in cw.children.values() {
                            live.insert(Arc::as_ptr(&c.w));
                        }
                    }
                }
                let states = live.len();
                if states <= 8192 {
                    if reset {
                        for &n in self.paths.values().chain(self.op_paths.values()) {
                            self.nodes[n].n = 0;
                        }
                        for &i in &used {
                            let e = &mut self.edges[i];
                            e.n = 0;
                            e.avail = 0;
                            e.total = 0.;
                            for cw in e.worlds.values_mut() {
                                for c in cw.children.values_mut() {
                                    c.n = 0;
                                    c.total = 0.;
                                }
                            }
                        }
                    }
                    let mut node_ids: Vec<_> = self
                        .paths
                        .values()
                        .chain(self.op_paths.values())
                        .copied()
                        .collect();
                    node_ids.sort();
                    node_ids.dedup();
                    let node_map: HashMap<_, _> = node_ids
                        .iter()
                        .enumerate()
                        .map(|(new, &old)| (old, new))
                        .collect();
                    let mut edge_ids: Vec<_> = used.iter().copied().collect();
                    edge_ids.sort();
                    let edge_map: HashMap<_, _> = edge_ids
                        .iter()
                        .enumerate()
                        .map(|(new, &old)| (old, new))
                        .collect();
                    self.nodes = std::mem::take(&mut self.nodes)
                        .into_iter()
                        .enumerate()
                        .filter_map(|(old, mut n)| {
                            if node_map.contains_key(&old) {
                                for e in &mut n.edges {
                                    *e = edge_map[e];
                                }
                                Some(n)
                            } else {
                                None
                            }
                        })
                        .collect();
                    self.edges = std::mem::take(&mut self.edges)
                        .into_iter()
                        .enumerate()
                        .filter_map(|(old, e)| {
                            if edge_map.contains_key(&old) {
                                Some(e)
                            } else {
                                None
                            }
                        })
                        .collect();
                    for n in self.paths.values_mut() {
                        *n = node_map[n];
                    }
                    for n in self.op_paths.values_mut() {
                        *n = node_map[n];
                    }
                    for e in self.op_edges.values_mut() {
                        *e = edge_map[e];
                    }
                    self.root = root;
                    self.templates.clear();
                    self.allocations.clear();
                    self.m.reused_nodes = self.paths.len();
                    self.m.reused_root_visits = self.nodes[self.paths[&vec![]]].n;
                    self.m.reuse_statistics_reset = reset;
                    self.m.reset_reason = if reset {
                        "hidden-belief-or-horizon-reset".into()
                    } else {
                        "compatible-subtree".into()
                    };
                    return self;
                }
                reason = "retained-state-limit";
            } else {
                reason = "retained-node-limit";
            }
        } else if matches_history {
            reason = "unexplored-observation";
        }
        let mut fresh = Self::new(root, 11);
        fresh.m.reset_reason = reason.into();
        fresh
    }
}
fn log_same(a: &Log, b: &Log, own: usize) -> bool {
    a.p == b.p
        && a.a == b.a
        && a.vp == b.vp
        && (!(a.a.k == 5 || a.a.k == 7) || a.p != own || a.ch == b.ch)
}
fn same_logs(a: &[Log], b: &[Log], own: usize) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| log_same(a, b, own))
}
pub fn actual_chance(w: &World, a: Act, rng: &mut rng::Rng) -> (i32, Option<[i32; 2]>) {
    match a.k {
        1 => {
            let d = [(rng.below(6) + 1) as i32, (rng.below(6) + 1) as i32];
            (d[0] + d[1], Some(d))
        }
        5 => (*w.e.state.deck.last().unwrap() as i32, None),
        7 if a.b >= 0 => {
            let hand = w.e.state.players[a.b as usize].hand;
            let mut x = rng.below(hand.iter().sum::<i32>() as usize) as i32;
            let mut r = 0;
            while x >= hand[r] {
                x -= hand[r];
                r += 1;
            }
            (r as i32, None)
        }
        _ => (-1, None),
    }
}
pub fn choose_temperature(rows: &[Row], temperature: f64, rng: &mut pcg::Pcg) -> usize {
    let max = rows
        .iter()
        .filter(|r| r.visits > 0)
        .map(|r| (r.visits as f64).ln() / temperature)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = rows
        .iter()
        .map(|r| {
            if r.visits > 0 {
                ((r.visits as f64).ln() / temperature - max).exp()
            } else {
                0.
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    assert!(total > 0.);
    let mut t = rng.uniform() * total;
    for (r, p) in rows.iter().zip(weights) {
        t -= p;
        if t < 0. {
            return r.action_id;
        }
    }
    rows.last().unwrap().action_id
}
fn repr_path(path: &TreePath) -> String {
    let mut parts = vec![];
    for s in path {
        let mut names: Vec<_> = (0..5)
            .filter(|r| s.denied & (1 << r) != 0)
            .map(|r| R[r])
            .collect();
        names.sort();
        let inner = names
            .iter()
            .map(|n| format!("'{n}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let denied = if names.len() == 1 {
            format!("({inner},)")
        } else {
            format!("({inner})")
        };
        parts.push(format!("({}, '{}', {})", s.id, hex(s.visible), denied));
    }
    let inner = parts.join(", ");
    if parts.len() == 1 {
        format!("({inner},)")
    } else {
        format!("({inner})")
    }
}
