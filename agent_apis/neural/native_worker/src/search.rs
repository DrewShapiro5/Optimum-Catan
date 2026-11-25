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
    value: f64,
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
    pub reused_opponent_nodes: usize,
    pub reused_world_states: usize,
    pub reused_root_visits: usize,
    pub tree_nodes: usize,
    pub tree_edges: usize,
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
    pub opponent: Option<Session>,
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
        let session = self
            .opponent
            .as_mut()
            .ok_or_else(|| "Frozen opponent session was not loaded".to_string())?;
        predict(session, features).map_err(|e| e.to_string())
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
    fn extract_actions(w: &World, enc: &Encoder) -> Vec<(Act, usize)> {
        let mut actions: Vec<_> = w.e.actions.iter().map(|&a| (a, enc.id(&w.e, a))).collect();
        actions.sort_by_key(|x| x.1);
        actions
    }
    fn process_prediction(
        &self,
        w: &World,
        enc: &Encoder,
        scores: Vec<f32>,
        raw_val: f32,
    ) -> Result<(Vec<(Act, usize, f64)>, f64), String> {
        if !scores.iter().all(|x| x.is_finite()) || !raw_val.is_finite() {
            return Err("Nonfinite model prediction".to_string());
        }
        let actor = w.e.state.current;
        let actions = Self::extract_actions(w, enc);
        let ids: Vec<_> = actions.iter().map(|x| x.1).collect();
        let priors = masked(&scores, &ids);
        let value = (1. / (1. + (-(raw_val as f64)).exp())) as f32 as f64;
        Ok((
            actions
                .into_iter()
                .zip(priors)
                .map(|((a, i), p)| (a, i, p))
                .collect(),
            if actor == self.own { value } else { 1. - value },
        ))
    }
    fn prediction(
        &mut self,
        w: &World,
        enc: &Encoder,
        models: &mut Models,
        opponent: bool,
    ) -> Result<(Vec<(Act, usize, f64)>, f64), String> {
        let actor = w.e.state.current;
        let actions = Self::extract_actions(w, enc);
        if opponent && actions.len() == 1 {
            return Ok((vec![(actions[0].0, actions[0].1, 1.)], 0.5));
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
        }?;
        self.m.inference_seconds += inference_start.elapsed().as_secs_f64();
        self.process_prediction(w, enc, scores, value)
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
    fn utility(w: &World, own: usize, p: f64, vp_margin_alpha: f64) -> f64 {
        if vp_margin_alpha <= 0.0 {
            return p;
        }
        let own_vp = w.e.state.players[own].actual as f64;
        let opp_vp = w.e.state.players[1 - own].actual as f64;
        let margin = 0.5 + 0.5 * ((own_vp - opp_vp) / 6.0).tanh();
        let sat = (2.0 * p - 1.0).powi(2);
        let weight = (vp_margin_alpha * sat).clamp(0.0, 1.0);
        (1.0 - weight) * p + weight * margin
    }
    fn expand_node(
        &mut self,
        w: &World,
        enc: &Encoder,
        pred: Vec<(Act, usize, f64)>,
        value: f64,
    ) -> usize {
        let edges = pred.into_iter().map(|(a, _, p)| self.edge(a, p)).collect();
        let i = self.nodes.len();
        self.nodes.push(Node {
            obs: w.key(enc, self.own),
            n: 0,
            vl_n: 0,
            value,
            edges,
        });
        i
    }
    fn pick_own_puct_edge(&self, w: &World, enc: &Encoder, ni: usize, use_vl: bool) -> usize {
        let parent_n = if use_vl {
            (self.nodes[ni].n + self.nodes[ni].vl_n).max(1) as f64
        } else {
            self.nodes[ni].n.max(1) as f64
        };
        let mut visited_n = 0usize;
        let mut visited_total = 0.0f64;
        let mut visited_prior = 0.0f64;
        for &i in &self.nodes[ni].edges {
            let e = &self.edges[i];
            if e.n > 0 {
                visited_n += e.n;
                visited_total += e.total;
                visited_prior += e.prior;
            }
        }
        let parent_q = if visited_n > 0 {
            visited_total / visited_n as f64
        } else {
            self.nodes[ni].value
        };
        let fpu = (parent_q - 0.20 * visited_prior.sqrt()).clamp(0.0, 1.0);
        let score = |i: usize| {
            let e = &self.edges[i];
            if use_vl {
                let n_eff = e.n + e.vl_n;
                let q = if n_eff == 0 {
                    fpu
                } else {
                    (e.total - e.vl_loss) / n_eff as f64
                };
                q + 1.5 * e.prior * parent_n.sqrt() / (1 + n_eff) as f64
            } else {
                (if e.n == 0 { fpu } else { e.mean() })
                    + 1.5 * e.prior * parent_n.sqrt() / (1 + e.n) as f64
            }
        };
        *self.nodes[ni]
            .edges
            .iter()
            .max_by(|&&a, &&b| {
                score(a).total_cmp(&score(b)).then_with(|| {
                    enc.id(&w.e, self.edges[b].a)
                        .cmp(&enc.id(&w.e, self.edges[a].a))
                })
            })
            .unwrap()
    }
    fn pick_adv_opp_edge(&mut self, w: &World, enc: &Encoder, ni: usize) -> usize {
        let legal_edges: Vec<usize> = self.nodes[ni]
            .edges
            .iter()
            .copied()
            .filter(|&i| w.e.actions.contains(&self.edges[i].a))
            .collect();
        let mut visited_n = 0usize;
        let mut visited_total = 0.0f64;
        let mut visited_prior = 0.0f64;
        for &i in &legal_edges {
            self.edges[i].avail += 1;
            let e = &self.edges[i];
            if e.n > 0 {
                visited_n += e.n;
                visited_total += e.total;
                visited_prior += e.prior;
            }
        }
        let opp_parent_q = if visited_n > 0 {
            1.0 - visited_total / visited_n as f64
        } else {
            1.0 - self.nodes[ni].value
        };
        let fpu_opp = (opp_parent_q - 0.20 * visited_prior.sqrt()).clamp(0.0, 1.0);
        let score = |i: usize| {
            let e = &self.edges[i];
            let q_opp = if e.n == 0 { fpu_opp } else { 1.0 - e.mean() };
            q_opp + 1.5 * e.prior * (e.avail.max(1) as f64).sqrt() / (1 + e.n) as f64
        };
        *legal_edges
            .iter()
            .max_by(|&&a, &&b| {
                score(a).total_cmp(&score(b)).then_with(|| {
                    enc.id(&w.e, self.edges[b].a)
                        .cmp(&enc.id(&w.e, self.edges[a].a))
                })
            })
            .unwrap()
    }
    fn sample_opp_edge(&mut self, path: &TreePath, pred: Vec<(Act, usize, f64)>) -> usize {
        let mut threshold = self.opponent.uniform();
        let mut selected = *pred.last().unwrap();
        for t in pred {
            threshold -= t.2;
            if threshold < 0. {
                selected = t;
                break;
            }
        }
        let k = (path.clone(), selected.1);
        if let Some(&i) = self.op_edges.get(&k) {
            i
        } else {
            let i = self.edge(selected.0, selected.2);
            self.op_edges.insert(k, i);
            i
        }
    }
    fn check_own_node_invariants(&self, w: &World, enc: &Encoder, ni: usize) {
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
    }
    fn finalize(&mut self, enc: &Encoder, start: Instant, root: &World) -> ResultData {
        let ni = self.paths[&vec![]];
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
        self.m.tree_nodes = self.nodes.len();
        self.m.tree_edges = self.edges.len();
        self.m.seconds = start.elapsed().as_secs_f64();
        ResultData {
            action_id: selected,
            sampled_action: selected,
            statistics: rows,
            metrics: self.m.clone(),
            allocations: self.allocations.clone(),
        }
    }
    fn backup_trace(
        &mut self,
        trace: &[(Option<usize>, usize, Key, Key)],
        v: f64,
        vl_loss: f64,
        use_vl: bool,
    ) {
        for &(node, edge, wk, ck) in trace.iter().rev() {
            if let Some(i) = node {
                if use_vl {
                    self.edges[edge].vl_n -= 1;
                    self.edges[edge].vl_loss -= vl_loss;
                    self.nodes[i].vl_n -= 1;
                }
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
    fn sample_world(
        &mut self,
        belief: &mut belief::Belief,
        root: &World,
        audit_allocations: bool,
    ) -> Arc<World> {
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
        w
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
            let mut hasher = Sha256::new();
            hasher.update(child.canonical());
            hasher.update([denied]);
            let key: Key = hasher.finalize().into();
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
    pub fn run(
        &mut self,
        enc: &Encoder,
        models: &mut Models,
        seed: u64,
        o: &Options,
        noise_override: Option<&[f64]>,
        audit_allocations: bool,
    ) -> Result<ResultData, String> {
        let placeholder = Search::new(self.root.clone(), 0);
        let moved = std::mem::replace(self, placeholder);
        let mut stepper = SearchStepper::new(moved, seed, o, noise_override, audit_allocations);
        let mut reply = None;
        loop {
            match stepper.step(enc, models, reply)? {
                StepOutcome::Done { search, result } => {
                    *self = search;
                    return Ok(result);
                }
                StepOutcome::NeedCandidate {
                    stepper: mut next_stepper,
                    features,
                } => {
                    let infer_start = Instant::now();
                    let out = models.predict_candidate_batch(features)?;
                    next_stepper.add_inference_seconds(infer_start.elapsed().as_secs_f64());
                    stepper = next_stepper;
                    reply = Some(out);
                }
            }
        }
    }
    pub fn advance(mut self, root: Arc<World>, enc: &Encoder) -> Self {
        let mut reason = "compatible-subtree";
        let delta = root.logs.len().saturating_sub(self.root.logs.len());
        let mut matches_history = delta > 0 && root.logs.len() >= self.root.logs.len();
        if matches_history {
            self.root.logs.for_each(|i, b| {
                if matches_history && !log_same(root.logs.get(i), b, self.own) {
                    matches_history = false;
                }
            });
        }
        if !matches_history {
            reason = "history-rewind-or-divergence";
        }
        // Rates can change after the first new action; derive its canonical ID on the old root.
        let matches_history = if matches_history {
            let matched =
                self.selected == Some(enc.id(&self.root.e, root.logs.get(self.root.logs.len()).a));
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
                let reset = root.logs.any(|l| l.p != self.own && l.a.k == 5)
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
                    self.m.reused_opponent_nodes = self.op_paths.len();
                    self.m.reused_world_states = states;
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
pub(crate) fn same_logs(a: &Logs, b: &Logs, own: usize) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut ok = true;
    a.for_each(|i, la| {
        if ok && !log_same(la, b.get(i), own) {
            ok = false;
        }
    });
    ok
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

pub enum StepOutcome {
    NeedCandidate {
        stepper: SearchStepper,
        features: Vec<Vec<f32>>,
    },
    Done {
        search: Search,
        result: ResultData,
    },
}

struct DescendState {
    w: Arc<World>,
    path: TreePath,
    trace: Vec<(Option<usize>, usize, Key, Key)>,
    rd: usize,
    depth: usize,
}

enum DescendOutcome {
    Finished,
    NeedOwnLeaf {
        w: Arc<World>,
        path: TreePath,
        trace: Vec<(Option<usize>, usize, Key, Key)>,
        features: Vec<f32>,
    },
    NeedAdvOpp {
        state: DescendState,
        existing_ni: Option<usize>,
        features: Vec<f32>,
    },
    NeedSelfOpp {
        state: DescendState,
        features: Vec<f32>,
    },
}

enum StepperPhase {
    Init,
    WaitRootNode,
    WaitRootPriors,
    ReadyToSimulate {
        belief: belief::Belief,
        longest: f64,
    },
    WaitSingleOwnLeaf {
        belief: belief::Belief,
        longest: f64,
        begin: Instant,
        w: Arc<World>,
        path: TreePath,
        trace: Vec<(Option<usize>, usize, Key, Key)>,
    },
    WaitSingleAdvOpp {
        belief: belief::Belief,
        longest: f64,
        begin: Instant,
        state: DescendState,
        existing_ni: Option<usize>,
    },
    WaitSingleSelfOpp {
        belief: belief::Belief,
        longest: f64,
        begin: Instant,
        state: DescendState,
    },
    WaitWaveOpp {
        belief: belief::Belief,
        longest: f64,
        begin: Instant,
        wave_target: usize,
        lane_idx: usize,
        pending: Vec<(Arc<World>, TreePath, Vec<(Option<usize>, usize, Key, Key)>)>,
        features_batch: Vec<Vec<f32>>,
        state: DescendState,
    },
    WaitWaveBatch {
        belief: belief::Belief,
        longest: f64,
        begin: Instant,
        pending: Vec<(Arc<World>, TreePath, Vec<(Option<usize>, usize, Key, Key)>)>,
    },
}

pub struct SearchStepper {
    search: Search,
    root: Arc<World>,
    seed: u64,
    o: Options,
    noise_override: Option<Vec<f64>>,
    audit_allocations: bool,
    start: Instant,
    lanes: usize,
    phase: StepperPhase,
}

impl SearchStepper {
    pub fn new(
        mut search: Search,
        seed: u64,
        o: &Options,
        noise_override: Option<&[f64]>,
        audit_allocations: bool,
    ) -> Self {
        let start = Instant::now();
        let previous = search.m.clone();
        search.m = Metrics::default();
        search.m.reused_nodes = previous.reused_nodes;
        search.m.reused_opponent_nodes = previous.reused_opponent_nodes;
        search.m.reused_world_states = previous.reused_world_states;
        search.m.reused_root_visits = previous.reused_root_visits;
        search.m.reuse_statistics_reset = previous.reuse_statistics_reset;
        search.m.reset_reason = previous.reset_reason;
        search.chance = rng::Rng::new(rng::named(seed, "chance"));
        search.world = rng::Rng::new(rng::named(seed, "worlds"));
        search.opponent = rng::Rng::new(rng::named(seed, "opponent"));
        search.templates.clear();
        search.allocations.clear();
        let root = search.root.clone();
        let lanes = if o.mcts_opponent == MctsOpponent::AdversarialPuct {
            1
        } else {
            o.lanes.max(1)
        };
        Self {
            search,
            root,
            seed,
            o: o.clone(),
            noise_override: noise_override.map(|s| s.to_vec()),
            audit_allocations,
            start,
            lanes,
            phase: StepperPhase::Init,
        }
    }

    pub fn add_inference_seconds(&mut self, secs: f64) {
        self.search.m.inference_seconds += secs;
    }

    fn take_one(reply: Option<Vec<(Vec<f32>, f32)>>) -> Result<(Vec<f32>, f32), String> {
        reply
            .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
            .ok_or_else(|| "Missing candidate inference reply".to_string())
    }

    fn step_descent(
        &mut self,
        enc: &Encoder,
        models: &mut Models,
        mut st: DescendState,
        use_vl: bool,
    ) -> Result<DescendOutcome, String> {
        loop {
            if st.w.e.state.current == self.search.own && st.w.e.winner() < 0 {
                self.search
                    .positions
                    .entry(st.path.clone())
                    .or_insert_with(|| st.w.clone());
            }
            self.search.m.max_action_depth = self.search.m.max_action_depth.max(st.depth);
            self.search.m.max_root_depth = self.search.m.max_root_depth.max(st.rd);
            self.search.m.max_completed_turn_depth = self
                .search
                .m
                .max_completed_turn_depth
                .max(st.w.e.state.turns - self.search.root.e.state.turns);
            let winner = st.w.e.winner();
            if winner >= 0 {
                let raw = (winner as usize == self.search.own) as u8 as f64;
                let v = Search::utility(&st.w, self.search.own, raw, self.o.vp_margin_alpha);
                self.search.backup_trace(&st.trace, v, self.o.virtual_loss, use_vl);
                return Ok(DescendOutcome::Finished);
            }
            if st.depth >= 128 {
                self.search.m.capped_simulations += 1;
                self.search.backup_trace(&st.trace, 0.5, self.o.virtual_loss, use_vl);
                return Ok(DescendOutcome::Finished);
            }
            let (edge, node, next_rd) = if st.w.e.state.current == self.search.own {
                let ni = if let Some(&ni) = self.search.paths.get(&st.path) {
                    ni
                } else {
                    let feature_start = Instant::now();
                    let features = st.w.features(enc, st.w.e.state.current).clone();
                    self.search.m.feature_seconds += feature_start.elapsed().as_secs_f64();
                    return Ok(DescendOutcome::NeedOwnLeaf {
                        w: st.w,
                        path: st.path,
                        trace: st.trace,
                        features,
                    });
                };
                self.search.check_own_node_invariants(&st.w, enc, ni);
                let ei = self.search.pick_own_puct_edge(&st.w, enc, ni, use_vl);
                if use_vl {
                    self.search.edges[ei].vl_n += 1;
                    self.search.edges[ei].vl_loss += self.o.virtual_loss;
                    self.search.nodes[ni].vl_n += 1;
                }
                (ei, Some(ni), st.rd + 1)
            } else if self.o.mcts_opponent == MctsOpponent::AdversarialPuct {
                if st.w.e.actions.len() == 1 {
                    let a = st.w.e.actions[0];
                    let aid = enc.id(&st.w.e, a);
                    let k = (st.path.clone(), aid);
                    let ei = if let Some(&i) = self.search.op_edges.get(&k) {
                        i
                    } else {
                        let i = self.search.edge(a, 1.0);
                        self.search.op_edges.insert(k, i);
                        i
                    };
                    (ei, None, st.rd)
                } else if let Some(&ni) = self.search.op_paths.get(&st.path) {
                    let existing_acts: Vec<Act> = self.search.nodes[ni]
                        .edges
                        .iter()
                        .map(|&i| self.search.edges[i].a)
                        .collect();
                    if st.w.e.actions.iter().any(|a| !existing_acts.contains(a)) {
                        let actor = st.w.e.state.current;
                        let feature_start = Instant::now();
                        let features = st.w.features(enc, actor).clone();
                        self.search.m.feature_seconds += feature_start.elapsed().as_secs_f64();
                        return Ok(DescendOutcome::NeedAdvOpp {
                            state: st,
                            existing_ni: Some(ni),
                            features,
                        });
                    }
                    let ei = self.search.pick_adv_opp_edge(&st.w, enc, ni);
                    (ei, Some(ni), st.rd)
                } else {
                    let actor = st.w.e.state.current;
                    let feature_start = Instant::now();
                    let features = st.w.features(enc, actor).clone();
                    self.search.m.feature_seconds += feature_start.elapsed().as_secs_f64();
                    return Ok(DescendOutcome::NeedAdvOpp {
                        state: st,
                        existing_ni: None,
                        features,
                    });
                }
            } else if st.w.e.actions.len() == 1 {
                let a = st.w.e.actions[0];
                let aid = enc.id(&st.w.e, a);
                let k = (st.path.clone(), aid);
                let ei = if let Some(&i) = self.search.op_edges.get(&k) {
                    i
                } else {
                    let i = self.search.edge(a, 1.0);
                    self.search.op_edges.insert(k, i);
                    i
                };
                (ei, None, st.rd)
            } else if self.o.mcts_opponent == MctsOpponent::FrozenMlp {
                let (pred, _) = self.search.prediction(&st.w, enc, models, true)?;
                let ei = self.search.sample_opp_edge(&st.path, pred);
                (ei, None, st.rd)
            } else {
                let actor = st.w.e.state.current;
                let feature_start = Instant::now();
                let features = st.w.features(enc, actor).clone();
                self.search.m.feature_seconds += feature_start.elapsed().as_secs_f64();
                return Ok(DescendOutcome::NeedSelfOpp {
                    state: st,
                    features,
                });
            };
            st = self.advance_descent(enc, st, edge, node, next_rd);
        }
    }

    fn advance_descent(
        &mut self,
        enc: &Encoder,
        mut st: DescendState,
        edge: usize,
        node: Option<usize>,
        next_rd: usize,
    ) -> DescendState {
        let (child, wk, ck, denied) = self.search.step(&st.w, edge);
        st.path.push(Step {
            id: enc.id(&st.w.e, self.search.edges[edge].a),
            visible: visible(&child, enc, self.search.own),
            denied,
        });
        st.trace.push((node, edge, wk, ck));
        DescendState {
            w: child,
            path: st.path,
            trace: st.trace,
            rd: next_rd,
            depth: st.depth + 1,
        }
    }

    fn emit_root_priors(mut self, enc: &Encoder) -> StepOutcome {
        let feature_start = Instant::now();
        let features = self.root.features(enc, self.root.e.state.current).clone();
        self.search.m.feature_seconds += feature_start.elapsed().as_secs_f64();
        self.search.m.evaluations += 1;
        self.phase = StepperPhase::WaitRootPriors;
        StepOutcome::NeedCandidate {
            stepper: self,
            features: vec![features],
        }
    }

    pub fn step(
        mut self,
        enc: &Encoder,
        models: &mut Models,
        mut reply: Option<Vec<(Vec<f32>, f32)>>,
    ) -> Result<StepOutcome, String> {
        loop {
            let phase = std::mem::replace(&mut self.phase, StepperPhase::Init);
            match phase {
                StepperPhase::Init => {
                    if self.root.e.actions.len() == 1 {
                        let id = enc.id(&self.root.e, self.root.e.actions[0]);
                        let result = ResultData {
                            action_id: id,
                            sampled_action: id,
                            statistics: vec![],
                            metrics: self.search.m.clone(),
                            allocations: vec![],
                        };
                        return Ok(StepOutcome::Done {
                            search: self.search,
                            result,
                        });
                    }
                    if self.search.paths.contains_key(&vec![]) {
                        return Ok(self.emit_root_priors(enc));
                    }
                    let feature_start = Instant::now();
                    let features = self.root.features(enc, self.root.e.state.current).clone();
                    self.search.m.feature_seconds += feature_start.elapsed().as_secs_f64();
                    self.search.m.evaluations += 1;
                    self.phase = StepperPhase::WaitRootNode;
                    return Ok(StepOutcome::NeedCandidate {
                        stepper: self,
                        features: vec![features],
                    });
                }
                StepperPhase::WaitRootNode => {
                    let (scores, raw_val) = Self::take_one(reply.take())?;
                    let (pred, raw_v) = self.search.process_prediction(&self.root, enc, scores, raw_val)?;
                    let v = Search::utility(&self.root, self.search.own, raw_v, self.o.vp_margin_alpha);
                    let ni = self.search.expand_node(&self.root, enc, pred, v);
                    self.search.paths.insert(vec![], ni);
                    return Ok(self.emit_root_priors(enc));
                }
                StepperPhase::WaitRootPriors => {
                    let (scores, raw_val) = Self::take_one(reply.take())?;
                    let (pred, _) = self.search.process_prediction(&self.root, enc, scores, raw_val)?;
                    let ni = self.search.paths[&vec![]];
                    let noise = if !self.o.noise {
                        vec![0.; pred.len()]
                    } else if let Some(x) = self.noise_override.as_ref() {
                        assert_eq!(x.len(), pred.len());
                        x.clone()
                    } else {
                        pcg::Pcg::new(rng::named(self.seed, "root-noise")).dirichlet(pred.len())
                    };
                    self.search.m.root_noise = noise.clone();
                    for ((_, id, p), n) in pred.iter().zip(noise) {
                        let ei = *self.search.nodes[ni]
                            .edges
                            .iter()
                            .find(|&&i| enc.id(&self.root.e, self.search.edges[i].a) == *id)
                            .unwrap();
                        self.search.edges[ei].prior = if self.o.noise { 0.75 * p + 0.25 * n } else { *p };
                    }
                    let posterior_start = Instant::now();
                    let belief = belief::Belief::new(&self.root, self.search.own);
                    self.search.m.posterior_seconds += posterior_start.elapsed().as_secs_f64();
                    self.search.m.belief_evidence = belief.evidence;
                    self.phase = StepperPhase::ReadyToSimulate {
                        belief,
                        longest: 0.0,
                    };
                }
                StepperPhase::ReadyToSimulate {
                    mut belief,
                    longest,
                } => {
                    if self.o.simulations.is_some_and(|n| self.search.m.simulations >= n)
                        || self
                            .o
                            .seconds
                            .is_some_and(|s| self.start.elapsed().as_secs_f64() + 1.25 * longest >= s)
                    {
                        if self.lanes > 1 {
                            assert!(self.search.edges.iter().all(|e| e.vl_n == 0));
                            assert!(self.search.nodes.iter().all(|n| n.vl_n == 0));
                        }
                        let result = self.search.finalize(enc, self.start, &self.root);
                        return Ok(StepOutcome::Done {
                            search: self.search,
                            result,
                        });
                    }
                    let begin = Instant::now();
                    if self.lanes == 1 {
                        let w = self.search.sample_world(&mut belief, &self.root, self.audit_allocations);
                        let st = DescendState {
                            w,
                            path: vec![],
                            trace: vec![],
                            rd: 0,
                            depth: 0,
                        };
                        if let Some(features) = self.finish_single_descent(
                            enc,
                            models,
                            belief,
                            longest,
                            begin,
                            st,
                        )? {
                            return Ok(StepOutcome::NeedCandidate {
                                stepper: self,
                                features,
                            });
                        }
                    } else {
                        let remaining = self
                            .o
                            .simulations
                            .map(|n| n.saturating_sub(self.search.m.simulations))
                            .unwrap_or(self.lanes);
                        let wave_target = self.lanes.min(remaining);
                        let pending = Vec::with_capacity(wave_target);
                        let features_batch = Vec::with_capacity(wave_target);
                        if let Some(features) = self.run_wave(
                            enc,
                            models,
                            belief,
                            longest,
                            begin,
                            wave_target,
                            0,
                            pending,
                            features_batch,
                            None,
                        )? {
                            return Ok(StepOutcome::NeedCandidate {
                                stepper: self,
                                features,
                            });
                        }
                    }
                }
                StepperPhase::WaitSingleOwnLeaf {
                    belief,
                    mut longest,
                    begin,
                    w,
                    path,
                    trace,
                } => {
                    let (scores, raw_val) = Self::take_one(reply.take())?;
                    let (pred, raw_v) = self.search.process_prediction(&w, enc, scores, raw_val)?;
                    let v = Search::utility(&w, self.search.own, raw_v, self.o.vp_margin_alpha);
                    let ni = self.search.expand_node(&w, enc, pred, v);
                    self.search.paths.insert(path, ni);
                    self.search.nodes[ni].n += 1;
                    self.search.backup_trace(&trace, v, 0.0, false);
                    self.search.m.simulations += 1;
                    longest = longest.max(begin.elapsed().as_secs_f64());
                    self.phase = StepperPhase::ReadyToSimulate { belief, longest };
                }
                StepperPhase::WaitSingleAdvOpp {
                    belief,
                    longest,
                    begin,
                    state,
                    existing_ni,
                } => {
                    let (scores, raw_val) = Self::take_one(reply.take())?;
                    let (pred, raw_v) = self.search.process_prediction(&state.w, enc, scores, raw_val)?;
                    let v = Search::utility(&state.w, self.search.own, raw_v, self.o.vp_margin_alpha);
                    let ni = if let Some(ni) = existing_ni {
                        let existing_acts: Vec<Act> = self.search.nodes[ni]
                            .edges
                            .iter()
                            .map(|&i| self.search.edges[i].a)
                            .collect();
                        for (a, _, p) in pred {
                            if !existing_acts.contains(&a) {
                                let ei = self.search.edge(a, p);
                                self.search.nodes[ni].edges.push(ei);
                            }
                        }
                        ni
                    } else {
                        let ni = self.search.expand_node(&state.w, enc, pred, v);
                        self.search.op_paths.insert(state.path.clone(), ni);
                        ni
                    };
                    let ei = self.search.pick_adv_opp_edge(&state.w, enc, ni);
                    let rd = state.rd;
                    let next_st = self.advance_descent(enc, state, ei, Some(ni), rd);
                    if let Some(features) = self.finish_single_descent(
                        enc,
                        models,
                        belief,
                        longest,
                        begin,
                        next_st,
                    )? {
                        return Ok(StepOutcome::NeedCandidate {
                            stepper: self,
                            features,
                        });
                    }
                }
                StepperPhase::WaitSingleSelfOpp {
                    belief,
                    longest,
                    begin,
                    state,
                } => {
                    let (scores, raw_val) = Self::take_one(reply.take())?;
                    let (pred, _) = self.search.process_prediction(&state.w, enc, scores, raw_val)?;
                    let ei = self.search.sample_opp_edge(&state.path, pred);
                    let rd = state.rd;
                    let next_st = self.advance_descent(enc, state, ei, None, rd);
                    if let Some(features) = self.finish_single_descent(
                        enc,
                        models,
                        belief,
                        longest,
                        begin,
                        next_st,
                    )? {
                        return Ok(StepOutcome::NeedCandidate {
                            stepper: self,
                            features,
                        });
                    }
                }
                StepperPhase::WaitWaveOpp {
                    belief,
                    longest,
                    begin,
                    wave_target,
                    lane_idx,
                    pending,
                    features_batch,
                    state,
                } => {
                    let (scores, raw_val) = Self::take_one(reply.take())?;
                    let (pred, _) = self.search.process_prediction(&state.w, enc, scores, raw_val)?;
                    let ei = self.search.sample_opp_edge(&state.path, pred);
                    let rd = state.rd;
                    let next_st = self.advance_descent(enc, state, ei, None, rd);
                    if let Some(features) = self.run_wave(
                        enc,
                        models,
                        belief,
                        longest,
                        begin,
                        wave_target,
                        lane_idx,
                        pending,
                        features_batch,
                        Some(next_st),
                    )? {
                        return Ok(StepOutcome::NeedCandidate {
                            stepper: self,
                            features,
                        });
                    }
                }
                StepperPhase::WaitWaveBatch {
                    belief,
                    mut longest,
                    begin,
                    pending,
                } => {
                    let outputs = reply
                        .take()
                        .ok_or_else(|| "Missing candidate batch inference reply".to_string())?;
                    for ((lw, path, trace), (scores, raw_val)) in
                        pending.into_iter().zip(outputs.into_iter())
                    {
                        let (pred, raw_v) = self.search.process_prediction(&lw, enc, scores, raw_val)?;
                        let v = Search::utility(&lw, self.search.own, raw_v, self.o.vp_margin_alpha);
                        if let Some(&existing_ni) = self.search.paths.get(&path) {
                            self.search.nodes[existing_ni].n += 1;
                        } else {
                            let new_ni = self.search.expand_node(&lw, enc, pred, v);
                            self.search.nodes[new_ni].n = 1;
                            self.search.paths.insert(path, new_ni);
                        }
                        self.search.backup_trace(&trace, v, self.o.virtual_loss, true);
                        self.search.m.simulations += 1;
                    }
                    longest = longest.max(begin.elapsed().as_secs_f64());
                    self.phase = StepperPhase::ReadyToSimulate { belief, longest };
                }
            }
        }
    }

    fn finish_single_descent(
        &mut self,
        enc: &Encoder,
        models: &mut Models,
        belief: belief::Belief,
        mut longest: f64,
        begin: Instant,
        st: DescendState,
    ) -> Result<Option<Vec<Vec<f32>>>, String> {
        match self.step_descent(enc, models, st, false)? {
            DescendOutcome::Finished => {
                self.search.m.simulations += 1;
                longest = longest.max(begin.elapsed().as_secs_f64());
                self.phase = StepperPhase::ReadyToSimulate { belief, longest };
                Ok(None)
            }
            DescendOutcome::NeedOwnLeaf {
                w,
                path,
                trace,
                features,
            } => {
                self.search.m.evaluations += 1;
                self.phase = StepperPhase::WaitSingleOwnLeaf {
                    belief,
                    longest,
                    begin,
                    w,
                    path,
                    trace,
                };
                Ok(Some(vec![features]))
            }
            DescendOutcome::NeedAdvOpp {
                state,
                existing_ni,
                features,
            } => {
                self.search.m.evaluations += 1;
                self.phase = StepperPhase::WaitSingleAdvOpp {
                    belief,
                    longest,
                    begin,
                    state,
                    existing_ni,
                };
                Ok(Some(vec![features]))
            }
            DescendOutcome::NeedSelfOpp { state, features } => {
                self.search.m.evaluations += 1;
                self.phase = StepperPhase::WaitSingleSelfOpp {
                    belief,
                    longest,
                    begin,
                    state,
                };
                Ok(Some(vec![features]))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_wave(
        &mut self,
        enc: &Encoder,
        models: &mut Models,
        mut belief: belief::Belief,
        mut longest: f64,
        begin: Instant,
        wave_target: usize,
        mut lane_idx: usize,
        mut pending: Vec<(Arc<World>, TreePath, Vec<(Option<usize>, usize, Key, Key)>)>,
        mut features_batch: Vec<Vec<f32>>,
        mut resumed_state: Option<DescendState>,
    ) -> Result<Option<Vec<Vec<f32>>>, String> {
        while lane_idx < wave_target {
            let st = if let Some(s) = resumed_state.take() {
                s
            } else {
                let w = self
                    .search
                    .sample_world(&mut belief, &self.root, self.audit_allocations);
                DescendState {
                    w,
                    path: vec![],
                    trace: vec![],
                    rd: 0,
                    depth: 0,
                }
            };
            match self.step_descent(enc, models, st, true)? {
                DescendOutcome::Finished => {
                    self.search.m.simulations += 1;
                    lane_idx += 1;
                }
                DescendOutcome::NeedOwnLeaf {
                    w,
                    path,
                    trace,
                    features,
                } => {
                    pending.push((w, path, trace));
                    features_batch.push(features);
                    lane_idx += 1;
                }
                DescendOutcome::NeedSelfOpp { state, features } => {
                    self.search.m.evaluations += 1;
                    self.phase = StepperPhase::WaitWaveOpp {
                        belief,
                        longest,
                        begin,
                        wave_target,
                        lane_idx,
                        pending,
                        features_batch,
                        state,
                    };
                    return Ok(Some(vec![features]));
                }
                DescendOutcome::NeedAdvOpp { .. } => {
                    return Err("AdversarialPuct requires lanes == 1".to_string());
                }
            }
        }
        if !pending.is_empty() {
            self.search.m.evaluations += pending.len();
            self.phase = StepperPhase::WaitWaveBatch {
                belief,
                longest,
                begin,
                pending,
            };
            return Ok(Some(features_batch));
        }
        longest = longest.max(begin.elapsed().as_secs_f64());
        self.phase = StepperPhase::ReadyToSimulate { belief, longest };
        Ok(None)
    }
}
