include!("base.rs");
mod batch_gpu;
mod belief;
mod pcg;
mod rng;
mod rules;
mod search;
mod worker;
use std::sync::OnceLock;
#[derive(Clone, Serialize, Deserialize, Debug)]
struct Log {
    p: usize,
    a: Act,
    ch: i32,
    vp: [i32; 2],
}
fn card(k: i32) -> Option<usize> {
    match k {
        8 => Some(0),
        9 => Some(2),
        10 => Some(1),
        11 => Some(3),
        _ => None,
    }
}
#[derive(Clone)]
struct LogTail {
    prev: Option<Arc<LogTail>>,
    log: Log,
}

#[derive(Clone)]
struct Logs {
    base: Arc<Vec<Log>>,
    tail: Option<Arc<LogTail>>,
    len: usize,
}

impl Logs {
    fn from_vec(v: Vec<Log>) -> Self {
        let len = v.len();
        Self {
            base: Arc::new(v),
            tail: None,
            len,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&self, log: Log) -> Self {
        Self {
            base: Arc::clone(&self.base),
            tail: Some(Arc::new(LogTail {
                prev: self.tail.clone(),
                log,
            })),
            len: self.len + 1,
        }
    }

    fn get(&self, idx: usize) -> &Log {
        let base_len = self.base.len();
        if idx < base_len {
            return &self.base[idx];
        }
        let mut steps_back = self.len - 1 - idx;
        let mut cur = self.tail.as_deref();
        while let Some(node) = cur {
            if steps_back == 0 {
                return &node.log;
            }
            steps_back -= 1;
            cur = node.prev.as_deref();
        }
        panic!("Log index {idx} out of bounds (len={})", self.len);
    }

    fn for_each<F: FnMut(usize, &Log)>(&self, mut f: F) {
        let base_len = self.base.len();
        for (i, l) in self.base.iter().enumerate() {
            f(i, l);
        }
        let tail_len = self.len - base_len;
        if tail_len > 0 {
            let mut refs = Vec::with_capacity(tail_len);
            let mut cur = self.tail.as_deref();
            while let Some(node) = cur {
                refs.push(&node.log);
                cur = node.prev.as_deref();
            }
            for (j, l) in refs.into_iter().rev().enumerate() {
                f(base_len + j, l);
            }
        }
    }

    fn any<F: FnMut(&Log) -> bool>(&self, mut f: F) -> bool {
        if self.base.iter().any(&mut f) {
            return true;
        }
        let mut cur = self.tail.as_deref();
        while let Some(node) = cur {
            if f(&node.log) {
                return true;
            }
            cur = node.prev.as_deref();
        }
        false
    }

    fn to_vec(&self) -> Vec<Log> {
        let mut out = Vec::with_capacity(self.len);
        out.extend(self.base.iter().cloned());
        let tail_len = self.len - self.base.len();
        if tail_len > 0 {
            let mut refs = Vec::with_capacity(tail_len);
            let mut cur = self.tail.as_deref();
            while let Some(node) = cur {
                refs.push(&node.log);
                cur = node.prev.as_deref();
            }
            out.extend(refs.into_iter().rev().cloned());
        }
        out
    }

    fn compact(&mut self) {
        if self.tail.is_some() {
            *self = Self::from_vec(self.to_vec());
        }
    }
}

#[derive(Clone)]
struct World {
    e: Engine,
    h: History,
    logs: Logs,
    features: [OnceLock<Vec<f32>>; 2],
    keys: [OnceLock<[u8; 32]>; 2],
    full: OnceLock<[u8; 32]>,
}
impl World {
    fn new(mut e: Engine, h: History, logs: Logs) -> Self {
        e.refresh();
        Self {
            e,
            h,
            logs,
            features: std::array::from_fn(|_| OnceLock::new()),
            keys: std::array::from_fn(|_| OnceLock::new()),
            full: OnceLock::new(),
        }
    }
    fn step(&self, a: Act, ch: i32) -> Self {
        assert!(self.e.actions.contains(&a));
        let p = self.e.state.current;
        let mut e = self.e.clone();
        e.apply(a, ch);
        e.refresh();
        let mut h = self.h.clone();
        h.update(p, a, ch);
        let logs = self.logs.push(Log {
            p,
            a,
            ch,
            vp: [e.state.players[0].vp, e.state.players[1].vp],
        });
        Self::new(e, h, logs)
    }
    fn features<'a>(&'a self, enc: &Encoder, p: usize) -> &'a Vec<f32> {
        self.features[p].get_or_init(|| enc.features(&self.e, &self.h, p))
    }
    fn hash_knowledge(&self, p: Option<usize>, hasher: &mut Sha256) {
        let mut turns = [0usize; 2];
        let mut count = 0u64;
        self.logs.for_each(|_, l| {
            if l.a.k == 0 {
                turns[l.p] += 1;
            }
            if l.a.k == 5 {
                count += 1;
                let ch = if p.is_none() || p == Some(l.p) {
                    l.ch
                } else {
                    -1
                };
                hasher.update([l.p as u8, 0u8]);
                hasher.update((turns[l.p] as u64).to_le_bytes());
                hasher.update(ch.to_le_bytes());
            }
            if let Some(c) = card(l.a.k) {
                count += 1;
                hasher.update([l.p as u8, 1u8]);
                hasher.update((turns[l.p] as u64).to_le_bytes());
                hasher.update((c as i32).to_le_bytes());
            }
        });
        hasher.update(count.to_le_bytes());
    }
    fn key(&self, enc: &Encoder, p: usize) -> [u8; 32] {
        *self.keys[p].get_or_init(|| {
            let mut hasher = Sha256::new();
            let features = self.features(enc, p);
            hasher.update((features.len() as u64).to_le_bytes());
            for (i, &v) in features.iter().enumerate() {
                let raw = if enc.is_proba[i] {
                    enc.base_features[i]
                } else {
                    v
                };
                let canon = if raw == 0.0 { 0.0_f32 } else { raw };
                hasher.update(canon.to_le_bytes());
            }
            self.hash_knowledge(Some(p), &mut hasher);
            if self.e.state.current == p {
                let mut legal: Vec<(usize, i32, i32, i32)> = self
                    .e
                    .actions
                    .iter()
                    .map(|&a| (enc.id(&self.e, a), a.k, a.a, a.b))
                    .collect();
                legal.sort_unstable();
                hasher.update((legal.len() as u64).to_le_bytes());
                for (id, k, a, b) in legal {
                    hasher.update((id as u64).to_le_bytes());
                    hasher.update(k.to_le_bytes());
                    hasher.update(a.to_le_bytes());
                    hasher.update(b.to_le_bytes());
                }
            } else {
                hasher.update(0u64.to_le_bytes());
            }
            hasher.finalize().into()
        })
    }
    fn canonical(&self) -> [u8; 32] {
        *self.full.get_or_init(|| {
            let mut hasher = Sha256::new();
            let s = &self.e.state;
            hasher.update((s.players.len() as u64).to_le_bytes());
            for p in &s.players {
                for &x in &p.hand {
                    hasher.update(x.to_le_bytes());
                }
                for &x in &p.dev {
                    hasher.update(x.to_le_bytes());
                }
                for &x in &p.played {
                    hasher.update(x.to_le_bytes());
                }
                for &x in &p.owned {
                    hasher.update([x as u8]);
                }
                hasher.update([p.rolled as u8, p.used_dev as u8]);
                hasher.update(p.vp.to_le_bytes());
                hasher.update(p.actual.to_le_bytes());
                for &x in &p.pieces {
                    hasher.update(x.to_le_bytes());
                }
                hasher.update([p.has_road as u8, p.has_army as u8]);
                hasher.update((p.road_length as u64).to_le_bytes());
            }
            let mut deck = [0i32; 5];
            for &d in &s.deck {
                deck[d] += 1;
            }
            for x in deck {
                hasher.update(x.to_le_bytes());
            }
            for &x in &s.bank {
                hasher.update(x.to_le_bytes());
            }
            hasher.update((s.buildings.len() as u64).to_le_bytes());
            for &b in &s.buildings {
                hasher.update(b[0].to_le_bytes());
                hasher.update(b[1].to_le_bytes());
            }
            hasher.update((s.roads.len() as u64).to_le_bytes());
            for &r in &s.roads {
                hasher.update(r.to_le_bytes());
            }
            hasher.update((s.robber as u64).to_le_bytes());
            hasher.update((s.current as u64).to_le_bytes());
            hasher.update((s.turn as u64).to_le_bytes());
            hasher.update((s.turns as u64).to_le_bytes());
            hasher.update((s.prompt as u64).to_le_bytes());
            hasher.update([
                s.initial as u8,
                s.discarding as u8,
                s.moving as u8,
                s.road_building as u8,
            ]);
            hasher.update((s.free_roads as u64).to_le_bytes());
            hasher.update((s.discards.len() as u64).to_le_bytes());
            for &d in &s.discards {
                hasher.update(d.to_le_bytes());
            }
            self.hash_knowledge(None, &mut hasher);
            hasher.finalize().into()
        })
    }
    fn sampled(&self, a: &belief::Allocation, own: usize) -> Self {
        let other = 1 - own;
        let mut e = self.e.clone();
        let hand = std::array::from_fn(|i| 19 - e.state.bank[i] - e.state.players[own].hand[i]);
        let q = &mut e.state.players[other];
        q.dev = std::array::from_fn(|i| a.old[i] + a.new[i]);
        q.owned = a.flags;
        q.actual = q.vp + q.dev[4];
        q.hand = hand;
        e.state.deck.clear();
        for i in [0, 2, 1, 3, 4] {
            for _ in 0..a.deck[i] {
                e.state.deck.push(i);
            }
        }
        let mut h = self.h.clone();
        h.new[other] = a.new;
        let mut logs = self.logs.to_vec();
        for &(step, card) in &a.draws {
            assert_eq!(logs[step].p, other);
            assert_eq!(logs[step].a.k, 5);
            logs[step].ch = card as i32;
        }
        let w = Self::new(e, h, Logs::from_vec(logs));
        assert_eq!(
            w.e.actions, self.e.actions,
            "Hidden sampled allocation changed own legal actions"
        );
        assert!(w.e.winner() < 0);
        w
    }
}

fn hex(k: Key) -> String {
    k.iter().map(|x| format!("{x:02x}")).collect()
}
type Key = [u8; 32];
fn visible(w: &World, enc: &Encoder, own: usize) -> Key {
    let mut hasher = Sha256::new();
    hasher.update(w.key(enc, own));
    hasher.update(w.e.winner().to_le_bytes());
    hasher.finalize().into()
}

#[derive(Clone, Deserialize)]
struct Task {
    job: Job,
    logs: Vec<Log>,
    id: String,
    start: Engine,
    start_rng: Vec<u32>,
    seat_seeds: [u64; 2],
    rng_state: Vec<u32>,
    #[serde(default)]
    prefix: usize,
}
fn task_world(t: &Task) -> World {
    let mut e = t.start.clone();
    let mut h = History::default();
    e.refresh();
    for l in &t.logs {
        assert_eq!(e.state.current, l.p);
        assert!(e.actions.contains(&l.a));
        e.apply(l.a, l.ch);
        e.refresh();
        h.update(l.p, l.a, l.ch);
        assert_eq!([e.state.players[0].vp, e.state.players[1].vp], l.vp);
    }
    let mut a = e.state.clone();
    let mut b = t.job.initial.state.clone();
    for s in [&mut a, &mut b] {
        for cs in &mut s.components {
            for c in cs {
                c.sort();
            }
        }
    }
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap(),
        "Replayed prefix state mismatch"
    );
    World::new(e, h, Logs::from_vec(t.logs.clone()))
}
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum MctsOpponent {
    #[default]
    FrozenMlp,
    SelfPolicy,
    AdversarialPuct,
}
#[derive(Clone)]
struct Options {
    simulations: Option<usize>,
    seconds: Option<f64>,
    noise: bool,
    lanes: usize,
    virtual_loss: f64,
    mcts_opponent: MctsOpponent,
    vp_margin_alpha: f64,
    relative_fpu: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = worker::parse_args(std::env::args().skip(1))?;
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || worker::serve(mode))?
        .join()
        .map_err(|_| "Native worker thread failed")?
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_log(p: usize, k: i32, a: i32, ch: i32) -> Log {
        Log {
            p,
            a: Act { k, a, b: -1 },
            ch,
            vp: [2, 2],
        }
    }

    #[test]
    fn logs_persistent_chain_preserves_order_and_branching() {
        let base = Logs::from_vec(vec![mk_log(0, 0, -1, -1), mk_log(1, 5, -1, -1)]);
        let branch_a = base.push(mk_log(0, 1, -1, 7)).push(mk_log(0, 8, -1, -1));
        let branch_b = base.push(mk_log(1, 1, -1, 6));

        assert_eq!(base.len(), 2);
        assert_eq!(branch_a.len(), 4);
        assert_eq!(branch_b.len(), 3);
        assert_eq!(branch_a.get(2).ch, 7);
        assert_eq!(branch_a.get(3).a.k, 8);
        assert_eq!(branch_b.get(2).ch, 6);

        let mut seen = Vec::new();
        branch_a.for_each(|i, l| seen.push((i, l.p, l.a.k, l.ch)));
        assert_eq!(seen, vec![(0, 0, 0, -1), (1, 1, 5, -1), (2, 0, 1, 7), (3, 0, 8, -1)]);

        assert!(branch_a.any(|l| l.p == 1 && l.a.k == 5));
        assert!(!branch_b.any(|l| l.a.k == 8));

        let mut compacted = branch_a.clone();
        compacted.compact();
        assert_eq!(compacted.to_vec().len(), 4);
        assert_eq!(compacted.get(3).a.k, 8);
        assert!(search::same_logs(&branch_a, &compacted, 0));
    }
}
