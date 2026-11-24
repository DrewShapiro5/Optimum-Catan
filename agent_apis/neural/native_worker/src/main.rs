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
struct World {
    e: Engine,
    h: History,
    logs: Arc<Vec<Log>>,
    features: [OnceLock<Vec<f32>>; 2],
    keys: [OnceLock<[u8; 32]>; 2],
    full: OnceLock<[u8; 32]>,
}
fn hash<T: Serialize>(v: &T) -> [u8; 32] {
    Sha256::digest(serde_json::to_vec(v).unwrap()).into()
}
impl World {
    fn new(mut e: Engine, h: History, logs: Vec<Log>) -> Self {
        e.refresh();
        Self {
            e,
            h,
            logs: Arc::new(logs),
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
        let mut logs = (*self.logs).clone();
        logs.push(Log {
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
    fn knowledge(&self, p: Option<usize>) -> Vec<(usize, u8, usize, i32)> {
        let mut turns = [0; 2];
        let mut out = vec![];
        for l in self.logs.iter() {
            if l.a.k == 0 {
                turns[l.p] += 1;
            }
            if l.a.k == 5 {
                out.push((
                    l.p,
                    0,
                    turns[l.p],
                    if p.is_none() || p == Some(l.p) {
                        l.ch
                    } else {
                        -1
                    },
                ));
            }
            if let Some(c) = card(l.a.k) {
                out.push((l.p, 1, turns[l.p], c as i32));
            }
        }
        out
    }
    fn key(&self, enc: &Encoder, p: usize) -> [u8; 32] {
        *self.keys[p].get_or_init(|| {
            let mut features: Vec<f64> = self.features(enc, p).iter().map(|&x| x as f64).collect();
            for (i, name) in enc.job.names.iter().enumerate() {
                if name.ends_with("_PROBA") {
                    features[i] = enc.job.static_features[i];
                }
            }
            let knowledge: Vec<_> = self
                .knowledge(Some(p))
                .into_iter()
                .map(|(seat, kind, turn, card)| {
                    serde_json::json!([
                        enc.job.colors[seat],
                        if kind == 0 { "buy" } else { "play" },
                        turn,
                        if card < 0 {
                            serde_json::Value::Null
                        } else {
                            serde_json::json!(D[card as usize])
                        }
                    ])
                })
                .collect();
            let mut legal = if self.e.state.current == p {
                self.e
                    .actions
                    .iter()
                    .map(|&a| legal_json(self, enc, a))
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            legal.sort_by_key(|a| serde_json::to_string(a).unwrap());
            hash(&serde_json::json!({"features":features,"knowledge":knowledge,"legal":legal}))
        })
    }
    fn canonical(&self) -> [u8; 32] {
        *self.full.get_or_init(|| {
            let s = &self.e.state;
            let players: Vec<_> = s
                .players
                .iter()
                .map(|p| {
                    serde_json::json!([
                        p.hand,
                        p.dev,
                        p.played,
                        p.owned,
                        p.rolled,
                        p.used_dev,
                        p.vp,
                        p.actual,
                        p.pieces,
                        p.has_road,
                        p.has_army,
                        p.road_length
                    ])
                })
                .collect();
            let mut deck = [0; 5];
            for &d in &s.deck {
                deck[d] += 1;
            }
            hash(&(
                players,
                deck,
                (
                    &s.bank,
                    &s.buildings,
                    &s.roads,
                    s.robber,
                    s.current,
                    s.turn,
                    s.turns,
                    s.prompt,
                ),
                (
                    s.initial,
                    s.discarding,
                    s.moving,
                    s.road_building,
                    s.free_roads,
                    &s.discards,
                ),
                self.knowledge(None),
            ))
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
        let mut logs = (*self.logs).clone();
        for &(step, card) in &a.draws {
            assert_eq!(logs[step].p, other);
            assert_eq!(logs[step].a.k, 5);
            logs[step].ch = card as i32;
        }
        let w = Self::new(e, h, logs);
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
fn legal_json(w: &World, enc: &Encoder, a: Act) -> serde_json::Value {
    let names = [
        "END_TURN",
        "ROLL",
        "BUILD_SETTLEMENT",
        "BUILD_ROAD",
        "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD",
        "DISCARD_RESOURCE",
        "MOVE_ROBBER",
        "PLAY_KNIGHT_CARD",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_MONOPOLY",
        "PLAY_ROAD_BUILDING",
        "MARITIME_TRADE",
    ];
    let v = match a.k {
        2 | 4 => serde_json::json!(a.a),
        3 => serde_json::json!(w.e.map.edges[a.a as usize]),
        6 | 10 => serde_json::json!(R[a.a as usize]),
        7 => serde_json::json!([
            enc.job.coordinates[a.a as usize],
            if a.b < 0 {
                serde_json::Value::Null
            } else {
                serde_json::json!(enc.job.colors[a.b as usize])
            }
        ]),
        9 => {
            if a.b < 0 {
                serde_json::json!([R[a.a as usize]])
            } else {
                serde_json::json!([R[a.a as usize], R[a.b as usize]])
            }
        }
        12 => {
            let id = enc.id(&w.e, a);
            let rate = enc.job.action_ids.iter().find(|x| x.id == id).unwrap().rate as usize;
            let mut v = vec![serde_json::json!(R[a.a as usize]); rate];
            v.resize(4, serde_json::Value::Null);
            v.push(serde_json::json!(R[a.b as usize]));
            serde_json::json!(v)
        }
        _ => serde_json::Value::Null,
    };
    serde_json::json!([enc.job.colors[w.e.state.current], names[a.k as usize], v])
}
fn visible(w: &World, enc: &Encoder, own: usize) -> Key {
    hash(
        &serde_json::json!({"observation":hex(w.key(enc,own)),"announced_terminal_winner":if w.e.winner()<0{serde_json::Value::Null}else{serde_json::json!(enc.job.colors[w.e.winner() as usize])}}),
    )
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
    World::new(e, h, t.logs.clone())
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
