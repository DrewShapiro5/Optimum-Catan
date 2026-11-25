use super::*;
const ORDER: [usize; 5] = [0, 2, 1, 3, 4];
const SUPPLY: [i32; 5] = [14, 2, 2, 2, 5];
#[derive(Clone)]
struct BEvent {
    kind: u8,
    index: usize,
    step: usize,
    bought: [i32; 5],
    played: [i32; 5],
    limit: i32,
}
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct Allocation {
    pub old: [i32; 5],
    pub new: [i32; 5],
    pub deck: [i32; 5],
    pub flags: [bool; 4],
    pub draws: Vec<(usize, usize)>,
}
pub struct Belief {
    events: Vec<BEvent>,
    bought: [i32; 5],
    played: [i32; 5],
    since: [i32; 5],
    total: i32,
    remaining: i32,
    maxvp: i32,
    cache: HashMap<(usize, [i32; 5], [i32; 5]), f64>,
    pub evidence: f64,
}
impl Belief {
    pub fn new(w: &World, own: usize) -> Self {
        let other = 1 - own;
        let mut b = Self {
            events: vec![],
            bought: [0; 5],
            played: [0; 5],
            since: [0; 5],
            total: w.e.state.players[other].dev.iter().sum(),
            remaining: w.e.state.deck.len() as i32,
            maxvp: 14 - w.e.state.players[other].vp,
            cache: HashMap::new(),
            evidence: 0.,
        };
        let mut previous = -999;
        let (mut own_old, mut own_new) = ([0; 5], [0; 5]);
        w.logs.for_each(|step, log| {
            let before_bought = b.bought;
            let before_played = b.played;
            let (mut kind, mut index) = (0, 0);
            if log.a.k == 5 {
                if log.p == own {
                    index = log.ch as usize;
                    kind = 1;
                    b.bought[index] += 1;
                    own_new[index] += 1;
                } else {
                    kind = 2;
                }
            } else if let Some(d) = card(log.a.k) {
                index = d;
                if log.p == other {
                    kind = 3;
                    b.played[index] += 1;
                    b.since[index] += 1;
                } else {
                    assert!(own_old[index] > 0, "Own old-card mismatch");
                    own_old[index] -= 1;
                }
            } else if log.a.k == 0 {
                if log.p == other {
                    kind = 4;
                    b.since = [0; 5];
                } else {
                    for i in 0..5 {
                        own_old[i] += own_new[i];
                    }
                    own_new = [0; 5];
                }
            }
            assert!(
                log.vp[own] + own_old[4] + own_new[4] < 15,
                "History already terminal"
            );
            let limit = 14 - log.vp[other];
            if kind != 0 || limit != previous {
                b.events.push(BEvent {
                    kind,
                    index,
                    step,
                    bought: before_bought,
                    played: before_played,
                    limit,
                });
                previous = limit;
            }
        });
        assert_eq!(
            std::array::from_fn::<_, 5, _>(|i| own_old[i] + own_new[i]),
            w.e.state.players[own].dev
        );
        b.evidence = b.mass(0, [0; 5], [0; 5]);
        assert!(b.evidence > 0., "Impossible development posterior");
        b
    }
    fn deck(b: [i32; 5], p: [i32; 5], o: [i32; 5], n: [i32; 5]) -> [i32; 5] {
        std::array::from_fn(|i| SUPPLY[i] - b[i] - p[i] - o[i] - n[i])
    }
    fn transitions(
        &self,
        pos: usize,
        o: [i32; 5],
        n: [i32; 5],
    ) -> Vec<(Option<usize>, f64, [i32; 5], [i32; 5])> {
        let e = &self.events[pos];
        let deck = Self::deck(e.bought, e.played, o, n);
        if deck.iter().any(|&x| x < 0) {
            return vec![];
        }
        let total = deck.iter().sum::<i32>() as f64;
        let mut v = vec![];
        match e.kind {
            2 => {
                for i in ORDER {
                    if deck[i] > 0 {
                        let mut nn = n;
                        nn[i] += 1;
                        v.push((Some(i), deck[i] as f64 / total, o, nn));
                    }
                }
            }
            1 => {
                if deck[e.index] > 0 {
                    v.push((Some(e.index), deck[e.index] as f64 / total, o, n));
                }
            }
            3 => {
                if o[e.index] > 0 {
                    let mut oo = o;
                    oo[e.index] -= 1;
                    v.push((None, 1., oo, n));
                }
            }
            4 => v.push((None, 1., std::array::from_fn(|i| o[i] + n[i]), [0; 5])),
            _ => v.push((None, 1., o, n)),
        }
        v.into_iter()
            .filter(|x| x.2[4] + x.3[4] <= e.limit)
            .collect()
    }
    fn mass(&mut self, pos: usize, o: [i32; 5], n: [i32; 5]) -> f64 {
        let key = (pos, o, n);
        if let Some(&v) = self.cache.get(&key) {
            return v;
        }
        let value = if pos == self.events.len() {
            let d = Self::deck(self.bought, self.played, o, n);
            (o.iter().sum::<i32>() + n.iter().sum::<i32>() == self.total
                && d.iter().sum::<i32>() == self.remaining
                && d.iter().all(|&x| x >= 0)
                && o[4] + n[4] <= self.maxvp.min(5)) as u8 as f64
        } else {
            let mut value = 0.;
            for (_, p, oo, nn) in self.transitions(pos, o, n) {
                value += p * self.mass(pos + 1, oo, nn);
            }
            value
        };
        self.cache.insert(key, value);
        value
    }
    pub fn sample(&mut self, rng: &mut rng::Rng) -> Allocation {
        let (mut o, mut n) = ([0; 5], [0; 5]);
        let mut draws = vec![];
        for pos in 0..self.events.len() {
            let mut weighted = vec![];
            let mut total = 0.;
            for t in self.transitions(pos, o, n) {
                let weight = t.1 * self.mass(pos + 1, t.2, t.3);
                total += weight;
                weighted.push((t, weight));
            }
            assert!(total > 0.);
            let mut threshold = rng.uniform() * total;
            let mut selected = weighted.last().unwrap().0;
            for (t, w) in weighted {
                threshold -= w;
                if threshold < 0. {
                    selected = t;
                    break;
                }
            }
            o = selected.2;
            n = selected.3;
            if self.events[pos].kind == 2 {
                draws.push((self.events[pos].step, selected.0.unwrap()));
            }
        }
        Allocation {
            old: o,
            new: n,
            deck: Self::deck(self.bought, self.played, o, n),
            flags: std::array::from_fn(|i| o[i] > 0 || self.since[i] > 0),
            draws,
        }
    }
}
