use crate::rules::{Act, Engine};
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::{collections::HashMap, path::Path, sync::Arc, time::Instant};
const R: [&str; 5] = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];
const D: [&str; 5] = [
    "KNIGHT",
    "MONOPOLY",
    "YEAR_OF_PLENTY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
];
#[derive(Clone, Deserialize)]
struct Id {
    action: Act,
    id: usize,
    rate: i32,
}
#[derive(Clone, Deserialize)]
struct Job {
    initial: Engine,
    names: Vec<String>,
    static_features: Vec<f64>,
    coordinates: Vec<[i32; 3]>,
    tile_ids: Vec<usize>,
    colors: Vec<String>,
    action_ids: Vec<Id>,
}
#[derive(Clone, Default)]
struct History {
    new: [[i32; 5]; 2],
    buys: [i32; 2],
}
impl History {
    fn update(&mut self, p: usize, a: Act, ch: i32) {
        if a.k == 0 {
            self.new[p] = [0; 5];
        }
        if a.k == 5 {
            self.new[p][ch as usize] += 1;
            self.buys[p] += 1;
        }
    }
}
struct Encoder {
    job: Arc<Job>,
    indices: HashMap<String, usize>,
}
impl Encoder {
    fn new(j: Arc<Job>) -> Self {
        Self {
            indices: j
                .names
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect(),
            job: j,
        }
    }
    fn features(&self, e: &Engine, h: &History, own: usize) -> Vec<f32> {
        let mut x: Vec<f32> = self.job.static_features.iter().map(|&x| x as f32).collect();
        let s = &e.state;
        let other = 1 - own;
        let mut put = |n: String, v: f32| {
            let i = self
                .indices
                .get(&n)
                .unwrap_or_else(|| panic!("Unknown feature {n}"));
            x[*i] = v;
        };
        for (rel, p) in [own, other].iter().copied().enumerate() {
            let q = &s.players[p];
            let pre = format!("P{rel}_");
            for (k, v) in [
                ("PUBLIC_VPS", q.vp as f32),
                ("HAS_ARMY", q.has_army as u8 as f32),
                ("HAS_ROAD", q.has_road as u8 as f32),
                ("HAS_ROLLED", q.rolled as u8 as f32),
                ("ROADS_LEFT", q.pieces[0] as f32),
                ("SETTLEMENTS_LEFT", q.pieces[1] as f32),
                ("CITIES_LEFT", q.pieces[2] as f32),
                ("LONGEST_ROAD_LENGTH", q.road_length as f32),
                ("NUM_RESOURCES_IN_HAND", q.hand.iter().sum::<i32>() as f32),
                ("NUM_DEVS_IN_HAND", q.dev.iter().sum::<i32>() as f32),
                ("DEV_PURCHASES", h.buys[p] as f32),
            ] {
                put(format!("{pre}{k}"), v);
            }
            for (i, d) in D.iter().enumerate() {
                if i != 4 {
                    put(format!("{pre}{d}_PLAYED"), q.played[i] as f32);
                }
            }
        }
        let q = &s.players[own];
        put("P0_ACTUAL_VPS".into(), q.actual as f32);
        put(
            "P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN".into(),
            q.used_dev as u8 as f32,
        );
        for (i, r) in R.iter().enumerate() {
            put(format!("P0_{r}_IN_HAND"), q.hand[i] as f32);
            put(
                format!("P1_INFERRED_{r}_IN_HAND"),
                (19 - s.bank[i] - q.hand[i]) as f32,
            );
            put(format!("BANK_{r}"), s.bank[i] as f32);
        }
        for (i, d) in D.iter().enumerate() {
            put(format!("P0_{d}_IN_HAND"), q.dev[i] as f32);
            put(format!("P0_NEW_{d}"), h.new[own][i] as f32);
            put(format!("P0_OLDER_{d}"), (q.dev[i] - h.new[own][i]) as f32);
            if i != 4 {
                put(format!("P0_{d}_OWNED_AT_START"), q.owned[i] as u8 as f32);
                let kind = [8, 10, 9, 11][i];
                put(
                    format!("P0_{d}_PLAYABLE"),
                    (s.current == own && e.actions.iter().any(|a| a.k == kind)) as u8 as f32,
                );
            }
        }
        let new = s.players[other].dev.iter().sum::<i32>();
        put(
            "P1_NEW_DEVS".into(),
            h.new[other].iter().sum::<i32>() as f32,
        );
        put(
            "P1_OLDER_DEVS".into(),
            (new - h.new[other].iter().sum::<i32>()) as f32,
        );
        for n in 0..54 {
            for rel in 0..2 {
                for k in ["SETTLEMENT", "CITY"] {
                    let p = if rel == 0 { own } else { other };
                    let kind = if k == "CITY" { 2 } else { 1 };
                    put(
                        format!("NODE{n}_P{rel}_{k}"),
                        (s.buildings[n] == [p as i32, kind]) as u8 as f32,
                    );
                }
            }
        }
        for (i, [a, b]) in e.map.edges.iter().enumerate() {
            for rel in 0..2 {
                let p = if rel == 0 { own } else { other };
                put(
                    format!("EDGE({a}, {b})_P{rel}_ROAD"),
                    (s.roads[i] == p as i32) as u8 as f32,
                );
            }
        }
        for (t, id) in self.job.tile_ids.iter().enumerate() {
            put(format!("TILE{id}_HAS_ROBBER"), (t == s.robber) as u8 as f32);
        }
        for (k, v) in [
            ("BANK_DEV_CARDS", s.deck.len() as f32),
            (
                "IS_DISCARDING",
                (s.current == own && e.actions.iter().any(|a| a.k == 6)) as u8 as f32,
            ),
            (
                "IS_MOVING_ROBBER",
                (s.current == own && e.actions.iter().any(|a| a.k == 7)) as u8 as f32,
            ),
            (
                "ACTOR_IS_BLUE",
                (self.job.colors[s.current] == "BLUE") as u8 as f32,
            ),
            (
                "OBSERVER_IS_BLUE",
                (self.job.colors[own] == "BLUE") as u8 as f32,
            ),
            ("DECISION_ACTOR_IS_SELF", (s.current == own) as u8 as f32),
            ("TURN_OWNER_IS_SELF", (s.turn == own) as u8 as f32),
            ("IS_SETUP", s.initial as u8 as f32),
            ("IS_FREE_ROAD", s.road_building as u8 as f32),
            ("FREE_ROADS_AVAILABLE", s.free_roads as f32),
            ("IS_MOVING_KNIGHT", s.moving as u8 as f32),
            ("DISCARD_REMAINING_SELF", s.discards[own] as f32),
            ("DISCARD_REMAINING_OTHER", s.discards[other] as f32),
        ] {
            put(k.into(), v);
        }
        for (i, n) in [
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "PLAY_TURN",
            "MOVE_ROBBER",
            "DISCARD",
        ]
        .iter()
        .enumerate()
        {
            put(format!("PHASE_{n}"), (s.prompt == i) as u8 as f32);
        }
        x
    }
    fn id(&self, e: &Engine, a: Act) -> usize {
        let mut rate = 0;
        if a.k == 12 {
            rate = 4;
            for (n, b) in e.state.buildings.iter().enumerate() {
                if b[0] == e.state.current as i32 {
                    for &r in &e.map.ports[n] {
                        if r < 0 {
                            rate = rate.min(3);
                        } else if r == a.a {
                            rate = 2;
                        }
                    }
                }
            }
        }
        self.job
            .action_ids
            .iter()
            .find(|v| v.action == a && v.rate == rate)
            .expect("Action ID missing")
            .id
    }
}
fn session(path: &Path) -> ort::Result<Session> {
    Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_file(path)
}
fn predict(s: &mut Session, x: Vec<f32>) -> ort::Result<(Vec<f32>, f32)> {
    let n = x.len();
    let input = Tensor::from_array(([1, n], x))?;
    let out = s.run(ort::inputs![input])?;
    Ok((
        out[0].try_extract_tensor::<f32>()?.1.to_vec(),
        out[1].try_extract_tensor::<f32>()?.1[0],
    ))
}
fn masked(scores: &[f32], ids: &[usize]) -> Vec<f64> {
    let max = ids
        .iter()
        .map(|&i| scores[i] as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let values: Vec<_> = ids
        .iter()
        .map(|&i| ((scores[i] as f64) - max).exp())
        .collect();
    let sum: f64 = values.iter().sum();
    values.into_iter().map(|x| x / sum).collect()
}
