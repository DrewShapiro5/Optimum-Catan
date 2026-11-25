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
    #[allow(dead_code)]
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
struct FeatureIndices {
    p_scalars: [[usize; 11]; 2],
    p_played: [[usize; 4]; 2],
    p0_actual_vps: usize,
    p0_used_dev: usize,
    p0_hand: [usize; 5],
    p1_inferred_hand: [usize; 5],
    bank_res: [usize; 5],
    p0_dev_in_hand: [usize; 5],
    p0_new_dev: [usize; 5],
    p0_older_dev: [usize; 5],
    p0_dev_owned_start: [usize; 4],
    p0_dev_playable: [usize; 4],
    p1_new_devs: usize,
    p1_older_devs: usize,
    nodes: [[[usize; 2]; 2]; 54],
    edges: Vec<[usize; 2]>,
    tiles_robber: Vec<usize>,
    bank_dev_cards: usize,
    is_discarding: usize,
    is_moving_robber: usize,
    actor_is_blue: usize,
    observer_is_blue: usize,
    decision_actor_is_self: usize,
    turn_owner_is_self: usize,
    is_setup: usize,
    is_free_road: usize,
    free_roads_available: usize,
    is_moving_knight: usize,
    discard_remaining_self: usize,
    discard_remaining_other: usize,
    phases: [usize; 5],
}

struct Encoder {
    job: Arc<Job>,
    base_features: Vec<f32>,
    is_proba: Vec<bool>,
    is_blue: Vec<bool>,
    idx: FeatureIndices,
    action_map: HashMap<(i32, i32, i32, i32), usize>,
}

fn feature_idx(map: &HashMap<&str, usize>, name: &str) -> usize {
    *map.get(name)
        .unwrap_or_else(|| panic!("Unknown feature {name}"))
}

impl Encoder {
    fn new(j: Arc<Job>) -> Self {
        let map: HashMap<&str, usize> = j
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let get = |name: &str| feature_idx(&map, name);

        let scalar_names = [
            "PUBLIC_VPS",
            "HAS_ARMY",
            "HAS_ROAD",
            "HAS_ROLLED",
            "ROADS_LEFT",
            "SETTLEMENTS_LEFT",
            "CITIES_LEFT",
            "LONGEST_ROAD_LENGTH",
            "NUM_RESOURCES_IN_HAND",
            "NUM_DEVS_IN_HAND",
            "DEV_PURCHASES",
        ];
        let p_scalars = std::array::from_fn(|rel| {
            std::array::from_fn(|k| get(&format!("P{rel}_{}", scalar_names[k])))
        });
        let p_played = std::array::from_fn(|rel| {
            std::array::from_fn(|i| get(&format!("P{rel}_{}_PLAYED", D[i])))
        });
        let p0_hand = std::array::from_fn(|i| get(&format!("P0_{}_IN_HAND", R[i])));
        let p1_inferred_hand =
            std::array::from_fn(|i| get(&format!("P1_INFERRED_{}_IN_HAND", R[i])));
        let bank_res = std::array::from_fn(|i| get(&format!("BANK_{}", R[i])));
        let p0_dev_in_hand = std::array::from_fn(|i| get(&format!("P0_{}_IN_HAND", D[i])));
        let p0_new_dev = std::array::from_fn(|i| get(&format!("P0_NEW_{}", D[i])));
        let p0_older_dev = std::array::from_fn(|i| get(&format!("P0_OLDER_{}", D[i])));
        let p0_dev_owned_start =
            std::array::from_fn(|i| get(&format!("P0_{}_OWNED_AT_START", D[i])));
        let p0_dev_playable = std::array::from_fn(|i| get(&format!("P0_{}_PLAYABLE", D[i])));
        let nodes = std::array::from_fn(|n| {
            std::array::from_fn(|rel| {
                [
                    get(&format!("NODE{n}_P{rel}_SETTLEMENT")),
                    get(&format!("NODE{n}_P{rel}_CITY")),
                ]
            })
        });
        let edges = j
            .initial
            .map
            .edges
            .iter()
            .map(|&[a, b]| {
                [
                    get(&format!("EDGE({a}, {b})_P0_ROAD")),
                    get(&format!("EDGE({a}, {b})_P1_ROAD")),
                ]
            })
            .collect();
        let tiles_robber = j
            .tile_ids
            .iter()
            .map(|id| get(&format!("TILE{id}_HAS_ROBBER")))
            .collect();
        let phase_names = [
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "PLAY_TURN",
            "MOVE_ROBBER",
            "DISCARD",
        ];
        let phases = std::array::from_fn(|i| get(&format!("PHASE_{}", phase_names[i])));

        let idx = FeatureIndices {
            p_scalars,
            p_played,
            p0_actual_vps: get("P0_ACTUAL_VPS"),
            p0_used_dev: get("P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"),
            p0_hand,
            p1_inferred_hand,
            bank_res,
            p0_dev_in_hand,
            p0_new_dev,
            p0_older_dev,
            p0_dev_owned_start,
            p0_dev_playable,
            p1_new_devs: get("P1_NEW_DEVS"),
            p1_older_devs: get("P1_OLDER_DEVS"),
            nodes,
            edges,
            tiles_robber,
            bank_dev_cards: get("BANK_DEV_CARDS"),
            is_discarding: get("IS_DISCARDING"),
            is_moving_robber: get("IS_MOVING_ROBBER"),
            actor_is_blue: get("ACTOR_IS_BLUE"),
            observer_is_blue: get("OBSERVER_IS_BLUE"),
            decision_actor_is_self: get("DECISION_ACTOR_IS_SELF"),
            turn_owner_is_self: get("TURN_OWNER_IS_SELF"),
            is_setup: get("IS_SETUP"),
            is_free_road: get("IS_FREE_ROAD"),
            free_roads_available: get("FREE_ROADS_AVAILABLE"),
            is_moving_knight: get("IS_MOVING_KNIGHT"),
            discard_remaining_self: get("DISCARD_REMAINING_SELF"),
            discard_remaining_other: get("DISCARD_REMAINING_OTHER"),
            phases,
        };

        let base_features = j.static_features.iter().map(|&x| x as f32).collect();
        let is_proba = j.names.iter().map(|n| n.ends_with("_PROBA")).collect();
        let is_blue = j.colors.iter().map(|c| c == "BLUE").collect();
        let mut action_map = HashMap::with_capacity(j.action_ids.len());
        for v in &j.action_ids {
            action_map
                .entry((v.action.k, v.action.a, v.action.b, v.rate))
                .or_insert(v.id);
        }

        Self {
            job: j,
            base_features,
            is_proba,
            is_blue,
            idx,
            action_map,
        }
    }

    fn features(&self, e: &Engine, h: &History, own: usize) -> Vec<f32> {
        let mut x = self.base_features.clone();
        let s = &e.state;
        let other = 1 - own;
        let idx = &self.idx;

        for (rel, p) in [own, other].iter().copied().enumerate() {
            let q = &s.players[p];
            let ps = &idx.p_scalars[rel];
            x[ps[0]] = q.vp as f32;
            x[ps[1]] = q.has_army as u8 as f32;
            x[ps[2]] = q.has_road as u8 as f32;
            x[ps[3]] = q.rolled as u8 as f32;
            x[ps[4]] = q.pieces[0] as f32;
            x[ps[5]] = q.pieces[1] as f32;
            x[ps[6]] = q.pieces[2] as f32;
            x[ps[7]] = q.road_length as f32;
            x[ps[8]] = q.hand.iter().sum::<i32>() as f32;
            x[ps[9]] = q.dev.iter().sum::<i32>() as f32;
            x[ps[10]] = h.buys[p] as f32;
            for i in 0..4 {
                x[idx.p_played[rel][i]] = q.played[i] as f32;
            }
        }

        let mut playable_mask = 0u8;
        let mut has_discard = false;
        let mut has_robber = false;
        if s.current == own {
            for a in &e.actions {
                match a.k {
                    8 => playable_mask |= 1 << 0,
                    10 => playable_mask |= 1 << 1,
                    9 => playable_mask |= 1 << 2,
                    11 => playable_mask |= 1 << 3,
                    6 => has_discard = true,
                    7 => has_robber = true,
                    _ => {}
                }
            }
        }

        let q = &s.players[own];
        x[idx.p0_actual_vps] = q.actual as f32;
        x[idx.p0_used_dev] = q.used_dev as u8 as f32;
        for i in 0..5 {
            x[idx.p0_hand[i]] = q.hand[i] as f32;
            x[idx.p1_inferred_hand[i]] = (19 - s.bank[i] - q.hand[i]) as f32;
            x[idx.bank_res[i]] = s.bank[i] as f32;
            x[idx.p0_dev_in_hand[i]] = q.dev[i] as f32;
            x[idx.p0_new_dev[i]] = h.new[own][i] as f32;
            x[idx.p0_older_dev[i]] = (q.dev[i] - h.new[own][i]) as f32;
        }
        for i in 0..4 {
            x[idx.p0_dev_owned_start[i]] = q.owned[i] as u8 as f32;
            x[idx.p0_dev_playable[i]] = ((playable_mask >> i) & 1) as f32;
        }

        let new = s.players[other].dev.iter().sum::<i32>();
        let p1_new = h.new[other].iter().sum::<i32>();
        x[idx.p1_new_devs] = p1_new as f32;
        x[idx.p1_older_devs] = (new - p1_new) as f32;

        for n in 0..54 {
            let b = s.buildings[n];
            for rel in 0..2 {
                let p = if rel == 0 { own } else { other };
                x[idx.nodes[n][rel][0]] = (b == [p as i32, 1]) as u8 as f32;
                x[idx.nodes[n][rel][1]] = (b == [p as i32, 2]) as u8 as f32;
            }
        }
        for (i, &[e0, e1]) in idx.edges.iter().enumerate() {
            let r = s.roads[i];
            x[e0] = (r == own as i32) as u8 as f32;
            x[e1] = (r == other as i32) as u8 as f32;
        }
        for (t, &ti) in idx.tiles_robber.iter().enumerate() {
            x[ti] = (t == s.robber) as u8 as f32;
        }

        x[idx.bank_dev_cards] = s.deck.len() as f32;
        x[idx.is_discarding] = has_discard as u8 as f32;
        x[idx.is_moving_robber] = has_robber as u8 as f32;
        x[idx.actor_is_blue] = self.is_blue[s.current] as u8 as f32;
        x[idx.observer_is_blue] = self.is_blue[own] as u8 as f32;
        x[idx.decision_actor_is_self] = (s.current == own) as u8 as f32;
        x[idx.turn_owner_is_self] = (s.turn == own) as u8 as f32;
        x[idx.is_setup] = s.initial as u8 as f32;
        x[idx.is_free_road] = s.road_building as u8 as f32;
        x[idx.free_roads_available] = s.free_roads as f32;
        x[idx.is_moving_knight] = s.moving as u8 as f32;
        x[idx.discard_remaining_self] = s.discards[own] as f32;
        x[idx.discard_remaining_other] = s.discards[other] as f32;
        for (i, &pi) in idx.phases.iter().enumerate() {
            x[pi] = (s.prompt == i) as u8 as f32;
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
        *self
            .action_map
            .get(&(a.k, a.a, a.b, rate))
            .expect("Action ID missing")
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
