//! Native port of this checkout's generated-action self-play rules.
//! Chance is supplied explicitly, allowing identical dice/steals/deck replays.
//! Legacy rust-catan rules and Python bindings are not used by this module.
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Map {
    pub edges: Vec<[usize; 2]>,
    pub tiles: Vec<Tile>,
    pub ports: Vec<Vec<i32>>,
    pub adjacent: Vec<Vec<usize>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tile {
    pub nodes: Vec<usize>,
    pub resource: i32,
    pub number: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Player {
    pub hand: [i32; 5],
    pub dev: [i32; 5],
    pub played: [i32; 5],
    pub owned: [bool; 4],
    pub rolled: bool,
    pub used_dev: bool,
    pub vp: i32,
    pub actual: i32,
    pub pieces: [i32; 3],
    pub has_road: bool,
    pub has_army: bool,
    pub road_length: usize,
    pub settlements: Vec<usize>,
    pub cities: Vec<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    pub players: Vec<Player>,
    pub bank: [i32; 5],
    pub deck: Vec<usize>,
    pub buildings: Vec<[i32; 2]>,
    pub roads: Vec<i32>,
    pub components: Vec<Vec<Vec<usize>>>,
    pub buildable: Vec<bool>,
    pub road_lengths: Vec<(usize, usize)>,
    pub road_color: i32,
    pub road_length: usize,
    pub robber: usize,
    pub current: usize,
    pub turn: usize,
    pub turns: usize,
    pub prompt: usize, // 0 initial settlement, 1 initial road, 2 turn, 3 robber, 4 discard
    pub initial: bool,
    pub discarding: bool,
    pub moving: bool,
    pub free_roads: usize,
    pub road_building: bool,
    pub discards: Vec<i32>,
    pub discard_limit: i32,
    pub friendly: bool,
    pub target: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Act {
    pub k: i32,
    pub a: i32,
    pub b: i32,
}
impl Act {
    fn new(k: i32, a: i32, b: i32) -> Self {
        Self { k, a, b }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Engine {
    pub map: Arc<Map>,
    pub state: Position,
    #[serde(skip)]
    pub actions: Vec<Act>,
}
const ROAD: [i32; 5] = [1, 1, 0, 0, 0];
const SETTLEMENT: [i32; 5] = [1, 1, 1, 1, 0];
const CITY: [i32; 5] = [0, 0, 0, 2, 3];
const DEV: [i32; 5] = [0, 0, 1, 1, 1];
impl Engine {
    fn connected(&self, p: usize, n: usize) -> bool {
        self.state.components[p].iter().any(|c| c.contains(&n))
    }
    fn roads_available(&self, p: usize) -> Vec<Act> {
        if self.state.players[p].pieces[0] == 0 {
            return vec![];
        }
        self.map
            .edges
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                self.state.roads[*i] < 0 && (self.connected(p, e[0]) || self.connected(p, e[1]))
            })
            .map(|(i, _)| Act::new(3, i as i32, -1))
            .collect()
    }
    fn can_dev(&self, p: usize, d: usize) -> bool {
        let q = &self.state.players[p];
        !q.used_dev && q.dev[d] > 0 && q.owned[d]
    }
    fn afford(&self, p: usize, cost: [i32; 5]) -> bool {
        (0..5).all(|r| self.state.players[p].hand[r] >= cost[r])
    }
    pub fn legal(&self) -> Vec<Act> {
        let s = &self.state;
        let p = s.current;
        let q = &s.players[p];
        if s.prompt == 0 {
            return s
                .buildable
                .iter()
                .enumerate()
                .filter(|(_, v)| **v)
                .map(|(n, _)| Act::new(2, n as i32, -1))
                .collect();
        }
        if s.prompt == 1 {
            let n = *q.settlements.last().unwrap();
            return self
                .roads_available(p)
                .into_iter()
                .filter(|a| self.map.edges[a.a as usize].contains(&n))
                .collect();
        }
        if s.prompt == 4 {
            return (0..5)
                .filter(|&r| q.hand[r] > 0 && s.discards[p] > 0)
                .map(|r| Act::new(6, r as i32, -1))
                .collect();
        }
        if s.prompt == 3 {
            let mut all = vec![];
            let mut friendly = vec![];
            for (t, tile) in self.map.tiles.iter().enumerate() {
                if t == s.robber {
                    continue;
                }
                let mut victims = vec![];
                let mut protected = false;
                for &n in &tile.nodes {
                    let owner = s.buildings[n][0];
                    if owner >= 0 && owner as usize != p {
                        let v = owner as usize;
                        protected |= s.players[v].vp < 3;
                        if s.players[v].hand.iter().sum::<i32>() > 0 && !victims.contains(&owner) {
                            victims.push(owner);
                        }
                    }
                }
                if victims.is_empty() {
                    victims.push(-1);
                }
                for v in victims {
                    let a = Act::new(7, t as i32, v);
                    all.push(a);
                    if !protected {
                        friendly.push(a);
                    }
                }
            }
            return if s.friendly && !friendly.is_empty() {
                friendly
            } else {
                all
            };
        }
        assert_eq!(s.prompt, 2);
        if s.road_building {
            return self.roads_available(p);
        }
        let mut a = vec![];
        if self.can_dev(p, 2) {
            // Match Catanatron's unusual one-card alternatives even if a different pair exists.
            for r in 0..5 {
                for t in r..5 {
                    if s.bank[r] >= if r == t { 2 } else { 1 } && s.bank[t] >= 1 {
                        a.push(Act::new(9, r as i32, t as i32));
                    } else {
                        if s.bank[r] > 0 {
                            a.push(Act::new(9, r as i32, -1));
                        }
                        if s.bank[t] > 0 {
                            a.push(Act::new(9, t as i32, -1));
                        }
                    }
                }
            }
        }
        if self.can_dev(p, 1) {
            for r in 0..5 {
                a.push(Act::new(10, r, -1));
            }
        }
        if self.can_dev(p, 0) {
            a.push(Act::new(8, -1, -1));
        }
        if self.can_dev(p, 3) && !self.roads_available(p).is_empty() {
            a.push(Act::new(11, -1, -1));
        }
        if !q.rolled {
            a.push(Act::new(1, -1, -1));
        } else {
            a.push(Act::new(0, -1, -1));
            if self.afford(p, ROAD) {
                a.extend(self.roads_available(p));
            }
            if q.pieces[1] > 0 && self.afford(p, SETTLEMENT) {
                for (n, &v) in s.buildable.iter().enumerate() {
                    if v && self.connected(p, n) {
                        a.push(Act::new(2, n as i32, -1));
                    }
                }
            }
            if q.pieces[2] > 0 && self.afford(p, CITY) {
                for &n in &q.settlements {
                    a.push(Act::new(4, n as i32, -1));
                }
            }
            if !s.deck.is_empty() && self.afford(p, DEV) {
                a.push(Act::new(5, -1, -1));
            }
            let mut rates = [4; 5];
            for (n, b) in s.buildings.iter().enumerate() {
                if b[0] == p as i32 {
                    for &r in &self.map.ports[n] {
                        if r < 0 {
                            for v in &mut rates {
                                *v = (*v).min(3);
                            }
                        } else {
                            rates[r as usize] = 2;
                        }
                    }
                }
            }
            for r in 0..5 {
                if q.hand[r] >= rates[r] {
                    for t in 0..5 {
                        if r != t && s.bank[t] > 0 {
                            a.push(Act::new(12, r as i32, t as i32));
                        }
                    }
                }
            }
        }
        a.sort();
        a.dedup();
        a
    }
    fn enemy(&self, n: usize, p: usize) -> bool {
        let o = self.state.buildings[n][0];
        o >= 0 && o != p as i32
    }
    fn edge_index(&self, a: usize, b: usize) -> usize {
        self.map
            .edges
            .iter()
            .position(|e| *e == [a.min(b), a.max(b)])
            .unwrap()
    }
    fn path_len(&self, n: usize, p: usize, used: u128) -> usize {
        let mut best = 0;
        for &v in &self.map.adjacent[n] {
            let e = self.edge_index(n, v);
            if self.state.roads[e] == p as i32 && !self.enemy(v, p) && used & (1u128 << e) == 0 {
                best = best.max(1 + self.path_len(v, p, used | (1u128 << e)));
            }
        }
        best
    }
    fn longest(&self, c: &[usize], p: usize) -> usize {
        c.iter().map(|&n| self.path_len(n, p, 0)).max().unwrap_or(0)
    }
    fn walk(&self, start: usize, p: usize) -> Vec<usize> {
        let mut agenda = vec![start];
        let mut seen = vec![];
        while let Some(n) = agenda.pop() {
            if !seen.contains(&n) {
                seen.push(n);
            }
            if self.enemy(n, p) {
                continue;
            }
            for &v in &self.map.adjacent[n] {
                if !seen.contains(&v) && self.state.roads[self.edge_index(n, v)] == p as i32 {
                    agenda.push(v);
                }
            }
        }
        seen
    }
    fn component(&self, p: usize, n: usize) -> Option<usize> {
        self.state.components[p].iter().position(|c| c.contains(&n))
    }
    fn set_length(&mut self, p: usize, len: usize, max: bool) {
        if let Some(v) = self.state.road_lengths.iter_mut().find(|v| v.0 == p) {
            v.1 = if max { v.1.max(len) } else { len };
        } else {
            self.state.road_lengths.push((p, len));
        }
    }
    fn maintain_road(&mut self, old: i32) {
        let s = &mut self.state;
        for &(p, n) in &s.road_lengths {
            s.players[p].road_length = n;
        }
        let new = s.road_color;
        if new < 0 || new == old {
            return;
        }
        let q = &mut s.players[new as usize];
        q.has_road = true;
        q.vp += 2;
        q.actual += 2;
        if old >= 0 {
            let q = &mut s.players[old as usize];
            q.has_road = false;
            q.vp -= 2;
            q.actual -= 2;
        }
    }
    fn pay(&mut self, p: usize, cost: [i32; 5]) {
        for r in 0..5 {
            self.state.players[p].hand[r] -= cost[r];
            self.state.bank[r] += cost[r];
        }
    }
    fn advance(&mut self, d: i32) {
        let s = &mut self.state;
        s.current =
            ((s.current as i32 + d + s.players.len() as i32) % s.players.len() as i32) as usize;
        s.turn = s.current;
        s.turns += 1;
    }
    fn play_dev(&mut self, p: usize, d: usize) {
        let old = self.state.players.iter().position(|q| q.has_army);
        let old_size = old.map(|o| self.state.players[o].played[0]).unwrap_or(0);
        let q = &mut self.state.players[p];
        q.dev[d] -= 1;
        q.played[d] += 1;
        q.used_dev = true;
        if d == 0 && q.played[0] >= 3 && old != Some(p) && (old.is_none() || q.played[0] > old_size)
        {
            q.has_army = true;
            q.vp += 2;
            q.actual += 2;
            if let Some(o) = old {
                let q = &mut self.state.players[o];
                q.has_army = false;
                q.vp -= 2;
                q.actual -= 2;
            }
        }
    }
    /// Apply an already legal action and an explicit chance result. No Python is called.
    pub fn apply(&mut self, a: Act, chance: i32) {
        let p = self.state.current;
        match a.k {
            0 => {
                let q = &mut self.state.players[p];
                q.used_dev = false;
                q.rolled = false;
                for d in 0..4 {
                    q.owned[d] = q.dev[d] > 0;
                }
                self.advance(1);
                self.state.prompt = 2;
            }
            1 => {
                self.state.players[p].rolled = true;
                if chance == 7 {
                    let s = &mut self.state;
                    for (i, q) in s.players.iter().enumerate() {
                        let n = q.hand.iter().sum::<i32>();
                        s.discards[i] = if n > s.discard_limit { n / 2 } else { 0 };
                    }
                    if let Some(i) = s.discards.iter().position(|&n| n > 0) {
                        s.current = i;
                        s.prompt = 4;
                        s.discarding = true;
                    } else {
                        s.prompt = 3;
                        s.moving = true;
                    }
                } else {
                    let mut payout = vec![[0; 5]; self.state.players.len()];
                    for (t, tile) in self.map.tiles.iter().enumerate() {
                        if t == self.state.robber || tile.number != chance || tile.resource < 0 {
                            continue;
                        }
                        for &n in &tile.nodes {
                            let b = self.state.buildings[n];
                            if b[0] >= 0 {
                                payout[b[0] as usize][tile.resource as usize] += b[1];
                            }
                        }
                    }
                    for r in 0..5 {
                        let total = payout.iter().map(|v| v[r]).sum::<i32>();
                        if total <= self.state.bank[r] {
                            self.state.bank[r] -= total;
                            for (i, v) in payout.iter().enumerate() {
                                self.state.players[i].hand[r] += v[r];
                            }
                        }
                    }
                    self.state.prompt = 2;
                }
            }
            2 => {
                let n = a.a as usize;
                let initial = self.state.initial;
                let old = self.state.road_color;
                self.state.buildings[n] = [p as i32, 1];
                self.state.players[p].settlements.push(n);
                {
                    let q = &mut self.state.players[p];
                    q.pieces[1] -= 1;
                    q.vp += 1;
                    q.actual += 1;
                }
                if initial {
                    self.state.components[p].push(vec![n]);
                } else {
                    // Preserve this checkout's connected-component and road-cut semantics.
                    let mut grouped: Vec<(i32, Vec<usize>)> = vec![];
                    for &v in &self.map.adjacent[n] {
                        let owner = self.state.roads[self.edge_index(n, v)];
                        if let Some(g) = grouped.iter_mut().find(|g| g.0 == owner) {
                            g.1.push(v);
                        } else {
                            grouped.push((owner, vec![v]));
                        }
                    }
                    for (owner, vs) in grouped {
                        if owner < 0 || owner == p as i32 {
                            continue;
                        }
                        let o = owner as usize;
                        if vs.len() == 2 {
                            let left = self.walk(vs[0], o);
                            let right = self.walk(vs[1], o);
                            let idx = self.component(o, n).unwrap();
                            self.state.components[o].remove(idx);
                            self.state.components[o].push(left);
                            self.state.components[o].push(right);
                            let len = self.state.components[o]
                                .iter()
                                .map(|c| self.longest(c, o))
                                .max()
                                .unwrap_or(0);
                            self.set_length(o, len, false);
                            // Python max keeps the first insertion on a tie, even below five.
                            if let Some(&(winner, length)) = self
                                .state
                                .road_lengths
                                .iter()
                                .max_by(|a, b| a.1.cmp(&b.1).then(std::cmp::Ordering::Greater))
                            {
                                self.state.road_color = winner as i32;
                                self.state.road_length = length;
                            }
                        } else if vs.len() == 1 {
                            if let Some(idx) = self.component(o, n) {
                                self.state.components[o][idx].retain(|&v| v != n);
                            }
                        }
                    }
                    self.pay(p, SETTLEMENT);
                    self.maintain_road(old);
                }
                self.state.buildable[n] = false;
                for &v in &self.map.adjacent[n] {
                    self.state.buildable[v] = false;
                }
                if initial {
                    if self.state.players[p].settlements.len() == 2 {
                        for tile in &self.map.tiles {
                            if tile.nodes.contains(&n) && tile.resource >= 0 {
                                let r = tile.resource as usize;
                                self.state.bank[r] -= 1;
                                self.state.players[p].hand[r] += 1;
                            }
                        }
                    }
                    self.state.prompt = 1;
                }
            }
            3 => {
                let e = a.a as usize;
                let [a, b] = self.map.edges[e];
                let old = self.state.road_color;
                self.state.roads[e] = p as i32;
                self.state.players[p].pieces[0] -= 1;
                let ai = self.component(p, a);
                let bi = self.component(p, b);
                let idx;
                if ai.is_none() && !self.enemy(a, p) {
                    idx = bi.unwrap();
                    self.state.components[p][idx].push(a);
                } else if bi.is_none() && !self.enemy(b, p) {
                    idx = ai.unwrap();
                    self.state.components[p][idx].push(b);
                } else if ai.is_some() && bi.is_some() && ai != bi {
                    let ia = ai.unwrap();
                    let ib = bi.unwrap();
                    let mut c = self.state.components[p][ia].clone();
                    for &v in &self.state.components[p][ib] {
                        if !c.contains(&v) {
                            c.push(v);
                        }
                    }
                    self.state.components[p][ia] = c;
                    self.state.components[p].remove(ib);
                    idx = if ib < ia { ia - 1 } else { ia };
                } else {
                    idx = ai.or(bi).unwrap();
                }
                let length = self.longest(&self.state.components[p][idx], p);
                self.set_length(p, length, true);
                if length >= 5 && length > self.state.road_length {
                    self.state.road_color = p as i32;
                    self.state.road_length = length;
                }
                if self.state.initial {
                    let count = self
                        .state
                        .players
                        .iter()
                        .map(|q| q.settlements.len())
                        .sum::<usize>();
                    let n = self.state.players.len();
                    if count < n {
                        self.advance(1);
                        self.state.prompt = 0;
                    } else if count == n {
                        self.state.prompt = 0;
                    } else if count == 2 * n {
                        self.state.initial = false;
                        self.state.prompt = 2;
                    } else {
                        self.advance(-1);
                        self.state.prompt = 0;
                    }
                } else {
                    self.maintain_road(old);
                    if self.state.road_building && self.state.free_roads > 0 {
                        self.state.free_roads -= 1;
                        if self.state.free_roads == 0 || self.roads_available(p).is_empty() {
                            self.state.road_building = false;
                            self.state.free_roads = 0;
                        }
                    } else {
                        self.pay(p, ROAD);
                    }
                }
            }
            4 => {
                let n = a.a as usize;
                self.state.buildings[n][1] = 2;
                let q = &mut self.state.players[p];
                q.settlements.retain(|&v| v != n);
                q.cities.push(n);
                q.pieces[1] += 1;
                q.pieces[2] -= 1;
                q.vp += 1;
                q.actual += 1;
                self.pay(p, CITY);
            }
            5 => {
                let d = chance as usize;
                // Search branches can force a draw other than the top card.
                // Match apply_buy_development_card: pop a matching top card,
                // otherwise remove the first occurrence without reordering.
                if self.state.deck.last() == Some(&d) {
                    self.state.deck.pop();
                } else {
                    let index = self.state.deck.iter().position(|&card| card == d).unwrap();
                    self.state.deck.remove(index);
                }
                self.state.players[p].dev[d] += 1;
                if d == 4 {
                    self.state.players[p].actual += 1;
                }
                self.pay(p, DEV);
            }
            6 => {
                let r = a.a as usize;
                self.state.players[p].hand[r] -= 1;
                self.state.bank[r] += 1;
                self.state.discards[p] -= 1;
                if self.state.discards[p] == 0 {
                    if let Some(i) =
                        ((p + 1)..self.state.players.len()).find(|&i| self.state.discards[i] > 0)
                    {
                        self.state.current = i;
                    } else {
                        self.state.current = self.state.turn;
                        self.state.prompt = 3;
                        self.state.discarding = false;
                        self.state.moving = true;
                    }
                }
            }
            7 => {
                self.state.robber = a.a as usize;
                if a.b >= 0 {
                    let r = chance as usize;
                    self.state.players[a.b as usize].hand[r] -= 1;
                    self.state.players[p].hand[r] += 1;
                }
                self.state.prompt = 2;
            }
            8 => {
                self.play_dev(p, 0);
                self.state.prompt = 3;
            }
            9 => {
                for r in [a.a, a.b] {
                    if r >= 0 {
                        self.state.bank[r as usize] -= 1;
                        self.state.players[p].hand[r as usize] += 1;
                    }
                }
                self.play_dev(p, 2);
                self.state.prompt = 2;
            }
            10 => {
                let r = a.a as usize;
                let mut total = 0;
                for (i, q) in self.state.players.iter_mut().enumerate() {
                    if i != p {
                        total += q.hand[r];
                        q.hand[r] = 0;
                    }
                }
                self.state.players[p].hand[r] += total;
                self.play_dev(p, 1);
                self.state.prompt = 2;
            }
            11 => {
                self.play_dev(p, 3);
                self.state.road_building = true;
                self.state.free_roads = 2;
                self.state.prompt = 2;
            }
            12 => {
                let r = a.a as usize;
                let t = a.b as usize;
                let mut rate = 4;
                for (n, b) in self.state.buildings.iter().enumerate() {
                    if b[0] == p as i32 {
                        for &port in &self.map.ports[n] {
                            if port < 0 {
                                rate = rate.min(3);
                            } else if port as usize == r {
                                rate = 2;
                            }
                        }
                    }
                }
                self.state.players[p].hand[r] -= rate;
                self.state.bank[r] += rate;
                self.state.players[p].hand[t] += 1;
                self.state.bank[t] -= 1;
            }
            _ => panic!("Unsupported action"),
        }
    }
    pub fn refresh(&mut self) {
        self.actions = self.legal();
    }
    pub fn winner(&self) -> i32 {
        self.state
            .players
            .iter()
            .enumerate()
            .filter(|(_, q)| q.actual >= self.state.target)
            .map(|(i, _)| i as i32)
            .last()
            .unwrap_or(-1)
    }
}
