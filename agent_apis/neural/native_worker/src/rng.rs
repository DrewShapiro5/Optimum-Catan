use sha2::{Digest, Sha256};
#[derive(Clone)]
pub struct Rng {
    mt: [u32; 624],
    at: usize,
}
pub fn named(seed: u64, name: &str) -> u64 {
    let x = Sha256::digest(format!("catan-benchmark-v1:{seed}:{name}").as_bytes());
    u64::from_be_bytes(x[..8].try_into().unwrap())
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Self {
            mt: [0; 624],
            at: 624,
        };
        r.mt[0] = 19650218;
        for i in 1..624 {
            r.mt[i] = 1812433253u32
                .wrapping_mul(r.mt[i - 1] ^ (r.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let key = if seed >> 32 == 0 {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        let (mut i, mut j) = (1, 0);
        for _ in 0..624.max(key.len()) {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1664525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
        }
        r.mt[0] = 0x80000000;
        r
    }
    pub fn from_state(state: &[u32]) -> Self {
        assert_eq!(state.len(), 625);
        Self {
            mt: state[..624].try_into().unwrap(),
            at: state[624] as usize,
        }
    }
    pub fn u32(&mut self) -> u32 {
        if self.at >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7fffffff);
                self.mt[i] =
                    self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.at = 0;
        }
        let mut y = self.mt[self.at];
        self.at += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    pub fn state(&self) -> Vec<u32> {
        let mut v = self.mt.to_vec();
        v.push(self.at as u32);
        v
    }
    pub fn uniform(&mut self) -> f64 {
        let a = (self.u32() >> 5) as u64;
        let b = (self.u32() >> 6) as u64;
        ((a << 26) + b) as f64 / 9007199254740992.
    }
    pub fn u64(&mut self) -> u64 {
        let low = self.u32() as u64;
        low | ((self.u32() as u64) << 32)
    }
    pub fn below(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        loop {
            let x = (self.u32() >> (32 - k)) as usize;
            if x < n {
                return x;
            }
        }
    }
}
