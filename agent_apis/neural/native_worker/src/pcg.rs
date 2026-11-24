//! Native SeedSequence/PCG64 compatibility and NumPy's standalone C distribution
//! kernels. No Python API is called or linked. See native/ licensing notices.
use std::ffi::c_void;
#[repr(C)]
struct BitGen {
    state: *mut c_void,
    next64: unsafe extern "C" fn(*mut c_void) -> u64,
    next32: unsafe extern "C" fn(*mut c_void) -> u32,
    nextdouble: unsafe extern "C" fn(*mut c_void) -> f64,
    nextraw: unsafe extern "C" fn(*mut c_void) -> u64,
}
extern "C" {
    fn random_standard_gamma(g: *mut BitGen, shape: f64) -> f64;
    fn random_beta(g: *mut BitGen, a: f64, b: f64) -> f64;
}
#[derive(Clone)]
pub struct Pcg {
    state: u128,
    inc: u128,
    cache: Option<u32>,
}
fn hashmix(mut value: u32, c: &mut u32) -> u32 {
    value ^= *c;
    *c = c.wrapping_mul(0x931e8875);
    value = value.wrapping_mul(*c);
    value ^ (value >> 16)
}
fn mix(x: u32, y: u32) -> u32 {
    let v = 0xca01f9ddu32
        .wrapping_mul(x)
        .wrapping_sub(0x4973f715u32.wrapping_mul(y));
    v ^ (v >> 16)
}
impl Pcg {
    pub fn new(seed: u64) -> Self {
        let entropy = if seed >> 32 == 0 {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        let mut pool = [0u32; 4];
        let mut c = 0x43b0d7e5;
        for i in 0..4 {
            pool[i] = hashmix(*entropy.get(i).unwrap_or(&0), &mut c);
        }
        for src in 0..4 {
            for dst in 0..4 {
                if src != dst {
                    pool[dst] = mix(pool[dst], hashmix(pool[src], &mut c));
                }
            }
        }
        let mut c = 0x8b51f9ddu32;
        let mut words = [0u32; 8];
        for i in 0..8 {
            let mut v = pool[i % 4] ^ c;
            c = c.wrapping_mul(0x58f38ded);
            v = v.wrapping_mul(c);
            words[i] = v ^ (v >> 16);
        }
        let seed: Vec<u64> = words
            .chunks(2)
            .map(|x| x[0] as u64 | ((x[1] as u64) << 32))
            .collect();
        let initial = ((seed[0] as u128) << 64) | seed[1] as u128;
        let sequence = ((seed[2] as u128) << 64) | seed[3] as u128;
        let mut p = Self {
            state: 0,
            inc: (sequence << 1) | 1,
            cache: None,
        };
        p.step();
        p.state = p.state.wrapping_add(initial);
        p.step();
        p
    }
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(0x2360ed051fc65da44385df649fccf645)
            .wrapping_add(self.inc);
    }
    pub fn raw(&mut self) -> u64 {
        self.step();
        ((self.state >> 64) as u64 ^ self.state as u64).rotate_right((self.state >> 122) as u32)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.raw() >> 11) as f64 * (1. / 9007199254740992.)
    }
    fn bits32(&mut self) -> u32 {
        if let Some(c) = self.cache.take() {
            c
        } else {
            let v = self.raw();
            self.cache = Some((v >> 32) as u32);
            v as u32
        }
    }
    fn bitgen(&mut self) -> BitGen {
        BitGen {
            state: self as *mut _ as *mut c_void,
            next64: next64,
            next32: next32,
            nextdouble: nextdouble,
            nextraw: next64,
        }
    }
    pub fn dirichlet(&mut self, n: usize) -> Vec<f64> {
        assert!(n > 0);
        let a = 10. / n as f64;
        let mut bg = self.bitgen();
        let mut vals = vec![0.; n];
        if a < 0.1 {
            let mut tail = vec![0.; n];
            let mut sum = 0.;
            for i in (0..n).rev() {
                sum += a;
                tail[i] = sum;
            }
            let mut acc = 1.;
            for i in 0..n - 1 {
                let v = unsafe { random_beta(&mut bg, a, tail[i + 1]) };
                vals[i] = acc * v;
                acc *= 1. - v;
            }
            vals[n - 1] = acc;
        } else {
            let mut sum = 0.;
            for v in &mut vals {
                *v = unsafe { random_standard_gamma(&mut bg, a) };
                sum += *v;
            }
            let inv = 1. / sum;
            for v in &mut vals {
                *v *= inv;
            }
        }
        vals
    }
}
unsafe extern "C" fn next64(s: *mut c_void) -> u64 {
    (*(s as *mut Pcg)).raw()
}
unsafe extern "C" fn next32(s: *mut c_void) -> u32 {
    (*(s as *mut Pcg)).bits32()
}
unsafe extern "C" fn nextdouble(s: *mut c_void) -> f64 {
    (*(s as *mut Pcg)).uniform()
}
