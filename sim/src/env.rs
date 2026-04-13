use clap::ValueEnum;
use rand::prelude::*;

use util::{self, dist, Pnt, Rect, SF32};

pub struct Env {
    pub rng: SmallRng,
    pub obs: Vec<Rect>,
    pub c: Clutter,
}

impl Env {
    pub fn new(c: Clutter, seed: Option<u64>) -> Self {
        match seed {
            Some(x) => Env {
                rng: SmallRng::seed_from_u64(x),
                obs: vec![],
                c,
            },
            None => Env {
                rng: SmallRng::from_entropy(),
                obs: vec![],
                c,
            },
        }
    }

    pub fn gen_obs(&mut self) {
        let (lo, hi) = match self.c {
            Clutter::Low => (3.0 * SF32, 4.0 * SF32),
            Clutter::Mid => (8.0 * SF32, 16.0 * SF32),
            Clutter::High => (17.0 * SF32, 23.0 * SF32),
            Clutter::Nah => (3.0 * SF32, 23.0 * SF32),
        };

        for _ in 0..self.rng.gen_range(lo as usize..hi as usize) {
            self.obs.push(Rect::random(&mut self.rng));
        }
    }

    pub fn obs_coord(&self) -> Vec<Pnt> {
        let mut ans = vec![];
        for block in &self.obs {
            ans.append(&mut block.to_coords());
        }

        return ans;
    }

    pub fn obs_bord(&self) -> Vec<Pnt> {
        let mut ans = vec![];
        for block in &self.obs {
            ans.append(&mut block.to_border_coords());
        }

        return ans;
    }

    pub fn obs_bord_lines(&self) -> Vec<(Pnt, Pnt)> {
        let mut ans = vec![];
        for block in &self.obs {
            ans.append(&mut block.to_border_lines());
        }

        return ans;
    }

    pub fn obs_bord_lines_local(&self, pos: Pnt, radius: f32) -> Vec<(Pnt, Pnt)> {
        let mut ans = vec![];
        'o: for block in &self.obs {
            let mut broke = false;
            'i: for i in block.to_coords() {
                if dist(i, pos) <= radius {
                    broke = true;
                    break 'i;
                }
            }

            if !broke { continue 'o; }

            ans.append(&mut block.to_border_lines());
        }

        return ans;
    }
}

#[derive(Clone, Copy, ValueEnum, Debug)]
pub enum Clutter {
    Low,
    Mid,
    High,
    Nah,
}