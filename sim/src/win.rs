use anyhow::Result;
use minifb::{Window, WindowOptions};
use rayon::prelude::*;

use util::*;
// use algebra::simple as alg;

use crate::colors::*;
use crate::rob::{Robot, VIEW_DIST};
use crate::env::{Clutter, Env};

pub struct Win {
    pub buf: Vec<u32>,
    pub win: Option<Window>,
    pub rob: Robot,
    pub env: Env,
}

impl Win {
    /// Create a new simulator with a window to display visual output.
    pub fn new_window(start: Pnt, scale: minifb::Scale, c: Clutter, seed: Option<u64>) -> Result<Self> {
        let mut me = Self {
            buf: vec![0; WIDTH * HEIGHT],
            win: Some(Window::new(
                "Robot Simulator",
                WIDTH,
                HEIGHT,
                WindowOptions {
                    scale,
                    ..WindowOptions::default()
                }
            )?),
            rob: Robot::new(start),
            env: Env::new(c, seed),
        };

        me.win.as_mut().unwrap().set_target_fps(0);

        me.env.gen_obs();
        // me.env.obs.clear();
        for block in &me.env.obs {
            for i in block.to_fb() {
                if i >= me.buf.len() { continue; }
                if dist(inv_coords_f32(i as u32), me.rob.pnt()) < 40.0 { continue; }
                me.buf[i] = DPURPLE;
            }
        }

        // let start_circle = util::gen_circle((start.0 as u32, start.1 as u32), 40);
        // for i in start_circle {
        //     me.buf[i] = 0xFF0001;
        // }

        Ok(me)
    }

    /// Create a new simulator without a window. Use when speed is key.
    pub fn new_headless(start: Pnt, c: Clutter, seed: Option<u64>) -> Result<Self> {
        let mut me = Self {
            buf: vec![0; WIDTH * HEIGHT],
            win: None,
            rob: Robot::new(start),
            env: Env::new(c, seed),
        };

        me.env.gen_obs();
        // me.env.obs.clear();
        for block in &me.env.obs {
            for i in block.to_fb() {
                if i >= me.buf.len() { continue; }
                if dist(inv_coords_f32(i as u32), me.rob.pnt()) < 40.0 { continue; }
                me.buf[i] = DPURPLE;
            }
        }

        // let start_circle = util::gen_circle((start.0 as u32, start.1 as u32), 40);
        // for i in start_circle {
        //     me.buf[i] = 0xFF0001;
        // }

        Ok(me)
    }

    pub fn update(&mut self) -> Result<()> {
        for (n, i) in self.buf.iter_mut().enumerate() {
            if n as u32 == self.rob.fb_coords()
                && *i != GREEN
                && *i != DPURPLE
                && *i != BLUE
            {
                *i = RED;
            } else if *i == YELLOW || *i == RED {
                *i = LYELLOW;
            } else if *i != GREEN
                && *i != DPURPLE
                && *i != LYELLOW
                && *i != BLUE
            {
                *i = BLACK;
            }

            // if n == self.rob.fb_coords() as usize - 1 {
            //     *i = GREEN;
            // }
        }

        for i in self.rob.fov() {
            if self.buf[i] != GREEN
                && self.buf[i] != DPURPLE
                && self.buf[i] != BLUE
                && self.buf[i] != RED
            {
                let line = util::gen_line(self.rob.pnt32(), inv_coords(i as u32), 1);
                if !line.iter().any(|x| self.buf[*x] == DPURPLE) {
                    self.buf[i] = YELLOW;
                }
            }
        }

        // lines_test(&mut self.buf);

        if let Some(ref mut x) = self.win { x.update_with_buffer(&self.buf, WIDTH, HEIGHT)?; }

        Ok(())
    }

    /// Draw some pixels to the buffer
    pub fn draw(&mut self, pix: &Vec<u32>, color: u32) -> Result<()> {
        for i in pix {
            if (*i as usize) < self.buf.len() {
                self.buf[*i as usize] = color;
            }
        }

        if let Some(ref mut x) = self.win { x.update_with_buffer(&self.buf, WIDTH, HEIGHT)?; }
        Ok(())
    }

    pub fn erase(&mut self, color: u32) -> Result<()> {
        for i in &mut self.buf {
            if *i == color {
                *i = BLACK;
            }
        }

        if let Some(ref mut x) = self.win { x.update_with_buffer(&self.buf, WIDTH, HEIGHT)?; }
        Ok(())
    }

    pub fn replace(&mut self, color: u32, into: u32) -> Result<()> {
        for i in &mut self.buf {
            if *i == color {
                *i = into;
            }
        }

        if let Some(ref mut x) = self.win { x.update_with_buffer(&self.buf, WIDTH, HEIGHT)?; }
        Ok(())
    }

    /// Check if the pixel at `i` is next to a specific color
    pub fn is_touching(&self, i: i32, color: u32, emarg: bool) -> bool {
        let up = (i - W32 as i32) as usize;
        let down = (i + W32 as i32) as usize;
        let left = (i - 1 as i32) as usize;
        let right = (i + 1 as i32) as usize;
        let botleft = (i + W32 as i32 - 1) as usize;
        let botright = (i + W32 as i32 + 1) as usize;
        let topleft = (i - W32 as i32 - 1) as usize;
        let topright = (i - W32 as i32 + 1) as usize;
        let up2 = (i - (2 * W32) as i32) as usize;
        let down2 = (i + (2 * W32) as i32) as usize;
        let left2 = (i - 2 as i32) as usize;
        let right2 = (i + 2 as i32) as usize;
        let up3 = (i - (3 * W32) as i32) as usize;
        let down3 = (i + (3 * W32) as i32) as usize;
        let left3 = (i - 3 as i32) as usize;
        let right3 = (i + 3 as i32) as usize;

        // Cardinal directions
        if up > 0 && up < self.buf.len() {
            if self.buf[up] == color { return true; }
        }
        
        if down < self.buf.len() {
            if self.buf[down] == color { return true; }
        }
        
        if left > 0 && left < self.buf.len() {
            if self.buf[left] == color { return true; }
        }
        
        if right < self.buf.len() {
            if self.buf[right] == color { return true; }
        }

        // Corners
        if topleft > 0 && topleft < self.buf.len() {
            if self.buf[topleft] == color { return true; }
        }
        
        if topright > 0 && topright < self.buf.len() {
            if self.buf[topright] == color { return true; }
        }
        
        if botleft > 0 && botleft < self.buf.len() {
            if self.buf[botleft] == color { return true; }
        }
        
        if botright < self.buf.len() {
            if self.buf[botright] == color { return true; }
        }

        // Gap of one pixel in cardinal directions
        if emarg {
            if up2 > 0 && up2 < self.buf.len() {
                if self.buf[up2] == color { return true; }
            }
            
            if down2 < self.buf.len() {
                if self.buf[down2] == color { return true; }
            }
            
            if left2 > 0 && left2 < self.buf.len() {
                if self.buf[left2] == color { return true; }
            }
            
            if right2 < self.buf.len() {
                if self.buf[right2] == color { return true; }
            }

            if up3 > 0 && up3 < self.buf.len() {
                if self.buf[up3] == color { return true; }
            }
            
            if down3 < self.buf.len() {
                if self.buf[down3] == color { return true; }
            }
            
            if left3 > 0 && left3 < self.buf.len() {
                if self.buf[left3] == color { return true; }
            }
            
            if right3 < self.buf.len() {
                if self.buf[right3] == color { return true; }
            }
        }

        return false;
    }
}

#[allow(dead_code)]
fn lines_test(buf: &mut Vec<u32>) {
    for i in gen_line((100, 100), (125, 150), 2) {
        buf[i] = BLUE;
    }

    for i in gen_line((100, 100), (150, 125), 2) {
        buf[i] = CYAN;
    }

    // for i in gen_line((100, 100), (50, 75), 2) {
    //     buf[i] = PINK;
    // }

    // for i in gen_line((100, 100), (75, 50), 2) {
    //     buf[i] = MAGENTA;
    // }

    // for i in gen_line((100, 100), (75, 150), 2) {
    //     buf[i] = ORANGE;
    // }

    // for i in gen_line((100, 100), (50, 125), 2) {
    //     buf[i] = BROWN;
    // }

    for i in gen_line((100, 100), (150, 75), 2) {
        buf[i] = GREEN;
    }

    for i in gen_line((100, 100), (125, 50), 2) {
        buf[i] = RED;
    }

    // for i in gen_line((100, 100), (100, 175), 2) {
    //     buf[i] = DGREEN;
    // }

    // for i in gen_line((100, 100), (100, 25), 2) {
    //     buf[i] = DCYAN;
    // }

    // for i in gen_line((100, 100), (175, 100), 2) {
    //     buf[i] = DBLUE;
    // }

    // for i in gen_line((100, 100), (25, 100), 2) {
    //     buf[i] = DRED;
    // }
}