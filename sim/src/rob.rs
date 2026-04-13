use std::f32::consts::PI;

use util::*;

pub const VIEW_DIST: f32 = 7.0 * SCALE as f32;

pub struct Robot {
    pub x: f32,
    pub y: f32,
    // pub front: f32,
    // pub left: f32,
    // pub right: f32,
    pub facing: f32,
}

impl Robot {
    /// Constructor.
    pub fn new(start: Pnt) -> Self {
        Robot {
            x: start.0,
            y: start.1,
            // front: 0.0,
            // left: 0.0,
            // right: 0.0,
            facing: PI / 4.0,
        }
    }

    /// Get the coords of the robot as framebuffer index.
    pub fn fb_coords(&self) -> u32 {
        let x = self.x.round() as u32;
        let y = if self.y.round() as usize <= HEIGHT {
            H32 - self.y.round() as u32
        } else {
            0
        };

        return x + y * W32;
    }

    /// Return the robot's position as a Pnt32.
    pub fn pnt32(&self) -> Pnt32 {
        return (self.x as u32, self.y as u32);
    }

    /// Return the robot's position as a Pnt.
    pub fn pnt(&self) -> Pnt {
        return (self.x, self.y);
    }

    /// Get the pixels in the FB that are in the robot's FOV.
    pub fn fov(&self) -> Vec<usize> {
        let mut pix = vec![];
        let mut angle_range = (
            (self.facing - (PI / 3.0)),
            (self.facing + (PI / 3.0)),
        );

        if angle_range.0 < 0.0 {
            angle_range.0 += 2.0 * PI;
            angle_range.1 += 2.0 * PI;
        }

        let range_low = coords((self.x - VIEW_DIST) as u32, (self.y + VIEW_DIST) as u32);
        let range_high = coords((self.x + VIEW_DIST) as u32, (self.y - VIEW_DIST) as u32);
        for i in range_low..range_high {
            let cart = (inv_coords(i).0 as f32, inv_coords(i).1 as f32);
            let dy = cart.1 - self.y;
            let dx = cart.0 - self.x;
            let mut angle = dy.atan2(dx);
            
            if dy < 0.0 || angle_range.1 > 2.0 * PI {
                angle += 2.0 * PI;
            }

            if dist(cart, (self.x, self.y)) <= VIEW_DIST
                && angle >= angle_range.0
                && angle <= angle_range.1
            {
                pix.push(i as usize);
            }
        }

        return pix;
    }

    /// Move forward one step.
    pub fn step(&mut self) {
        let (dy, dx) = self.facing.sin_cos();
        self.x += dx;
        self.y += dy;
    }

    /// Rotate to `r` degrees (0 is east)
    pub fn rot(&mut self, r: f32) {
        self.facing = r.to_radians();

        if self.facing >= 2.0 * PI {
            self.facing -= 2.0 * PI;
        }

        if self.facing < 0.0 {
            self.facing += 2.0 * PI;
        }
    }

    /// Move forward `d` steps.
    pub fn step_by(&mut self, d: f32) {
        let (dy, dx) = self.facing.sin_cos();
        self.x += d * dx;
        self.y += d * dy;
    }

    /// Rotate by `r` degrees.
    pub fn rot_by(&mut self, r: f32) {
        self.facing += r.to_radians();

        if self.facing >= 2.0 * PI {
            self.facing -= 2.0 * PI;
        }

        if self.facing < 0.0 {
            self.facing += 2.0 * PI;
        }
    }
}