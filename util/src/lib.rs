pub mod macros;
pub mod trees;
pub mod wl;

use std::{collections::{HashMap, HashSet}, sync::Arc};

use anyhow::Result;
use rand::prelude::*;
use plotters::prelude::*;
use chrono::prelude::*;
use trees::*;

pub const SCALE: usize = 3;
pub const WSCALE: usize = 3;
pub const S32: u32 = SCALE as u32;
pub const SF32: f32 = 2.5;
pub const WIDTH: usize = 250;
pub const HEIGHT: usize = 250;
pub const W32: u32 = WIDTH as u32;
pub const H32: u32 = HEIGHT as u32;

pub type Pnt = (f32, f32);
pub type Pnt32 = (u32, u32);

pub type Tree = HashMap<usize, Arc<Node>>;
pub type CostFn = Option<(
    fn(Arc<Node>) -> f32,
    Arc<Node>,
    f32,
)>;

#[derive(Debug)]
pub struct Rect {
    pub coords: Pnt32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn random(rng: &mut SmallRng) -> Self {
        let coords = (rng.gen_range(10 * S32..W32), rng.gen_range(10 * S32..H32));
        let width = rng.gen_range(1 * S32..10 * S32);
        let height = rng.gen_range(1 * S32..10 * S32);

        Rect {
            coords,
            width,
            height,
        }
    }

    pub fn to_fb(&self) -> Vec<usize> {
        let mut pix = vec![];

        for i in self.coords.1 .. self.coords.1 + self.height {
            for j in self.coords.0 .. self.coords.0 + self.width {
                if i <= H32 && j <= W32 {
                    pix.push(coords(j, i) as usize);
                }
            }
        }

        return pix;
    }

    pub fn to_coords(&self) -> Vec<Pnt> {
        let mut pix = vec![];

        for i in self.coords.1 .. self.coords.1 + self.height {
            for j in self.coords.0 .. self.coords.0 + self.width {
                pix.push((j as f32, i as f32));
            }
        }

        return pix;
    }

    pub fn to_border_coords(&self) -> Vec<Pnt> {
        let mut pix = vec![];

        for i in self.coords.0 .. self.coords.0 + self.width {
            pix.push((i as f32, self.coords.1 as f32));
            pix.push((i as f32, self.coords.1 as f32 + self.height as f32));
        }

        for i in self.coords.1 .. self.coords.1 + self.height {
            pix.push((self.coords.0 as f32, i as f32));
            pix.push((self.coords.0 as f32 + self.width as f32, i as f32));
        }

        return pix;
    }

    pub fn to_border_lines(&self) -> Vec<(Pnt, Pnt)> {
        return vec![
            (
                (self.coords.0 as f32, self.coords.1 as f32),
                (self.coords.0 as f32, (self.coords.1 + self.height) as f32),
            ),
            (
                (self.coords.0 as f32, self.coords.1 as f32),
                ((self.coords.0 + self.width) as f32, self.coords.1 as f32),
            ),
            (
                ((self.coords.0 + self.width) as f32, self.coords.1 as f32),
                ((self.coords.0 + self.width) as f32, (self.coords.1 + self.height) as f32),
            ),
            (
                (self.coords.0 as f32, (self.coords.1 + self.height) as f32),
                ((self.coords.0 + self.width) as f32, (self.coords.1 + self.height) as f32),
            ),
        ];
    }
}

/********************************************************/
/****************   UTILS FROM RRT_LIB   ****************/
/********************************************************/

/// Distance between two points p1 and p2.
pub fn dist(p1: Pnt, p2: Pnt) -> f32 {
    return ((p1.1 - p2.1).powf(2.0) + (p1.0 - p2.0).powf(2.0)).sqrt();
}

/// Distance between two points p1 and p2.
pub fn dist_pnt_u8(p1: Pnt, p2: (u8, u8)) -> f32 {
    return ((p1.1 - p2.1 as f32).powf(2.0) + (p1.0 - p2.0 as f32).powf(2.0)).sqrt();
}

/// Calculate the span of the RRT (largest distance between any two nodes).
pub fn max_distance(tree: &Tree) -> f32 {
    let mut max_dist = 0.0;
    
    // Convert tree to a Vec for easier iteration
    let nodes: Vec<&Arc<Node>> = tree.values().collect();
    
    // Compare each pair of nodes
    for i in 0..nodes.len() {
        for j in (i+1)..nodes.len() {
            let p1 = nodes[i].pnt();
            let p2 = nodes[j].pnt();
            let distance = dist(p1, p2);
            
            if distance > max_dist {
                max_dist = distance;
            }
        }
    }
    
    return max_dist;
}

/// Approximate the area covered by the RRT using a grid occupancy method.
pub fn grid_occupancy_area(tree: &Tree, cell_size: f32) -> f32 {
    let mut occupied_cells = HashSet::new();
    
    // Convert each node position to a grid cell
    for node in tree.values() {
        let cell_x = (node.x / cell_size).floor() as i32;
        let cell_y = (node.y / cell_size).floor() as i32;
        occupied_cells.insert((cell_x, cell_y));
    }
    
    // Calculate area
    return (occupied_cells.len() as f32) * cell_size * cell_size;
}

/// Approximate the area of the RRT using a bounding box.
pub fn bbox_area(tree: &Tree) -> f32 {
    if tree.is_empty() {
        return 0.0;
    }
    
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    
    // Find the bounding box
    for node in tree.values() {
        min_x = min_x.min(node.x);
        min_y = min_y.min(node.y);
        max_x = max_x.max(node.x);
        max_y = max_y.max(node.y);
    }
    
    // Calculate the area of the bounding box
    (max_x - min_x) * (max_y - min_y)
}

#[allow(dead_code)]
/// Distance between point and line.
pub fn to_line(point: Pnt, line: (Pnt, Pnt)) -> f32 {
    let distance = dist(line.0, line.1);
    let ratio = (
        (point.0 - line.0.0) * (line.1.0 - line.0.0) 
        + (point.1 - line.0.1) * (line.1.1 - line.0.1)
    ) / (distance.powf(2.0));
    
    let int = (line.0.0 + ratio * (line.1.0 - line.0.0), line.0.1 + ratio * (line.1.1 - line.0.1));

    return dist(int, point);
}

/// Find closest node to a point, and optionally supply a cost function.
pub fn closest(nodes: &Tree, point: Pnt, cost_fn: CostFn) -> Arc<Node> {
    // Variable to store the closest point and the associated distance
    let mut closest = (1_000_000_000_000.0, nodes[&0].clone());
    
    // If a cost function is provided, use it
    if let Some(f) = cost_fn {
        for (_, node) in nodes {
            // Apply the cost function, using arguments if provided, and adjust cost by a factor
            let cost = dist(node.pnt(), point) + f.0(f.1.clone()) * f.2;

            // If the cost is less than the smallest cost found, update `closest`
            if cost < closest.0 && node.pnt() != point {
                closest = (dist(node.pnt(), point), node.clone());
            }
        }
    } else {
        // If no cost function provided, determine based purely on distance
        for (_, node) in nodes {
            if dist(node.pnt(), point) < closest.0 && node.pnt() != point {
                closest = (dist(node.pnt(), point), node.clone());
            }
        }
    }

    return closest.1;
}

/// Find closest node to a point and its index, and optionally supply a cost function.
pub fn closest_idx(nodes: &Tree, point: Pnt, cost_fn: CostFn) -> (Arc<Node>, usize) {
    // Variable to store the closest point and the associated distance
    let mut closest = (1_000_000_000_000.0, nodes[&0].clone(), 0);
    
    // If a cost function is provided, use it
    if let Some(f) = cost_fn {
        for (n, node) in nodes{
            // Apply the cost function, using arguments if provided, and adjust cost by a factor
            let cost = dist(node.pnt(), point) + f.0(f.1.clone()) * f.2;

            // If the cost is less than the smallest cost found, update `closest`
            if cost < closest.0 && node.pnt() != point {
                closest = (dist(node.pnt(), point), node.clone(), *n);
            }
        }
    } else {
        // If no cost function provided, determine based purely on distance
        for (n, node) in nodes {
            if dist(node.pnt(), point) < closest.0 && node.pnt() != point {
                closest = (dist(node.pnt(), point), node.clone(), *n);
            }
        }
    }

    return (closest.1, closest.2);
}

/// Find closest node to a point
pub fn closest_pnt(nodes: &Vec<Pnt>, point: Pnt) -> Pnt {
    // Variable to store the closest point and the associated distance
    let mut closest: (f32, Pnt) = (1_000_000_000_000.0, (0.0, 0.0));
    
    // If no cost function provided, determine based purely on distance
    for node in nodes {
        if dist(*node, point) < closest.0 && *node != point {
            closest = (dist(*node, point), node.clone());
        }
    }

    return closest.1;
}

pub fn nearby(nodes: &Tree, point: Pnt, radius: f32) -> Vec<Arc<Node>> {
    // Variable to store the closest point and the associated distance
    let mut nearby = Vec::new();

    for (_, node) in nodes {
        // If a node is within the radius, add it to `nearby`
        if dist(node.pnt(), point) <= radius && node.pnt() != point {
            nearby.push(node.clone());
        }
    }

    return nearby;
}

pub fn nearby_pnt(nodes: &Vec<Pnt>, point: Pnt, radius: f32) -> Vec<Pnt> {
    // Variable to store the closest point and the associated distance
    let mut nearby = Vec::new();

    for node in nodes {
        // If a node is within the radius, add it to `nearby`
        if dist(*node, point) <= radius {
            nearby.push(node.clone());
        }
    }

    return nearby;
}

pub fn fname_gen(utc: DateTime<Utc>, base: &str, suffix: &str, ext: &str) -> String {
    format!("{}{}-{:02}-{:02}_{:02}-{:02}-{:02}-{:04}{}.{}",
        base,
        utc.year(), utc.month(), utc.day(),
        utc.hour(), utc.minute(), utc.second(), utc.nanosecond(),
        suffix, ext
    )
} 

pub fn utc_format(utc: DateTime<Utc>) -> String {
    format!("{}-{:02}-{:02}_{:02}-{:02}-{:02}-{:04}",
        utc.year(), utc.month(), utc.day(),
        utc.hour(), utc.minute(), utc.second(), utc.nanosecond(),
    )
}

/// Output the graph to "\<out>".
pub fn make_chart(series: Vec<(&Tree, &RGBColor, ChartStyle)>, out: &str) -> Result<()> {
    let root = BitMapBackend::new(out, (1035, 1035)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..WIDTH as f32, 0f32..WIDTH as f32)?;

    chart.configure_mesh().draw()?;

    for graph in series {
        match graph.2 {
            ChartStyle::Node => {
                chart.draw_series(graph.0.iter().map(|(_, point)|
                    Circle::new(point.pnt(), 2, graph.1.filled())
                ))?;
            },
            ChartStyle::Edge => {
                for (_, i) in graph.0 {
                    let rabbit = if let Some(n) = i.par.lock().unwrap().upgrade() {
                        vec![n.clone().pnt(), i.pnt()]
                    } else {
                        vec![i.pnt()]
                    };

                    chart.draw_series(
                        LineSeries::new(
                            rabbit.iter().map(|point| *point),
                            graph.1,
                        )
                    )?;
                }
            },
            ChartStyle::NodeEdge => {
                for (_, i) in graph.0 {
                    let rabbit = if let Some(n) = i.par.lock().unwrap().upgrade() {
                        vec![n.clone().pnt(), i.pnt()]
                    } else {
                        vec![i.pnt()]
                    };

                    chart.draw_series(
                        LineSeries::new(
                            rabbit.iter().map(|point| *point),
                            graph.1,
                        )
                    )?;
                }

                chart.draw_series(graph.0.iter().map(|(_, point)|
                    Circle::new(point.pnt(), 2, graph.1.filled())
                ))?;
            }
        }
    }

    root.present()?;

    Ok(())
}

pub enum ChartStyle {
    Node,
    Edge,
    NodeEdge,
}

pub fn buf_to_pair<T: Copy>(buf: &mut Vec<T>) -> Vec<(T, T)> {
    let mut out = vec![];
    for i in (0..buf.len()).step_by(2) {
        out.push((buf[i], buf[i + 1]));
    }

    return out;
}

pub fn buf_to_ord<T: Copy>(ord: usize, buf: &mut Vec<T>) -> Vec<Vec<T>> {
    let mut out = vec![];
    for i in (0..buf.len()).step_by(ord) {
        let mut tmp = vec![];
        for j in 0..ord {
            tmp.push(buf[i + j]);
        }
        
        out.push(tmp);
    }

    return out;
}

/*****************************************************/
/****************   UTILS FROM SIML   ****************/
/*****************************************************/

/// Converts `x` and `y` coords to pixel buffer index
pub fn coords(x: u32, y: u32) -> u32 { x + (if H32 >= y { H32 - y - 1 } else { 0 }) * W32 }

/// Converts pixel buffer index to `x` and `y` coords
pub fn inv_coords(idx: u32) -> (u32, u32) {
    let x = idx % W32;
    let y = idx / W32;

    return (x, (if H32 >= y { H32 - y } else { 0 }));
}

/// Converts pixel buffer index to `x` and `y` coords
pub fn inv_coords_f32(idx: u32) -> (f32, f32) {
    let x = idx % W32;
    let y = idx / W32;

    return (x as f32, (if H32 >= y { H32 - y } else { 0 }) as f32);
}

/// Determines if a pixel `p` in the buffer is in a `*` shape around `q`.
/// Used to generate the marker that represents the agent.
pub fn is_agent(p: i32, q: i32, r: u32) -> bool {
    for i in 0 .. r {
        let a = p + i as i32;
        let b = p - i as i32;
        let c = p + (i * W32) as i32;
        let d = p - (i * W32) as i32;

        if a == q || b == q { return true; }
        if c == q || d == q { return true; }

        if c - i as i32 == q || c + i as i32 == q { return true; }
        if d - i as i32 == q || d + i as i32 == q { return true; }
    }

    return false;
}

pub fn gen_line(a: Pnt32, b: Pnt32, width: usize) -> Vec<usize> {
    let mut pix = vec![];
    
    // Point p is the leftmost point
    let mut p = if a.0 > b.0 { b } else { a };
    let mut q = if p == a { b } else { a };

    let dx = q.0 as i32 - p.0 as i32;

    // If the line is vertical and p is above q, swap p and q
    if dx == 0 && q.1 < p.1 {
        let tmp = p;
        p = q;
        q = tmp;
    }

    let dy = q.1 as i32 - p.1 as i32;
    let dx = q.0 as i32 - p.0 as i32;

    if dy.abs() > dx.abs() {
        // up 1, over dx/dy
        if q.1 > p.1 {
            let mut hz = p.0 as f32;
            for i in p.1..q.1 {
                hz += dx as f32 / dy as f32;
                if coords(hz as u32, i as u32) >= W32 * H32 { continue; }
                pix.push(coords(hz as u32, i as u32) as usize);
            }
        } else {
            let mut hz = q.0 as f32;
            for i in q.1..p.1 {
                hz += dx as f32 / dy as f32;
                if coords(hz as u32, i as u32) >= W32 * H32 { continue; }
                pix.push(coords(hz as u32, i as u32) as usize);
            }
        }
    } else {
        let mut vt = p.1 as f32;
        // over 1, up dy/dx
        for i in p.0..q.0 {
            vt += dy as f32 / dx as f32;
            if coords(i as u32, vt as u32) >= W32 * H32 { continue; }
            pix.push(coords(i as u32, vt as u32) as usize);
        }
    }

    if dx != 0 {
        for i in 0..=(width / 2) {
            let mut up: Vec<usize> = pix.iter().map(|x| x + i * WIDTH).collect();
            let mut down: Vec<usize> = pix.iter().map(|x|
                if i * WIDTH <= *x { x - i * WIDTH } else { 0 }
            ).collect();

            pix.append(&mut up);
            pix.append(&mut down);
        }
    } else {
        for i in 0..=(width / 2) {
            let mut up: Vec<usize> = pix.iter().map(|x| x + i).collect();
            let mut down: Vec<usize> = pix.iter().map(|x|
                if i <= *x { x - i } else { 0 }
            ).collect();

            pix.append(&mut up);
            pix.append(&mut down);
        }
    }

    return pix;
}

pub fn gen_circle(p: Pnt32, r: u32) -> Vec<usize> {
    // let mut circle = vec![];
    let area = Rect {
        coords: (p.0 - r, p.1 - r),
        width: 2 * r,
        height: 2 * r,
    };

    // for i in area.to_fb() {
    //     if dist(inv_coords_f32(i as u32), (p.0 as f32, p.1 as f32)) <= r as f32 {
    //         circle.push(i);
    //     }
    // }

    // return circle;
    return area.to_fb().into_iter().filter(|i| dist(inv_coords_f32(*i as u32), (p.0 as f32, p.1 as f32)) <= r as f32).collect();
}

pub fn tuple_type<In, Out: From<In>>(input: (In, In)) -> (Out, Out) {
    (input.0.into(), input.1.into())
}