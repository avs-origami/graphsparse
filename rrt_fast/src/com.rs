use std::sync::{Arc, Mutex};

use anyhow::Result;

use rayon::prelude::*;
use rrt_lib::rrt::Algorithm;
use util::Pnt32;
use util::{self, info, dist, dist_pnt_u8, trees::Node, Pnt, Tree};
use sim::int_api::{Cmd::*, Req::*, Sim, SimApi};
use sim::colors as col;

const OBS_RADIUS: u32 = 2;
const EDGE_CLEAR: usize = 2;

/// Check for a clear line of sight between two points.
pub fn lsc(map: &Vec<u32>, a: Pnt32, b: Pnt32, k: usize) -> bool {
    let line4 = util::gen_line(a, b, k);
    for c in line4 {
        if c >= map.len() { continue; }
        if map[c] == col::DPURPLE {
            return false;
        }
    }

    return true;
}

/// Generate new nodes in the tree, checking that they lie within the known free space
pub fn gen_nodes(rrt: &mut Algorithm, map: &Vec<u32>, num: usize) {
    // print_time!("gen nodes");
    for _ in 0..num {
        rrt.gen_node(|x, d| {
            let circle5 = util::gen_circle(x.pnt32(), OBS_RADIUS);
            let circle7 = util::gen_circle(x.pnt32(), 7);

            for i in circle5 {
                if i >= map.len() { continue; }
                if map[i] == col::DPURPLE {
                    if d { info!("circle5"); }
                    return false;
                }
            }

            if d { info!("{:?}", &circle7); }

            if map.contains(&col::LYELLOW) {
                for i in circle7 {
                    if i >= map.len() { continue; }
                    if map[i] == col::LYELLOW || map[i] == col::YELLOW || map[i] == col::GREEN || map[i] == col::RED {
                        return true;
                    }
                }
            } else {
                return true;
            }

            if d { info!("not circle7"); }
            return false;
        });
    }
}

/// Generate new nodes in the tree, only checking that they are not within obstacles
pub fn gen_nodes_unchecked(rrt: &mut Algorithm, map: &Vec<u32>, num: usize) {
    for _ in 0..num {
        rrt.gen_node(|x, d| {
            let circle = util::gen_circle(x.pnt32(), OBS_RADIUS);
            for i in circle {
                if i >= map.len() { continue; }
                if map[i] == col::DPURPLE {
                    return false;
                }
            }


            return true;
        });
    }
}

/// Update the knowledge of the environment and remove bad nodes
pub fn update_knowledge(
    rrt: &mut Algorithm,
    sml: Arc<Sim>,
    map: &mut Vec<u32>,
) -> Result<()> {
    // Add newly discovered areas to the map.
    sml.cmd(Ping)?;
    *map = sml.req(Pixbuf)?;

    // Remove invalid nodes and their children
    let to_del: Vec<usize> = rrt.tree.par_iter().filter_map(|(idx, node)| {
        if *idx == 0 { return None; }

        let circle5 = util::gen_circle(node.pnt32(), OBS_RADIUS);
        for i in circle5 {
            if i >= map.len() { continue; }
            if map[i] == col::DPURPLE {
                return Some(*idx);
            }
        }

        let par = node.par.lock().unwrap().upgrade().unwrap();
        if !lsc(map, node.pnt32(), par.pnt32(), EDGE_CLEAR) {
            return Some(*idx);
        }

        return None;
    }).collect();

    for idx in to_del {
        rrt.del_subtree(idx)
    }

    Ok(())
}

pub fn update_frontiers(
    frontiers: &mut Vec<Arc<Node>>,
    tree: &Tree,
    map: &Vec<u32>,
    current: Arc<Node>
) {
    // print_time!("upd_front");
    // Inlines `gen_circle(node, 2)` + the color check to avoid allocating
    // two `Vec<usize>`s per tree node.
    const R: u32 = 2;
    const R_SQ_F: f32 = (R * R) as f32;

    frontiers.clear();
    'a: for (_, node) in tree {
        let p = node.pnt32();
        let x_start = p.0.saturating_sub(R);
        let y_start = p.1.saturating_sub(R);
        let x_end = (p.0 + R).min(util::W32);
        let y_end = (p.1 + R).min(util::H32);
        let cx = p.0 as f32;
        let cy = p.1 as f32;

        for y in y_start..y_end {
            for x in x_start..x_end {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy > R_SQ_F { continue; }

                let j = util::coords(x, y) as usize;
                if map[j] == col::LYELLOW || map[j] == col::YELLOW {
                    continue 'a;
                }
            }
        }

        frontiers.push(node.clone());
    }

    frontiers.push(current.clone());
}

pub fn goto(
    dest: &Arc<Node>,
    current: &mut Arc<Node>,
    sml: Arc<Sim>,
    map: &mut Vec<u32>,
    rrt: &mut Algorithm,
) -> Result<bool> {
    let vb = rrt.verbose;
    // Determine how to reach the destination node
    let dy = dest.y - current.y;
    let dx = dest.x - current.x;

    let theta = dy.atan2(dx);
    let r = dist(dest.pnt(), current.pnt());
    let start_pos = sml.req::<f32>(Pos)?;
    if vb {
        eprintln!(
            "goto: r={:.2} dest_idx={} cur_idx={} start=({:.1},{:.1}) theta={:.1}°",
            r, dest.idx, current.idx, start_pos[0], start_pos[1], theta.to_degrees()
        );
    }

    // Move the robot to the destination node
    sml.cmd(Rot(theta.to_degrees()))?;

    let mut moves_made = 0usize;
    let mut success = true;
    for i in 0..r.round() as usize {
        // We try to step in the specified direction, but if the robot doesn't move, try
        // other angles since it is probably stuck on an obstacle.
        let mut moved = false;
        let mut used_offset = 0.0f32;
        for of in (0..360).step_by(15) {
            let off = of as f32;
            let before = sml.req::<f32>(Pos)?;
            if off != 0.0 { sml.cmd(Rot((theta.to_degrees() + off).rem_euclid(360.0)))?; }
            sml.cmd(StepBy(3.0))?;
            let after = sml.req::<f32>(Pos)?;
            if (before[0] - after[0]).abs() > 1.5 || (before[1] - after[1]).abs() > 1.5 {
                moved = true;
                moves_made += 1;
                used_offset = off;
                break;
            }
        }
        // If nothing at any wiggle angle moved, the position is fully wedged
        // — no point continuing this goto.
        if !moved {
            success = false;
            if vb { eprintln!("goto: fully wedged at step i={}, aborting", i); }
            break;
        }
        // Re-point toward the true target for the next iteration so wiggle
        // detours don't send us off-course indefinitely.
        sml.cmd(Rot(theta.to_degrees()))?;

        // Re-check LSC either on the periodic schedule or whenever a wiggle was needed.
        let wiggled = used_offset != 0.0;
        let periodic = i % (sim::VIEW_DIST as usize - 4) == (sim::VIEW_DIST as usize - 5);
        if wiggled || periodic {
            sml.cmd(Ping)?;
            *map = sml.req(Pixbuf)?;

            // Draw the LSC line from the robot's ACTUAL position, not from
            // `current` (the tree node we started from -- it's stale after
            // any amount of movement).
            let pos = sml.req::<f32>(Pos)?;
            let cur_pnt = (pos[0] as u32, pos[1] as u32);
            if !lsc(&*map, cur_pnt, dest.pnt32(), 2) {
                success = false;
                if vb { eprintln!("goto: LSC re-check failed at i={} pos=({:.1},{:.1}) wiggled={}", i, pos[0], pos[1], wiggled); }
                break;
            }
        }
    }
    
    if vb { eprintln!("goto: step loop done, success={} moves={}/{}", success, moves_made, r.round() as usize); }

    sml.cmd(AddScore)?;

    // Update the current node
    let pos = sml.req(Pos)?;
    if vb { eprintln!("goto: end pos=({:.1},{:.1})", pos[0], pos[1]); }

    // If not at the expected location, update the current position
    if dist((pos[0], pos[1]), dest.pnt()) > 1.5 {
        let par = current.clone();
        *current = Arc::new(Node::new((pos[0], pos[1]), par, rrt.idx));   // TODO: FIX THIS INDEX
        rrt.idx += 1;
    } else {
        *current = dest.clone();
    }

    Ok(success)
}