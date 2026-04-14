use std::sync::{Arc, Mutex};

use anyhow::Result;

use rrt_lib::rrt::Algorithm;
use util::Pnt32;
use util::{self, info, dist, dist_pnt_u8, trees::Node, Pnt, Tree};
use sim::int_api::{Cmd::*, Req::*, Sim, SimApi};
use sim::colors as col;

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
            let circle5 = util::gen_circle(x.pnt32(), 5);
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
            let circle = util::gen_circle(x.pnt32(), 5);
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
    // print_time!("upd_know");
    let mut to_del = vec![];

    'd: for idx in rrt.tree.keys() {
        if *idx == 0 { continue; }

        let circle5 = util::gen_circle(rrt.tree[idx].pnt32(), 5);
        for i in circle5 {
            if i >= map.len() { continue; }
            if map[i] == col::DPURPLE {
                to_del.push(*idx);
                continue 'd;
            }
        }

        let line4 = util::gen_line(
            rrt.tree[idx].pnt32(),
            rrt.tree[idx].par.lock().unwrap().upgrade().unwrap().pnt32(),
            2
        );

        for i in line4 {
            if i >= map.len() { continue; }
            if map[i] == col::DPURPLE {
                to_del.push(*idx);
                continue 'd;
            }
        }
    }

    for idx in to_del {
        rrt.del(idx)
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
    frontiers.clear();
    'a: for (_, i) in tree {
        let circle = util::gen_circle(i.pnt32(), 2);
        for j in circle {
            if j >= map.len() { continue; }
            if map[j] == col::LYELLOW || map[j] == col::YELLOW {
                continue 'a;
            }
        }

        frontiers.push(i.clone());
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
    // Determine how to reach the destination node
    let dy = dest.y - current.y;
    let dx = dest.x - current.x;

    let theta = dy.atan2(dx);
    let r = dist(dest.pnt(), current.pnt());

    // Move the robot to the destination node
    sml.cmd(Rot(theta.to_degrees()))?;

    let mut success = true;
    for i in 0..r.round() as usize {
        sml.cmd(Step)?;

        if i % (sim::VIEW_DIST as usize - 4) == (sim::VIEW_DIST as usize - 5) {
            // Update the obstacle space
            sml.cmd(Ping)?;
            *map = sml.req(Pixbuf)?;

            // If line of sight is no longer satisfied, break
            if !lsc(&*map, current.pnt32(), dest.pnt32(), 2) {
                success = false;
                break;
            }
        }
    }

    sml.cmd(AddScore)?;

    // Update the current node
    let pos = sml.req(Pos)?;

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