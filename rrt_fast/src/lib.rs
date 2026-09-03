use std::sync::Arc;
use std::collections::VecDeque;
use std::time::Instant;

use anyhow::Result;
use measure_time::print_time;
use pysparse::opts::GlobalOpts;
use rrt_lib::rrt::Algorithm;
use sim::int_api::{Sim, SimApi, Cmd::*, Req::*};
use pysparse::py;
use util::trees::Node;
use util::{dist, Pnt, Tree};
use sim::colors as col;
use rayon::prelude::*;

pub mod com;
use com::*;

pub const START: Pnt = (0.0, 0.0);
pub const STEP: f32 = 1.0;
pub const TIMEOUT: u64 = 30;
pub const COV_MAX: f32 = 95.0;

pub enum Exit {
    Timeout,
    Ok,
    Finish,
}

pub struct RrtInc {
    pub sml: Arc<Sim>,
    pub rrt: Algorithm,
    pub current: Arc<Node>,
    pub map: Vec<u32>,
    pub visited: Vec<Arc<Node>>,
    pub frontiers: Vec<Arc<Node>>,
    pub clearance: f32,
    pub total_dist: f32,
    pub counter: usize,
    pub regen_attempts: usize,
    pub greg: usize,
    pub total_time: u64,
    pub coverage: f32,
    pub initial_coverage: f32,
    pub dumps: f32,
    pub num_visited: usize,
    pub opts: GlobalOpts
}

impl RrtInc {
    pub fn new(start: Option<Pnt>, opts: GlobalOpts) -> Result<(RrtInc, std::thread::JoinHandle<()>)> {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(12).build_global();

        let rrt = Algorithm::new(start.unwrap_or(START), STEP, opts.seed, opts.verbose)?;
        let current = rrt.start.clone();

        let sml = Arc::new(Sim::new(
            sim::args::Args {
                child_alg: None,
                child_alg_args: None,
                clutter: opts.clutter,
                scale: None,
                seed: opts.seed,
                phasing: false
            },
            start.unwrap_or(START),
            !opts.render,
        )?);

        let mut me = RrtInc {
            sml,
            rrt,
            current,
            map: vec![],
            visited: vec![],
            frontiers: vec![],
            clearance: 4.0,
            total_dist: 0.0,
            counter: 1,
            regen_attempts: 0,
            greg: 0,
            total_time: 0,
            coverage: 0.0,
            initial_coverage: 0.0,
            dumps: 0.0,
            num_visited: 0,
            opts,
        };

        let st = me.init()?;

        Ok((me, st))
    }

    pub fn init(&mut self) -> Result<std::thread::JoinHandle<()>> {
        let st = self.sml.clone().run();
        let dim = self.sml.req(Scale)?;
        self.rrt.set_dim((dim[0], dim[1]))?;

        // Reveal the area immediately around the robot before doing anything else.
        self.sml.cmd(RotBy(360.0))?;
        self.initial_coverage = self.sml.req::<f32>(Coverage)?[0];
        Ok(st)
    }

    pub fn reset(&mut self, start: Option<Pnt>, handle: &mut std::thread::JoinHandle<()>) -> Result<()> {
        let _ = self.sml.cmd(Done);
        
        let start = if let Some(start) = start {
            start
        } else {
            self.rrt.start.pnt()
        };

        self.rrt = Algorithm::new(start, self.rrt.step, self.rrt.seed, self.opts.verbose)?;
        self.current = self.rrt.start.clone();
        self.map = vec![];
        self.visited = vec![];
        self.frontiers = vec![];
        self.total_dist = 0.0;
        self.sml = Arc::new(Sim::new(
            self.sml.args.clone(),
            start,
            !self.opts.render,
        )?);

        self.counter = 1;
        self.total_time = 0;
        self.coverage = 0.0;
        self.initial_coverage = 0.0;
        self.num_visited = 0;
        self.greg = 0;
        *handle = self.init()?;

        Ok(())
    }

    pub fn step(&mut self) -> Result<(Exit, f32)> {
        let vb = self.opts.verbose;
        if vb { eprintln!("step:enter"); }
        // print_time!("rithm.step");
        let mut gain = 0.0;
        // print_time!("step");
        let now = Instant::now();
        // Update the knowledge of the environment
        update_knowledge(
            &mut self.rrt,
            self.sml.clone(),
            &mut self.map
        )?;
        if vb { eprintln!("step:uk1"); }

        // Using this information, build the tree in the known self.free space
        if self.counter == 1 {
            gen_nodes_unchecked(&mut self.rrt, &self.map, 50 / STEP as usize);
        } else {
            gen_nodes(&mut self.rrt, &self.map, 50 / STEP as usize);
        }
        if vb { eprintln!("step:gen"); }

        // update_frontiers(&mut self.frontiers, &self.rrt.tree, &self.map, self.current.clone());
        // Update the obstacle space and self.free space
        update_knowledge(
            &mut self.rrt,
            self.sml.clone(),
            &mut self.map
        )?;
        if vb { eprintln!("step:uk2"); }

        // Re-anchor current if update_knowledge or a prior goto orphaned it.
        // Keeps current at the robot's actual position but re-registers it in
        // the tree under the nearest LOS-clear anchor. We must always maintain
        // self.current as a member of the tree for pathfinding logic to work.
        //
        // Cheap distance pre-filter (ANCHOR_R) skips the `lsc` call
        // on nodes that are obviously too far to improve runtime.
        if !self.rrt.tree.contains_key(&self.current.idx) {
            const ANCHOR_R: f32 = 30.0;
            let anchor = self.rrt.tree.values()
                .filter(|n| dist(n.pnt(), self.current.pnt()) < ANCHOR_R)
                .filter(|n| lsc(&self.map, self.current.pnt32(), n.pnt32(), 2))
                .min_by(|a, b| {
                    dist(a.pnt(), self.current.pnt())
                        .partial_cmp(&dist(b.pnt(), self.current.pnt()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned();

            if let Some(anchor) = anchor {
                assert!(
                    anchor.idx != self.current.idx,
                    "re-anchor would create self-loop on idx={}", self.current.idx
                );
                
                // If current was not previously part of the RRT, it cannot have children,
                // so just make sure there aren't any stale child references hanging around.
                self.current.child.lock().unwrap().clear();
                *self.current.num_child.lock().unwrap() = 0;

                *self.current.par.lock().unwrap() = Arc::downgrade(&anchor);
                *anchor.num_child.lock().unwrap() += 1;
                anchor.child.lock().unwrap()
                    .insert(self.current.idx, Arc::downgrade(&self.current));
                self.rrt.tree.insert(self.current.idx, self.current.clone());
            }
            // else: no tree node has LOS to the robot. It will get re-parented
            // in a future iteration once the tree grows some.
        }
        if vb { eprintln!("step:reanchor"); }

        // Attempt to move to a node in the tree
        self.regen_attempts = 0;
        'i: loop {
            if vb { eprintln!("step:i (regen={})", self.regen_attempts); }
            // print_time!("'i loop");

            // Add nodes outside the self.free space to self.frontiers
            update_frontiers(&mut self.frontiers, &self.rrt.tree, &self.map, self.current.clone());
            if vb { eprintln!("step:i uf done nf={}", self.frontiers.len()); }

            // Make sure we haven't exceeded the timeout
            if now.elapsed().as_secs() > TIMEOUT || self.total_time > TIMEOUT { self.greg += self.regen_attempts + 1; return Ok((Exit::Timeout, gain)); }

            // Create a set of nodes that are previously visited to avoid obviously
            // useless moves. A previously visited node may be traveled through
            // as part of pathfinding, but will never be selected as a destination.
            let visited_idx: std::collections::HashSet<usize> = self.visited.iter().map(|n| n.idx).collect();

            // Determine the destination node
            let mut dest_maybe = None;

            if self.regen_attempts == 0 || self.regen_attempts % 5 != 0 {
                // Try to find a direct line of sight to a frontier node
                // print_time!("'a for before 10 attempts");
                if vb { eprintln!("step:i direct scanning"); }
                'a: for (_, i) in self.frontiers.iter().rev().enumerate() {
                    // if n > 3000 { break 'a; }

                    // Skip nodes we've already visited.
                    if visited_idx.contains(&i.idx) {
                        continue 'a;
                    }

                    // Determine if there is a clear line of sight to the destination
                    if !lsc(&self.map, self.current.pnt32(), i.pnt32(), 2) {
                        continue 'a;
                    }

                    if *i == self.current.clone() {
                        continue 'a;
                    }

                    // If all conditions met, set dest_maybe and break
                    dest_maybe = Some(i);
                    break 'a;
                }
                if vb { eprintln!("step:i direct done dest_maybe={}", dest_maybe.is_some()); }
            } else if self.frontiers.len() > 1 {
                // If there isn't a line of sight to a valid frontier node, pathfind to one
                if let None = dest_maybe {
                    if vb { eprintln!("tp:enter frontiers={}", self.frontiers.len()); }

                    // // Determine the farthest frontier node
                    // let mut far = self.current.clone();

                    // // print_time!("'a for farthest frontier");
                    // 'a: for (_, i) in self.frontiers.iter().enumerate() {
                    //     if dist(
                    //         self.current.pnt(), i.pnt()
                    //     ) > dist(
                    //         self.current.pnt(), far.pnt()
                    //     ) {
                    //         // Ensure that the node is valid

                    //         if !lsc(&self.map, self.current.pnt32(), i.pnt32(), 2) {
                    //             continue 'a;
                    //         }

                    //         // If it's farther and it's valid, set it as the destination
                    //         far = i.clone();
                    //     }
                    // }

                    // Pick the frontier farthest from the current location, since this encourages
                    // the robot to explore other parts of the tree that may be more promising
                    // locations for information gain.
                    let far = self.frontiers.par_iter()
                        .filter(|&i| i.idx != self.current.idx && !visited_idx.contains(&i.idx))
                        .reduce_with(|a, b| {
                            if dist(self.current.pnt(), b.pnt()) > dist(self.current.pnt(), a.pnt()) {
                                b
                            } else {
                                a
                            }
                        }).unwrap_or(&self.current).clone();
                    if vb { eprintln!("tp:far idx={} cur_in_tree={}", far.idx, self.rrt.tree.contains_key(&self.current.idx)); }

                    // Path from self.current to root
                    let mut path_start: VecDeque<Arc<Node>> = [self.current.clone()].into();
                    // Path from destination to root
                    let mut path_end: VecDeque<Arc<Node>> = [far.clone()].into();
                    let mut path_found = false;
                    let mut last = self.current.clone();

                    // Find the path from the self.current node to the root.
                    // WALK_CAP prevents infinite loops in case a cycle is
                    // introduced to the RRT by accident.
                    const WALK_CAP: usize = 100_000_000_000;
                    let mut walk_a = 0usize;
                    let mut cycle_a = false;
                    while ! path_found {
                        walk_a += 1;
                        if walk_a > WALK_CAP {
                            eprintln!(
                                "tp:walk_a CYCLE last_idx={} par_idx={:?}",
                                last.idx,
                                last.par.lock().unwrap().upgrade().map(|p| p.idx)
                            );
                            cycle_a = true;
                            break;
                        }
                        let next = last.par.lock().unwrap().upgrade();
                        match next {
                            Some(n) => {
                                if !lsc(
                                    &self.map,
                                    path_start[0].pnt32(),
                                    n.pnt32(),
                                    2
                                ) && lsc(
                                    &self.map,
                                    path_start[0].pnt32(),
                                    last.pnt32(),
                                    2
                                ) {
                                    path_start.push_front(last.clone());
                                }

                                last = n.clone();
                            },
                            None => {
                                // Since we don't push nodes to the path unless LOS is broken
                                // between the most recently pushed node and the next node, it
                                // is possible that we reach the root node after traveling down
                                // a chain of nodes without pushing any of them. Therefore, we
                                // must make sure that the terminal node of this half-path is
                                // included, otherwise we could end up with an LOS-break in our
                                // path when we expect that it is guaranteed to be traversable
                                // without stopping unexpectedly.
                                if path_start.front().map(|n| n.idx) != Some(last.idx) {
                                    path_start.push_front(last.clone());
                                }

                                path_found = true;
                            }
                        }
                    }
                    if vb { eprintln!("tp:walk_a done ({} hops, cycle={})", walk_a, cycle_a); }

                    path_found = false;
                    last = far.clone();

                    // Find the path from the destination to the root
                    let mut walk_b = 0usize;
                    let mut cycle_b = false;
                    while ! path_found {
                        walk_b += 1;
                        if walk_b > WALK_CAP {
                            eprintln!(
                                "tp:walk_b CYCLE last_idx={} par_idx={:?}",
                                last.idx,
                                last.par.lock().unwrap().upgrade().map(|p| p.idx)
                            );
                            cycle_b = true;
                            break;
                        }
                        let next = last.par.lock().unwrap().upgrade();
                        match next {
                            Some(n) => {
                                if !lsc(
                                    &self.map,
                                    path_end[0].pnt32(),
                                    n.pnt32(),
                                    2
                                ) && lsc(
                                    &self.map,
                                    path_end[0].pnt32(),
                                    last.pnt32(),
                                    2
                                ) {
                                    path_end.push_front(last.clone());
                                }

                                last = n.clone();
                            },
                            None => {
                                // Since we don't push nodes to the path unless LOS is broken
                                // between the most recently pushed node and the next node, it
                                // is possible that we reach the root node after traveling down
                                // a chain of nodes without pushing any of them. Therefore, we
                                // must make sure that the terminal node of this half-path is
                                // included, otherwise we could end up with an LOS-break in our
                                // path when we expect that it is guaranteed to be traversable
                                // without stopping unexpectedly.
                                if path_end.front().map(|n| n.idx) != Some(last.idx) {
                                    path_end.push_front(last.clone());
                                }
                                path_found = true;
                            }
                        }
                    }
                    if vb { eprintln!("tp:walk_b done ({} hops, cycle={})", walk_b, cycle_b); }

                    // If we found a cycle in the RRT, continue to the next attempt.
                    if cycle_a || cycle_b {
                        self.regen_attempts += 1;
                        gen_nodes(&mut self.rrt, &self.map, 25 / STEP as usize);
                        continue 'i;
                    }

                    // Determine the length of the shortest path (to avoid index errors)
                    let len = if path_start.len() < path_end.len() {
                        path_start.len()
                    } else {
                        path_end.len()
                    };

                    // Both paths are arranged from root -> [start/end]point, so we find
                    // the first index where the paths diverge and join the paths at the
                    // previous index (the furthest node from root shared by both paths,
                    // i.e. the least common ancestor of both paths or LCA).
                    let mut com = 0;
                    for i in 1..len {
                        if path_start[i] != path_end[i] {
                            break;
                        }

                        com = i;
                    }

                    // path_start[com..].rev() walks from current to the LCA.
                    // path_end[(com+1)..] walks from just past the LCA to far.
                    // We skip the LCA in path_end to avoid duplicating it.
                    let mut path: Vec<Arc<Node>> = path_start.make_contiguous()[com..].to_vec().into_iter().rev().collect();
                    let end_start = (com + 1).min(path_end.len());
                    path.append(&mut path_end.make_contiguous()[end_start..].to_vec());

                    //self.sml.cmd(DrawTree2(&path));
                    //dbg!(&path);
                    if vb { eprintln!("tp:path_len={}", path.len()); }

                    // Move to each node in path
                    let mut exit = false;
                    // print_time!("'b for move along path");
                    let mut bi = 0usize;
                    'b: for dest in path {
                        if vb { eprintln!("tp:b iter {} dest_idx={}", bi, dest.idx); }
                        bi += 1;
                        let last = self.current.clone();
                        if !goto(&dest, &mut self.current, self.sml.clone(), &mut self.map, &mut self.rrt)? {
                            exit = true;
                        }

                        self.num_visited += 1;

                        // self.num_visited += 1;

                        // Add the self.current location to the list of self.visited nodes
                        self.visited.push(self.current.clone());

                        self.counter += 1;
                        self.total_dist += dist(last.pnt(), self.current.pnt());

                        gain = self.coverage;
                        self.coverage = self.sml.req::<f32>(Coverage)?[0];
                        gain = self.coverage - gain;
                        if self.coverage >= COV_MAX { self.greg += self.regen_attempts + 1; return Ok((Exit::Finish, gain)); }

                        if now.elapsed().as_secs() > TIMEOUT || self.total_time > TIMEOUT { self.greg += self.regen_attempts + 1; return Ok((Exit::Timeout, gain)); }
                        // if *ctrlc_exit.lock().unwrap() { break 'o; }

                        if self.sml.req::<f32>(Coverage)?[0] >= self.dumps * 5.0 {
                            self.sml.cmd(StatsTime(now.elapsed().as_secs_f32()))?;
                            self.sml.cmd(StatsDist(self.total_dist))?;
                            self.sml.cmd(Stats(self.dumps * 5.0))?;
                            self.dumps += 1.0;
                        }

                        if exit { break 'b; }
                    }
                    if vb { eprintln!("tp:b done"); }

                    break 'i;
                }
            }

            // If a suitable destination was found, unwrap it
            let dest = if let Some(n) = dest_maybe {
                n
            } else {
                // Otherwise, make some new nodes and try again
                if vb { eprintln!("step:i fallthrough regen->{}", self.regen_attempts + 1); }
                self.regen_attempts += 1;
                if vb { eprintln!("step:i pre-gen_nodes tree={}", self.rrt.len()); }
                gen_nodes(&mut self.rrt, &self.map, 25 / STEP as usize);
                if vb { eprintln!("step:i post-gen_nodes tree={}", self.rrt.len()); }
                continue 'i;
            };

            let last = self.current.clone();

            if vb { eprintln!("step:i pre-goto dest_idx={} dest@{:?} cur@{:?}",
                dest.idx, dest.pnt(), self.current.pnt()); }
            {
                // print_time!("goto");
                goto(&dest, &mut self.current, self.sml.clone(), &mut self.map, &mut self.rrt)?;
                self.num_visited += 1;
            }
            if vb { eprintln!("step:i post-goto cur_idx={} cur@{:?}", self.current.idx, self.current.pnt()); }

            // Add the self.current location to the list of self.visited nodes
            self.visited.push(self.current.clone());

            self.counter += 1;
            self.total_dist += dist(last.pnt(), self.current.pnt());

            gain = self.coverage;
            if vb { eprintln!("step:i pre-req(Coverage)#1"); }
            self.coverage = self.sml.req::<f32>(Coverage)?[0];
            if vb { eprintln!("step:i post-req(Coverage)#1 cov={}", self.coverage); }
            gain = self.coverage - gain;
            if self.coverage >= COV_MAX { self.greg += self.regen_attempts + 1; return Ok((Exit::Finish, gain)); }
            if now.elapsed().as_secs() > TIMEOUT || self.total_time > TIMEOUT { self.greg += self.regen_attempts + 1; return Ok((Exit::Timeout, gain)); }
            // if *ctrlc_exit.lock().unwrap() { break 'o; }

            if self.sml.req::<f32>(Coverage)?[0] >= self.dumps * 5.0 {
                self.sml.cmd(StatsTime(now.elapsed().as_secs_f32()))?;
                self.sml.cmd(StatsDist(self.total_dist))?;
                self.sml.cmd(Stats(self.dumps * 5.0))?;
                self.dumps += 1.0;
            }

            // Moving to this node was successful
            break 'i;
        }

        self.total_time += now.elapsed().as_secs();
        self.greg += self.regen_attempts + 1;
        Ok((Exit::Ok, gain))
    }
}