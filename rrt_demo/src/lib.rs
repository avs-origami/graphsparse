use std::sync::Arc;
use std::collections::VecDeque;
use std::time::Instant;

use anyhow::Result;
use measure_time::print_time;
use pysparse::opts::GlobalOpts;
use rrt_lib::rrt::Algorithm;
use sim::demo_api::{Sim, SimApi, Cmd::*, Req::*};
use pysparse::py;
use util::trees::Node;
use util::{dist, Pnt, Tree};
use sim::colors as col;
use rayon::prelude::*;

pub mod com;
use com::*;

pub const START: Pnt = (0.0, 0.0);
pub const STEP: f32 = 1.0;
pub const TIMEOUT: u64 = 60;
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
    pub dumps: f32,
    pub num_visited: usize,
    pub opts: GlobalOpts
}

impl RrtInc {
    pub fn new(start: Option<Pnt>, opts: GlobalOpts) -> Result<(RrtInc, std::thread::JoinHandle<()>)> {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(15).build_global();
        
        let rrt = Algorithm::new(start.unwrap_or(START), STEP, None)?;
        let current = rrt.start.clone();

        let sml = Arc::new(Sim::new(
            sim::args::Args {
                child_alg: None,
                child_alg_args: None,
                clutter: opts.clutter,
                scale: None,
                seed: None,
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
        Ok(st)
    }

    pub fn reset(&mut self, start: Option<Pnt>, handle: &mut std::thread::JoinHandle<()>) -> Result<()> {
        let _ = self.sml.cmd(Done);
        
        let start = if let Some(start) = start {
            start
        } else {
            self.rrt.start.pnt()
        };

        self.rrt = Algorithm::new(start, self.rrt.step, self.rrt.seed)?;
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
        self.num_visited = 0;
        self.greg = 0;
        *handle = self.init()?;

        Ok(())
    }

    pub fn step(&mut self) -> Result<(Exit, f32)> {
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

        // Using this information, build the tree in the known self.free space
        if self.counter == 1 {
            gen_nodes_unchecked(&mut self.rrt, &self.map, 50 / STEP as usize);
        } else {
            gen_nodes(&mut self.rrt, &self.map, 50 / STEP as usize);
        }

        // update_frontiers(&mut self.frontiers, &self.rrt.tree, &self.map, self.current.clone());
        // Update the obstacle space and self.free space
        update_knowledge(
            &mut self.rrt,
            self.sml.clone(),
            &mut self.map
        )?;

        // Attempt to move to a node in the tree
        self.regen_attempts = 0;
        'i: loop {
            // print_time!("'i loop");

            // Add nodes outside the self.free space to self.frontiers
            update_frontiers(&mut self.frontiers, &self.rrt.tree, &self.map, self.current.clone());

            // Make sure we haven't exceeded the timeout
            if now.elapsed().as_secs() > TIMEOUT || self.total_time > TIMEOUT { self.greg += self.regen_attempts; return Ok((Exit::Timeout, gain)); }

            // Determine the destination node
            let mut dest_maybe = None;

            if self.regen_attempts == 0 || self.regen_attempts % 5 != 0 {
                // Try to find a direct line of sight to a frontier node
                // print_time!("'a for before 10 attempts");
                'a: for (_, i) in self.frontiers.iter().rev().enumerate() {
                    // if n > 3000 { break 'a; }

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
            } else if self.frontiers.len() > 1 {
                // If there isn't a line of sight to a valid frontier node, pathfind to one
                if let None = dest_maybe {
                    if !self.rrt.tree.contains_key(&self.current.idx) {
                        let mut nearest = self.rrt.start.clone();
                        for (_, node) in &self.rrt.tree {
                            if dist(node.pnt(), self.current.pnt()) < dist(nearest.pnt(), self.current.pnt())
                               && lsc(&self.map, self.current.pnt32(), node.pnt32(), 2)
                            {
                                nearest = node.clone();
                            }
                        }
                    }

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

                    let far = self.frontiers.par_iter()
                        .filter(|&i| lsc(&self.map, self.current.pnt32(), i.pnt32(), 2))
                        .reduce_with(|a, b| {
                            if dist(self.current.pnt(), b.pnt()) > dist(self.current.pnt(), a.pnt()) {
                                b
                            } else {
                                a
                            }
                        }).unwrap_or(&self.current).clone();

                    // Path from self.current to root
                    let mut path_start: VecDeque<Arc<Node>> = [self.current.clone()].into();
                    // Path from destination to root
                    let mut path_end: VecDeque<Arc<Node>> = [far.clone()].into();
                    let mut path_found = false;
                    let mut last = self.current.clone();

                    // Find the path from the self.current node to the root
                    while ! path_found {
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
                            None => path_found = true,
                        }
                    }

                    path_found = false;
                    last = far.clone();

                    // Find the path from the destination to the root
                    while ! path_found {
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
                            None => path_found = true,
                        }
                    }

                    // Determine the length of the shortest path (to avoid index errors)
                    let len = if path_start.len() < path_end.len() {
                        path_start.len()
                    } else {
                        path_end.len()
                    };

                    // Determine the index of the last node common to both paths
                    let mut com = 0;
                    for i in 0..len {
                        if path_start[i] != path_end[i] && i != 0 {
                            com = i - 1;
                            break;
                        }
                    }

                    // Truncate both paths to only nodes after `com` and join them
                    let mut path: Vec<Arc<Node>> = path_start.make_contiguous()[com..].to_vec().into_iter().rev().collect();
                    path.append(&mut path_end.make_contiguous()[com..].to_vec());

                    //self.sml.cmd(DrawTree2(&path));
                    //dbg!(&path);

                    // Move to each node in path
                    let mut exit = false;
                    // print_time!("'b for move along path");
                    'b: for dest in path {
                        let last = self.current.clone();
                        if !goto(&dest, &mut self.current, self.sml.clone(), &mut self.map, &mut self.rrt)? {
                            exit = true;
                        }

                        self.num_visited += 1;

                        self.num_visited += 1;

                        // Add the self.current location to the list of self.visited nodes
                        self.visited.push(self.current.clone());

                        self.counter += 1;
                        self.total_dist += dist(last.pnt(), self.current.pnt());

                        gain = self.coverage;
                        self.coverage = self.sml.req::<f32>(Coverage)?[0];
                        gain = self.coverage - gain;
                        if self.coverage >= COV_MAX { self.greg += self.regen_attempts; return Ok((Exit::Finish, gain)); }

                        if now.elapsed().as_secs() > TIMEOUT || self.total_time > TIMEOUT { self.greg += self.regen_attempts; return Ok((Exit::Timeout, gain)); }
                        // if *ctrlc_exit.lock().unwrap() { break 'o; }

                        if self.sml.req::<f32>(Coverage)?[0] >= self.dumps * 5.0 {
                            self.sml.cmd(StatsTime(now.elapsed().as_secs_f32()))?;
                            self.sml.cmd(StatsDist(self.total_dist))?;
                            self.sml.cmd(Stats(self.dumps * 5.0))?;
                            self.dumps += 1.0;
                        }

                        if exit { break 'b; }
                    }

                    break 'i;
                }
            }

            // If a suitable destination was found, unwrap it
            let dest = if let Some(n) = dest_maybe {
                n
            } else {
                // Otherwise, make some new nodes and try again
                self.regen_attempts += 1;
                gen_nodes(&mut self.rrt, &self.map, 25 / STEP as usize);
                continue 'i;
            };

            let last = self.current.clone();
            
            {
                // print_time!("goto");
                goto(&dest, &mut self.current, self.sml.clone(), &mut self.map, &mut self.rrt)?;
                self.num_visited += 1;
            }

            // Add the self.current location to the list of self.visited nodes
            self.visited.push(self.current.clone());

            self.counter += 1;
            self.total_dist += dist(last.pnt(), self.current.pnt());

            gain = self.coverage;
            self.coverage = self.sml.req::<f32>(Coverage)?[0];
            gain = self.coverage - gain;
            if self.coverage >= COV_MAX { self.greg += self.regen_attempts; return Ok((Exit::Finish, gain)); }
            if now.elapsed().as_secs() > TIMEOUT || self.total_time > TIMEOUT { self.greg += self.regen_attempts; return Ok((Exit::Timeout, gain)); }
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
        self.greg += self.regen_attempts;
        Ok((Exit::Ok, gain))
    }
}