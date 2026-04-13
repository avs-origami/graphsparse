use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use rand::prelude::*;
use rand::distr::Uniform;
use rand::rngs::ReseedingRng;
use rand_chacha::ChaCha20Core;

use util::*;
use util::trees::*;

/// The RRT algorithm.
#[derive(Clone, Debug)]
pub struct Algorithm {
    pub start: Arc<Node>,
    pub step: f32,
    pub tree: Tree,
    pub idx: usize,
    pub dim: (f32, f32),
    pub rng: ReseedingRng<ChaCha20Core, StdRng>,
    pub dist: (Uniform<f32>, Uniform<f32>),
    pub seed: Option<u64>
}

impl Algorithm {
    /// Constructor.
    pub fn new(start: Pnt, step: f32, seed: Option<u64>) -> Result<Self> {
        match seed {
            Some(x) => Ok(Algorithm {
                start: Arc::new(Node::start(start)),
                step,
                tree: HashMap::from([(0, Arc::new(Node::start(start)))]),
                idx: 1,
                dim: (100.0, 100.0),
                rng: ReseedingRng::new(0, StdRng::seed_from_u64(x))?,
                dist: (Uniform::new(0.0, 100.0)?, Uniform::new(0.0, 100.0)?),
                seed,
            }),
            None => Ok(Algorithm {
                start: Arc::new(Node::start(start)),
                step,
                tree: HashMap::from([(0, Arc::new(Node::start(start)))]),
                idx: 1,
                dim: (100.0, 100.0),
                rng: ReseedingRng::new(512, StdRng::from_os_rng())?,
                dist: (Uniform::new(0.0, 100.0)?, Uniform::new(0.0, 100.0)?),
                seed,
            }),
        }
    }

    /// Set the dimensions of the environment.
    pub fn set_dim(&mut self, dim: (f32, f32)) -> Result<()> {
        self.dim = dim;
        self.dist = (Uniform::new(0.0, dim.0)?, Uniform::new(0.0, dim.1)?);
        Ok(())
    }

    /// Generate a node.
    pub fn gen_node(&mut self, mut discard: impl FnMut(&Node, bool) -> bool) {
        let mut attempts = 0;
        let mut debug = false;
        loop {
            // This aims to fix an issue with the RNG "converging" occasionally
            attempts += 1;
            if attempts % 500 == 0 {
                info!("reseeding RNG");
                match self.seed {
                    Some(x) => self.rng = ReseedingRng::new(0, StdRng::seed_from_u64(x+1)).unwrap(),
                    None => self.rng.reseed().unwrap(),
                }

                #[allow(deprecated)]
                std::thread::sleep_ms(23);
                debug = true;
            }

            // Randomly generate a point
            // let point: (f32, f32) = (self.rng.gen_range(0.0..self.dim.0), self.rng.gen_range(0.0..self.dim.1));
            let point: (f32, f32) = (self.dist.0.sample(&mut self.rng), self.dist.1.sample(&mut self.rng));
            if debug { dbg!(&point); }
            // Find the closest node to the point
            let parent = closest(&self.tree, point, None);
            // Move one unit away in the direction of the random point, and make a new node
            let dy = point.1 - parent.y;
            let dx = point.0 - parent.x;
            let mag = dist(point, parent.pnt()) / self.step;
            let node = Node::new((parent.x + dx/mag, parent.y + dy/mag), parent.clone(), self.idx);

            if debug {
                // info!("{}", discard(&node, debug));
                info!("{}", dist(node.pnt(), parent.pnt()));
                info!("{:?}", node.pnt());
                info!("{}", self.tree.len());
                info!("{:?}", parent.pnt());
            }

            // Can be used to discard points if needed
            // dist(node.pnt(), closest(&self.tree, node.pnt(), None).pnt()) >= self.step
            if 0.0 < node.x && node.x < self.dim.0
                && 0.0 < node.y && node.y < self.dim.1
                && discard(&node, debug)
            {
                self.idx += 1;
                let rnode = Arc::new(node);
                *parent.num_child.lock().unwrap() += 1;
                parent.child.lock().unwrap().insert(rnode.idx, Arc::downgrade(&rnode));
                self.tree.insert(rnode.idx, rnode);
                break;
            }
        }
    }

    /// Remove a node from tree, updating any relevant information about other nodes
    /// that might change as a result of this change.
    pub fn del(&mut self, idx: usize) {
        if idx == 0 { return; }
        // Remove the node at idx from the tree
        let rem = self.tree.remove(&idx).unwrap();

        // Lock the parent node, and decrement num_child to reflect the removal
        // of the desired node.
        let par = rem.par.lock().unwrap().upgrade().unwrap().clone();
        *par.num_child.lock().unwrap() -= 1;
        
        for ch in rem.child.lock().unwrap().iter() {
            if let Some(x) = ch.1.upgrade() {
                // This child still exists in the tree, so give it a new parent
                // and add it to the parent's list of children.
                *x.par.lock().unwrap() = Arc::downgrade(&par.clone());
                par.child.lock().unwrap().insert(x.idx, Arc::downgrade(&x.clone()));

                // The parent now has one more child.
                *par.num_child.lock().unwrap() += 1;
            }
        }

        // Drop the removed node, and remove its pointer from the parent's list
        // of child nodes.
        drop(rem);
        let it = par.child.lock().unwrap().clone();
        for ch in it.iter() {
            if let None = ch.1.upgrade() {
                par.child.lock().unwrap().remove(ch.0);
            } else if !self.tree.contains_key(ch.0) {
                par.child.lock().unwrap().remove(ch.0);
            }
        }
    }

    pub fn len(&self) -> usize {
        return self.tree.len();
    }
}