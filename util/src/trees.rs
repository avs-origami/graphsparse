use std::fmt;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use serde::{Serialize, Deserialize};

use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Representation of a node in the search tree.
pub struct Node {
    /// X-coordinate
    pub x: f32,
    /// Y-coordinate
    pub y: f32,
    /// Index
    pub idx: usize,
    /// Parent
    pub par: Arc<Mutex<Weak<Node>>>,
    /// Children
    pub child: Arc<Mutex<HashMap<usize, Weak<Node>>>>,
    /// Number of children
    pub num_child: Arc<Mutex<usize>>,
}

impl Node {
    /// Constructor.
    pub fn new(point: Pnt, parent: Arc<Node>, idx: usize) -> Self {
        Node {
            x: point.0,
            y: point.1,
            idx,
            par: Arc::new(Mutex::new(Arc::downgrade(&parent))),
            child: Arc::new(Mutex::new(HashMap::new())),
            num_child: Arc::new(Mutex::new(0)),
        }
    }

    /// Create the starting node.
    pub fn start(start: Pnt) -> Self {
        Node {
            x: start.0,
            y: start.1,
            idx: 0,
            par: Arc::new(Mutex::new(Weak::new())),
            child: Arc::new(Mutex::new(HashMap::new())),
            num_child: Arc::new(Mutex::new(0)),
        }
    }

    /// The node as a point (self.x, self.y).
    pub fn pnt(&self) -> Pnt {
        return (self.x, self.y);
    }

    /// Same as self.pnt(), but returns a Pnt32
    pub fn pnt32(&self) -> Pnt32 {
        return (self.x.round() as u32, self.y.round() as u32);
    }

    #[allow(dead_code)]
    /// The node as a point (self.x, self.y) with coords rounded.
    pub fn pnt_rnd(&self) -> Pnt {
        return (self.x.round(), self.y.round());
    }

    /// The node as a point (self.x, self.y) with coords truncated.
    pub fn pnt_trunc(&self) -> Pnt {
        return (self.x.trunc(), self.y.trunc());
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        if self.x == other.x && self.y == other.y {
            return true;
        }

        return false;
    }

    fn ne(&self, other: &Self) -> bool {
        if !self.eq(other) {
            return true;
        }

        return false;
    }
}

impl Eq for Node {}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ x: {}, y: {}, idx: {} }}", self.x, self.y, self.idx)
    }
}