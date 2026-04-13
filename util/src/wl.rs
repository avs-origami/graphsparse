//! This is a helper module to generate the Weisfeiler-Lehman rooted subgraphs
//! for use by the graph2vec encoder module.

use std::{collections::HashMap, sync::Arc};

use crate::trees::Node;

#[derive(Clone, Copy, Debug)]
pub struct WLabel {
    pub deg: usize,
    pub parent: usize,
    pub child: usize,
    pub par_dist_rnd: usize,
}

impl From<WLabel> for String {
    fn from(value: WLabel) -> Self {
        format!(
            "{}.{}.{}.{}",
            value.deg,
            value.parent,
            value.child,
            value.par_dist_rnd
        )
    }
}

#[derive(Clone, Debug)]
pub struct Subgraph<T> {
    pub v: Vec<T>,
}

impl<T> Subgraph<T> {
    pub fn empty() -> Self {
        Self {
            v: vec![],
        }
    }

    pub fn push(&mut self, val: T) {
        self.v.push(val)
    }
}

impl<T: Copy> Into<String> for Subgraph<T> where String: From<T> {
    fn into(self) -> String {
        let mut ret = String::new();
        for w in &self.v {
            let l: String = (*w).into();
            ret.push_str(&l);
            ret.push_str("!")
        }

        return ret;
    }
}

pub fn get_wl(node: Arc<Node>, graph: &HashMap<usize, Arc<Node>>, depth: usize) -> Subgraph<WLabel> {
    let mut sg = Subgraph::empty();
    if depth == 0 {
        // Return label if depth 0: line 3 of GetWLSubgraph in graph2vec paper
        let mut deg = *node.num_child.lock().unwrap();
        let mut parent = 0;
        let child = deg;
        let mut par_dist_rnd = 0;

        if let Some(x) = node.par.lock().unwrap().upgrade() {
            parent = 1;
            deg += 1;
            par_dist_rnd = ((crate::dist(node.pnt(), x.pnt()) / 20.0).round() * 20.0) as usize;
        }

        sg.push(WLabel {
            deg,
            parent,
            child,
            par_dist_rnd,
        });
    } else {
        // Get the WL subgraph of the current node at shallower depth
        let wl_root = get_wl(node.clone(), graph, depth - 1);

        // Calculate set M: line 6 of GetWLSubgraph in graph2vec paper
        let mut wl_hood: Vec<WLabel> = vec![];

        let ch_lock = node.child.lock().unwrap();
        let ch_clone = ch_lock.clone();
        drop(ch_lock);
        for np in &ch_clone {
            wl_hood.extend(&get_wl(np.1.upgrade().unwrap(), graph, depth - 1).v);
        }

        let lock = node.par.lock().unwrap();
        let val = lock.upgrade().clone();
        drop(lock);
        if let Some(x) = val {
            wl_hood.extend(&get_wl(x.clone(), graph, depth - 1).v);
        }

        wl_hood.sort_by(|a, b| a.deg.cmp(&b.deg));

        // Extend sg: line 7 of GetWLSubgraph in graph2vec paper
        sg.v.extend(&wl_root.v);
        sg.v.extend(&wl_hood);
    }

    return sg;
}

pub fn get_str(node: Arc<Node>, graph: &HashMap<usize, Arc<Node>>, depth: usize) -> String {
    get_wl(node, graph, depth).into()
}