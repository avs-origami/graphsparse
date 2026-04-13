pub mod rrt;

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{seq::IteratorRandom, thread_rng, Rng};

    #[test]
    /// Ensure that when the tree is grown, there are no orphans and all
    /// node properties are consistent.
    fn grow_tree() {
        let mut alg = rrt::Algorithm::new((0.0, 0.0), 1.0, None).unwrap();
        let mut rng = thread_rng();
        for _ in 0..1000 {
            alg.gen_node(|_| rng.gen_bool(0.5));
        }
        
        for (_, node) in &alg.tree {
            assert_eq!(node.child.lock().unwrap().len(), *node.num_child.lock().unwrap());

            if node.idx == 0 {
                assert!(node.par.lock().unwrap().upgrade().is_none());
            } else {
                assert!(node.par.lock().unwrap().upgrade().is_some());
            }
        }
    }

    #[test]
    /// Ensure that upon removing nodes from the tree, there are no orphans and
    /// all node properties are consistent.
    fn del_node() {
        let mut alg = rrt::Algorithm::new((0.0, 0.0), 1.0, None).unwrap();
        for _ in 0..1000 {
            alg.gen_node(|_| true);
        }

        for _ in 0..500 {
            let rand_idx = alg.tree.keys().choose(&mut alg.rng).unwrap();
            if *rand_idx != 0 { alg.del(*rand_idx); }
        }

        for _ in 0..1000 {
            alg.gen_node(|_| true);
        }

        for _ in 0..500 {
            let rand_idx = alg.tree.keys().choose(&mut alg.rng).unwrap();
            if *rand_idx != 0 { alg.del(*rand_idx); }
        }

        for (_, node) in &alg.tree {
            assert_eq!(node.child.lock().unwrap().len(), *node.num_child.lock().unwrap());

            for ch in &*node.child.lock().unwrap() {
                assert!(ch.1.upgrade().is_some());
            }

            if node.idx == 0 {
                assert!(node.par.lock().unwrap().upgrade().is_none());
            } else {
                assert!(node.par.lock().unwrap().upgrade().is_some());
            }
        }
    }
}