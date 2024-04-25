use crate::{tensor::NodeRefCount, NodeID};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    mem,
    sync::Arc,
};

#[derive(Default, Debug)]
pub struct GraphMemoryManagement {
    nodes: HashMap<NodeRefCount, Vec<NodeID>>,
    roots: HashSet<NodeID>,
    statuses: HashMap<NodeID, NodeMemoryStatus>,
}

impl Display for GraphMemoryManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "{} {} {}",
                self.nodes.len(),
                self.roots.len(),
                self.statuses.len()
            )
            .as_str(),
        )
    }
}

#[derive(Debug, Clone)]
enum NodeMemoryStatus {
    Useful,
    Unavailable,
    Unknown,
}

#[derive(Clone)]
enum Mode {
    Retain,
    Explore,
}

impl GraphMemoryManagement {
    /// Register a new node with its parent.
    pub fn register(&mut self, node: NodeRefCount, children: Vec<NodeID>) {
        let node_id = *node.as_ref();

        for parent_id in children.iter() {
            self.roots.remove(parent_id);
        }

        self.roots.insert(node_id);
        self.nodes.insert(node, children);
    }

    /// Free the node from the state.
    pub fn consume_node(&mut self, node_id: NodeID) {
        self.roots.remove(&node_id);
        self.nodes.remove(&node_id);
    }

    /// Free all nodes whose backward call has become impossible
    ///
    /// This function goes into three steps, which must happen for all roots
    /// before going into the next step. Then it deletes what can be safely deleted
    pub(crate) fn free_unusable(&mut self, mut on_free_graph: impl FnMut(&NodeID)) {
        let roots = self.roots.clone();
        let mut new_roots = HashSet::new();
        let mut deletables = Vec::new();

        // When consuming nodes with a backward pass, some other backward passes become
        // unavailable because some of their children have been consumed. They are
        // identified here.
        for root in roots.clone() {
            self.unavailable_propagation(root);
        }

        // Among the available nodes that remain, some may be useless if no
        // available node with a tensor reference exist in their ancestry.
        // But some may seem useless from some root but be useful in another one,
        // hence the need to iterate on all roots.
        for root in roots.clone() {
            self.useful_propagation(root, Mode::Explore);
        }

        // New roots are the roots of a useful sub-tree.
        // Deletables are everything not marked as useful.
        for root in roots.clone() {
            self.identify_roots_and_deletables(root, &mut new_roots, &mut deletables);
        }

        // Replace roots by the new ones and delete everything not useful anymore
        println!("{:?}", new_roots);
        println!("{:?}", deletables);
        mem::swap(&mut self.roots, &mut new_roots);
        self.statuses.clear();
        for node_to_delete in deletables {
            self.nodes.remove(&node_to_delete);
            on_free_graph(&node_to_delete)
        }
    }

    fn unavailable_propagation(&mut self, node_id: NodeID) -> NodeMemoryStatus {
        // If already visited
        if let Some(status) = self.statuses.get(&node_id) {
            return status.clone();
        }

        match self.nodes.get(&node_id).cloned() {
            // If node exists and any of its children is unavailable, it is unavailable as well
            Some(children) => {
                let mut node_status = NodeMemoryStatus::Unknown;
                for child in children {
                    let child_status = self.unavailable_propagation(child);
                    if let NodeMemoryStatus::Unavailable = child_status {
                        node_status = NodeMemoryStatus::Unavailable;
                    }
                }
                self.statuses.insert(node_id.clone(), node_status.clone());
                node_status
            }
            // If node does not exist, it was deleted, and all its ancestors are unavailable
            None => {
                self.statuses
                    .insert(node_id.clone(), NodeMemoryStatus::Unavailable);
                NodeMemoryStatus::Unavailable
            }
        }
    }

    fn useful_propagation(&mut self, node_id: NodeID, mode: Mode) {
        let children = self.nodes.get(&node_id).cloned().unwrap_or(vec![]);

        match mode {
            Mode::Retain => {
                self.statuses.insert(node_id, NodeMemoryStatus::Useful);
                for child in children {
                    self.useful_propagation(child, Mode::Retain)
                }
            }
            Mode::Explore => {
                let node_status = self
                    .statuses
                    .get(&node_id)
                    .expect("All nodes should have received a status at this point")
                    .clone();

                match node_status {
                    NodeMemoryStatus::Useful => {
                        // Nothing to do, was already tagged by some other path
                    }
                    NodeMemoryStatus::Unavailable => {
                        // Even if this node is useless, it is still possible that a descendant is useful if referenced
                        for child in children {
                            self.useful_propagation(child, Mode::Explore);
                        }
                    }
                    NodeMemoryStatus::Unknown => {
                        // If this node is referenced and not unavailable,
                        // then it is useful and we must retain all descendants

                        let mut mode = Mode::Explore;
                        if self.is_referenced(node_id) {
                            self.statuses.insert(node_id, NodeMemoryStatus::Useful);
                            mode = Mode::Retain;
                        }

                        for child in children {
                            self.useful_propagation(child, mode.clone());
                        }
                    }
                }
            }
        }
    }

    fn is_referenced(&self, node_id: NodeID) -> bool {
        Arc::strong_count(
            self.nodes
                .keys()
                .find(|key| key.as_ref().eq(&node_id))
                .expect("Node should be in the nodes map"),
        ) > 1
    }

    fn identify_roots_and_deletables(
        &self,
        node_id: NodeID,
        new_roots: &mut HashSet<NodeID>,
        to_delete: &mut Vec<NodeID>,
    ) {
        let current_status = self
            .statuses
            .get(&node_id)
            .expect("Node should have status");

        match current_status {
            NodeMemoryStatus::Useful => {
                new_roots.insert(node_id);
            }
            _ => {
                let children = self.nodes.get(&node_id).cloned().unwrap_or(vec![]);
                for child in children {
                    self.identify_roots_and_deletables(child, new_roots, to_delete)
                }
                to_delete.push(node_id);
            }
        }
    }
}
