use crate::{tensor::NodeRefCount, NodeID};
use std::{
    collections::{HashMap, HashSet},
    mem,
    sync::Arc,
};

#[derive(Default, Debug)]
pub struct GraphMemoryManagement {
    nodes: HashMap<NodeRefCount, Vec<NodeID>>,
    leaves: HashSet<NodeID>,
    statuses: HashMap<NodeID, NodeMemoryStatus>,
}

#[derive(Debug, Clone, PartialEq)]
enum NodeMemoryStatus {
    Useful,
    Unavailable,
    Unknown,
}

impl GraphMemoryManagement {
    /// Register a new node with its parent.
    pub fn register(&mut self, node: NodeRefCount, parents: Vec<NodeID>) {
        let node_id = *node.as_ref();

        for parent_id in parents.iter() {
            self.leaves.remove(parent_id);
        }

        self.leaves.insert(node_id);
        self.nodes.insert(node, parents);
    }

    /// Free the node from the state.
    pub fn consume_node(&mut self, node_id: NodeID) {
        if !self.is_referenced(node_id) {
            self.leaves.remove(&node_id);
            self.nodes.remove(&node_id);
        }
    }

    /// Free all nodes whose backward call has become impossible
    ///
    /// This function goes into three steps, which must happen for all leaves
    /// before going into the next step. Then it deletes what can be safely deleted
    pub(crate) fn free_unavailable_nodes(&mut self, mut on_free_graph: impl FnMut(&NodeID)) {
        let leaves = self.leaves.clone();
        let mut new_leaves = HashSet::new();
        let mut deletables = Vec::new();

        // When consuming nodes with a backward pass, some other backward passes become
        // unavailable because some of their parents have been consumed. They are
        // identified here.
        for leaf in leaves.clone() {
            self.unavailable_propagation(leaf);
        }

        // Among the available nodes that remain, some may be useless if no
        // available node with a tensor reference exist in their descendance.
        // But some may seem useless from some leaf but be useful from another one,
        // hence the need to iterate on all leaves.
        self.useful_propagation(leaves.clone());

        // New leaves are the roots of a useful backward sub-tree.
        // Deletables are everything not marked as useful.
        for leaf in leaves {
            self.identify_leaves_and_deletables(leaf, &mut new_leaves, &mut deletables);
        }

        // Replace leaves by the new ones and delete everything not useful anymore
        mem::swap(&mut self.leaves, &mut new_leaves);

        self.clear_unused_roots(&mut deletables);

        self.statuses.clear();
        for node_to_delete in deletables {
            self.nodes.remove(&node_to_delete);
            on_free_graph(&node_to_delete)
        }
    }

    fn clear_unused_roots(&mut self, to_delete: &mut Vec<NodeID>) {
        for (id, parents) in self.nodes.iter() {
            let is_useful = matches!(self.statuses.get(id), Some(NodeMemoryStatus::Useful));

            if !is_useful && Arc::strong_count(id) == 1 && parents.is_empty() {
                to_delete.push(*id.as_ref())
            }
        }
    }

    fn unavailable_propagation(&mut self, node_id: NodeID) -> NodeMemoryStatus {
        // If already visited
        if let Some(status) = self.statuses.get(&node_id) {
            return status.clone();
        }

        match self.nodes.get(&node_id).cloned() {
            // If node exists and any of its parents is unavailable, it is unavailable as well
            // If node exists but the parents vec is empty, it is a tensor that never had parents;
            //  the status remains unknown
            Some(parents) => {
                let mut node_status = NodeMemoryStatus::Unknown;
                for parent in parents {
                    let parent_status = self.unavailable_propagation(parent);
                    if let NodeMemoryStatus::Unavailable = parent_status {
                        node_status = NodeMemoryStatus::Unavailable;
                    }
                }
                self.statuses.insert(node_id, node_status.clone());
                node_status
            }
            // If node does not exist, it was
            // deleted, so this and all its descendants are unavailable
            None => {
                self.statuses.insert(node_id, NodeMemoryStatus::Unavailable);
                NodeMemoryStatus::Unavailable
            }
        }
    }

    fn useful_propagation(&mut self, leaves: HashSet<NodeID>) {
        // Accumulate visited nodes
        let mut explored = HashSet::new();
        let mut tagged_useful = HashSet::new();

        // Queue of nodes to visit
        let mut to_tag_useful = PopNodeSet::default();
        let mut to_explore = PopNodeSet::new(leaves);

        // Utilitary function to iterate over a node's parents
        let parents = |node_id| {
            self.nodes
                .get(&node_id)
                .cloned()
                .unwrap_or_default()
                .into_iter()
        };

        loop {
            // Pop a node id, greedily looking at tag_useful ones first
            let (node_id, status) = match to_tag_useful.pop() {
                Some(node_id) => (node_id, NodeMemoryStatus::Useful),
                None => match to_explore.pop() {
                    Some(node_id) => {
                        let node_status = self
                            .statuses
                            .get(&node_id)
                            .expect("All nodes should have received a status during unavailable_propagation")
                            .to_owned();

                        if let NodeMemoryStatus::Unknown = node_status {
                            match self.is_referenced(node_id) {
                                true => (node_id, NodeMemoryStatus::Useful),
                                false => (node_id, NodeMemoryStatus::Unknown),
                            }
                        } else {
                            (node_id, node_status)
                        }
                    }
                    None => {
                        // There are no nodes in the queues anymore
                        break;
                    }
                },
            };

            match status {
                NodeMemoryStatus::Useful => {
                    tagged_useful.insert(node_id);
                    for parent in parents(node_id) {
                        // The node can be explored, as long as it's not already tagged useful
                        if !(tagged_useful.contains(&parent) || to_tag_useful.contains(&parent)) {
                            to_tag_useful.insert(parent);
                        }
                    }
                }
                _ => {
                    explored.insert(node_id);
                    for parent in parents(node_id) {
                        if !(explored.contains(&parent) || to_explore.contains(&parent)) {
                            to_explore.insert(parent);
                        }
                    }
                }
            }

            self.statuses.insert(node_id, status);
        }
    }

    fn identify_leaves_and_deletables(
        &self,
        leaf_id: NodeID,
        new_leaves: &mut HashSet<NodeID>,
        to_delete: &mut Vec<NodeID>,
    ) {
        let mut visited = HashSet::new();
        let mut to_visit = vec![leaf_id];

        while let Some(node_id) = to_visit.pop() {
            visited.insert(node_id);

            match self
                .statuses
                .get(&node_id)
                .expect("Node should have status")
            {
                NodeMemoryStatus::Useful => {
                    new_leaves.insert(node_id);
                }
                _ => {
                    to_delete.push(node_id);

                    for parent in self
                        .nodes
                        .get(&node_id)
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                    {
                        if !visited.contains(&parent) {
                            to_visit.push(parent);
                        }
                    }
                }
            };
        }
    }

    fn is_referenced(&self, node_id: NodeID) -> bool {
        match self.nodes.get_key_value(&node_id) {
            Some((key, _value)) => Arc::strong_count(key) > 1,
            None => panic!("Node should be in the nodes map"),
        }
    }
}

/// Wrapper over hash set for fast popping of any node
#[derive(new, Default)]
struct PopNodeSet {
    hash_set: HashSet<NodeID>,
}

impl PopNodeSet {
    #[inline(always)]
    fn pop(&mut self) -> Option<NodeID> {
        self.hash_set
            .iter()
            .next()
            .copied()
            .and_then(|node_id| self.hash_set.take(&node_id))
    }

    #[inline(always)]
    fn contains(&self, node_id: &NodeID) -> bool {
        self.hash_set.contains(node_id)
    }

    #[inline(always)]
    fn insert(&mut self, node_id: NodeID) {
        self.hash_set.insert(node_id);
    }
}
