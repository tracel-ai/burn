use crate::{tensor::NodeRefCount, NodeID};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// Keeps a version on the graphs created during autodiff with the reference count of each node.
///
/// When all nodes in a graph have only one reference, the graph can be freed.
#[derive(Default, Debug)]
pub struct GraphMemoryManagement {
    graphs: HashMap<GraphId, GraphState>,
    owned: HashSet<GraphId>,
}

#[derive(new, Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct GraphId {
    node: NodeID,
}

#[derive(Debug)]
enum GraphState {
    Merged(GraphId),
    Owned(Vec<NodeRefCount>),
}

impl GraphMemoryManagement {
    /// Register a new node with its parent.
    pub fn register(&mut self, node: NodeRefCount, parents: Vec<NodeID>) {
        let node_id = *node.as_ref();
        let graph_id = GraphId::new(node_id);

        self.insert_owned_graph(graph_id, vec![node.clone()]);

        if !parents.is_empty() {
            let graph_ids = parents.into_iter().map(GraphId::new);
            if let Some(parent_graph_id) = self.merge_graph(graph_ids) {
                self.merge_graph([graph_id, parent_graph_id]);
            }
        }
    }

    /// Free the given graph calling the given function for each node deleted.
    pub fn free_graph<F>(&mut self, graph_id: GraphId, mut func: F)
    where
        F: FnMut(&NodeID),
    {
        self.owned.remove(&graph_id);
        let graph = match self.graphs.remove(&graph_id) {
            Some(graph) => graph,
            None => return,
        };

        let graph = match graph {
            GraphState::Merged(graph) => {
                self.free_graph(graph, func);
                return;
            }
            GraphState::Owned(graph) => graph,
        };

        for node_id in graph.into_iter() {
            func(&node_id);
            self.graphs.remove(&GraphId::new(*node_id));
        }
    }

    /// Find the graphs where all nodes are orphan.
    ///
    /// The returned graphs can be safely freed.
    pub fn find_orphan_graphs(&self) -> Vec<GraphId> {
        self.owned
            .iter()
            .filter(|id| self.is_orphan(id))
            .copied()
            .collect()
    }

    fn is_orphan(&self, id: &GraphId) -> bool {
        let graph = match self.graphs.get(id) {
            Some(val) => val,
            None => return false,
        };

        let nodes = match graph {
            GraphState::Merged(_) => return false,
            GraphState::Owned(nodes) => nodes,
        };

        for node in nodes {
            if Arc::strong_count(node) > 1 {
                return false;
            }
        }

        true
    }

    fn insert_owned_graph(&mut self, graph_id: GraphId, nodes: Vec<NodeRefCount>) {
        self.graphs.insert(graph_id, GraphState::Owned(nodes));
        self.owned.insert(graph_id);
    }

    fn merge_graph<I: IntoIterator<Item = GraphId>>(&mut self, graph_ids: I) -> Option<GraphId> {
        let graph_ids = graph_ids.into_iter();
        let graph_ids = graph_ids.collect::<Vec<_>>();

        let mut merged = HashSet::new();

        let mut updated_nodes = Vec::new();
        let mut updated_graph_id = None;

        for id in graph_ids {
            let graph_id = match self.find_owned_graph(id) {
                Some(val) => val,
                None => continue,
            };

            if updated_graph_id.is_none() {
                updated_graph_id = Some(graph_id);
            }

            merged.insert(graph_id);
        }

        let updated_graph_id = match updated_graph_id {
            Some(val) => val,
            None => return None,
        };

        for id in merged {
            let mut updated_state = GraphState::Merged(updated_graph_id);
            let state = self.graphs.get_mut(&id).unwrap();
            self.owned.remove(&id);

            core::mem::swap(state, &mut updated_state);

            if let GraphState::Owned(nodes) = updated_state {
                updated_nodes.extend(nodes)
            };
        }

        self.insert_owned_graph(updated_graph_id, updated_nodes);

        Some(updated_graph_id)
    }

    fn find_owned_graph(&mut self, graph_id: GraphId) -> Option<GraphId> {
        let graph = match self.graphs.get(&graph_id) {
            Some(val) => val,
            None => return None,
        };

        let merged_graph_id = match graph {
            GraphState::Merged(graph_id) => graph_id,
            GraphState::Owned(_) => return Some(graph_id),
        };

        self.find_owned_graph(*merged_graph_id)
    }
}

impl core::fmt::Display for GraphMemoryManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Graphs Memory Management with {} owned graphs and total of {} graphs\n",
            self.owned.len(),
            self.graphs.len()
        ))?;
        for (id, state) in self.graphs.iter() {
            f.write_fmt(format_args!("Graph {} => ", id.node.value))?;
            match state {
                GraphState::Merged(id) => f.write_fmt(format_args!("Merged {}", id.node.value))?,
                GraphState::Owned(nodes) => {
                    f.write_str("Owned")?;
                    for node in nodes {
                        f.write_fmt(format_args!(" {}", node.value))?;
                    }
                }
            }
            f.write_str("\n")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_graph_memory_management_connect_graphs() {
        let mut graph_mm = GraphMemoryManagement::default();

        let node_1 = Arc::new(NodeID::new());
        let node_2 = Arc::new(NodeID::new());
        let node_3 = Arc::new(NodeID::new());
        let node_4 = Arc::new(NodeID::new());
        let node_5 = Arc::new(NodeID::new());

        graph_mm.register(node_1.clone(), vec![]);
        graph_mm.register(node_2.clone(), vec![*node_1]);
        assert_eq!(graph_mm.owned.len(), 1, "A single connected graph.");

        graph_mm.register(node_3.clone(), vec![]);
        graph_mm.register(node_4.clone(), vec![*node_3]);
        assert_eq!(graph_mm.owned.len(), 2, "Two connected graphs.");

        graph_mm.register(node_5.clone(), vec![*node_1, *node_3]);
        assert_eq!(
            graph_mm.owned.len(),
            1,
            "Two connected graphs are merged into one."
        );
    }

    #[test]
    fn test_graph_memory_management_find_orphans() {
        let mut graph_mm = GraphMemoryManagement::default();

        let node_1 = Arc::new(NodeID::new());
        let node_2 = Arc::new(NodeID::new());

        graph_mm.register(node_1.clone(), vec![]);
        graph_mm.register(node_2.clone(), vec![*node_1]);

        core::mem::drop(node_1);
        assert_eq!(
            graph_mm.find_orphan_graphs().len(),
            0,
            "Not all nodes are dropped"
        );

        core::mem::drop(node_2);
        assert_eq!(
            graph_mm.find_orphan_graphs().len(),
            1,
            "All nodes are dropped"
        );
    }

    #[test]
    fn test_graph_memory_management_free_graph_from_any_node() {
        let mut graph_mm = GraphMemoryManagement::default();

        // Create a graph and free(node_1)
        let node_1 = Arc::new(NodeID::new());
        let node_2 = Arc::new(NodeID::new());

        graph_mm.register(node_1.clone(), vec![]);
        graph_mm.register(node_2.clone(), vec![*node_1]);

        let mut node_ids = Vec::new();
        graph_mm.free_graph(GraphId::new(*node_1.as_ref()), |id| node_ids.push(*id));

        assert!(node_ids.contains(&node_1));
        assert!(node_ids.contains(&node_2));

        assert_eq!(graph_mm.graphs.len(), 0);
        assert_eq!(graph_mm.owned.len(), 0);

        // Same but with free(node_2);
        graph_mm.register(node_1.clone(), vec![]);
        graph_mm.register(node_2.clone(), vec![*node_1]);

        let mut node_ids = Vec::new();
        graph_mm.free_graph(GraphId::new(*node_2.as_ref()), |id| node_ids.push(*id));

        assert!(node_ids.contains(&node_1));
        assert!(node_ids.contains(&node_2));

        assert_eq!(graph_mm.graphs.len(), 0);
        assert_eq!(graph_mm.owned.len(), 0);
    }
}
