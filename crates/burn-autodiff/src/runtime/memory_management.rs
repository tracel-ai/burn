use crate::{tensor::NodeRefCount, NodeID};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(new, Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct GraphId {
    node: NodeID,
}

#[derive(Debug)]
enum GraphState {
    Merged(GraphId),
    Owned(Vec<NodeRefCount>),
}

#[derive(Default, Debug)]
pub struct GraphsMemoryManagement {
    graphs: HashMap<GraphId, GraphState>,
    owned: HashSet<GraphId>,
}

impl core::fmt::Display for GraphsMemoryManagement {
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

impl GraphsMemoryManagement {
    pub fn register(&mut self, rc: NodeRefCount, parents: Vec<NodeID>) {
        let node_id = *rc.as_ref();
        let graph_id = GraphId::new(node_id);

        self.insert_owned_graph(graph_id, vec![rc.clone()]);

        if !parents.is_empty() {
            let graph_ids = parents.into_iter().map(GraphId::new);
            if let Some(parent_graph_id) = self.merge_graph(graph_ids) {
                self.merge_graph([graph_id, parent_graph_id]);
            }
        }
    }

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
        }
    }

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_graph_memory_management() {
        let mut graph_mm = GraphsMemoryManagement::default();

        let node_1 = Arc::new(NodeID::new());
        let node_2 = Arc::new(NodeID::new());
        let node_3 = Arc::new(NodeID::new());
        let node_4 = Arc::new(NodeID::new());
        let node_5 = Arc::new(NodeID::new());

        // node_1 is root
        graph_mm.register(node_1.clone(), vec![]);
        graph_mm.register(node_2.clone(), vec![*node_1]);
        graph_mm.register(node_3.clone(), vec![]);
        graph_mm.register(node_4.clone(), vec![*node_3]);
        graph_mm.register(node_5.clone(), vec![*node_1, *node_3]);

        core::mem::drop(node_1);
        core::mem::drop(node_2);
        core::mem::drop(node_3);
        core::mem::drop(node_4);
        core::mem::drop(node_5);

        let disconected_graphs = graph_mm.find_orphan_graphs();

        assert_eq!(disconected_graphs.len(), 1);
    }
}
