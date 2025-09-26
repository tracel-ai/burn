use super::{AutodiffClient, server::AutodiffServer};
use crate::{
    NodeID,
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::{Parent, StepBoxed},
    tensor::{AutodiffTensor, NodeRefCount},
};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use burn_common::stub::Mutex;
use burn_tensor::backend::Backend;
use hashbrown::HashMap;

#[derive(Clone, new, Debug)]
pub struct GraphMutexClient;

pub struct GraphLocator {
    graphs: HashMap<NodeID, Arc<Graph>>,
}

struct Graph {
    server: Mutex<AutodiffServer>,
    id: NodeID,
}

static STATE: spin::Mutex<Option<GraphLocator>> = spin::Mutex::new(None);

impl GraphMutexClient {
    fn clean(nodes: Vec<NodeID>) {
        let mut state = STATE.lock();
        if let Some(locator) = state.as_mut() {
            for node in nodes {
                locator.graphs.remove(&node);
            }
        }
    }
    fn graph(node: NodeID, parents: &[Parent]) -> Arc<Graph> {
        let mut state = STATE.lock();

        match state.as_mut() {
            Some(locator) => locator.select(node, parents),
            None => {
                let mut locator = GraphLocator {
                    graphs: HashMap::new(),
                };
                let stream = locator.select(node, parents);
                *state = Some(locator);
                stream
            }
        }
    }
}

impl AutodiffClient for GraphMutexClient {
    fn register(
        &self,
        _stream_id: burn_common::stream_id::StreamId,
        node_id: NodeRefCount,
        step: StepBoxed,
        actions: CheckpointerBuilder,
    ) {
        let stream = GraphMutexClient::graph(*node_id, step.parents());
        let mut server = stream.server.lock().unwrap();
        server.register(node_id, step, actions);
    }

    fn backward<B: Backend>(&self, root: AutodiffTensor<B>) -> Gradients {
        let node_id = root.node.id;
        let stream = GraphMutexClient::graph(root.node.id, &[]);

        let grads = Gradients::new::<B>(root.node, root.primitive);
        let mut server = stream.server.lock().unwrap();
        let mut to_clean = Vec::new();

        let grads = server.backward(grads, node_id, |n| to_clean.push(*n));
        Self::clean(to_clean);
        grads
    }
}

impl GraphLocator {
    fn select(&mut self, node: NodeID, parents: &[Parent]) -> Arc<Graph> {
        let mut graphs = self.select_many(node, parents);

        if graphs.len() == 1 {
            let graph = graphs.pop().unwrap();
            if graph.id != node {
                self.graphs.insert(node, graph.clone());
            }

            return graph;
        }

        self.merge(node, graphs)
    }

    fn select_many(&mut self, node: NodeID, parents: &[Parent]) -> Vec<Arc<Graph>> {
        let mut servers = HashMap::<NodeID, Arc<Graph>>::new();

        if let Some(val) = self.graphs.get(&node) {
            if parents.is_empty() {
                return vec![val.clone()];
            }
            servers.insert(val.id, val.clone());
        }

        for parent in parents {
            match self.graphs.get(&parent.id) {
                Some(val) => servers.insert(val.id, val.clone()),
                None => continue,
            };
        }

        if servers.is_empty() {
            return match self.graphs.get(&node) {
                Some(old) => vec![old.clone()],
                None => {
                    let server = Arc::new(Graph {
                        server: Mutex::new(AutodiffServer::default()),
                        id: node,
                    });

                    self.graphs.insert(node, server.clone());
                    vec![server]
                }
            };
        }

        servers.drain().map(|(_, v)| v).collect()
    }

    fn merge(&mut self, node: NodeID, mut graphs: Vec<Arc<Graph>>) -> Arc<Graph> {
        let mut graph_ids = Vec::with_capacity(graphs.len());
        let main = graphs.pop().unwrap();

        let mut server = main.server.lock().unwrap();

        for graph in graphs.drain(..) {
            let mut locked = graph.server.lock().unwrap();
            let mut ser = AutodiffServer::default();
            core::mem::swap(&mut ser, &mut locked);
            server.extend(ser);
            graph_ids.push(graph.id);
        }

        for gid in graph_ids {
            self.graphs.insert(gid, main.clone());
        }
        self.graphs.insert(node, main.clone());

        core::mem::drop(server);

        main
    }
}
