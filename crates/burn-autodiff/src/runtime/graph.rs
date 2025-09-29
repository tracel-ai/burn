use super::{AutodiffClient, server::AutodiffServer};
use crate::{
    NodeId,
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::{Parent, StepBoxed},
    runtime::server::NodeCleaner,
    tensor::{AutodiffTensor, NodeRefCount},
};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use burn_common::stub::Mutex;
use burn_tensor::backend::Backend;
use hashbrown::HashMap;
use spin::MutexGuard;

/// A client for managing multiple graphs using mutex-based synchronization.
///
/// The biggest benefit of using this client implementation is that each graph can modify its own
/// data without blocking other graphs, which is essential for multi-device training.
///
/// # Notes
///
/// The [AutodiffServer] fully supports multiple graphs with sharing nodes, however those type of
/// graphs will be stored under a single mutex-protected graph by the client, limiting
/// parralelisation.
#[derive(Clone, new, Debug)]
pub struct GraphMutexClient;

/// Manages a collection of graphs, mapping [node ids](NodeId) to their respective graph.
///
/// The `GraphLocator` is responsible for selecting and merging graphs based on their IDs and parent
/// dependencies, ensuring proper synchronization and server allocation.
///
/// # Notes
///
/// Multiple node ids can point to the same graph, where the autodiff graph is stored.
pub struct GraphLocator {
    graphs: HashMap<NodeId, Arc<Graph>>,
}

/// Represents a single computation graph with a mutex-protected server.
///
/// Each `Graph` contains an [AutodiffServer] and the original [NodeId] where the server was
/// first created.
pub(crate) struct Graph {
    server: Mutex<AutodiffServer>,
    id: NodeId,
}

static STATE: spin::Mutex<Option<GraphLocator>> = spin::Mutex::new(None);

impl GraphMutexClient {
    /// Retrieves or creates a graph for the given [NodeId] and parent dependencies.
    ///
    /// # Parameters
    /// - `node`: The unique identifier for the stream.
    /// - `parents`: A slice of parent nodes that the stream depends on.
    ///
    /// # Returns
    /// An `Arc<Graph>` representing the selected or newly created stream.
    fn graph(node: NodeId, parents: &[Parent]) -> Arc<Graph> {
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
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let graph = GraphMutexClient::graph(*node_id, step.parents());
        let mut server = graph.server.lock().unwrap();
        server.register(node_id, step, actions);
    }

    fn backward<B: Backend>(&self, root: AutodiffTensor<B>) -> Gradients {
        let node_id = root.node.id;
        let graph = GraphMutexClient::graph(root.node.id, &[]);

        let grads = Gradients::new::<B>(root.node, root.primitive);
        let mut server = graph.server.lock().unwrap();

        server.backward::<GraphCleaner>(grads, node_id)
    }
}

struct GraphCleaner<'a> {
    guard: MutexGuard<'a, Option<GraphLocator>>,
}

impl<'a> NodeCleaner for GraphCleaner<'a> {
    fn init() -> Self {
        let guard = STATE.lock();
        Self { guard }
    }

    fn clean(&mut self, node: &NodeId) {
        if let Some(state) = self.guard.as_mut() {
            state.graphs.remove(node);
        }
    }
}

impl GraphLocator {
    /// Selects a single graph for the given [NodeId], considering parent dependencies.
    ///
    /// If multiple graphs are found, they are merged into a single one.
    ///
    /// # Parameters
    /// - `node`: The node ID of the graph to select.
    /// - `parents`: A slice of parent nodes that the graph depends on.
    ///
    /// # Returns
    ///
    /// An `Arc<Graph>` representing the selected or merged graph.
    pub(crate) fn select(&mut self, node: NodeId, parents: &[Parent]) -> Arc<Graph> {
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

    fn select_many(&mut self, node: NodeId, parents: &[Parent]) -> Vec<Arc<Graph>> {
        let mut graphs = HashMap::<NodeId, Arc<Graph>>::new();

        if let Some(val) = self.graphs.get(&node) {
            if parents.is_empty() {
                return vec![val.clone()];
            }
            graphs.insert(val.id, val.clone());
        }

        for parent in parents {
            match self.graphs.get(&parent.id) {
                Some(graph) => graphs.insert(graph.id, graph.clone()),
                None => continue,
            };
        }

        if graphs.is_empty() {
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

        graphs.drain().map(|(_, v)| v).collect()
    }

    fn merge(&mut self, node: NodeId, mut graphs: Vec<Arc<Graph>>) -> Arc<Graph> {
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
