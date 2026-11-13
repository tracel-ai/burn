use super::{AutodiffClient, server::AutodiffServer};
use crate::{
    NodeId,
    checkpoint::builder::CheckpointerBuilder,
    grads::{BackwardMode, Gradients},
    graph::{Parent, StepBoxed},
    runtime::server::NodeCleaner,
    tensor::{AutodiffTensor, NodeRefCount},
};
use alloc::vec::Vec;

use burn_backend::Backend;
use burn_std::sync::Arc;

use alloc::collections::BTreeMap;
use hashbrown::HashSet;

#[cfg(feature = "std")]
use parking_lot::{Mutex, MutexGuard};

#[cfg(not(feature = "std"))]
use spin::{Mutex, MutexGuard};

/// A client for managing multiple graphs using mutex-based synchronization.
///
/// The biggest benefit of using this client implementation is that each graph can modify its own
/// data without blocking other graphs, which is essential for multi-device training.
///
/// # Notes
///
/// The [AutodiffServer] fully supports multiple graphs with sharing nodes, however those type of
/// graphs will be stored under a single mutex-protected graph by the client, limiting
/// parallelisation.
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
#[derive(Default)]
pub struct GraphLocator {
    graphs: BTreeMap<NodeId, Arc<Graph>>,
    /// One entry per distinct live graph, keyed by the surviving graph's origin.
    ///
    /// `graphs` holds one alias per *node*, so it grows with the total number of
    /// registered operations rather than the number of live graphs. Sweeping
    /// `graphs` on every backward pass therefore costs more over the lifetime of
    /// a long-running process; this map keeps the same graphs deduplicated so a
    /// sweep stays proportional to the number of live graphs. A merge folds the
    /// absorbed graph's nodes into the survivor, so an entry can serve nodes
    /// that were registered under a different origin.
    origins: BTreeMap<NodeId, Arc<Graph>>,
    /// We keep a mapping of each original node id (graph id) => all nodes that point to that graph.
    /// This is to ensure that when merging graphs, we correctly move all previous graphs to
    /// the new merged one.
    keys: BTreeMap<NodeId, HashSet<NodeId>>,
}

/// Represents a single computation graph with a mutex-protected server.
///
/// Each `Graph` contains an [AutodiffServer] and the original [NodeId] where the server was
/// first created.
pub(crate) struct Graph {
    origin: NodeId,
    state: Mutex<GraphState>,
}

#[derive(Default)]
struct GraphState {
    server: AutodiffServer,
}

impl core::fmt::Debug for Graph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graph")
            .field("origin", &self.origin)
            .finish()
    }
}

static STATE: Mutex<Option<GraphLocator>> = Mutex::new(None);

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
                let mut locator = GraphLocator::default();
                let stream = locator.select(node, parents);
                *state = Some(locator);
                stream
            }
        }
    }
}

impl AutodiffClient for GraphMutexClient {
    fn register(&self, node_id_ref: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let node_id = *node_id_ref;
        let graph = GraphMutexClient::graph(node_id, step.parents());
        let mut state = graph.state.lock();

        state.server.register(node_id_ref, step, actions);
    }

    fn backward<B: Backend>(&self, root: AutodiffTensor<B>, mode: BackwardMode) -> Gradients {
        let node_id = root.node.id;
        let graph = GraphMutexClient::graph(root.node.id, &[]);

        let grads = {
            let mut state = graph.state.lock();
            state
                .server
                .backward::<GraphCleaner, B>(root.node, root.primitive, node_id, mode)
        }; // lock released

        GraphCleaner::cleanup_orphaned_entries();

        grads
    }
}

struct GraphCleaner<'a> {
    guard: MutexGuard<'a, Option<GraphLocator>>,
}

impl<'a> GraphCleaner<'a> {
    fn cleanup_orphaned_entries() {
        let graphs = {
            // Get the available graphs and release the lock. Iterate the
            // deduplicated set: `graphs` holds one alias per node, so cloning
            // it would cost O(total registered nodes) on every backward pass.
            match STATE.lock().as_ref() {
                Some(state) => state.origins.values().cloned().collect::<Vec<_>>(),
                None => return,
            }
        };

        let mut should_remove = Vec::new();
        for graph in graphs.iter() {
            {
                // Best-effort sweep: a graph whose mutex is contended is in
                // use — possibly by this very thread, when a `Backward` step
                // itself runs an inner `backward()` (the outer backward holds
                // its graph's lock across step execution, and this mutex is
                // not reentrant). Skip it; a later backward's sweep will
                // collect its orphans.
                let Some(mut guard) = graph.state.try_lock() else {
                    continue;
                };
                // Double safety: in case it was marked as no longer useful, but other
                // nodes are still relevant, we only check which nodes can safely be removed.
                if !guard.server.maybe_useful() {
                    guard
                        .server
                        .free_unused_roots(|node| should_remove.push(*node));
                }
            }
        }

        if !should_remove.is_empty() {
            let mut state = STATE.lock();
            if let Some(state) = state.as_mut() {
                for node in should_remove {
                    state.remove_entry(&node);
                }
            }
        }
    }
}

impl<'a> NodeCleaner for GraphCleaner<'a> {
    fn init() -> Self {
        let guard = STATE.lock();
        Self { guard }
    }

    fn clean(&mut self, node: &NodeId) {
        if let Some(state) = self.guard.as_mut() {
            state.remove_entry(node);
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
        match self.analyse(node, parents) {
            GraphAnalysis::NoCollision(graph) => {
                if graph.origin != node {
                    self.graphs.insert(node, graph.clone());
                    self.register_key(graph.origin, node);
                }

                graph
            }
            GraphAnalysis::Collisions(graphs) => self.merge(node, graphs),
        }
    }

    /// Analyses the graph for a given node and its parents, returning the associated `GraphAnalysis`.
    fn analyse(&mut self, node: NodeId, parents: &[Parent]) -> GraphAnalysis {
        // If no parents, there is no collision, therefore a single graph is ok.
        if parents.is_empty() {
            let graph = match self.graphs.get(&node) {
                Some(val) => val.clone(),
                None => self.new_graph(node),
            };
            return GraphAnalysis::NoCollision(graph);
        };

        // We collect all graphs of parents and of the current node based on their origin node id.
        let mut graphs = BTreeMap::<NodeId, Arc<Graph>>::new();

        if let Some(val) = self.graphs.get(&node) {
            graphs.insert(val.origin, val.clone());
        }

        for parent in parents {
            match self.graphs.get(&parent.id) {
                Some(graph) => graphs.insert(graph.origin, graph.clone()),
                None => continue,
            };
        }

        if graphs.is_empty() {
            return match self.graphs.get(&node) {
                Some(old) => GraphAnalysis::NoCollision(old.clone()),
                None => GraphAnalysis::NoCollision(self.new_graph(node)),
            };
        }

        if graphs.len() == 1 {
            return GraphAnalysis::NoCollision(graphs.into_values().next().unwrap());
        }

        GraphAnalysis::Collisions(graphs)
    }

    /// Merges multiple graphs associated with a node into a single graph.
    fn merge(&mut self, node: NodeId, graphs: BTreeMap<NodeId, Arc<Graph>>) -> Arc<Graph> {
        // `into_values()` on the `BTreeMap` yields graphs in ascending `NodeId` order, so the "main"
        // graph and the merge order are deterministic (a `HashMap` here picked an arbitrary,
        // run-to-run-varying main, which permuted the merged steps and the backward op stream).
        let mut graphs = graphs.into_values();

        let main = graphs.next().expect("At least one graph");
        self.register_key(main.origin, node);

        let mut state = main.state.lock();

        for graph in graphs {
            self.merge_two(&mut state, &main, graph);
        }

        self.graphs.insert(main.origin, main.clone());
        self.graphs.insert(node, main.clone());
        self.origins.insert(main.origin, main.clone());

        core::mem::drop(state);

        main
    }

    /// Registers a key for a given origin node.
    fn register_key(&mut self, origin: NodeId, key: NodeId) {
        self.keys.entry(origin).or_default();

        if origin != key {
            // Register this node to point to the origin graph
            self.keys.get_mut(&origin).unwrap().insert(key);
        }
    }

    /// Merges two graphs by combining their states and updating graph mappings.
    fn merge_two(&mut self, main_state: &mut GraphState, main: &Arc<Graph>, merged: Arc<Graph>) {
        let mut locked = merged.state.lock();
        let mut state_old = GraphState::default();
        core::mem::swap(&mut state_old, &mut locked);
        main_state.server.extend(state_old.server);

        // Re-map merged origin to the main graph
        self.graphs.insert(merged.origin, main.clone());
        self.origins.remove(&merged.origin);

        // Move all keys (node IDs) from the merged graph to the main graph
        if let Some(locator_keys) = self.keys.remove(&merged.origin) {
            for k in locator_keys.iter() {
                self.graphs.insert(*k, main.clone());
            }

            let locator_keys_main = self
                .keys
                .get_mut(&main.origin)
                .expect("Should be init before the merge.");
            locator_keys_main.extend(locator_keys);
        }
    }

    /// Creates a new graph for a given node.
    fn new_graph(&mut self, origin: NodeId) -> Arc<Graph> {
        let graph = Arc::new(Graph {
            origin,
            state: Mutex::new(GraphState::default()),
        });
        self.graphs.insert(origin, graph.clone());
        self.origins.insert(origin, graph.clone());
        self.keys.insert(origin, HashSet::new());
        graph
    }

    fn remove_entry(&mut self, node: &NodeId) {
        if let Some(graph) = self.graphs.remove(node) {
            let origin = graph.origin;

            // `keys[origin]` holds every alias of the graph except the origin
            // node itself, so an absent entry means there are no aliases left.
            let aliases_left = match self.keys.get_mut(&origin) {
                Some(entry) => {
                    entry.remove(node);
                    !entry.is_empty()
                }
                None => false,
            };
            if !aliases_left {
                self.keys.remove(&origin);
            }

            // A graph is unreachable once neither its origin node nor any of its
            // aliases remain in `graphs`. Mirror the alias prune in `origins` so
            // dead graphs do not linger there between sweeps.
            let origin_alive = self
                .graphs
                .get(&origin)
                .is_some_and(|current| Arc::ptr_eq(current, &graph));
            if !aliases_left && !origin_alive {
                self.origins.remove(&origin);
            }
        }
    }
}

/// Represents the analysis result of graph operations for a given node and its parents.
#[derive(Debug)]
enum GraphAnalysis {
    /// No collision detected, contains the graph associated with the node.
    NoCollision(Arc<Graph>),
    /// Collision detected, contains a map of node IDs to their associated graphs.
    Collisions(BTreeMap<NodeId, Arc<Graph>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node() -> NodeId {
        NodeId::new()
    }

    #[test]
    fn origins_hold_one_entry_per_graph() {
        let mut locator = GraphLocator::default();
        let root = node();
        let graph = locator.select(root, &[]);

        for _ in 0..64 {
            let child = node();
            let selected = locator.select(child, &[Parent::new(root, false)]);
            assert!(Arc::ptr_eq(&selected, &graph));
        }

        // One alias per node, but a single distinct graph.
        assert_eq!(locator.graphs.len(), 65);
        assert_eq!(locator.origins.len(), 1);
    }

    #[test]
    fn removing_the_last_alias_drops_the_origin() {
        let mut locator = GraphLocator::default();
        let root = node();
        locator.select(root, &[]);
        let child = node();
        locator.select(child, &[Parent::new(root, false)]);

        locator.remove_entry(&child);
        assert_eq!(locator.graphs.len(), 1);
        assert_eq!(locator.origins.len(), 1);

        locator.remove_entry(&root);
        assert_eq!(locator.graphs.len(), 0);
        assert_eq!(locator.origins.len(), 0);
    }

    #[test]
    fn merging_graphs_keeps_one_origin() {
        let mut locator = GraphLocator::default();
        let left = node();
        let right = node();
        locator.select(left, &[]);
        locator.select(right, &[]);

        let child = node();
        let merged = locator.select(
            child,
            &[Parent::new(left, false), Parent::new(right, false)],
        );

        assert_eq!(locator.origins.len(), 1);
        assert!(locator.origins.contains_key(&merged.origin));
    }

    /// Removing a merged-away graph's nodes must not drop the surviving graph
    /// from `origins`. An origin that is gone from `graphs` is only dead once
    /// the *surviving* graph has no aliases left either; a wrong prune here is
    /// a slow leak rather than a crash, because the sweep silently stops
    /// visiting the graph and its orphans are never collected.
    #[test]
    fn removing_a_merged_graph_keeps_the_surviving_origin() {
        let mut locator = GraphLocator::default();

        let left = node();
        locator.select(left, &[]);
        let left_alias = node();
        locator.select(left_alias, &[Parent::new(left, false)]);

        let right = node();
        locator.select(right, &[]);
        let right_alias = node();
        locator.select(right_alias, &[Parent::new(right, false)]);

        let child = node();
        let merged = locator.select(
            child,
            &[Parent::new(left, false), Parent::new(right, false)],
        );
        assert_eq!(locator.origins.len(), 1);
        assert!(locator.origins.contains_key(&merged.origin));

        // A merge keeps the graph with the smallest origin id, so which side
        // was absorbed is only known from the result.
        let (absorbed, absorbed_alias, survivor_alias) = if merged.origin == left {
            (right, right_alias, left_alias)
        } else {
            (left, left_alias, right_alias)
        };

        // Every node of the absorbed graph now aliases the survivor, so
        // removing them one by one must leave its origin entry alone.
        for alias in [absorbed, absorbed_alias] {
            locator.remove_entry(&alias);
            assert_eq!(
                locator.origins.len(),
                1,
                "removing an absorbed node dropped the surviving graph"
            );
            assert!(locator.origins.contains_key(&merged.origin));
        }

        // The survivor's own nodes still hold it, and the last one removes it.
        locator.remove_entry(&survivor_alias);
        locator.remove_entry(&child);
        assert_eq!(locator.origins.len(), 1);
        assert!(locator.origins.contains_key(&merged.origin));

        locator.remove_entry(&merged.origin);
        assert_eq!(locator.origins.len(), 0);
        assert!(locator.graphs.is_empty());
    }
}
