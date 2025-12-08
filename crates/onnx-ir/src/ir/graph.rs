//! ONNX graph representation
//!
//! This module contains the OnnxGraph struct which represents a complete
//! ONNX computational graph with nodes, inputs, and outputs.

use super::argument::Argument;
use super::node::{Node, RawNode};
use crate::tensor_store::ValueStore;

/// ONNX graph representation containing fully processed nodes
///
/// After finalization, all Arguments hold immutable `ValueStore` references
/// for accessing tensor data. The graph itself owns a `ValueStore` to ensure
/// the tensor data lives as long as the graph.
#[derive(Debug, Clone, Default)]
pub struct OnnxGraph {
    /// The nodes of the graph (after conversion from RawNode).
    pub nodes: Vec<Node>,

    /// The inputs of the graph.
    pub inputs: Vec<Argument>,

    /// The outputs of the graph.
    pub outputs: Vec<Argument>,

    /// Immutable value store for tensor data lookup
    /// All Arguments' ValueStoreRef::Final references point to this
    pub(crate) value_store: Option<ValueStore>,
}

/// Intermediate graph representation used during processing
///
/// This holds RawNode instances while type inference and processing is happening.
/// After processing is complete, it gets converted to OnnxGraph via convert_to_graph().
///
/// During construction, Arguments hold `ValueStoreRef::Building` with access to GraphState.
/// During finalization, these are converted to `ValueStoreRef::Final` with immutable access.
#[derive(Debug, Clone)]
pub struct OnnxGraphBuilder {
    /// The nodes of the graph (before conversion to final Node enum).
    pub nodes: Vec<RawNode>,

    /// The inputs of the graph.
    pub inputs: Vec<Argument>,

    /// The outputs of the graph.
    pub outputs: Vec<Argument>,

    /// Reference to GraphState during construction (for building value_store)
    pub(crate) graph_state: Option<std::rc::Rc<std::cell::RefCell<crate::graph_state::GraphState>>>,
}

impl OnnxGraphBuilder {
    /// Convert this OnnxGraphBuilder to an OnnxGraph by converting all RawNodes to Nodes
    ///
    /// This recursively converts subgraphs for control flow nodes (If, Loop, Scan).
    /// All Arguments are converted from `ValueStoreRef::Building` to `ValueStoreRef::Final`.
    pub fn convert_to_graph(mut self, opset: usize) -> OnnxGraph {
        // Build immutable ValueStore from GraphState
        let value_store = self
            .graph_state
            .as_ref()
            .map(|gs| gs.borrow().build_value_store());

        // Convert RawNodes to Nodes
        let mut nodes = convert_builders_to_nodes(std::mem::take(&mut self.nodes), opset);

        // Attach value_store to all Arguments
        if let Some(ref vs) = value_store {
            finalize_arguments_in_nodes(&mut nodes, vs);
            for input in &mut self.inputs {
                input.set_value_store(vs.clone());
            }
            for output in &mut self.outputs {
                output.set_value_store(vs.clone());
            }
        }

        OnnxGraph {
            nodes,
            inputs: self.inputs,
            outputs: self.outputs,
            value_store,
        }
    }
}

/// Recursively attach value_store to Arguments in all nodes (including subgraphs)
///
/// For outer-scope Static arguments (constants converted from parent graph), preserves their
/// existing value_store since it contains the tensor data they reference.
fn finalize_arguments_in_nodes(nodes: &mut [Node], value_store: &ValueStore) {
    for node in nodes {
        // Attach value_store to the node's inputs
        for arg in node.inputs_mut() {
            // Preserve value_store for Static arguments that already have a store containing
            // their data. After to_static() converts a Constant to Static, the name is cleared
            // but we can check if the existing store has the data_id.
            let should_preserve = if let crate::ir::ValueSource::Static(data_id) = arg.value_source
            {
                // If the argument already has a store that contains this data_id, preserve it
                arg.value_store
                    .as_ref()
                    .map(|store| store.get_tensor_data(data_id).is_some())
                    .unwrap_or(false)
            } else {
                false
            };

            if !should_preserve {
                arg.set_value_store(value_store.clone());
            }
        }
        // Outputs are always local, so always set the store
        for arg in node.outputs_mut() {
            arg.set_value_store(value_store.clone());
        }

        // Recursively process subgraphs
        finalize_subgraphs_in_node(node);
    }
}

/// Recursively finalize subgraphs within a node
fn finalize_subgraphs_in_node(node: &mut Node) {
    match node {
        Node::If(n) => {
            finalize_subgraph(&mut n.config.then_branch);
            finalize_subgraph(&mut n.config.else_branch);
        }
        Node::Loop(n) => {
            finalize_subgraph(&mut n.config.body);
        }
        Node::Scan(n) => {
            finalize_subgraph(&mut n.config.body);
        }
        _ => {}
    }
}

/// Attach value_store to a subgraph using its own value_store
///
/// Subgraphs have their own GraphState with their own constants (e.g., conv weights
/// within an If branch). We must use the subgraph's own value_store to access these
/// constants, not the parent's value_store.
///
/// However, outer-scope Static arguments already have the parent's value_store set,
/// which contains the tensor data they reference. We must preserve those references.
fn finalize_subgraph(graph: &mut OnnxGraph) {
    // Use the subgraph's own value_store if it has one
    if let Some(ref vs) = graph.value_store {
        let value_store = vs.clone();
        finalize_arguments_in_nodes(&mut graph.nodes, &value_store);
        for input in &mut graph.inputs {
            // Preserve value_store for Static arguments that already have a store containing their data
            let should_preserve =
                if let crate::ir::ValueSource::Static(data_id) = input.value_source {
                    input
                        .value_store
                        .as_ref()
                        .map(|store| store.get_tensor_data(data_id).is_some())
                        .unwrap_or(false)
                } else {
                    false
                };

            if !should_preserve {
                input.set_value_store(value_store.clone());
            }
        }
        for output in &mut graph.outputs {
            // Outputs are always local
            output.set_value_store(value_store.clone());
        }
    }
}

/// Convert a vector of RawNodes to Nodes
fn convert_builders_to_nodes(builders: Vec<RawNode>, opset: usize) -> Vec<Node> {
    let registry = crate::processor::get_processor_registry();

    builders
        .into_iter()
        .map(|builder| {
            let processor = registry.get(&builder.node_type);

            log::debug!(
                "Converting node '{}' of type {:?}",
                builder.name,
                builder.node_type
            );

            processor.build_node(builder, opset)
        })
        .collect()
}
