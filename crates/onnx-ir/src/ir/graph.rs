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
fn finalize_arguments_in_nodes(nodes: &mut [Node], value_store: &ValueStore) {
    for node in nodes {
        // Attach value_store to the node's inputs/outputs
        for arg in node.inputs_mut() {
            arg.set_value_store(value_store.clone());
        }
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
fn finalize_subgraph(graph: &mut OnnxGraph) {
    // Use the subgraph's own value_store if it has one
    if let Some(ref vs) = graph.value_store {
        let value_store = vs.clone();
        finalize_arguments_in_nodes(&mut graph.nodes, &value_store);
        for input in &mut graph.inputs {
            input.set_value_store(value_store.clone());
        }
        for output in &mut graph.outputs {
            output.set_value_store(value_store.clone());
        }
    }
}

/// Convert a vector of RawNodes to Nodes, handling subgraphs recursively
pub fn finalize_graph_nodes(builders: &mut Vec<RawNode>, opset: usize) -> Vec<Node> {
    let taken_builders = std::mem::take(builders);
    convert_builders_to_nodes(taken_builders, opset)
}

/// Convert a vector of RawNodes to Nodes, handling subgraphs recursively
fn convert_builders_to_nodes(builders: Vec<RawNode>, opset: usize) -> Vec<Node> {
    let registry = crate::processor::get_processor_registry();

    builders
        .into_iter()
        .map(|builder| {
            let processor = registry.get(&builder.node_type);

            // For control flow nodes with subgraphs, we need to convert those subgraphs first
            let builder = convert_subgraphs_in_attributes(builder, opset);

            // Debug: log which node is being converted
            log::debug!(
                "Converting node '{}' of type {:?}",
                builder.name,
                builder.node_type
            );

            // Now build the node
            processor.build_node(builder, opset)
        })
        .collect()
}

/// Convert any subgraphs in node attributes from OnnxGraphBuilder to OnnxGraph
fn convert_subgraphs_in_attributes(mut builder: RawNode, opset: usize) -> RawNode {
    use crate::ir::AttributeValue;

    for attr_value in builder.attrs.values_mut() {
        match attr_value {
            AttributeValue::GraphBuilder(subgraph_builder) => {
                // Build value store from subgraph's GraphState
                let value_store = subgraph_builder
                    .graph_state
                    .as_ref()
                    .map(|gs| gs.borrow().build_value_store());

                // Convert the subgraph's RawNodes to Nodes
                let nodes =
                    convert_builders_to_nodes(std::mem::take(&mut subgraph_builder.nodes), opset);

                // Create a new OnnxGraph with converted nodes
                *attr_value = AttributeValue::Graph(OnnxGraph {
                    nodes,
                    inputs: std::mem::take(&mut subgraph_builder.inputs),
                    outputs: std::mem::take(&mut subgraph_builder.outputs),
                    value_store,
                });
            }
            AttributeValue::GraphBuilders(subgraph_builders) => {
                let converted_graphs: Vec<OnnxGraph> = subgraph_builders
                    .iter_mut()
                    .map(|subgraph_builder| {
                        let value_store = subgraph_builder
                            .graph_state
                            .as_ref()
                            .map(|gs| gs.borrow().build_value_store());

                        let nodes = convert_builders_to_nodes(
                            std::mem::take(&mut subgraph_builder.nodes),
                            opset,
                        );

                        OnnxGraph {
                            nodes,
                            inputs: std::mem::take(&mut subgraph_builder.inputs),
                            outputs: std::mem::take(&mut subgraph_builder.outputs),
                            value_store,
                        }
                    })
                    .collect();

                *attr_value = AttributeValue::Graphs(converted_graphs);
            }
            _ => {}
        }
    }

    builder
}
