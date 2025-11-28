//! ONNX graph representation
//!
//! This module contains the OnnxGraph struct which represents a complete
//! ONNX computational graph with nodes, inputs, and outputs.

use super::argument::Argument;
use super::node::{Node, RawNode};

/// ONNX graph representation containing fully processed nodes
#[derive(Debug, Clone, Default)]
pub struct OnnxGraph {
    /// The nodes of the graph (after conversion from RawNode).
    pub nodes: Vec<Node>,

    /// The inputs of the graph.
    pub inputs: Vec<Argument>,

    /// The outputs of the graph.
    pub outputs: Vec<Argument>,

    /// Reference to GraphState to keep tensor data alive for .value() access
    /// This ensures Arguments can access tensor data via their data_id
    pub(crate) _graph_data: Option<std::rc::Rc<std::cell::RefCell<crate::graph_state::GraphState>>>,
}

/// Intermediate graph representation used during processing
///
/// This holds RawNode instances while type inference and processing is happening.
/// After processing is complete, it gets converted to OnnxGraph via convert_to_graph().
#[derive(Debug, Clone)]
pub struct OnnxGraphBuilder {
    /// The nodes of the graph (before conversion to final Node enum).
    pub nodes: Vec<RawNode>,

    /// The inputs of the graph.
    pub inputs: Vec<Argument>,

    /// The outputs of the graph.
    pub outputs: Vec<Argument>,

    /// Reference to GraphState to keep tensor data alive for .value() access
    pub(crate) _graph_data: Option<std::rc::Rc<std::cell::RefCell<crate::graph_state::GraphState>>>,
}

impl OnnxGraphBuilder {
    /// Convert this OnnxGraphBuilder to an OnnxGraph by converting all RawNodes to Nodes
    ///
    /// This recursively converts subgraphs for control flow nodes (If, Loop, Scan).
    pub fn convert_to_graph(mut self, opset: usize) -> OnnxGraph {
        let nodes = convert_builders_to_nodes(std::mem::take(&mut self.nodes), opset);

        OnnxGraph {
            nodes,
            inputs: self.inputs,
            outputs: self.outputs,
            _graph_data: self._graph_data,
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
                // Convert the subgraph's RawNodes to Nodes
                let nodes =
                    convert_builders_to_nodes(std::mem::take(&mut subgraph_builder.nodes), opset);

                // Create a new OnnxGraph with converted nodes
                *attr_value = AttributeValue::Graph(OnnxGraph {
                    nodes,
                    inputs: std::mem::take(&mut subgraph_builder.inputs),
                    outputs: std::mem::take(&mut subgraph_builder.outputs),
                    _graph_data: subgraph_builder._graph_data.clone(),
                });
            }
            AttributeValue::GraphBuilders(subgraph_builders) => {
                let converted_graphs: Vec<OnnxGraph> = subgraph_builders
                    .iter_mut()
                    .map(|subgraph_builder| {
                        let nodes = convert_builders_to_nodes(
                            std::mem::take(&mut subgraph_builder.nodes),
                            opset,
                        );

                        OnnxGraph {
                            nodes,
                            inputs: std::mem::take(&mut subgraph_builder.inputs),
                            outputs: std::mem::take(&mut subgraph_builder.outputs),
                            _graph_data: subgraph_builder._graph_data.clone(),
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
