//! ONNX graph representation
//!
//! This module contains the OnnxGraph struct which represents a complete
//! ONNX computational graph with nodes, inputs, and outputs.

use super::argument::Argument;
use super::node::NodeBuilder;

/// ONNX graph representation
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// The nodes of the graph.
    pub nodes: Vec<NodeBuilder>,

    /// The inputs of the graph.
    pub inputs: Vec<Argument>,

    /// The outputs of the graph.
    pub outputs: Vec<Argument>,

    /// Reference to GraphState to keep tensor data alive for .value() access
    /// This ensures Arguments can access tensor data via their data_id
    pub(crate) _graph_data: Option<std::rc::Rc<std::cell::RefCell<crate::graph_state::GraphState>>>,
}
