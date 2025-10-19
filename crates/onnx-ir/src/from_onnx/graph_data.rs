//! Graph data management for ONNX conversion
//!
//! This module orchestrates the ONNX to IR conversion process by managing:
//! - Node storage and ordering
//! - Graph inputs and outputs
//! - Name mapping between ONNX and IR
//! - Delegation to specialized modules (TensorStore, ConstantBuilder)

use std::collections::{HashMap, HashSet};

use crate::ir::{ArgType, Argument, Node, NodeType, TensorData, TensorId};
use crate::protos::{TensorProto, ValueInfoProto};

use super::constant_builder;
use super::tensor_store::TensorStore;

/// Manages intermediate state during ONNX graph conversion
#[derive(Debug)]
pub struct GraphData {
    /// The nodes that have been processed, used to copy the outputs to a child node
    pub(super) processed_nodes: Vec<Node>,
    /// The inputs of the graph
    inputs: Vec<Argument>,
    /// The outputs of the graph
    outputs: Vec<Argument>,
    /// Maps ONNX names to graph input indices
    graph_input_map: HashMap<String, usize>,
    /// Maps ONNX names to node outputs (node_index, output_index)
    node_output_map: HashMap<String, (usize, usize)>,
    /// Maps IR input names (input1, input2) back to ONNX names
    input_key_map: HashMap<String, String>,
    /// Tracks which graph inputs have been used by nodes
    passed_inputs: HashSet<usize>,
    /// Maps constant output names to node indices
    pub(super) constant_nodes: HashMap<String, usize>,
    /// Central tensor data store
    pub(super) tensor_store: TensorStore,
}

impl GraphData {
    /// Create new GraphData from ONNX proto structures
    pub(crate) fn new(
        inputs: &[ValueInfoProto],
        outputs: &[ValueInfoProto],
        initializers: &[TensorProto],
    ) -> Self {
        let mut tensor_store = TensorStore::new();
        let mut graph_input_map = HashMap::new();
        let mut node_output_map = HashMap::new();
        let mut input_key_map = HashMap::new();

        // Convert all initializers to Constant nodes
        let processed_constants =
            constant_builder::process_initializers(initializers, &mut tensor_store);
        let processed_nodes = processed_constants.nodes;
        let constant_nodes = processed_constants.constant_nodes;

        // Map initializer names to their constant node outputs
        for (i, initializer) in initializers.iter().enumerate() {
            node_output_map.insert(initializer.name.clone(), (i, 0));
        }

        let outputs = outputs
            .iter()
            .map(|x| Argument::try_from(x.clone()).unwrap())
            .collect::<Vec<Argument>>();

        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let in_name = format!("input{}", i + 1);

                // Only add to graph_input_map if not already an initializer
                if !node_output_map.contains_key(&x.name) {
                    graph_input_map.insert(x.name.clone(), i);
                }
                input_key_map.insert(in_name.clone(), x.name.clone());

                let mut arg = Argument::try_from(x.clone()).unwrap();
                arg.name = in_name;
                arg
            })
            .collect::<Vec<Argument>>();

        Self {
            inputs,
            outputs,
            processed_nodes,
            graph_input_map,
            node_output_map,
            input_key_map,
            passed_inputs: HashSet::new(),
            constant_nodes,
            tensor_store,
        }
    }

    /// Get the value of an input from the original input name. Used during proto conversion
    pub(crate) fn init_in(&self, proto_str: &str) -> Argument {
        if let Some(&i) = self.graph_input_map.get(proto_str) {
            self.inputs[i].clone()
        } else if let Some(&(node_idx, output_idx)) = self.node_output_map.get(proto_str) {
            self.processed_nodes[node_idx].outputs[output_idx].clone()
        } else {
            log::warn!("Input {proto_str} not found, should only happen when peeking");
            Argument::new(proto_str.to_string())
        }
    }

    /// Mark the graph_inputs to a node as passed
    fn mark_input_passed(&mut self, node: &Node) {
        node.inputs.iter().for_each(|node_input| {
            if let Some(onnx_name) = self.input_key_map.get(&node_input.name)
                && let Some(&i) = self.graph_input_map.get(onnx_name)
            {
                self.passed_inputs.insert(i);
            }
        });
    }

    /// Add a node to the graph
    ///
    /// This function does three things:
    ///     1. marks the inputs as passed
    ///     2. maps the old output names to the node output
    ///     3. renames the node output
    pub(super) fn add_node(&mut self, mut node: Node) {
        log::debug!("Adding node {:?}", &node.name);
        self.mark_input_passed(&node);
        let node_idx = self.processed_nodes.len();
        let mut out_count = 1;
        for output in node.outputs.iter_mut() {
            self.node_output_map
                .insert(output.name.clone(), (node_idx, out_count - 1));
            output.name = format!("{}_out{}", node.name, out_count);
            out_count += 1;
        }

        // Register Constant nodes so they can be found during lifting
        if node.node_type == NodeType::Constant {
            for output in &node.outputs {
                self.constant_nodes
                    .insert(output.name.clone(), self.processed_nodes.len());
            }
        }

        self.processed_nodes.push(node);
    }

    /// Consumes the graph data and returns the processed nodes, filtered inputs, and outputs
    pub(super) fn consume(mut self) -> (Vec<Node>, Vec<Argument>, Vec<Argument>) {
        let mut filtered_inputs = Vec::new();
        for (i, input) in self.inputs.into_iter().enumerate() {
            if self.passed_inputs.contains(&i) {
                filtered_inputs.push(input);
            }
        }
        self.inputs = filtered_inputs;
        let outputs = self
            .outputs
            .into_iter()
            .filter_map(|x| {
                if let Some(&(node_idx, output_idx)) = self.node_output_map.get(&x.name) {
                    Some(self.processed_nodes[node_idx].outputs[output_idx].clone())
                } else if let Some(&i) = self.graph_input_map.get(&x.name) {
                    // Output maps directly to an input (e.g., when Identity nodes are removed)
                    Some(self.inputs[i].clone())
                } else {
                    None
                }
            })
            .collect();
        (self.processed_nodes, self.inputs, outputs)
    }

    /// Check if a value is available in constant nodes
    pub(crate) fn has_value(&self, name: &str) -> bool {
        self.constant_nodes.contains_key(name)
    }

    /// Get the type of a graph output by name
    pub(crate) fn get_output_type(&self, name: &str) -> Option<&crate::ir::ArgType> {
        self.outputs
            .iter()
            .find(|out| out.name == name)
            .map(|out| &out.ty)
    }

    /// Register a test constant in GraphData
    #[cfg(test)]
    pub(crate) fn register_test_constant(
        &mut self,
        name: String,
        data: crate::ir::Data,
        shape: Vec<usize>,
    ) {
        let (constant_node, _) = constant_builder::create_test_constant(
            name.clone(),
            data,
            shape,
            &mut self.tensor_store,
        );

        let node_idx = self.processed_nodes.len();
        self.constant_nodes.insert(name, node_idx);
        self.processed_nodes.push(constant_node);
    }

    /// Allocate a new tensor ID and store data in central store
    /// Returns the allocated ID
    pub(crate) fn store_tensor_data(&mut self, data: TensorData) -> TensorId {
        self.tensor_store.store(data)
    }

    /// Get tensor data by ID from central store
    pub(crate) fn get_tensor_data(&self, id: TensorId) -> Option<&TensorData> {
        self.tensor_store.get(id)
    }

    /// Get mutable tensor data by ID from central store
    pub(crate) fn get_tensor_data_mut(&mut self, id: TensorId) -> Option<&mut TensorData> {
        self.tensor_store.get_mut(id)
    }

    /// Set the expected type for an argument
    ///
    /// This is called by Argument::should_be() to record type expectations.
    /// Note: Currently unused - this is a stub for future type inference enhancements.
    pub(crate) fn set_expected_type(&mut self, _arg_name: String, _expected_ty: ArgType) {
        // Stub method - expected_types field was removed since it's never read
        // Kept for compatibility with Argument::should_be() calls
    }

    /// Get the data_id for a constant by output name
    ///
    /// This is used by Argument::to_static() to look up the data_id of a constant node
    pub(crate) fn get_constant_data_id_by_output(&self, output_name: &str) -> Option<TensorId> {
        self.constant_nodes
            .get(output_name)
            .and_then(|&idx| self.processed_nodes.get(idx))
            .and_then(|node| node.inputs.first())
            .and_then(|input| input.data_id)
    }

    /// Alias for get_constant_data_id_by_output (for test utilities)
    #[cfg(test)]
    pub(crate) fn get_constant_data_id(&self, name: &str) -> Option<TensorId> {
        self.get_constant_data_id_by_output(name)
    }
}
