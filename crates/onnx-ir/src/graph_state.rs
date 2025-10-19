//! Graph state management for ONNX conversion
//!
//! This module manages the mutable state during ONNX to IR conversion:
//! - Node storage and ordering
//! - Graph inputs and outputs
//! - Name mapping between ONNX and IR names
//! - Tensor data storage

use std::collections::{HashMap, HashSet};

use crate::ir::{ArgType, Argument, Node, NodeType, TensorData, TensorId};
use crate::proto_conversion::argument_from_initializer;
use crate::protos::{TensorProto, ValueInfoProto};

use super::tensor_store::TensorStore;

/// Result of processing initializers
struct ProcessedConstants {
    /// Created Constant nodes
    nodes: Vec<Node>,
    /// Map of constant output names to node indices
    constant_nodes: HashMap<String, usize>,
}

/// Create a Constant node with Static input and Constant output
fn create_constant_node(
    node_name: String,
    output_name: String,
    ty: ArgType,
    data_id: TensorId,
) -> Node {
    Node {
        node_type: NodeType::Constant,
        name: node_name,
        inputs: vec![Argument {
            name: String::new(),
            ty: ty.clone(),
            data_id: Some(data_id),
            value_source: crate::ir::ValueSource::Static,
            value_store: None,
        }],
        outputs: vec![Argument {
            name: output_name,
            ty,
            data_id: None,
            value_source: crate::ir::ValueSource::Constant,
            value_store: None,
        }],
        attrs: HashMap::new(),
        config: None,
    }
}

/// Convert ONNX initializers to Constant nodes, store in tensor store
fn process_initializers(
    initializers: &[TensorProto],
    tensor_store: &mut TensorStore,
) -> ProcessedConstants {
    let mut nodes = Vec::new();
    let mut constant_nodes = HashMap::new();

    for initializer in initializers.iter() {
        let (_arg, data) = argument_from_initializer(initializer);

        // Allocate ID and store tensor data
        let data_id = tensor_store.store(data);

        let idx = nodes.len();
        let const_name = format!("constant{}", idx + 1);
        let output_name = format!("{}_out1", const_name);

        let constant_node =
            create_constant_node(const_name, output_name.clone(), _arg.ty.clone(), data_id);

        constant_nodes.insert(output_name, idx);
        nodes.push(constant_node);
    }

    ProcessedConstants {
        nodes,
        constant_nodes,
    }
}

#[cfg(test)]
/// Create a test constant node with tensor data
fn create_test_constant(
    name: String,
    data: crate::ir::Data,
    shape: Vec<usize>,
    tensor_store: &mut TensorStore,
) -> (Node, usize) {
    use crate::ir::TensorData;

    let elem_type = match &data {
        crate::ir::Data::Bool(_) | crate::ir::Data::Bools(_) => crate::ElementType::Bool,
        crate::ir::Data::Float16(_) | crate::ir::Data::Float16s(_) => crate::ElementType::Float16,
        crate::ir::Data::Float32(_) | crate::ir::Data::Float32s(_) => crate::ElementType::Float32,
        crate::ir::Data::Float64(_) | crate::ir::Data::Float64s(_) => crate::ElementType::Float64,
        crate::ir::Data::Uint16(_) | crate::ir::Data::Uint16s(_) => crate::ElementType::Uint16,
        crate::ir::Data::Uint8(_) | crate::ir::Data::Uint8s(_) => crate::ElementType::Uint8,
        crate::ir::Data::Int8(_) | crate::ir::Data::Int8s(_) => crate::ElementType::Int8,
        crate::ir::Data::Int32(_) | crate::ir::Data::Int32s(_) => crate::ElementType::Int32,
        crate::ir::Data::Int64(_) | crate::ir::Data::Int64s(_) => crate::ElementType::Int64,
        crate::ir::Data::String(_) | crate::ir::Data::Strings(_) => crate::ElementType::String,
    };

    let ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
        elem_type,
        rank: shape.len(),
        static_shape: Some(shape.clone()),
    });

    let data_id = tensor_store.store(TensorData { data, shape });

    let output_name = format!("{}_const_out", name);
    let const_node_name = format!("{}_const", name);
    let constant_node = create_constant_node(const_node_name, output_name, ty, data_id);

    // Return node and a placeholder index (caller will assign proper index)
    (constant_node, 0)
}

/// Mutable state container for ONNX graph conversion
#[derive(Debug)]
pub struct GraphState {
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

impl GraphState {
    /// Create new GraphState from ONNX proto structures
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
        let processed_constants = process_initializers(initializers, &mut tensor_store);
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

    /// Add a node (marks inputs as passed, maps outputs, renames outputs)
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

    /// Consume and return (nodes, filtered inputs, outputs)
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

    /// Register a test constant in GraphState
    #[cfg(test)]
    pub(crate) fn register_test_constant(
        &mut self,
        name: String,
        data: crate::ir::Data,
        shape: Vec<usize>,
    ) {
        let (constant_node, _) =
            create_test_constant(name.clone(), data, shape, &mut self.tensor_store);

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

    /// Get data_id for a constant by output name
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
