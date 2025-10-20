//! Graph state management for ONNX conversion
//!
//! This module manages the mutable state during ONNX to IR conversion:
//! - Node storage and ordering
//! - Graph inputs and outputs
//! - Name mapping between ONNX and IR names
//! - Tensor data storage

use std::collections::HashMap;

use crate::ir::{ArgType, Argument, Node, NodeType, TensorData, TensorId};
use crate::proto_conversion::argument_from_initializer;
use crate::protos::{TensorProto, ValueInfoProto};

use super::tensor_store::TensorStore;

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

        // Convert all initializers to Constant nodes
        let processed_nodes = process_initializers(initializers, &mut tensor_store);

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
            .filter_map(|x| {
                // Skip inputs that are initializers (they become constant nodes instead)
                if node_output_map.contains_key(&x.name) {
                    return None;
                }

                // Only real graph inputs get added
                let in_name = format!("input{}", graph_input_map.len() + 1);
                graph_input_map.insert(x.name.clone(), graph_input_map.len());

                let mut arg = Argument::try_from(x.clone()).unwrap();
                arg.name = in_name;
                Some(arg)
            })
            .collect::<Vec<Argument>>();

        Self {
            inputs,
            outputs,
            processed_nodes,
            graph_input_map,
            node_output_map,
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

    /// Add a node (maps outputs, renames outputs)
    pub(super) fn add_node(&mut self, mut node: Node) {
        log::debug!("Adding node {:?}", &node.name);
        let node_idx = self.processed_nodes.len();
        let mut out_count = 1;
        for output in node.outputs.iter_mut() {
            self.node_output_map
                .insert(output.name.clone(), (node_idx, out_count - 1));
            output.name = format!("{}_out{}", node.name, out_count);
            out_count += 1;
        }

        self.processed_nodes.push(node);
    }

    /// Consume and return (nodes, inputs, outputs)
    pub(super) fn consume(self) -> (Vec<Node>, Vec<Argument>, Vec<Argument>) {
        let outputs = self
            .outputs
            .into_iter()
            .filter_map(|x| {
                if let Some(&(node_idx, output_idx)) = self.node_output_map.get(&x.name) {
                    Some(self.processed_nodes[node_idx].outputs[output_idx].clone())
                } else if let Some(&i) = self.graph_input_map.get(&x.name) {
                    // Output references a graph input directly (passthrough)
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
        self.processed_nodes.iter().any(|node| {
            node.node_type == NodeType::Constant && node.outputs.iter().any(|o| o.name == name)
        })
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
        let (constant_node, _) = create_test_constant(name, data, shape, &mut self.tensor_store);
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
        self.processed_nodes
            .iter()
            .find(|node| {
                node.node_type == NodeType::Constant
                    && node.outputs.iter().any(|o| o.name == output_name)
            })
            .and_then(|node| node.inputs.first())
            .and_then(|input| input.data_id)
    }

    /// Alias for get_constant_data_id_by_output (for test utilities)
    #[cfg(test)]
    pub(crate) fn get_constant_data_id(&self, name: &str) -> Option<TensorId> {
        self.get_constant_data_id_by_output(name)
    }
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
fn process_initializers(initializers: &[TensorProto], tensor_store: &mut TensorStore) -> Vec<Node> {
    initializers
        .iter()
        .enumerate()
        .map(|(idx, initializer)| {
            let (_arg, data) = argument_from_initializer(initializer);

            // Allocate ID and store tensor data
            let data_id = tensor_store.store(data);

            let const_name = format!("constant{}", idx + 1);
            let output_name = format!("{}_out1", const_name);

            create_constant_node(const_name, output_name, _arg.ty.clone(), data_id)
        })
        .collect()
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

    // Convert Data enum to TensorData using burn-tensor
    let tensor_data = match data {
        crate::ir::Data::Bool(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Bools(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Float16(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Float16s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Float32(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Float32s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Float64(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Float64s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Uint16(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Uint16s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Uint8(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Uint8s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Int8(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Int8s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Int32(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Int32s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::Int64(v) => TensorData::new(vec![v], shape.clone()),
        crate::ir::Data::Int64s(v) => TensorData::new(v, shape.clone()),
        crate::ir::Data::String(_) | crate::ir::Data::Strings(_) => {
            panic!("String tensors not supported in burn-tensor")
        }
    };

    let elem_type = tensor_data.elem_type();

    let ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
        elem_type,
        rank: shape.len(),
        static_shape: Some(shape.clone()),
    });

    let data_id = tensor_store.store(tensor_data);

    // Use name directly as output name for test lookups (no _const_out suffix)
    let const_node_name = format!("{}_const", name);
    let constant_node = create_constant_node(const_node_name, name, ty, data_id);

    // Return node and a placeholder index (caller will assign proper index)
    (constant_node, 0)
}
