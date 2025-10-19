//! Graph data management for ONNX conversion
//!
//! This module handles the intermediate state during ONNX graph conversion,
//! including node storage, input/output mapping, and constant tracking.

use std::collections::{HashMap, HashSet};

use crate::ir::{ArgType, Argument, Node, NodeType, TensorData, TensorId};
use crate::protos::{TensorProto, ValueInfoProto};

/// Represents where an input comes from - either a graph input or a node output
#[derive(Debug, Clone)]
pub(crate) enum IOEntry {
    /// Input from a graph input at the given index
    In(usize),
    /// Input from a node output (node_index, output_index)
    Node(usize, usize),
}

/// Manages intermediate state during ONNX graph conversion
#[derive(Debug)]
pub struct GraphData {
    /// The nodes that have been processed, used to copy the outputs to a child node
    pub(super) processed_nodes: Vec<Node>,
    /// The inputs of the graph
    inputs: Vec<Argument>,
    /// The outputs of the graph
    outputs: Vec<Argument>,
    /// The initializers of the graph
    pub(crate) initializers: HashMap<String, Argument>,
    /// Maps the original input name to a graph input
    input_name_map: HashMap<String, IOEntry>,
    /// Maps the updated input name to the original input name. Required to check if the input is an initializer
    input_key_map: HashMap<String, String>,
    /// Tracks which inputs have been used by nodes
    passed_inputs: HashSet<usize>,
    /// Maps constant output names to node indices
    pub(super) constant_nodes: HashMap<String, usize>,

    /// Central tensor data store: ID -> TensorData
    /// All constant/static tensor data is stored here with unique IDs
    pub(super) tensor_data: HashMap<TensorId, TensorData>,
    /// Next available tensor ID
    pub(super) next_tensor_id: TensorId,
}

impl GraphData {
    /// Create a Constant node with Static input
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

    /// Create new GraphData from ONNX proto structures
    pub(crate) fn new(
        inputs: &[ValueInfoProto],
        outputs: &[ValueInfoProto],
        initializers: &[TensorProto],
    ) -> Self {
        let mut input_name_map = HashMap::new();
        let mut input_key_map = HashMap::new();
        let mut processed_nodes = Vec::new();
        let mut constant_nodes = HashMap::new();
        let mut tensor_data = HashMap::new();
        let mut next_tensor_id = 0;

        // Convert all initializers to Constant nodes
        let mut constants_map: HashMap<String, Argument> = HashMap::new();
        for initializer in initializers.iter() {
            let (arg, data) = Argument::from_initializer(initializer);

            // Allocate ID and store tensor data
            let data_id = next_tensor_id;
            next_tensor_id += 1;
            tensor_data.insert(data_id, data);

            let idx = processed_nodes.len();
            let const_name = format!("constant{}", idx + 1);
            let output_name = format!("{}_out1", const_name);

            let constant_node = Self::create_constant_node(
                const_name,
                output_name.clone(),
                arg.ty.clone(),
                data_id,
            );

            constant_nodes.insert(output_name.clone(), idx);
            input_name_map.insert(initializer.name.clone(), IOEntry::Node(idx, 0));
            constants_map.insert(initializer.name.clone(), arg);
            processed_nodes.push(constant_node);
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

                // Only add to input_name_map if not already mapped to a constant
                if !input_name_map.contains_key(&x.name) {
                    input_name_map.insert(x.name.clone(), IOEntry::In(i));
                }
                input_key_map.insert(in_name.clone(), x.name.clone());

                let mut arg = Argument::try_from(x.clone()).unwrap();
                if constants_map.contains_key(&x.name) {
                    log::warn!(
                        "Input {} is also an initializer. Initializer as default values are currently not supported",
                        x.name
                    );
                }

                arg.name = in_name;
                arg
            })
            .collect::<Vec<Argument>>();

        Self {
            inputs,
            outputs,
            initializers: constants_map,
            processed_nodes,
            input_name_map,
            input_key_map,
            passed_inputs: HashSet::new(),
            constant_nodes,
            tensor_data,
            next_tensor_id,
        }
    }

    /// Get the value of an input from the original input name. Used during proto conversion
    pub(crate) fn init_in(&self, proto_str: &str) -> Argument {
        match self.input_name_map.get(proto_str) {
            None => {
                //NOTE: if initializers are guaranteed to be unique, (I think they are
                //need to confirm) then we could pop the initializer from the map
                if let Some(init_arg) = self.initializers.get(proto_str) {
                    init_arg.clone()
                } else {
                    log::warn!("Input {proto_str} not found, should only happen when peeking");
                    Argument::new(proto_str.to_string())
                }
            }
            Some(IOEntry::In(i)) => self.inputs[*i].clone(),
            Some(IOEntry::Node(i, j)) => self.processed_nodes[*i].outputs[*j].clone(),
        }
    }

    /// Mark the graph_inputs to a node as passed, unless they are also initializers
    fn mark_input_passed(&mut self, node: &Node) {
        // we have to double map the inputs because the input might be replaced by an initializer
        node.inputs.iter().for_each(|node_input| {
            if let Some(old_input_name) = self.input_key_map.get(&node_input.name)
                && !self.initializers.contains_key(old_input_name)
            {
                match self.input_name_map.get(old_input_name) {
                    Some(IOEntry::In(i)) => {
                        self.passed_inputs.insert(*i);
                    }
                    _ => {
                        panic!("Should not happen, please report this error");
                    }
                }
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
        let mut out_count = 1;
        for output in node.outputs.iter_mut() {
            self.input_name_map.insert(
                output.name.clone(),
                IOEntry::Node(self.processed_nodes.len(), out_count - 1),
            );
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
        let passed_inputs = self.passed_inputs;
        let mut filtered_inputs = Vec::new();
        for (i, input) in self.inputs.into_iter().enumerate() {
            if passed_inputs.contains(&i) {
                filtered_inputs.push(input);
            }
        }
        self.inputs = filtered_inputs;
        let outputs = self
            .outputs
            .into_iter()
            .filter_map(|x| match self.input_name_map.get(&x.name) {
                Some(IOEntry::Node(i, j)) => Some(self.processed_nodes[*i].outputs[*j].clone()),
                Some(IOEntry::In(i)) => {
                    // Output maps directly to an input (e.g., when Identity nodes are removed)
                    Some(self.inputs[*i].clone())
                }
                _ => None,
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
        use crate::ir::TensorData;

        let elem_type = match &data {
            crate::ir::Data::Bool(_) | crate::ir::Data::Bools(_) => crate::ElementType::Bool,
            crate::ir::Data::Float16(_) | crate::ir::Data::Float16s(_) => {
                crate::ElementType::Float16
            }
            crate::ir::Data::Float32(_) | crate::ir::Data::Float32s(_) => {
                crate::ElementType::Float32
            }
            crate::ir::Data::Float64(_) | crate::ir::Data::Float64s(_) => {
                crate::ElementType::Float64
            }
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

        let data_id = self.store_tensor_data(TensorData { data, shape });

        let output_name = format!("{}_const_out", name);
        let const_node_name = format!("{}_const", name);
        let constant_node = Self::create_constant_node(const_node_name, output_name, ty, data_id);

        let node_idx = self.processed_nodes.len();
        self.constant_nodes.insert(name, node_idx);
        self.processed_nodes.push(constant_node);
    }

    /// Allocate a new tensor ID and store data in central store
    /// Returns the allocated ID
    pub(crate) fn store_tensor_data(&mut self, data: TensorData) -> TensorId {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        self.tensor_data.insert(id, data);
        id
    }

    /// Get tensor data by ID from central store
    pub(crate) fn get_tensor_data(&self, id: TensorId) -> Option<&TensorData> {
        self.tensor_data.get(&id)
    }

    /// Get mutable tensor data by ID from central store
    pub(crate) fn get_tensor_data_mut(&mut self, id: TensorId) -> Option<&mut TensorData> {
        self.tensor_data.get_mut(&id)
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
