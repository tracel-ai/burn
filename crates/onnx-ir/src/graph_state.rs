//! Graph state management for ONNX conversion
//!
//! This module manages the mutable state during ONNX to IR conversion:
//! - Node storage and ordering
//! - Graph inputs and outputs
//! - Name mapping between ONNX and IR names
//! - Tensor data storage

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::ir::{ArgType, Argument, NodeBuilder, NodeType, TensorData, TensorId};
use crate::proto_conversion::argument_from_initializer;
use crate::protos::{TensorProto, ValueInfoProto};

use super::tensor_store::TensorStore;

/// Shared registry for ensuring unique node names across sibling subgraphs
#[derive(Debug, Default)]
struct NameRegistryInner {
    seen_names: HashSet<String>,
    node_type_counters: HashMap<crate::ir::NodeType, usize>,
}

/// Wrapper for shared name registry
#[derive(Debug, Clone)]
pub struct NameRegistry(Rc<RefCell<NameRegistryInner>>);

impl Default for NameRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NameRegistry {
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(NameRegistryInner {
            seen_names: HashSet::new(),
            node_type_counters: HashMap::new(),
        })))
    }

    /// Generate a unique node name based on node type and counter
    pub fn generate_node_name(&self, node_type: &crate::ir::NodeType) -> String {
        let mut inner = self.0.borrow_mut();

        // Increment counter for this node type
        let counter = inner
            .node_type_counters
            .entry(node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);

        let name = format!("{}{}", node_type, counter).to_lowercase();

        // Also add to seen_names
        inner.seen_names.insert(name.clone());

        name
    }

    /// Set initial counter for a node type (used to account for initializers)
    #[allow(dead_code)]
    pub fn set_initial_counter(&self, node_type: &crate::ir::NodeType, count: usize) {
        let mut inner = self.0.borrow_mut();
        if count > 0 {
            inner.node_type_counters.insert(node_type.clone(), count);
        }
    }
}

/// Mutable state container for ONNX graph conversion
#[derive(Debug)]
pub struct GraphState {
    /// The nodes that have been processed, used to copy the outputs to a child node
    pub(super) processed_nodes: Vec<NodeBuilder>,
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
    /// Maps ONNX value names to their type info (from value_info)
    value_info_map: HashMap<String, ArgType>,
    /// Optional shared name registry for ensuring unique names across subgraphs
    name_registry: Option<NameRegistry>,
}

impl GraphState {
    /// Create new GraphState from ONNX proto structures
    pub(crate) fn new(
        inputs: &[ValueInfoProto],
        outputs: &[ValueInfoProto],
        initializers: &[TensorProto],
        value_infos: &[ValueInfoProto],
    ) -> Self {
        Self::new_with_registry(inputs, outputs, initializers, value_infos, None)
    }

    /// Create new GraphState with optional shared name registry
    pub(crate) fn new_with_registry(
        inputs: &[ValueInfoProto],
        outputs: &[ValueInfoProto],
        initializers: &[TensorProto],
        value_infos: &[ValueInfoProto],
        name_registry: Option<NameRegistry>,
    ) -> Self {
        let mut tensor_store = TensorStore::new();
        let mut graph_input_map = HashMap::new();
        let mut node_output_map = HashMap::new();
        let mut value_info_map = HashMap::new();

        // Convert all initializers to Constant nodes
        let processed_nodes =
            process_initializers(initializers, &mut tensor_store, name_registry.as_ref());

        // Map initializer names to their constant node outputs
        for (i, initializer) in initializers.iter().enumerate() {
            node_output_map.insert(initializer.name.clone(), (i, 0));
        }

        // Store value_info for intermediate values
        for value_info in value_infos {
            if let Ok(arg) = Argument::try_from(value_info.clone()) {
                value_info_map.insert(value_info.name.clone(), arg.ty);
            }
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
                // Preserve the original ONNX input name for better generated code usability
                graph_input_map.insert(x.name.clone(), graph_input_map.len());

                let arg = Argument::try_from(x.clone()).unwrap();
                // arg.name is already set from x.name via try_from
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
            value_info_map,
            name_registry,
        }
    }

    /// Get the value of an input from the original input name. Used during proto conversion
    pub(crate) fn init_in(&self, proto_str: &str) -> Argument {
        // Sanitize the ONNX name to match our internal sanitized names
        let sanitized = crate::proto_conversion::sanitize_name(proto_str);

        // Check graph inputs (uses original ONNX names as keys)
        if let Some(&i) = self.graph_input_map.get(proto_str) {
            self.inputs[i].clone()
        }
        // Check node outputs (uses sanitized names as keys)
        else if let Some(&(node_idx, output_idx)) = self.node_output_map.get(&sanitized) {
            self.processed_nodes[node_idx].outputs[output_idx].clone()
        }
        // Also check with original name for initializers (they use original names as keys)
        else if let Some(&(node_idx, output_idx)) = self.node_output_map.get(proto_str) {
            self.processed_nodes[node_idx].outputs[output_idx].clone()
        } else {
            log::warn!("Input {proto_str} not found, should only happen when peeking");
            Argument::new(sanitized)
        }
    }

    /// Add a node (maps outputs, renames outputs)
    pub(super) fn add_node(&mut self, mut node: NodeBuilder) {
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

    /// Get reference to node output map (maps original ONNX names to node outputs)
    pub(crate) fn node_output_map(&self) -> &HashMap<String, (usize, usize)> {
        &self.node_output_map
    }

    /// Get reference to the name registry (if available)
    pub(crate) fn name_registry(&self) -> Option<&NameRegistry> {
        self.name_registry.as_ref()
    }

    /// Consume and return (nodes, inputs, outputs)
    pub(super) fn consume(self) -> (Vec<NodeBuilder>, Vec<Argument>, Vec<Argument>) {
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

    /// Get type from value_info for intermediate values
    pub(crate) fn get_value_info_type(&self, name: &str) -> Option<&crate::ir::ArgType> {
        self.value_info_map.get(name)
    }

    /// Register a test constant in GraphState
    #[cfg(test)]
    pub(crate) fn register_test_constant(&mut self, name: String, tensor_data: TensorData) {
        let (constant_node, _) = create_test_constant(name, tensor_data, &mut self.tensor_store);
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
            .and_then(|input| match input.value_source {
                crate::ir::ValueSource::Static(data_id) => Some(data_id),
                _ => None,
            })
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
) -> NodeBuilder {
    NodeBuilder {
        node_type: NodeType::Constant,
        name: node_name,
        inputs: vec![Argument {
            name: String::new(),
            ty: ty.clone(),
            value_source: crate::ir::ValueSource::Static(data_id),
            value_store: None,
        }],
        outputs: vec![Argument {
            name: output_name,
            ty,
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
    name_registry: Option<&NameRegistry>,
) -> Vec<NodeBuilder> {
    initializers
        .iter()
        .enumerate()
        .map(|(idx, initializer)| {
            let (_arg, data) = argument_from_initializer(initializer);

            // Allocate ID and store tensor data
            let data_id = tensor_store.store(data);

            // Generate unique name using registry if available
            let const_name = if let Some(registry) = name_registry {
                registry.generate_node_name(&crate::ir::NodeType::Constant)
            } else {
                format!("constant{}", idx + 1)
            };
            let output_name = format!("{}_out1", const_name);

            create_constant_node(const_name, output_name, _arg.ty.clone(), data_id)
        })
        .collect()
}

#[cfg(test)]
/// Create a test constant node with tensor data
fn create_test_constant(
    name: String,
    tensor_data: TensorData,
    tensor_store: &mut TensorStore,
) -> (NodeBuilder, usize) {
    use crate::ir::TensorDataExt;
    let elem_type = tensor_data.elem_type();
    let shape = tensor_data.shape.to_vec();

    let ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
        dtype: elem_type,
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
