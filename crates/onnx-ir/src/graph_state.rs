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

use crate::ir::{ArgType, Argument, DataId, NodeType, RawNode, TensorData};
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
    pub(super) processed_nodes: Vec<RawNode>,
    /// The inputs of the graph
    inputs: Vec<Argument>,
    /// The outputs of the graph
    outputs: Vec<Argument>,
    /// Maps ONNX names to graph input indices
    graph_input_map: HashMap<String, usize>,
    /// Maps ONNX names to node outputs (node_index, output_index)
    node_output_map: HashMap<String, (usize, usize)>,
    /// Central tensor data store (shared via Rc for Arguments to reference)
    pub(super) tensor_store: Rc<TensorStore>,
    /// Maps constant output names to their data IDs (shared via Rc)
    /// Updated whenever a Constant node is created
    constant_map: Rc<HashMap<String, DataId>>,
    /// Maps ONNX value names to their type info (from value_info)
    value_info_map: HashMap<String, ArgType>,
    /// Optional shared name registry for ensuring unique names across subgraphs
    name_registry: Option<NameRegistry>,
}

impl GraphState {
    /// Create new GraphState from ONNX proto structures
    #[doc(hidden)]
    pub fn new(
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
        let mut constant_map = HashMap::new();
        let mut graph_input_map = HashMap::new();
        let mut node_output_map = HashMap::new();
        let mut value_info_map = HashMap::new();

        // Convert all initializers to Constant nodes
        let processed_nodes = process_initializers(
            initializers,
            &mut tensor_store,
            &mut constant_map,
            name_registry.as_ref(),
        );

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
            tensor_store: Rc::new(tensor_store),
            constant_map: Rc::new(constant_map),
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
            Argument::from_name(sanitized)
        }
    }

    /// Add a node (maps outputs, renames outputs)
    ///
    /// For Constant nodes, also registers the output name → data_id mapping
    /// in constant_map for fast lookup during lift_constants.
    pub(super) fn add_node(&mut self, mut node: RawNode) {
        let node_idx = self.processed_nodes.len();
        let mut out_count = 1;

        // Get data_id for Constant nodes (from their Static input)
        let constant_data_id = if node.node_type == NodeType::Constant {
            node.inputs
                .first()
                .and_then(|input| match input.value_source {
                    crate::ir::ValueSource::Static(data_id) => Some(data_id),
                    _ => None,
                })
        } else {
            None
        };

        for output in node.outputs.iter_mut() {
            self.node_output_map
                .insert(output.name.clone(), (node_idx, out_count - 1));
            output.name = format!("{}_out{}", node.name, out_count);

            // Register constant output name → data_id for fast lookup
            if let Some(data_id) = constant_data_id {
                Rc::make_mut(&mut self.constant_map).insert(output.name.clone(), data_id);
            }

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
    pub(super) fn consume(self) -> (Vec<RawNode>, Vec<Argument>, Vec<Argument>) {
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
    #[doc(hidden)]
    #[allow(dead_code)] // Used by tests in node/ modules
    pub fn register_test_constant(&mut self, name: String, tensor_data: TensorData) {
        let (constant_node, data_id) = create_test_constant(
            name.clone(),
            tensor_data,
            Rc::make_mut(&mut self.tensor_store),
        );
        // Register in constant_map (output name is the same as input name for test constants)
        Rc::make_mut(&mut self.constant_map).insert(name, data_id);
        self.processed_nodes.push(constant_node);
    }

    /// Register a constant output name to its data_id
    /// Called when a Constant node is created during node conversion
    #[allow(dead_code)] // Available for future use or external consumers
    pub(crate) fn register_constant(&mut self, output_name: String, data_id: DataId) {
        Rc::make_mut(&mut self.constant_map).insert(output_name, data_id);
    }

    /// Allocate a new tensor ID and store data in central store
    /// Returns the allocated ID
    pub(crate) fn store_tensor_data(&mut self, data: TensorData) -> DataId {
        Rc::make_mut(&mut self.tensor_store).store(data)
    }

    /// Get tensor data by ID from central store
    #[allow(dead_code)] // May be useful for downstream consumers
    pub(crate) fn get_tensor_data(&self, id: DataId) -> Option<&TensorData> {
        self.tensor_store.get(id)
    }

    /// Get data_id for a constant by output name (O(1) lookup via constant_map)
    pub(crate) fn get_constant_data_id_by_output(&self, output_name: &str) -> Option<DataId> {
        // First try the constant_map (O(1) lookup)
        if let Some(&data_id) = self.constant_map.get(output_name) {
            return Some(data_id);
        }

        // Fallback: scan processed_nodes (for backwards compatibility during transition)
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
    #[doc(hidden)]
    #[allow(dead_code)] // Used by tests in node/ modules
    pub fn get_constant_data_id(&self, name: &str) -> Option<DataId> {
        self.get_constant_data_id_by_output(name)
    }

    /// Get reference to the constant_map
    #[allow(dead_code)] // May be useful for downstream consumers
    pub(crate) fn constant_map(&self) -> &HashMap<String, DataId> {
        &self.constant_map
    }

    /// Get Rc reference to the constant_map (for cheap preservation across state reset)
    pub(crate) fn constant_map_rc(&self) -> Rc<HashMap<String, DataId>> {
        self.constant_map.clone()
    }

    /// Restore tensor_store and constant_map from Rc references (no data copying)
    /// Used in post-processing to preserve stores across GraphState reset
    pub(crate) fn restore_stores(
        &mut self,
        tensor_store: Rc<TensorStore>,
        constant_map: Rc<HashMap<String, DataId>>,
    ) {
        self.tensor_store = tensor_store;
        self.constant_map = constant_map;
    }

    /// Build a ValueStore from the current state
    /// Returns cloned Rc references to the tensor_store and constant_map
    pub(crate) fn build_value_store(&self) -> crate::tensor_store::ValueStore {
        use crate::tensor_store::ValueStore;

        ValueStore::new(self.tensor_store.clone(), self.constant_map.clone())
    }
}

/// Create a Constant node with Static input and Constant output
fn create_constant_node(
    node_name: String,
    output_name: String,
    ty: ArgType,
    data_id: DataId,
) -> RawNode {
    RawNode {
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
    }
}

/// Convert ONNX initializers to Constant nodes, store in tensor store
fn process_initializers(
    initializers: &[TensorProto],
    tensor_store: &mut TensorStore,
    constant_map: &mut HashMap<String, DataId>,
    name_registry: Option<&NameRegistry>,
) -> Vec<RawNode> {
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

            // Register in constant_map for fast lookup
            constant_map.insert(output_name.clone(), data_id);

            create_constant_node(const_name, output_name, _arg.ty.clone(), data_id)
        })
        .collect()
}

/// Create a test constant node with tensor data
/// Returns (node, data_id) for registering in constant_map
#[allow(dead_code)] // Used by register_test_constant
fn create_test_constant(
    name: String,
    tensor_data: TensorData,
    tensor_store: &mut TensorStore,
) -> (RawNode, DataId) {
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

    // Return node and data_id for registering in constant_map
    (constant_node, data_id)
}
