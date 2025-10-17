use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::node_remap::remap_node_type;
use crate::util::verify_opsets;

use super::{
    coalesce::coalesce,
    ir::{ElementType, OnnxGraph, TensorData},
    processor::ProcessorRegistry,
    proto_conversion::convert_node_proto,
    protos::{ModelProto, NodeProto, TensorProto, ValueInfoProto},
};

use super::ir::{ArgType, Argument, Node, NodeType};

use protobuf::Message;

// Lazily initialized processor registry
use std::sync::OnceLock;

static PROCESSOR_REGISTRY: OnceLock<ProcessorRegistry> = OnceLock::new();

fn get_processor_registry() -> &'static ProcessorRegistry {
    PROCESSOR_REGISTRY.get_or_init(ProcessorRegistry::with_standard_processors)
}

use crate::protos::tensor_proto::DataType as DT;
use protobuf::Enum;

pub fn element_type_from_proto(dt_i32: i32) -> Result<ElementType, String> {
    match DT::from_i32(dt_i32).ok_or_else(|| format!("unknown dtype {}", dt_i32))? {
        DT::FLOAT => Ok(ElementType::Float32),
        DT::DOUBLE => Ok(ElementType::Float64),
        DT::FLOAT16 => Ok(ElementType::Float16),
        DT::INT64 => Ok(ElementType::Int64),
        DT::INT32 => Ok(ElementType::Int32),
        DT::UINT16 => Ok(ElementType::Uint16),
        DT::UINT8 => Ok(ElementType::Uint8),
        DT::INT8 => Ok(ElementType::Int8),
        DT::BOOL => Ok(ElementType::Bool),
        DT::STRING => Ok(ElementType::String),
        other => Err(format!("unsupported dtype {:?}", other)),
    }
}
/// Minimum required ONNX opset version
pub const MIN_OPSET_VERSION: i64 = 16;

#[derive(Debug, Clone)]
pub(crate) enum IOEntry {
    In(usize),
    Node(usize, usize),
}

pub struct GraphData {
    /// The nodes that have been processed, used to copy the outputs to a child node
    processed_nodes: Vec<Node>,
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
    /// Reference counts for constant nodes (keyed by output name)
    constant_references: HashMap<String, usize>,
    /// Maps constant output names to node indices
    constant_nodes: HashMap<String, usize>,
    /// Tracks which node indices correspond to initializers (for lifting logic)
    initializer_nodes: HashSet<usize>,
    /// Nodes marked for removal
    nodes_to_remove: HashSet<usize>,
    /// Cached values from consumed constants (constant node removed, but value still accessible)
    /// Also used for lifted constants that need to be accessible via into_value()
    pub(crate) consumed_values: HashMap<String, TensorData>,
    /// Type expectations set via should_be() method
    /// Maps argument names to their expected types
    expected_types: HashMap<String, ArgType>,
}

impl GraphData {
    pub(crate) fn new(
        inputs: &[ValueInfoProto],
        outputs: &[ValueInfoProto],
        initializers: &[TensorProto],
    ) -> Self {
        let mut input_name_map = HashMap::new();
        let mut input_key_map = HashMap::new();
        let mut processed_nodes = Vec::new();
        let mut constant_references = HashMap::new();
        let mut constant_nodes = HashMap::new();

        // Convert all initializers to Constant nodes immediately
        let mut constants_map: HashMap<String, Argument> = HashMap::new();
        let mut tensor_data_map: HashMap<String, TensorData> = HashMap::new();
        for initializer in initializers.iter() {
            let (arg, tensor_data) = Argument::from_initializer(initializer);
            constants_map.insert(initializer.name.clone(), arg);
            tensor_data_map.insert(initializer.name.clone(), tensor_data);
        }

        // Track initializer node indices for lifting logic
        let mut initializer_nodes = HashSet::new();

        // Create Constant nodes for all initializers
        for (idx, initializer) in initializers.iter().enumerate() {
            if let Some(arg) = constants_map.get(&initializer.name) {
                // Use same naming scheme as other constants: "constant1", "constant2", etc.
                let const_name = format!("constant{}", idx + 1);
                let output_name = format!("{}_out1", const_name);

                let mut constant_node = Node {
                    node_type: NodeType::Constant,
                    name: const_name.clone(),
                    inputs: vec![],
                    outputs: vec![Argument {
                        name: output_name.clone(),
                        ty: arg.ty.clone(),
                        value_store: None,
                    }],
                    attrs: HashMap::new(),
                    config: None,
                };

                // Store the value in the 'value' attribute
                if let Some(tensor_data) = tensor_data_map.get(&initializer.name) {
                    constant_node.attrs.insert(
                        "value".to_string(),
                        crate::ir::AttributeValue::Tensor(tensor_data.clone()),
                    );
                }

                let node_idx = processed_nodes.len();

                // Register this constant
                constant_nodes.insert(output_name.clone(), node_idx);
                constant_references.insert(output_name.clone(), 0);

                // Mark this as an initializer node for lifting logic
                initializer_nodes.insert(node_idx);

                // Map the original initializer name to this constant node
                input_name_map.insert(initializer.name.clone(), IOEntry::Node(node_idx, 0));

                processed_nodes.push(constant_node);
            }
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
            constant_references,
            constant_nodes,
            initializer_nodes,
            nodes_to_remove: HashSet::new(),
            consumed_values: HashMap::new(),
            expected_types: HashMap::new(),
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

    /// This function does three things:
    ///     1. marks the inputs as passed
    ///     2. maps the old output names to the node output
    ///     3. renames the node output
    fn add_node(&mut self, mut node: Node) {
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
        self.processed_nodes.push(node);
    }

    /// Consumes the graph data and returns the processed nodes, filtered inputs, outputs, and nodes to remove
    fn consume(mut self) -> (Vec<Node>, Vec<Argument>, Vec<Argument>, HashSet<usize>) {
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
        (
            self.processed_nodes,
            self.inputs,
            outputs,
            self.nodes_to_remove,
        )
    }

    /// Used to get the output of the graph by name. Only used to remap unsqueeze nodes
    pub fn get_graph_output(&self, name: &str) -> Option<&Argument> {
        self.outputs.iter().find(|x| x.name == name)
    }

    // Since Nodes are added at the end of conversion, the current index is the length of the processed nodes
    /// Get the current index of the processed nodes. Useful when lifting values or marking nodes for removal
    pub fn get_current_index(&self) -> usize {
        self.processed_nodes.len()
    }

    /// Register a constant node and initialize its reference count to 0
    pub(crate) fn register_constant(&mut self, output_name: String, node_idx: usize) {
        self.constant_nodes.insert(output_name.clone(), node_idx);
        self.constant_references.insert(output_name, 0);
    }

    /// Increment the reference count for a constant
    pub(crate) fn increment_constant_ref(&mut self, output_name: &str) {
        if let Some(count) = self.constant_references.get_mut(output_name) {
            *count += 1;
        }
    }

    /// Decrement the reference count for a constant and mark it for removal if count reaches 0
    /// Returns true if the constant was marked for removal
    pub(crate) fn decrement_constant_ref(&mut self, output_name: &str) -> bool {
        if let Some(count) = self.constant_references.get_mut(output_name) {
            if *count > 0 {
                *count -= 1;
            }
            if *count == 0
                && let Some(&node_idx) = self.constant_nodes.get(output_name)
            {
                self.nodes_to_remove.insert(node_idx);
                return true;
            }
        }
        false
    }

    /// Check if an argument name corresponds to a constant node
    pub(crate) fn is_constant(&self, arg_name: &str) -> bool {
        self.constant_nodes.contains_key(arg_name)
    }

    /// Get the value from a constant node by output name
    pub(crate) fn get_constant_value(&self, output_name: &str) -> Option<&Node> {
        self.constant_nodes
            .get(output_name)
            .and_then(|&idx| self.processed_nodes.get(idx))
    }

    /// Get mutable access to a constant node by output name
    pub(crate) fn get_constant_value_mut(&mut self, output_name: &str) -> Option<&mut Node> {
        if let Some(&idx) = self.constant_nodes.get(output_name) {
            self.processed_nodes.get_mut(idx)
        } else {
            None
        }
    }

    /// Check if a value is available (either in active constants or consumed cache)
    pub(crate) fn has_value(&self, name: &str) -> bool {
        self.constant_nodes.contains_key(name) || self.consumed_values.contains_key(name)
    }

    /// Get the tensor data value for a constant by name
    /// Returns cloned data from either active constant node or consumed cache
    pub(crate) fn get_value(&self, name: &str) -> Option<TensorData> {
        // Check consumed cache first (faster)
        if let Some(cached) = self.consumed_values.get(name) {
            return Some(cached.clone());
        }

        // Otherwise extract from constant node
        self.get_constant_value(name).and_then(|node| {
            node.attrs.get("value").and_then(|attr| {
                if let crate::ir::AttributeValue::Tensor(tensor) = attr {
                    Some(tensor.clone())
                } else {
                    None
                }
            })
        })
    }

    /// Mark a constant as consumed, removing it from active nodes and caching its value
    pub(crate) fn mark_consumed(&mut self, name: &str) {
        // Extract value from constant node before removing it
        if let Some(node) = self.get_constant_value(name)
            && let Some(crate::ir::AttributeValue::Tensor(tensor)) = node.attrs.get("value")
        {
            self.consumed_values
                .insert(name.to_string(), tensor.clone());
        }

        // Decrement reference count and mark for removal
        self.decrement_constant_ref(name);
    }

    /// Attach value store reference to an argument
    /// This allows the argument to access constant values without explicitly passing GraphData
    pub(crate) fn attach_value_store_to_arg(
        &self,
        arg: &mut Argument,
        store: std::rc::Rc<std::cell::RefCell<GraphData>>,
    ) {
        arg.value_store = Some(store);
    }

    /// Helper to create a shared reference to GraphData
    /// This is used when converting nodes to attach value stores
    pub(crate) fn create_shared_ref(self) -> std::rc::Rc<std::cell::RefCell<GraphData>> {
        std::rc::Rc::new(std::cell::RefCell::new(self))
    }

    /// Get mutable access to processed nodes
    /// This is used to clear value_stores before consuming GraphData
    pub(crate) fn get_processed_nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.processed_nodes
    }

    /// Register a test constant in GraphData. This is used by test utilities to add constant
    /// values that can be retrieved via `into_value()`.
    pub(crate) fn register_test_constant(
        &mut self,
        name: String,
        data: crate::ir::Data,
        shape: Vec<usize>,
    ) {
        use crate::ir::{AttributeValue, NodeType, TensorData};

        // Determine element type from data
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

        let output_name = format!("{}_const_out", name);
        let const_node_name = format!("{}_const", name);

        let tensor_data = TensorData {
            data,
            shape: shape.clone(),
        };

        let mut constant_node = Node {
            node_type: NodeType::Constant,
            name: const_node_name,
            inputs: vec![],
            outputs: vec![Argument {
                name: output_name.clone(),
                ty: crate::ir::ArgType::Tensor(crate::ir::TensorType {
                    elem_type,
                    rank: shape.len(),
                    static_shape: Some(shape),
                }),
                value_store: None,
            }],
            attrs: HashMap::new(),
            config: None,
        };

        // Store the value in the 'value' attribute
        constant_node
            .attrs
            .insert("value".to_string(), AttributeValue::Tensor(tensor_data));

        // Register this constant
        let node_idx = self.processed_nodes.len();
        self.constant_nodes.insert(name.clone(), node_idx);
        self.constant_references.insert(name.clone(), 0);

        self.processed_nodes.push(constant_node);
    }

    /// Set the expected type for an argument
    /// This is called by Argument::should_be() to record type expectations
    pub(crate) fn set_expected_type(&mut self, arg_name: String, expected_ty: ArgType) {
        self.expected_types.insert(arg_name, expected_ty);
    }

    /// Get the expected type for an argument, if any
    pub(crate) fn get_expected_type(&self, arg_name: &str) -> Option<&ArgType> {
        self.expected_types.get(arg_name)
    }

    /// Check if a constant output name corresponds to an initializer
    /// Uses the initializer_nodes set to identify initializers
    /// Returns false for ONNX Constant nodes
    pub(crate) fn is_initializer_output(&self, output_name: &str) -> bool {
        self.constant_nodes
            .get(output_name)
            .map(|&node_idx| self.initializer_nodes.contains(&node_idx))
            .unwrap_or(false)
    }
}

#[derive(Default)]
pub(crate) struct OnnxGraphBuilder {
    node_name_counter: HashMap<NodeType, usize>,
}

impl OnnxGraphBuilder {
    /// Run iterative type inference with preference propagation
    /// This alternates between type inference and preference collection until convergence
    fn iterative_type_inference_with_preferences(
        &self,
        nodes: &mut [Node],
        opset: usize,
    ) {
        use crate::processor::ArgPreference;

        let registry = get_processor_registry();

        // Track collected preferences: (producer_output_name, consumer_name, pref_type_str)
        let mut collected_preferences: HashSet<(String, String, String)> = HashSet::new();

        let max_iterations = 100; // Safety limit to prevent infinite loops

        for iteration in 1..=max_iterations {
            log::debug!("Type inference iteration {}", iteration);

            // Step 1: Build OutputPreferences map from collected preferences
            let mut node_preferences: HashMap<String, crate::processor::OutputPreferences> = HashMap::new();

            for (output_name, consumer_name, pref_type_str) in &collected_preferences {
                let pref = match pref_type_str.as_str() {
                    "Scalar" => ArgPreference::Scalar,
                    "Shape" => ArgPreference::Shape,
                    "Tensor" => ArgPreference::Tensor,
                    _ => continue,
                };

                // Find producer node for this output
                for node in nodes.iter() {
                    if node.outputs.iter().any(|o| &o.name == output_name) {
                        node_preferences
                            .entry(node.name.clone())
                            .or_default()
                            .add(output_name.clone(), consumer_name.clone(), pref);
                        break;
                    }
                }
            }

            // Step 2: Run infer_types on all nodes with current preferences
            for node in nodes.iter_mut() {
                let prefs = node_preferences
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_else(crate::processor::OutputPreferences::new);

                let processor = registry.get(&node.node_type);
                let _ = processor.infer_types(node, opset, &prefs);
            }

            // Step 2.5: Sync input types from producer outputs
            // After inference, propagate output types to consuming nodes' inputs
            let output_types: HashMap<String, ArgType> = nodes
                .iter()
                .flat_map(|n| n.outputs.iter().map(|o| (o.name.clone(), o.ty.clone())))
                .collect();

            for node in nodes.iter_mut() {
                for input in &mut node.inputs {
                    if let Some(new_type) = output_types.get(&input.name) {
                        input.ty = new_type.clone();
                    }
                }
            }

            // Step 3: Collect NEW input_preferences based on inferred types
            let mut new_preferences_found = false;

            for consumer_node in nodes.iter() {
                let processor = registry.get(&consumer_node.node_type);

                if let Ok(Some(input_prefs)) = processor.input_preferences(consumer_node, opset) {
                    // For each input this consumer has preferences for
                    for input in &consumer_node.inputs {
                        let requested_types = input_prefs.get(&input.name);

                        if requested_types.is_empty() {
                            continue;
                        }

                        // Find which node produces this input
                        for producer_node in nodes.iter() {
                            if let Some(output) = producer_node.outputs.iter().find(|o| o.name == input.name) {
                                // Check each requested preference type
                                for req_type in requested_types {
                                    let pref_type_str = match req_type {
                                        ArgPreference::Scalar => "Scalar",
                                        ArgPreference::Shape => "Shape",
                                        ArgPreference::Tensor => "Tensor",
                                    }.to_string();

                                    let key = (output.name.clone(), consumer_node.name.clone(), pref_type_str);

                                    // Only add if this is a NEW preference
                                    if !collected_preferences.contains(&key) {
                                        collected_preferences.insert(key.clone());
                                        new_preferences_found = true;

                                        log::debug!(
                                            "Iteration {}: Node {} requests {:?} for output {} from node {}",
                                            iteration,
                                            consumer_node.name,
                                            req_type,
                                            output.name,
                                            producer_node.name
                                        );
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }

            // Step 4: Check convergence
            if !new_preferences_found {
                log::debug!("Type inference converged after {} iterations", iteration);
                return;
            }
        }

        log::warn!("Type inference iteration limit ({}) reached without convergence", max_iterations);
    }

    pub(crate) fn build(mut self, model_proto: &ModelProto) -> OnnxGraph {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Extract opset version from model (default ONNX domain)
        let opset_version = model_proto
            .opset_import
            .iter()
            .find(|opset| opset.domain.is_empty())
            .map(|opset| opset.version as usize)
            .unwrap_or(MIN_OPSET_VERSION as usize);

        // Initialize Constant node counter to account for initializers
        let num_initializers = model_proto.graph.initializer.len();
        if num_initializers > 0 {
            self.node_name_counter
                .insert(NodeType::Constant, num_initializers);
        }

        let mut graph_data = GraphData::new(
            &model_proto.graph.input,
            &model_proto.graph.output,
            &model_proto.graph.initializer,
        );

        // Wrap GraphData in Rc<RefCell<>> to allow shared mutable access
        let graph_data_rc = Rc::new(RefCell::new(graph_data));

        for t in &model_proto.graph.initializer {
            log::debug!(
                "init name={:?} dtype={:?} dims={:?} raw_len={} i32={} i64={} f32={} f64={}",
                t.name,
                crate::protos::tensor_proto::DataType::from_i32(t.data_type),
                t.dims,
                t.raw_data.len(),
                t.int32_data.len(),
                t.int64_data.len(),
                t.float_data.len(),
                t.double_data.len(),
            );
        }

        let mut node_iter = model_proto.graph.node.iter().peekable();

        // Track which constants were lifted into configs
        // These should be filtered out of the final graph
        let mut lifted_constants = HashSet::new();

        // PASS 2: Process nodes with collected preferences
        while let Some(node_proto) = node_iter.next() {
            let mut node = convert_node_proto(node_proto, &graph_data_rc.borrow());

            // Attach value_store to all arguments in the node
            for arg in &mut node.inputs {
                arg.value_store = Some(graph_data_rc.clone());
            }
            for arg in &mut node.outputs {
                arg.value_store = Some(graph_data_rc.clone());
            }

            remap_node_type(&mut node);
            self.handle_node_renaming(&mut node);

            // Track node type before coalesce
            let node_type_before_coalesce = node.node_type.clone();

            coalesce(&mut node, &mut node_iter, &mut graph_data_rc.borrow_mut());

            // Re-attach value_stores after coalesce (which may add new inputs from fusion)
            for arg in &mut node.inputs {
                arg.value_store = Some(graph_data_rc.clone());
            }
            for arg in &mut node.outputs {
                arg.value_store = Some(graph_data_rc.clone());
            }

            // If coalesce changed the node type (e.g., Gemm->Linear, MatMul->Linear), rename it
            if node.node_type != node_type_before_coalesce {
                self.handle_node_renaming(&mut node);
            }

            // Convert Identity nodes with constant inputs to Constant nodes
            // This allows burn-import to access the constant values via into_value()
            if node.node_type == NodeType::Identity && !node.inputs.is_empty() {
                let input_name = &node.inputs[0].name;
                let has_constant_input = {
                    let graph_data = graph_data_rc.borrow();
                    graph_data.is_constant(input_name)
                };

                if has_constant_input {
                    // Convert Identity to Constant node
                    let constant_value = {
                        let graph_data = graph_data_rc.borrow();
                        graph_data.get_value(input_name)
                    };

                    if let Some(tensor_data) = constant_value {
                        log::debug!("Converting Identity node {} to Constant (input: {})", node.name, input_name);

                        node.node_type = NodeType::Constant;
                        node.attrs.insert(
                            "value".to_string(),
                            crate::ir::AttributeValue::Tensor(tensor_data),
                        );
                        node.inputs.clear(); // Constant nodes have no inputs

                        // Rename since we changed type
                        self.handle_node_renaming(&mut node);

                        // Re-attach value_stores after renaming
                        for arg in &mut node.outputs {
                            arg.value_store = Some(graph_data_rc.clone());
                        }
                    }
                }
            }

            // NOTE: potential start of custom functions
            // can filter, coalesce, or modify the nodes here
            // args : node, peek_iter, graph_data

            log::debug!("Processing node: {}", node.name);
            let registry = get_processor_registry();
            let processor = registry.get(&node.node_type);

            // Register ALL Constant nodes so their values can be accessed via has_value() and get_value()
            // This includes: initializer constants (already registered), converted Identity nodes,
            // and explicit ONNX Constant nodes
            if node.node_type == NodeType::Constant && !node.outputs.is_empty() {
                let future_output_name = format!("{}_out1", node.name);
                let node_idx = {
                    let graph_data = graph_data_rc.borrow();
                    graph_data.get_current_index()
                };

                // Only register if not already registered (e.g., initializer constants)
                {
                    let mut graph_data = graph_data_rc.borrow_mut();
                    if !graph_data.constant_nodes.contains_key(&future_output_name) {
                        graph_data
                            .constant_nodes
                            .insert(future_output_name.clone(), node_idx);
                        graph_data.constant_references.insert(future_output_name, 0);
                    }
                } // Explicitly drop mutable borrow here
            }

            // Lift constants (ensure constant inputs are accessible)
            // lift_constants returns a list of input names that COULD be lifted
            // We filter by has_value() to only lift actual constants
            let potential_lifts = processor
                .lift_constants(&mut node, opset_version)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to lift constants for node {} (type: {:?}): {:?}",
                        node.name, node.node_type, e
                    )
                });

            // Filter to only lift inputs that are constants (have values available)
            // All constants are liftable - initializers, ONNX Constant nodes, etc.
            // Check GraphData directly to avoid RefCell borrow conflicts
            let lifted: Vec<String> = {
                let graph_data = graph_data_rc.borrow();
                potential_lifts
                    .into_iter()
                    .filter(|input_name| graph_data.has_value(input_name))
                    .collect()
            }; // Drop immutable borrow here

            // Make lifted constants accessible by caching their values
            // Identity nodes with constant values have already been converted to Constant nodes
            for input_name in &lifted {
                {
                    let mut graph_data = graph_data_rc.borrow_mut();

                    // Get the value from the constant and cache it
                    if let Some(value) = graph_data.get_value(input_name) {
                        // Cache the value to ensure it stays available for burn-import
                        graph_data.consumed_values.insert(input_name.clone(), value);
                        log::debug!("Lifted constant {} for node {}", input_name, node.name);

                        // Track this as a lifted constant for potential removal
                        // Lifted constants have their values in node configs and shouldn't be model parameters
                        lifted_constants.insert(input_name.clone());
                        log::debug!("Marked constant {} for removal", input_name);
                    } else {
                        log::warn!(
                            "Failed to lift constant {} for node {} - value not found",
                            input_name,
                            node.name
                        );
                    }
                } // Explicitly drop mutable borrow before next iteration
            }

            // Extract config first
            let config = processor
                .extract_config(&node, opset_version)
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to extract config for node {} (type: {:?}): {:?}",
                        node.name, node.node_type, e
                    )
                });
            node.config = config;

            // Add node to graph (type inference happens later in iterative loop)
            graph_data_rc.borrow_mut().add_node(node);
        }

        // Run iterative type inference with preference propagation
        // This allows preferences to be collected based on inferred types,
        // enabling scenarios like Concat requesting Shape types after seeing Shape inputs
        log::debug!("Starting iterative type inference with preference propagation");
        {
            // Temporarily extract nodes to avoid holding mutable borrow during iteration
            // (iteration may need immutable borrows for into_value() calls)
            let mut nodes = std::mem::take(&mut graph_data_rc.borrow_mut().processed_nodes);
            self.iterative_type_inference_with_preferences(&mut nodes, opset_version);
            graph_data_rc.borrow_mut().processed_nodes = nodes;
        }

        // Cache all Constant node values for burn-import to access
        // This ensures burn-import can generate code for ALL constants, not just lifted ones
        {
            let mut graph_data = graph_data_rc.borrow_mut();

            // Collect constant nodes that need caching (to avoid borrow issues)
            let constant_outputs: Vec<String> = graph_data
                .processed_nodes
                .iter()
                .filter(|node| node.node_type == NodeType::Constant && !node.outputs.is_empty())
                .map(|node| node.outputs[0].name.clone())
                .collect();

            // Cache values for all constant nodes
            for output_name in constant_outputs {
                if !graph_data.consumed_values.contains_key(&output_name)
                    && let Some(value) = graph_data.get_value(&output_name)
                {
                    graph_data
                        .consumed_values
                        .insert(output_name.clone(), value);
                    log::debug!("Cached constant {} value for burn-import", output_name);
                }
            }
        }

        // Extract the processed graph data and preserve consumed_values for burn-import
        let (mut processed_nodes, inputs, outputs, nodes_to_remove) = {
            let mut graph_data = graph_data_rc.borrow_mut();

            // Extract consumed_values before consuming
            let consumed_values = std::mem::take(&mut graph_data.consumed_values);

            // Consume the old graph_data
            let result =
                std::mem::replace(&mut *graph_data, GraphData::new(&[], &[], &[])).consume();

            // Restore consumed_values so burn-import can access them via into_value()
            graph_data.consumed_values = consumed_values;

            (result.0, result.1, result.2, result.3)
        };

        // Determine which lifted constants can be removed
        // A lifted constant can be removed if it's ONLY used by ONE node (the node that lifted it)
        //
        // Build a map of constant_name -> count of nodes that use it
        let mut constant_user_counts: HashMap<String, usize> = HashMap::new();
        for node in &processed_nodes {
            for input in &node.inputs {
                if lifted_constants.contains(&input.name) {
                    *constant_user_counts.entry(input.name.clone()).or_insert(0) += 1;
                }
            }
        }

        // Filter out nodes marked for removal AND lifted constants used by only one node
        log::debug!(
            "Filtering nodes: total={}, nodes_to_remove={:?}, lifted_constants={}",
            processed_nodes.len(),
            nodes_to_remove,
            lifted_constants.len()
        );

        let mut i = 0;
        processed_nodes.retain(|node| {
            let is_lifted_constant = node
                .outputs
                .first()
                .is_some_and(|output| lifted_constants.contains(&output.name));

            // If this is a lifted constant, check how many nodes use it
            // Remove if used by only 1 node (the node that lifted it)
            let used_by_multiple = if is_lifted_constant {
                node.outputs
                    .first()
                    .and_then(|output| constant_user_counts.get(&output.name))
                    .map(|count| *count > 1)
                    .unwrap_or(false)
            } else {
                false
            };

            let keep = !nodes_to_remove.contains(&i) && (!is_lifted_constant || used_by_multiple);

            if !keep {
                log::debug!(
                    "Filtering out node at index {}: {} (lifted={}, used_by_multiple={})",
                    i,
                    node.name,
                    is_lifted_constant,
                    used_by_multiple
                );
            }
            i += 1;
            keep
        });

        log::debug!("After filtering: {} nodes remain", processed_nodes.len());

        // TODO Update graph inputs and outputs to match the processed nodes inputs and outputs
        // This is necessary for the graph to be valid
        // ConstantOfShape updates input to be Shape argument and output Tensor dim is updated
        OnnxGraph {
            nodes: processed_nodes,
            inputs,
            outputs,
        }
    }

    fn handle_node_renaming(&mut self, node: &mut Node) {
        self.node_name_counter
            .entry(node.node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);
        let new_name = format!(
            "{}{}",
            node.node_type, self.node_name_counter[&node.node_type]
        )
        .to_lowercase();

        log::debug!("Renaming node {:?} to {new_name:?}", &node.name);

        node.name.clone_from(&new_name);
    }

}

/// Parses an ONNX model file and converts it to an intermediate representation.
///
/// This function reads an ONNX model from the specified path, validates its opset version,
/// and transforms it into our internal graph representation for further processing.
///
/// # Arguments
///
/// * `onnx_path` - Path to the ONNX model file
///
/// # Returns
///
/// * `OnnxGraph` - The internal graph representation of the ONNX model
///
/// # Panics
///
/// * If the file cannot be opened or read
/// * If the ONNX model cannot be parsed
/// * If the model uses an unsupported opset version (must be >= MIN_OPSET_VERSION)
/// * If the nodes in the graph are not topologically sorted
pub fn parse_onnx(onnx_path: &Path) -> OnnxGraph {
    log::info!("Parsing ONNX file: {}", onnx_path.display());

    // Open the file
    let mut file = File::open(onnx_path)
        .unwrap_or_else(|_| panic!("Unable to open file: {}", onnx_path.display()));
    let onnx_model: ModelProto =
        Message::parse_from_reader(&mut file).expect("Unable to parse ONNX file");

    // Check opset versions - must be >= MIN_OPSET_VERSION
    if !verify_opsets(&onnx_model.opset_import, MIN_OPSET_VERSION) {
        panic!(
            "Unsupported ONNX opset version. This implementation requires opset {MIN_OPSET_VERSION} or higher. \
            Please upgrade your model using the ONNX shape inference tool. \
            See documentation (https://burn.dev/books/burn/import/onnx-model.html) for details."
        );
    }

    // ONNX nodes must be topologically sorted per spec:
    // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    debug_assert!(
        onnx_model.graph.node.is_top_sorted(),
        "Nodes are not topologically sorted"
    );
    log::debug!("Number of nodes: {:?}", onnx_model.graph.node.len());
    log::debug!("Number of inputs: {:?}", onnx_model.graph.input.len());

    log::debug!(
        "Number of initializers: {:?}",
        onnx_model.graph.initializer.len()
    );

    log::debug!("Number of outputs: {:?}", onnx_model.graph.output.len());

    // Debug information about opset versions
    for opset in &onnx_model.opset_import {
        log::debug!(
            "Opset domain: {:?}, version: {:?}",
            if opset.domain.is_empty() {
                "<default>"
            } else {
                &opset.domain
            },
            opset.version
        );
    }

    let builder = OnnxGraphBuilder::default();
    let graph = builder.build(&onnx_model);

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());

    graph
}

// Define a trait for topological sorting
trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

impl TopologicalSortable for Vec<NodeProto> {
    fn is_top_sorted(&self) -> bool {
        // Iterate over each node in the vector
        for (node_position, node) in self.iter().enumerate() {
            // Iterate over each output of the node
            for output in &node.output {
                // If the output is empty, we don't want to check the rest of the graph, inputs and outputs that are optional
                // can end up as empty strings, so we can't use that as a reason to count the graph as not sorted
                if output.is_empty() {
                    continue;
                }
                // Iterate over each other node in the vector
                for (other_node_position, other_node) in self.iter().enumerate() {
                    // If the other node has an input that matches the current output
                    if other_node.input.contains(output) {
                        // If the position of the current node is greater than the position of the other node
                        if node_position > other_node_position {
                            // The vector is not topologically sorted
                            return false;
                        }
                    }
                }
            }
        }

        // The vector is topologically sorted
        true
    }
}

/// Get the value of a constant node from its attributes
pub fn convert_constant_value(node: &Node) -> Argument {
    // A value can be stored in any of these attributes
    let keys = [
        "value",
        "value_float",
        "value_floats",
        "value_int",
        "value_ints",
        "value_string",
        "value_strings",
        "sparse_value",
    ];

    let value = keys
        .iter()
        .find_map(|&key| node.attrs.get(key).cloned())
        .expect("Constant should have a value");

    Argument::from(value)
}
