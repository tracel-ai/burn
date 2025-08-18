use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::node_remap::remap_node_type;
use crate::util::verify_opsets;

use super::{
    coalesce::coalesce,
    ir::{Data, ElementType, OnnxGraph, TensorData, TensorType},
    proto_conversion::convert_node_proto,
    protos::{ModelProto, NodeProto, TensorProto, ValueInfoProto},
};

use super::ir::{ArgType, Argument, Node, NodeType};
use super::rank_inference::rank_inference;

use protobuf::Message;

const LIFT_CONSTANTS_FOR_NODE_TYPES: [NodeType; 28] = [
    NodeType::BatchNormalization,
    NodeType::Clip,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::Conv3d,
    NodeType::ConvTranspose1d,
    NodeType::ConvTranspose2d,
    NodeType::ConvTranspose3d,
    NodeType::Dropout,
    NodeType::Expand,
    NodeType::Gather,
    NodeType::GroupNormalization,
    NodeType::InstanceNormalization,
    NodeType::LayerNormalization,
    NodeType::Linear,
    NodeType::OneHot,
    NodeType::PRelu,
    NodeType::Pad,
    NodeType::ReduceSum,
    NodeType::Reshape,
    NodeType::Resize,
    NodeType::Slice,
    NodeType::Split,
    NodeType::Squeeze,
    NodeType::Tile,
    NodeType::TopK,
    NodeType::Trilu,
    NodeType::Unsqueeze,
];

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
}

impl GraphData {
    pub(crate) fn new(
        inputs: &[ValueInfoProto],
        outputs: &[ValueInfoProto],
        initializers: &[TensorProto],
    ) -> Self {
        let mut input_name_map = HashMap::new();
        let mut input_key_map = HashMap::new();

        let constants = initializers
            .iter()
            .map(|x| (x.name.clone(), Argument::from_initializer(x)))
            .collect::<HashMap<String, Argument>>();
        let outputs = outputs
            .iter()
            .map(|x| Argument::try_from(x.clone()).unwrap())
            .collect::<Vec<Argument>>();
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let in_name = format!("input{}", i + 1);

                input_name_map.insert(x.name.clone(), IOEntry::In(i));
                input_key_map.insert(in_name.clone(), x.name.clone());

                let mut arg = Argument::try_from(x.clone()).unwrap();
                if let Some(initial_arg) = constants.get(&x.name) && arg.value.is_none() {
                        log::warn!("Input {} is also an initializer. Initializer as default values are currently not supported", x.name);
                        arg.copy_value(initial_arg);
                }

                arg.name = in_name;
                arg
            })
            .collect::<Vec<Argument>>();
        Self {
            inputs,
            outputs,
            initializers: constants,
            processed_nodes: Vec::new(),
            input_name_map,
            input_key_map,
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
                    Some(IOEntry::In(i)) => self.inputs[*i].passed = true,
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

    /// Consumes the graph data and returns the processed nodes, filtered inputs and outputs
    fn consume(mut self) -> (Vec<Node>, Vec<Argument>, Vec<Argument>) {
        self.inputs.retain(|x| x.passed);
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

    /// Used to get the output of the graph by name. Only used to remap unsqueeze nodes
    pub fn get_graph_output(&self, name: &str) -> Option<&Argument> {
        self.outputs.iter().find(|x| x.name == name)
    }

    // Since Nodes are added at the end of conversion, the current index is the length of the processed nodes
    /// Get the current index of the processed nodes. Useful when lifting values or marking nodes for removal
    pub fn get_current_index(&self) -> usize {
        self.processed_nodes.len()
    }
}

#[derive(Default)]
pub(crate) struct OnnxGraphBuilder {
    /// Nodes to remove. Note may be moved to graph data if we implement support for custom ops
    nodes_to_remove: HashSet<usize>,
    /// Map from constant node output names to indices of constant nodes
    constants_map: HashMap<String, usize>,
    /// Node types that should be lifted to constants
    constants_types: HashSet<NodeType>,
    node_name_counter: HashMap<NodeType, usize>,
}

impl OnnxGraphBuilder {
    pub(crate) fn build(mut self, model_proto: &ModelProto) -> OnnxGraph {
        self.constants_types = LIFT_CONSTANTS_FOR_NODE_TYPES.into_iter().collect();

        let mut graph_data = GraphData::new(
            &model_proto.graph.input,
            &model_proto.graph.output,
            &model_proto.graph.initializer,
        );

        let mut node_iter = model_proto.graph.node.iter().peekable();

        while let Some(node_proto) = node_iter.next() {
            let mut node = convert_node_proto(node_proto, &graph_data);

            remap_node_type(&mut node);
            self.handle_node_renaming(&mut node);
            coalesce(&mut node, &mut node_iter, &graph_data);
            self.handle_identity(&mut node);
            self.check_constants(&mut node, &graph_data);
            self.convert_initializer_inputs_to_constants(&mut node, &mut graph_data);
            // NOTE: potential start of custom functions
            // can filter, coalesce, or modify the nodes here
            // args : node, peek_iter, graph_data
            self.handle_unsqueeze(&mut node, &graph_data);

            rank_inference(&mut node);
            graph_data.add_node(node);
        }

        let (mut processed_nodes, inputs, outputs) = graph_data.consume();

        // Convert Constant nodes to Shape type when used with Shape in binary operations
        self.convert_shape_constants(&mut processed_nodes);

        // Remove the graph inputs/output that are not used by any node
        let mut i = 0;
        processed_nodes.retain(|_| {
            let keep = !self.nodes_to_remove.contains(&i);
            i += 1;
            keep
        });

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

    /// Convert Constant nodes to Shape type when used with Shape in operations like Add, Sub, Mul, Div, and Concat
    fn convert_shape_constants(&self, nodes: &mut [Node]) {
        // Find constants that need to be converted to Shape type
        let mut constants_to_convert = self.find_shape_constants(nodes);

        // If no constants need conversion, return early
        if constants_to_convert.is_empty() {
            return;
        }

        // Get actual ranks from constant tensor data
        self.update_constant_ranks(nodes, &mut constants_to_convert);

        // Apply the conversions to constants and their uses
        self.apply_shape_conversions(nodes, &constants_to_convert);
    }

    /// Find constants that should be converted to Shape type based on their usage
    fn find_shape_constants(&self, nodes: &[Node]) -> HashMap<String, usize> {
        let mut constants_to_convert = HashMap::new();

        for node in nodes {
            let shape_inputs = self.get_shape_compatible_inputs(node);

            for (input_name, expected_rank) in shape_inputs {
                constants_to_convert.insert(input_name, expected_rank);
            }
        }

        constants_to_convert
    }

    /// Get inputs that should be converted to Shape type for a given node
    fn get_shape_compatible_inputs(&self, node: &Node) -> Vec<(String, usize)> {
        let mut shape_inputs = Vec::new();

        match node.node_type {
            // Binary operations: convert rank-1 tensors if the other input is Shape
            NodeType::Add | NodeType::Sub | NodeType::Mul | NodeType::Div => {
                if node.inputs.len() != 2 {
                    return shape_inputs;
                }

                // Find if there's a Shape input
                let shape_rank = node.inputs.iter().find_map(|input| {
                    if let ArgType::Shape(rank) = input.ty {
                        Some(rank)
                    } else {
                        None
                    }
                });

                if let Some(shape_rank) = shape_rank {
                    // Mark rank-1 tensors for conversion
                    for input in &node.inputs {
                        if matches!(&input.ty, ArgType::Tensor(t) if t.rank == 1) {
                            shape_inputs.push((input.name.clone(), shape_rank));
                        }
                    }
                }
            }
            // Concat: convert rank-1 tensors if any input is Shape
            NodeType::Concat => {
                let has_shape = node
                    .inputs
                    .iter()
                    .any(|i| matches!(i.ty, ArgType::Shape(_)));

                if has_shape {
                    for input in &node.inputs {
                        if matches!(&input.ty, ArgType::Tensor(t) if t.rank == 1) {
                            // Actual rank will be determined from tensor data
                            shape_inputs.push((input.name.clone(), 0));
                        }
                    }
                }
            }
            _ => {}
        }

        shape_inputs
    }

    /// Update the conversion map with actual ranks from constant tensor data
    fn update_constant_ranks(
        &self,
        nodes: &[Node],
        constants_to_convert: &mut HashMap<String, usize>,
    ) {
        use crate::ir::AttributeValue;

        for node in nodes {
            if node.node_type != NodeType::Constant {
                continue;
            }

            let Some(output) = node.outputs.first() else {
                continue;
            };

            if !constants_to_convert.contains_key(&output.name) {
                continue;
            }

            // Get actual rank from tensor data
            if let ArgType::Tensor(tensor) = &output.ty
                && tensor.rank == 1
                && let Some(AttributeValue::Tensor(tensor_data)) = node.attrs.get("value")
                && tensor_data.shape.len() == 1
            {
                let actual_rank = tensor_data.shape[0];
                constants_to_convert.insert(output.name.clone(), actual_rank);
                log::debug!(
                    "Constant {} will be converted to Shape({})",
                    output.name,
                    actual_rank
                );
            }
        }
    }

    /// Apply Shape type conversions to constants and update their uses
    fn apply_shape_conversions(
        &self,
        nodes: &mut [Node],
        constants_to_convert: &HashMap<String, usize>,
    ) {
        for node in nodes.iter_mut() {
            match node.node_type {
                NodeType::Constant => {
                    // Convert constant output to Shape type
                    if let Some(output) = node.outputs.first_mut()
                        && let Some(&shape_rank) = constants_to_convert.get(&output.name)
                        && matches!(&output.ty, ArgType::Tensor(t) if t.rank == 1)
                    {
                        output.ty = ArgType::Shape(shape_rank);
                        log::debug!(
                            "Converted constant {} to Shape({})",
                            output.name,
                            shape_rank
                        );
                    }
                }
                NodeType::Add
                | NodeType::Sub
                | NodeType::Mul
                | NodeType::Div
                | NodeType::Concat => {
                    // Update input types and re-run rank inference if needed
                    let mut needs_reinference = false;

                    for input in &mut node.inputs {
                        if let Some(&shape_rank) = constants_to_convert.get(&input.name)
                            && matches!(&input.ty, ArgType::Tensor(t) if t.rank == 1)
                        {
                            input.ty = ArgType::Shape(shape_rank);
                            needs_reinference = true;
                            log::debug!(
                                "Updated {} input {} to Shape({})",
                                node.node_type,
                                input.name,
                                shape_rank
                            );
                        }
                    }

                    // Re-run rank inference for Concat if inputs changed
                    if needs_reinference && node.node_type == NodeType::Concat {
                        rank_inference(node);
                        log::debug!("Re-ran rank inference for Concat node {}", node.name);
                    }
                }
                _ => {}
            }
        }
    }

    fn check_constants(&mut self, node: &mut Node, graph_data: &GraphData) {
        if node.node_type == NodeType::Constant {
            self.constants_map.insert(
                format!("{}_out{}", &node.name, 1),
                graph_data.get_current_index(),
            );
        } else if node.node_type == NodeType::ConstantOfShape {
            // Special handling for ConstantOfShape - check first input
            log::debug!("Checking ConstantOfShape node {} for constants", &node.name);
            if let Some(input) = node.inputs.first_mut() {
                log::debug!("Checking first input {:?} for const", input);
                if let Some(const_idx) = self.constants_map.get(&input.name) {
                    let constant = &graph_data.processed_nodes[*const_idx];
                    log::debug!(
                        "Input {} matched constant node {}",
                        &input.name,
                        &constant.name
                    );
                    if !constant.inputs.is_empty() && constant.inputs[0].value.is_some() {
                        // The value comes from Identity inputs
                        input.value.clone_from(&constant.inputs[0].value);
                        input.ty = constant.inputs[0].ty.clone();
                    } else {
                        let arg = convert_constant_value(constant);
                        input.value = arg.value;
                        input.ty = arg.ty;
                    }
                    // The constant values are now embedded in the input, so we can remove the constant node
                    self.nodes_to_remove.insert(*const_idx);
                    log::debug!(
                        "Lifted and removed constant node {} for ConstantOfShape",
                        constant.name
                    );
                }
            }
        } else if self.constants_types.contains(&node.node_type) {
            log::debug!("Checking node {} for constants", &node.name);
            for input in node.inputs.iter_mut().skip(1) {
                log::debug!("Checking input {input:?} for const");
                if let Some(const_idx) = self.constants_map.get(&input.name) {
                    let constant = &graph_data.processed_nodes[*const_idx];
                    log::debug!(
                        "Input {} matched constant node {}",
                        &input.name,
                        &constant.name
                    );
                    if !constant.inputs.is_empty() && constant.inputs[0].value.is_some() {
                        // The value comes from Identity inputs
                        input.value.clone_from(&constant.inputs[0].value);
                        input.ty = constant.inputs[0].ty.clone();
                    } else {
                        let arg = convert_constant_value(constant);
                        input.value = arg.value;
                        input.ty = arg.ty;
                    }
                    self.nodes_to_remove.insert(*const_idx);
                }
            }
        }
    }

    /// Check if the unsqueeze node has a rhs value (rhs is constant) and if not remap it to a reshape
    /// Needs to be called after node renaming to ensure that the rhs name is correct
    /// Needs to be called after constant lifting to ensure that the rhs value exists
    fn handle_unsqueeze(&mut self, node: &mut Node, graph_data: &GraphData) {
        if node.node_type == NodeType::Unsqueeze
            && node.inputs.len() > 1
            && node.inputs[1].value.is_none()
        {
            //if the output has a shape, it's only because it's a graph output
            if let Some(out_arg) = graph_data.get_graph_output(&node.outputs[0].name) {
                remap_unsqueeze_to_reshape(node, out_arg);
            }
        }
    }

    /// Convert inputs with values (initializers) to Constant nodes
    fn convert_initializer_inputs_to_constants(
        &mut self,
        node: &mut Node,
        graph_data: &mut GraphData,
    ) {
        // Skip if this is already a Constant node
        if node.node_type == NodeType::Constant {
            return;
        }

        // Check each input for initializer values
        for (idx, input) in node.inputs.iter_mut().enumerate() {
            if let Some(value) = &input.value {
                // Skip inputs that will be lifted by the constant lifting mechanism
                // For nodes in the constant lifting list, skip inputs at index >= 1
                // (these will be lifted by check_constants later)
                if self.constants_types.contains(&node.node_type) && idx >= 1 {
                    log::debug!(
                        "Skipping constant creation for {} input {} (will be lifted)",
                        node.name,
                        idx
                    );
                    continue;
                }

                // Skip first input for ConstantOfShape as it's handled in check_constants
                if node.node_type == NodeType::ConstantOfShape && idx == 0 {
                    log::debug!(
                        "Skipping constant creation for ConstantOfShape {} input {} (handled in check_constants)",
                        node.name,
                        idx
                    );
                    continue;
                }

                // This input is an initializer - create a Constant node for it
                // Use the existing constant naming convention
                self.node_name_counter
                    .entry(NodeType::Constant)
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
                let const_name = format!("constant{}", self.node_name_counter[&NodeType::Constant]);
                log::debug!(
                    "Creating Constant node {} for initializer {} at position {}",
                    const_name,
                    input.name,
                    idx
                );

                // Create a new Constant node
                let mut const_node = Node {
                    node_type: NodeType::Constant,
                    name: const_name.clone(),
                    inputs: vec![],
                    outputs: vec![input.clone()],
                    attrs: HashMap::new(),
                };

                // Set the output name to match what the consuming node expects
                const_node.outputs[0].name = format!("{}_out1", const_name);
                const_node.attrs.insert(
                    "value".to_string(),
                    crate::ir::AttributeValue::Tensor(value.clone()),
                );

                // Add the Constant node to the graph
                graph_data.add_node(const_node);

                // Update the input to reference the Constant node's output
                input.name = format!("{}_out1", const_name);
                input.value = None; // Clear the value since it's now in the Constant node
            }
        }
    }

    fn handle_identity(&mut self, node: &mut Node) {
        if node.node_type == NodeType::Identity {
            // If Identity node has a constant/initializer input, convert it to a Constant node
            if let Some(value) = &node.inputs[0].value {
                log::debug!(
                    "Converting Identity node with constant input to Constant node: {}",
                    &node.name
                );

                // Convert Identity to Constant
                node.node_type = NodeType::Constant;

                // Update the name to use constant naming convention
                self.node_name_counter
                    .entry(NodeType::Constant)
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
                let new_name = format!("constant{}", self.node_name_counter[&NodeType::Constant]);
                log::debug!("Renaming {} to {}", node.name, new_name);
                node.name = new_name;

                // Move the value from input to an attribute
                node.attrs.insert(
                    "value".to_string(),
                    crate::ir::AttributeValue::Tensor(value.clone()),
                );

                // Clear the inputs since Constant nodes don't have inputs
                node.inputs.clear();

                // The output remains the same
            } else {
                // For Identity nodes without constant inputs, we only optimize them away if they are
                // part of a processing chain (not standalone). For now, let's pass them through to
                // burn-import and let it handle them.
                log::debug!(
                    "Identity node without constant - will pass through to burn-import: {}",
                    &node.name
                );
            }
        }
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

/// Remap the unsqueeze node to a reshape node, Should only be called after
/// node renaming has been done. avoids marking rhs as passed so that it can be
/// properly deleted if nothing else uses it
/// Remap the unsqueeze node to a reshape node
pub(crate) fn remap_unsqueeze_to_reshape(node: &mut Node, out_arg: &Argument) {
    if let Some(value) = &out_arg.value {
        let shape_vec = value.shape.clone();
        let inner = shape_vec
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();
        let shape_len = inner.len();
        let new_rhs_value = Some(TensorData {
            shape: vec![shape_len],
            data: Data::Int64s(inner),
        });
        //moving the remap to here
        let rhs_arg = Argument {
            name: format!("{}_generated_const", &node.name),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: Some(vec![shape_len]),
            }),
            value: new_rhs_value,
            passed: false,
        };
        // ? should this replace the old input (reuse the old key) or should it be a new key
        // going with new key for now
        node.inputs[1] = rhs_arg;
        node.outputs[0] = out_arg.clone();
        node.node_type = NodeType::Reshape;
    }
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
