use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::onnx::node_remap::remap_node_type;

use super::{
    coalesce::coalesce,
    ir::{Data, OnnxGraph, TensorType},
    proto_conversion::convert_node_proto,
    protos::{ModelProto, NodeProto, TensorProto, ValueInfoProto},
};

use super::dim_inference::dim_inference;
use super::ir::{ArgType, Argument, Node, NodeType};

use protobuf::Message;

const LIFT_CONSTANTS_FOR_NODE_TYPES: [NodeType; 12] = [
    NodeType::BatchNormalization,
    NodeType::Clip,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::Dropout,
    NodeType::Expand,
    NodeType::Reshape,
    NodeType::Resize,
    NodeType::Unsqueeze,
    NodeType::ReduceSum,
    NodeType::Slice,
    NodeType::Squeeze,
];

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
        inputs: &Vec<ValueInfoProto>,
        outputs: &Vec<ValueInfoProto>,
        initializers: &Vec<TensorProto>,
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
                if let Some(initial_arg) = constants.get(&x.name) {
                    if arg.value.is_none() {
                        log::warn!("Input {} is also an initializer. Initializer as default values are currently not supported", x.name);
                        arg.copy_value(initial_arg);
                    }
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
                    log::warn!(
                        "Input {} not found, should only happen when peeking",
                        proto_str
                    );
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
            if let Some(old_input_name) = self.input_key_map.get(&node_input.name) {
                if !self.initializers.contains_key(old_input_name) {
                    match self.input_name_map.get(old_input_name) {
                        Some(IOEntry::In(i)) => self.inputs[*i].passed = true,
                        _ => {
                            panic!("Should not happen, please report this error");
                        }
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
        log::debug!("adding node {:?}", &node.name);
        self.mark_input_passed(&node);
        let mut out_count = 1;
        for output in node.outputs.iter_mut() {
            self.input_name_map.insert(
                output.name.clone(),
                IOEntry::Node(self.processed_nodes.len(), 0),
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
    /// Map from identity node output names to indices of identity nodes
    identity_idx: HashMap<String, usize>,
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
            self.handle_identity(&mut node, &graph_data);
            self.check_constants(&mut node, &graph_data);
            // NOTE: potential start of custom functions
            // can filter, coalesce, or modify the nodes here
            // args : node, peek_iter, graph_data
            self.handle_unsqueeze(&mut node, &graph_data);

            dim_inference(&mut node);
            graph_data.add_node(node);
        }

        let (mut processed_nodes, inputs, outputs) = graph_data.consume();
        // Remove the graph inputs/output that are not used by any node
        let mut i = 0;
        processed_nodes.retain(|_| {
            let keep = !self.nodes_to_remove.contains(&i);
            i += 1;
            keep
        });
        OnnxGraph {
            nodes: processed_nodes,
            inputs,
            outputs,
        }
    }

    fn handle_node_renaming(&mut self, node: &mut Node) {
        log::debug!("renaming node {:?}", &node.name);
        self.node_name_counter
            .entry(node.node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);
        let new_name = format!(
            "{}{}",
            node.node_type, self.node_name_counter[&node.node_type]
        )
        .to_lowercase();
        node.name.clone_from(&new_name);
    }

    fn check_constants(&mut self, node: &mut Node, graph_data: &GraphData) {
        if node.node_type == NodeType::Constant
            || (node.node_type == NodeType::Identity && node.inputs[0].value.is_some())
        {
            self.constants_map.insert(
                format!("{}_out{}", &node.name, 1),
                graph_data.get_current_index(),
            );
        } else if self.constants_types.contains(&node.node_type) {
            log::debug!("checking node {} for constants", &node.name);
            for input in node.inputs.iter_mut().skip(1) {
                log::debug!("checking input {:?} for const", input);
                if let Some(const_idx) = self.constants_map.get(&input.name) {
                    let constant = &graph_data.processed_nodes[*const_idx];
                    log::debug!(
                        "input {} matched constant node {}",
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

    fn handle_identity(&mut self, node: &mut Node, graph_data: &GraphData) {
        if node.node_type == NodeType::Identity && node.inputs[0].value.is_none() {
            log::debug!("\nfound identity node:\n{:?}\n", &node);
            let i = graph_data.get_current_index();
            //map the output name to check for pass through values
            self.identity_idx.insert(format!("{}_out1", &node.name), i);
            self.nodes_to_remove.insert(i);
        } else {
            node.inputs.iter_mut().for_each(|x| {
                if let Some(identity_idx) = self.identity_idx.get(&x.name) {
                    let input_name = &graph_data.processed_nodes[*identity_idx].inputs[0].name;

                    x.name.clone_from(input_name);
                }
            });
        }
    }
}

/// Open an onnx file and convert it to a Graph (intermediate representation)
///
/// # Arguments
///
/// * `onnx_path` - Path to the onnx file
///
/// # Returns
///
/// * `OnnxGraph` - The graph representation of the onnx file
///
/// # Panics
///
/// * If the file cannot be opened
/// * If the file cannot be parsed
/// * If the nodes are not topologically sorted
pub fn parse_onnx(onnx_path: &Path) -> OnnxGraph {
    log::info!("Parsing ONNX file: {}", onnx_path.display());

    // Open the file
    let mut file = File::open(onnx_path).expect("Unable to open file");
    let onnx_model: ModelProto =
        Message::parse_from_reader(&mut file).expect("Unable to parse ONNX file");

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
    match &out_arg.ty {
        ArgType::Tensor(output_tensor) => {
            let inner = output_tensor
                .shape
                .clone()
                .unwrap()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>();
            let shape_len = inner.len();
            let new_rhs_value = Some(Data::Int64s(inner));
            //moving the remap to here
            let rhs_arg = Argument {
                name: format!("{}_generated_const", &node.name),
                ty: ArgType::Tensor(TensorType {
                    elem_type: super::ir::ElementType::Int64,
                    dim: 1,
                    shape: Some(vec![shape_len]),
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
        _ => {}
    }
}
// Define a trait for topological sorting
trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

impl TopologicalSortable for Vec<NodeProto> {
    fn is_top_sorted(&self) -> bool {
        // Create a hashmap to store the position of each node in the vector
        let position: HashMap<String, usize> = self
            .iter()
            .enumerate()
            .map(|(idx, node)| (node.name.clone(), idx))
            .collect();

        // Iterate over each node in the vector
        for node in self {
            // Iterate over each output of the node
            for output in &node.output {
                // Iterate over each other node in the vector
                for other_node in self {
                    // If the other node has an input that matches the current output
                    if other_node.input.contains(output) {
                        // If the position of the current node is greater than the position of the other node
                        if position[&node.name] > position[&other_node.name] {
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
pub(crate) fn convert_constant_value(node: &Node) -> Argument {
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
