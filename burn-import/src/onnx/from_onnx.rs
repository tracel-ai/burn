use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

use crate::onnx::{
    coalesce::coalesce, ir::TensorType, node_remap::remap_node_type,
    proto_conversion::convert_node_proto,
};

use super::dim_inference::dim_inference;
use super::ir::{ArgType, Argument, Node, NodeType, ONNXGraph, Tensor};
use super::protos::{ModelProto, TensorProto};

use protobuf::Message;

const LIFT_CONSTANTS_FOR_NODE_TYPES: [NodeType; 6] = [
    NodeType::BatchNormalization,
    NodeType::Clip,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::Dropout,
    NodeType::Reshape,
];

/// Open an onnx file and convert it to a Graph (intermediate representation)
///
/// # Arguments
///
/// * `onnx_path` - Path to the onnx file
///
/// # Returns
///
/// * `ONNXGraph` - The graph representation of the onnx file
///
/// # Panics
///
/// * If the file cannot be opened
/// * If the file cannot be parsed
/// * If the nodes are not topologically sorted
pub fn parse_onnx(onnx_path: &Path) -> ONNXGraph {
    log::info!("Parsing ONNX file: {}", onnx_path.display());

    // Open the file
    let mut file = File::open(onnx_path).expect("Unable to open file");
    let onnx_model: ModelProto =
        Message::parse_from_reader(&mut file).expect("Unable to parse ONNX file");

    log::debug!("Number of nodes: {:?}", onnx_model.graph.node.len());
    log::debug!("Number of inputs: {:?}", onnx_model.graph.input.len());

    log::debug!(
        "Number of initializers: {:?}",
        onnx_model.graph.initializer.len()
    );

    log::debug!("Number of outputs: {:?}", onnx_model.graph.output.len());

    // Convert the nodes
    let mut nodes: Vec<Node> = vec![];
    for onnx_node in onnx_model.graph.node.iter() {
        let mut node = convert_node_proto(onnx_node);
        remap_node_type(&mut node);
        nodes.push(node);
    }

    // ONNX nodes must be topologically sorted per spec:
    // https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    assert!(nodes.is_top_sorted(), "Nodes are not topologically sorted");

    // Move inputs with initializers to states
    move_inputs_to_state(&mut nodes, &onnx_model.graph.initializer);

    // Handle Identity nodes (expects inputs to be moved to states)
    handle_identity(&mut nodes);

    // Lift constants to initializers (expects inputs to be moved to states)
    lift_constants(&mut nodes);

    // Coalesce and transform nodes
    coalesce(&mut nodes);

    // Rename nodes and inputs, save the mapping for later
    let old_node_names = rename_nodes(&mut nodes);

    // This function collects the inputs of an ONNX model and returns them as a vector of Arguments.
    let mut inputs = onnx_model
        .graph
        .input
        .iter()
        .map(|x| Argument::try_from(x.clone()).unwrap())
        .collect();

    // Map each output in the model's graph to an Argument and collect them into a vector.
    let mut outputs = onnx_model
        .graph
        .output
        .iter()
        .map(|x| Argument::try_from(x.clone()).unwrap())
        .collect();

    let old_input_names = rename_inputs(&mut nodes, &mut inputs, &mut outputs);

    // Infer shapes and update the inputs and outputs
    dim_inference(&mut nodes, &inputs, &mut outputs);

    // Remove the graph inputs/output that are not used by any node
    remove_unused_graph_inputs(&mut inputs, &mut outputs, &nodes);

    log::info!("Finished parsing ONNX file: {}", onnx_path.display());

    ONNXGraph {
        nodes,
        inputs,
        outputs,
        old_node_names,
        old_input_names,
    }
}

/// This function moves inputs that are also present
/// in the initializer to the node's states vector.
/// It also removes inputs that are already present in the states vector.
///
/// # Arguments
///
/// * `nodes` - A mutable reference to a vector of nodes
/// * `initializers` - A vector of TensorProto
fn move_inputs_to_state(nodes: &mut Vec<Node>, initializers: &[TensorProto]) {
    // Convert initializers to hashmap for faster lookup
    let initializers = initializers
        .iter()
        .map(|x| (x.name.clone(), x.clone()))
        .collect::<HashMap<String, TensorProto>>();

    // Iterate over each node in the graph
    nodes.iter_mut().for_each(|node| {
        for input in node.inputs.iter_mut() {
            // If there is a corresponding initializer for the input, then move the data to the input value
            if let Some(initializer) = initializers.get(&input.name) {
                move_initializer_data(initializer, input);
            }
        }
    });
}

fn move_initializer_data(initializer: &TensorProto, input: &mut Argument) {
    // If the input name matches the tensor name in the initializer
    // Convert the initializer to a tensor
    let tensor = Tensor::try_from(initializer.clone()).expect("Invalid tensor");

    if tensor.dim == 0 {
        // Convert zero dim tensor to scalar
        if let Some(data) = tensor.data {
            input.value = Some(data.into_scalar());
        } else {
            input.value = None;
        }

        // Update the input type
        input.ty = ArgType::Scalar(tensor.elem_type);
    } else {
        // Move the tensor data to the input value
        input.value = tensor.data.clone();

        // Update the input type
        input.ty = ArgType::Tensor(TensorType {
            dim: tensor.dim,
            elem_type: tensor.elem_type,
            shape: tensor.shape,
        });
    }
}

/// Lift constants from the graph into the states vector for known node types.
///
/// The primary reason to move constants into the states vector is to reduce the number of nodes in the graph,
/// and consistently utilize the same interface for all nodes (constant inputs and inputs with initializers are
/// treated the same way). This simplification aids code generation.
///
/// For example, if we have a graph ([Const1, Const2, Conv2d1]) where the Conv2d node has 3 inputs
/// (graph_input, const2_out1, const_out2), we can lift the constants into the states of the Conv2d node.
/// const2_out1 and const_out2 are used for the weights and bias of the Conv2d node.
/// After lifting, we will have a graph ([Conv2d1]) where the Conv2d node has 1 input (graph_input) and 2 states.
///
/// Also note that often times, Conv2d node's inputs are not constants, but they are initializers. Initializers
/// move to the states vector as well, using the `move_inputs_to_state` function.
///
///
/// # Arguments
///
/// * `nodes` - A mutable reference to a vector of nodes
///
/// # Panics
///
/// Panics if the node's output is not a constant.
fn lift_constants(nodes: &mut Vec<Node>) {
    log::info!("Lifting constants into the states");

    // create a set to hold the node types to process
    let node_types_to_process: HashSet<NodeType> =
        LIFT_CONSTANTS_FOR_NODE_TYPES.into_iter().collect();

    // create a new vector to hold the graph's constants (index by the node's name)
    let constants = nodes
        .iter()
        .filter(|node| node.node_type == NodeType::Constant || node.node_type == NodeType::Identity)
        .map(|node| (node.outputs[0].name.clone(), node.clone()))
        .collect::<HashMap<String, Node>>();

    // create a set to hold the IDs of constants to be removed
    let mut constant_to_removed = HashSet::<String>::new();

    for node in nodes.iter_mut() {
        // Skip the node if it is not in the set of node types to process
        if !node_types_to_process.contains(&node.node_type) {
            continue;
        }

        // Skip the first input because it is the node's true input and not a constant/state
        node.inputs
            .iter_mut()
            .skip(1) // TODO make configurable
            .for_each(|input| {
                if let Some(constant) = constants.get(&input.name) {
                    if !constant.inputs.is_empty() && constant.inputs[0].value.is_some() {
                        // The value comes from Identity inputs
                        if let Some(constant_input) = constant.inputs.first() {
                            input.ty = constant_input.ty.clone();
                            input.value = constant_input.value.clone();
                        }
                    } else {
                        // The value comes from an attribute
                        let arg = convert_constant_value(constant); // get the value of the constant

                        input.value = arg.value; // set the input's value to the constant's value
                        input.ty = arg.ty; // set the input's type to the constant's type
                                           // remove the constant from the graph
                    }
                    constant_to_removed.insert(constant.name.clone());
                }
            });
    }

    // remove the constants that were moved to the states vector
    nodes.retain(|node| !constant_to_removed.contains(&node.name));

    log::debug!(
        "The number of constants lifted: {}",
        constant_to_removed.len()
    );
}

fn handle_identity(nodes: &mut Vec<Node>) {
    log::info!("Handling identity nodes");

    let mut nodes_to_remove = HashSet::new();

    let identity_nodes = nodes
        .iter()
        .filter(|node| node.node_type == NodeType::Identity)
        .cloned()
        .collect::<Vec<Node>>();

    // Handle pass-through nodes.
    for identity_node in identity_nodes {
        if identity_node.node_type == NodeType::Identity && identity_node.inputs[0].value.is_none()
        {
            let input_name = &identity_node.inputs[0].name;
            let output_name = &identity_node.outputs[0].name;

            // Replace the identity node's output with its input in the connected nodes.
            for node in nodes.iter_mut() {
                if let Some(matched_input) = node.inputs.iter_mut().find(|x| x.name == *output_name)
                {
                    matched_input.name = input_name.clone();
                }
            }

            nodes_to_remove.insert(identity_node);
        }
    }

    // Remove the identity nodes.
    nodes.retain(|node| !nodes_to_remove.contains(node));
}

/// Rename the nodes in the graph to be unique and return a map of the old names to the new names.
fn rename_nodes(nodes: &mut Vec<Node>) -> HashMap<String, String> {
    let mut old_names = HashMap::new();
    let mut counter: HashMap<NodeType, usize> = HashMap::new();

    for node in nodes.iter_mut() {
        // keep track of the number of nodes of each type
        counter
            .entry(node.node_type.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);

        let old_name = node.name.clone();
        let new_name = format!("{}{}", node.node_type, counter[&node.node_type]).to_lowercase();

        node.name = new_name.clone();

        old_names.insert(old_name, new_name);
    }

    old_names
}

/// Rename the inputs and output in the graph and return a map of
/// the old names to the new names.
///
/// The inputs are renamed to be unique and to be in the format of
/// conv2_in1, conv2_in2, etc. This is done to be consistent with
/// the naming convention of the nodes and allow to be used as rust identifiers.
fn rename_inputs(
    nodes: &mut Vec<Node>,
    inputs: &mut Vec<Argument>,
    outputs: &mut Vec<Argument>,
) -> HashMap<String, String> {
    let mut old_names = HashMap::new();

    // rename all graph input names to follow input1, input2, input3, etc.
    // (assumes the input names are already unique)
    let mut counter = 1;
    for input in inputs.iter_mut() {
        let old_name = input.name.clone();
        let new_name = format!("input{}", counter);
        input.name = new_name.clone();
        old_names.insert(old_name, new_name);
        counter += 1;
    }

    for node in nodes.iter_mut() {
        let mut counter = 1;

        // loop through node outputs and rename them and store the new name <-> old name mapping
        for output in node.outputs.iter_mut() {
            let old_name = output.name.clone();
            let new_name = format!("{}_out{}", node.name, counter);
            output.name = new_name.clone();
            old_names.insert(old_name, new_name);
            counter += 1;
        }
    }

    for node in nodes.iter_mut() {
        // loop through node inputs and rename them with previously replaced names
        // and mark them as passed if they are in the old_names map (i.e. they are node outputs)
        for input in node.inputs.iter_mut() {
            if let Some(new_name) = old_names.get(&input.name) {
                input.name = new_name.clone();
                input.passed = true;
            } else {
                input.name = "".to_string(); // Rename to a placeholder
                input.passed = false;
            }
        }
    }

    // Rename the graph outputs
    for output in outputs.iter_mut() {
        if let Some(new_name) = old_names.get(&output.name) {
            output.name = new_name.clone();
        } else {
            log::warn!("Output {:?} not found in old_names", output.name);
        }
    }

    old_names
}

/// Removes the graph inputs/output that are not used by any node.
///
/// In older ONNX models, the inputs and outputs are not always used by the nodes.
/// For example, the input could be used as a state instead of an input. Since the
/// inputs with initializers are moved to the states vector, the inputs vector could
/// contain unused inputs. The same is true for the outputs.
///
/// Generally, it's a good idea to remove unused inputs/outputs because it makes the
/// generated code cleaner and easier to read.
fn remove_unused_graph_inputs(
    inputs: &mut Vec<Argument>,
    outputs: &mut Vec<Argument>,
    nodes: &Vec<Node>,
) {
    // Remove inputs that are not used by any node
    inputs.retain(|input| {
        for node in nodes.iter() {
            if node.inputs.iter().any(|x| x.name == input.name) {
                return true;
            }
        }
        false
    });

    // Remove outputs that are not used by any node
    outputs.retain(|output| {
        for node in nodes.iter() {
            if node.outputs.iter().any(|x| x.name == output.name) {
                return true;
            }
        }
        false
    });
}

// Define a trait for topological sorting
trait TopologicalSortable {
    fn is_top_sorted(&self) -> bool;
}

impl TopologicalSortable for Vec<Node> {
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
            for output in &node.outputs {
                // Iterate over each other node in the vector
                for other_node in self {
                    // If the other node has an input that matches the current output
                    if other_node.inputs.contains(output) {
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
