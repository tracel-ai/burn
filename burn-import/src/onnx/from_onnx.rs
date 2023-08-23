use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
    str::{from_utf8, FromStr},
};

use super::dim_inference::dim_inference;
use super::ir::{
    ArgType, Argument, AttributeValue, Attributes, ElementType, Node, NodeType, ONNXGraph, State,
    Tensor, TensorData,
};
use super::protos::{
    attribute_proto::AttributeType, tensor_proto::DataType, tensor_shape_proto::dimension::Value,
    type_proto, AttributeProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
    ValueInfoProto,
};
use super::{coalesce::coalesce, ir::StateType};

use bytemuck::cast_slice;
use protobuf::{Enum, Message};

const LIFT_CONSTANTS_FOR_NODE_TYPES: [NodeType; 5] = [
    NodeType::BatchNormalization,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::Dropout,
    NodeType::Reshape,
];

/// Error type for parsing ONNX model
#[derive(Debug)]
pub enum ParseError {
    VariantNotFound,
}

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
        nodes.push(convert_node_proto(onnx_node));
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

fn to_string(bytes: Vec<u8>) -> String {
    from_utf8(bytes.as_slice()).unwrap().to_string()
}

fn to_string_vec(bytes: Vec<Vec<u8>>) -> Vec<String> {
    bytes.iter().map(|b| to_string(b.clone())).collect()
}

fn convert_shape(shape: Vec<i64>) -> Vec<usize> {
    shape.iter().map(|s| *s as usize).collect()
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<TensorProto> for Tensor {
    type Error = ParseError;
    fn try_from(tensor: TensorProto) -> Result<Tensor, Self::Error> {
        let (elem_type, data) = match DataType::from_i32(tensor.data_type).unwrap() {
            DataType::FLOAT => (
                ElementType::Float32,
                // Convert the raw data to a vector of floats
                if !tensor.raw_data.is_empty() {
                    TensorData::Float32(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    TensorData::Float32(tensor.float_data)
                },
            ),
            DataType::INT32 => (
                ElementType::Int32,
                // Convert the raw data to a vector of ints
                if !tensor.raw_data.is_empty() {
                    TensorData::Int32(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    TensorData::Int32(tensor.int32_data)
                },
            ),
            DataType::INT64 => (
                ElementType::Int64,
                // Convert the raw data to a vector of ints
                if !tensor.raw_data.is_empty() {
                    TensorData::Int64(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    TensorData::Int64(tensor.int64_data)
                },
            ),
            DataType::DOUBLE => (
                ElementType::Float64,
                // Convert the raw data to a vector of floats
                if !tensor.raw_data.is_empty() {
                    TensorData::Float64(cast_slice(&tensor.raw_data[..]).to_vec())
                } else {
                    TensorData::Float64(tensor.double_data)
                },
            ),
            DataType::BOOL => (ElementType::Bool, {
                assert!(!tensor.raw_data.is_empty());
                TensorData::Bool(tensor.raw_data.iter().map(|x| *x != 0).collect())
            }),
            // TODO : Add more types
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };
        let shape = convert_shape(tensor.dims);

        Ok(Tensor {
            elem_type,
            dim: shape.len(),
            shape: Some(shape),
            data: Some(data),
        })
    }
}

impl TryFrom<TensorShapeProto> for Vec<usize> {
    type Error = ParseError;
    fn try_from(shape: TensorShapeProto) -> Result<Vec<usize>, Self::Error> {
        let mut result = Vec::new();

        for dim in shape.dim {
            if let Value::DimValue(value) = dim.value.unwrap() {
                result.push(value as usize);
            }
        }

        Ok(result)
    }
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<&type_proto::Tensor> for Tensor {
    type Error = ParseError;
    fn try_from(tensor: &type_proto::Tensor) -> Result<Tensor, Self::Error> {
        let elem_type = match DataType::from_i32(tensor.elem_type).unwrap() {
            DataType::FLOAT => ElementType::Float32,
            DataType::INT32 => ElementType::Int32,
            DataType::INT64 => ElementType::Int64,
            DataType::DOUBLE => ElementType::Float64,

            // TODO : Add more types
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };

        let shape_proto = tensor.shape.clone().unwrap();
        let shape: Vec<usize> = shape_proto.try_into().unwrap();

        Ok(Tensor {
            elem_type,
            dim: shape.len(),
            shape: Some(shape),
            data: None,
        })
    }
}

fn convert_vec_tensor_proto(tensors: Vec<TensorProto>) -> Result<Vec<Tensor>, ParseError> {
    let mut result = Vec::new();
    for tensor in tensors {
        result.push(Tensor::try_from(tensor)?);
    }
    Ok(result)
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<AttributeProto> for AttributeValue {
    type Error = ParseError;

    fn try_from(attr: AttributeProto) -> Result<AttributeValue, Self::Error> {
        let value = match attr.type_.unwrap() {
            AttributeType::FLOAT => AttributeValue::Float32(attr.f),
            AttributeType::INT => AttributeValue::Int64(attr.i),
            AttributeType::STRING => AttributeValue::String(to_string(attr.s)),

            // warning: tensor can be empty TODO: check if it is empty
            AttributeType::TENSOR => AttributeValue::Tensor(Tensor::try_from(attr.t.unwrap())?),

            // Graph is not supported for now
            // AttributeType::GRAPH => AttributeValue::Graph(attr.g),
            AttributeType::FLOATS => AttributeValue::Float32s(attr.floats),
            AttributeType::INTS => AttributeValue::Int64s(attr.ints),
            AttributeType::STRINGS => AttributeValue::Strings(to_string_vec(attr.strings)),
            AttributeType::TENSORS => {
                AttributeValue::Tensors(convert_vec_tensor_proto(attr.tensors)?)
            }
            // AttributeType::GRAPHS => AttributeValue::Graphs(attr.graphs),
            // AttributeType::SPARSE_TENSORS => AttributeValue::SparseTensors(attr.sparse_tensors),
            // AttributeType::SPARSE_TENSOR => AttributeValue::SparseTensor(attr.sparse_tensor),
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };

        Ok(value)
    }
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
pub fn convert_vec_attrs_proto(attrs: Vec<AttributeProto>) -> Attributes {
    let mut result = Attributes::new();
    for attr in attrs {
        result.insert(attr.name.clone(), AttributeValue::try_from(attr).unwrap());
    }
    result
}

pub fn convert_node_proto(node: &NodeProto) -> Node {
    let name = node.name.clone();

    log::debug!("Converting ONNX node with type {:?}", node.op_type.as_str());

    let inputs = node
        .input
        .clone()
        .into_iter()
        .map(|x| Argument {
            name: x,
            ty: ArgType::Tensor(Tensor::default()),
        })
        .collect();

    let outputs = node
        .output
        .clone()
        .into_iter()
        .map(|x| Argument {
            name: x,
            ty: ArgType::Tensor(Tensor::default()),
        })
        .collect();
    let attrs = convert_vec_attrs_proto(node.attribute.clone());

    let node_type = NodeType::from_str(node.op_type.as_str()).expect("Unknown node type");

    let mut node = Node {
        node_type,
        name,
        inputs,
        outputs,
        states: vec![],
        attrs,
    };

    remap_node_type(&mut node);

    node
}

/// Remap node type using kernel shape
fn remap_node_with_kernel_shape<F>(node: &mut Node, new_node_type: F)
where
    F: FnOnce(&Vec<i64>) -> NodeType,
{
    if let AttributeValue::Int64s(ints) = node.attrs.get("kernel_shape").unwrap() {
        node.node_type = new_node_type(ints);
    } else {
        panic!("kernel_shape is not an int64s");
    }
}

/// Remap node type to a more specific one
fn remap_node_type(node: &mut Node) {
    match node.node_type {
        NodeType::Conv => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::Conv1d,
            2 => NodeType::Conv2d,
            _ => panic!("Only conv 1d and 2d are supported"),
        }),
        NodeType::MaxPool => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::MaxPool1d,
            2 => NodeType::MaxPool2d,
            _ => panic!("Only max_pool 1d and 2d are supported"),
        }),
        NodeType::AveragePool => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::AveragePool1d,
            2 => NodeType::AveragePool2d,
            _ => panic!("Only avg_pool 1d and 2d are supported"),
        }),
        _ => (),
    }
}

impl TryFrom<ValueInfoProto> for Argument {
    type Error = ParseError;

    fn try_from(value: ValueInfoProto) -> Result<Argument, Self::Error> {
        let name = value.name.clone();
        let proto_type = value.type_.unwrap();

        if !proto_type.has_tensor_type() {
            panic!("Unsupported argument type {:?}", proto_type);
        }

        let tensor_proto = proto_type.tensor_type();

        let elem_type = match DataType::from_i32(tensor_proto.elem_type).unwrap() {
            DataType::FLOAT => ElementType::Float32,
            DataType::INT32 => ElementType::Int32,
            DataType::INT64 => ElementType::Int64,
            DataType::DOUBLE => ElementType::Float64,
            DataType::BOOL => ElementType::Bool,
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };

        let tensor: Tensor = Tensor {
            dim: tensor_proto.shape.dim.len(),
            elem_type,
            shape: None,
            data: None,
        };

        let ty = ArgType::Tensor(tensor);

        Ok(Argument { ty, name })
    }
}

impl TryFrom<ValueInfoProto> for State {
    type Error = ParseError;

    fn try_from(value: ValueInfoProto) -> Result<State, Self::Error> {
        let name = value.name.clone();
        let proto_type = value.type_.unwrap();

        if !proto_type.has_tensor_type() {
            panic!("Unsupported argument type {:?}", proto_type);
        }

        let tensor_proto = proto_type.tensor_type();
        let tensor: Tensor = tensor_proto.try_into().unwrap();
        let ty = StateType::Tensor(tensor);

        Ok(State { name, ty })
    }
}

// This function moves inputs that are also present in the initializer to the node's states vector.
// It also removes inputs that are already present in the states vector.
fn move_inputs_to_state(nodes: &mut Vec<Node>, initializer: &[TensorProto]) {
    // Iterate over each node in the graph
    nodes.iter_mut().for_each(|node| {
        // Create a new vector to hold the node's states
        let mut node_states = Vec::new();
        // Create a new vector to hold the node's inputs
        let mut inputs = Vec::new();

        // Iterate over each input in the node's inputs vector
        for input in node.inputs.iter() {
            // Iterate over each tensor in the initializer
            for init in initializer.iter() {
                // If the input name matches the tensor name in the initializer
                if init.name == input.name {
                    // Add the tensor to the node's states vector
                    node_states.push(State {
                        name: init.name.clone(),
                        ty: StateType::Tensor(init.clone().try_into().unwrap()),
                    });
                }
            }
        }

        // Swap the node's inputs vector with the temporary inputs vector
        core::mem::swap(&mut inputs, &mut node.inputs);

        // Filter out inputs that are already present in the node's states vector
        node.inputs = inputs
            .into_iter()
            .filter(|input| {
                for init in node_states.iter() {
                    if init.name == input.name {
                        return false;
                    }
                }

                true
            })
            .collect();

        // Set the node's states vector to the temporary node_states vector
        node.states.append(&mut node_states);
    });
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
        .filter(|node| node.node_type == NodeType::Constant) // filter out non-constant nodes
        .map(|node| (node.outputs[0].name.clone(), node.clone()))
        .collect::<HashMap<String, Node>>();

    // create a set to hold the IDs of constants to be removed
    let mut constant_to_removed = HashSet::<String>::new();

    for node in nodes.iter_mut() {
        // skip if not in the set or len <= 1
        if !node_types_to_process.contains(&node.node_type) || node.inputs.len() <= 1 {
            continue;
        }

        // create a new vector to hold the node's states
        let mut node_states = Vec::new();

        let mut inputs_to_remove = Vec::new();

        // Skip the first input because it is the node's true input and not a constant/state
        node.inputs.iter().skip(1).for_each(|input| {
            if let Some(constant) = constants.get(&input.name) {
                // if the input is a constant, get its ID and node
                let value = get_constant_value(constant).unwrap(); // get the value of the constant

                let state = match value {
                    AttributeValue::Tensor(tensor) => State {
                        // if the value is a tensor, create a new State object with the tensor as its type
                        name: input.name.clone(),
                        ty: StateType::Tensor(tensor),
                    },
                    _ => todo!("Support non tensor constant type"),
                };
                node_states.push(state); // add the new state to the node's states vector
                constant_to_removed.insert(constant.name.clone());
                inputs_to_remove.push(input.name.clone());
            }
        });

        // append the node's states vector to the new vector created in the previous step
        node.states.append(&mut node_states);

        // remove the inputs that were moved to the states vector
        node.inputs.retain(|x| !inputs_to_remove.contains(&x.name))
    }

    // remove the constants that were moved to the states vector
    nodes.retain(|node| !constant_to_removed.contains(&node.name));

    log::debug!(
        "The number of constants removed: {}",
        constant_to_removed.len()
    );
}

/// Handle Identity nodes.
///
/// There are two types of Identity nodes:
/// 1. Pass-through nodes that are used to connect two nodes. These are removed from the graph.
/// 2. Nodes that act as a constant (its input has initializer). Change the node type to Constant.
fn handle_identity(nodes: &mut Vec<Node>) {
    log::info!("Handling identity nodes");

    let mut identity_nodes_to_remove = Vec::new();

    for node in nodes
        .iter_mut()
        .filter(|node| node.node_type == NodeType::Identity)
    {
        // if the node has states, it is a constant. Move the data to value attribute for consistency.
        if let Some(state) = node.states.first() {
            match &state.ty {
                // Currently there is only tensor type
                StateType::Tensor(tensor) => {
                    node.attrs
                        .insert("value".to_string(), AttributeValue::Tensor(tensor.clone()));
                    node.states.clear();
                }
            }
            node.node_type = NodeType::Constant;
            log::debug!("Converted identity node ({}) to constant", node.name);
        } else {
            // Support pass through identity node
            identity_nodes_to_remove.push(node.clone());
        }
    }

    // Remove the identity nodes that are only used to connect two nodes
    for identity_node in identity_nodes_to_remove.iter() {
        let input = identity_node
            .inputs
            .first()
            .expect("Pass through Identity node should have at least one input");
        let output = identity_node
            .outputs
            .first()
            .expect("Pass through Identity node should have at least one ");

        // find the node that uses the identity node's output
        for node in nodes.iter_mut() {
            if let Some(matched_input) = node.inputs.iter_mut().find(|x| x.name == output.name) {
                // replace the identity node's output with the identity node's input
                matched_input.name = input.name.clone();
            }
        }
    }

    // remove the identity nodes that are used to connect two nodes
    nodes.retain(|node| !identity_nodes_to_remove.contains(node));
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

/// Rename the inputs and output in the graph and return a map of the old names to the new names.
///
/// The inputs are renamed to be unique and to be in the format of conv2_in1, conv2_in2, etc.
/// This is done to be consistent with the naming convention of the nodes and allow to be used as rust identifiers.
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
        for input in node.inputs.iter_mut() {
            if let Some(new_name) = old_names.get(&input.name) {
                input.name = new_name.clone();
            } else {
                panic!("Input {} not found in old_names", input.name);
            }
        }
    }

    // Rename the graph outputs
    for output in outputs.iter_mut() {
        if let Some(new_name) = old_names.get(&output.name) {
            output.name = new_name.clone();
        } else {
            panic!("Output {} not found in old_names", output.name);
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
pub(crate) fn get_constant_value(node: &Node) -> Option<AttributeValue> {
    // A value can be stored in any of these attributes
    let value_keys = [
        "value",
        "value_float",
        "value_floats",
        "value_int",
        "value_ints",
        "value_string",
        "value_strings",
        "sparse_value",
    ];

    value_keys
        .iter()
        .find_map(|&key| node.attrs.get(key).cloned())
}
