use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
    str::{from_utf8, FromStr},
};

use super::coalesce::coalesce;
use super::ir::{
    ArgType, Argument, AttributeValue, Attributes, ElementType, Node, NodeType, ONNXGraph, Tensor,
    TensorData,
};
use super::protos::{
    attribute_proto::AttributeType, tensor_proto::DataType, tensor_shape_proto::dimension::Value,
    type_proto, AttributeProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
    ValueInfoProto,
};
use super::shape_inference::shape_inference;

use bytemuck::cast_slice;
use protobuf::{Enum, Message};
use topological_sort::TopologicalSort;

const STATEFUL_NODE_TYPES: [NodeType; 6] = [
    NodeType::Conv,
    NodeType::Conv1d,
    NodeType::Conv2d,
    NodeType::BatchNormalization,
    NodeType::Dropout,
    NodeType::Linear,
];

/// Error type for parsing ONNX model
#[derive(Debug)]
pub enum ParseError {
    VariantNotFound,
}

/// Open an onnx file and convert it to a Graph (intermediate representation)
pub fn parse_onnx(onnx_path: &Path) -> ONNXGraph {
    // Open the file
    let mut file = File::open(onnx_path).expect("Unable to open file");
    let onnx_model: ModelProto =
        Message::parse_from_reader(&mut file).expect("Unable to parse ONNX file");

    // Convert the nodes
    let mut nodes: Vec<Node> = vec![];
    for onnx_node in onnx_model.graph.node.iter() {
        nodes.push(convert_node_proto(onnx_node));
    }

    // Get the names of the initializers
    let check_if_initializer: HashSet<String> = onnx_model
        .graph
        .initializer
        .iter()
        .map(|x| x.name.clone())
        .collect();

    // Move inputs to initializers
    move_inputs_to_initializer(&mut nodes, &check_if_initializer);

    // Get the topological sort of the nodes and the top nodes
    let (ts, top_nodes) = get_top_nodes(&nodes);

    // Sort the nodes
    top_sort_nodes(&mut nodes, ts);

    // Collect inputs, outputs and initializers
    let mut inputs = collect_inputs(&onnx_model, &check_if_initializer, top_nodes);
    let mut outputs = collect_outputs(&onnx_model, check_if_initializer);
    let initializers = collect_initializers(onnx_model);

    // Copy the initializers to the nodes
    copy_initializer_info_to_nodes_level(&mut nodes, &initializers);

    // Coalesce and transform nodes
    coalesce(&mut nodes);

    // Rename nodes and inputs, save the mapping for later
    let old_node_names = rename_nodes(&mut nodes);
    let old_input_names = rename_inputs(&mut nodes, &mut inputs, &mut outputs);

    // Infer shapes and update the inputs and outputs
    shape_inference(&mut nodes, &inputs, &mut outputs);

    ONNXGraph {
        nodes,
        inputs,
        outputs,
        initializers,
        old_node_names,
        old_input_names,
    }
}

/// Collect initializers
fn collect_initializers(onnx_model: ModelProto) -> Vec<Argument> {
    let mut initializers: Vec<Argument> = vec![];
    for initializer in onnx_model.graph.initializer.iter() {
        let tensor_proto = initializer.clone();

        let name = tensor_proto.name.clone();

        // FIXME data conversion for the tensor is incorrect
        let tensor: Tensor = tensor_proto.try_into().unwrap();
        let arg_type = Some(ArgType::Tensor(tensor));
        let arg = Argument { name, arg_type };
        initializers.push(arg);
    }
    initializers
}

/// Collect outputs
fn collect_outputs(
    onnx_model: &ModelProto,
    check_if_initializer: HashSet<String>,
) -> Vec<Argument> {
    // TODO: filter out the outputs that are not used in the graph
    let outputs: Vec<Argument> = onnx_model
        .graph
        .output
        .iter()
        .filter(|x| !check_if_initializer.contains(x.name.as_str()))
        .map(|i| Argument::try_from(i.clone()).unwrap())
        .collect();
    outputs
}

/// Collect inputs
fn collect_inputs(
    onnx_model: &ModelProto,
    check_if_initializer: &HashSet<String>,
    top_nodes: HashSet<String>,
) -> Vec<Argument> {
    let inputs: Vec<Argument> = onnx_model
        .graph
        .input
        .iter()
        .filter(|x| !check_if_initializer.contains(x.name.as_str()))
        .filter(|x| top_nodes.contains(&x.name))
        .map(|x| Argument::try_from(x.clone()).unwrap())
        .collect();
    inputs
}

/// Sort the nodes in topological order
fn top_sort_nodes(nodes: &mut Vec<Node>, mut ts: TopologicalSort<Node>) {
    *nodes = vec![];
    while let Some(node) = ts.pop() {
        nodes.push(node);
    }
}

/// Get the top nodes in the graph
fn get_top_nodes(nodes: &Vec<Node>) -> (TopologicalSort<Node>, HashSet<String>) {
    // Get the names of the top nodes (first nodes in the graph to receive the input)
    // Sometimes onnx will pass inputs to be used as weights and biases but they are not truly inputs
    let ts = topsort(nodes);
    let mut top_nodes: HashSet<String> = HashSet::new();

    for node in ts.peek_all() {
        for input in node.inputs.iter() {
            top_nodes.insert(input.name.clone());
        }
    }
    (ts, top_nodes)
}

/// Move nodes's inputs and outputs to initializers if they are in the initializer list
fn move_inputs_to_initializer(nodes: &mut Vec<Node>, check_if_initializer: &HashSet<String>) {
    for node in nodes.iter_mut() {
        node.initializers = node
            .inputs
            .iter()
            .filter(|x| check_if_initializer.contains(&x.name))
            .cloned()
            .collect();

        // Remove the initializers from the inputs and outputs
        node.inputs
            .retain(|x| !check_if_initializer.contains(&x.name));
        node.outputs
            .retain(|x| !check_if_initializer.contains(&x.name));
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
            // TODO : Add more types
            _ => {
                return Err(ParseError::VariantNotFound);
            }
        };
        let shape = convert_shape(tensor.dims);
        let name = tensor.name;

        Ok(Tensor {
            name: Some(name),
            elem_type,
            shape,
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

        let name = None;

        Ok(Tensor {
            name,
            elem_type,
            shape,
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
    let inputs = node
        .input
        .clone()
        .into_iter()
        .map(|x| Argument {
            name: x,
            arg_type: None,
        })
        .collect();
    let outputs = node
        .output
        .clone()
        .into_iter()
        .map(|x| Argument {
            name: x,
            arg_type: None,
        })
        .collect();
    let attrs = convert_vec_attrs_proto(node.attribute.clone());

    let node_type = NodeType::from_str(node.op_type.as_str()).unwrap();

    let is_stateful = STATEFUL_NODE_TYPES.contains(&node_type);

    let mut node = Node {
        node_type,
        name,
        inputs,
        outputs,
        initializers: vec![],
        attrs,
        is_stateful,
    };

    remap_node_type(&mut node);

    node
}

/// Remap node type to a more specific one
fn remap_node_type(node: &mut Node) {
    match node.node_type {
        NodeType::Conv => {
            if let AttributeValue::Int64s(ints) = node.attrs.get("kernel_shape").unwrap() {
                node.node_type = match ints.len() {
                    1 => NodeType::Conv1d,
                    2 => NodeType::Conv2d,
                    _ => todo!(),
                };
            } else {
                panic!("kernel_shape is not an int64s");
            }
        }
        _ => (),
    }
}

/// Convert a vector of AttributeProto to a HashMap of AttributeValue
impl TryFrom<ValueInfoProto> for Argument {
    type Error = ParseError;

    fn try_from(value: ValueInfoProto) -> Result<Argument, Self::Error> {
        let name = value.name.clone();
        let proto_type = value.type_.unwrap();

        let mut arg_type = None;

        if proto_type.has_tensor_type() {
            let tensor_proto = proto_type.tensor_type();

            let tensor: Tensor = tensor_proto.try_into().unwrap();

            arg_type = Some(ArgType::Tensor(tensor));
        }
        Ok(Argument { name, arg_type })
    }
}

/// Copy the initializers to the nodes
fn copy_initializer_info_to_nodes_level(nodes: &mut Vec<Node>, initializers: &Vec<Argument>) {
    for node in nodes.iter_mut() {
        for node_initializer in node.initializers.iter_mut() {
            *node_initializer = initializers
                .iter()
                .find(|x| x.name == node_initializer.name)
                .unwrap()
                .clone();
        }
    }
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

/// Rename the inputs in the graph and return a map of the old names to the new names.
///
/// The inputs are renamed to be unique and to be in the format of conv2_in1, conv2_in2, etc.
/// This is done to be consistent with the naming convention of the nodes and allow to be used as rust identifiers.
fn rename_inputs(
    nodes: &mut Vec<Node>,
    inputs: &mut Vec<Argument>,
    outputs: &mut Vec<Argument>,
) -> HashMap<String, String> {
    let mut old_names = HashMap::new();

    let mut counter = 1;
    for input in inputs.iter_mut() {
        let old_name = input.name.clone();
        let new_name = format!("input{}", counter);

        input.name = new_name.clone();

        old_names.insert(old_name, new_name);
        counter += 1;
    }

    let mut counter: HashMap<String, usize> = HashMap::new();

    for node in nodes.iter_mut() {
        // keep track of the number of nodes of each type
        counter
            .entry(node.name.clone())
            .and_modify(|e| *e += 1)
            .or_insert(1);

        // loop through node inputs and rename them with previously replaced names
        for input in node.inputs.iter_mut() {
            if let Some(new_name) = old_names.get(&input.name) {
                input.name = new_name.clone();
            }
        }

        // loop through node outputs and rename them and store the new name <-> old name mapping
        for output in node.outputs.iter_mut() {
            let old_name = output.name.clone();
            let new_name = format!("{}_out{}", node.name, counter[&node.name]);
            output.name = new_name.clone();
            old_names.insert(old_name, new_name);
        }
    }

    // Rename the graph outputs
    for output in outputs.iter_mut() {
        if let Some(new_name) = old_names.get(&output.name) {
            output.name = new_name.clone();
        }
    }

    old_names
}

/// Find the node that produces the given output
fn lookup_node_by_output(nodes: &Vec<Node>, input: &str) -> Option<Node> {
    for node in nodes.iter() {
        if node.outputs.iter().any(|x| x.name == *input) {
            return Some(node.clone());
        }
    }
    None
}

/// Sort nodes in topological order
pub fn topsort(nodes: &Vec<Node>) -> TopologicalSort<Node> {
    let mut ts = TopologicalSort::new();

    for node in nodes.iter() {
        for input in node.inputs.iter() {
            match lookup_node_by_output(nodes, input.name.as_str()) {
                Some(prec) => ts.add_dependency(prec, node.clone()),
                None => {}
            }
        }
    }

    ts
}
