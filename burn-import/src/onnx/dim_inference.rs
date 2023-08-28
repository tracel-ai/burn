use std::collections::HashMap;

use protobuf::Enum;

use super::{
    ir::{ArgType, Argument, AttributeValue, Data, ElementType, Node, NodeType, TensorType},
    op_configuration::flatten_config,
    protos::tensor_proto::DataType,
};

struct TensorDimUpdater {
    arguments: HashMap<String, Argument>,
}

impl TensorDimUpdater {
    fn new(inputs: &[Argument]) -> Self {
        let mut arguments: HashMap<String, Argument> = HashMap::with_capacity(inputs.len());

        inputs.iter().for_each(|input| {
            arguments.insert(input.name.clone(), input.clone());
        });

        Self { arguments }
    }
    /// Update tensor inputs from the registered arguments and returns the number of input
    /// updated.
    fn update_tensor_inputs(&self, node: &mut Node) -> usize {
        self.update_arguments(&mut node.inputs)
    }

    /// Update the arguments struct from the node output tensors and return the number of output
    /// updated.
    fn update_tensor_outputs(&mut self, node: &Node) -> usize {
        node.outputs
            .iter()
            .map(|arg| {
                self.arguments.insert(arg.name.clone(), arg.clone());
            })
            .count()
    }

    fn update_arguments(&self, arguments: &mut [Argument]) -> usize {
        arguments
            .iter_mut()
            .filter_map(|input| self.arguments.get(&input.name).map(|arg| (arg, input)))
            .map(|(arg, input)| {
                input.ty = arg.ty.clone();
            })
            .count()
    }
}

/// Infer the dimension of each output tensor and update them.
pub fn dim_inference(
    nodes: &mut Vec<Node>,
    graph_inputs: &Vec<Argument>,
    graph_outputs: &mut Vec<Argument>,
) {
    let mut updater = TensorDimUpdater::new(graph_inputs);

    for node in nodes.iter_mut() {
        updater.update_tensor_inputs(node);

        match node.node_type {
            NodeType::Conv1d => conv1d_update_outputs(node),
            NodeType::Conv2d => conv2d_update_outputs(node),
            NodeType::MaxPool2d => same_as_input(node),
            NodeType::Linear => linear_update_outputs(node),
            NodeType::Flatten => flatten_update_outputs(node),
            NodeType::Relu => same_as_input(node),
            NodeType::LogSoftmax => same_as_input(node),
            NodeType::BatchNormalization => same_as_input(node),
            NodeType::Add => same_as_input(node),
            NodeType::Sub => same_as_input(node),
            NodeType::Mul => same_as_input(node),
            NodeType::Cast => cast_update_outputs(node),
            NodeType::Div => same_as_input(node),
            NodeType::Sqrt => same_as_input(node),
            NodeType::Tanh => same_as_input(node),
            NodeType::Softmax => same_as_input(node),
            NodeType::ReduceMean => mean_update_outputs(node),
            NodeType::Constant => constant_update_outputs(node),
            NodeType::Equal => equal_update_outputs(node),
            NodeType::Shape => shape_update_outputs(node),
            NodeType::Unsqueeze => unsqueeze_update_outputs(node),
            NodeType::Sigmoid => same_as_input(node),
            NodeType::Transpose => same_as_input(node),
            NodeType::Concat => concat_update_outputs(node),
            NodeType::Reshape => reshape_update_outputs(node),
            NodeType::Dropout => same_as_input(node),
            NodeType::GlobalAveragePool => same_as_input(node),
            NodeType::AveragePool2d => same_as_input(node),
            NodeType::Clip => same_as_input(node),
            // Intentionally letting outputs leave unchanged but issue a warning so IR file can be generated.
            _ => temporary_pass_through_stub(node),
        }

        updater.update_tensor_outputs(node);
    }

    updater.update_arguments(graph_outputs);
}

fn constant_update_outputs(node: &mut Node) {
    // Fix the tensor dimension of the output when the value is tensor

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

    let matched_value = keys.iter().find_map(|&key| node.attrs.get(key).cloned());

    node.outputs[0].ty = match matched_value {
        Some(value) => match &value {
            // The value is stored in an attribute
            AttributeValue::Tensor(tensor) => ArgType::Tensor(TensorType {
                elem_type: tensor.elem_type.clone(),
                dim: tensor.dim,
                shape: tensor.shape.clone(),
            }),
            AttributeValue::Float32(_) => ArgType::Scalar(ElementType::Float32),
            AttributeValue::Float32s(value) => ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                dim: 1,
                shape: Some(vec![value.len()]),
            }),
            AttributeValue::Int64(_) => ArgType::Scalar(ElementType::Int64),
            AttributeValue::Int64s(value) => ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                dim: 1,
                shape: Some(vec![value.len()]),
            }),
            ty => panic!("Constant value of {:?} is not supported", ty),
        },
        None => panic!("Constant node must have a value attribute"),
    };
}

/// Infer the shape of the output tensor of a Conv2d node
fn linear_update_outputs(node: &mut Node) {
    // Extract the configuration of the linear layer (inputs are known)
    let node_input = &node.inputs[0];
    let weight = &node.inputs[1];

    // Calculate the output shape. Usually we do not use shapes, but since the input shape is
    // known, we can calculate the output shape.
    if let ArgType::Tensor(tensor) = node_input.clone().ty {
        let mut tensor = tensor.clone();
        let mut shape = tensor.shape.clone().unwrap();

        if let ArgType::Tensor(weight_tensor) = weight.clone().ty {
            let last = shape.last_mut().unwrap();
            *last = *weight_tensor.shape.unwrap().first().unwrap();
        } else {
            panic!("Weight must be a tensor");
        }

        tensor.shape = Some(shape);

        // Update the output tensor
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update the output type using "to" attribute
fn cast_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Cast: multiple inputs are not supported");
    }
    let output = &mut node.outputs[0];

    // Extract cast type and update the output tensor
    let elem_type = match node.attrs.get("to") {
        Some(value) => match &value {
            AttributeValue::Int64(type_id) => match DataType::from_i32(*type_id as i32).unwrap() {
                DataType::FLOAT => ElementType::Float32,
                DataType::INT32 => ElementType::Int32,
                DataType::INT64 => ElementType::Int64,
                DataType::DOUBLE => ElementType::Float64,
                _ => panic!("Cast: unsupported type"),
            },
            _ => panic!("'to' attribute must be an Int64"),
        },
        None => panic!("Constant node must have a value attribute"),
    };

    match output.ty.clone() {
        ArgType::Tensor(tensor) => {
            if tensor.dim == 0 {
                // treat 0-dim tensor as scalar
                output.ty = ArgType::Scalar(elem_type);
            } else {
                todo!("Cast: support casting from different tensor types");
            }
        }
        ArgType::Scalar(_scalar) => {
            output.ty = ArgType::Scalar(elem_type);
        }
        _ => panic!("Cast: only scalar input is valid"),
    }
}

fn concat_update_outputs(node: &mut Node) {
    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match &input.ty {
            ArgType::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .unwrap();

    node.outputs[0].ty = ArgType::Tensor(tensor.clone());
}

fn reshape_update_outputs(node: &mut Node) {
    assert_eq!(node.inputs.len(), 2);

    let shape = if let Some(Data::Int64s(ref shape)) = node.inputs[1].value {
        shape
    } else {
        panic!("Reshape: int64s shape is expected per ONNX spec");
    };

    // The output dimension is the same as the shape length
    let dim = shape.len();
    let elem_type = match node.inputs[0].ty.clone() {
        ArgType::Tensor(tensor) => tensor.elem_type,
        _ => panic!("Reshape: invalid input type"),
    };

    let shape = shape.iter().map(|&dim| dim as usize).collect();

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type,
        dim,
        shape: Some(shape),
    });
}

fn mean_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Mean: multiple inputs are not supported");
    }

    // Extract the configuration of the linear layer (inputs are known)
    let node_input = &mut node.inputs[0];
    let tensor = match node_input.clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    let dim_only = match node.attrs.get("axes") {
        Some(value) => match &value {
            AttributeValue::Int64(_) => true,
            AttributeValue::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => false,
    };

    if dim_only {
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType { dim: 1, ..tensor });
    }
}
/// Infers the shape of a Unsqueeze node and replaces the shape of the output tensor.
///
/// # Remarks
///
/// Unsqueeze is not implemented fully. This is left WIP from the past.
fn unsqueeze_update_outputs(node: &mut Node) {
    let node_input = node
        .inputs
        .first_mut()
        .expect("Unsqueeze: an input is required");

    if let ArgType::Tensor(tensor) = &mut node_input.ty {
        tensor.dim += 1;

        // add a new dimension to the input tensor by extending the shape
        // TODO: support unsqueezing configurations
        if let Some(shape) = &mut tensor.shape {
            shape.insert(0, 1);
        } else {
            todo!("Unsqueeze: support unsqueezing a tensor without shape");
        }

        node.outputs[0].ty = ArgType::Tensor(tensor.clone());
    } else {
        panic!("Only tensor input is valid");
    }
}

fn same_as_input(node: &mut Node) {
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

/// Temporary pass-through stub for dimension inference so that we can export the IR model.
fn temporary_pass_through_stub(node: &mut Node) {
    log::warn!(
        "Must implement dimension inference for {:?}",
        node.node_type
    );
}

fn equal_update_outputs(node: &mut Node) {
    let input1_type = node.inputs[0].ty.clone();

    match input1_type {
        ArgType::Tensor(tensor) => {
            // if the input is a tensor, the output is a tensor of bool
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor
            });
        }
        ArgType::Scalar(_) => {
            node.outputs[0].ty = ArgType::Scalar(ElementType::Bool);
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn shape_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Gather: multiple inputs are not supported: {:?}", node);
    }

    // Extract the configuration of the linear layer (inputs are known)
    let node_input = &mut node.inputs[0];
    if let ArgType::Tensor(tensor) = node_input.clone().ty {
        // Update the output tensor
        node.outputs[0].ty = ArgType::Shape(tensor.dim);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a Flatten node and replaces the shape of the output tensor.
fn flatten_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Flatten: multiple inputs are not supported");
    }
    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match &input.ty {
            ArgType::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .unwrap();

    let input_dim = tensor.dim;

    let (start_dim, end_dim) = flatten_config(node);

    let collapsed_dims = end_dim - start_dim;
    let output_dim = input_dim - collapsed_dims;

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        dim: output_dim,
        ..tensor.clone()
    });
}

/// Infers the shape of a Conv1d node and replaces the shape of the output tensor.
fn conv1d_update_outputs(node: &mut Node) {
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(tensor) = node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a Conv2d node and replaces the shape of the output tensor.
fn conv2d_update_outputs(node: &mut Node) {
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(tensor) = node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        panic!("Only tensor input is valid");
    }
}
