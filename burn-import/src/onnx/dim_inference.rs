use std::collections::HashMap;

use protobuf::Enum;

use super::{
    from_onnx::get_constant_value,
    ir::{
        ArgType, Argument, AttributeValue, ElementType, Node, NodeType, StateType, Tensor,
        TensorData,
    },
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
            NodeType::Pow => same_as_input(node),
            NodeType::Mul => same_as_input(node),
            NodeType::Cast => cast_update_outputs(node),
            NodeType::Div => same_as_input(node),
            NodeType::Sqrt => same_as_input(node),
            NodeType::Tanh => same_as_input(node),
            NodeType::Softmax => same_as_input(node),
            NodeType::Erf => same_as_input(node),
            NodeType::ReduceMean => mean_update_outputs(node),
            NodeType::Constant => constant_update_outputs(node),
            NodeType::Equal => equal_update_outputs(node),
            NodeType::Shape => shape_update_outputs(node),
            NodeType::Unsqueeze => unsqueeze_update_outputs(node),
            NodeType::Slice => slice_update_outputs(node),
            NodeType::MatMul => same_as_input(node),
            NodeType::Sigmoid => same_as_input(node),
            NodeType::Transpose => same_as_input(node),
            NodeType::Concat => concat_update_outputs(node),
            NodeType::Reshape => reshape_update_outputs(node),
            NodeType::Dropout => same_as_input(node),
            NodeType::GlobalAveragePool => same_as_input(node),
            NodeType::AveragePool2d => same_as_input(node),
            _ => todo!(
                "shape inference for {:?} is not implemented",
                node.node_type
            ),
        }

        updater.update_tensor_outputs(node);
    }

    updater.update_arguments(graph_outputs);
}

fn constant_update_outputs(node: &mut Node) {
    // Fix the tensor dimension of the output when the value is tensor
    match get_constant_value(node) {
        Some(value) => match &value {
            // The value is stored in an attribute
            AttributeValue::Tensor(tensor) => {
                node.outputs[0].ty = ArgType::Tensor(tensor.clone());
            }
            _ => todo!("Support other constant value types"),
        },
        None => panic!("Constant node must have a value attribute"),
    };
}

/// Infer the shape of the output tensor of a Conv2d node
fn linear_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Linear: multiple inputs are not supported");
    }

    // Extract the configuration of the linear layer (inputs are known)
    let node_input = &mut node.inputs[0];

    if let ArgType::Tensor(tensor) = node_input.clone().ty {
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
    // Extract the shape information from the state
    let shape = match node.states.first() {
        Some(state) => match &state.ty {
            StateType::Tensor(tensor) => match tensor.data.as_ref() {
                Some(TensorData::Int64(data)) => data.clone(),
                _ => panic!("Reshape: invalid state data for shape"),
            },
        },
        None => panic!("Reshape: missing state required for shape"),
    };

    // The output dimension is the same as the shape length
    let dim = shape.len();
    let elem_type = match node.inputs[0].ty.clone() {
        ArgType::Tensor(tensor) => tensor.elem_type,
        _ => panic!("Reshape: invalid input type"),
    };

    node.outputs[0].ty = ArgType::Tensor(Tensor {
        elem_type,
        dim,
        data: None,
        shape: None,
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
        node.outputs[0].ty = ArgType::Tensor(Tensor { dim: 1, ..tensor });
    }
}

fn unsqueeze_update_outputs(node: &mut Node) {
    let node_input = &mut node
        .inputs
        .first()
        .expect("Unsqueeze: an input is required");

    let (dim, elem_type) = match node_input.clone().ty {
        ArgType::Tensor(tensor) => (tensor.dim, tensor.elem_type),
        _ => panic!("Input must be a tensor"),
    };

    node.outputs[0].ty = ArgType::Tensor(Tensor {
        elem_type,
        dim: dim + 1,
        data: None,
        shape: None,
    });
}

fn slice_update_outputs(node: &mut Node) {
    if node.inputs.is_empty() {
        panic!("Slice: inputs required: {:?}", node);
    }

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

fn same_as_input(node: &mut Node) {
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

fn equal_update_outputs(node: &mut Node) {
    let input1_type = node.inputs[0].ty.clone();

    match input1_type {
        ArgType::Tensor(tensor) => {
            // if the input is a tensor, the output is a tensor of bool
            node.outputs[0].ty = ArgType::Tensor(Tensor {
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

    node.outputs[0].ty = ArgType::Tensor(Tensor {
        dim: output_dim,
        ..tensor.clone()
    });
}

/// Infers the shape of a Conv1d node and replaces the shape of the output tensor.
fn conv1d_update_outputs(node: &mut Node) {
    // copy the type from the previous output to the nodeent input
    if node.inputs.len() != 1 {
        panic!("Conv1d: multiple inputs are not supported");
    }

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(tensor) = node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a Conv2d node and replaces the shape of the output tensor.
fn conv2d_update_outputs(node: &mut Node) {
    // copy the type from the previous output to the nodeent input
    if node.inputs.len() != 1 {
        panic!("Conv2d: multiple inputs are not supported");
    }

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(tensor) = node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        panic!("Only tensor input is valid");
    }
}
