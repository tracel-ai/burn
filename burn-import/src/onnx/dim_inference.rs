use std::collections::HashMap;

use super::{
    ir::{ArgType, Argument, AttributeValue, Node, NodeType, TensorArg},
    op_configuration::flatten_config,
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
            NodeType::Conv2d => conv2d_update_outputs(node),
            NodeType::MaxPool2d => max_pool2d_update_outputs(node),
            NodeType::Linear => linear_update_outputs(node),
            NodeType::Flatten => flatten_update_outputs(node),
            NodeType::Relu => same_as_input(node),
            NodeType::LogSoftmax => same_as_input(node),
            NodeType::BatchNormalization => same_as_input(node),
            NodeType::Add => same_as_input(node),
            NodeType::Sub => same_as_input(node),
            NodeType::Pow => same_as_input(node),
            NodeType::Mul => same_as_input(node),
            NodeType::Cast => same_as_input(node),
            NodeType::Div => same_as_input(node),
            NodeType::Sqrt => same_as_input(node),
            NodeType::Softmax => same_as_input(node),
            NodeType::Erf => same_as_input(node),
            NodeType::ReduceMean => mean_update_outputs(node),
            NodeType::Constant => {
                node.outputs[0].ty = ArgType::Constant;
            }
            NodeType::Equal => same_as_input(node),
            NodeType::Shape => shape_update_outputs(node),
            NodeType::Unsqueeze => unsqueeze_update_outputs(node),
            NodeType::Slice => slice_update_outputs(node),
            NodeType::MatMul => same_as_input(node),
            NodeType::Sigmoid => same_as_input(node),
            NodeType::Transpose => same_as_input(node),
            NodeType::Concat => concat_update_outputs(node),
            NodeType::Reshape => reshape_update_outputs(node),
            NodeType::Dropout => same_as_input(node),
            NodeType::GlobalAveragePool => same_as_input(node), //FIXME use correct output
            _ => todo!(
                "shape inference for {:?} is not implemented",
                node.node_type
            ),
        }

        updater.update_tensor_outputs(node);
    }

    updater.update_arguments(graph_outputs);
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
        node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
    } else {
        panic!("Only tensor input is valid");
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

    node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
}

fn reshape_update_outputs(node: &mut Node) {
    let dim = *node
        .inputs
        .iter()
        .filter_map(|input| match &input.ty {
            ArgType::Tensor(tensor) => Some(tensor.dim),
            _ => None,
        })
        .collect::<Vec<_>>()
        .last()
        .unwrap();

    node.outputs[0].ty = ArgType::Tensor(TensorArg { dim });
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
        node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: 1 });
    }
}

fn unsqueeze_update_outputs(node: &mut Node) {
    if node.inputs.is_empty() {
        panic!("Unsqueeze: inputs required: {:?}", node);
    }

    let node_input = &mut node.inputs[0];
    let dim = match node_input.clone().ty {
        ArgType::Tensor(tensor) => tensor.dim,
        ArgType::Shape(dim) => dim,
        ArgType::Constant => panic!("Needs shape or tensor"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: dim + 1 });
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

    node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
}

fn same_as_input(node: &mut Node) {
    node.outputs[0].ty = node.inputs[0].ty.clone();
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

    let (start_dim, end_dim) = flatten_config(node);

    node.outputs[0].ty = ArgType::Tensor(TensorArg {
        dim: end_dim - start_dim,
    });
}

/// Infers the shape of a Conv2d node and replaces the shape of the output tensor.
///
/// The shape of the output tensor is calculated by running the actual convolution operation.
fn conv2d_update_outputs(node: &mut Node) {
    // copy the type from the previous output to the nodeent input
    if node.inputs.len() != 1 {
        panic!("Conv2d: multiple inputs are not supported");
    }

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(tensor) = node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a MaxPool2d node and replaces the shape of the output tensor.
///
/// The shape of the output tensor is calculated by running the actual convolution operation.
fn max_pool2d_update_outputs(node: &mut Node) {
    // copy the type from the previous output to the node input
    if node.inputs.len() != 1 {
        panic!("Pool2d: multiple inputs are not supported");
    }

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(tensor) = node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
    } else {
        panic!("Only tensor input is valid");
    }
}
