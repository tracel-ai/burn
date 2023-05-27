use std::collections::HashMap;

use super::{
    ir::{ArgType, Argument, Node, NodeType, Tensor},
    op_configuration::{conv2d_config, flatten_config, linear_config},
};

use burn::tensor;
use burn_ndarray::NdArrayBackend;

struct TensorShapeUpdater {
    arguments: HashMap<String, Argument>,
}

impl TensorShapeUpdater {
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
            .filter(|output| match &output.arg_type {
                Some(ty) => match ty {
                    ArgType::Tensor(_) => true,
                },
                None => false,
            })
            .map(|arg| {
                self.arguments.insert(arg.name.clone(), arg.clone());
            })
            .count()
    }

    fn update_arguments(&self, arguments: &mut [Argument]) -> usize {
        arguments
            .iter_mut()
            .filter_map(|input| self.arguments.get(&input.name).map(|arg| (arg, input)))
            .filter_map(|(arg, input)| match &arg.arg_type {
                Some(arg) => match arg {
                    ArgType::Tensor(tensor) => Some((tensor, input)),
                },
                None => None,
            })
            .map(|(tensor, input)| {
                input.arg_type = Some(ArgType::Tensor(tensor.clone()));
            })
            .count()
    }
}

/// Infer the shape of each node and replace the shape of the output tensor
pub fn shape_inference(
    nodes: &mut Vec<Node>,
    graph_inputs: &Vec<Argument>,
    graph_outputs: &mut Vec<Argument>,
) {
    let mut updater = TensorShapeUpdater::new(graph_inputs);

    for node in nodes.iter_mut() {
        updater.update_tensor_inputs(node);

        match node.node_type {
            NodeType::Conv2d => conv2d_update_outputs(node),
            NodeType::Linear => linear_update_outputs(node),
            NodeType::Flatten => flatten_update_outputs(node),
            NodeType::Relu => element_wise_update_outputs(node),
            NodeType::LogSoftmax => element_wise_update_outputs(node),
            NodeType::BatchNormalization => element_wise_update_outputs(node),
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
fn linear_update_outputs(curr: &mut Node) {
    if curr.inputs.len() != 1 {
        panic!("Linear: multiple inputs are not supported");
    }

    // Extract the configuration of the linear layer (inputs are known)
    let config = linear_config(curr);

    // Replace the output tensor
    let curr_input = &mut curr.inputs[0];
    let ArgType::Tensor(tensor) = curr_input.clone().arg_type.unwrap();
    let mut new_shape = tensor.shape.clone();
    // Update the last dimension of the shape
    new_shape[tensor.shape.len() - 1] = config.d_input;

    // Update the output tensor
    curr.outputs[0].arg_type = Some(ArgType::Tensor(Tensor {
        name: None,
        shape: new_shape,
        data: None,
        elem_type: tensor.elem_type,
    }));
}

fn element_wise_update_outputs(curr: &mut Node) {
    curr.outputs[0].arg_type = curr.inputs[0].arg_type.clone();
}

/// Infers the shape of a Flatten node and replaces the shape of the output tensor.
fn flatten_update_outputs(curr: &mut Node) {
    if curr.inputs.len() != 1 {
        panic!("Flatten: multiple inputs are not supported");
    }

    let curr_input = &mut curr.inputs[0];

    let ArgType::Tensor(tensor) = curr_input.clone().arg_type.unwrap();

    let input_shape = tensor.shape;

    let (start_dim, end_dim) = flatten_config(curr);

    // calculate the new shape (code is taken from the flatten op)
    // use the same logic as in the flatten op
    // unfortunately the output tensor's dimensions (D2) are not known at compile time
    // that's why we have to calculate the new shape at runtime
    let mut new_dims = vec![0; input_shape.len() - (end_dim - start_dim)];
    let mut flatten_dims = 1;
    for i in input_shape[start_dim..=end_dim].iter() {
        flatten_dims *= i;
    }
    new_dims[..start_dim].copy_from_slice(&input_shape[..start_dim]);
    new_dims[start_dim] = flatten_dims;
    new_dims[start_dim + 1..].copy_from_slice(&input_shape[end_dim + 1..]);

    curr.outputs[0].arg_type = Some(ArgType::Tensor(Tensor {
        name: None,
        shape: new_dims,
        data: None,
        elem_type: tensor.elem_type,
    }));
}

/// Infers the shape of a Conv2d node and replaces the shape of the output tensor.
///
/// The shape of the output tensor is calculated by running the actual convolution operation.
fn conv2d_update_outputs(curr: &mut Node) {
    // copy the type from the previous output to the current input
    if curr.inputs.len() != 1 {
        panic!("Conv2d: multiple inputs are not supported");
    }

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let ArgType::Tensor(tensor) = curr.inputs[0].clone().arg_type.unwrap();

    let elem_type = tensor.elem_type;
    if tensor.shape.len() != 4 {
        panic!("Conv2d: input tensor must be 4D");
    }

    // using the real configuration, run through op and calculate an actual shape of the output tensor
    let config = conv2d_config(curr);
    let conv2d = config.init();

    let mut input_shape: [usize; 4] = [0; 4];
    input_shape.copy_from_slice(tensor.shape.as_slice());
    let input = tensor::Tensor::<NdArrayBackend<f32>, 4>::zeros(input_shape);
    let output = conv2d.forward(input);

    let output_shape = output.shape().dims.to_vec();

    curr.outputs[0].arg_type = Some(ArgType::Tensor(Tensor {
        name: None,
        shape: output_shape,
        data: None,
        elem_type,
    }));
}
