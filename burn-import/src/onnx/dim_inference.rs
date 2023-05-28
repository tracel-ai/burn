use std::collections::HashMap;

use super::{
    ir::{ArgType, Argument, Node, NodeType, TensorArg},
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
            .filter(|output| match &output.ty {
                ArgType::Tensor(_) => true,
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
            .map(|(arg, input)| match &arg.ty {
                ArgType::Tensor(tensor) => (tensor, input),
            })
            .map(|(tensor, input)| {
                input.ty = ArgType::Tensor(tensor.clone());
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
    let curr_input = &mut curr.inputs[0];
    let ArgType::Tensor(tensor) = curr_input.clone().ty;

    // Update the output tensor
    curr.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
}

fn element_wise_update_outputs(curr: &mut Node) {
    curr.outputs[0].ty = curr.inputs[0].ty.clone();
}

/// Infers the shape of a Flatten node and replaces the shape of the output tensor.
fn flatten_update_outputs(curr: &mut Node) {
    if curr.inputs.len() != 1 {
        panic!("Flatten: multiple inputs are not supported");
    }

    let (start_dim, end_dim) = flatten_config(curr);

    curr.outputs[0].ty = ArgType::Tensor(TensorArg {
        dim: end_dim - start_dim,
    });
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
    let ArgType::Tensor(tensor) = curr.inputs[0].clone().ty;

    curr.outputs[0].ty = ArgType::Tensor(TensorArg { dim: tensor.dim });
}
