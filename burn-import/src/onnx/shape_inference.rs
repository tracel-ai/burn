use std::collections::HashMap;

use burn::tensor;

use burn_ndarray::NdArrayBackend;

use super::{
    ir::{ArgType, Argument, Node, NodeType, Tensor},
    op_configuration::{conv2d_config, flatten_config, linear_config},
};

/// Infer the shape of each node and replace the shape of the output tensor
pub fn shape_inference(
    nodes: &mut Vec<Node>,
    graph_inputs: &Vec<Argument>,
    graph_outputs: &mut Vec<Argument>,
) {
    let mut prev_outputs: HashMap<String, Argument> = HashMap::new();

    for output in graph_inputs.iter() {
        prev_outputs.insert(output.name.clone(), output.clone());
    }

    for node in nodes.iter_mut() {
        match node.node_type {
            NodeType::Conv2d => conv2d(node, &prev_outputs),
            NodeType::Linear => linear(node, &prev_outputs),
            NodeType::Relu => relu(node, &prev_outputs),
            NodeType::Flatten => flatten(node, &prev_outputs),
            NodeType::LogSoftmax => log_softmax(node, &prev_outputs),
            _ => todo!(
                "shape inference for {:?} is not implemented",
                node.node_type
            ),
        }

        for output in node.outputs.iter() {
            prev_outputs.insert(output.name.clone(), output.clone());
        }
    }

    //update the outputs of the graph from prev_outputs
    for output in graph_outputs.iter_mut() {
        let arg = prev_outputs.get(output.name.as_str()).unwrap();
        output.arg_type = arg.arg_type.clone();
    }
}

/// Infer the shape of the output tensor of a Conv2d node
fn linear(curr: &mut Node, prev_outpus: &HashMap<String, Argument>) {
    if curr.inputs.len() != 1 {
        panic!("Linear: multiple inputs are not supported");
    }

    // Fill in the missing information about the input tensor from the previous outputs
    let prev_node_output = prev_outpus.get(curr.inputs[0].name.as_str()).unwrap();
    curr.inputs[0].arg_type = prev_node_output.arg_type.clone();

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

/// Infers the shape of a Relu node and replaces the shape of the output tensor.
fn relu(curr: &mut Node, prev_outpus: &HashMap<String, Argument>) {
    if curr.inputs.len() != 1 {
        panic!("Relu: multiple inputs are not supported");
    }

    // Fill in the missing information about the input tensor from the previous outputs
    let prev_node_output = prev_outpus.get(curr.inputs[0].name.as_str()).unwrap();

    curr.inputs[0].arg_type = prev_node_output.arg_type.clone();

    curr.outputs[0].arg_type = prev_node_output.arg_type.clone();
}

/// Infers the shape of a Flatten node and replaces the shape of the output tensor.
fn flatten(curr: &mut Node, prev_outpus: &HashMap<String, Argument>) {
    if curr.inputs.len() != 1 {
        panic!("Flatten: multiple inputs are not supported");
    }

    // Fill in the missing information about the input tensor from the previous outputs
    let prev_node_output = prev_outpus.get(curr.inputs[0].name.as_str()).unwrap();

    curr.inputs[0].arg_type = prev_node_output.arg_type.clone();

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

/// Infers the shape of a LogSoftmax node and replaces the shape of the output tensor.
fn log_softmax(curr: &mut Node, prev_outpus: &HashMap<String, Argument>) {
    if curr.inputs.len() != 1 {
        panic!("LogSoftmax: multiple inputs are not supported");
    }

    // Fill in the missing information about the input tensor from the previous outputs
    let prev_node_output = prev_outpus.get(curr.inputs[0].name.as_str()).unwrap();
    curr.inputs[0].arg_type = prev_node_output.arg_type.clone();
    curr.outputs[0].arg_type = prev_node_output.arg_type.clone();
}

/// Infers the shape of a Conv2d node and replaces the shape of the output tensor.
///
/// The shape of the output tensor is calculated by running the actual convolution operation.
fn conv2d(curr: &mut Node, prev_outpus: &HashMap<String, Argument>) {
    // copy the type from the previous output to the current input

    if curr.inputs.len() != 1 {
        panic!("Conv2d: multiple inputs are not supported");
    }

    // Fill in the missing information about the input tensor from the previous outputs
    let curr_input = &mut curr.inputs[0];
    let prev = prev_outpus.get(curr_input.name.as_str()).unwrap();
    curr_input.arg_type = prev.arg_type.clone();

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let ArgType::Tensor(tensor) = curr_input.clone().arg_type.unwrap();

    let elem_type = tensor.elem_type;

    if tensor.shape.len() != 4 {
        panic!("Conv2d: input tensor must be 4D");
    }

    let mut input_shape: [usize; 4] = [0; 4];
    input_shape.copy_from_slice(tensor.shape.as_slice());

    // using the real configuration, run through op and calculate an actual shape of the output tensor
    let config = conv2d_config(curr);

    let conv2d = config.init();

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
