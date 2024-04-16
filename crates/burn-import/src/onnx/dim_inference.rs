use core::panic;

use protobuf::Enum;

use super::{
    from_onnx::OnnxGraphIO,
    ir::{ArgType, AttributeValue, Data, ElementType, Node, NodeType, TensorType},
    op_configuration::flatten_config,
    protos::tensor_proto::DataType,
};

/// Infer the dimension of each output tensor and update them.
pub fn dim_inference(node: &mut Node, graph_io: &mut OnnxGraphIO) {
    match node.node_type {
        NodeType::Add => same_as_input(node),
        NodeType::AveragePool2d => same_as_input(node),
        NodeType::BatchNormalization => same_as_input(node),
        NodeType::Cast => cast_update_outputs(node),
        NodeType::Clip => same_as_input(node),
        NodeType::Concat => concat_update_outputs(node),
        NodeType::Constant => constant_update_outputs(node),
        NodeType::Conv1d => conv1d_update_outputs(node),
        NodeType::Conv2d => conv2d_update_outputs(node),
        NodeType::Cos => same_as_input(node),
        NodeType::Div => same_as_input(node),
        NodeType::Dropout => same_as_input(node),
        NodeType::Equal => equal_update_outputs(node),
        NodeType::Erf => same_as_input(node),
        NodeType::Exp => same_as_input(node),
        NodeType::Flatten => flatten_update_outputs(node),
        NodeType::Gelu => same_as_input(node),
        NodeType::GatherElements => same_as_input(node),
        NodeType::GlobalAveragePool => same_as_input(node),
        NodeType::ConvTranspose2d => conv_transpose2d_update_outputs(node),
        NodeType::Linear => linear_update_outputs(node),
        NodeType::Log => same_as_input(node),
        NodeType::LogSoftmax => same_as_input(node),
        NodeType::MaxPool2d => same_as_input(node),
        NodeType::Mul => same_as_input(node),
        NodeType::Neg => same_as_input(node),
        NodeType::Not => same_as_input(node),
        NodeType::Reciprocal => same_as_input(node),
        NodeType::ReduceMax => reduce_max_update_outputs(node),
        NodeType::ReduceMean => reduce_mean_update_outputs(node),
        NodeType::Relu => same_as_input(node),
        NodeType::Reshape => reshape_update_outputs(node),
        NodeType::Shape => shape_update_outputs(node),
        NodeType::Sigmoid => same_as_input(node),
        NodeType::Sin => same_as_input(node),
        NodeType::Softmax => same_as_input(node),
        NodeType::Sqrt => same_as_input(node),
        NodeType::Sub => same_as_input(node),
        NodeType::Tanh => same_as_input(node),
        NodeType::Transpose => same_as_input(node),
        NodeType::Unsqueeze => unsqueeze_update_output(node),
        NodeType::Pow => same_as_input(node),
        NodeType::LeakyRelu => same_as_input(node),
        // Intentionally letting outputs leave unchanged but issue a warning so IR file can be generated.
        _ => temporary_pass_through_stub(node),
    }

    graph_io.update_tensor_output(node);
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
    let input = &mut node.inputs[0];
    let output = &mut node.outputs[0];

    // Extract cast type and update the output tensor
    let elem_type = match node.attrs.get("to") {
        Some(value) => match &value {
            AttributeValue::Int64(type_id) => match DataType::from_i32(*type_id as i32).unwrap() {
                DataType::FLOAT => ElementType::Float32,
                DataType::INT32 => ElementType::Int32,
                DataType::INT64 => ElementType::Int64,
                DataType::DOUBLE => ElementType::Float64,
                DataType::BOOL => ElementType::Bool,
                _ => panic!("Cast: unsupported type"),
            },
            _ => panic!("'to' attribute must be an Int64"),
        },
        None => panic!("Constant node must have a value attribute"),
    };

    match input.ty.clone() {
        ArgType::Tensor(tensor) => {
            if tensor.dim == 0 {
                // treat 0-dim tensor as scalar
                output.ty = ArgType::Scalar(elem_type);
                input.ty = ArgType::Scalar(tensor.elem_type);
            } else {
                // Cast input and output are the same shape, but possibly different types
                output.ty = ArgType::Tensor(TensorType {
                    elem_type,
                    dim: tensor.dim,
                    shape: tensor.shape.clone(),
                });
            }
        }
        ArgType::Scalar(_scalar) => {
            output.ty = ArgType::Scalar(elem_type);
        }
        _ => panic!("Cast: only scalar and tensor inputs are valid"),
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

fn reduce_mean_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Mean: multiple inputs are not supported");
    }

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
        // NOTE: ReduceMean w/o keepdims reduces to a scalar value, but Burn doesn't have
        // 0-dim tensor so we can't track or perform other ops on that value if we call
        // `.into_scalar()` on the result of `tensor.max()`
        // node.outputs[0].ty = ArgType::Scalar(tensor.elem_type);
        // Instead, we return a tensor of rank 1 (the result of `tensor.max()`)
        node.outputs[0].ty = ArgType::Tensor(TensorType { dim: 1, ..tensor });
    }
}

//fn __unsqueeze_shape
/// Infers the shape of the output from the input and axes
/// Right now, this should only be called if the rhs is a constant
fn unsqueeze_update_output(node: &mut Node) {
    if node.inputs.len() != 2 {
        panic!("Unsqueeze: wrong number of inputs");
    }
    // get the values while making sure the types are correct
    let (input, axes) = match (&node.inputs[0].ty, &node.inputs[1].ty) {
        (ArgType::Tensor(tensor), ArgType::Tensor(_axes)) => (
            tensor.clone(),
            match &node.inputs[1].value {
                Some(value) => match &value {
                    Data::Int64s(axes) => Some(axes.clone()),
                    _ => panic!("Unsqueeze: invalid input types"),
                },
                None => None,
            },
        ),
        _ => panic!("Unsqueeze: invalid input types"),
    };
    //need output way up here to avoid borrowing issues
    let mut tensor = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Unsqueeze: invalid output types"),
    };
    match &axes {
        //case 1: axes is constant -> output shape is input shape with 1s inserted at the axes
        Some(dim_indices) => {
            let output_rank = (dim_indices.len() + input.dim) as i64;
            let mut dim_indices = dim_indices
                .to_vec()
                .iter()
                .map(|&d| {
                    if (-output_rank..output_rank).contains(&d) {
                        (if d < 0 { d + output_rank } else { d }) as usize
                    } else {
                        panic!("Unsqueeze: invalid axis")
                    }
                })
                .collect::<Vec<usize>>();
            dim_indices.sort_unstable();
            let mut new_dims = vec![1; output_rank as usize];

            tensor.dim = output_rank as usize;
            let old_dims = input.shape.unwrap();
            //Now use this to copy the chunks of the dims
            let mut prev_idx: usize = 0;
            let mut current_left_b: usize = 0;
            let mut current_right_b: usize = 0;
            let mut offset: usize = 0;

            dim_indices.iter().for_each(|d| {
                //check if there is space for at least one dimension
                if prev_idx < *d {
                    current_right_b = *d - offset;

                    //copy the chunks of the dims
                    if current_right_b < old_dims.len() {
                        new_dims[prev_idx..*d]
                            .copy_from_slice(&old_dims[current_left_b..current_right_b])
                    } else {
                        new_dims[prev_idx..*d].copy_from_slice(&old_dims[current_left_b..]);
                    }
                    prev_idx = *d + 1;
                    //offset is equal to the number of extracted elements from the original shape
                    offset += current_right_b - current_left_b;
                    current_left_b = current_right_b;
                } else {
                    //it's sorted so the only reason this would happen
                    //is if multiple indices are the same
                    prev_idx += 1;
                }
            });
            //copy over anything past the index of the last new dimension
            if current_left_b < old_dims.len() {
                new_dims[prev_idx..].copy_from_slice(&old_dims[current_left_b..]);
            }
            tensor.shape = Some(new_dims);
            node.outputs[0].ty = ArgType::Tensor(tensor.clone());
        }

        //case 3: output shape is dynamic -> black magic or unsupported
        None => {
            panic!("Unsqueeze: dynamic output shape is not currently supported");
        }
    }
}

fn same_as_input(node: &mut Node) {
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

/// Temporary pass-through stub for dimension inference so that we can export the IR model.
fn temporary_pass_through_stub(node: &Node) {
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
        panic!("Shape: multiple inputs are not supported: {:?}", node);
    }

    let node_input = &mut node.inputs[0];
    if let ArgType::Tensor(_tensor) = node_input.clone().ty {
        // Output tensor is 1D int64
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            dim: 1,
            ..Default::default()
        });
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

/// Infers the shape of a ConvTranspose2d node and replaces the shape of the output tensor.
fn conv_transpose2d_update_outputs(node: &mut Node) {
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(tensor) = node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a ReduceMax node and replaces the shape of the output tensor.
fn reduce_max_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ReduceMax: multiple inputs are not supported");
    }

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
        // NOTE: ReduceMax w/o keepdims reduces to a scalar value, but Burn doesn't have
        // 0-dim tensor so we can't track or perform other ops on that value if we call
        // `.into_scalar()` on the result of `tensor.max()`
        // node.outputs[0].ty = ArgType::Scalar(tensor.elem_type);
        // Instead, we return a tensor of rank 1 (the result of `tensor.max()`)
        node.outputs[0].ty = ArgType::Tensor(TensorType { dim: 1, ..tensor });
    }
}
