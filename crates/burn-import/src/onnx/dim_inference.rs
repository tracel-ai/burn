use core::cmp::max;
use core::panic;

use protobuf::Enum;

use super::{
    from_onnx::OnnxGraphIO,
    ir::{
        ArgType, AttributeValue, ConversionNode, Data, ElementType, Node as ProcessedNode,
        NodeType, TensorType,
    },
    node_remap::remap_unsqueeze_to_reshape,
    op_configuration::flatten_config,
    protos::tensor_proto::DataType,
};

/// Infer the dimension of each output tensor and update them.
pub fn dim_inference(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    match node.node_type {
        NodeType::Add => same_as_input(node),
        NodeType::AveragePool1d => same_as_input(node),
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
        NodeType::LayerNormalization => same_as_input(node),
        NodeType::Linear => linear_update_outputs(node),
        NodeType::Log => same_as_input(node),
        NodeType::LogSoftmax => same_as_input(node),
        NodeType::MatMul => matmul_update_outputs(node),
        NodeType::Min => same_as_input(node),
        NodeType::Max => same_as_input(node),
        NodeType::MaxPool1d => same_as_input(node),
        NodeType::MaxPool2d => same_as_input(node),
        NodeType::Mul => same_as_input(node),
        NodeType::Neg => same_as_input(node),
        NodeType::Not => same_as_input(node),
        NodeType::Greater => greater_update_outputs(node),
        NodeType::GreaterOrEqual => greater_or_equal_update_outputs(node),
        NodeType::Less => less_update_outputs(node),
        NodeType::LessOrEqual => less_or_equal_update_outputs(node),
        NodeType::Reciprocal => same_as_input(node),
        NodeType::ReduceMax => reduce_max_update_outputs(node),
        NodeType::ReduceMean => reduce_mean_update_outputs(node),
        NodeType::ReduceSum => reduce_sum_update_outputs(node),
        NodeType::Relu => same_as_input(node),
        NodeType::Reshape => reshape_update_outputs(node),
        NodeType::Shape => shape_update_outputs(node),
        NodeType::Sigmoid => same_as_input(node),
        NodeType::Sign => same_as_input(node),
        NodeType::Sin => same_as_input(node),
        NodeType::Softmax => same_as_input(node),
        NodeType::Sqrt => same_as_input(node),
        NodeType::Sub => same_as_input(node),
        NodeType::Tanh => same_as_input(node),
        NodeType::Transpose => same_as_input(node),
        NodeType::Unsqueeze => unsqueeze_update_output(node),
        NodeType::Pow => same_as_input(node),
        NodeType::LeakyRelu => same_as_input(node),
        NodeType::PRelu => same_as_input(node),
        NodeType::Where => where_update_outputs(node),
        NodeType::Squeeze => squeeze_update_output(node),
        NodeType::RandomUniform => random_update_output(node),
        NodeType::RandomNormal => random_update_output(node),
        // Intentionally letting outputs leave unchanged but issue a warning so IR file can be generated.
        _ => temporary_pass_through_stub(node, graph_io),
    }

    // graph_io.update_tensor_output(node);
}

fn constant_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
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

    graph_io.set_type(
        &node.outputs[0],
        match matched_value {
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
        },
    );
}


/// Infer the shape of a node's output with an explicit shape attribute
/// for the Random operations with explicit shape
///
/// This includes the `RandomUniform`, `RandomNormal` operators
///
/// Also reads & interprets an optional `dtype` attribute
fn random_update_output(node: &mut Node) {
    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
        .unwrap_or(DataType::FLOAT);

    let mut shape = node
        .attrs
        .get("shape")
        .expect("required shape attribute missing")
        .clone()
        .into_i64s();

    let elem_type = match dtype {
        DataType::FLOAT => ElementType::Float32,
        DataType::DOUBLE => ElementType::Float64,
        _ => panic!("tensor with type {dtype:?} not supported for random output"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type,
        dim: shape.len(),
        shape: Some(
            shape
                .drain(..)
                .map(usize::try_from)
                .collect::<Result<Vec<usize>, _>>()
                .unwrap(),
        ),
    })
}

/// Infer the shape of a node's output with an explicit shape attribute
/// for the Random operations with explicit shape
///
/// This includes the `RandomUniform`, `RandomNormal` operators
///
/// Also reads & interprets an optional `dtype` attribute
fn random_update_output(node: &mut Node) {
    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
        .unwrap_or(DataType::FLOAT);

    let mut shape = node
        .attrs
        .get("shape")
        .expect("required shape attribute missing")
        .clone()
        .into_i64s();

    let elem_type = match dtype {
        DataType::FLOAT => ElementType::Float32,
        DataType::DOUBLE => ElementType::Float64,
        _ => panic!("tensor with type {dtype:?} not supported for random output"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type,
        dim: shape.len(),
        shape: Some(
            shape
                .drain(..)
                .map(usize::try_from)
                .collect::<Result<Vec<usize>, _>>()
                .unwrap(),
        ),
    })
}

/// Infer the shape of the output tensor of a Conv2d node
fn linear_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    // Extract the configuration of the linear layer (inputs are known)
    let node_input = &node.inputs[0];
    let weight_key = &node.inputs[1];

    // Calculate the output shape. Usually we do not use shapes, but since the input shape is
    // known, we can calculate the output shape.
    if let ArgType::Tensor(tensor) = graph_io.get_type(node_input) {
        let mut tensor = tensor.clone();

        // Update the shape of the output tensor if it's known
        if let Some(mut shape) = tensor.shape.clone() {
            if let ArgType::Tensor(weight_tensor) = graph_io.get_type(weight_key) {
                let last = shape.last_mut().unwrap();
                *last = *weight_tensor.shape.as_ref().unwrap().first().unwrap();
            } else {
                panic!("Weight must be a tensor");
            }
            tensor.shape = Some(shape);
        }

        // Update the output tensor
        graph_io.set_type(&node.outputs[0], ArgType::Tensor(tensor));
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update the output type using "to" attribute
fn cast_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
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

    match graph_io.get_type(input).clone() {
        ArgType::Tensor(tensor) => {
            if tensor.dim == 0 {
                // treat 0-dim tensor as scalar
                graph_io.set_type(output, ArgType::Scalar(elem_type));
                graph_io.set_type(input, ArgType::Scalar(tensor.elem_type));
            } else {
                // Cast input and output are the same shape, but possibly different types
                graph_io.set_type(
                    output,
                    ArgType::Tensor(TensorType {
                        elem_type,
                        dim: tensor.dim,
                        shape: tensor.shape.clone(),
                    }),
                );
            }
        }
        ArgType::Scalar(_scalar) => {
            graph_io.set_type(output, ArgType::Scalar(elem_type));
        }
        _ => panic!("Cast: only scalar and tensor inputs are valid"),
    }

    log::debug!(
        "Cast: input type: {:?}, output type: {:?}",
        graph_io.get_type(input),
        graph_io.get_type(output)
    );
}

fn concat_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match graph_io.get_type(input) {
            ArgType::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .unwrap();

    graph_io.set_type(&node.outputs[0], ArgType::Tensor(tensor.clone()));
}

fn reshape_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    let shape = if node.inputs.len() == 2 {
        match graph_io.get_value(&node.inputs[1]) {
            Some(value) => match value {
                Data::Int64s(shape) => Some(shape.clone()),
                _ => panic!("Reshape: invalid input types"),
            },
            None => None,
        }
    } else {
        node.attrs.get("shape").cloned().map(|v| v.into_i64s())
    };

    let output = match graph_io.get_type(&node.outputs[0]) {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Reshape: invalid output types"),
    };

    if let Some(shape) = shape {
        graph_io.set_type(
            &node.outputs[0],
            ArgType::Tensor(TensorType {
                dim: shape.len(),
                shape: None, // shape is calculated at runtime
                ..output
            }),
        );
    }
}

fn greater_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn less_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn greater_or_equal_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn less_or_equal_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn greater_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn less_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn greater_or_equal_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn less_or_equal_update_outputs(node: &mut Node) {
    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}


fn reduce_mean_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    if node.inputs.len() != 1 {
        panic!("Mean: multiple inputs are not supported");
    }

    let node_input = &mut node.inputs[0];
    let tensor = match graph_io.get_type(node_input) {
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
        graph_io.set_type(&node.outputs[0], ArgType::Tensor(tensor.clone()));
    } else {
        // NOTE: ReduceMean w/o keepdims reduces to a scalar value, but Burn doesn't have
        // 0-dim tensor so we can't track or perform other ops on that value if we call
        // `.into_scalar()` on the result of `tensor.max()`
        // node.outputs[0].ty = ArgType::Scalar(tensor.elem_type);
        // Instead, we return a tensor of rank 1 (the result of `tensor.max()`)
        graph_io.set_type(
            &node.outputs[0],
            ArgType::Tensor(TensorType {
                dim: 1,
                ..tensor.clone()
            }),
        );
    }
}

/// Update the output tensor dimension
fn squeeze_update_output(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    let axes = if node.inputs.len() == 2 {
        match graph_io.get_value(&node.inputs[1]) {
            Some(value) => match value {
                Data::Int64s(axes) => Some(axes.clone()),
                _ => panic!("Squeeze: invalid input types"),
            },
            None => None,
        }
    } else {
        node.attrs.get("axes").cloned().map(|v| v.into_i64s())
    };

    if axes.is_none() {
        panic!("Squeeze must specify an axis");
    } else if axes.as_ref().unwrap().len() > 1 {
        panic!(
            "Squeeze must specify only 1 axis, found {:?}",
            axes.as_ref().unwrap().len()
        );
    }

    let input_dim = match graph_io.get_type(&node.inputs[0]) {
        ArgType::Tensor(tensor) => tensor.dim,
        _ => panic!("Squeeze: invalid input type"),
    };

    let output_elem = match graph_io.get_type(&node.outputs[0]) {
        ArgType::Tensor(tensor) => tensor.elem_type.clone(),
        _ => panic!("Squeeze: invalid output type"),
    };

    graph_io.set_type(
        &node.outputs[0],
        ArgType::Tensor(TensorType {
            dim: input_dim - 1,
            shape: None, // shape is tracked and calculated at runtime
            elem_type: output_elem,
        }),
    );
}

/// Update the output tensor dimension based on the "axes" attribute or the second input
fn unsqueeze_update_output(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    let axes = if node.inputs.len() == 2 {
        match graph_io.get_value(&node.inputs[1]) {
            Some(value) => match value {
                Data::Int64s(axes) => Some(axes.clone()),
                _ => panic!("Unsqueeze: invalid input types"),
            },
            None => None,
        }
    } else {
        node.attrs.get("axes").cloned().map(|v| v.into_i64s())
    };

    if axes.is_none() {
        // Remap unsqueeze to reshape if axes doesn't have a value
        // since we don't support runtime shapes
        remap_unsqueeze_to_reshape(node, graph_io);
        reshape_update_outputs(node, graph_io);
    }

    let input_dim = match graph_io.get_type(&node.inputs[0]) {
        ArgType::Tensor(tensor) => tensor.dim,
        ArgType::Scalar(_) => 0, // treat scalar as 0-dim tensor
        _ => panic!("Unsqueeze: invalid input type"),
    };

    let output_elem = match graph_io.get_type(&node.outputs[0]) {
        ArgType::Tensor(tensor) => tensor.elem_type.clone(),
        ArgType::Scalar(elem_type) => elem_type.clone(),
        _ => panic!("Unsqueeze: invalid output type"),
    };

    if let Some(axes) = axes {
        graph_io.set_type(
            &node.outputs[0],
            ArgType::Tensor(TensorType {
                dim: input_dim + axes.len(),
                shape: None, // shape is tracked and calculated at runtime
                elem_type: output_elem,
            }),
        );
    }
}

fn same_as_input(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    graph_io.copy_type(&node.inputs[0], &node.outputs[0]);
}

/// Temporary pass-through stub for dimension inference so that we can export the IR model.
fn temporary_pass_through_stub(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    log::warn!("Must implement dimension inference for {:?}", node);
    log::warn!("Temporarily setting the output type to the input type.");
    same_as_input(node, graph_io);
}

fn equal_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    let input1_type = graph_io.get_type(&node.inputs[0]);

    match input1_type {
        ArgType::Tensor(tensor) => {
            // if the input is a tensor, the output is a tensor of bool
            graph_io.set_type(
                &node.outputs[0],
                ArgType::Tensor(TensorType {
                    elem_type: ElementType::Bool,
                    ..tensor.clone()
                }),
            );
            // node.outputs[0].ty = ArgType::Tensor(TensorType {
            //     elem_type: ElementType::Bool,
            //     ..tensor.clone()
            // });
        }
        ArgType::Scalar(_) => {
            graph_io.set_type(&node.outputs[0], ArgType::Scalar(ElementType::Bool));
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn shape_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    if node.inputs.len() != 1 {
        panic!("Shape: multiple inputs are not supported: {:?}", node);
    }

    if let ArgType::Tensor(_tensor) = graph_io.get_type(&node.inputs[0]) {
        // Output tensor is 1D int64
        graph_io.set_type(
            &node.outputs[0],
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                dim: 1,
                ..Default::default()
            }),
        );
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a Flatten node and replaces the shape of the output tensor.
fn flatten_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    if node.inputs.len() != 1 {
        panic!("Flatten: multiple inputs are not supported");
    }
    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match graph_io.get_type(input) {
            ArgType::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .unwrap();

    let input_dim = tensor.dim;

    //? is there a way to separate out flatten_config from dim_inference? may want to consider inlining
    //to avoid the need for the conversion
    let (start_dim, end_dim) = flatten_config(&ProcessedNode::new(node.clone(), graph_io));

    let collapsed_dims = end_dim - start_dim;
    let output_dim = input_dim - collapsed_dims;

    graph_io.set_type(
        &node.outputs[0],
        ArgType::Tensor(TensorType {
            dim: output_dim,
            ..tensor.clone()
        }),
    );
}

/// Infers the shape of a Conv1d node and replaces the shape of the output tensor.
fn conv1d_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(_) = graph_io.get_type(&node.inputs[0]) {
        graph_io.copy_type(&node.inputs[0], &node.outputs[0]);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a Conv2d node and replaces the shape of the output tensor.
fn conv2d_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(_) = graph_io.get_type(&node.inputs[0]) {
        graph_io.copy_type(&node.inputs[0], &node.outputs[0]);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infers the shape of a ConvTranspose2d node and replaces the shape of the output tensor.
fn conv_transpose2d_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    if let ArgType::Tensor(_) = graph_io.get_type(&node.inputs[0]) {
        graph_io.copy_type(&node.inputs[0], &node.outputs[0]);
    } else {
        panic!("Only tensor input is valid");
    }
}

fn matmul_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    // NOTE: matmul only supported for float tensors
    match (
        graph_io.get_type(&node.inputs[0]),
        graph_io.get_type(&node.inputs[1]),
    ) {
        (ArgType::Tensor(a), ArgType::Tensor(b)) => {
            // With broadcasting support, output dim has to be computed based on the inputs
            let mut out_dim = max(a.dim, b.dim);

            // Matrix-vector or vector-matrix product
            if (a.dim >= 2 && b.dim == 1) || (a.dim == 1 && b.dim >= 2) {
                out_dim -= 1;
            }

            graph_io.set_type(
                &node.outputs[0],
                ArgType::Tensor(TensorType {
                    elem_type: a.elem_type.clone(),
                    dim: out_dim,
                    shape: a.shape.clone(),
                }),
            );
        }
        _ => panic!("Only tensor input is valid"),
    }
}

/// Infers the shape of a ReduceMax node and replaces the shape of the output tensor.
fn reduce_max_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    if node.inputs.len() != 1 {
        panic!("ReduceMax: multiple inputs are not supported");
    }

    let node_input = &node.inputs[0];
    let tensor = match graph_io.get_type(node_input) {
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

    graph_io.set_type(
        &node.outputs[0],
        if dim_only {
            ArgType::Tensor(tensor.clone())
        } else {
            // NOTE: ReduceMax w/o keepdims reduces to a scalar value, but Burn doesn't have
            // 0-dim tensor so we can't track or perform other ops on that value if we call
            // `.into_scalar()` on the result of `tensor.max()`
            // node.outputs[0].ty = ArgType::Scalar(tensor.elem_type);
            // Instead, we return a tensor of rank 1 (the result of `tensor.max()`)
            ArgType::Tensor(TensorType {
                dim: 1,
                ..tensor.clone()
            })
        },
    )
}

/// Infers the shape of a ReduceSum node and replaces the shape of the output tensor.
fn reduce_sum_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    let node_input = &node.inputs[0];
    let tensor = match graph_io.get_type(node_input) {
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

    let dim_only = match node.inputs.get(1).and_then(|arg| graph_io.get_value(arg)) {
        Some(value) => match &value {
            Data::Int64(_) => true,
            Data::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => dim_only,
    };

    graph_io.set_type(
        &node.outputs[0],
        if dim_only {
            ArgType::Tensor(tensor.clone())
        } else {
            // NOTE: ReduceSum w/o keepdims reduces to a scalar value, but Burn doesn't have
            // 0-dim tensor so we can't track or perform other ops on that value if we call
            // `.into_scalar()` on the result of `tensor.sum()`
            // node.outputs[0].ty = ArgType::Scalar(tensor.elem_type);
            // Instead, we return a tensor of rank 1 (the result of `tensor.sum()`)
            ArgType::Tensor(TensorType {
                dim: 1,
                ..tensor.clone()
            })
        },
    )
}

fn where_update_outputs(node: &mut ConversionNode, graph_io: &mut OnnxGraphIO) {
    match (
        graph_io.get_type(&node.inputs[0]),
        graph_io.get_type(&node.inputs[1]),
        graph_io.get_type(&node.inputs[2]),
    ) {
        (ArgType::Tensor(condition), ArgType::Tensor(x), ArgType::Tensor(y)) => {
            // With broadcasting support, output dim has to be computed based on the inputs
            graph_io.set_type(
                &node.outputs[0],
                ArgType::Tensor(TensorType {
                    elem_type: x.elem_type.clone(),
                    dim: max(condition.dim, max(x.dim, y.dim)),
                    ..Default::default()
                }),
            );
        }
        _ => panic!("Only tensor input is valid"),
    }
}
