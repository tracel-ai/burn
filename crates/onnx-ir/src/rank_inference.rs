use core::cmp::max;
use core::panic;

use protobuf::Enum;

use crate::{
    ir::{ArgType, AttributeValue, Data, ElementType, Node, NodeType, TensorData, TensorType},
    protos::tensor_proto::DataType,
    util::{flatten_config, shape_config},
};

/// Infer the rank of each output tensor and update them.
pub fn rank_inference(node: &mut Node) {
    match node.node_type {
        NodeType::Add => same_as_input_broadcast(node),
        NodeType::ArgMax => argmax_update_outputs(node),
        NodeType::AveragePool1d => same_as_input(node),
        NodeType::AveragePool2d => same_as_input(node),
        NodeType::BatchNormalization => same_as_input(node),
        NodeType::Cast => cast_update_outputs(node),
        NodeType::Clip => same_as_input(node),
        NodeType::Concat => concat_update_outputs(node),
        NodeType::Constant => constant_update_outputs(node),
        NodeType::ConstantOfShape => constant_of_shape_update_output(node),
        NodeType::Conv1d => conv1d_update_outputs(node),
        NodeType::Conv2d => conv2d_update_outputs(node),
        NodeType::Cos => same_as_input(node),
        NodeType::Div => same_as_input_broadcast(node),
        NodeType::Dropout => same_as_input(node),
        NodeType::Equal => elementwise_comparison_outputs(node),
        NodeType::Erf => same_as_input(node),
        NodeType::Exp => same_as_input(node),
        NodeType::Expand => expand_update_outputs(node),
        NodeType::Flatten => flatten_update_outputs(node),
        NodeType::Gelu => same_as_input(node),
        NodeType::Gather => gather_update_outputs(node),
        NodeType::GatherElements => same_as_input(node),
        NodeType::Greater => elementwise_comparison_outputs(node),
        NodeType::GreaterOrEqual => elementwise_comparison_outputs(node),
        NodeType::HardSigmoid => same_as_input(node),
        NodeType::GlobalAveragePool => same_as_input(node),
        NodeType::ConvTranspose1d => conv_transpose1d_update_outputs(node),
        NodeType::ConvTranspose2d => conv_transpose2d_update_outputs(node),
        NodeType::LayerNormalization => same_as_input(node),
        NodeType::LeakyRelu => same_as_input(node),
        NodeType::Less => elementwise_comparison_outputs(node),
        NodeType::LessOrEqual => elementwise_comparison_outputs(node),
        NodeType::Linear => linear_update_outputs(node),
        NodeType::Log => same_as_input(node),
        NodeType::LogSoftmax => same_as_input(node),
        NodeType::MatMul => matmul_update_outputs(node),
        NodeType::Max => same_as_input_broadcast(node),
        NodeType::MaxPool1d => same_as_input(node),
        NodeType::MaxPool2d => same_as_input(node),
        NodeType::Min => same_as_input_broadcast(node),
        NodeType::Mul => same_as_input(node),
        NodeType::Neg => same_as_input(node),
        NodeType::Not => same_as_input(node),
        NodeType::Pad => same_as_input(node),
        NodeType::PRelu => same_as_input_broadcast(node),
        NodeType::Pow => same_as_input_broadcast(node),
        NodeType::RandomNormal => random_update_output(node),
        NodeType::RandomNormalLike => random_like_update_output(node),
        NodeType::RandomUniform => random_update_output(node),
        NodeType::RandomUniformLike => random_like_update_output(node),
        NodeType::Range => range_update_outputs(node),
        NodeType::Reciprocal => same_as_input(node),
        NodeType::ReduceMax => reduce_max_update_outputs(node),
        NodeType::ReduceMin => reduce_min_update_outputs(node),
        NodeType::ReduceMean => reduce_mean_update_outputs(node),
        NodeType::ReduceProd => reduce_prod_update_outputs(node),
        NodeType::ReduceSum => reduce_sum_update_outputs(node),
        NodeType::Relu => same_as_input(node),
        NodeType::Reshape => reshape_update_outputs(node),
        NodeType::Resize => same_as_input(node),
        NodeType::Shape => shape_update_outputs(node),
        NodeType::Sigmoid => same_as_input(node),
        NodeType::Sign => same_as_input(node),
        NodeType::Sin => same_as_input(node),
        NodeType::Slice => same_as_input(node),
        NodeType::Softmax => same_as_input(node),
        NodeType::Squeeze => squeeze_update_output(node),
        NodeType::Sqrt => same_as_input(node),
        NodeType::Sub => same_as_input_broadcast(node),
        NodeType::Sum => same_as_input_broadcast(node),
        NodeType::Tanh => same_as_input(node),
        NodeType::Transpose => same_as_input(node),
        NodeType::Trilu => same_as_input(node),
        NodeType::Unsqueeze => unsqueeze_update_output(node),
        NodeType::Where => where_update_outputs(node),
        // Intentionally letting outputs leave unchanged but issue a warning so IR file can be generated.
        _ => temporary_pass_through_stub(node),
    }
}

fn constant_update_outputs(node: &mut Node) {
    // Fix the tensor rank of the output when the value is tensor

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
            AttributeValue::Tensor(tensor) if tensor.rank == 0 => {
                ArgType::Scalar(tensor.elem_type.clone())
            }
            AttributeValue::Tensor(tensor) => ArgType::Tensor(TensorType {
                elem_type: tensor.elem_type.clone(),
                rank: tensor.rank,
            }),
            AttributeValue::Float32(_) => ArgType::Scalar(ElementType::Float32),
            AttributeValue::Float32s(_) => ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 1,
            }),
            AttributeValue::Int64(_) => ArgType::Scalar(ElementType::Int64),
            AttributeValue::Int64s(_) => ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
            }),
            ty => panic!("Constant value of {:?} is not supported", ty),
        },
        None => panic!("Constant node must have a value attribute"),
    };
}
fn constant_of_shape_update_output(node: &mut Node) {
    let value_type = node
        .attrs
        .get("value")
        .map(|v| v.clone().into_tensor().elem_type)
        .unwrap_or(ElementType::Float32); // If not given, defaults to 0 as float32

    let dim = match &node.inputs[0].ty {
        ArgType::Shape(dim) => *dim,
        ArgType::Tensor(tensor_type) => tensor_type.rank,
        _ => panic!("ConstantOfShape node must have a Tensor or Shape type input"),
    };

    // Fix the input type to be a shape
    node.inputs[0].ty = ArgType::Shape(dim);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: value_type,
        rank: dim,
    });
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

    let rank = node
        .attrs
        .get("shape")
        .expect("required shape attribute missing")
        .clone()
        .into_i64s()
        .len();

    let elem_type = match dtype {
        DataType::FLOAT => ElementType::Float32,
        DataType::DOUBLE => ElementType::Float64,
        _ => panic!("tensor with type {dtype:?} not supported for random output"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType { elem_type, rank })
}

/// Reads & interprets an optional `dtype` attribute
/// This includes the `RandomUniformLike` and `RandomNormalLike` operators
fn random_like_update_output(node: &mut Node) {
    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
        .unwrap_or(DataType::FLOAT);

    let elem_type = match dtype {
        DataType::FLOAT => ElementType::Float32,
        DataType::FLOAT16 => ElementType::Float16,
        DataType::DOUBLE => ElementType::Float64,
        _ => panic!("Tensor with type {dtype:?} not supported for random output"),
    };

    if let ArgType::Tensor(tensor) = &node.inputs[0].clone().ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: tensor.rank,
        });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Infer the shape of the output tensor
fn linear_update_outputs(node: &mut Node) {
    // Extract the configuration of the linear layer (inputs are known)
    let node_input = &node.inputs[0];

    if let ArgType::Tensor(tensor) = node_input.clone().ty {
        // Update the output tensor
        node.outputs[0].ty = ArgType::Tensor(tensor.clone());
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
            if tensor.rank == 0 {
                // treat 0-dim tensor as scalar
                output.ty = ArgType::Scalar(elem_type);
                input.ty = ArgType::Scalar(tensor.elem_type);
            } else {
                // Cast input and output are the same shape, but possibly different types
                output.ty = ArgType::Tensor(TensorType {
                    elem_type,
                    rank: tensor.rank,
                });
            }
        }
        ArgType::Scalar(_scalar) => {
            output.ty = ArgType::Scalar(elem_type);
        }
        _ => panic!("Cast: only scalar and tensor inputs are valid"),
    }

    log::debug!(
        "Cast: input type: {:?}, output type: {:?}",
        input.ty,
        output.ty
    );
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
    let shape = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) => shape.clone(),
            _ => panic!("Reshape: missing or invalid shape input"),
        }
    } else {
        node.attrs
            .get("shape")
            .expect("Reshape: missing shape attribute")
            .clone()
            .into_i64s()
    };

    let output = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Reshape: invalid output types"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: shape.len(),
        ..output
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
        node.outputs[0].ty = ArgType::Tensor(TensorType { rank: 1, ..tensor });
    }
}

fn argmax_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Mean: multiple inputs are not supported");
    }

    let node_input = &mut node.inputs[0];
    let tensor = match node_input.clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Note: argmax in burn does not support keepdims=false
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: tensor.rank,
        elem_type: ElementType::Int64,
    });
}

/// Update the output tensor rank
fn squeeze_update_output(node: &mut Node) {
    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(TensorData {
                data: Data::Int64s(axes),
                ..
            }) => axes.clone(),
            _ => panic!("Squeeze: missing or invalid shape input"),
        }
    } else {
        node.attrs
            .get("axes")
            .expect("Squeeze: missing axes attribute")
            .clone()
            .into_i64s()
    };

    let input_dim = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Squeeze: invalid input type"),
    };

    let new_dim = input_dim - axes.len();

    let output_elem = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.elem_type.clone(),
        _ => panic!("Squeeze: invalid output type"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: new_dim,
        elem_type: output_elem,
    });
}

/// Updates the output type for operations that take more than one Tensor/Scalar of the a type
/// and returns the same type while supporting broadcasting
///
/// This is mostly the elementwise math operations, i.e. add, sub, mul, max etc.
fn same_as_input_broadcast(node: &mut Node) {
    if node.inputs.iter().all(|input| input.ty.is_scalar()) {
        // If all inputs are scalar, the output is a scalar as well
        node.outputs[0].ty = node.inputs[0].ty.clone();
    } else {
        // else, if any input is a Tensor, use it's datatype,
        // or the input consists solely from Scalars and Shapes,
        // which should result in another Shape
        node.outputs[0].ty = node
            .inputs
            .iter()
            .find(|input| input.ty.is_tensor())
            .map(|input| input.ty.clone())
            .unwrap_or_else(|| ArgType::Shape(0)); //Shape dim will be set by broadcast calculation
    }
}

/// Update the output tensor rank based on the "axes" attribute or the second input
fn unsqueeze_update_output(node: &mut Node) {
    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(TensorData {
                data: Data::Int64s(axes),
                ..
            }) => axes.clone(),
            _ => panic!("Unsqueeze: missing or invalid shape input"),
        }
    } else {
        node.attrs
            .get("axes")
            .expect("Unsqueeze: missing axes attribute")
            .clone()
            .into_i64s()
    };

    let input_dim = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0, // treat scalar as 0-dim tensor
        _ => panic!("Unsqueeze: invalid input type"),
    };

    let output_elem = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.elem_type.clone(),
        ArgType::Scalar(elem_type) => elem_type.clone(),
        _ => panic!("Unsqueeze: invalid output type"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: input_dim + axes.len(),
        elem_type: output_elem,
    });
}

fn same_as_input(node: &mut Node) {
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

/// Temporary pass-through stub for rank inference so that we can export the IR model.
fn temporary_pass_through_stub(node: &mut Node) {
    log::warn!("Must implement rank inference for {:?}", node);
    log::warn!("Temporarily setting the output type to the input type.");
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

/// Sets the output for binary operators resulting in a boolean output,
/// i.e., comparison operators like Equal, Greater, Less, etc.
///
/// Support for broadcasting is assumed
fn elementwise_comparison_outputs(node: &mut Node) {
    let input1_type = &node.inputs[0].ty;
    let input2_type = &node.inputs[1].ty;

    match (input1_type, input2_type) {
        (ArgType::Tensor(tensor), _) | (_, ArgType::Tensor(tensor)) => {
            // if one of the inputs is a tensor, the output is a tensor of bool
            assert_ne!(
                tensor.rank, 0,
                "Got a rank 0 Tensor, that should have been a Scalar!"
            );
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                ..tensor.clone()
            });
        }
        (ArgType::Scalar(_), ArgType::Scalar(_)) => {
            // if both inputs are scalars, the result is a scalar bool
            node.outputs[0].ty = ArgType::Scalar(ElementType::Bool);
        }
        _ => {
            panic!("Invalid input types for comparison op: {input1_type:?}, {input2_type:?}")
        }
    }
}

fn expand_update_outputs(node: &mut Node) {
    let shape = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) => shape.clone(),
            _ => vec![], //panic!("Expand: missing or invalid shape input"),
        }
    } else {
        node.attrs
            .get("shape")
            .expect("Expand: missing shape attribute")
            .clone()
            .into_i64s()
    };

    let output = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Expand: invalid output types"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: shape.len(),
        ..output
    });
}

fn shape_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Shape: multiple inputs are not supported: {:?}", node);
    }

    let (start, end) = shape_config(node);
    let dim = end - start;
    node.outputs[0].ty = ArgType::Shape(dim);
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

    let input_dim = tensor.rank;

    let (start_dim, end_dim) = flatten_config(node);

    let collapsed_dims = end_dim - start_dim;
    let output_dim = input_dim - collapsed_dims;

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: output_dim,
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

/// Infers the shape of a ConvTranspose1d node and replaces the shape of the output tensor.
fn conv_transpose1d_update_outputs(node: &mut Node) {
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

fn matmul_update_outputs(node: &mut Node) {
    // NOTE: matmul only supported for float tensors
    match (node.inputs[0].ty.clone(), node.inputs[1].ty.clone()) {
        (ArgType::Tensor(a), ArgType::Tensor(b)) => {
            // With broadcasting support, output dim has to be computed based on the inputs
            let mut out_dim = max(a.rank, b.rank);

            // Matrix-vector or vector-matrix product
            if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                out_dim -= 1;
            }

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: a.elem_type.clone(),
                rank: out_dim,
            });
        }
        _ => panic!("Only tensor input is valid"),
    }
}

fn range_update_outputs(node: &mut Node) {
    if node.inputs.len() != 3 {
        panic!("Range: expected 3 inputs, found {}", node.inputs.len());
    }

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: 1,
    });
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
        node.outputs[0].ty = ArgType::Tensor(TensorType { rank: 1, ..tensor });
    }
}

fn reduce_min_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ReduceMin: multiple inputs are not supported");
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
        node.outputs[0].ty = ArgType::Tensor(TensorType { rank: 1, ..tensor });
    }
}

/// Infers the shape of a ReduceProd node and replaces the shape of the output tensor.
fn reduce_prod_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ReduceProd: multiple inputs are not supported");
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
        node.outputs[0].ty = ArgType::Tensor(TensorType { rank: 1, ..tensor });
    }
}

/// Infers the shape of a ReduceSum node and replaces the shape of the output tensor.
fn reduce_sum_update_outputs(node: &mut Node) {
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

    let dim_only = match node.inputs.get(1).and_then(|arg| arg.value.as_ref()) {
        Some(value) => value.rank == 1 || value.rank == 0,
        None => dim_only,
    };

    if dim_only {
        node.outputs[0].ty = ArgType::Tensor(tensor);
    } else {
        // NOTE: ReduceSum w/o keepdims reduces to a scalar value, but Burn doesn't have
        // 0-dim tensor so we can't track or perform other ops on that value if we call
        // `.into_scalar()` on the result of `tensor.sum()`
        // node.outputs[0].ty = ArgType::Scalar(tensor.elem_type);
        // Instead, we return a tensor of rank 1 (the result of `tensor.sum()`)
        node.outputs[0].ty = ArgType::Tensor(TensorType { rank: 1, ..tensor });
    }
}

fn where_update_outputs(node: &mut Node) {
    let condition = &node.inputs[0].ty;
    let x = &node.inputs[1].ty;
    let y = &node.inputs[2].ty;
    let elem_type = x.elem_type().clone();
    assert_eq!(
        *condition.elem_type(),
        ElementType::Bool,
        "Where condition must be boolean!"
    );
    assert_eq!(
        elem_type,
        *y.elem_type(),
        "Where x and y have different element types!"
    );

    let output_rank = max(condition.rank(), max(x.rank(), y.rank()));
    if output_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(elem_type);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: output_rank,
        });
    }
}

fn gather_update_outputs(node: &mut Node) {
    if node.inputs.len() != 2 {
        panic!("Gather requires two inputs: data and indices");
    }

    let indices_dim = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0,
        _ => panic!("Only tensor indices is valid, got {:?}", node.inputs[1].ty),
    };

    match &node.inputs[0].ty {
        ArgType::Tensor(input_tensor) => {
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_dim + input_tensor.rank - 1;

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: input_tensor.elem_type.clone(),
                rank: output_rank,
            });
        }
        ArgType::Shape(_dim) => {
            let shape_dim = 1;
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_dim + shape_dim - 1;

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: output_rank,
            })
        }
        ty => panic!("Only tensor/shape input is valid but received: {:?}", ty),
    }
}
