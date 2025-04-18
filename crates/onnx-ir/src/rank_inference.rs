use core::cmp::max;
use core::panic;

use protobuf::Enum;

use crate::{
    ir::{ArgType, AttributeValue, Data, ElementType, Node, NodeType, TensorType},
    protos::tensor_proto::DataType,
    util::shape_config,
};

/// Infer the rank of each output tensor and update them based solely on rank inference.
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
        NodeType::Cosh => same_as_input(node),
        NodeType::Div => same_as_input_broadcast(node),
        NodeType::Dropout => same_as_input(node),
        NodeType::Equal => elementwise_comparison_outputs(node),
        NodeType::Erf => same_as_input(node),
        NodeType::Exp => same_as_input(node),
        NodeType::Expand => expand_update_outputs(node),
        NodeType::Floor => same_as_input(node),
        NodeType::Flatten => flatten_update_outputs(node),
        NodeType::Gelu => same_as_input(node),
        NodeType::Gather => gather_update_outputs(node),
        NodeType::GatherElements => same_as_input(node),
        NodeType::Gemm => gemm_output_shape(node),
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
        NodeType::OneHot => one_hot_output_shape(node),
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
        NodeType::Sinh => same_as_input(node),
        NodeType::Slice => same_as_input(node),
        NodeType::Softmax => same_as_input(node),
        NodeType::Split => split_update_outputs(node),
        NodeType::Squeeze => squeeze_update_output(node),
        NodeType::Sqrt => same_as_input(node),
        NodeType::Sub => same_as_input_broadcast(node),
        NodeType::Sum => same_as_input_broadcast(node),
        NodeType::Tan => same_as_input(node),
        NodeType::Tanh => same_as_input(node),
        NodeType::TopK => top_k_update_output(node),
        NodeType::Transpose => same_as_input(node),
        NodeType::Trilu => same_as_input(node),
        NodeType::Unsqueeze => unsqueeze_update_output(node),
        NodeType::Where => where_update_outputs(node),
        _ => temporary_pass_through_stub(node),
    }
}

/// Update output type for constant nodes based on attribute values, focusing on rank only.
fn constant_update_outputs(node: &mut Node) {
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
            AttributeValue::Tensor(tensor) if tensor.rank == 0 => {
                ArgType::Scalar(tensor.elem_type.clone())
            }
            AttributeValue::Tensor(tensor) => ArgType::Tensor(TensorType {
                elem_type: tensor.elem_type.clone(),
                rank: tensor.rank,
                static_shape: None,
            }),
            AttributeValue::Float32(_) => ArgType::Scalar(ElementType::Float32),
            AttributeValue::Float32s(_) => ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 1,
                static_shape: None,
            }),
            AttributeValue::Int64(_) => ArgType::Scalar(ElementType::Int64),
            AttributeValue::Int64s(_) => ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: None,
            }),
            ty => panic!("Constant value of {:?} is not supported", ty),
        },
        None => panic!("Constant node must have a value attribute"),
    };
}

/// Updates the output rank for a ConstantOfShape node based on the rank of its input.
fn constant_of_shape_update_output(node: &mut Node) {
    let value_type = node
        .attrs
        .get("value")
        .map(|v| v.clone().into_tensor().elem_type)
        .unwrap_or(ElementType::Float32); // If not given, defaults to 0 as float32

    let rank = match &node.inputs[0].ty {
        ArgType::Shape(rank) => *rank,
        ArgType::Tensor(tensor_type) => tensor_type
            .static_shape
            .as_ref()
            .and_then(|shape| shape.first())
            .copied()
            .expect("ConstantOfShape node must have a Tensor with a non-empty static shape value"),
        _ => panic!("ConstantOfShape node requires a Tensor or Shape type as input"),
    };

    // Update the input type to be a shape
    node.inputs[0].ty = ArgType::Shape(rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: value_type,
        rank,
        static_shape: None,
    });
}

/// Update output rank for Random operations with explicit shape attribute.
fn random_update_output(node: &mut Node) {
    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
        .unwrap_or(DataType::FLOAT);
    let shape = node
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
        rank: shape.len(),
        static_shape: None,
    });
}

/// Update output rank for RandomLike operations based on input rank.
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

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: tensor.rank,
            static_shape: tensor.static_shape.clone(),
        });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for Linear operations (same as input rank).
fn linear_update_outputs(node: &mut Node) {
    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output type for Cast operations, preserving rank.
fn cast_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Cast: multiple inputs are not supported");
    }
    let input = &mut node.inputs[0];
    let output = &mut node.outputs[0];

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
        None => panic!("Cast node must have a 'to' attribute"),
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
                    static_shape: None,
                });
            }
        }
        ArgType::Scalar(_) => output.ty = ArgType::Scalar(elem_type),
        _ => panic!("Cast: only scalar and tensor inputs are valid"),
    }
}

/// Update output rank for Concat (same as first tensor input).
fn concat_update_outputs(node: &mut Node) {
    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match &input.ty {
            ArgType::Tensor(tensor) => Some(tensor.clone()),
            _ => None,
        })
        .unwrap();

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type,
        rank: tensor.rank,
        static_shape: None,
    });
}

/// Update output rank for Reshape based on shape input if constant, otherwise use input rank.
fn reshape_update_outputs(node: &mut Node) {
    let shape = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(shape) => Some(shape.clone()),
                _ => panic!("Reshape: invalid input types"),
            },
            None => None,
        }
    } else {
        node.attrs.get("shape").cloned().map(|v| v.into_i64s())
    };

    let output = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Reshape: invalid output types"),
    };

    let rank = match &shape {
        Some(s) => s.len(),
        None => output.rank,
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank,
        static_shape: None,
        ..output
    });
}

/// Update output rank for ReduceMean based on axes.
fn reduce_mean_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ReduceMean: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
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

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: if dim_only { tensor.rank } else { 1 },
        static_shape: None,
    });
}

/// Update output rank for ArgMax (same as input rank).
fn argmax_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ArgMax: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Note: argmax in burn does not support keepdims=false
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: tensor.rank,
        static_shape: None,
    });
}

/// Update output rank for Squeeze based on axes.
fn squeeze_update_output(node: &mut Node) {
    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(axes) => Some(axes.clone()),
                _ => panic!("Squeeze: invalid input types"),
            },
            None => None,
        }
    } else {
        node.attrs.get("axes").cloned().map(|v| v.into_i64s())
    };

    let axes = axes.unwrap_or_else(|| panic!("Squeeze must specify an axis"));
    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Squeeze: invalid input type"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank: input_rank - axes.len(),
        static_shape: None,
    });
}

/// Update output rank for broadcasting operations (e.g., Add, Sub) to max input rank.
fn same_as_input_broadcast(node: &mut Node) {
    let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => acc.max(tensor.rank),
        ArgType::Scalar(_) => acc,
        _ => panic!("Unsupported input type for broadcasting operation"),
    });

    if max_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(node.inputs[0].ty.elem_type().clone());
    } else {
        let elem_type = node
            .inputs
            .iter()
            .find_map(|input| match &input.ty {
                ArgType::Tensor(tensor) => Some(tensor.elem_type.clone()),
                _ => None,
            })
            .unwrap_or_else(|| node.inputs[0].ty.elem_type().clone());

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: max_rank,
            static_shape: None,
            // Removed call to set_broadcasting_output_shape
        });
    }
}

/// Update output rank for Unsqueeze based on axes.
/// Update the output tensor dimension based on the "axes" attribute or the second input
fn unsqueeze_update_output(node: &mut Node) {
    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match value.data {
                Data::Int64s(ref axes) => Some(axes.clone()),
                _ => panic!("Unsqueeze: invalid input types"),
            },
            // We cannot support dynamic input for axes
            // because it would require dynamic rank
            // burn supports static rank.
            // The output rank is determined by the input rank and the axes attribute.
            None => panic!("Unsqueeze: Dynamic input for axes is not supported"),
        }
    } else {
        node.attrs.get("axes").cloned().map(|v| v.into_i64s())
    };
    if axes.is_none() {
        return;
    }

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0, // treat scalar as 0-dim tensor
        _ => panic!("Unsqueeze: invalid input type"),
    };

    let output_elem = match &node.outputs[0].ty {
        ArgType::Tensor(_) => node.inputs[0].ty.elem_type().clone(),
        ArgType::Scalar(elem_type) => elem_type.clone(),
        _ => panic!("Unsqueeze: invalid output type"),
    };

    if let Some(axes) = axes {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            rank: input_rank + axes.len(),
            static_shape: None, // shape is tracked and calculated at runtime
            elem_type: output_elem,
        });
    }
}

/// Preserve input rank for operations like Relu, Sigmoid, etc.
fn same_as_input(node: &mut Node) {
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

/// Update output rank for TopK (same as input rank).
fn top_k_update_output(node: &mut Node) {
    let rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("TopK: invalid input type"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank,
        static_shape: None,
    });
    node.outputs[1].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank,
        static_shape: None,
    });
}

/// Temporary stub preserves input type for unhandled operations.
fn temporary_pass_through_stub(node: &mut Node) {
    log::warn!("Must implement rank inference for {:?}", node);
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

/// Update output rank for comparison operations (e.g., Equal, Greater) to max input rank.
fn elementwise_comparison_outputs(node: &mut Node) {
    let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => acc.max(tensor.rank),
        ArgType::Scalar(_) => acc,
        _ => panic!("Invalid input type for comparison op"),
    });

    if max_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(ElementType::Bool);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Bool,
            rank: max_rank,
            static_shape: None,
        });
    }
}

/// Updates the output rank and shape for the Expand operation based on the provided shape input.
/// If the shape is a constant, the rank and static shape of the output are set accordingly.
/// If the shape is dynamic, the rank is inferred from the static shape of the shape input.
fn expand_update_outputs(node: &mut Node) {
    let shape = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(shape) => Some(shape.clone()),
                _ => panic!("Expand operation encountered invalid input types"),
            },
            None => None,
        }
    } else {
        panic!("Expand operation requires exactly two inputs");
    };

    let output = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Expand operation encountered invalid output types"),
    };

    if let Some(shape) = shape {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            rank: shape.len(),
            static_shape: Some(shape.into_iter().map(|dim| dim as usize).collect()),
            ..output
        });
    } else {
        // When the shape cannot be determined statically (i.e., the second argument 'shape' is passed dynamically),
        // infer the rank from the static shape of the input tensor.
        let output_rank = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor
                .static_shape
                .as_ref()
                .expect("Shape input must have a static shape defined")
                .first()
                .copied()
                .expect("Static shape must contain at least one element"),
            ArgType::Shape(rank) => *rank,
            _ => panic!("Shape input must be of tensor or shape type",),
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            rank: output_rank,
            static_shape: None, // The exact shape cannot be determined statically
            ..output
        });
    }
}

/// Update output type for Shape operation (rank 1).
fn shape_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Shape: multiple inputs are not supported: {:?}", node);
    }
    let (start, end) = shape_config(node);
    let dim = end - start;
    node.outputs[0].ty = ArgType::Shape(dim);
}

/// Update output type for Flatten operation (rank 2).
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

    // Flatten to a 2D tensor
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: 2,
        ..tensor.clone()
    });
}

/// Update output rank for Conv1d (same as input).
fn conv1d_update_outputs(node: &mut Node) {
    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for Conv2d (same as input).
fn conv2d_update_outputs(node: &mut Node) {
    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for ConvTranspose1d (same as input).
fn conv_transpose1d_update_outputs(node: &mut Node) {
    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for ConvTranspose2d (same as input).
fn conv_transpose2d_update_outputs(node: &mut Node) {
    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for MatMul based on input ranks.
fn matmul_update_outputs(node: &mut Node) {
    match (&node.inputs[0].ty, &node.inputs[1].ty) {
        (ArgType::Tensor(a), ArgType::Tensor(b)) => {
            let mut out_rank = max(a.rank, b.rank);
            if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                out_rank -= 1;
            }
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: a.elem_type.clone(),
                rank: out_rank,
                static_shape: None,
            });
        }
        _ => panic!("Only tensor inputs are valid"),
    }
}

/// Update output rank for Range (always rank 1).
fn range_update_outputs(node: &mut Node) {
    if node.inputs.len() != 3 {
        panic!("Range: expected 3 inputs, found {}", node.inputs.len());
    }
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: 1,
        static_shape: None,
    });
}

/// Update output rank for ReduceMax based on axes.
fn reduce_max_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ReduceMax: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
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

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: if dim_only { tensor.rank } else { 1 },
        static_shape: None,
    });
}

/// Update output rank for ReduceMin based on axes.
fn reduce_min_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ReduceMin: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
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

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: if dim_only { tensor.rank } else { 1 },
        static_shape: None,
    });
}

/// Update output rank for ReduceProd based on axes.
fn reduce_prod_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("ReduceProd: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
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

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: if dim_only { tensor.rank } else { 1 },
        static_shape: None,
    });
}

/// Update output rank for ReduceSum based on axes.
fn reduce_sum_update_outputs(node: &mut Node) {
    let tensor = match &node.inputs[0].ty {
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
    } || match node.inputs.get(1).and_then(|arg| arg.value.as_ref()) {
        Some(value) => match &value.data {
            Data::Int64(_) => true,
            Data::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => false,
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: if dim_only { tensor.rank } else { 1 },
        static_shape: None,
    });
}

/// Update output rank for Where to max input rank.
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
            static_shape: None,
        });
    }
}

/// Update output rank for Gather based on input and indices ranks.
fn gather_update_outputs(node: &mut Node) {
    if node.inputs.len() != 2 {
        panic!("Gather requires two inputs: data and indices");
    }

    let indices_rank = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0,
        _ => panic!("Only tensor indices is valid, got {:?}", node.inputs[1].ty),
    };

    match &node.inputs[0].ty {
        ArgType::Tensor(input_tensor) => {
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_rank + input_tensor.rank - 1;
            if output_rank == 0 {
                node.outputs[0].ty = ArgType::Scalar(input_tensor.elem_type.clone());
            } else {
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: input_tensor.elem_type.clone(),
                    rank: output_rank,
                    static_shape: None,
                });
            }
        }
        ArgType::Shape(_) => {
            let shape_rank = 1;
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_rank + shape_rank - 1;
            if output_rank == 0 {
                node.outputs[0].ty = ArgType::Scalar(ElementType::Int64);
            } else {
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: output_rank,
                    static_shape: None,
                });
            }
        }
        ty => panic!("Only tensor/shape input is valid but received: {:?}", ty),
    }
}

/// Update output rank for Split (same as input).
fn split_update_outputs(node: &mut Node) {
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Split: Input must be a tensor"),
    };

    for output_arg in node.outputs.iter_mut() {
        output_arg.ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
    }
}

/// Update output rank for OneHot (input rank + 1).
fn one_hot_output_shape(node: &mut Node) {
    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("OneHot: invalid input type"),
    };
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.outputs[0].ty.elem_type().clone(),
        rank: input_rank + 1,
        static_shape: None,
    });
}

fn gemm_output_shape(node: &mut Node) {
    let a_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Input A should be a tensor!"),
    };
    let b_rank = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Input B should be a tensor!"),
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: max(a_rank, b_rank),
        static_shape: None,
        elem_type: match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.elem_type.clone(),
            _ => panic!("Unexpected type for input A"),
        },
    });
}
