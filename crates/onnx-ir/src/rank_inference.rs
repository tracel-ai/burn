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
    log::debug!("Inferring rank for node: {}", node.name);

    match node.node_type {
        NodeType::Add => same_as_input_broadcast(node),
        NodeType::ArgMax => argmax_update_outputs(node),
        NodeType::AveragePool1d => same_as_input(node),
        NodeType::AveragePool2d => same_as_input(node),
        NodeType::BatchNormalization => same_as_input(node),
        NodeType::Cast => cast_update_outputs(node),
        NodeType::Clip => same_as_input(node),
        NodeType::Concat => {
            concat_update_outputs_safe(node).unwrap_or_else(|e| {
                panic!("Concat rank inference failed: {:?}", e);
            });
        }
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

    log::debug!(
        "Rank inference result for {}: {:?}",
        node.name,
        node.outputs
    );
}

/// Update output type for constant nodes based on attribute values, focusing on rank only.
fn constant_update_outputs(node: &mut Node) {
    log::debug!("Constant rank inference for node {}", node.name);

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
    log::debug!("Constant found attribute: {}", matched_value.is_some());

    node.outputs[0].ty = match matched_value {
        Some(value) => match &value {
            AttributeValue::Tensor(tensor) if tensor.shape.is_empty() => {
                log::debug!("Constant as scalar for {}", node.name);
                ArgType::Scalar(tensor.elem_type())
            }
            AttributeValue::Tensor(tensor) => {
                log::debug!(
                    "Constant tensor with rank {} for {}",
                    tensor.shape.len(),
                    node.name
                );
                ArgType::Tensor(TensorType {
                    elem_type: tensor.elem_type(),
                    rank: tensor.shape.len(),
                    static_shape: None,
                })
            }
            AttributeValue::Float32(_) => {
                log::debug!("Constant Float32 scalar for {}", node.name);
                ArgType::Scalar(ElementType::Float32)
            }
            AttributeValue::Float32s(_) => {
                log::debug!("Constant Float32s tensor with rank 1 for {}", node.name);
                ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                })
            }
            AttributeValue::Int64(_) => {
                log::debug!("Constant Int64 scalar for {}", node.name);
                ArgType::Scalar(ElementType::Int64)
            }
            AttributeValue::Int64s(_) => {
                log::debug!("Constant Int64s tensor with rank 1 for {}", node.name);
                ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: None,
                })
            }
            ty => panic!("Constant value of {:?} is not supported", ty),
        },
        None => panic!("Constant node must have a value attribute"),
    };
}

/// Updates the output rank for a ConstantOfShape node based on the rank of its input.
fn constant_of_shape_update_output(node: &mut Node) {
    log::debug!("ConstantOfShape rank inference for node {}", node.name);

    let value_type = node
        .attrs
        .get("value")
        .map(|v| v.clone().into_tensor().elem_type())
        .unwrap_or(ElementType::Float32); // If not given, defaults to 0 as float32
    log::debug!(
        "ConstantOfShape value type for {}: {:?}",
        node.name,
        value_type
    );

    let rank = match &node.inputs[0].ty {
        ArgType::Shape(rank) => {
            log::debug!(
                "ConstantOfShape input is Shape with rank {} for {}",
                rank,
                node.name
            );
            *rank
        }
        ArgType::Tensor(tensor_type) => {
            log::debug!("ConstantOfShape input is Tensor for {}", node.name);
            let r = tensor_type
                .static_shape
                .as_ref()
                .and_then(|shape| shape.first())
                .copied()
                .expect(
                    "ConstantOfShape node must have a Tensor with a non-empty static shape value",
                );
            log::debug!(
                "ConstantOfShape derived rank from tensor: {} for {}",
                r,
                node.name
            );
            r
        }
        _ => panic!("ConstantOfShape node requires a Tensor or Shape type as input"),
    };

    // Update the input type to be a shape
    node.inputs[0].ty = ArgType::Shape(rank);
    log::debug!(
        "ConstantOfShape updated input to Shape({}) for {}",
        rank,
        node.name
    );

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: value_type,
        rank,
        static_shape: None,
    });
    log::debug!("ConstantOfShape output rank for {}: {}", node.name, rank);
}

/// Update output rank for Random operations with explicit shape attribute.
fn random_update_output(node: &mut Node) {
    log::debug!("Random rank inference for node {}", node.name);

    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
        .unwrap_or(DataType::FLOAT);
    log::debug!("Random dtype for {}: {:?}", node.name, dtype);

    let shape = node
        .attrs
        .get("shape")
        .expect("required shape attribute missing")
        .clone()
        .into_i64s();
    log::debug!("Random shape for {}: {:?}", node.name, shape);

    let elem_type = match dtype {
        DataType::FLOAT => ElementType::Float32,
        DataType::DOUBLE => ElementType::Float64,
        _ => panic!("tensor with type {dtype:?} not supported for random output"),
    };

    let rank = shape.len();
    log::debug!("Random output rank for {}: {}", node.name, rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type,
        rank,
        static_shape: None,
    });
}

/// Update output rank for RandomLike operations based on input rank.
fn random_like_update_output(node: &mut Node) {
    log::debug!("RandomLike rank inference for node {}", node.name);

    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
        .unwrap_or(DataType::FLOAT);
    log::debug!("RandomLike dtype for {}: {:?}", node.name, dtype);

    let elem_type = match dtype {
        DataType::FLOAT => ElementType::Float32,
        DataType::FLOAT16 => ElementType::Float16,
        DataType::DOUBLE => ElementType::Float64,
        _ => panic!("Tensor with type {dtype:?} not supported for random output"),
    };

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("RandomLike input rank for {}: {}", node.name, tensor.rank);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: tensor.rank,
            static_shape: tensor.static_shape.clone(),
        });

        log::debug!("RandomLike output rank for {}: {}", node.name, tensor.rank);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for Linear operations (same as input rank).
fn linear_update_outputs(node: &mut Node) {
    log::debug!("Linear rank inference for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("Linear input rank for {}: {}", node.name, tensor.rank);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });

        log::debug!("Linear output rank for {}: {}", node.name, tensor.rank);
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
    log::debug!("Concat rank inference for node {}", node.name);

    // Find the first tensor input, with better error handling
    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match &input.ty {
            ArgType::Tensor(tensor) => Some(tensor.clone()),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!(
                "Concat node '{}' has no valid tensor inputs. Inputs: {:?}",
                node.name,
                node.inputs
                    .iter()
                    .map(|i| format!("{:?}", i.ty))
                    .collect::<Vec<_>>()
            )
        });

    log::debug!("Concat using input rank for {}: {}", node.name, tensor.rank);

    // Verify all inputs have compatible ranks
    for (idx, input) in node.inputs.iter().enumerate() {
        match &input.ty {
            ArgType::Tensor(t) => {
                if t.rank != tensor.rank {
                    panic!(
                        "Concat node '{}' has mismatched ranks. Expected rank {} but input {} has rank {}",
                        node.name, tensor.rank, idx, t.rank
                    );
                }
            }
            _ => {
                panic!(
                    "Concat node '{}' has non-tensor input at position {}. Input type: {:?}",
                    node.name, idx, input.ty
                );
            }
        }
    }

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type,
        rank: tensor.rank,
        static_shape: None,
    });

    log::debug!("Concat output rank for {}: {}", node.name, tensor.rank);
}

/// Update output rank for Reshape based on shape input if constant, otherwise use input rank.
fn reshape_update_outputs(node: &mut Node) {
    log::debug!("Reshape rank inference for node {}", node.name);

    let shape = if node.inputs.len() == 2 {
        log::debug!("Reshape node {} has shape as second input", node.name);
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(shape) => {
                    log::debug!("Reshape node {} has constant shape: {:?}", node.name, shape);
                    Some(shape.clone())
                }
                _ => panic!("Reshape: invalid input types"),
            },
            None => {
                log::debug!(
                    "Reshape node {} has dynamic shape as second input",
                    node.name
                );
                None
            }
        }
    } else {
        log::debug!("Reshape node {} using shape from attributes", node.name);
        node.attrs.get("shape").cloned().map(|v| {
            let shape = v.into_i64s();
            log::debug!("Reshape node {} shape attribute: {:?}", node.name, shape);
            shape
        })
    };

    let output = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Reshape: invalid output types"),
    };

    let rank = match &shape {
        Some(s) => s.len(),
        None => output.rank,
    };

    log::debug!("Reshape output rank for node {}: {}", node.name, rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank,
        static_shape: None,
        ..output
    });
}

/// Update output rank for ReduceMean based on axes.
fn reduce_mean_update_outputs(node: &mut Node) {
    log::debug!("ReduceMean rank inference for node {}", node.name);

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

    let output_rank = if dim_only { tensor.rank } else { 1 };
    log::debug!("ReduceMean output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: output_rank,
        static_shape: None,
    });
}

/// Update output rank for ArgMax (same as input rank).
fn argmax_update_outputs(node: &mut Node) {
    log::debug!("ArgMax rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ArgMax: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    log::debug!("ArgMax input rank for {}: {}", node.name, tensor.rank);

    // Note: argmax in burn does not support keepdims=false
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: tensor.rank,
        static_shape: None,
    });

    log::debug!("ArgMax output rank for {}: {}", node.name, tensor.rank);
}

/// Update output rank for Squeeze based on axes.
fn squeeze_update_output(node: &mut Node) {
    log::debug!("Squeeze rank inference for node {}", node.name);

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
    log::debug!("Squeeze axes for {}: {:?}", node.name, axes);

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Squeeze: invalid input type"),
    };
    log::debug!("Squeeze input rank for {}: {}", node.name, input_rank);

    let output_rank = input_rank - axes.len();
    log::debug!("Squeeze output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank: output_rank,
        static_shape: None,
    });
}

/// Update output rank for broadcasting operations (e.g., Add, Sub) to max input rank.
fn same_as_input_broadcast(node: &mut Node) {
    log::debug!("Broadcasting operation for node {}", node.name);

    let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => acc.max(tensor.rank),
        ArgType::Scalar(_) => acc,
        _ => panic!("Unsupported input type for broadcasting operation"),
    });

    log::debug!("Max rank for broadcasting node {}: {}", node.name, max_rank);

    if max_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(node.inputs[0].ty.elem_type().clone());
        log::debug!("Scalar result for node {}", node.name);
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
        log::debug!(
            "Tensor result for node {} with rank {}",
            node.name,
            max_rank
        );
    }
}

/// Update output rank for Unsqueeze based on axes.
/// Update the output tensor dimension based on the "axes" attribute or the second input
fn unsqueeze_update_output(node: &mut Node) {
    log::debug!("Unsqueeze rank inference for node {}", node.name);

    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(a) => Some(a.clone()),
                _ => panic!("Unsqueeze: invalid input types"),
            },
            None => None,
        }
    } else {
        let axes = node.attrs.get("axes").cloned().map(|v| {
            let axes = v.into_i64s();
            log::debug!(
                "Unsqueeze axes from attribute for {}: {:?}",
                node.name,
                axes
            );
            axes
        });
        axes
    };

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => {
            0 // treat scalar as 0-dim tensor
        }
        _ => panic!("Unsqueeze: invalid input type"),
    };

    let output_elem = match &node.outputs[0].ty {
        ArgType::Tensor(_) => node.inputs[0].ty.elem_type().clone(),
        ArgType::Scalar(elem_type) => elem_type.clone(),
        _ => panic!("Unsqueeze: invalid output type"),
    };

    let output_rank = if let Some(axes) = axes {
        input_rank + axes.len()
    } else if let ArgType::Tensor(tensor) = &node.inputs[1].ty {
        if let Some(static_shape) = &tensor.static_shape {
            input_rank + *static_shape.first().expect("Empty shape")
        } else {
            panic!("Unsqueeze: should have static shape")
        }
    } else {
        panic!("Unsqueeze: missing axes information")
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape: None, // shape is tracked and calculated at runtime
        elem_type: output_elem,
    });

    log::debug!("Unsqueeze output rank for {}: {}", node.name, output_rank);
}

/// Preserve input rank for operations like Relu, Sigmoid, etc.
fn same_as_input(node: &mut Node) {
    log::debug!("Copying input type to output for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("Input rank for {}: {}", node.name, tensor.rank);
    } else if let ArgType::Scalar(_) = &node.inputs[0].ty {
        log::debug!("Input is scalar for {}", node.name);
    }

    node.outputs[0].ty = node.inputs[0].ty.clone();
    log::debug!("Output type is same as input for {}", node.name);
}

/// Update output rank for TopK (same as input rank).
fn top_k_update_output(node: &mut Node) {
    log::debug!("TopK rank inference for node {}", node.name);

    let rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("TopK: invalid input type"),
    };
    log::debug!("TopK input rank for {}: {}", node.name, rank);

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

    log::debug!(
        "TopK output rank for {}: {} (both outputs)",
        node.name,
        rank
    );
}

/// Temporary stub preserves input type for unhandled operations.
fn temporary_pass_through_stub(node: &mut Node) {
    log::warn!(
        "Must implement rank inference for node type {:?} (name: {})",
        node.node_type,
        node.name
    );

    if let Some(input_rank) = node.inputs.first().map(|input| match &input.ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0,
        _ => 0,
    }) {
        log::debug!(
            "Passing through input rank {} for unhandled node {}",
            input_rank,
            node.name
        );
    }

    node.outputs[0].ty = node.inputs[0].ty.clone();
    log::debug!(
        "Using pass-through inference for unhandled node type {:?} ({})",
        node.node_type,
        node.name
    );
}

/// Update output type for comparison operations (e.g., Equal, Greater) to max input rank.
fn elementwise_comparison_outputs(node: &mut Node) {
    log::debug!("Elementwise comparison for node {}", node.name);

    let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => acc.max(tensor.rank),
        ArgType::Scalar(_) => acc,
        _ => panic!("Invalid input type for comparison op"),
    });

    log::debug!("Max rank for comparison node {}: {}", node.name, max_rank);

    if max_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(ElementType::Bool);
        log::debug!("Scalar boolean result for node {}", node.name);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Bool,
            rank: max_rank,
            static_shape: None,
        });
        log::debug!(
            "Tensor boolean result for node {} with rank {}",
            node.name,
            max_rank
        );
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
    log::debug!(
        "Shape operation for node {}: start={}, end={}, dim={}",
        node.name,
        start,
        end,
        dim
    );
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
    log::debug!("Conv1d rank inference for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("Conv1d input rank for {}: {}", node.name, tensor.rank);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });

        log::debug!("Conv1d output rank for {}: {}", node.name, tensor.rank);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for Conv2d (same as input).
fn conv2d_update_outputs(node: &mut Node) {
    log::debug!("Conv2d rank inference for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("Conv2d input rank for {}: {}", node.name, tensor.rank);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });

        log::debug!("Conv2d output rank for {}: {}", node.name, tensor.rank);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for ConvTranspose1d (same as input).
fn conv_transpose1d_update_outputs(node: &mut Node) {
    log::debug!("ConvTranspose1d rank inference for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!(
            "ConvTranspose1d input rank for {}: {}",
            node.name,
            tensor.rank
        );

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });

        log::debug!(
            "ConvTranspose1d output rank for {}: {}",
            node.name,
            tensor.rank
        );
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for ConvTranspose2d (same as input).
fn conv_transpose2d_update_outputs(node: &mut Node) {
    log::debug!("ConvTranspose2d rank inference for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!(
            "ConvTranspose2d input rank for {}: {}",
            node.name,
            tensor.rank
        );

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });

        log::debug!(
            "ConvTranspose2d output rank for {}: {}",
            node.name,
            tensor.rank
        );
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Update output rank for MatMul based on input ranks.
fn matmul_update_outputs(node: &mut Node) {
    log::debug!("MatMul rank inference for node {}", node.name);

    match (&node.inputs[0].ty, &node.inputs[1].ty) {
        (ArgType::Tensor(a), ArgType::Tensor(b)) => {
            log::debug!(
                "MatMul input ranks for {}: a.rank={}, b.rank={}",
                node.name,
                a.rank,
                b.rank
            );

            let mut out_rank = max(a.rank, b.rank);
            if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                out_rank -= 1;
                log::debug!(
                    "MatMul special case for node {}: reducing output rank",
                    node.name
                );
            }

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: a.elem_type.clone(),
                rank: out_rank,
                static_shape: None,
            });

            log::debug!("MatMul output rank for {}: {}", node.name, out_rank);
        }
        _ => panic!("Only tensor inputs are valid"),
    }
}

/// Update output rank for Range (always rank 1).
fn range_update_outputs(node: &mut Node) {
    log::debug!("Range rank inference for node {}", node.name);

    if node.inputs.len() != 3 {
        panic!("Range: expected 3 inputs, found {}", node.inputs.len());
    }
    log::debug!(
        "Range operation always produces rank 1 tensor for {}",
        node.name
    );

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: 1,
        static_shape: None,
    });

    log::debug!("Range output rank for {}: 1", node.name);
}

/// Update output rank for ReduceMax based on axes.
fn reduce_max_update_outputs(node: &mut Node) {
    log::debug!("ReduceMax rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ReduceMax: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };
    log::debug!("ReduceMax input rank for {}: {}", node.name, tensor.rank);

    let dim_only = match node.attrs.get("axes") {
        Some(value) => match &value {
            AttributeValue::Int64(_) => true,
            AttributeValue::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => false,
    };

    let output_rank = if dim_only { tensor.rank } else { 1 };
    log::debug!("ReduceMax output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: output_rank,
        static_shape: None,
    });
}

/// Update output rank for ReduceMin based on axes.
fn reduce_min_update_outputs(node: &mut Node) {
    log::debug!("ReduceMin rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ReduceMin: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };
    log::debug!("ReduceMin input rank for {}: {}", node.name, tensor.rank);

    let dim_only = match node.attrs.get("axes") {
        Some(value) => match &value {
            AttributeValue::Int64(_) => true,
            AttributeValue::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => false,
    };

    let output_rank = if dim_only { tensor.rank } else { 1 };
    log::debug!("ReduceMin output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: output_rank,
        static_shape: None,
    });
}

/// Update output rank for ReduceProd based on axes.
fn reduce_prod_update_outputs(node: &mut Node) {
    log::debug!("ReduceProd rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ReduceProd: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };
    log::debug!("ReduceProd input rank for {}: {}", node.name, tensor.rank);

    let dim_only = match node.attrs.get("axes") {
        Some(value) => match &value {
            AttributeValue::Int64(_) => true,
            AttributeValue::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => false,
    };

    let output_rank = if dim_only { tensor.rank } else { 1 };
    log::debug!("ReduceProd output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: output_rank,
        static_shape: None,
    });
}

/// Update output rank for ReduceSum based on axes.
fn reduce_sum_update_outputs(node: &mut Node) {
    log::debug!("ReduceSum rank inference for node {}", node.name);

    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };
    log::debug!("ReduceSum input rank for {}: {}", node.name, tensor.rank);

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

    let output_rank = if dim_only { tensor.rank } else { 1 };
    log::debug!("ReduceSum output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: output_rank,
        static_shape: None,
    });
}

/// Update output rank for Where to max input rank.
fn where_update_outputs(node: &mut Node) {
    log::debug!("Where rank inference for node {}", node.name);

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

    log::debug!(
        "Where input ranks for {}: condition={}, x={}, y={}",
        node.name,
        condition.rank(),
        x.rank(),
        y.rank()
    );

    let output_rank = max(condition.rank(), max(x.rank(), y.rank()));
    log::debug!("Where output rank for {}: {}", node.name, output_rank);

    if output_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(elem_type);
        log::debug!("Where result for {} is scalar", node.name);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: output_rank,
            static_shape: None,
        });
        log::debug!(
            "Where result for {} is tensor with rank {}",
            node.name,
            output_rank
        );
    }
}

/// Update output rank for Gather based on input and indices ranks.
fn gather_update_outputs(node: &mut Node) {
    log::debug!("Gather rank inference for node {}", node.name);

    if node.inputs.len() != 2 {
        panic!("Gather requires two inputs: data and indices");
    }

    let indices_rank = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0,
        _ => panic!("Only tensor indices is valid, got {:?}", node.inputs[1].ty),
    };
    log::debug!("Gather indices rank for {}: {}", node.name, indices_rank);

    match &node.inputs[0].ty {
        ArgType::Tensor(input_tensor) => {
            log::debug!(
                "Gather input tensor rank for {}: {}",
                node.name,
                input_tensor.rank
            );
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_rank + input_tensor.rank - 1;
            log::debug!("Gather output rank for {}: {}", node.name, output_rank);

            if output_rank == 0 {
                node.outputs[0].ty = ArgType::Scalar(input_tensor.elem_type.clone());
                log::debug!("Gather result for {} is scalar", node.name);
            } else {
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: input_tensor.elem_type.clone(),
                    rank: output_rank,
                    static_shape: None,
                });
                log::debug!(
                    "Gather result for {} is tensor with rank {}",
                    node.name,
                    output_rank
                );
            }
        }
        ArgType::Shape(_) => {
            log::debug!("Gather input is shape for {}", node.name);
            let shape_rank = 1;
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_rank + shape_rank - 1;
            log::debug!(
                "Gather output rank for {} with shape input: {}",
                node.name,
                output_rank
            );

            if output_rank == 0 {
                node.outputs[0].ty = ArgType::Scalar(ElementType::Int64);
                log::debug!("Gather result for {} is scalar (from shape)", node.name);
            } else {
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: output_rank,
                    static_shape: None,
                });
                log::debug!(
                    "Gather result for {} is tensor with rank {} (from shape)",
                    node.name,
                    output_rank
                );
            }
        }
        ty => panic!("Only tensor/shape input is valid but received: {:?}", ty),
    }
}

/// Update output rank for Split (same as input).
fn split_update_outputs(node: &mut Node) {
    log::debug!("Split rank inference for node {}", node.name);

    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Split: Input must be a tensor"),
    };
    log::debug!("Split input rank for {}: {}", node.name, tensor.rank);
    log::debug!(
        "Split will generate {} outputs for {}",
        node.outputs.len(),
        node.name
    );

    for (i, output_arg) in node.outputs.iter_mut().enumerate() {
        output_arg.ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
        log::debug!("Split output {} rank for {}: {}", i, node.name, tensor.rank);
    }
}

/// Update output rank for OneHot (input rank + 1).
fn one_hot_output_shape(node: &mut Node) {
    log::debug!("OneHot rank inference for node {}", node.name);

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("OneHot: invalid input type"),
    };
    log::debug!("OneHot input rank for {}: {}", node.name, input_rank);

    let output_rank = input_rank + 1;
    log::debug!("OneHot output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.outputs[0].ty.elem_type().clone(),
        rank: output_rank,
        static_shape: None,
    });
}

fn gemm_output_shape(node: &mut Node) {
    log::debug!("Gemm rank inference for node {}", node.name);

    let a_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Input A should be a tensor!"),
    };
    let b_rank = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Input B should be a tensor!"),
    };

    log::debug!(
        "Gemm input ranks for {}: a_rank={}, b_rank={}",
        node.name,
        a_rank,
        b_rank
    );

    let output_rank = max(a_rank, b_rank);
    log::debug!("Gemm output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape: None,
        elem_type: match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.elem_type.clone(),
            _ => panic!("Unexpected type for input A"),
        },
    });
}

#[derive(Debug)]
pub enum RankInferenceError {
    MismatchedRanks,
    NonTensorInput,
    EmptyInputs,
}

pub fn concat_update_outputs_safe(node: &mut Node) -> Result<(), RankInferenceError> {
    if node.inputs.is_empty() {
        return Err(RankInferenceError::EmptyInputs);
    }

    let first_rank = match node.inputs[0].ty {
        ArgType::Tensor(ref tensor) => tensor.rank,
        _ => return Err(RankInferenceError::NonTensorInput),
    };

    // Check that all inputs are tensors with the same rank
    for input in &node.inputs {
        match input.ty {
            ArgType::Tensor(ref tensor) if tensor.rank == first_rank => {}
            ArgType::Tensor(_) => return Err(RankInferenceError::MismatchedRanks),
            _ => return Err(RankInferenceError::NonTensorInput),
        }
    }

    // Update output rank
    if let ArgType::Tensor(ref mut tensor) = node.outputs[0].ty {
        tensor.rank = first_rank;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, TensorType};

    #[test]
    fn test_concat_rank_inference() {
        // Create a test node with valid tensor inputs
        let node = Node {
            name: "test_concat".to_string(),
            node_type: NodeType::Concat,
            inputs: vec![
                Argument {
                    name: "input1".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 4,
                        static_shape: None,
                    }),
                    value: None,
                    passed: false,
                },
                Argument {
                    name: "input2".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 4,
                        static_shape: None,
                    }),
                    value: None,
                    passed: false,
                },
            ],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 0,
                    static_shape: None,
                }),
                value: None,
                passed: false,
            }],
            attrs: std::collections::HashMap::new(),
        };

        // Test successful case
        let mut success_node = node.clone();
        assert!(concat_update_outputs_safe(&mut success_node).is_ok());
        assert_eq!(
            success_node.outputs[0].ty,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 4,
                static_shape: None,
            })
        );

        // Test mismatched ranks
        let mut node_mismatched = node.clone();
        if let ArgType::Tensor(ref mut tensor) = node_mismatched.inputs[1].ty {
            tensor.rank = 3;
        }
        assert!(matches!(
            concat_update_outputs_safe(&mut node_mismatched),
            Err(RankInferenceError::MismatchedRanks)
        ));

        // Test non-tensor input
        let mut node_non_tensor = node.clone();
        node_non_tensor.inputs[1].ty = ArgType::Scalar(ElementType::Float32);
        assert!(matches!(
            concat_update_outputs_safe(&mut node_non_tensor),
            Err(RankInferenceError::NonTensorInput)
        ));

        // Test empty inputs
        let mut node_empty = node.clone();
        node_empty.inputs.clear();
        assert!(matches!(
            concat_update_outputs_safe(&mut node_empty),
            Err(RankInferenceError::EmptyInputs)
        ));
    }
}
