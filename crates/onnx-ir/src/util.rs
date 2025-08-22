use crate::ir::{ArgType, Node, TensorType};

use crate::protos::OperatorSetIdProto;

pub fn shape_config(curr: &Node) -> (usize, usize) {
    if curr.inputs.len() != 1 {
        panic!(
            "Shape: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the rank of the input (works for both Tensor and Shape types)
    let input_rank = match &curr.inputs.first().unwrap().ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Shape(size) => *size,
        _ => panic!("Shape operation requires Tensor or Shape input"),
    };

    // Default: all axes up to the last one (included)
    let mut start_dim: i64 = 0;
    let mut end_dim: i64 = input_rank as i64;

    // Extract the attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "start" => start_dim = value.clone().into_i64(),
            "end" => end_dim = value.clone().into_i64(),
            _ => {}
        }
    }

    // If dim is negative, it is counted from the end
    if start_dim < 0 {
        start_dim += input_rank as i64;
    }
    if end_dim < 0 {
        end_dim += input_rank as i64;
    }

    (start_dim as usize, end_dim as usize)
}

/// Check whether the provided operator set version is supported.
///
/// # Arguments
///
/// * `opset` - The operator set to check
/// * `min_version` - The minimum supported version
///
/// # Returns
///
/// * `bool` - True if the opset version is supported, false otherwise
///
/// # Panics
///
/// * If the domain is not supported
pub fn check_opset_version(opset: &OperatorSetIdProto, min_version: i64) -> bool {
    match opset.domain.as_str() {
        // Standard ONNX operators
        "" => opset.version >= min_version,
        // ONNX ML operators - commonly used for traditional ML operators
        "ai.onnx.ml" => opset.version >= 1, // ML operators are generally stable from version 1
        // Add support for other domains as needed
        _ => {
            panic!(
                "Unsupported ONNX domain: '{}'. Only standard ONNX ('') and ML ('ai.onnx.ml') domains are supported",
                opset.domain
            );
        }
    }
}

/// Verify that all operator sets in a model are supported.
///
/// # Arguments
///
/// * `opsets` - The operator sets to check
/// * `min_version` - The minimum supported version
///
/// # Returns
///
/// * `bool` - True if all opset versions are supported, false otherwise
pub fn verify_opsets(opsets: &[OperatorSetIdProto], min_version: i64) -> bool {
    for opset in opsets {
        if !check_opset_version(opset, min_version) {
            return false;
        }
    }
    true
}

/// Preserve input rank for operations like Relu, Sigmoid, etc.
pub fn same_as_input(node: &mut Node) {
    log::debug!("Copying input type to output for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("Input rank for {}: {}", node.name, tensor.rank);
    } else if let ArgType::Scalar(_) = &node.inputs[0].ty {
        log::debug!("Input is scalar for {}", node.name);
    }

    node.outputs[0].ty = node.inputs[0].ty.clone();
    log::debug!("Output type is same as input for {}", node.name);
}

/// Update output rank for broadcasting operations (e.g., Add, Sub) to max input rank.
pub fn same_as_input_broadcast(node: &mut Node) {
    log::debug!("Broadcasting operation for node {}", node.name);

    // Check if any input is a Shape type
    let has_shape_input = node
        .inputs
        .iter()
        .any(|input| matches!(&input.ty, ArgType::Shape(_)));

    if has_shape_input {
        // If any input is a Shape, find the first Shape input and use its rank for the output
        let shape_rank = node
            .inputs
            .iter()
            .find_map(|input| match &input.ty {
                ArgType::Shape(rank) => Some(*rank),
                _ => None,
            })
            .expect("Shape input must exist");

        log::debug!(
            "Shape input detected for node {}, output will be Shape with rank {}",
            node.name,
            shape_rank
        );
        node.outputs[0].ty = ArgType::Shape(shape_rank);
        return;
    }

    let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => acc.max(tensor.rank),
        ArgType::Scalar(_) => acc,
        ArgType::Shape(_) => unreachable!("Shape case handled above"),
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
        });
        log::debug!(
            "Tensor result for node {} with rank {}",
            node.name,
            max_rank
        );
    }
}

/// Temporary stub preserves input type for unhandled operations.
pub fn temporary_pass_through_stub(node: &mut Node) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, ElementType, NodeType};
    use std::collections::HashMap;

    fn create_test_node(op_type: NodeType, input_ranks: Vec<usize>) -> Node {
        let mut inputs = Vec::new();

        for (i, rank) in input_ranks.iter().enumerate() {
            inputs.push(Argument {
                name: format!("input_{i}"),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: *rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            });
        }

        let outputs = vec![Argument {
            name: "output".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        Node {
            node_type: op_type.clone(),
            name: format!("test_{op_type:?}").to_lowercase(),
            inputs,
            outputs,
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn test_same_as_input() {
        let mut node = create_test_node(NodeType::Relu, vec![3]);
        same_as_input(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_same_as_input_broadcast_max_rank() {
        let mut node = create_test_node(NodeType::Add, vec![2, 4, 3]);
        same_as_input_broadcast(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4); // max(2, 4, 3) = 4
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_same_as_input_broadcast_with_scalar() {
        let mut node = create_test_node(NodeType::Add, vec![3]);
        // Add a scalar input
        node.inputs.push(Argument {
            name: "scalar_input".to_string(),
            ty: ArgType::Scalar(ElementType::Float32),
            value: None,
            passed: true,
        });

        same_as_input_broadcast(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3); // Scalar doesn't affect rank
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_temporary_pass_through_stub() {
        let mut node = create_test_node(NodeType::Identity, vec![5]);
        temporary_pass_through_stub(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 5);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_same_as_input_broadcast_with_shape() {
        let mut node = create_test_node(NodeType::Add, vec![3]);
        // Add a Shape input
        node.inputs.push(Argument {
            name: "shape_input".to_string(),
            ty: ArgType::Shape(3),
            value: None,
            passed: true,
        });

        same_as_input_broadcast(&mut node);

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 3);
            }
            _ => panic!("Expected shape output when one input is Shape"),
        }
    }

    #[test]
    fn test_same_as_input_broadcast_shape_and_scalar() {
        let mut node = Node {
            node_type: NodeType::Mul,
            name: "test_mul".to_string(),
            inputs: vec![
                Argument {
                    name: "shape_input".to_string(),
                    ty: ArgType::Shape(4),
                    value: None,
                    passed: true,
                },
                Argument {
                    name: "scalar_input".to_string(),
                    ty: ArgType::Scalar(ElementType::Int64),
                    value: None,
                    passed: true,
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
                passed: true,
            }],
            attrs: HashMap::new(),
        };

        same_as_input_broadcast(&mut node);

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 4);
            }
            _ => panic!("Expected shape output when one input is Shape"),
        }
    }
}
