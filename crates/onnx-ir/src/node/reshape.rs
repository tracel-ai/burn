use crate::ir::{ArgType, Argument, Data, Node, TensorData, TensorType};

/// Configuration for the Reshape operation.
#[derive(Debug, Clone)]
pub struct ReshapeConfig {
    pub shape: ReshapeInput,
}

/// Represents either a static value or a runtime argument for reshape shape.
#[derive(Debug, Clone)]
pub enum ReshapeInput {
    /// Static shape known at compile time.
    Static(Vec<i64>),
    /// Runtime shape determined during execution.
    Runtime(Argument),
}

/// Update output rank for Reshape based on shape input if constant, otherwise use input rank.
pub fn reshape_update_outputs(node: &mut Node) {
    let input_tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Reshape: invalid input types"),
    };

    // Try to determine output rank from various sources
    let output_rank = infer_reshape_output_rank(node);

    // Get static shape if available from output
    let static_shape = match &node.outputs[0].ty {
        ArgType::Tensor(t) => t.static_shape.clone(),
        _ => None,
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape,
        elem_type: input_tensor.elem_type.clone(),
    });
}

/// Infer output rank for reshape operation from available information
fn infer_reshape_output_rank(node: &Node) -> usize {
    // Case 1: Static shape from constant shape input or attribute
    if let Some(shape) = get_static_shape(node) {
        return shape.len();
    }

    // Case 2: Dynamic shape - try to infer from shape input type
    if node.inputs.len() == 2 {
        match &node.inputs[1].ty {
            ArgType::Tensor(shape_tensor) => {
                if let Some(dims) = &shape_tensor.static_shape
                    && !dims.is_empty()
                {
                    return dims[0];
                }
            }
            ArgType::Shape(rank) => {
                // Shape type directly gives us the rank
                return *rank;
            }
            _ => {}
        }
    }

    // Case 3: Use output's static_shape if available
    if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty
        && let Some(shape) = &output_tensor.static_shape
    {
        return shape.len();
    }

    // Case 4: No rank information available - this is an error
    panic!(
        "Reshape node {} has dynamic shape with no rank information available. \
         Cannot determine output rank.",
        node.name
    )
}

/// Extract static shape from reshape node if available
fn get_static_shape(node: &Node) -> Option<Vec<i64>> {
    // Check shape input
    if node.inputs.len() == 2
        && let Some(value) = &node.inputs[1].value
        && let Data::Int64s(shape) = &value.data
    {
        return Some(shape.clone());
    }

    None
}

/// Creates a configuration for reshape operation based on the ONNX Reshape operator.
/// Returns either static shape or runtime argument for reshape.
pub fn reshape_config(node: &Node) -> ReshapeConfig {
    let allowzero = node
        .attrs
        .get("allowzero")
        .map(|v| v.clone().into_i64())
        .unwrap_or(0);

    // Check the allowzero attribute
    // (see https://onnx.ai/onnx/operators/onnx__Reshape.html#attributes)
    if allowzero != 0 {
        panic!("Zero shape size is not supported");
    }

    if node.inputs.len() != 2 {
        panic!("Reshape requires exactly 2 inputs");
    }

    let shape = match &node.inputs[1].ty {
        ArgType::Tensor(_) => {
            match &node.inputs[1].value {
                Some(TensorData { data, shape, .. }) => {
                    assert_eq!(shape.len(), 1, "Reshape: shape tensor must be 1D");
                    ReshapeInput::Static(data.clone().into_i64s())
                }
                None => {
                    // Runtime shape input from tensor
                    ReshapeInput::Runtime(node.inputs[1].clone())
                }
            }
        }
        ArgType::Shape(_) => {
            // Runtime shape input from Shape node
            ReshapeInput::Runtime(node.inputs[1].clone())
        }
        _ => panic!("Reshape: second input must be either a Tensor or Shape type"),
    };

    ReshapeConfig { shape }
}

/// Legacy function that returns shape as Vec<i64> - kept for backward compatibility
pub fn reshape_config_vec(node: &Node) -> Vec<i64> {
    let config = reshape_config(node);
    match config.shape {
        ReshapeInput::Static(shape) => shape,
        ReshapeInput::Runtime(_) => {
            panic!("reshape_config_vec cannot be used with runtime shape inputs")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementType;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(allowzero: i64, shape_vec: Vec<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 4, None)
            .input_tensor_i64_data("shape", shape_vec.clone(), vec![shape_vec.len()])
            .output_tensor_f32("reshaped", 2, None);

        if allowzero != 0 {
            builder = builder.attr_int("allowzero", allowzero);
        }

        builder.build()
    }

    fn create_runtime_reshape_node() -> Node {
        NodeBuilder::new(NodeType::Reshape, "test_runtime_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("shape", 0, None) // No static value - runtime input
            .output_tensor_f32("reshaped", 2, None)
            .build()
    }

    fn create_reshape_with_shape_input() -> Node {
        NodeBuilder::new(NodeType::Reshape, "test_reshape_with_shape")
            .input_tensor_f32("data", 4, None)
            .add_input("shape", ArgType::Shape(2))
            .output_tensor_f32("reshaped", 2, None)
            .build()
    }

    #[test]
    fn test_reshape_config_basic() {
        let node = create_test_node(0, vec![2, 3]);
        let config = reshape_config(&node);
        match config.shape {
            ReshapeInput::Static(shape) => assert_eq!(shape, vec![2, 3]),
            _ => panic!("Expected static shape"),
        }
    }

    #[test]
    #[should_panic(expected = "Zero shape size is not supported")]
    fn test_reshape_config_allowzero_not_supported() {
        let node = create_test_node(1, vec![2, 3]);
        let _ = reshape_config(&node);
    }

    #[test]
    fn test_reshape_config_runtime() {
        let node = create_runtime_reshape_node();
        let config = reshape_config(&node);
        match config.shape {
            ReshapeInput::Runtime(arg) => assert_eq!(arg.name, "shape"),
            _ => panic!("Expected runtime shape"),
        }
    }

    #[test]
    #[should_panic(expected = "Reshape requires exactly 2 inputs")]
    fn test_reshape_config_no_shape_input() {
        let mut node = create_test_node(0, vec![2, 3]);
        node.inputs.pop(); // Remove the shape input
        let _ = reshape_config(&node);
    }

    #[test]
    #[should_panic(expected = "shape tensor must be 1D")]
    fn test_reshape_config_invalid_shape_dim() {
        let mut node = create_test_node(0, vec![2, 3]);
        // Modify the shape tensor's shape to be 2D
        if let Some(tensor_data) = &mut node.inputs[1].value {
            tensor_data.shape = vec![2, 1];
        }
        let _ = reshape_config(&node);
    }

    #[test]
    fn test_reshape_update_outputs_basic() {
        let mut node = create_test_node(0, vec![2, 3]);

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_update_outputs_int() {
        let mut node = create_test_node(0, vec![2, 3]);
        node.inputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int32,
            rank: 4,
            static_shape: None,
        });

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.elem_type, ElementType::Int32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_config_with_shape_type() {
        let node = create_reshape_with_shape_input();
        let config = reshape_config(&node);
        match config.shape {
            ReshapeInput::Runtime(arg) => assert_eq!(arg.name, "shape"),
            _ => panic!("Expected runtime shape"),
        }
    }

    #[test]
    fn test_reshape_update_outputs_with_shape_type() {
        let mut node = create_reshape_with_shape_input();

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2); // Should get rank from Shape(2) input
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
