//! Unsqueeze operation implementation for ONNX graphs.
//!
//! This module handles the unsqueeze operation which adds dimensions of size 1 to tensors.
//! It includes an important optimization for Int scalar to Shape conversion, which is the
//! reverse of the squeeze operation and critical for efficient dynamic shape handling in
//! ONNX models.

use crate::{
    Argument, TensorData,
    ir::{ArgType, Data, Node, TensorType},
};

/// Update output rank for Unsqueeze based on axes.
/// Update the output tensor dimension based on the "axes" attribute or the second input
pub fn unsqueeze_update_output(node: &mut Node) {
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
        node.attrs.get("axes").cloned().map(|v| {
            let axes = v.into_i64s();
            log::debug!(
                "Unsqueeze axes from attribute for {}: {:?}",
                node.name,
                axes
            );
            axes
        })
    };

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => {
            0 // treat scalar as 0-dim tensor
        }
        _ => panic!("Unsqueeze: invalid input type"),
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

    // Determine the output type based on input type and output rank.
    //
    // Special case: Int scalar -> Shape[1] conversion
    // ================================================
    // When an Int scalar is unsqueezed to rank 1, we produce a Shape type instead of a Tensor.
    // This is the reverse operation of squeeze(Shape[1]) -> Scalar.
    //
    // Why this optimization matters:
    // 1. Performance: Many reshape operations in ONNX models follow this pattern where shape
    //    information flows through squeeze/unsqueeze operations. Keeping them as Shape types
    //    avoids unnecessary tensor allocations.
    //
    // 2. Memory efficiency: Both scalars and Shape types are stored on CPU, so this conversion
    //    is essentially free - no device transfers are needed.
    //
    // 3. Type consistency: This maintains type symmetry with the squeeze operation, allowing
    //    shape information to flow naturally through the graph.
    //
    // 4. Flexibility: If a downstream operation requires a GPU tensor, the Shape can be
    //    converted to a Tensor at that point. This lazy conversion strategy ensures we only
    //    pay the cost when necessary.
    //
    // This optimization is particularly important for dynamic shape scenarios where reshape
    // operations compute their output shapes at runtime using these squeeze/unsqueeze patterns.
    match &node.inputs[0].ty {
        ArgType::Scalar(elem_type) if output_rank == 1 => {
            match elem_type {
                crate::ir::ElementType::Int32 | crate::ir::ElementType::Int64 => {
                    // Unsqueeze Int scalar to Shape[1] (reverse of squeeze operation)
                    node.outputs[0].ty = ArgType::Shape(1);
                }
                _ => {
                    // Other scalar types unsqueeze to tensor
                    node.outputs[0].ty = ArgType::Tensor(TensorType {
                        rank: output_rank,
                        static_shape: None,
                        elem_type: elem_type.clone(),
                    });
                }
            }
        }
        _ => {
            // Regular tensor or scalar to tensor conversion
            let output_elem = match &node.outputs[0].ty {
                ArgType::Tensor(_) => node.inputs[0].ty.elem_type().clone(),
                ArgType::Scalar(elem_type) => elem_type.clone(),
                ArgType::Shape(_) => crate::ir::ElementType::Int64, // Shape elements are always i64
            };

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                rank: output_rank,
                static_shape: None, // shape is tracked and calculated at runtime
                elem_type: output_elem,
            });
        }
    }

    log::debug!("Unsqueeze output rank for {}: {}", node.name, output_rank);
}

/// Axes specification for the Unsqueeze operation.
#[derive(Debug, Clone)]
pub enum UnsqueezeConfig {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes that will be determined during execution.
    Runtime(Argument),
}

/// Creates UnsqueezeAxes configuration from the node attributes.
///
/// Note: This function should only execute if the second input is a constant.
/// If it wasn't and the output shape was known, unsqueeze has been remapped to reshape.
pub fn unsqueeze_config(node: &Node) -> UnsqueezeConfig {
    // Check if axes attribute exists
    for (key, value) in node.attrs.iter() {
        if key.as_str() == "axes" {
            return UnsqueezeConfig::Static(value.clone().into_i64s());
        }
    }

    assert!(
        !node.inputs.is_empty(),
        "Unsqueeze: axes tensor must be present"
    );

    let input_value = &node.inputs[1];

    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.rank, 1, "Unsqueeze: axes tensor must be 1D");
            if let Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) = input_value.value.as_ref()
            {
                UnsqueezeConfig::Static(shape.clone())
            } else {
                UnsqueezeConfig::Runtime(node.inputs[1].clone())
            }
        }
        _ => panic!("Arg for unsqueeze must be tensor or scalar"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    // Implement custom equality for UnsqueezeConfig to make testing easier
    impl PartialEq<UnsqueezeConfig> for UnsqueezeConfig {
        fn eq(&self, other: &UnsqueezeConfig) -> bool {
            match (self, other) {
                (UnsqueezeConfig::Static(a), UnsqueezeConfig::Static(b)) => a == b,
                (UnsqueezeConfig::Runtime(a), UnsqueezeConfig::Runtime(b)) => a.name == b.name,
                _ => false,
            }
        }
    }

    fn create_test_node_with_attr(input_rank: usize, axes: Vec<i64>) -> Node {
        let builder = NodeBuilder::new(NodeType::Unsqueeze, "test_unsqueeze")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Y", 0, None) // Will be updated
            .attr_ints("axes", axes);

        builder.build()
    }

    fn create_test_node_with_input(input_rank: usize, axes: Vec<i64>, with_value: bool) -> Node {
        let axes_len = axes.len();
        let mut builder = NodeBuilder::new(NodeType::Unsqueeze, "test_unsqueeze")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Y", 0, None); // Will be updated

        // Add axes input with or without value
        if with_value {
            builder = builder.input_tensor_i64_data("axes", axes.clone(), vec![axes_len]);
        } else {
            // Input without value
            builder = builder.input_tensor_i64("axes", 1, Some(vec![axes_len]));
        }

        builder.build()
    }

    // Tests for unsqueeze_update_output function

    #[test]
    fn test_unsqueeze_with_attr() {
        let mut node = create_test_node_with_attr(2, vec![0, 3]);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4); // 2 + 2 = 4
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_with_input() {
        let mut node = create_test_node_with_input(3, vec![1, 2, 4], true);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 6); // 3 + 3 = 6
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar_float() {
        let mut node = create_test_node_with_attr(0, vec![0]);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1); // 0 + 1 = 1
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar_int_to_shape() {
        let mut node = create_test_node_with_attr(0, vec![0]);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Int64);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1); // Scalar unsqueezed to Shape[1]
            }
            _ => panic!("Expected Shape output for Int scalar unsqueeze"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar_int32_to_shape() {
        let mut node = create_test_node_with_attr(0, vec![0]);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Int32);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1); // Scalar unsqueezed to Shape[1]
            }
            _ => panic!("Expected Shape output for Int32 scalar unsqueeze"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar_int_multiple_axes() {
        // Test that Int scalar with multiple axes produces a tensor, not shape
        let mut node = create_test_node_with_attr(0, vec![0, 1]);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Int64);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2); // 0 + 2 = 2
            }
            _ => panic!("Expected tensor output for multi-axis unsqueeze"),
        }
    }

    #[test]
    #[should_panic(expected = "Unsqueeze: invalid input type")]
    fn test_unsqueeze_invalid_input() {
        let mut node = create_test_node_with_attr(2, vec![0]);
        node.inputs[0].ty = ArgType::Shape(1);
        unsqueeze_update_output(&mut node);
    }

    // Tests for unsqueeze_config function

    #[test]
    fn test_unsqueeze_config_with_attr() {
        // Test with axes provided as attribute
        let axes = vec![0, 2, 4];
        let node = create_test_node_with_attr(3, axes.clone());

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_with_static_input() {
        // Test with axes provided as input tensor with static value
        let axes = vec![1, 3];
        let node = create_test_node_with_input(2, axes.clone(), true);

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_with_runtime_input() {
        // Test with axes provided as input tensor but without static value
        let axes = vec![0, 2];
        let node = create_test_node_with_input(2, axes.clone(), false);

        let config = unsqueeze_config(&node);

        // Should return a Runtime config since the axes are only known at runtime
        match config {
            UnsqueezeConfig::Static(_) => panic!("Expected Runtime config"),
            UnsqueezeConfig::Runtime(arg) => {
                assert_eq!(arg.name, "axes");
            }
        }
    }

    #[test]
    fn test_unsqueeze_config_negative_axes() {
        // Test with negative axes (should be handled by the caller)
        let axes = vec![-1, -3];
        let node = create_test_node_with_attr(3, axes.clone());

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_empty_axes() {
        // Test with empty axes array (edge case)
        let axes = vec![];
        let node = create_test_node_with_attr(2, axes.clone());

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_unsqueeze_config_missing_axes() {
        // Test with neither axes attribute nor input
        let mut node = create_test_node_with_attr(2, vec![0]);
        node.attrs.clear(); // Remove the axes attribute
        node.inputs = vec![node.inputs[0].clone()]; // Remove the axes input

        let _ = unsqueeze_config(&node);
    }

    #[test]
    #[should_panic(expected = "Unsqueeze: axes tensor must be 1D")]
    fn test_unsqueeze_config_invalid_axes_rank() {
        // Test with axes tensor that is not 1D
        let mut node = create_test_node_with_input(2, vec![0, 1], true);
        if let ArgType::Tensor(ref mut tensor) = node.inputs[1].ty {
            tensor.rank = 2; // Invalid rank for axes
        }

        let _ = unsqueeze_config(&node);
    }

    #[test]
    #[should_panic(expected = "Arg for unsqueeze must be tensor or scalar")]
    fn test_unsqueeze_config_invalid_axes_type() {
        // Test with axes input that is not a tensor
        let mut node = create_test_node_with_input(2, vec![0], false);
        node.inputs[1].ty = ArgType::Shape(1); // Invalid type for axes

        let _ = unsqueeze_config(&node);
    }
}
