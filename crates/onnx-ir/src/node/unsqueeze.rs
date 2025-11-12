//! # Unsqueeze
//!
//! Inserts single-dimensional entries (dimensions of size 1) at specified positions in the tensor's shape.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Unsqueeze.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with required 'axes' attribute.
//! - **Opset 11**: Clarified semantics and behavior for negative axis values.
//! - **Opset 13**: Changed 'axes' from attribute to required input, enabling dynamic axes specification at runtime.
//!
//! **Implementation Note**: This implementation requires opset 13+ (axes as input). The change from attribute to input provides greater flexibility for dynamic shape operations.
//!
//! TODO: Axes range validation not implemented - ONNX spec requires axes values in [-r, r-1] range where r = rank(expanded), but extract_config and infer_types do not validate this constraint - Missing validation in extract_config after line 151
//!
//! TODO: Missing duplicate axes validation - ONNX spec states axes order doesn't matter but doesn't allow duplicates, implementation doesn't check for duplicate values in axes - Should validate uniqueness after to_i64_vec
//!
//! TODO: Missing test coverage for negative axes - Tests exist for positive axes but no test validates negative axis values work correctly per opset 11+ spec - Need test case with negative axes like [-1, -3]
//!
//! TODO: Missing test coverage for zero-size tensor - No test validates unsqueeze behavior with zero-size input tensor (e.g., shape [0, 3]) - Should add test case
//!
//! TODO: Missing test coverage for duplicate axes error case - No test verifies that duplicate axes are rejected - Need negative test case
//!
//! TODO: Missing test coverage for out-of-range axes - No test validates axes range checking per spec [-r, r-1] - Need negative test cases
//!
//! ## Special Optimizations
//!
//! This module includes an important optimization for Int scalar to Shape conversion, which is the
//! reverse of the squeeze operation and critical for efficient dynamic shape handling in ONNX models.

use crate::ir::{ArgType, NodeBuilder, NodeConfig, RuntimeInputRef, TensorDataExt, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Axes specification for the Unsqueeze operation.
#[derive(Debug, Clone)]
pub enum UnsqueezeConfig {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl NodeConfig for UnsqueezeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct UnsqueezeProcessor;

impl NodeProcessor for UnsqueezeProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 13,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, opset: usize) -> Result<(), ProcessError> {
        // Lift axes input (input[1]) if present
        // In opset 13+, axes is a required input
        // In opset <13, axes is an attribute
        if opset >= 13 && node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get reference to config for type inference
        let config = node.config::<UnsqueezeConfig>();

        // Extract axes for type inference
        let axes = match config {
            UnsqueezeConfig::Static(axes) => Some(axes.clone()),
            UnsqueezeConfig::Runtime(_) => None,
        };

        self.infer_with_axes(node, axes)
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Check if axes attribute exists (only valid in opset <13)
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axes" {
                if opset >= 13 {
                    return Err(ProcessError::Custom(
                        "Unsqueeze: axes must be provided as input (not attribute) in opset 13+"
                            .to_string(),
                    ));
                }
                let config = UnsqueezeConfig::Static(value.clone().into_i64s());
                return Ok(Some(Box::new(config)));
            }
        }

        // In opset 13+, axes must be provided as second input
        // In opset <13, if no axes attribute, axes must be provided as input
        if node.inputs.len() < 2 {
            if opset >= 13 {
                return Err(ProcessError::InvalidInputCount {
                    expected: 2,
                    actual: node.inputs.len(),
                });
            } else {
                return Err(ProcessError::Custom(
                    "Unsqueeze: axes must be provided as either attribute or input".to_string(),
                ));
            }
        }

        let input_value = &node.inputs[1];

        let config = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => {
                // Validate tensor rank if it's non-zero
                // (rank of 0 means not yet inferred, which is OK during initial config extraction)
                if tensor.rank != 0 && tensor.rank != 1 {
                    return Err(ProcessError::Custom(
                        "Unsqueeze: axes tensor must be 1D".to_string(),
                    ));
                }

                if let Some(tensor_data) = input_value.value().as_ref() {
                    // Validate actual tensor data shape
                    if tensor_data.shape.len() != 1 {
                        return Err(ProcessError::Custom(
                            "Unsqueeze: axes tensor must be 1D".to_string(),
                        ));
                    }
                    // TODO: Missing duplicate axes validation - ONNX spec states axes order doesn't matter but doesn't allow duplicates, implementation doesn't check for duplicate values in axes - Should validate uniqueness after to_i64_vec
                    match tensor_data.to_i64_vec() {
                        Ok(axes) => UnsqueezeConfig::Static(axes),
                        Err(_) => {
                            return Err(ProcessError::Custom(
                                "Unsqueeze: axes tensor must be Int32 or Int64".to_string(),
                            ));
                        }
                    }
                } else {
                    // Runtime input - store reference instead of cloning the argument
                    UnsqueezeConfig::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
                }
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        Ok(Some(Box::new(config)))
    }
}

impl UnsqueezeProcessor {
    fn infer_with_axes(
        &self,
        node: &mut NodeBuilder,
        axes: Option<Vec<i64>>,
    ) -> Result<(), ProcessError> {
        let input_rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            ArgType::Scalar(_) => 0,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Scalar".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        let output_rank = if let Some(axes) = axes {
            input_rank + axes.len()
        } else if node.inputs.len() == 2 {
            if let ArgType::Tensor(tensor) = &node.inputs[1].ty {
                if let Some(static_shape) = &tensor.static_shape {
                    input_rank
                        + *static_shape.first().ok_or_else(|| {
                            ProcessError::Custom("Unsqueeze: empty axes shape".to_string())
                        })?
                } else {
                    return Err(ProcessError::Custom(
                        "Unsqueeze: missing static shape for axes".to_string(),
                    ));
                }
            } else {
                return Err(ProcessError::Custom(
                    "Unsqueeze: missing axes information".to_string(),
                ));
            }
        } else {
            return Err(ProcessError::Custom(
                "Unsqueeze: missing axes information".to_string(),
            ));
        };

        // Special case: Int scalar -> Shape[1] conversion (reverse of squeeze)
        match &node.inputs[0].ty {
            ArgType::Scalar(elem_type) if output_rank == 1 => match elem_type {
                crate::ir::DType::I32 | crate::ir::DType::I64 => {
                    node.outputs[0].ty = ArgType::Shape(1);
                }
                _ => {
                    node.outputs[0].ty = ArgType::Tensor(TensorType {
                        rank: output_rank,
                        static_shape: None,
                        dtype: *elem_type,
                    });
                }
            },
            _ => {
                let output_elem = match &node.outputs[0].ty {
                    ArgType::Tensor(_) => node.inputs[0].ty.elem_type(),
                    ArgType::Scalar(elem_type) => *elem_type,
                    ArgType::Shape(_) => crate::ir::DType::I64,
                };

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    rank: output_rank,
                    static_shape: None,
                    dtype: output_elem,
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    // Implement custom equality for UnsqueezeConfig to make testing easier
    impl PartialEq<UnsqueezeConfig> for UnsqueezeConfig {
        fn eq(&self, other: &UnsqueezeConfig) -> bool {
            match (self, other) {
                (UnsqueezeConfig::Static(a), UnsqueezeConfig::Static(b)) => a == b,
                (UnsqueezeConfig::Runtime(a), UnsqueezeConfig::Runtime(b)) => a == b,
                _ => false,
            }
        }
    }

    fn create_test_node_with_attr(input_rank: usize, axes: Vec<i64>) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Unsqueeze, "test_unsqueeze")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Y", 0, None) // Will be updated
            .attr_ints("axes", axes)
    }

    fn create_test_node_with_input(
        input_rank: usize,
        axes: Vec<i64>,
        with_value: bool,
    ) -> TestNodeBuilder {
        let axes_len = axes.len();
        let mut builder = TestNodeBuilder::new(NodeType::Unsqueeze, "test_unsqueeze")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Y", 0, None); // Will be updated

        // Add axes input with or without value
        if with_value {
            builder = builder.input_tensor_i64_data("axes", axes.clone(), vec![axes_len]);
        } else {
            // Input without value
            builder = builder.input_tensor_i64("axes", 1, Some(vec![axes_len]));
        }

        builder
    }

    // Tests for unsqueeze_update_output function

    #[test]
    fn test_unsqueeze_with_attr() {
        let mut node = create_test_node_with_attr(2, vec![0, 3]).build();
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4); // 2 + 2 = 4
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_with_input() {
        let mut node =
            create_test_node_with_input(3, vec![1, 2, 4], true).build_with_graph_data(16);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 6); // 3 + 3 = 6
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar_float() {
        let mut node = create_test_node_with_attr(0, vec![0]).build();
        node.inputs[0].ty = ArgType::Scalar(DType::F32);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 1); // 0 + 1 = 1
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar_int_to_shape() {
        let mut node = create_test_node_with_attr(0, vec![0]).build();
        node.inputs[0].ty = ArgType::Scalar(DType::I64);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1); // Scalar unsqueezed to Shape[1]
            }
            _ => panic!("Expected Shape output for Int scalar unsqueeze"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar_int32_to_shape() {
        let mut node = create_test_node_with_attr(0, vec![0]).build();
        node.inputs[0].ty = ArgType::Scalar(DType::I32);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();

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
        let mut node = create_test_node_with_attr(0, vec![0, 1]).build();
        node.inputs[0].ty = ArgType::Scalar(DType::I64);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 2); // 0 + 2 = 2
            }
            _ => panic!("Expected tensor output for multi-axis unsqueeze"),
        }
    }

    #[test]
    fn test_unsqueeze_invalid_input() {
        let mut node = create_test_node_with_attr(2, vec![0]).build();
        node.inputs[0].ty = ArgType::Shape(1);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 13, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    // Tests for unsqueeze_config function

    #[test]
    fn test_unsqueeze_config_with_attr() {
        let axes = vec![0, 2, 4];
        let node = create_test_node_with_attr(3, axes.clone()).build();

        let mut node = node;
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();
        let config = node.config::<UnsqueezeConfig>();

        assert_eq!(*config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_with_static_input() {
        let axes = vec![1, 3];
        let node = create_test_node_with_input(2, axes.clone(), true).build_with_graph_data(16);

        let mut node = node;
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<UnsqueezeConfig>();

        assert_eq!(*config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_with_runtime_input() {
        let axes = vec![0, 2];
        let node = create_test_node_with_input(2, axes.clone(), false).build();

        let mut node = node;
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<UnsqueezeConfig>();

        match config {
            UnsqueezeConfig::Static(_) => panic!("Expected Runtime config"),
            UnsqueezeConfig::Runtime(name) => {
                assert_eq!(name.name, "axes");
            }
        }
    }

    #[test]
    fn test_unsqueeze_config_negative_axes() {
        let axes = vec![-1, -3];
        let node = create_test_node_with_attr(3, axes.clone()).build();

        let mut node = node;
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();
        let config = node.config::<UnsqueezeConfig>();

        assert_eq!(*config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_empty_axes() {
        let axes = vec![];
        let node = create_test_node_with_attr(2, axes.clone()).build();

        let mut node = node;
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        // Use opset 11 for attribute-based axes (pre-opset 13)
        let config = processor.extract_config(&node, 11).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 13, &prefs).unwrap();
        let config = node.config::<UnsqueezeConfig>();

        assert_eq!(*config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_missing_axes() {
        let mut node = create_test_node_with_attr(2, vec![0]).build();
        node.attrs.clear(); // Remove the axes attribute
        node.inputs = vec![node.inputs[0].clone()]; // Remove the axes input

        let node = node;
        let processor = UnsqueezeProcessor;
        let _prefs = OutputPreferences::new();

        // Test opset 13+ requires axes as input
        let result = processor.extract_config(&node, 13);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));

        // Test opset <13 requires axes as either attribute or input
        let result = processor.extract_config(&node, 11);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_unsqueeze_config_invalid_axes_rank() {
        let mut node = create_test_node_with_input(2, vec![0, 1], true).build_with_graph_data(16);
        if let ArgType::Tensor(ref mut tensor) = node.inputs[1].ty {
            tensor.rank = 2; // Invalid rank for axes
        }

        let node = node;
        let processor = UnsqueezeProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_unsqueeze_config_invalid_axes_type() {
        let mut node = create_test_node_with_input(2, vec![0], false).build();
        node.inputs[1].ty = ArgType::Shape(1); // Invalid type for axes

        let node = node;
        let processor = UnsqueezeProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_unsqueeze_attr_rejected_in_opset_13_plus() {
        // Test that attributes are rejected in opset 13+
        let node = create_test_node_with_attr(2, vec![0]).build();
        let processor = UnsqueezeProcessor;

        let result = processor.extract_config(&node, 13);
        assert!(matches!(result, Err(ProcessError::Custom(_))));

        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
