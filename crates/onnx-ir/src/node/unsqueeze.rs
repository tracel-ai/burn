//! # Unsqueeze
//!
//! Inserts single-dimensional entries (dimensions of size 1) at specified positions in the tensor's shape.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Unsqueeze.html>
//!
//! ## Attributes
//!
//! None in opset 13+. In earlier versions (opset 11 and below), `axes` was an attribute.
//!
//! ## Inputs
//!
//! - `data` (T): Original tensor
//! - `axes` (tensor(int64)): List of integers indicating the dimensions to be inserted.
//!   Negative values count dimensions from the back. Accepted range is [-r, r-1] where r = rank(expanded).
//!   The order of values in axes does not matter and can come in any order.
//!
//! ## Outputs
//!
//! - `expanded` (T): Reshaped tensor with same data as input, with dimensions of size 1 inserted at specified positions
//!
//! ## Opset Versions
//!
//! - **Opset 13+**: `axes` is a required input (allows dynamic specification)
//! - **Opset 11**: `axes` was an attribute (fixed at graph construction time)
//!
//! The change from attribute to input in opset 13 provides greater flexibility,
//! enabling dynamic axes specification at runtime.
//!
//! ## Special Optimizations
//!
//! This module includes an important optimization for Int scalar to Shape conversion, which is the
//! reverse of the squeeze operation and critical for efficient dynamic shape handling in ONNX models.

use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::{
    TensorData,
    ir::{ArgType, Data, Node, NodeConfig, RuntimeInputRef, TensorType},
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
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift axes input (input[1]) if present
        // Note: axes can also be an attribute, but we only lift the input version
        // FIXME: The spec states that axes is a required input in opset 13+, but the
        // extract_config method allows for axes as an attribute (backward compatibility).
        // The opset validation and axes requirement should be more strictly enforced.
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::processor::validate_opset(opset, 13)?;

        // Validate we have at least one input
        crate::processor::validate_min_inputs(node, 1)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

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
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Check if axes attribute exists
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axes" {
                let config = UnsqueezeConfig::Static(value.clone().into_i64s());
                return Ok(Some(Box::new(config)));
            }
        }

        // Axes must be provided as second input
        if node.inputs.len() < 2 {
            return Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: node.inputs.len(),
            });
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

                if let Some(TensorData {
                    data: Data::Int64s(shape),
                    shape: data_shape,
                    ..
                }) = input_value.value().as_ref()
                {
                    // Validate actual tensor data shape
                    if data_shape.len() != 1 {
                        return Err(ProcessError::Custom(
                            "Unsqueeze: axes tensor must be 1D".to_string(),
                        ));
                    }
                    UnsqueezeConfig::Static(shape.clone())
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
    fn infer_with_axes(&self, node: &mut Node, axes: Option<Vec<i64>>) -> Result<(), ProcessError> {
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
                crate::ir::ElementType::Int32 | crate::ir::ElementType::Int64 => {
                    node.outputs[0].ty = ArgType::Shape(1);
                }
                _ => {
                    node.outputs[0].ty = ArgType::Tensor(TensorType {
                        rank: output_rank,
                        static_shape: None,
                        elem_type: elem_type.clone(),
                    });
                }
            },
            _ => {
                let output_elem = match &node.outputs[0].ty {
                    ArgType::Tensor(_) => node.inputs[0].ty.elem_type().clone(),
                    ArgType::Scalar(elem_type) => elem_type.clone(),
                    ArgType::Shape(_) => crate::ir::ElementType::Int64,
                };

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    rank: output_rank,
                    static_shape: None,
                    elem_type: output_elem,
                });
            }
        }

        Ok(())
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
                (UnsqueezeConfig::Runtime(a), UnsqueezeConfig::Runtime(b)) => a == b,
                _ => false,
            }
        }
    }

    fn create_test_node_with_attr(input_rank: usize, axes: Vec<i64>) -> NodeBuilder {
        NodeBuilder::new(NodeType::Unsqueeze, "test_unsqueeze")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Y", 0, None) // Will be updated
            .attr_ints("axes", axes)
    }

    fn create_test_node_with_input(
        input_rank: usize,
        axes: Vec<i64>,
        with_value: bool,
    ) -> NodeBuilder {
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

        builder
    }

    // Tests for unsqueeze_update_output function

    #[test]
    fn test_unsqueeze_with_attr() {
        let mut node = create_test_node_with_attr(2, vec![0, 3]).build();
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        let mut node =
            create_test_node_with_input(3, vec![1, 2, 4], true).build_with_graph_data(16);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        let mut node = create_test_node_with_attr(0, vec![0]).build();
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        let mut node = create_test_node_with_attr(0, vec![0]).build();
        node.inputs[0].ty = ArgType::Scalar(ElementType::Int64);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        node.inputs[0].ty = ArgType::Scalar(ElementType::Int32);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        node.inputs[0].ty = ArgType::Scalar(ElementType::Int64);
        let processor = UnsqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let result = processor.extract_config(&node, 16);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));
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
}
