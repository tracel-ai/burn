//! # Reduce Operations (ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceSumSquare)
//!
//! Reduction operations that compute aggregates along specified axes of a tensor. These operations
//! reduce the input tensor by applying an aggregation function (sum, mean, max, min, product, or
//! sum of squares) along the specified axes.
//!
//! **ONNX Specs**:
//! - ReduceSum: <https://onnx.ai/onnx/operators/onnx__ReduceSum.html>
//! - ReduceMean: <https://onnx.ai/onnx/operators/onnx__ReduceMean.html>
//! - ReduceMax: <https://onnx.ai/onnx/operators/onnx__ReduceMax.html>
//! - ReduceMin: <https://onnx.ai/onnx/operators/onnx__ReduceMin.html>
//! - ReduceProd: <https://onnx.ai/onnx/operators/onnx__ReduceProd.html>
//! - ReduceSumSquare: <https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html>
//!
//! ## Common Attributes
//! - `axes` (ints, optional): List of integers indicating the dimensions to reduce. If not specified,
//!   all dimensions are reduced. For opset >= 18, axes can be provided as an optional input instead.
//! - `keepdims` (int, default=1): Whether to keep the reduced dimensions (with size 1) in the output.
//!   If set to 0, the reduced dimensions are removed from the output shape.
//!
//! ## Inputs
//! - `data` (T): Input tensor to be reduced.
//! - `axes` (optional, int64): Axes along which to reduce (opset >= 18).
//!
//! ## Outputs
//! - `reduced` (T): Reduced output tensor. If all dimensions are reduced and keepdims=0, the output
//!   will be a scalar.
//!
//! ## Opset Versions
//! - **Opset 1-10**: Earlier versions with different attribute handling
//! - **Opset 11-12**: Standardized behavior with axes attribute, added noop_with_empty_axes
//! - **Opset 13-17**: Extended type support (bfloat16, uint/int types)
//! - **Opset 18+**: Axes moved from attribute to optional input tensor for dynamic shapes
//!
//! ## Type Constraints
//! - T: tensor(float16), tensor(float32), tensor(float64), tensor(int32), tensor(int64)

use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::{ArgType, Node, NodeConfig, TensorType};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct ReduceConfig {
    pub dims: Vec<usize>,
    pub keepdims: bool,
}

impl NodeConfig for ReduceConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

impl ReduceConfig {
    pub fn new(dims: Vec<usize>, keepdims: bool) -> Self {
        Self { dims, keepdims }
    }
}

pub struct ReduceProcessor;

impl NodeProcessor for ReduceProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift axes input (input[1]) if present
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
        // Opset validation
        crate::processor::validate_opset(opset, 11)?;

        // Validate input count
        crate::processor::validate_min_inputs(node, 1)?;

        // Validate input type and extract tensor info
        let (tensor_rank, tensor_elem_type, tensor_static_shape) = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => (tensor.rank, tensor.dtype, tensor.static_shape.clone()),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Get config values before using them
        let dims = node.config::<ReduceConfig>().dims.clone();
        let keepdims = if node.config::<ReduceConfig>().keepdims {
            1
        } else {
            0
        };

        // Determine if the output should be a scalar
        let should_be_scalar = keepdims == 0 && (dims.is_empty() || dims.len() == tensor_rank);

        if should_be_scalar {
            // Output is a scalar
            node.outputs[0].ty = ArgType::Scalar(tensor_elem_type);
        } else {
            // Output is a tensor
            let output_rank = if keepdims == 1 {
                tensor_rank
            } else {
                tensor_rank - dims.len()
            };

            // Infer static shape based if given
            let output_shape = tensor_static_shape.and_then(|mut shape| {
                // Only process static shape if it's complete (matches tensor rank)
                if shape.len() != tensor_rank {
                    return None;
                }

                if keepdims == 1 {
                    for dim in &dims {
                        shape[*dim] = 1;
                    }
                    Some(shape)
                } else {
                    for dim in dims.iter().rev() {
                        shape.remove(*dim);
                    }
                    Some(shape)
                }
            });

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: tensor_elem_type,
                rank: output_rank,
                static_shape: output_shape,
            });
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Validate input type and extract tensor info
        let tensor_rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Extract axes and keepdims attributes
        let mut axes = Vec::new();
        let mut keepdims = 1;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axes" => axes = value.clone().into_i64s(),
                "keepdims" => keepdims = value.clone().into_i64(),
                _ => {}
            }
        }

        // Process axes from additional input (if available)
        if let Some(value) = node.inputs.get(1).and_then(|argument| argument.value()) {
            axes = value.to_vec::<i64>().unwrap();
        }

        let mut dims: Vec<usize> = axes
            .into_iter()
            .map(|mut dim| {
                if dim < 0 {
                    // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
                    dim += tensor_rank as i64;
                }
                dim as usize
            })
            .collect();

        // Sort the dimensions to ensure consistent order
        dims.sort();

        let config = ReduceConfig::new(dims, keepdims == 1);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::bool_assert_comparison)]

    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axes: Option<Vec<i64>>, keepdims: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::ReduceMax, "test_reduce_max")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("reduced", 3, None);

        if let Some(axes_val) = axes {
            builder = builder.attr_ints("axes", axes_val);
        }
        if let Some(kd) = keepdims {
            builder = builder.attr_int("keepdims", kd);
        }

        builder.build()
    }

    #[test]
    fn test_reduce_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [1]);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [1]); // -2 + 3 = 1
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_axes() {
        let node = create_test_node(None, Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, Vec::<usize>::new());
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [0, 1]);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0));
        let mut node = node;

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [1]);
        assert_eq!(config.keepdims, false);
    }

    #[test]
    fn test_reduce_update_outputs_scalar_no_axes_no_keepdims() {
        // Test that reduce with no axes and keepdims=false produces a scalar output
        let mut node = create_test_node(None, Some(0));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(_) => {
                // This is the expected case - scalar output
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_scalar_all_dims_no_keepdims() {
        // Test that reduce with all dimensions and keepdims=false produces a scalar output
        let mut node = create_test_node(Some(vec![0, 1, 2]), Some(0));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(_) => {
                // This is the expected case - scalar output
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_partial_dims_no_keepdims() {
        // Test that reduce with partial dimensions and keepdims=false produces a tensor output
        let mut node = create_test_node(Some(vec![1]), Some(0));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should be rank 2 (3 - 1 = 2)
                assert_eq!(tensor.rank, 2);
            }
            ArgType::Scalar(_) => {
                panic!("Expected tensor output but got scalar");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_with_keepdims() {
        // Test that reduce with keepdims=true always produces a tensor output
        let mut node = create_test_node(None, Some(1));
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain original rank when keepdims=true
                assert_eq!(tensor.rank, 3);
            }
            ArgType::Scalar(_) => {
                panic!("Expected tensor output but got scalar when keepdims=true");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_partial_static_shape_keepdims() {
        // Regression test for partial static_shape with keepdims=true
        // This was causing "index out of bounds" panic before the fix
        let mut node = NodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![768])) // Rank 3 but only last dim known
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![2]) // Reduce on dimension 2
            .attr_int("keepdims", 1)
            .build();

        // This should not panic
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain rank 3 with keepdims=true
                assert_eq!(tensor.rank, 3);
                // Static shape should be None since input shape was partial
                assert_eq!(tensor.static_shape, None);
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_partial_static_shape_no_keepdims() {
        // Regression test for partial static_shape without keepdims
        let mut node = NodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![768])) // Rank 3 but only last dim known
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![1]) // Reduce on dimension 1
            .attr_int("keepdims", 0)
            .build();

        // This should not panic
        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should be rank 2 (3 - 1 = 2) without keepdims
                assert_eq!(tensor.rank, 2);
                // Static shape should be None since input shape was partial
                assert_eq!(tensor.static_shape, None);
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_complete_static_shape_keepdims() {
        // Test that complete static_shape is properly updated with keepdims=true
        let mut node = NodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![2, 4, 768])) // Complete shape
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![2]) // Reduce on dimension 2
            .attr_int("keepdims", 1)
            .build();

        let processor = ReduceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain rank 3 with keepdims=true
                assert_eq!(tensor.rank, 3);
                // Static shape should be updated: [2, 4, 768] -> [2, 4, 1]
                assert_eq!(tensor.static_shape, Some(vec![2, 4, 1]));
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }
}
