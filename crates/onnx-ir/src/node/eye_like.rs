//! # EyeLike
//!
//! Generates a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__EyeLike.html>
//!
//! ## Attributes
//! - `dtype` (int, optional): Data type for output. Must be a valid data type from TensorProto.
//!   If not specified, the type of the input tensor is used.
//! - `k` (int, default=0): Index of the diagonal to populate with ones.
//!   - `k=0`: Main diagonal (default)
//!   - `k>0`: Upper diagonal (k positions above main)
//!   - `k<0`: Lower diagonal (k positions below main)
//!
//! ## Inputs
//! - `input` (T1): 2D input tensor used only for shape reference. Must be rank 2.
//!
//! ## Outputs
//! - `output` (T2): 2D output tensor with the same shape as input, containing ones on diagonal k
//!   and zeros everywhere else.
//!
//! ## Opset Versions
//! - **Opset 9+**: Initial version with dtype and k attributes

use crate::ir::{ArgType, DType, Node, NodeConfig, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::proto_conversion::element_type_from_proto;

use std::any::Any;

/// Configuration for EyeLike operations
#[derive(Debug, Clone, new)]
pub struct EyeLikeConfig {
    /// Data type of the output tensor (optional, defaults to input type)
    pub dtype: Option<DType>,
    /// Diagonal offset (0 = main diagonal, >0 = upper, <0 = lower)
    pub k: i64,
}

impl NodeConfig for EyeLikeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct EyeLikeProcessor;

impl NodeProcessor for EyeLikeProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Extract tensor info and validate
        let (input_rank, input_elem_type, input_static_shape) = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                if tensor.rank != 2 {
                    return Err(ProcessError::Custom(
                        "EyeLike operation requires 2D tensor input".to_string(),
                    ));
                }
                (tensor.rank, tensor.dtype, tensor.static_shape.clone())
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Get reference to config for type inference
        let config = node.config::<EyeLikeConfig>();

        // Output type is either specified dtype or input type
        let output_type = config.dtype.unwrap_or(input_elem_type);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: output_type,
            rank: input_rank,
            static_shape: input_static_shape,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut dtype = None;
        let mut k = 0i64; // default to main diagonal

        // Extract attributes
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "dtype" => {
                    let dtype_i32 = value.clone().into_i32();
                    dtype = Some(element_type_from_proto(dtype_i32).map_err(|e| {
                        ProcessError::InvalidAttribute {
                            name: "dtype".to_string(),
                            reason: format!("Unsupported dtype for EyeLike: {}", e),
                        }
                    })?);
                }
                "k" => {
                    k = value.clone().into_i64();
                }
                // TODO: Add validation for unexpected attributes (currently silently ignored)
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for EyeLike: {}", key),
                    });
                }
            }
        }

        let config = EyeLikeConfig { dtype, k };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;
    use protobuf::Enum;

    #[test]
    fn test_eye_like_update_output() {
        let mut node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .output_tensor_f32("output", 2, None) // rank will be updated
            .build();

        let processor = EyeLikeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![3, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_eye_like_config_default() {
        let node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![4, 4]))
            .output_tensor_f32("output", 2, None)
            .build();

        let mut node = node;

        let processor = EyeLikeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<EyeLikeConfig>();
        assert_eq!(config.k, 0);
        assert_eq!(config.dtype, None);
    }

    #[test]
    fn test_eye_like_config_with_attributes() {
        let node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![4, 4]))
            .output_tensor_f32("output", 2, None)
            .attr_int("k", -1)
            .attr_int("dtype", DataType::INT64.value() as i64)
            .build();

        let mut node = node;

        let processor = EyeLikeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<EyeLikeConfig>();
        assert_eq!(config.k, -1);
        assert_eq!(config.dtype, Some(DType::I64));
    }

    #[test]
    fn test_eye_like_update_output_with_dtype() {
        let mut node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .output_tensor_f32("output", 2, None)
            .attr_int("dtype", DataType::INT32.value() as i64)
            .build();

        let processor = EyeLikeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I32);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![3, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    // TODO: Add test for non-2D input - Should return error for rank != 2 per spec - Missing constraint validation test
    // TODO: Add test for non-square matrices - Test rectangular matrices (e.g., 3x5, 5x3) - Missing edge case test
    // TODO: Add test for 1x1 matrix - Edge case with single element - Test exists in onnx-tests but not in unit tests
    // TODO: Add test for large k values - When k is larger than matrix dimensions, should produce all zeros - Test exists in onnx-tests but not in unit tests
    // TODO: Add test for large negative k values - When |k| is larger than matrix dimensions, should produce all zeros - Test exists in onnx-tests but not in unit tests
    // TODO: Add test for different output dtypes - Spec supports many types, test more than just I32 and I64 - Missing type coverage
    // TODO: Add test for unexpected attributes - Should reject unknown attributes per implementation - Missing attribute validation test
    // TODO: Add test for opset < 9 - Should fail per spec, EyeLike introduced in opset 9 - Missing opset validation test
}
