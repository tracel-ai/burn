//! # ConstantOfShape
//!
//! Generates a tensor with a given shape filled with a constant value.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html>
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version with shape input and optional value attribute.
//! - **Opset 20**: Added support for bfloat16, int4, uint4, and float8 value types.

use crate::ir::{
    ArgType, DType, Node, NodeBuilder, NodeConfig, RuntimeInputRef, TensorData, TensorDataExt,
    TensorType,
};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Configuration for the ConstantOfShape operation.
#[derive(Debug, Clone, Default)]
pub struct ConstantOfShapeConfig {
    /// Shape information (static or runtime).
    pub shape: ConstantOfShapeShape,
    /// The fill value. If None, defaults to 0.0f32.
    pub value: Option<TensorData>,
}

/// Shape information for the ConstantOfShape operation.
#[derive(Debug, Clone)]
pub enum ConstantOfShapeShape {
    /// Static shape information known at compile time.
    Static(Vec<i64>),
    /// Runtime shape that will be determined during execution .
    Runtime(RuntimeInputRef),
}

impl Default for ConstantOfShapeShape {
    fn default() -> Self {
        Self::Static(vec![])
    }
}

impl NodeConfig for ConstantOfShapeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ConstantOfShapeProcessor;

impl NodeProcessor for ConstantOfShapeProcessor {
    type Config = ConstantOfShapeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // Only lift shape input (input[0]) if it has a static value
        // Runtime shapes should remain in the graph
        if !node.inputs.is_empty() && node.inputs[0].is_constant() {
            node.inputs[0].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate input type
        match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                // For tensor inputs representing shapes, the rank should be 1
                if tensor.rank != 1 {
                    return Err(ProcessError::Custom(
                        "ConstantOfShape: shape tensor must be 1D".to_string(),
                    ));
                }
                if !matches!(tensor.dtype, DType::I64) {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Int64".to_string(),
                        actual: format!("{:?}", tensor.dtype),
                    });
                }
            }
            ArgType::Shape(_) => {
                // Shapes are always 1-D int64 data, so nothing to validate here
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        }

        // TODO: According to spec, the 'value' attribute is a one-element tensor
        // FIXME: Need to validate that it contains exactly one element - currently not checked
        // TODO: Add test for multi-element value tensor (should error)
        // TODO: Add test for negative shape values - spec says "all values must be >= 0"
        // TODO: Add test for very large shape dimensions - potential memory/overflow issues
        // TODO: Add test for opset 20+ types (bfloat16, int4, uint4, float8) - mentioned in spec
        let _config = self
            .extract_config(node, _opset)
            .expect("Config extraction failed");

        let value_type = node
            .attrs
            .get("value")
            .map(|v| v.clone().into_tensor().elem_type())
            .unwrap_or(DType::F32); // If not given, defaults to 0 as float32

        let rank = match &node.inputs[0].ty {
            ArgType::Shape(rank) => *rank,
            ArgType::Tensor(tensor_type) => {
                // First check if we have a lifted constant value (most reliable)
                if let Some(tensor_data) = node.inputs[0].value() {
                    // The tensor data contains the shape values
                    // For a shape tensor, the length of the data is the output rank
                    match tensor_data.to_i64_vec() {
                        Ok(shape_vec) => shape_vec.len(),
                        Err(_) => {
                            return Err(ProcessError::Custom(format!(
                                "ConstantOfShape node {} requires Int32 or Int64 shape input",
                                node.name
                            )));
                        }
                    }
                } else if let Some(shape) = &tensor_type.static_shape {
                    // Fall back to static shape if no constant value
                    shape.first().copied().ok_or_else(|| {
                        ProcessError::Custom(
                            "ConstantOfShape node must have a non-empty static shape value"
                                .to_string(),
                        )
                    })?
                } else {
                    return Err(ProcessError::Custom(format!(
                        "ConstantOfShape node {} must have either a constant value or a static shape",
                        node.name
                    )));
                }
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Update the input type to be a shape
        node.inputs[0].ty = ArgType::Shape(rank);

        // Optimization: When input is Shape(1) and value type is Int64,
        // output Shape(1) directly instead of a tensor. This is a common pattern
        // in ONNX models where ConstantOfShape is used to create shape arrays.
        // Downstream operations can cast to tensor if needed.
        // This optimization improves performance by keeping shape operations in the Shape domain.
        if rank == 1 && value_type == DType::I64 {
            // Special optimization for Shape(1) with Int64 values
            node.outputs[0].ty = ArgType::Shape(1);
        } else if rank == 0 {
            // When rank is 0, output should be a scalar
            node.outputs[0].ty = ArgType::Scalar(value_type);
        } else {
            // General case: output is a tensor
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: value_type,
                rank,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Check if we have static values or need runtime resolution
        let shape = match node.inputs[0].value() {
            Some(tensor_data) => match tensor_data.to_i64_vec() {
                Ok(shape) => ConstantOfShapeShape::Static(shape),
                Err(_) => {
                    return Err(ProcessError::Custom(format!(
                        "ConstantOfShape node {} requires Int32 or Int64 shape data",
                        node.name
                    )));
                }
            },
            None => {
                // Runtime input - store reference instead of cloning the argument
                ConstantOfShapeShape::Runtime(RuntimeInputRef::new(node.inputs[0].name.clone(), 0))
            }
        };

        // Extract the value attribute if present
        let value = node.attrs.get("value").map(|v| v.clone().into_tensor());

        let config = ConstantOfShapeConfig { shape, value };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::ConstantOfShape {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, NodeType, TensorData};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(input_ty: ArgType) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::ConstantOfShape, "test_constantofshape")
            .add_input("shape", input_ty)
            .output_tensor_f32("output", 0, None) // Will be updated
    }

    #[test]
    fn test_shape_input() {
        let mut node = create_test_node(ArgType::Shape(3)).build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_tensor_input_with_static_shape() {
        let mut node = create_test_node(ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 1,
            static_shape: Some(vec![4]),
        }))
        .build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_custom_value_type() {
        let mut node = create_test_node(ArgType::Shape(2)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData::new(vec![7i64], vec![])),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_invalid_input_type() {
        let mut node = create_test_node(ArgType::Scalar(DType::F32)).build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_no_static_shapes_with_value_attr() {
        // Simulates the scenario after constant lifting where the input has a value

        let mut node = TestNodeBuilder::new(NodeType::ConstantOfShape, "constantofshape1")
            .input_tensor_i64_data("constant180_out1", vec![2, 3, 4], vec![3])
            .output_default("/model/encoder/patch_encoder/ConstantOfShape_output_0")
            .attr_tensor("value", TensorData::new(vec![1i64], vec![1]))
            .build_with_graph_data(16);

        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Verify the output has the correct rank
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 3); // Output rank should be 3
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_scalar_output_with_shape_0() {
        // Test when input is Shape(0), output should be Scalar
        let mut node = create_test_node(ArgType::Shape(0)).build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::F32);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_scalar_output_with_tensor_shape_0() {
        // Test when input is a tensor with static shape [0], output should be Scalar
        let mut node = create_test_node(ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 1,
            static_shape: Some(vec![0]), // Shape is [0], meaning rank-0 output
        }))
        .build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::F32);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_scalar_output_with_custom_value() {
        // Test scalar output with custom value type
        let mut node = create_test_node(ArgType::Shape(0)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData::new(vec![42i64], vec![])),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::I64);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_shape_optimization_with_int64() {
        // Test Shape(1) -> Shape(1) optimization when value type is Int64
        let mut node = create_test_node(ArgType::Shape(1)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData::new(vec![5i64], vec![])),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1);
            }
            _ => panic!("Expected Shape(1) output for Shape(1) input with Int64 value"),
        }
    }

    #[test]
    fn test_shape_1_with_float_no_optimization() {
        // Test that Shape(1) with Float32 does NOT get optimized (outputs Tensor)
        let mut node = create_test_node(ArgType::Shape(1)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData::new(vec![1.5f32], vec![])),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected Tensor output for Shape(1) input with Float32 value"),
        }
    }

    #[test]
    fn test_shape_1_default_value_no_optimization() {
        // Test that Shape(1) with default value (Float32) does NOT get optimized
        let mut node = create_test_node(ArgType::Shape(1)).build();
        // No value attribute means default Float32
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected Tensor output for Shape(1) input with default Float32 value"),
        }
    }
}
