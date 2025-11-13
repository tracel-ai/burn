//! # Expand
//!
//! Broadcasts input tensor to a target shape using numpy-style broadcasting.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Expand.html>
//!
//! ## Opset Versions
//! - **Opset 8**: Initial version (replaces deprecated Tile for broadcasting)
//! - **Opset 13**: Extended type support (bfloat16)

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::{
    DType,
    ir::{ArgType, Node, NodeBuilder, NodeConfig, RuntimeInputRef, TensorDataExt, TensorType},
};
use std::any::Any;

/// Shape information for the Expand operation.
#[derive(Debug, Clone)]
pub enum ExpandShape {
    /// Static shape information known at compile time.
    Static(Vec<i64>),
    /// Runtime shape determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for ExpandShape {
    fn default() -> Self {
        ExpandShape::Static(Vec::new())
    }
}

impl NodeConfig for ExpandShape {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ExpandProcessor;

impl NodeProcessor for ExpandProcessor {
    type Config = ExpandShape;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 8,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // Only lift shape input (input[1]) if it has a static value
        // Runtime shapes should remain in the graph
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate no unexpected attributes - Expand has no attributes per spec - Missing attribute validation

        // Validate shape input type
        match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => {
                if tensor.rank != 1 {
                    return Err(ProcessError::Custom(
                        "Expand: shape tensor must be 1D".to_string(),
                    ));
                }
                if !matches!(tensor.dtype, DType::I64) {
                    return Err(ProcessError::Custom(
                        "Expand: shape tensor must have element type int64".to_string(),
                    ));
                }
            }
            ArgType::Shape(_) => {
                // Shapes are always 1-D int64 data, so nothing to validate here
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        }

        // Get reference to config for type inference
        let config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");

        // Get input element type - Expand should preserve the input's element type
        let input_elem_type = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.dtype,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Determine output type based on config
        match config {
            ExpandShape::Static(shape) => {
                // TODO: Validate shape values are positive or -1 per ONNX spec - Negative values other than -1 are invalid - Missing constraint validation
                // TODO: Validate broadcasting rules - Per spec, input shape and target shape must be compatible for broadcasting - Missing broadcast validation
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: input_elem_type,
                    rank: shape.len(),
                    static_shape: Some(shape.iter().map(|&dim| dim as usize).collect()),
                });
            }
            ExpandShape::Runtime(_) => {
                // When the shape cannot be determined statically, infer the rank from the shape input
                let output_rank = match &node.inputs[1].ty {
                    ArgType::Shape(rank) => *rank,
                    ArgType::Tensor(tensor) => {
                        if let Some(static_shape) = &tensor.static_shape {
                            static_shape[0]
                        } else {
                            // Check if output already has a rank set from ONNX
                            match &node.outputs[0].ty {
                                ArgType::Tensor(TensorType { rank, .. }) if *rank > 0 => *rank,
                                _ => {
                                    return Err(ProcessError::Custom(format!(
                                        "Cannot determine output rank for Expand node {} with fully dynamic shape tensor",
                                        node.name
                                    )));
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Tensor or Shape".to_string(),
                            actual: format!("{:?}", node.inputs[1].ty),
                        });
                    }
                };

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: input_elem_type,
                    rank: output_rank,
                    static_shape: None,
                });
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract config
        let config = match node.inputs[1].value() {
            Some(tensor_data) => match tensor_data.to_i64_vec() {
                Ok(shape) => ExpandShape::Static(shape),
                Err(_) => {
                    return Err(ProcessError::Custom(
                        "Expand: shape data type must be int32 or int64".to_string(),
                    ));
                }
            },
            None => {
                // Runtime shape - store reference instead of cloning the argument
                ExpandShape::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
            }
        };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let _config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Expand {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(
        input_rank: usize,
        shape_value: Option<Vec<i64>>,
        shape_type: Option<ArgType>,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_f32("input", input_rank, None)
            .output_tensor_f32("output", 0, None); // Rank 0 will be updated

        if let Some(shape) = shape_value {
            builder = builder.input_tensor_i64_data("shape", shape.clone(), vec![shape.len()]);
        } else if let Some(st) = shape_type {
            // Use the provided custom shape type
            builder = builder.add_input("shape", st);
        } else {
            // Default case with dynamic shape
            builder = builder.input_tensor_i64("shape", 1, Some(vec![3]));
        }

        builder
    }

    #[test]
    fn test_expand_with_constant_shape() {
        let mut node = create_test_node(2, Some(vec![2, 3, 4]), None).build_with_graph_data(16);

        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, Some(vec![2, 3, 4]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_with_dynamic_shape() {
        let mut node = create_test_node(2, None, None).build();

        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, None);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_with_incorrect_inputs() {
        let mut node = create_test_node(2, Some(vec![2, 3, 4]), None).build_with_graph_data(16);
        // Remove one input to make it invalid
        node.inputs.pop();

        let processor = ExpandProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));
    }

    // Tests for expand_config function

    #[test]
    fn test_expand_config_with_static_shape() {
        let node = create_test_node(2, Some(vec![2, 3, 4]), None).build_with_graph_data(16);
        let mut node = node;
        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match config {
            ExpandShape::Static(shape) => {
                assert_eq!(*shape, vec![2, 3, 4]);
            }
            ExpandShape::Runtime(_) => panic!("Expected Static config, got Runtime"),
        }
    }

    #[test]
    fn test_expand_config_with_runtime_shape() {
        let node = create_test_node(2, None, None).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match config {
            ExpandShape::Static(_) => panic!("Expected Runtime config, got Static"),
            ExpandShape::Runtime(name) => {
                assert_eq!(name.name, "shape");
            }
        }
    }

    #[test]
    fn test_expand_config_with_shape_type() {
        let shape_type = ArgType::Shape(3);
        let node = create_test_node(2, None, Some(shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match config {
            ExpandShape::Static(_) => panic!("Expected Runtime config, got Static"),
            ExpandShape::Runtime(name) => {
                assert_eq!(name.name, "shape");
            }
        }
    }

    #[test]
    fn test_expand_config_with_invalid_shape_rank() {
        let invalid_shape_type = ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 2, // Invalid rank, should be 1
            static_shape: None,
        });
        let node = create_test_node(2, None, Some(invalid_shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_expand_config_with_invalid_shape_type() {
        let invalid_shape_type = ArgType::Tensor(TensorType {
            dtype: DType::F32, // Invalid element type, should be Int64
            rank: 1,
            static_shape: None,
        });
        let node = create_test_node(2, None, Some(invalid_shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_expand_config_with_invalid_input_type() {
        let invalid_shape_type = ArgType::Scalar(DType::I64);
        let node = create_test_node(2, None, Some(invalid_shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_expand_config_with_invalid_value_type() {
        // Create a node with shape input that has Float32 type instead of Int64
        let node = TestNodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_f32("input", 2, None)
            .input_tensor_f32_data("shape", vec![2.0, 3.0, 4.0], vec![3]) // Wrong type - Float32 instead of Int64
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(16);

        let node = node;
        let processor = ExpandProcessor;
        let result = processor.extract_config(&node, 16);
        match result {
            Err(ProcessError::Custom(_)) => {}
            _ => panic!("Expected ProcessError::Custom for invalid shape data type"),
        }
    }

    #[test]
    fn test_expand_update_outputs_with_shape_input() {
        // Test Expand with Shape type as shape input
        let mut node = create_test_node(2, None, Some(ArgType::Shape(4))).build();

        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4); // Shape(4) means output will be rank 4
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_update_outputs_with_shape_input_static_value() {
        // Test Expand with shape input that has static values
        let mut node = TestNodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_f32("input", 2, None)
            .input_tensor_i64_data("shape", vec![5, 10, 15], vec![3]) // Static shape values
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(16);

        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, Some(vec![5, 10, 15]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_preserves_input_element_type() {
        // Test that Expand preserves the input element type for different types

        // Test Float32 -> Float32
        {
            let mut node = TestNodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_f32("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_f32("output", 0, None)
                .build_with_graph_data(16);

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: DType::I64, // Wrong type
                rank: 0,
                static_shape: None,
            });

            let processor = ExpandProcessor;
            let prefs = OutputPreferences::new();
            let _config = processor.extract_config(&node, 16).unwrap();
            processor.infer_types(&mut node, 16, &prefs).unwrap();

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.dtype,
                        DType::F32,
                        "Expand should preserve Float32 input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }

        // Test Int64 -> Int64
        {
            let mut node = TestNodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_i64("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_i64("output", 0, None)
                .build_with_graph_data(16);

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: DType::F32, // Wrong type
                rank: 0,
                static_shape: None,
            });

            let processor = ExpandProcessor;
            let prefs = OutputPreferences::new();
            let _config = processor.extract_config(&node, 16).unwrap();
            processor.infer_types(&mut node, 16, &prefs).unwrap();

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.dtype,
                        DType::I64,
                        "Expand should preserve Int64 input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }

        // Test Bool -> Bool
        {
            let mut node = TestNodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_bool("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_bool("output", 0, None)
                .build_with_graph_data(16);

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: DType::F32, // Wrong type
                rank: 0,
                static_shape: None,
            });

            let processor = ExpandProcessor;
            let prefs = OutputPreferences::new();
            let _config = processor.extract_config(&node, 16).unwrap();
            processor.infer_types(&mut node, 16, &prefs).unwrap();

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.dtype,
                        DType::Bool,
                        "Expand should preserve Bool input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }
    }

    #[test]
    fn test_expand_with_mismatched_output_type() {
        // Test that Expand corrects output type even when initially set incorrectly
        // This simulates the case where ONNX might have wrong type info
        let mut node = TestNodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_i64("input", 2, None) // Input is Int64
            .input_tensor_i64_data("shape", vec![2, 3], vec![2])
            .output_tensor_f32("output", 0, None) // Output incorrectly set to Float32
            .build_with_graph_data(16);

        let processor = ExpandProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(
                    tensor.dtype,
                    DType::I64,
                    "Expand should use input type (Int64) not initial output type (Float32)"
                );
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![2, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    // TODO: Add test for invalid shape values - Test negative values other than -1 (e.g., -2, -3) should return error - Missing constraint validation test
    // TODO: Add test for shape with value -1 - Per spec, -1 means copy from input dimension - Missing edge case test
    // TODO: Add test for incompatible broadcasting - Test case where input shape cannot be broadcast to target shape - Missing broadcast validation test
    // TODO: Add test for zero in target shape - Test behavior when target shape contains 0 - Missing edge case test
    // TODO: Add test for expanding scalar to tensor - Test input with rank 0 expanded to higher rank - Missing edge case test
    // TODO: Add test for different data types - Spec supports many types (all numeric types, bool, strings) - Only testing f32, i64, bool
    // TODO: Add test for opset < 8 - Should fail per spec, Expand introduced in opset 8 - Missing opset validation test
    // TODO: Add test for unexpected attributes - Should validate and reject unknown attributes - Missing attribute validation test
}
