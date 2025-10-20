//! # Gather
//!
//! Gathers elements from input tensor along a specified axis using indices.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Gather.html>
//!
//! ## Description
//!
//! Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather entries of the
//! axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and
//! concatenates them in an output tensor of rank q + (r - 1).
//!
//! ## Attributes
//!
//! - `axis` (int64, default=0): Which axis to gather on. Negative value means counting
//!   dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
//!
//! ## Inputs
//!
//! - `data` (T): Tensor of rank r >= 1
//! - `indices` (Tind): Tensor of int32/int64 indices, of any rank q. All index values are
//!   expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of
//!   the index values are out of bounds.
//!
//! ## Outputs
//!
//! - `output` (T): Tensor of rank q + (r - 1)
//!
//! ## Type Constraints
//!
//! - T: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double),
//!   tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8),
//!   tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
//! - Tind: tensor(int32), tensor(int64)
//!
//! ## Opset Versions
//!
//! - **Opset 1**: Initial version, basic gather functionality
//! - **Opset 11**: Added support for negative indices; out-of-bounds indices now raise an error
//! - **Opset 13**: Added bfloat16 type support

use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{InputPreferences, NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for the Gather operation.
#[derive(Debug, Clone)]
pub struct GatherConfig {
    pub axis: usize,
}

impl NodeConfig for GatherConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct GatherProcessor;

impl NodeProcessor for GatherProcessor {
    fn input_preferences(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        use crate::processor::ArgPreference;

        if node.inputs.len() < 2 {
            return Ok(None);
        }

        // When gathering from Shape data, prefer indices to be Shape type
        if node.inputs[0].ty.is_shape() {
            Ok(Some(
                InputPreferences::new().add(&node.inputs[1].name, ArgPreference::Shape),
            ))
        } else {
            Ok(None)
        }
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::processor::validate_opset(opset, 11)?;

        // Validate input count
        crate::processor::validate_input_count(node, 2)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // Extract the input rank for axis normalization
        let input_dim = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank as i64,
            ArgType::Shape(shape_rank) => *shape_rank as i64,
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        // Extract the axis attribute (default: 0 per ONNX spec)
        let mut axis: i64 = 0;
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()
            }
        }

        // Normalize negative axis
        if axis < 0 {
            axis += input_dim;
        }

        // Validate axis is within bounds
        if axis < 0 || axis >= input_dim {
            return Err(ProcessError::InvalidAttribute {
                name: "axis".to_string(),
                reason: format!("axis {} is out of bounds for rank {}", axis, input_dim),
            });
        }

        // Infer output type based on indices rank
        let indices_rank = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            ArgType::Scalar(_) => 0,
            ArgType::Shape(_shape_rank) => {
                // Shape is always a 1D array, but when used as indices for Gather,
                // we treat it as rank 1 for the ONNX gather formula
                1 // Shape indices are always treated as rank 1 for gather
            }
        };

        match &node.inputs[0].ty {
            ArgType::Tensor(input_tensor) => {
                // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
                let output_rank = indices_rank + input_tensor.rank - 1;

                if output_rank == 0 {
                    // Output is scalar when gathering a single element
                    node.outputs[0].ty = ArgType::Scalar(input_tensor.elem_type.clone());
                } else {
                    // Output is tensor
                    node.outputs[0].ty = ArgType::Tensor(TensorType {
                        elem_type: input_tensor.elem_type.clone(),
                        rank: output_rank,
                        static_shape: None,
                    });
                }
            }
            ArgType::Shape(_shape_rank) => {
                // When gathering from a shape:
                // - If indices are scalar (rank 0), output is a scalar (single dimension value)
                // - Otherwise, output is a shape with same dimension as indices
                if indices_rank == 0 {
                    node.outputs[0].ty = ArgType::Scalar(crate::ir::ElementType::Int64);
                } else {
                    // For Shape indices, use the actual shape rank (number of elements)
                    let output_shape_rank = match &node.inputs[1].ty {
                        ArgType::Shape(shape_rank) => *shape_rank,
                        ArgType::Tensor(_) => indices_rank, // For tensor indices, use computed rank
                        _ => indices_rank,
                    };
                    node.outputs[0].ty = ArgType::Shape(output_shape_rank);
                }
            }
            ty => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", ty),
                });
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract the input rank for axis normalization
        let input_dim = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank as i64,
            ArgType::Shape(shape_rank) => *shape_rank as i64,
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        // Extract the axis attribute (default: 0 per ONNX spec)
        let mut axis: i64 = 0;
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()
            }
        }

        // Normalize negative axis
        if axis < 0 {
            axis += input_dim;
        }

        let config = GatherConfig {
            axis: axis as usize,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize, is_shape: bool) -> NodeBuilder {
        // Start building the node with the appropriate input type
        let mut builder = NodeBuilder::new(NodeType::Gather, "test_gather").attr_int("axis", axis);

        if is_shape {
            builder = builder.add_input("data", ArgType::Shape(input_rank));
        } else {
            builder = builder.input_tensor_f32("data", input_rank, None);
        }

        // Add indices and output
        builder
            .input_tensor_i64("indices", 1, None)
            .output_tensor_f32("output", input_rank, None)
    }

    #[test]
    fn test_gather_config_basic() {
        let node = create_test_node(0, 3, false).build();
        let mut node = node;
        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<GatherConfig>();
        assert_eq!(config.axis, 0);
    }

    #[test]
    fn test_gather_config_negative_axis() {
        let node = create_test_node(-2, 3, false).build();
        let mut node = node;
        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<GatherConfig>();
        assert_eq!(config.axis, 1); // -2 + 3 = 1
    }

    #[test]
    fn test_gather_config_shape_input() {
        let node = create_test_node(0, 4, true).build(); // Shape of a 4D tensor
        let mut node = node;
        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<GatherConfig>();
        assert_eq!(config.axis, 0);
    }

    #[test]
    fn test_gather_config_missing_index() {
        let mut node = create_test_node(0, 3, false).build();
        node.inputs.pop(); // Remove the indices input
        let mut node = node;
        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn test_gather_update_outputs_scalar_result() {
        // Test gather with scalar indices on 1D tensor -> scalar output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_scalar_gather")
            .attr_int("axis", 0)
            .input_tensor_f32("data", 1, None)
            .add_input("indices", ArgType::Scalar(crate::ir::ElementType::Int64))
            .output_tensor_f32("output", 1, None)
            .build();

        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output scalar, not tensor
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, crate::ir::ElementType::Float32);
            }
            other => panic!("Expected scalar output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_tensor_result() {
        // Test gather with 1D indices on 2D tensor -> 2D tensor output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_tensor_gather")
            .attr_int("axis", 0)
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("indices", 1, None)
            .output_tensor_f32("output", 2, None)
            .build();

        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output tensor with rank 2 (1 + 2 - 1)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.elem_type, crate::ir::ElementType::Float32);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_shape_indices() {
        // Test gather with Shape indices - this was the bug that caused the original issue
        // Gathering from a shape tensor using shape indices should work correctly
        let mut node = NodeBuilder::new(NodeType::Gather, "test_gather_shape_indices")
            .attr_int("axis", 0)
            .input_shape("data", 3) // Shape input (represents shape of a 3D tensor)
            .add_input("indices", ArgType::Shape(1)) // Shape(1) indices - this was causing the panic
            .output_shape("output", 1) // Output should be Shape(1)
            .build();

        // This should not panic - it was panicking before the fix
        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output Shape(1) since we're gathering from Shape(3) with Shape(1) indices
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1);
            }
            other => panic!("Expected Shape output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_shape_scalar_indices() {
        // Test gather with scalar indices on shape input -> scalar output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_gather_shape_scalar")
            .attr_int("axis", 0)
            .input_shape("data", 2) // Shape input (represents shape of a 2D tensor)
            .add_input("indices", ArgType::Scalar(crate::ir::ElementType::Int64)) // Scalar indices
            .output_tensor_i64("output", 0, None) // Will be updated by gather_update_outputs
            .build();

        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output scalar when gathering from shape with scalar indices
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected scalar output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_shape_with_shape_indices_rank_2() {
        // Test gather from Shape with Shape(2) indices -> Shape(2) output
        // This tests our fix where Shape indices preserve their rank in the output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_gather_shape_shape_2")
            .attr_int("axis", 0)
            .input_shape("data", 4) // Shape input (represents shape of a 4D tensor)
            .add_input("indices", ArgType::Shape(2)) // Shape(2) indices
            .output_shape("output", 1) // Initial output, will be updated
            .build();

        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output Shape(2) since indices are Shape(2)
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 2, "Expected Shape(2) output for Shape(2) indices");
            }
            other => panic!("Expected Shape(2) output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_shape_with_shape_indices_rank_3() {
        // Test gather from Shape with Shape(3) indices -> Shape(3) output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_gather_shape_shape_3")
            .attr_int("axis", 0)
            .input_shape("data", 5) // Shape input (represents shape of a 5D tensor)
            .add_input("indices", ArgType::Shape(3)) // Shape(3) indices
            .output_shape("output", 1) // Initial output, will be updated
            .build();

        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output Shape(3) since indices are Shape(3)
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 3, "Expected Shape(3) output for Shape(3) indices");
            }
            other => panic!("Expected Shape(3) output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_shape_with_tensor_indices() {
        // Test gather from Shape with Tensor indices -> Shape output with computed rank
        let mut node = NodeBuilder::new(NodeType::Gather, "test_gather_shape_tensor")
            .attr_int("axis", 0)
            .input_shape("data", 4) // Shape input
            .input_tensor_i64("indices", 1, None) // 1D tensor indices
            .output_shape("output", 1) // Initial output, will be updated
            .build();

        let processor = GatherProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output Shape(1) for 1D tensor indices (indices_rank = 1)
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1, "Expected Shape(1) output for 1D tensor indices");
            }
            other => panic!("Expected Shape(1) output, got {:?}", other),
        }
    }
}
