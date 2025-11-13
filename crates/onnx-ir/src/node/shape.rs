//! # Shape
//!
//! Extracts the shape of an input tensor as a 1D int64 tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Shape.html>
//!
//! ## Special Features
//! - `start` attribute (int, optional, opset 15+): Starting dimension for partial shape extraction.
//!   If omitted, defaults to 0. Negative values count from the end. Values are clamped to [0, rank].
//! - `end` attribute (int, optional, opset 15+): Ending dimension (exclusive) for partial shape extraction.
//!   If omitted, defaults to rank. Negative values count from the end. Values are clamped to [0, rank].
//!
//! **FIXME**: The spec mentions values should be clamped to [0, rank], but the implementation does
//! not perform clamping. Negative indices are normalized but out-of-bounds positive values are not
//! clamped, which could lead to incorrect results or panics.
//!
//! ## Opset Versions
//! - **Opset 1-14**: Outputs full shape as 1D int64 tensor (no attributes).
//! - **Opset 15**: Added `start` and `end` attributes to enable partial shape extraction,
//!   allowing selection of a slice of dimensions from the input shape.
//! - **Opset 19**: Added support for bfloat16 input data type.
//! - **Opset 21**: Added support for int4, uint4, and float8 input data types.

use crate::ir::{ArgType, Node, NodeBuilder, NodeConfig};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Configuration for the Shape operation.
#[derive(Debug, Clone, Default)]
pub struct ShapeConfig {
    pub start: usize,
    pub end: usize,
}

impl NodeConfig for ShapeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ShapeProcessor;

impl NodeProcessor for ShapeProcessor {
    type Config = ShapeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Determine output dimension based on input type
        let dim = match &node.inputs[0].ty {
            ArgType::Tensor(_) => {
                // Shape of a Tensor: output has (end - start) elements
                let config = self
                    .extract_config(node, opset)
                    .expect("Config extraction failed");
                config.end - config.start
            }
            ArgType::Shape(_) => {
                // Shape of a Shape: output is always a 1-element array containing the length
                1
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Infer output type - Shape always outputs Shape type
        node.outputs[0].ty = ArgType::Shape(dim);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract the rank/dimension count from the input
        let rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            ArgType::Shape(rank) => *rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Extract attributes
        let mut start_dim: i64 = 0;
        let mut end_dim: i64 = rank as i64;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "start" => start_dim = value.clone().into_i64(),
                "end" => end_dim = value.clone().into_i64(),
                _ => {}
            }
        }

        // Handle negative indices
        if start_dim < 0 {
            start_dim += rank as i64;
        }
        if end_dim < 0 {
            end_dim += rank as i64;
        }

        // TODO: Missing clamping to [0, rank] as per ONNX spec - out-of-bounds positive values are not clamped.
        // The spec explicitly states: "Values are clamped to [0, rank]" but implementation only normalizes negative indices.
        // This could lead to panics or incorrect results when start/end exceed rank.
        // Should add: start_dim = start_dim.max(0).min(rank); end_dim = end_dim.max(0).min(rank);

        // Calculate dimensions
        let start = start_dim as usize;
        let end = end_dim as usize;

        let config = ShapeConfig { start, end };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Shape {
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
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(start: Option<i64>, end: Option<i64>, rank: usize) -> NodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Shape, "test_shape")
            .input_tensor_f32("data", rank, None)
            .output_tensor_i64("shape", 1, None);

        if let Some(start_val) = start {
            builder = builder.attr_int("start", start_val);
        }

        if let Some(end_val) = end {
            builder = builder.attr_int("end", end_val);
        }

        builder.build()
    }

    #[test]
    fn test_shape_config_defaults() {
        let mut node = create_test_node(None, None, 4);
        let processor = ShapeProcessor;
        let prefs = OutputPreferences::new();

        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.start, 0);
        assert_eq!(config.end, 4);

        // Should always output Shape
        assert!(matches!(node.outputs[0].ty, ArgType::Shape(4)));
    }

    #[test]
    fn test_shape_config_with_start() {
        let mut node = create_test_node(Some(1), None, 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.start, 1);
        assert_eq!(config.end, 4);
    }

    #[test]
    fn test_shape_config_with_end() {
        let mut node = create_test_node(None, Some(3), 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.start, 0);
        assert_eq!(config.end, 3);
    }

    #[test]
    fn test_shape_config_with_start_and_end() {
        let mut node = create_test_node(Some(1), Some(3), 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.start, 1);
        assert_eq!(config.end, 3);
    }

    #[test]
    fn test_shape_config_negative_dims() {
        let mut node = create_test_node(Some(-2), Some(-1), 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.start, 2); // -2 + 4 = 2
        assert_eq!(config.end, 3); // -1 + 4 = 3
    }

    #[test]
    fn test_shape_config_multiple_inputs() {
        let mut node = create_test_node(None, None, 4);
        // Add an extra input to cause error
        node.inputs.push(crate::ir::Argument {
            name: "extra".to_string(),
            ty: crate::ir::ArgType::Tensor(crate::ir::TensorType {
                dtype: crate::ir::DType::F32,
                rank: 4,
                static_shape: None,
            }),
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });

        let processor = ShapeProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }

    // TODO: Missing test for start/end clamping behavior - ONNX spec requires clamping to [0, rank].
    // Need tests for: start > rank, end > rank, start < -rank, end < -rank.
    // These edge cases are mentioned in spec but not validated or tested.

    // TODO: Missing test for opset 15 features - start and end attributes were added in opset 15.
    // Need test to verify behavior when opset < 15 (should not have start/end attributes).

    // TODO: Missing test for zero-rank tensors (scalars) - what should Shape return for rank-0 input?
    // ONNX spec doesn't explicitly cover this edge case.

    #[test]
    fn test_shape_output_type() {
        // Shape operation always outputs Shape type
        let mut node = TestNodeBuilder::new(NodeType::Shape, "test_shape")
            .input_tensor_f32("data", 3, None)
            .output_tensor_i64("shape", 1, None)
            .build();

        let processor = ShapeProcessor;
        let prefs = OutputPreferences::new();

        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should always output Shape type
        assert!(matches!(node.outputs[0].ty, ArgType::Shape(3)));
    }
}
