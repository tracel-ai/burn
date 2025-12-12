//! # CumSum
//!
//! Performs cumulative sum of the input tensor along the given axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__CumSum.html>
//!
//! ## Opset Versions
//! - **Opset 11**: Initial version
//! - **Opset 14**: Added bfloat16 and float16 support

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::TensorDataExt;
use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for CumSum operation
#[derive(Debug, Clone, new)]
pub struct CumSumConfig {
    /// The axis along which to compute cumulative sum
    pub axis: usize,
    /// If true, the j-th output is the sum of the first (j-1) elements (excludes current)
    pub exclusive: bool,
    /// If true, perform cumulative sum in reverse direction
    pub reverse: bool,
}

/// Node representation for CumSum operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct CumSumNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: CumSumConfig,
}

pub(crate) struct CumSumProcessor;

impl NodeProcessor for CumSumProcessor {
    type Config = CumSumConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(2), // x and axis
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift axis input (input[1]) to static value
        if node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output type is the same as input type
        crate::processor::same_as_input(node);
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract axis from input[1] (scalar tensor)
        let axis_input = &node.inputs[1];
        let axis_vec = axis_input
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("CumSum: axis must be a constant value".to_string())
            })?
            .to_i64_vec()
            .map_err(|e| ProcessError::Custom(format!("CumSum: failed to extract axis: {}", e)))?;

        // Axis is a scalar (0-D tensor), so it should have exactly one element
        let axis_value = if axis_vec.is_empty() {
            return Err(ProcessError::Custom(
                "CumSum: axis tensor is empty".to_string(),
            ));
        } else {
            axis_vec[0]
        };

        // Get tensor rank for negative axis handling
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Handle negative axis
        let axis = if axis_value < 0 {
            (tensor.rank as i64 + axis_value) as usize
        } else {
            axis_value as usize
        };

        // Validate axis
        if axis >= tensor.rank {
            return Err(ProcessError::Custom(format!(
                "CumSum: axis {} is out of bounds for tensor of rank {}",
                axis_value, tensor.rank
            )));
        }

        // Extract exclusive attribute (default: 0)
        let exclusive = node
            .attrs
            .get("exclusive")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        // Extract reverse attribute (default: 0)
        let reverse = node
            .attrs
            .get("reverse")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(false);

        Ok(CumSumConfig {
            axis,
            exclusive,
            reverse,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::CumSum(CumSumNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(axis: i64, exclusive: i64, reverse: i64, rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::CumSum, "test_cumsum")
            .input_tensor_f32("x", rank, None)
            .input_tensor_i64_data("axis", vec![axis], vec![])
            .output_tensor_f32("y", rank, None)
            .attr_int("exclusive", exclusive)
            .attr_int("reverse", reverse)
            .build_with_graph_data(14)
    }

    #[test]
    fn test_cumsum_config_default() {
        let node = create_test_node(0, 0, 0, 3);
        let processor = CumSumProcessor;
        let config = processor.extract_config(&node, 14).unwrap();
        assert_eq!(config.axis, 0);
        assert!(!config.exclusive);
        assert!(!config.reverse);
    }

    #[test]
    fn test_cumsum_config_exclusive() {
        let node = create_test_node(1, 1, 0, 3);
        let processor = CumSumProcessor;
        let config = processor.extract_config(&node, 14).unwrap();
        assert_eq!(config.axis, 1);
        assert!(config.exclusive);
        assert!(!config.reverse);
    }

    #[test]
    fn test_cumsum_config_reverse() {
        let node = create_test_node(0, 0, 1, 3);
        let processor = CumSumProcessor;
        let config = processor.extract_config(&node, 14).unwrap();
        assert_eq!(config.axis, 0);
        assert!(!config.exclusive);
        assert!(config.reverse);
    }

    #[test]
    fn test_cumsum_config_exclusive_reverse() {
        let node = create_test_node(2, 1, 1, 3);
        let processor = CumSumProcessor;
        let config = processor.extract_config(&node, 14).unwrap();
        assert_eq!(config.axis, 2);
        assert!(config.exclusive);
        assert!(config.reverse);
    }

    #[test]
    fn test_cumsum_config_negative_axis() {
        let node = create_test_node(-1, 0, 0, 3);
        let processor = CumSumProcessor;
        let config = processor.extract_config(&node, 14).unwrap();
        assert_eq!(config.axis, 2); // -1 + 3 = 2
    }

    #[test]
    fn test_cumsum_type_inference() {
        let mut node = create_test_node(0, 0, 0, 3);
        let processor = CumSumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 14, &prefs).unwrap();

        // Output should have same type as input
        match (&node.inputs[0].ty, &node.outputs[0].ty) {
            (ArgType::Tensor(input_t), ArgType::Tensor(output_t)) => {
                assert_eq!(input_t.dtype, output_t.dtype);
                assert_eq!(input_t.rank, output_t.rank);
            }
            _ => panic!("Expected tensor types"),
        }
    }
}
