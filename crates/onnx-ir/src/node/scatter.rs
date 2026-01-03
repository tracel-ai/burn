//! # Scatter
//!
//! Scatter elements into a tensor along a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Scatter.html>
//!
//! ## Description
//!
//! Given `data`, `indices`, and `updates` tensors, this operation copies data from `updates`
//! into a copy of `data` at positions specified by `indices`. The `axis` attribute specifies
//! which axis to scatter along.
//!
//! For each entry in `updates`:
//! `output[indices[i][j][k]][j][k] = updates[i][j][k]` (if axis=0)
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
//! - **Opset 9**: Initial version (deprecated in favor of ScatterElements).
//! - **Opset 11**: Added support for negative indices.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for the Scatter operation.
#[derive(Debug, Clone, new)]
pub struct ScatterConfig {
    /// The axis along which to scatter. Default is 0.
    pub axis: i64,
}

/// Node representation for Scatter operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ScatterNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ScatterConfig,
}

pub(crate) struct ScatterProcessor;

impl NodeProcessor for ScatterProcessor {
    type Config = ScatterConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9, // Scatter was introduced in opset 9
            max_opset: None,
            inputs: InputSpec::Exact(3), // data, indices, updates
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output has the same type and shape as input data
        node.outputs[0].ty = node.inputs[0].ty.clone();
        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut axis: i64 = 0;
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Scatter: {}", key),
                    });
                }
            }
        }
        Ok(ScatterConfig { axis })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");
        Node::Scatter(ScatterNode {
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

    fn create_test_node(axis: i64) -> RawNode {
        TestNodeBuilder::new(NodeType::Scatter, "test_scatter")
            .attr_int("axis", axis)
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("indices", 2, None)
            .input_tensor_f32("updates", 2, None)
            .output_tensor_f32("output", 2, None)
            .build()
    }

    #[test]
    fn test_scatter_config_basic() {
        let node = create_test_node(0);
        let processor = ScatterProcessor;
        let config = processor.extract_config(&node, 11).unwrap();
        assert_eq!(config.axis, 0);
    }

    #[test]
    fn test_scatter_config_axis_1() {
        let node = create_test_node(1);
        let processor = ScatterProcessor;
        let config = processor.extract_config(&node, 11).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_scatter_config_negative_axis() {
        let node = create_test_node(-1);
        let processor = ScatterProcessor;
        let config = processor.extract_config(&node, 11).unwrap();
        assert_eq!(config.axis, -1);
    }

    #[test]
    fn test_scatter_infer_types() {
        let mut node = create_test_node(0);
        let processor = ScatterProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 11, &prefs).unwrap();
        // Output should have same type as input data
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }
}

