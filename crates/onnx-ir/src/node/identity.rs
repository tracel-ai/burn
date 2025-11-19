//! # Identity
//!
//! Passes input through unchanged - typically eliminated during post-processing.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Identity.html>
//!
//! ## Opset Versions
//! - **Opset 1-13**: Basic identity operation
//! - **Opset 14**: Added bfloat16 support
//! - **Opset 16**: Added optional sequence support
//! - **Opset 19**: Added int4/uint4 support
//!
//! ## Note
//! Identity nodes are typically eliminated during the post-processing phase to simplify
//! the graph. They exist primarily for graph construction and optimization purposes.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for Identity operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct IdentityNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct IdentityProcessor;

impl NodeProcessor for IdentityProcessor {
    type Config = ();

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
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Identity passes input type through unchanged
        node.outputs[0].ty = node.inputs[0].ty.clone();

        Ok(())
    }

    fn extract_config(
        &self,
        _node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Identity(IdentityNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_identity_type_inference() {
        let mut node = TestNodeBuilder::new(NodeType::Identity, "test")
            .input_tensor_f32("input", 3, None)
            .output_tensor_f32("output", 3, None)
            .build();

        let processor = IdentityProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should have same type as input
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);

        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, 3);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_identity_scalar() {
        let mut node = TestNodeBuilder::new(NodeType::Identity, "test")
            .input_scalar_f32("input")
            .output_scalar_f32("output")
            .build();

        let processor = IdentityProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should have same type as input
        assert!(matches!(node.outputs[0].ty, ArgType::Scalar(_)));
    }
}
