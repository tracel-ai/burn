//! # Hyperbolic Tangent Operation
//!
//! Element-wise hyperbolic tangent operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Tanh.html>
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 1-5**: Basic support
//! - **Opset 6+**: Improved shape inference
//! - **Opset 13+**: Added support for bfloat16

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};
use onnx_ir_derive::NodeBuilderDerive;

/// Node representation for Tanh operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct TanhNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for hyperbolic tangent operation
pub(crate) struct TanhProcessor;

impl NodeProcessor for TanhProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 6,
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
        validate_opset(opset, 6)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Tanh(TanhNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
