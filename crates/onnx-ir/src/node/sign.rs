//! # Sign Operation
//!
//! Element-wise sign operation (returns -1, 0, or 1).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Sign.html>
//!
//! ## Type Constraints
//!
//! T: Numeric tensor types (float, int)
//!
//! ## Opset Versions
//! - **Opset 9+**: Initial version
//! - **Opset 13+**: Added support for bfloat16

use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};

/// Node representation for Sign operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct SignNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for sign operation
pub(crate) struct SignProcessor;

impl NodeProcessor for SignProcessor {
    type Config = ();

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
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 9)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Sign(SignNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
