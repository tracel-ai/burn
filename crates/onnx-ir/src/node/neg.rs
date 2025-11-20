//! # Negation Operation
//!
//! Element-wise negation operation (-x).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Neg.html>
//!
//! ## Type Constraints
//!
//! T: Numeric tensor types (float, int)
//!
//! ## Opset Versions
//! - **Opset 1-5**: Basic support
//! - **Opset 6+**: Improved shape inference
//! - **Opset 13+**: Added support for bfloat16

use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};

/// Node representation for Neg operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct NegNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for negation operation
pub(crate) struct NegProcessor;

impl NodeProcessor for NegProcessor {
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
        Node::Neg(NegNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
