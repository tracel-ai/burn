//! # Mean
//!
//! Element-wise mean of each of the input tensors with multidirectional (Numpy-style)
//! broadcasting support. All inputs and outputs must have the same data type.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Mean.html>
//!
//! ## Opset Versions
//!
//! - **Opset 1-5**: Basic element-wise mean
//! - **Opset 6-7**: Improved broadcasting support
//! - **Opset 8**: Multidirectional (Numpy-style) broadcasting
//! - **Opset 13**: Extended type support including bfloat16

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for Mean operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MeanNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for Mean operation (variadic element-wise mean)
pub(crate) struct MeanProcessor;

impl NodeProcessor for MeanProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 8,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Mean(MeanNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
