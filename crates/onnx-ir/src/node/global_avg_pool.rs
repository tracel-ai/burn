//! # GlobalAveragePool
//!
//! Applies global average pooling to the input tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{ArgType, Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for GlobalAveragePool operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct GlobalAveragePoolNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct GlobalAveragePoolProcessor;

impl NodeProcessor for GlobalAveragePoolProcessor {
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
        // Get input tensor type
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Output has the same type and rank as input (spatial dimensions become 1)
        node.outputs[0].ty = ArgType::Tensor(input_tensor.clone());

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
        Node::GlobalAveragePool(GlobalAveragePoolNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_global_avg_pool_type_inference() {
        let mut node = TestNodeBuilder::new(NodeType::GlobalAveragePool, "test")
            .input_tensor_f32("input", 4, None)
            .output_tensor_f32("output", 4, None)
            .build();

        let processor = GlobalAveragePoolProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should have same type and rank as input
        if let ArgType::Tensor(output_tensor) = &node.outputs[0].ty {
            assert_eq!(output_tensor.dtype, DType::F32);
            assert_eq!(output_tensor.rank, 4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
