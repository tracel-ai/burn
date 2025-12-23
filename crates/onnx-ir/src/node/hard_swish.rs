//! # HardSwish
//!
//! Applies the HardSwish activation function element-wise.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__HardSwish.html>
//!
//! ## Formula
//! ```text
//! y = x * max(0, min(1, (x + 3) / 6))
//! ```
//!
//! ## Type Constraints
//! - `T`: float16, float32, float64, bfloat16
//!
//! ## Opset Versions
//! - **Opset 14**: Initial version
//! - **Opset 22**: Added bfloat16 support

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use onnx_ir_derive::NodeBuilder;

/// Node representation for HardSwish operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct HardSwishNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct HardSwishProcessor;

impl NodeProcessor for HardSwishProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 14,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate no unexpected attributes
        if !node.attrs.is_empty() {
            let keys: Vec<String> = node.attrs.keys().cloned().collect();
            return Err(ProcessError::InvalidAttribute {
                name: keys[0].clone(),
                reason: format!(
                    "HardSwish does not support any attributes, found: {:?}",
                    keys
                ),
            });
        }

        // Output type is same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::HardSwish(HardSwishNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use burn_tensor::DType;

    fn create_test_node() -> RawNode {
        TestNodeBuilder::new(NodeType::HardSwish, "test_hard_swish")
            .input_tensor_f32("X", 4, Some(vec![1, 3, 224, 224]))
            .output_tensor_f32("Y", 0, None)
            .build()
    }

    #[test]
    fn test_hard_swish_type_inference() {
        let mut node = create_test_node();
        let processor = HardSwishProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check output type matches input
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
                assert_eq!(tensor.static_shape, Some(vec![1, 3, 224, 224]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_hard_swish_no_attributes_allowed() {
        let mut node = create_test_node();
        node.attrs.insert(
            "invalid_attr".to_string(),
            crate::ir::AttributeValue::Float32(0.5),
        );
        let processor = HardSwishProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_hard_swish_shape_preservation() {
        let mut node = TestNodeBuilder::new(NodeType::HardSwish, "test_hard_swish")
            .input_tensor_f32("X", 2, Some(vec![10, 20]))
            .output_tensor_f32("Y", 0, None)
            .build();

        let processor = HardSwishProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        if let ArgType::Tensor(tensor) = &node.outputs[0].ty {
            assert_eq!(tensor.static_shape, Some(vec![10, 20]));
        } else {
            panic!("Expected tensor output");
        }
    }
}
