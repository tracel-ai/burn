//! # IsNaN
//!
//! Returns which elements of the input are NaN (Not a Number).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__IsNaN.html>
//!
//! ## Type Constraints
//!
//! - T1: tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(float8e4m3fn),
//!   tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz)
//! - T2: tensor(bool)
//!
//! ## Opset Versions
//! - **Opset 9-12**: Initial version
//! - **Opset 13-19**: Extended type support (added bfloat16)
//! - **Opset 20+**: Added float8 type variants support
//!
//! ## Missing Test Coverage
//! - TODO: No test for mixed NaN/Inf/finite values in same tensor - Current test only has NaN and finite
//! - TODO: No test for zero-size tensors - Edge case for empty tensor handling
//! - TODO: No test validating that input must be floating-point type - Integer inputs should be rejected
//! - TODO: No test for higher-rank tensors (3D, 4D) - Only 2D tensor tested
//! - TODO: No test for positive/negative NaN variants - Some platforms distinguish signaling/quiet NaN

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::{Node, NodeBuilder};

pub struct IsNaNProcessor;

impl NodeProcessor for IsNaNProcessor {
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
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input dtype is floating-point - Type constraint T1: tensor(float16), tensor(float), tensor(double), tensor(bfloat16), tensor(float8*) not enforced - Integer inputs should be rejected - burn/crates/onnx-ir/src/node/is_nan.rs:42

        // TODO: Validate that no unexpected attributes are present
        // The spec states "None" for attributes
        if let Some((key, _value)) = node.attrs.iter().next() {
            return Err(ProcessError::InvalidAttribute {
                name: key.clone(),
                reason: format!("IsNaN does not accept any attributes, found: {}", key),
            });
        }

        // Output is boolean tensor with same shape as input
        crate::node::comparison::elementwise_comparison_outputs(node);

        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder) -> Node {
        Node::IsNaN {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_is_nan_basic() {
        let mut node = TestNodeBuilder::new(NodeType::IsNaN, "test_is_nan")
            .input_tensor_f32("data", 4, None)
            .output_tensor_bool("output", 4, None)
            .build();

        let processor = IsNaNProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should be boolean with same rank as input
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::Bool);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_is_nan_scalar() {
        let mut node = TestNodeBuilder::new(NodeType::IsNaN, "test_is_nan")
            .add_input("data", ArgType::Scalar(DType::F32))
            .add_output("output", ArgType::Scalar(DType::Bool))
            .build();

        let processor = IsNaNProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should be boolean scalar
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::Bool);
            }
            _ => panic!("Expected scalar output"),
        }
    }
}
