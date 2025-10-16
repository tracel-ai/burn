use crate::Node;
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

pub struct IsNaNProcessor;

impl NodeProcessor for IsNaNProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::util::validate_opset(opset, 9)?;
        crate::util::validate_input_count(node, 1)?;
        crate::util::validate_output_count(node, 1)?;

        // Output is boolean tensor with same shape as input
        crate::node::comparison::elementwise_comparison_outputs(node);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn test_is_nan_basic() {
        let mut node = NodeBuilder::new(NodeType::IsNaN, "test_is_nan")
            .input_tensor_f32("data", 4, None)
            .output_tensor_bool("output", 4, None)
            .build();

        let processor = IsNaNProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should be boolean with same rank as input
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Bool);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_is_nan_scalar() {
        let mut node = NodeBuilder::new(NodeType::IsNaN, "test_is_nan")
            .add_input("data", ArgType::Scalar(ElementType::Float32))
            .add_output("output", ArgType::Scalar(ElementType::Bool))
            .build();

        let processor = IsNaNProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should be boolean scalar
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Bool);
            }
            _ => panic!("Expected scalar output"),
        }
    }
}
