//! # Range
//!
//! Generates a tensor containing a sequence of numbers that begin at `start` and extends by
//! increments of `delta` up to `limit` (exclusive).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Range.html>
//!
//! ## Description
//!
//! The Range operator generates a 1-D tensor containing a sequence of evenly spaced values.
//! The generated sequence starts at `start` and increments by `delta` until reaching `limit`
//! (exclusive). This is similar to Python's `range()` function or NumPy's `arange()`.
//!
//! The number of elements in the output is computed as:
//! `number_of_elements = max(ceil((limit - start) / delta), 0)`
//!
//! Note that `limit` is **exclusive** - the output will not include the limit value itself.
//!
//! ## Type Constraints
//!
//! - T: tensor(double), tensor(float), tensor(int16), tensor(int32), tensor(int64)
//!
//! ## Opset Versions
//!
//! - **Opset 11**: Initial version with scalar inputs for start, limit, and delta.

use crate::ir::{ArgType, Node, NodeConfig, RuntimeInputRef, TensorDataExt, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Configuration for the Range operation.
#[derive(Debug, Clone)]
pub struct RangeConfig {
    pub start: RangeInput,
    pub limit: RangeInput,
    pub delta: RangeInput,
}

impl NodeConfig for RangeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Represents either a static value or a runtime argument for range parameters.
#[derive(Debug, Clone)]
pub enum RangeInput {
    /// Static value known at compile time.
    Static(i64),
    /// Runtime argument determined during execution .
    Runtime(RuntimeInputRef),
}

pub struct RangeProcessor;

impl NodeProcessor for RangeProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Only lift inputs that have static values
        // Runtime inputs (no value) should remain in the graph
        if !node.inputs.is_empty() && node.inputs[0].is_constant() {
            node.inputs[0].to_static()?;
        }

        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate that all three inputs have the same dtype (type T constraint)
        // ONNX spec requires start, limit, delta to all have the same type T.
        // Current implementation infers output dtype from start (inputs[0]) but doesn't
        // validate that limit and delta have matching types. Mismatched types should be rejected.
        // Should add validation: inputs[1].dtype == inputs[0].dtype && inputs[2].dtype == inputs[0].dtype
        // Location: After validate_output_count, before infer output dtype

        // TODO: Validate that all inputs are scalar tensors
        // ONNX spec requires start, limit, delta to be scalar tensors (rank 0 or shape [1]).
        // Implementation extracts scalar values in extract_config but doesn't validate in infer_types.
        // Should validate tensor ranks are 0 or shapes are [1].
        // Location: After dtype validation

        // TODO: Missing test coverage for delta=0 edge case
        // What happens when delta is 0? Should produce empty output or error.
        // Spec says "number_of_elements = max(ceil((limit - start) / delta), 0)"
        // Division by zero case not tested. Add test: range_zero_delta

        // TODO: Missing test coverage for float types
        // Spec supports float and double, but tests only use int64.
        // Add tests: range_float32, range_float64

        // TODO: Missing test coverage for negative delta (descending range)
        // Tests only cover positive delta (ascending). Spec allows negative delta.
        // Add test: range_negative_delta (e.g., start=10, limit=0, delta=-2)

        // TODO: Missing validation for empty range cases
        // When start >= limit with positive delta, or start <= limit with negative delta,
        // the range should be empty. No test validates this. Add test: range_empty

        // Infer output dtype from input types (all inputs should have the same type T)
        let output_dtype = node.inputs[0].ty.elem_type();

        // Range operation always produces rank 1 tensor
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: output_dtype,
            rank: 1,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Helper function to extract range input
        fn get_range_input(
            node: &Node,
            index: usize,
            param_name: &str,
        ) -> Result<RangeInput, ProcessError> {
            let input = node.inputs.get(index).ok_or_else(|| {
                ProcessError::MissingInput(format!("Range: {} parameter is required", param_name))
            })?;

            match input.value() {
                None => Ok(RangeInput::Runtime(RuntimeInputRef::new(
                    input.name.clone(),
                    index,
                ))),
                Some(tensor_data) => match tensor_data.scalar_i64() {
                    Ok(value) => Ok(RangeInput::Static(value)),
                    Err(_) => Err(ProcessError::TypeMismatch {
                        expected: "scalar int value".to_string(),
                        actual: format!("{} must be a scalar int value", param_name),
                    }),
                },
            }
        }

        let start = get_range_input(node, 0, "start")?;
        let limit = get_range_input(node, 1, "limit")?;
        let delta = get_range_input(node, 2, "delta")?;

        let config = RangeConfig {
            start,
            limit,
            delta,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node() -> Node {
        NodeBuilder::new(NodeType::Range, "test_range")
            .input_scalar_i64("start")
            .input_scalar_i64("limit")
            .input_scalar_i64("delta")
            .output_tensor_i64("output", 0, None) // Rank 0 will be updated
            .build()
    }

    #[test]
    fn test_range_output() {
        let mut node = create_test_node();
        let processor = RangeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_range_missing_inputs() {
        let mut node = create_test_node();
        node.inputs.pop();
        let processor = RangeProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 3,
                actual: 2
            })
        ));
    }
}
