//! # GatherElements
//!
//! Gathers data from the input tensor at indices specified by the indices tensor along a given axis.
//! Unlike `Gather`, which uses array indexing, `GatherElements` performs per-element indexing where
//! each element in the indices tensor specifies which element to select from the corresponding position
//! in the input tensor along the specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GatherElements.html>
//!
//! ## Key Difference from Gather
//! - `Gather`: Uses array-style indexing to select entire slices. Output rank = indices_rank + data_rank - 1
//! - `GatherElements`: Uses per-element indexing. Output shape = indices shape, same rank as indices
//!
//! ## Type Constraints
//! - T: All tensor types
//! - Tind: int32, int64
//!
//! ## Opset Versions
//! - **Opset 11**: Initial version with per-element indexing along a specified axis.
//! - **Opset 13**: Added bfloat16 support and clarified negative index handling.
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, NodeBuilder, RuntimeInputRef, TensorDataExt};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for the GatherElements operation.
#[derive(Debug, Clone, new)]
pub struct GatherElementsConfig {
    pub indices: GatherElementsInput,
    pub axis: usize,
}

/// Node representation for GatherElements operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GatherElementsNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: GatherElementsConfig,
}

/// Represents either a static value or a runtime argument for gather elements indices.
#[derive(Debug, Clone)]
pub enum GatherElementsInput {
    /// Static value known at compile time.
    Static(Vec<i64>),
    /// Runtime argument determined during execution.
    Runtime(RuntimeInputRef),
}

impl Default for GatherElementsInput {
    fn default() -> Self {
        GatherElementsInput::Static(Vec::new())
    }
}

pub(crate) struct GatherElementsProcessor;

impl NodeProcessor for GatherElementsProcessor {
    type Config = GatherElementsConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate indices tensor type is int32 or int64 per ONNX spec - Missing type constraint validation
        // TODO: Validate data and indices have same rank per ONNX spec - Spec requires rank(data) == rank(indices) - Missing rank validation

        // Output has the same shape as indices input, same type as data input
        if let crate::ir::ArgType::Tensor(data_tensor) = &node.inputs[0].ty
            && let crate::ir::ArgType::Tensor(indices_tensor) = &node.inputs[1].ty
        {
            node.outputs[0].ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
                dtype: data_tensor.dtype,
                rank: indices_tensor.rank,
                static_shape: indices_tensor.static_shape.clone(),
            });
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract the input rank for axis normalization
        let input_dim = match &node.inputs[0].ty {
            crate::ir::ArgType::Tensor(tensor) => tensor.rank as i64,
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        // Extract the axis attribute (default: 0 per ONNX spec)
        let mut axis: i64 = 0;
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for GatherElements: {}", key),
                    });
                }
            }
        }

        // Normalize negative axis
        if axis < 0 {
            axis += input_dim;
        }

        // Validate axis is within bounds
        if axis < 0 || axis >= input_dim {
            return Err(ProcessError::InvalidAttribute {
                name: "axis".to_string(),
                reason: format!("axis {} is out of bounds for rank {}", axis, input_dim),
            });
        }

        // Get indices input
        let indices_input = &node.inputs[1];

        let indices = if let Some(value) = indices_input.value() {
            // Static indices - convert to i64 vec
            match value.to_i64_vec() {
                Ok(indices) => GatherElementsInput::Static(indices),
                Err(_) => {
                    return Err(ProcessError::Custom(
                        "GatherElements indices must be int32 or int64".to_string(),
                    ));
                }
            }
        } else {
            // Runtime indices
            GatherElementsInput::Runtime(RuntimeInputRef::new(indices_input.name.clone(), 1))
        };

        let config = GatherElementsConfig {
            indices,
            axis: axis as usize,
        };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::GatherElements(GatherElementsNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

// TODO: Add unit tests for GatherElements - No tests found for this operator - Missing test coverage entirely
// TODO: Add test for rank validation - data and indices must have same rank per spec - Missing constraint test
// TODO: Add test for indices type validation - indices must be int32 or int64 - Missing type constraint test
// TODO: Add test for negative indices - Opset 13+ clarifies negative index handling - Missing negative indices test
// TODO: Add test for out-of-bounds indices - Spec requires error for out-of-bounds - Missing bounds checking test
// TODO: Add test for axis validation - Test axis out of range should return error - Missing constraint test
// TODO: Add test for unexpected attributes - Should reject unknown attributes per implementation - Missing attribute validation test
