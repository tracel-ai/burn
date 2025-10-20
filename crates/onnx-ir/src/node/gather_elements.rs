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
//! ## Attributes
//! - `axis` (int64, default=0): Axis along which to gather. Negative values count from the end.
//!   Accepted range is [-r, r-1] where r = rank(data).
//!
//! ## Inputs
//! - `data` (T): Input data tensor of rank r >= 1
//! - `indices` (Tind): Indices tensor of rank r (same rank as data). Element type must be int32 or int64
//!
//! ## Outputs
//! - `output` (T): Output tensor with same shape as indices and same element type as data
//!
//! ## Type Constraints
//! - T: All tensor types
//! - Tind: int32, int64
//!
//! ## Opset Versions
//! - **Opset 11**: Initial version with per-element indexing along a specified axis.
//! - **Opset 13**: Added bfloat16 support and clarified negative index handling.

use crate::ir::{Node, NodeConfig, RuntimeInputRef};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for the GatherElements operation.
#[derive(Debug, Clone)]
pub struct GatherElementsConfig {
    pub indices: GatherElementsInput,
    pub axis: usize,
}

impl NodeConfig for GatherElementsConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Represents either a static value or a runtime argument for gather elements indices.
#[derive(Debug, Clone)]
pub enum GatherElementsInput {
    /// Static value known at compile time.
    Static(Vec<i64>),
    /// Runtime argument determined during execution.
    Runtime(RuntimeInputRef),
}

pub struct GatherElementsProcessor;

impl NodeProcessor for GatherElementsProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // GatherElements was introduced in opset 11
        crate::processor::validate_opset(opset, 11)?;

        // GatherElements requires 2 inputs: data and indices
        crate::processor::validate_input_count(node, 2)?;

        // GatherElements has 1 output
        crate::processor::validate_output_count(node, 1)?;

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
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
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
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()
            }
            // TODO: Add validation for unexpected attributes (currently silently ignored)
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
        Ok(Some(Box::new(config)))
    }
}
