use crate::ir::{Data, Node, NodeConfig, RuntimeInputRef};
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
        crate::util::validate_opset(opset, 11)?;

        // GatherElements requires 2 inputs: data and indices
        crate::util::validate_input_count(node, 2)?;

        // GatherElements has 1 output
        crate::util::validate_output_count(node, 1)?;

        // Output has the same shape as indices input, same type as data input
        if let crate::ir::ArgType::Tensor(data_tensor) = &node.inputs[0].ty
            && let crate::ir::ArgType::Tensor(indices_tensor) = &node.inputs[1].ty
        {
            node.outputs[0].ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
                elem_type: data_tensor.elem_type.clone(),
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

        let indices = if let Some(value) = indices_input.into_value() {
            // Static indices
            match &value.data {
                Data::Int64s(vals) => GatherElementsInput::Static(vals.clone()),
                Data::Int32s(vals) => {
                    let int64_vals = vals.iter().map(|&v| v as i64).collect::<Vec<_>>();
                    GatherElementsInput::Static(int64_vals)
                }
                other => {
                    return Err(ProcessError::Custom(format!(
                        "GatherElements indices must be int32 or int64, got {:?}",
                        other
                    )));
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
