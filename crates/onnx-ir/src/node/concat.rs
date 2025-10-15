use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for Concat operation
#[derive(Debug, Clone)]
pub struct ConcatConfig {
    pub axis: usize,
}

impl NodeConfig for ConcatConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ConcatProcessor;

impl NodeProcessor for ConcatProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::util::validate_opset(opset, 4)?;

        // Validate we have at least one input
        crate::util::validate_min_inputs(node, 1)?;

        // Validate output count
        crate::util::validate_output_count(node, 1)?;

        // Extract the axis attribute (required per ONNX spec)
        let mut axis: Option<i64> = None;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = Some(value.clone().into_i64());
                break;
            }
        }

        let axis = axis.ok_or_else(|| ProcessError::MissingAttribute("axis".to_string()))?;

        // extract the rank based on input type
        let rank = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.rank as i64,
            ArgType::Shape(_) => 1, // Shapes are 1D
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // if axis is negative, it is counted from the end
        let normalized_axis = if axis < 0 { axis + rank } else { axis };

        // Validate axis is within bounds
        if normalized_axis < 0 || normalized_axis >= rank {
            return Err(ProcessError::InvalidAttribute {
                name: "axis".to_string(),
                reason: format!("axis {} is out of bounds for rank {}", axis, rank),
            });
        }

        // For shapes, axis must be 0 (since they're 1D)
        if matches!(&node.inputs.first().unwrap().ty, ArgType::Shape(_)) && normalized_axis != 0 {
            return Err(ProcessError::InvalidAttribute {
                name: "axis".to_string(),
                reason: format!(
                    "Concat on Shape inputs only supports axis=0, got axis={}",
                    axis
                ),
            });
        }

        let config = ConcatConfig {
            axis: normalized_axis as usize,
        };
        node.config = Some(Box::new(config));

        // Infer output type
        log::debug!("Concat rank inference for node {}", node.name);

        // Check if we have mixed Shape and rank-1 tensor inputs
        let has_shape = node
            .inputs
            .iter()
            .any(|i| matches!(i.ty, ArgType::Shape(_)));
        let has_rank1_tensor = node
            .inputs
            .iter()
            .any(|i| matches!(&i.ty, ArgType::Tensor(t) if t.rank == 1));

        if has_shape && has_rank1_tensor {
            // Mixed inputs that will be unified after constant conversion
            // Calculate provisional rank by summing Shape ranks and estimating tensor contributions
            let mut provisional_rank: usize = 0;

            for input in &node.inputs {
                match &input.ty {
                    ArgType::Shape(rank) => {
                        provisional_rank += rank;
                    }
                    ArgType::Tensor(t) if t.rank == 1 => {
                        // For constant tensors, use their actual dimension count
                        // For dynamic tensors, assume 1 element (will be corrected after conversion)
                        let contribution =
                            input.into_value().as_ref().map(|v| v.shape[0]).unwrap_or(1);
                        provisional_rank += contribution;

                        log::debug!(
                            "Concat {}: rank-1 tensor {} contributes {} to provisional rank",
                            node.name,
                            input.name,
                            contribution
                        );
                    }
                    _ => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Shape or rank-1 Tensor".to_string(),
                            actual: format!("{:?}", input.ty),
                        });
                    }
                }
            }

            // Output as Shape type since we have Shape inputs
            // The rank is provisional and will be corrected after constant conversion
            node.outputs[0].ty = ArgType::Shape(provisional_rank);
            log::debug!(
                "Concat {} has mixed Shape/Tensor inputs, using provisional Shape({}) output",
                node.name,
                provisional_rank
            );
            return Ok(());
        }

        // Get the first input type - it determines the output type
        let first_input_type = &node.inputs[0].ty;

        match first_input_type {
            ArgType::Tensor(tensor) => {
                log::debug!(
                    "Concat using tensor input rank for {}: {}",
                    node.name,
                    tensor.rank
                );

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: tensor.elem_type.clone(),
                    rank: tensor.rank,
                    static_shape: None,
                });

                log::debug!("Concat output rank for {}: {}", node.name, tensor.rank);
            }
            ArgType::Shape(shape_rank) => {
                log::debug!(
                    "Concat using shape input rank for {}: {}",
                    node.name,
                    shape_rank
                );

                // When concatenating shapes, we sum up their ranks
                let total_rank: usize = node
                    .inputs
                    .iter()
                    .map(|input| match &input.ty {
                        ArgType::Shape(rank) => Ok(*rank),
                        _ => Err(ProcessError::TypeMismatch {
                            expected: "Shape".to_string(),
                            actual: format!("{:?}", input.ty),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .sum();

                node.outputs[0].ty = ArgType::Shape(total_rank);

                log::debug!("Concat output shape rank for {}: {}", node.name, total_rank);
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", first_input_type),
                });
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract the axis attribute (required per ONNX spec)
        let mut axis: Option<i64> = None;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = Some(value.clone().into_i64());
                break;
            }
        }

        let axis = axis.ok_or_else(|| ProcessError::MissingAttribute("axis".to_string()))?;

        // extract the rank based on input type
        let rank = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.rank as i64,
            ArgType::Shape(_) => 1, // Shapes are 1D
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // if axis is negative, it is counted from the end
        let normalized_axis = if axis < 0 { axis + rank } else { axis };

        let config = ConcatConfig {
            axis: normalized_axis as usize,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize, num_inputs: usize) -> Node {
        NodeBuilder::new(NodeType::Concat, "test_concat")
            .input_tensors_f32::<Vec<usize>>("data", num_inputs, input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_concat_config_basic() {
        let node = create_test_node(1, 3, 2);
        let mut node = node;
        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConcatConfig>();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_concat_config_negative_axis() {
        let node = create_test_node(-2, 3, 2);
        let mut node = node;
        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConcatConfig>();
        assert_eq!(config.axis, 1); // -2 + 3 = 1
    }

    #[test]
    fn test_concat_config_shape_input() {
        let node = NodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .output_shape("output", 5)
            .attr_int("axis", 0) // Required attribute
            .build();

        let mut node = node;
        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConcatConfig>();
        assert_eq!(config.axis, 0); // Shape concat uses axis 0
    }

    #[test]
    fn test_concat_config_missing_axis() {
        let node = NodeBuilder::new(NodeType::Concat, "test_concat")
            .input_tensor_f32("data1", 3, None)
            .input_tensor_f32("data2", 3, None)
            .output_tensor_f32("output", 3, None)
            .build();

        let mut node = node;
        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::MissingAttribute(_))));
    }

    #[test]
    fn test_concat_config_axis_out_of_bounds() {
        let node = NodeBuilder::new(NodeType::Concat, "test_concat")
            .input_tensor_f32("data1", 3, None)
            .input_tensor_f32("data2", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_int("axis", 3)
            .build();

        let mut node = node;
        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_concat_update_outputs_shape() {
        let mut node = NodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .input_shape("shape3", 1)
            .output_shape("output", 0) // Will be updated
            .attr_int("axis", 0) // Required attribute
            .build();

        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check that output is Shape with sum of input ranks
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => assert_eq!(*rank, 6), // 2 + 3 + 1
            _ => panic!("Expected Shape output"),
        }
    }

    #[test]
    fn test_concat_config_shape_negative_axis() {
        let node = NodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .output_shape("output", 5)
            .attr_int("axis", -1) // -1 should become 0 for 1D shapes
            .build();

        let mut node = node;
        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConcatConfig>();
        assert_eq!(config.axis, 0); // -1 + 1 = 0
    }

    #[test]
    fn test_concat_config_shape_invalid_axis() {
        let node = NodeBuilder::new(NodeType::Concat, "test_concat_shape")
            .input_shape("shape1", 2)
            .input_shape("shape2", 3)
            .output_shape("output", 5)
            .attr_int("axis", 1)
            .build();

        let mut node = node;
        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_concat_mixed_inputs() {
        let mut node = NodeBuilder::new(NodeType::Concat, "test_concat_mixed")
            .input_shape("shape1", 2)
            .input_tensor_f32("tensor1", 3, None)
            .output_shape("output", 0)
            .attr_int("axis", 0)
            .build();

        let processor = ConcatProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
