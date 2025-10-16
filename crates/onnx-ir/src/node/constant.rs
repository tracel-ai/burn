use crate::ir::{ArgType, AttributeValue, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

pub struct ConstantProcessor;

impl NodeProcessor for ConstantProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::util::validate_opset(opset, 9)?;
        crate::util::validate_output_count(node, 1)?;

        log::debug!("Constant rank inference for node {}", node.name);

        let keys = [
            "value",
            "value_float",
            "value_floats",
            "value_int",
            "value_ints",
            "value_string",
            "value_strings",
            "sparse_value",
        ];

        let matched_value = keys.iter().find_map(|&key| node.attrs.get(key).cloned());
        log::debug!("Constant found attribute: {}", matched_value.is_some());

        // First, determine the base type from the constant value
        let base_type = match matched_value {
            Some(value) => match &value {
                AttributeValue::Tensor(tensor) if tensor.shape.is_empty() => {
                    log::debug!("Constant as scalar for {}", node.name);
                    ArgType::Scalar(tensor.elem_type())
                }
                AttributeValue::Tensor(tensor) => {
                    log::debug!(
                        "Constant tensor with rank {} for {}",
                        tensor.shape.len(),
                        node.name
                    );
                    ArgType::Tensor(TensorType {
                        elem_type: tensor.elem_type(),
                        rank: tensor.shape.len(),
                        static_shape: Some(tensor.shape.clone()),
                    })
                }
                AttributeValue::Float32(_) => {
                    log::debug!("Constant Float32 scalar for {}", node.name);
                    ArgType::Scalar(ElementType::Float32)
                }
                AttributeValue::Float32s(values) => {
                    log::debug!("Constant Float32s tensor with rank 1 for {}", node.name);
                    ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 1,
                        static_shape: Some(vec![values.len()]),
                    })
                }
                AttributeValue::Int64(_) => {
                    log::debug!("Constant Int64 scalar for {}", node.name);
                    ArgType::Scalar(ElementType::Int64)
                }
                AttributeValue::Int64s(values) => {
                    log::debug!("Constant Int64s tensor with rank 1 for {}", node.name);
                    ArgType::Tensor(TensorType {
                        elem_type: ElementType::Int64,
                        rank: 1,
                        static_shape: Some(vec![values.len()]),
                    })
                }
                ty => {
                    return Err(ProcessError::Custom(format!(
                        "Constant value of {:?} is not supported",
                        ty
                    )));
                }
            },
            None => {
                return Err(ProcessError::MissingAttribute("value".to_string()));
            }
        };

        // Check output preferences to see if consumers want this converted
        let output_name = &node.outputs[0].name;
        let preferences = output_preferences.get(output_name);

        // Apply preferences if any exist
        node.outputs[0].ty = if !preferences.is_empty() {
            // Check if any consumer wants Shape type
            let wants_shape = preferences
                .iter()
                .any(|(_, ty)| matches!(ty, crate::processor::ArgPreference::Shape));

            // Check if any consumer wants Scalar type
            let wants_scalar = preferences
                .iter()
                .any(|(_, ty)| matches!(ty, crate::processor::ArgPreference::Scalar));

            match &base_type {
                // Convert 1D tensor to Shape if requested and we have static shape info
                ArgType::Tensor(tensor) if tensor.rank == 1 && wants_shape => {
                    if let Some(shape) = tensor.static_shape.as_ref() {
                        if let Some(&shape_rank) = shape.first() {
                            log::debug!(
                                "Converting constant {} from Tensor(rank=1) to Shape({})",
                                node.name,
                                shape_rank
                            );
                            ArgType::Shape(shape_rank)
                        } else {
                            // Empty shape for rank-1 tensor is invalid, keep as Tensor
                            log::warn!(
                                "Constant {} has rank 1 but empty static_shape, keeping as Tensor",
                                node.name
                            );
                            base_type
                        }
                    } else {
                        // No static shape info, keep as Tensor
                        log::warn!(
                            "Constant {} lacks static_shape information, keeping as Tensor",
                            node.name
                        );
                        base_type
                    }
                }
                // Convert scalar-compatible tensor to Scalar if requested
                ArgType::Tensor(tensor) if tensor.rank == 0 && wants_scalar => {
                    log::debug!(
                        "Converting constant {} from Tensor(rank=0) to Scalar",
                        node.name
                    );
                    ArgType::Scalar(tensor.elem_type.clone())
                }
                // Otherwise keep base type
                _ => {
                    log::debug!(
                        "Constant {} keeping base type {:?} (preferences: {:?})",
                        node.name,
                        base_type,
                        preferences
                    );
                    base_type
                }
            }
        } else {
            log::debug!("Constant {} has no preferences, using base type", node.name);
            base_type
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{NodeType, TensorData};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node() -> Node {
        NodeBuilder::new(NodeType::Constant, "test_constant")
            .output_tensor_f32("output", 0, None) // This will be overwritten
            .build()
    }

    #[test]
    fn test_constant_scalar_float() {
        let mut node = create_test_node();
        node.attrs
            .insert("value_float".to_string(), AttributeValue::Float32(6.14));

        let processor = ConstantProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    fn test_constant_tensor() {
        let mut node = create_test_node();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![2, 3],
                data: crate::ir::Data::Float32s(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            }),
        );

        let processor = ConstantProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_constant_missing_value() {
        let mut node = create_test_node();

        let processor = ConstantProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::MissingAttribute { .. })));
    }

    #[test]
    fn test_constant_1d_tensor_to_shape_with_preferences() {
        let mut node = create_test_node();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![3], // 1D tensor with 3 elements
                data: crate::ir::Data::Int64s(vec![10, 20, 30]),
            }),
        );

        // Create preferences requesting Shape type
        let mut prefs = OutputPreferences::new();
        prefs.add(
            node.outputs[0].name.clone(),
            "consumer_node",
            crate::processor::ArgPreference::Shape,
        );

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should be converted to Shape(3) since we have 3 elements
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 3);
            }
            other => panic!("Expected Shape output, got {:?}", other),
        }
    }

    #[test]
    fn test_constant_1d_tensor_without_preferences() {
        let mut node = create_test_node();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![3],
                data: crate::ir::Data::Int64s(vec![10, 20, 30]),
            }),
        );

        // No preferences
        let prefs = OutputPreferences::new();

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should remain as Tensor
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.elem_type, ElementType::Int64);
            }
            other => panic!("Expected Tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_constant_rank0_tensor_to_scalar_with_preferences() {
        let mut node = create_test_node();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![], // rank 0 tensor
                data: crate::ir::Data::Float32(42.0),
            }),
        );

        // Create preferences requesting Scalar type
        let mut prefs = OutputPreferences::new();
        prefs.add(
            node.outputs[0].name.clone(),
            "consumer_node",
            crate::processor::ArgPreference::Scalar,
        );

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should already be Scalar (rank 0 tensor is treated as scalar by default)
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            other => panic!("Expected Scalar output, got {:?}", other),
        }
    }

    #[test]
    fn test_constant_2d_tensor_ignores_shape_preference() {
        let mut node = create_test_node();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![2, 3], // 2D tensor - cannot convert to Shape
                data: crate::ir::Data::Float32s(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            }),
        );

        // Create preferences requesting Shape type (which shouldn't apply to 2D tensor)
        let mut prefs = OutputPreferences::new();
        prefs.add(
            node.outputs[0].name.clone(),
            "consumer_node",
            crate::processor::ArgPreference::Shape,
        );

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should remain as Tensor (2D tensor cannot be converted to Shape)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.elem_type, ElementType::Float32);
            }
            other => panic!("Expected Tensor output, got {:?}", other),
        }
    }
}
