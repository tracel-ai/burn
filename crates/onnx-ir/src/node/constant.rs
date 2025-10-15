use crate::ir::{ArgType, AttributeValue, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

pub struct ConstantProcessor;

impl NodeProcessor for ConstantProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        if opset < 9 {
            return Err(ProcessError::UnsupportedOpset {
                required: 9,
                actual: opset,
            });
        }

        // Validate output count (constants don't have inputs)
        if node.outputs.len() != 1 {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: node.outputs.len(),
            });
        }

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

        node.outputs[0].ty = match matched_value {
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
}
