use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

pub struct RandomLikeProcessor;

impl NodeProcessor for RandomLikeProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // RandomLike operations support opset 1+
        if opset < 1 {
            return Err(ProcessError::UnsupportedOpset {
                required: 1,
                actual: opset,
            });
        }

        // Validate input count
        if node.inputs.is_empty() {
            return Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 0,
            });
        }

        // Validate output count
        if node.outputs.len() != 1 {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: node.outputs.len(),
            });
        }

        log::debug!("RandomLike rank inference for node {}", node.name);

        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
            .unwrap_or(DataType::FLOAT);
        log::debug!("RandomLike dtype for {}: {:?}", node.name, dtype);

        let elem_type = match dtype {
            DataType::FLOAT => ElementType::Float32,
            DataType::FLOAT16 => ElementType::Float16,
            DataType::DOUBLE => ElementType::Float64,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "dtype".to_string(),
                    reason: format!("Tensor with type {dtype:?} not supported for random output"),
                });
            }
        };

        if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
            log::debug!("RandomLike input rank for {}: {}", node.name, tensor.rank);

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type,
                rank: tensor.rank,
                static_shape: tensor.static_shape.clone(),
            });

            log::debug!("RandomLike output rank for {}: {}", node.name, tensor.rank);

            Ok(())
        } else {
            Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: "Only tensor input is valid".to_string(),
            })
        }
    }

    fn extract_config(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn crate::ir::NodeConfig>>, ProcessError> {
        // RandomLike has no config
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: i32, input_rank: usize, static_shape: Option<Vec<usize>>) -> Node {
        NodeBuilder::new(NodeType::RandomNormalLike, "test_random_like")
            .input_tensor_f32("input", input_rank, static_shape)
            .output_tensor_f32("output", 0, None) // Rank 0 will be updated
            .attr_int("dtype", dtype as i64)
            .build()
    }

    #[test]
    fn test_random_like_float() {
        let mut node = create_test_node(DataType::FLOAT.value(), 3, None);
        let processor = RandomLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_like_double() {
        let mut node = create_test_node(DataType::DOUBLE.value(), 2, Some(vec![5, 10]));
        let processor = RandomLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![5, 10]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_like_invalid_input() {
        let mut node = create_test_node(DataType::FLOAT.value(), 2, None);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        let processor = RandomLikeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_random_like_unsupported_type() {
        let mut node = create_test_node(DataType::INT32.value(), 2, None);
        let processor = RandomLikeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }
}
