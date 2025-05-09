use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

/// Update output rank for RandomLike operations based on input rank.
pub fn random_like_update_output(node: &mut Node) {
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
        _ => panic!("Tensor with type {dtype:?} not supported for random output"),
    };

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("RandomLike input rank for {}: {}", node.name, tensor.rank);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: tensor.rank,
            static_shape: tensor.static_shape.clone(),
        });

        log::debug!("RandomLike output rank for {}: {}", node.name, tensor.rank);
    } else {
        panic!("Only tensor input is valid");
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
        random_like_update_output(&mut node);

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
        random_like_update_output(&mut node);

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
    #[should_panic(expected = "Only tensor input is valid")]
    fn test_random_like_invalid_input() {
        let mut node = create_test_node(DataType::FLOAT.value(), 2, None);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        random_like_update_output(&mut node);
    }

    #[test]
    #[should_panic(expected = "Tensor with type INT32 not supported for random output")]
    fn test_random_like_unsupported_type() {
        let mut node = create_test_node(DataType::INT32.value(), 2, None);
        random_like_update_output(&mut node);
    }
}
