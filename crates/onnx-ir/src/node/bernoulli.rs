use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

/// Update output rank for Bernoulli operation.
pub fn bernoulli_update_output(node: &mut Node) {
    log::debug!("Bernoulli rank inference for node {}", node.name);

    // Get the tensor type and its rank
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Bernoulli: only tensor input is valid"),
    };
    let rank = tensor.rank;
    let static_shape = tensor.static_shape.clone();

    // Infer elem type based on the dtype of the input tensor
    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap());

    log::debug!("Bernoulli: dtype for {}: {:?}", node.name, dtype);

    let elem_type = dtype.map_or(tensor.elem_type, |dtype| match dtype {
        DataType::FLOAT => ElementType::Float32,
        DataType::INT32 => ElementType::Int32,
        DataType::INT64 => ElementType::Int64,
        DataType::DOUBLE => ElementType::Float64,
        DataType::BOOL => ElementType::Bool,
        _ => panic!("Bernoulli: tensor with type {dtype:?} not supported for random output"),
    });
    log::debug!("Bernoulli: elem type for {}: {:?}", node.name, elem_type);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type,
        rank,
        static_shape,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: Option<i32>, static_shape: Option<Vec<usize>>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Bernoulli, "test_bernoulli")
            .input_tensor_f32("input", 4, static_shape) // Rank 0 will be updated
            .output_tensor_f32("output", 0, None); // Rank 0 will be updated

        if let Some(dtype) = dtype {
            builder = builder.attr_int("dtype", dtype as i64)
        }

        builder.build()
    }

    #[test]
    fn test_bernoulli_int() {
        let mut node = create_test_node(Some(DataType::INT32.value()), Some(vec![3, 4, 2]));
        bernoulli_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int32);
                assert_eq!(tensor.static_shape, Some(vec![3, 4, 2]));
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_no_cast() {
        let mut node = create_test_node(None, Some(vec![3, 4, 2]));
        bernoulli_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_no_static_shape() {
        let mut node = create_test_node(None, None);
        bernoulli_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Bernoulli: only tensor input is valid")]
    fn test_bernoulli_invalid_input() {
        let mut node = create_test_node(Some(DataType::FLOAT.value()), None);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        bernoulli_update_output(&mut node);
    }
}
