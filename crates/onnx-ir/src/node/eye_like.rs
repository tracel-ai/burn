use crate::from_onnx::element_type_from_proto;
use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Configuration for EyeLike operations
#[derive(Debug, Clone, new)]
pub struct EyeLikeConfig {
    /// Data type of the output tensor (optional, defaults to input type)
    pub dtype: Option<ElementType>,
    /// Diagonal offset (0 = main diagonal, >0 = upper, <0 = lower)
    pub k: i64,
}

/// Create an EyeLike configuration from the node
pub fn eye_like_config(node: &Node) -> EyeLikeConfig {
    let mut dtype = None;
    let mut k = 0i64; // default to main diagonal

    // Extract attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "dtype" => {
                let dtype_i32 = value.clone().into_i32();
                dtype = Some(
                    element_type_from_proto(dtype_i32)
                        .unwrap_or_else(|e| panic!("Unsupported dtype for EyeLike: {}", e)),
                );
            }
            "k" => {
                k = value.clone().into_i64();
            }
            _ => {}
        }
    }

    EyeLikeConfig { dtype, k }
}

/// Update output for EyeLike - output has same shape as input, but may have different dtype
pub fn eye_like_update_output(node: &mut Node) {
    log::debug!("EyeLike rank inference for node {}", node.name);

    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.rank, 2, "Input rank must be 2D tensor");

            let config = eye_like_config(node);
            // Output type is either specified dtype or input type
            let output_type = config.dtype.unwrap_or_else(|| tensor.elem_type.clone());

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: output_type,
                rank: tensor.rank,
                static_shape: tensor.static_shape.clone(),
            });
            log::debug!("EyeLike output tensor rank: {}", tensor.rank);
        }
        _ => panic!("EyeLike operation requires 2D tensor input"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;
    use protobuf::Enum;

    #[test]
    fn test_eye_like_update_output() {
        let mut node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .output_tensor_f32("output", 2, None) // rank will be updated
            .build();

        eye_like_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![3, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_eye_like_config_default() {
        let node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![4, 4]))
            .output_tensor_f32("output", 2, None)
            .build();

        let config = eye_like_config(&node);
        assert_eq!(config.k, 0);
        assert_eq!(config.dtype, None);
    }

    #[test]
    fn test_eye_like_config_with_attributes() {
        let node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![4, 4]))
            .output_tensor_f32("output", 2, None)
            .attr_int("k", -1)
            .attr_int("dtype", DataType::INT64.value() as i64)
            .build();

        let config = eye_like_config(&node);
        assert_eq!(config.k, -1);
        assert_eq!(config.dtype, Some(ElementType::Int64));
    }

    #[test]
    fn test_eye_like_update_output_with_dtype() {
        let mut node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .output_tensor_f32("output", 2, None)
            .attr_int("dtype", DataType::INT32.value() as i64)
            .build();

        eye_like_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int32);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![3, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
