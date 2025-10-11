use crate::from_onnx::element_type_from_proto;
use crate::ir::{ArgType, ElementType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, ProcessorContext};
use std::any::Any;

/// Configuration for EyeLike operations
#[derive(Debug, Clone, new)]
pub struct EyeLikeConfig {
    /// Data type of the output tensor (optional, defaults to input type)
    pub dtype: Option<ElementType>,
    /// Diagonal offset (0 = main diagonal, >0 = upper, <0 = lower)
    pub k: i64,
}

impl NodeConfig for EyeLikeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct EyeLikeProcessor;

impl NodeProcessor for EyeLikeProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (9, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
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

        let config = EyeLikeConfig { dtype, k };
        node.config = Some(Box::new(config));
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        log::debug!("EyeLike rank inference for node {}", node.name);

        // Extract tensor info before calling process_config
        let (input_rank, input_elem_type, input_static_shape) = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2, "Input rank must be 2D tensor");
                (
                    tensor.rank,
                    tensor.elem_type.clone(),
                    tensor.static_shape.clone(),
                )
            }
            _ => panic!("EyeLike operation requires 2D tensor input"),
        };

        let processor = EyeLikeProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(node, &context, graph_data);

        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<EyeLikeConfig>()
            .unwrap();
        // Output type is either specified dtype or input type
        let output_type = config.dtype.clone().unwrap_or(input_elem_type);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: output_type,
            rank: input_rank,
            static_shape: input_static_shape,
        });
        log::debug!("EyeLike output tensor rank: {}", input_rank);
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

        let processor = EyeLikeProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process_forward(&mut node, &context, &mut graph_data);

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

        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = node;

        let processor = EyeLikeProcessor;

        let context = ProcessorContext::new(16);

        processor.process_config(&mut node, &context, &mut graph_data);

        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<EyeLikeConfig>()
            .unwrap();
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

        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = node;

        let processor = EyeLikeProcessor;

        let context = ProcessorContext::new(16);

        processor.process_config(&mut node, &context, &mut graph_data);

        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<EyeLikeConfig>()
            .unwrap();
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

        let processor = EyeLikeProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process_forward(&mut node, &context, &mut graph_data);

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
