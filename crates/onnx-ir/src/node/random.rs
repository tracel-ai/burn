use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, ProcessorContext};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

pub struct RandomProcessor;

impl NodeProcessor for RandomProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (1, None)
    }

    fn process(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        log::debug!("Random rank inference for node {}", node.name);

        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
            .unwrap_or(DataType::FLOAT);
        log::debug!("Random dtype for {}: {:?}", node.name, dtype);

        let shape = node
            .attrs
            .get("shape")
            .expect("required shape attribute missing")
            .clone()
            .into_i64s();
        log::debug!("Random shape for {}: {:?}", node.name, shape);

        let elem_type = match dtype {
            DataType::FLOAT => ElementType::Float32,
            DataType::DOUBLE => ElementType::Float64,
            _ => panic!("tensor with type {dtype:?} not supported for random output"),
        };

        let rank = shape.len();
        log::debug!("Random output rank for {}: {}", node.name, rank);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank,
            static_shape: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: i32, shape: Vec<i64>) -> Node {
        NodeBuilder::new(NodeType::RandomNormal, "test_random")
            .output_tensor_f32("output", 0, None) // Rank 0 will be updated
            .attr_int("dtype", dtype as i64)
            .attr_ints("shape", shape)
            .build()
    }

    #[test]
    fn test_random_normal_float() {
        let mut node = create_test_node(DataType::FLOAT.value(), vec![2, 3, 4]);
        let processor = RandomProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_normal_double() {
        let mut node = create_test_node(DataType::DOUBLE.value(), vec![5]);
        let processor = RandomProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "required shape attribute missing")]
    fn test_random_normal_missing_shape() {
        // Create node and then manually remove the shape attribute
        let mut node = create_test_node(DataType::FLOAT.value(), vec![2, 3]);
        node.attrs.remove("shape");
        let processor = RandomProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process(&mut node, &context, &mut graph_data);
    }

    #[test]
    #[should_panic(expected = "tensor with type INT32 not supported for random output")]
    fn test_random_normal_unsupported_type() {
        let mut node = create_test_node(DataType::INT32.value(), vec![2, 3]);
        let processor = RandomProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process(&mut node, &context, &mut graph_data);
    }
}
