use crate::ir::{ArgType, Node, TensorType};

pub fn one_hot_config(curr: &Node) -> (usize, [f32; 2], i64) {
    let depth = curr.inputs[1]
        .value
        .clone()
        .expect("OneHot: Only constant depth is currently supported")
        .data
        .into_i64();

    let values = curr.inputs[2]
        .value
        .clone()
        .expect("OneHot: Only constant on/off values is currently supported")
        .data
        .into_f32s();

    let axis = curr
        .attrs
        .get("axis")
        .map(|val| val.clone().into_i64())
        .unwrap_or(-1);

    (depth as usize, values.try_into().unwrap(), axis)
}

/// Update output rank for OneHot (input rank + 1).
pub fn one_hot_output_shape(node: &mut Node) {
    log::debug!("OneHot rank inference for node {}", node.name);

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("OneHot: invalid input type"),
    };
    log::debug!("OneHot input rank for {}: {}", node.name, input_rank);

    let output_rank = input_rank + 1;
    log::debug!("OneHot output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.outputs[0].ty.elem_type().clone(),
        rank: output_rank,
        static_shape: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(depth: i64, values: Vec<f32>, axis: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::OneHot, "test_one_hot")
            .input_tensor_i64("indices", 2, None)
            .input_scalar_tensor_i64("depth", depth)
            .input_tensor_f32_data("values", values.clone(), vec![2]) // always [off_value, on_value]
            .output_tensor_f32("output", 3, None); // rank increases by 1

        if let Some(axis_val) = axis {
            builder = builder.attr_int("axis", axis_val);
        }

        builder.build()
    }

    #[test]
    fn test_one_hot_config_basic() {
        let node = create_test_node(5, vec![0.0, 1.0], None);
        let (depth, values, axis) = one_hot_config(&node);
        assert_eq!(depth, 5);
        assert_eq!(values, [0.0, 1.0]);
        assert_eq!(axis, -1); // default axis
    }

    #[test]
    fn test_one_hot_config_with_axis() {
        let node = create_test_node(5, vec![0.0, 1.0], Some(1));
        let (depth, values, axis) = one_hot_config(&node);
        assert_eq!(depth, 5);
        assert_eq!(values, [0.0, 1.0]);
        assert_eq!(axis, 1);
    }

    #[test]
    fn test_one_hot_config_custom_values() {
        let node = create_test_node(10, vec![-1.0, 2.0], None);
        let (depth, values, axis) = one_hot_config(&node);
        assert_eq!(depth, 10);
        assert_eq!(values, [-1.0, 2.0]); // custom off/on values
        assert_eq!(axis, -1);
    }

    #[test]
    #[should_panic(expected = "Only constant depth is currently supported")]
    fn test_one_hot_config_no_depth_value() {
        let mut node = create_test_node(5, vec![0.0, 1.0], None);
        node.inputs[1].value = None; // Remove depth value
        let _ = one_hot_config(&node);
    }

    #[test]
    #[should_panic(expected = "Only constant on/off values is currently supported")]
    fn test_one_hot_config_no_values() {
        let mut node = create_test_node(5, vec![0.0, 1.0], None);
        node.inputs[2].value = None; // Remove values
        let _ = one_hot_config(&node);
    }
}
