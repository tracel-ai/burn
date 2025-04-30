use crate::ir::Node;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ArgType, Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node(depth: i64, values: Vec<f32>, axis: Option<i64>) -> Node {
        let inputs = vec![
            Argument {
                name: "indices".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 2,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "depth".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 0,
                    static_shape: None,
                }),
                value: Some(TensorData {
                    data: Data::Int64(depth),
                    shape: vec![],
                }),
                passed: true,
            },
            Argument {
                name: "values".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(TensorData {
                    data: Data::Float32s(values),
                    shape: vec![2], // always [off_value, on_value]
                }),
                passed: true,
            },
        ];

        let mut attrs = HashMap::new();
        if let Some(axis_val) = axis {
            attrs.insert("axis".to_string(), AttributeValue::Int64(axis_val));
        }

        Node {
            node_type: NodeType::OneHot,
            name: "test_one_hot".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3, // rank increases by 1
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
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
