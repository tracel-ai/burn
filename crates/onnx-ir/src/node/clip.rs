use crate::ir::{Data, Node};

pub fn clip_config(node: &Node) -> (Option<f64>, Option<f64>) {
    let mut min_result: Option<f64> = None;
    let mut max_result: Option<f64> = None;

    // For Clip Opset 6+ , the min and max values are attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "min" => {
                let min = value.clone().into_f32() as f64;
                min_result = Some(min);
            }
            "max" => {
                let max = value.clone().into_f32();
                max_result = Some(max as f64);
            }
            _ => {}
        }
    }

    // For Clip Opset 11+ , the min and max values are inputs
    // Get the min and max values from the input values
    if min_result.is_none() && max_result.is_none() {
        let min = node.inputs.get(1).and_then(|arg| arg.value.clone());
        let max = node.inputs.get(2).and_then(|arg| arg.value.clone());

        if min_result.is_none() && min.is_some() {
            let min = min.unwrap().data.into_scalar();
            min_result = match min {
                Data::Float16(min) => Some(f32::from(min) as f64),
                Data::Float32(min) => Some(min as f64),
                Data::Float64(min) => Some(min),
                _ => panic!("Clip: only float min is supported"),
            };
        }

        if max_result.is_none() && max.is_some() {
            let max = max.unwrap().data.into_scalar();
            max_result = match max {
                Data::Float16(max) => Some(f32::from(max) as f64),
                Data::Float32(max) => Some(max as f64),
                Data::Float64(max) => Some(max),
                _ => panic!("Clip: only float max is supported"),
            };
        }
    }

    if min_result.is_none() && max_result.is_none() {
        panic!("Clip: min and max values must be either attributes or inputs");
    }

    (min_result, max_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ArgType, Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node_with_attributes(min: Option<f32>, max: Option<f32>) -> Node {
        let inputs = vec![Argument {
            name: "X".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 4,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        if let Some(min_val) = min {
            attrs.insert("min".to_string(), AttributeValue::Float32(min_val));
        }
        if let Some(max_val) = max {
            attrs.insert("max".to_string(), AttributeValue::Float32(max_val));
        }

        Node {
            node_type: NodeType::Clip,
            name: "test_clip".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "Y".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    fn create_test_node_with_inputs(min: Option<f32>, max: Option<f32>) -> Node {
        let mut inputs = vec![Argument {
            name: "X".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 4,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        // Add min input
        inputs.push(Argument {
            name: "min".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0,
                static_shape: None,
            }),
            value: min.map(|val| TensorData {
                data: Data::Float32(val),
                shape: vec![],
            }),
            passed: true,
        });

        // Add max input
        inputs.push(Argument {
            name: "max".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0,
                static_shape: None,
            }),
            value: max.map(|val| TensorData {
                data: Data::Float32(val),
                shape: vec![],
            }),
            passed: true,
        });

        Node {
            node_type: NodeType::Clip,
            name: "test_clip".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "Y".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn test_clip_config_with_attributes() {
        let node = create_test_node_with_attributes(Some(-1.0), Some(1.0));
        let (min, max) = clip_config(&node);
        assert_eq!(min, Some(-1.0));
        assert_eq!(max, Some(1.0));
    }

    #[test]
    fn test_clip_config_with_attributes_min_only() {
        let node = create_test_node_with_attributes(Some(-1.0), None);
        let (min, max) = clip_config(&node);
        assert_eq!(min, Some(-1.0));
        assert_eq!(max, None);
    }

    #[test]
    fn test_clip_config_with_attributes_max_only() {
        let node = create_test_node_with_attributes(None, Some(1.0));
        let (min, max) = clip_config(&node);
        assert_eq!(min, None);
        assert_eq!(max, Some(1.0));
    }

    #[test]
    fn test_clip_config_with_inputs() {
        let node = create_test_node_with_inputs(Some(-1.0), Some(1.0));
        let (min, max) = clip_config(&node);
        assert_eq!(min, Some(-1.0));
        assert_eq!(max, Some(1.0));
    }

    #[test]
    fn test_clip_config_with_inputs_min_only() {
        let node = create_test_node_with_inputs(Some(-1.0), None);
        let (min, max) = clip_config(&node);
        assert_eq!(min, Some(-1.0));
        assert_eq!(max, None);
    }

    #[test]
    fn test_clip_config_with_inputs_max_only() {
        let node = create_test_node_with_inputs(None, Some(1.0));
        let (min, max) = clip_config(&node);
        assert_eq!(min, None);
        assert_eq!(max, Some(1.0));
    }

    #[test]
    #[should_panic(expected = "Clip: min and max values must be either attributes or inputs")]
    fn test_clip_config_no_min_max() {
        let node = create_test_node_with_attributes(None, None);
        let _ = clip_config(&node);
    }
}
