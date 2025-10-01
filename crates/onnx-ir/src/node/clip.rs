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

        if min_result.is_none()
            && let Some(min) = min
        {
            let min = min.data.into_scalar();
            min_result = match min {
                Data::Float16(min) => Some(f32::from(min) as f64),
                Data::Float32(min) => Some(min as f64),
                Data::Float64(min) => Some(min),
                Data::Int32(min) => Some(min as f64),
                Data::Int64(min) => Some(min as f64),
                _ => panic!("Clip: unsupported min data type {:?}", min),
            };
        }

        if max_result.is_none()
            && let Some(max) = max
        {
            let max = max.data.into_scalar();
            max_result = match max {
                Data::Float16(max) => Some(f32::from(max) as f64),
                Data::Float32(max) => Some(max as f64),
                Data::Float64(max) => Some(max),
                Data::Int32(max) => Some(max as f64),
                Data::Int64(max) => Some(max as f64),
                _ => panic!("Clip: unsupported max data type {:?}", max),
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
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node_with_attributes(min: Option<f32>, max: Option<f32>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None);

        if let Some(min_val) = min {
            builder = builder.attr_float("min", min_val);
        }

        if let Some(max_val) = max {
            builder = builder.attr_float("max", max_val);
        }

        builder.build()
    }

    fn create_test_node_with_inputs(min: Option<f32>, max: Option<f32>) -> Node {
        NodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .input_scalar_tensor_f32("min", min)
            .input_scalar_tensor_f32("max", max)
            .output_tensor_f32("Y", 4, None)
            .build()
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
