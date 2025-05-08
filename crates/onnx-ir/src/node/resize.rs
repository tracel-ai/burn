use crate::ir::{ArgType, Node, TensorData};

pub fn resize_config(node: &Node) -> (String, Vec<f32>, Vec<usize>) {
    let mut mode: String = "".to_string();

    let mut scales: Vec<f32>;
    let mut sizes: Vec<usize>;

    let input = if let ArgType::Tensor(tensor) = &node
        .inputs
        .first()
        .expect("Resize: Input tensor must be present")
        .ty
    {
        tensor
    } else {
        panic!("Resize: input must be a tensor")
    };

    // Note: we are ignoring some attributes because results are approximately the same
    // and we are not supporting all the attributes of the Resize operator.
    // However, some attributes are important to be checked and we are checking
    // against the default values of the attributes.
    // TODO revisit this when we have more Resize operators in the model
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "antialias" => assert_eq!(
                value.clone().into_i32(),
                0,
                "Resize: antialias other than 0 is not supported"
            ),
            "axes" => panic!("Resize: custom axes attribute is not supported"),
            "coordinate_transformation_mode" => {
                log::warn!("Resize: coordinate_transformation_mode is ignored")
            }

            "cubic_coeff_a" => log::warn!("Resize: cubic_coeff_a is ignored"),
            "exclude_outside" => assert_eq!(
                value.clone().into_i32(),
                0,
                "Resize: exclude_outside other than 0 is not supported"
            ),
            "extrapolation_value" => assert_eq!(
                value.clone().into_f32(),
                0.0,
                "Resize: extrapolation_value other than 0.0 is not supported"
            ),
            "keep_aspect_ratio_policy" => {
                assert_eq!(
                    value.clone().into_string().to_lowercase(),
                    "stretch",
                    "Resize: keep_aspect_ratio_policy other than 'stretch' is not supported"
                )
            }
            "mode" => mode = value.clone().into_string().to_lowercase(),
            "nearest_mode" => log::warn!("Resize: nearest_mode is ignored"),

            _ => {}
        }
    }

    let roi: Vec<f32> = node
        .inputs
        .get(1)
        .map(|input| {
            if let Some(TensorData { data, .. }) = &input.value {
                data.clone().into_f32s()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();

    scales = node
        .inputs
        .get(2)
        .map(|input| {
            if let Some(TensorData { data, .. }) = &input.value {
                data.clone().into_f32s()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();

    sizes = node
        .inputs
        .get(3)
        .map(|input| {
            if let Some(TensorData { data, .. }) = &input.value {
                data.clone()
                    .into_i64s()
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();

    if mode.is_empty() {
        panic!("Resize: mode attribute is required")
    }

    if !roi.is_empty() {
        panic!("Resize: roi input is not supported")
    }

    if scales.is_empty() && sizes.is_empty() {
        panic!("Resize: either scales or sizes input is required")
    }

    if !scales.is_empty() {
        assert!(scales.len() == input.rank);
        // ignore the fist two items from scales
        // because they are the batch and channel dimensions
        scales = scales.iter().skip(2).cloned().collect();
    }

    if !sizes.is_empty() {
        assert!(sizes.len() == input.rank);
        // ignore the fist two items from sizes
        // because they are the batch and channel dimensions
        sizes = sizes.iter().skip(2).cloned().collect();
    }

    (mode, scales, sizes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        mode: &str,
        scales: Option<Vec<f32>>,
        sizes: Option<Vec<i64>>,
        roi: Option<Vec<f32>>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None) // N,C,H,W format
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", mode);
            
        // Add ROI input if provided
        if let Some(roi_data) = roi {
            builder = builder.input_tensor_f32_data("roi", roi_data.clone(), vec![8]); 
            // For 4D input (start x, start y, end x, end y)
        } else {
            // Empty ROI still needs to be added as a placeholder
            builder = builder.input_tensor_f32("roi", 1, None);
        }
        
        // Add scales input if provided
        if let Some(scales_data) = scales {
            builder = builder.input_tensor_f32_data("scales", scales_data.clone(), vec![4]); 
            // N,C,H,W scales
        } else {
            // Empty scales still needs to be added as a placeholder
            builder = builder.input_tensor_f32("scales", 1, None);
        }
        
        // Add sizes input if provided
        if let Some(sizes_data) = sizes {
            builder = builder.input_tensor_i64_data("sizes", sizes_data.clone(), vec![4]); 
            // N,C,H,W sizes
        } else {
            // Empty sizes still needs to be added as a placeholder
            builder = builder.input_tensor_i64("sizes", 1, None);
        }
        
        builder.build()
    }

    #[test]
    fn test_resize_config_with_scales() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]), // Keep N,C same, double H,W
            None,
            None,
        );
        let (mode, scales, sizes) = resize_config(&node);
        assert_eq!(mode, "nearest");
        assert_eq!(scales, vec![2.0, 2.0]); // Only the spatial scales (H,W)
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_resize_config_with_sizes() {
        let node = create_test_node(
            "linear",
            None,
            Some(vec![1, 3, 224, 224]), // Fixed output size
            None,
        );
        let (mode, scales, sizes) = resize_config(&node);
        assert_eq!(mode, "linear");
        assert!(scales.is_empty());
        assert_eq!(sizes, vec![224, 224]); // Only the spatial sizes (H,W)
    }

    #[test]
    #[should_panic(expected = "Resize: roi input is not supported")]
    fn test_resize_config_with_roi() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]),
            None,
            Some(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), // ROI values
        );
        let _ = resize_config(&node);
    }

    #[test]
    #[should_panic(expected = "Resize: either scales or sizes input is required")]
    fn test_resize_config_no_scales_or_sizes() {
        let node = create_test_node("nearest", None, None, None);
        let _ = resize_config(&node);
    }

    #[test]
    #[should_panic(expected = "Resize: mode attribute is required")]
    fn test_resize_config_no_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None);
        node.attrs.clear(); // Remove all attributes including mode
        let _ = resize_config(&node);
    }
}
