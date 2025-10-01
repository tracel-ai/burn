use crate::ir::{ArgType, Argument, Node, TensorData};
use std::str::FromStr;

/// Interpolation mode for resize operation
#[derive(Debug, Clone, PartialEq)]
pub enum ResizeMode {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation (bilinear for 2D, trilinear for 3D)
    Linear,
    /// Cubic interpolation
    Cubic,
}

impl FromStr for ResizeMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nearest" => Ok(ResizeMode::Nearest),
            "linear" => Ok(ResizeMode::Linear),
            "cubic" => Ok(ResizeMode::Cubic),
            _ => Err(format!("Unsupported resize mode: {}", s)),
        }
    }
}

/// Configuration for the Resize operation.
#[derive(Debug, Clone)]
pub struct ResizeConfig {
    pub mode: ResizeMode,
    pub scales: Option<ResizeScales>,
    pub sizes: Option<ResizeSizes>,
}

/// Represents either a static value or a runtime argument for resize scales.
#[derive(Debug, Clone)]
pub enum ResizeScales {
    /// Static scales known at compile time.
    Static(Vec<f32>),
    /// Runtime scales determined during execution.
    Runtime(Argument),
}

/// Represents either a static value or a runtime argument for resize sizes.
#[derive(Debug, Clone)]
pub enum ResizeSizes {
    /// Static sizes known at compile time.
    Static(Vec<usize>),
    /// Runtime sizes determined during execution.
    Runtime(Argument),
}

pub fn resize_config(node: &Node) -> ResizeConfig {
    let mut mode: Option<ResizeMode> = None;

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
            "mode" => {
                mode = Some(
                    value
                        .clone()
                        .into_string()
                        .parse::<ResizeMode>()
                        .expect("Failed to parse resize mode"),
                )
            }
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

    // Extract scales input (3rd input)
    let scales = extract_scales_input(node, input.rank);

    // Extract sizes input (4th input)
    let sizes = extract_sizes_input(node, input.rank);

    let mode = mode.expect("Resize: mode attribute is required");

    if !roi.is_empty() {
        panic!("Resize: roi input is not supported")
    }

    // Check that at least one of scales or sizes is provided
    if scales.is_none() && sizes.is_none() {
        panic!("Resize: either scales or sizes input is required")
    }

    ResizeConfig {
        mode,
        scales,
        sizes,
    }
}

/// Extract scales input as either static or runtime
fn extract_scales_input(node: &Node, input_rank: usize) -> Option<ResizeScales> {
    match node.inputs.get(2) {
        Some(input) => {
            // Skip empty inputs (those with empty names are placeholders)
            if input.name.is_empty() {
                return None;
            }

            match &input.ty {
                ArgType::Tensor(_) => {
                    // Check if it's a constant tensor
                    match &input.value {
                        Some(TensorData { data, .. }) => {
                            let mut scales = data.clone().into_f32s();
                            if scales.is_empty() {
                                return None;
                            }
                            assert!(scales.len() == input_rank);
                            // ignore the first two items from scales
                            // because they are the batch and channel dimensions
                            scales = scales.iter().skip(2).cloned().collect();
                            Some(ResizeScales::Static(scales))
                        }
                        None => Some(ResizeScales::Runtime(input.clone())),
                    }
                }
                ArgType::Shape(_) => {
                    // Shape input for scales - treat as runtime
                    Some(ResizeScales::Runtime(input.clone()))
                }
                _ => None,
            }
        }
        None => None,
    }
}

/// Extract sizes input as either static or runtime
fn extract_sizes_input(node: &Node, input_rank: usize) -> Option<ResizeSizes> {
    match node.inputs.get(3) {
        Some(input) => {
            // Skip empty inputs (those with empty names are placeholders)
            if input.name.is_empty() {
                return None;
            }

            match &input.ty {
                ArgType::Tensor(_) => {
                    // Check if it's a constant tensor
                    match &input.value {
                        Some(TensorData { data, .. }) => {
                            let mut sizes: Vec<usize> = data
                                .clone()
                                .into_i64s()
                                .iter()
                                .map(|&x| x as usize)
                                .collect();
                            if sizes.is_empty() {
                                return None;
                            }
                            assert!(sizes.len() == input_rank);
                            // ignore the first two items from sizes
                            // because they are the batch and channel dimensions
                            sizes = sizes.iter().skip(2).cloned().collect();
                            Some(ResizeSizes::Static(sizes))
                        }
                        None => Some(ResizeSizes::Runtime(input.clone())),
                    }
                }
                ArgType::Shape(_rank) => {
                    // Shape input for sizes - this is the key case we're fixing
                    // The Shape type represents the shape of a tensor, which is exactly what we need
                    Some(ResizeSizes::Runtime(input.clone()))
                }
                _ => None,
            }
        }
        None => None,
    }
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
            // Empty ROI still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_f32("", 1, None);
        }

        // Add scales input if provided
        if let Some(scales_data) = scales {
            builder = builder.input_tensor_f32_data("scales", scales_data.clone(), vec![4]);
            // N,C,H,W scales
        } else {
            // Empty scales still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_f32("", 1, None);
        }

        // Add sizes input if provided
        if let Some(sizes_data) = sizes {
            builder = builder.input_tensor_i64_data("sizes", sizes_data.clone(), vec![4]);
            // N,C,H,W sizes
        } else {
            // Empty sizes still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_i64("", 1, None);
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
        let config = resize_config(&node);
        assert_eq!(config.mode, ResizeMode::Nearest);
        match config.scales {
            Some(ResizeScales::Static(scales)) => {
                assert_eq!(scales, vec![2.0, 2.0]); // Only the spatial scales (H,W)
            }
            _ => panic!("Expected static scales"),
        }
        assert!(config.sizes.is_none(), "Expected no sizes");
    }

    #[test]
    fn test_resize_config_with_sizes() {
        let node = create_test_node(
            "linear",
            None,
            Some(vec![1, 3, 224, 224]), // Fixed output size
            None,
        );
        let config = resize_config(&node);
        assert_eq!(config.mode, ResizeMode::Linear);
        assert!(config.scales.is_none(), "Expected no scales");
        match config.sizes {
            Some(ResizeSizes::Static(sizes)) => {
                assert_eq!(sizes, vec![224, 224]); // Only the spatial sizes (H,W)
            }
            _ => panic!("Expected static sizes"),
        }
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
