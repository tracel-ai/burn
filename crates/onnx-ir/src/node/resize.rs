//! # Resize
//!
//! Resizes input tensor using various interpolation methods.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Resize.html>
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version with scales and sizes inputs.
//! - **Opset 11**: Added coordinate_transformation_mode attribute for more control over interpolation. Added support for linear mode (previously only nearest).
//! - **Opset 13**: Added cubic mode support and cubic_coeff_a attribute. Added antialias attribute for downsampling.
//! - **Opset 18**: Added keep_aspect_ratio_policy and axes attributes for selective resizing.
//! - **Opset 19**: Added antialiasing improvements and clarified coordinate transformation modes.
//!
//! **Implementation Note**: This implementation requires opset 11+ for coordinate transformation mode support. Many attributes are ignored or have restricted values (see validation in infer_types).

use crate::ir::{ArgType, Node, NodeConfig, RuntimeInputRef};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use std::any::Any;
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
    /// Coordinate transformation mode (default: "half_pixel")
    pub coordinate_transformation_mode: String,
    /// Cubic coefficient for cubic interpolation (default: -0.75)
    pub cubic_coeff_a: f32,
    /// Nearest mode rounding strategy (default: "round_prefer_floor")
    pub nearest_mode: String,
    /// Exclude outside weights (default: 0)
    pub exclude_outside: i32,
    /// Extrapolation value for tf_crop_and_resize mode (default: 0.0)
    pub extrapolation_value: f32,
    /// Antialias flag (default: 0) - opset 13+
    pub antialias: i32,
}

impl NodeConfig for ResizeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Represents either a static value or a runtime argument for resize scales.
#[derive(Debug, Clone)]
pub enum ResizeScales {
    /// Static scales known at compile time.
    Static(Vec<f32>),
    /// Runtime scales determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Represents either a static value or a runtime argument for resize sizes.
#[derive(Debug, Clone)]
pub enum ResizeSizes {
    /// Static sizes known at compile time.
    Static(Vec<usize>),
    /// Runtime sizes determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Extract scales input as either static or runtime
fn extract_scales_input(node: &Node, input_rank: usize) -> Option<ResizeScales> {
    match node.inputs.get(2) {
        Some(input) => {
            // Skip optional inputs (those that were never provided)
            if input.is_optional() {
                return None;
            }

            match &input.ty {
                ArgType::Tensor(_) => {
                    // Check if it's a static value (lifted constant) or constant
                    match input.value() {
                        Some(tensor_data) => {
                            let mut scales: Vec<f32> = tensor_data.to_vec().unwrap();
                            if scales.is_empty() {
                                return None;
                            }
                            assert!(scales.len() == input_rank);
                            // ignore the first two items from scales
                            // because they are the batch and channel dimensions
                            scales = scales.iter().skip(2).cloned().collect();
                            Some(ResizeScales::Static(scales))
                        }
                        None => {
                            // Runtime input - store reference instead of cloning the argument
                            Some(ResizeScales::Runtime(RuntimeInputRef::new(
                                input.name.clone(),
                                2,
                            )))
                        }
                    }
                }
                ArgType::Shape(_) => {
                    // Shape input for scales - store reference instead of cloning the argument
                    Some(ResizeScales::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        2,
                    )))
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
            // Skip optional inputs (those that were never provided)
            if input.is_optional() {
                return None;
            }

            match &input.ty {
                ArgType::Tensor(_) => {
                    // Check if it's a static value (lifted constant) or constant
                    match input.value() {
                        Some(tensor_data) => {
                            let i64_sizes: Vec<i64> = tensor_data.to_vec().unwrap();
                            let mut sizes: Vec<usize> =
                                i64_sizes.iter().map(|&x| x as usize).collect();
                            if sizes.is_empty() {
                                return None;
                            }
                            assert!(sizes.len() == input_rank);
                            // ignore the first two items from sizes
                            // because they are the batch and channel dimensions
                            sizes = sizes.iter().skip(2).cloned().collect();
                            Some(ResizeSizes::Static(sizes))
                        }
                        None => {
                            // Runtime input - store reference instead of cloning the argument
                            Some(ResizeSizes::Runtime(RuntimeInputRef::new(
                                input.name.clone(),
                                3,
                            )))
                        }
                    }
                }
                ArgType::Shape(_rank) => {
                    // Shape input for sizes - store reference instead of cloning the argument
                    // The Shape type represents the shape of a tensor, which is exactly what we need
                    Some(ResizeSizes::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        3,
                    )))
                }
                _ => None,
            }
        }
        None => None,
    }
}

pub struct ResizeProcessor;

impl NodeProcessor for ResizeProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Range(1, 4),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift roi input (input[1]) if present and constant
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Lift scales input (input[2]) if present and constant
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        // Lift sizes input (input[3]) if present and constant
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add maximum input count validation
        // Spec allows 1-4 inputs (X, roi, scales, sizes). Should validate max 4 inputs.
        // Location: After validate_min_inputs

        // TODO: Missing validation for scales and sizes both provided
        // Spec states: "Either 'scales' or 'sizes' MUST be provided, MUST NOT provide both."
        // Implementation checks at least one is provided (line 331-335) but doesn't reject both.
        // Should add validation: if both scales and sizes are non-None, return error.
        // Location: After extracting config, before checking at least one exists

        // TODO: Missing test coverage for cubic mode
        // ResizeMode::Cubic is defined and parsed but no test validates cubic interpolation.
        // Add test: resize_cubic_mode

        // TODO: Missing test coverage for different coordinate_transformation_mode values
        // Implementation accepts and stores coordinate_transformation_mode but doesn't validate values.
        // Spec defines: "half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric", "tf_crop_and_resize"
        // Tests don't verify different modes produce different/correct results.
        // Add tests: resize_align_corners, resize_asymmetric, resize_tf_crop_and_resize

        // TODO: Missing test coverage for nearest_mode variations
        // Spec defines nearest_mode: "round_prefer_floor", "round_prefer_ceil", "floor", "ceil"
        // No tests validate different rounding behaviors.
        // Add tests: resize_nearest_floor, resize_nearest_ceil

        // TODO: Missing test coverage for keep_aspect_ratio_policy (opset 18+)
        // Attribute is validated to reject non-"stretch" values (lines 291-299) but no test.
        // Add test: resize_keep_aspect_ratio_not_stretch (should fail)

        // TODO: Missing test coverage for axes attribute (opset 18+)
        // Attribute is validated to reject custom axes (lines 262-266) but no test.
        // Add test: resize_custom_axes (should fail)

        // FIXME: ROI input support is missing
        // Implementation explicitly rejects non-empty ROI (lines 309-325) but ROI is part of spec.
        // ROI (Region of Interest) allows resizing only a subregion. This is a spec deviation.
        // Impact: MEDIUM - Models using ROI-based resize will fail.
        // Either implement ROI support or clearly document limitation.

        // TODO: Missing test coverage for 1D and 3D tensors
        // Tests only cover 4D tensors (N,C,H,W). Spec supports any rank >= 1.
        // Add tests: resize_1d, resize_3d, resize_5d

        // TODO: Missing test coverage for antialias attribute validation
        // antialias is validated to reject non-zero (lines 254-260) but no test.
        // Add test: resize_antialias_enabled (should fail with current implementation)

        // Note: we are ignoring some attributes because results are approximately the same
        // and we are not supporting all the attributes of the Resize operator.
        // However, some attributes are important to be checked and we are checking
        // against the default values of the attributes.
        // TODO revisit this when we have more Resize operators in the model

        // TODO: Missing validation for coordinate_transformation_mode values
        // Implementation stores coordinate_transformation_mode string but doesn't validate it
        // against spec-defined values. Invalid mode strings are accepted silently.
        // Should validate mode is one of: "half_pixel", "pytorch_half_pixel", "align_corners",
        // "asymmetric", "tf_crop_and_resize", "tf_half_pixel_for_nn"
        // Location: extract_config method after extracting coordinate_transformation_mode
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "antialias" => {
                    if value.clone().into_i32() != 0 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "antialias".to_string(),
                            reason: "antialias other than 0 is not supported".to_string(),
                        });
                    }
                }
                "axes" => {
                    return Err(ProcessError::InvalidAttribute {
                        name: "axes".to_string(),
                        reason: "custom axes attribute is not supported".to_string(),
                    });
                }
                "coordinate_transformation_mode" => {
                    // Ignored: approximate results are acceptable
                }
                "cubic_coeff_a" => {
                    // Ignored: approximate results are acceptable
                }
                "exclude_outside" => {
                    if value.clone().into_i32() != 0 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "exclude_outside".to_string(),
                            reason: "exclude_outside other than 0 is not supported".to_string(),
                        });
                    }
                }
                "extrapolation_value" => {
                    if value.clone().into_f32() != 0.0 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "extrapolation_value".to_string(),
                            reason: "extrapolation_value other than 0.0 is not supported"
                                .to_string(),
                        });
                    }
                }
                "keep_aspect_ratio_policy" => {
                    if value.clone().into_string().to_lowercase() != "stretch" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "keep_aspect_ratio_policy".to_string(),
                            reason:
                                "keep_aspect_ratio_policy other than 'stretch' is not supported"
                                    .to_string(),
                        });
                    }
                }
                "mode" => {} // Validated in extract_config
                "nearest_mode" => {
                    // Ignored: approximate results are acceptable
                }
                _ => {}
            }
        }

        let roi: Vec<f32> = node
            .inputs
            .get(1)
            .map(|input| {
                if let Some(tensor_data) = input.value() {
                    tensor_data.to_vec().unwrap()
                } else {
                    vec![]
                }
            })
            .unwrap_or_default();

        if !roi.is_empty() {
            return Err(ProcessError::Custom(
                "Resize: roi input is not supported".to_string(),
            ));
        }

        // Get reference to config for validation
        let config = node.config::<ResizeConfig>();

        // Check that at least one of scales or sizes is provided
        if config.scales.is_none() && config.sizes.is_none() {
            return Err(ProcessError::Custom(
                "Resize: either scales or sizes input is required".to_string(),
            ));
        }

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut mode: Option<ResizeMode> = None;
        let mut coordinate_transformation_mode = "half_pixel".to_string();
        let mut cubic_coeff_a = -0.75f32;
        let mut nearest_mode = "round_prefer_floor".to_string();
        let mut exclude_outside = 0i32;
        let mut extrapolation_value = 0.0f32;
        let mut antialias = 0i32;

        let input = if let ArgType::Tensor(tensor) = &node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("input".to_string()))?
            .ty
        {
            tensor
        } else {
            return Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", node.inputs.first().unwrap().ty),
            });
        };

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "mode" => {
                    mode = Some(
                        value
                            .clone()
                            .into_string()
                            .parse::<ResizeMode>()
                            .map_err(|e| ProcessError::InvalidAttribute {
                                name: "mode".to_string(),
                                reason: format!("Failed to parse resize mode: {}", e),
                            })?,
                    )
                }
                "coordinate_transformation_mode" => {
                    coordinate_transformation_mode = value.clone().into_string();
                }
                "cubic_coeff_a" => {
                    cubic_coeff_a = value.clone().into_f32();
                }
                "nearest_mode" => {
                    nearest_mode = value.clone().into_string();
                }
                "exclude_outside" => {
                    exclude_outside = value.clone().into_i32();
                }
                "extrapolation_value" => {
                    extrapolation_value = value.clone().into_f32();
                }
                "antialias" => {
                    antialias = value.clone().into_i32();
                }
                _ => {}
            }
        }

        // Extract scales input (3rd input)
        let scales = extract_scales_input(node, input.rank);

        // Extract sizes input (4th input)
        let sizes = extract_sizes_input(node, input.rank);

        let mode = mode.ok_or_else(|| ProcessError::MissingAttribute("mode".to_string()))?;

        let config = ResizeConfig {
            mode,
            scales,
            sizes,
            coordinate_transformation_mode,
            cubic_coeff_a,
            nearest_mode,
            exclude_outside,
            extrapolation_value,
            antialias,
        };
        Ok(Some(Box::new(config)))
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
    ) -> NodeBuilder {
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

        builder
    }

    #[test]
    fn test_resize_config_with_scales() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]), // Keep N,C same, double H,W
            None,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ResizeConfig>();
        assert_eq!(config.mode, ResizeMode::Nearest);
        match &config.scales {
            Some(ResizeScales::Static(scales)) => {
                assert_eq!(*scales, vec![2.0, 2.0]); // Only the spatial scales (H,W)
            }
            _ => panic!("Expected static scales"),
        }
        assert!(config.sizes.is_none(), "Expected no sizes");
        // Verify default attribute values
        assert_eq!(config.coordinate_transformation_mode, "half_pixel");
        assert_eq!(config.cubic_coeff_a, -0.75);
        assert_eq!(config.nearest_mode, "round_prefer_floor");
        assert_eq!(config.exclude_outside, 0);
        assert_eq!(config.extrapolation_value, 0.0);
        assert_eq!(config.antialias, 0);
    }

    #[test]
    fn test_resize_config_with_sizes() {
        let node = create_test_node(
            "linear",
            None,
            Some(vec![1, 3, 224, 224]), // Fixed output size
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ResizeConfig>();
        assert_eq!(config.mode, ResizeMode::Linear);
        assert!(config.scales.is_none(), "Expected no scales");
        match &config.sizes {
            Some(ResizeSizes::Static(sizes)) => {
                assert_eq!(*sizes, vec![224, 224]); // Only the spatial sizes (H,W)
            }
            _ => panic!("Expected static sizes"),
        }
    }

    #[test]
    fn test_resize_config_with_roi() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]),
            None,
            Some(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), // ROI values
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_config_no_scales_or_sizes() {
        let node = create_test_node("nearest", None, None, None).build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_config_no_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None)
            .build_with_graph_data(16);
        node.attrs.clear(); // Remove all attributes including mode
        let node = node;
        let processor = ResizeProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::MissingAttribute(_))));
    }
}
