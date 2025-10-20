//! # IsInf
//!
//! Maps infinity to true and other values to false.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__IsInf.html>
//!
//! ## Attributes
//! - `detect_negative` (int, default=1): Whether to detect negative infinity.
//!   Set to 1 to map negative infinity to true, or 0 to map it to false.
//! - `detect_positive` (int, default=1): Whether to detect positive infinity.
//!   Set to 1 to map positive infinity to true, or 0 to map it to false.
//!
//! ## Inputs
//! - `X` (T1): Input tensor of floating-point type.
//!
//! ## Outputs
//! - `Y` (T2): Output boolean tensor with same shape as input, indicating which elements are infinity.
//!
//! ## Type Constraints
//! - `T1`: Floating-point tensors (float16, float, double, bfloat16, float8 variants)
//! - `T2`: Boolean tensor
//!
//! ## Opset Versions
//! - **Opset 10-19**: Initial version with detect_negative and detect_positive attributes
//! - **Opset 20+**: Extended type support (added float8 variants)

use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use crate::{Node, NodeConfig};
use std::any::Any;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsInfConfig {
    pub detect_negative: bool,
    pub detect_positive: bool,
}

impl IsInfConfig {
    pub fn new(detect_negative: bool, detect_positive: bool) -> Self {
        Self {
            detect_negative,
            detect_positive,
        }
    }
}

impl NodeConfig for IsInfConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct IsInfProcessor;

impl NodeProcessor for IsInfProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 10)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        // Validate unexpected attributes before config extraction
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "detect_negative" | "detect_positive" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for IsInf: {}", key),
                    });
                }
            }
        }

        // Output is boolean tensor with same shape as input
        crate::node::comparison::elementwise_comparison_outputs(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract detect_negative and detect_positive attributes
        let mut detect_negative = true;
        let mut detect_positive = true;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "detect_negative" => detect_negative = value.clone().into_i64() != 0,
                "detect_positive" => detect_positive = value.clone().into_i64() != 0,
                _ => {}
            }
        }

        let config = IsInfConfig::new(detect_negative, detect_positive);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(detect_negative: Option<i64>, detect_positive: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::IsInf, "test_is_inf")
            .input_tensor_f32("data", 4, None)
            .output_tensor_bool("output", 4, None);

        // Add attributes
        if let Some(v) = detect_negative {
            builder = builder.attr_int("detect_negative", v);
        }
        if let Some(v) = detect_positive {
            builder = builder.attr_int("detect_positive", v);
        }

        builder.build()
    }

    #[test]
    fn test_is_inf_config_default() {
        let node = create_test_node(None, None);
        let mut node = node;
        let processor = IsInfProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<IsInfConfig>();

        // Both should default to true if not specified according to the spec
        assert!(config.detect_negative);
        assert!(config.detect_positive);
    }

    #[test]
    fn test_is_inf_only_neg() {
        let node = create_test_node(Some(1), Some(0));
        let mut node = node;
        let processor = IsInfProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<IsInfConfig>();

        assert!(config.detect_negative);
        assert!(!config.detect_positive);
    }

    #[test]
    fn test_is_inf_only_pos() {
        let node = create_test_node(Some(0), Some(1));
        let mut node = node;
        let processor = IsInfProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<IsInfConfig>();

        assert!(!config.detect_negative);
        assert!(config.detect_positive);
    }

    #[test]
    fn test_is_inf_detect_none() {
        let node = create_test_node(Some(0), Some(0));
        let mut node = node;
        let processor = IsInfProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<IsInfConfig>();

        assert!(!config.detect_negative);
        assert!(!config.detect_positive);
    }
}
