//! # IsInf
//!
//! Maps infinity to true and other values to false.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__IsInf.html>
//!
//! ## Type Constraints
//! - `T1`: Floating-point tensors (float16, float, double, bfloat16, float8 variants)
//! - `T2`: Boolean tensor
//!
//! ## Opset Versions
//! - **Opset 10-19**: Initial version with detect_negative and detect_positive attributes
//! - **Opset 20+**: Extended type support (added float8 variants)

use derive_new::new;
use onnx_ir_derive::NodeBuilderDerive;

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::ir::{Argument, Node, NodeBuilder};

#[derive(Debug, Clone, PartialEq, Eq, new)]
pub struct IsInfConfig {
    pub detect_negative: bool,
    pub detect_positive: bool,
}

/// Node representation for IsInf operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct IsInfNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: IsInfConfig,
}

pub(crate) struct IsInfProcessor;

impl NodeProcessor for IsInfProcessor {
    type Config = IsInfConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
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
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
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
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::IsInf(IsInfNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(detect_negative: Option<i64>, detect_positive: Option<i64>) -> NodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::IsInf, "test_is_inf")
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(!config.detect_negative);
        assert!(!config.detect_positive);
    }
}
