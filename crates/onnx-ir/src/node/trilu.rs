use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use crate::{Data, Node, NodeConfig, TensorData};
use std::any::Any;

/// Configuration for the Trilu operation.
#[derive(Debug, Clone, PartialEq)]
pub struct TriluConfig {
    /// Whether to return the upper triangular matrix.
    pub upper: bool,
    /// The diagonal offset.
    pub diagonal: i64,
}

impl TriluConfig {
    /// Creates a TriluConfig from the node attributes and inputs.
    pub fn new(upper: bool, diagonal: i64) -> Self {
        Self { upper, diagonal }
    }
}

impl NodeConfig for TriluConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct TriluProcessor;

impl NodeProcessor for TriluProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift diagonal input (input[1]) if present
        if node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Trilu implementation supports opset 14+
        crate::processor::validate_opset(opset, 14)?;

        // Validate input count (1 or 2 inputs)
        crate::processor::validate_min_inputs(node, 1)?;
        if node.inputs.len() > 2 {
            return Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: node.inputs.len(),
            });
        }

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut upper = true;
        let mut diagonal = 0;
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "upper" {
                upper = value.clone().into_i64() != 0
            }
        }
        // The second input of the Trilu node is the diagonal value, coming from a constant node
        if let Some(diagonal_arg) = node.inputs.get(1)
            && let Some(TensorData {
                data: Data::Int64(diagonal_val),
                ..
            }) = &diagonal_arg.value()
        {
            diagonal = *diagonal_val;
        }

        let config = TriluConfig::new(upper, diagonal);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    /// Helper function to create test nodes for Trilu tests
    fn create_test_node(upper_attr: Option<i64>, diagonal_input: Option<i64>) -> NodeBuilder {
        let mut builder = NodeBuilder::new(NodeType::Trilu, "test_trilu")
            .input_tensor_f32("X", 2, None) // Typically a matrix
            .output_tensor_f32("Y", 2, None);

        // Add diagonal input if provided
        if let Some(diag) = diagonal_input {
            builder = builder.input_scalar_tensor_i64("k", Some(diag));
        }

        // Add upper attribute if provided
        if let Some(upper) = upper_attr {
            builder = builder.attr_int("upper", upper);
        }

        builder
    }

    #[test]
    fn test_trilu_config_default() {
        // Test with no attributes or inputs - should use defaults (upper=true, diagonal=0)
        let node = create_test_node(None, None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: true,
                diagonal: 0
            }
        );
    }

    #[test]
    fn test_trilu_config_upper_true() {
        // Test with upper=1 attribute
        let node = create_test_node(Some(1), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: true,
                diagonal: 0
            }
        );
    }

    #[test]
    fn test_trilu_config_upper_false() {
        // Test with upper=0 attribute (lower triangular)
        let node = create_test_node(Some(0), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: false,
                diagonal: 0
            }
        );
    }

    #[test]
    fn test_trilu_config_with_diagonal() {
        // Test with diagonal=2 input (offset 2 above main diagonal)
        let node = create_test_node(None, Some(2)).build_with_graph_data(16);

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: true,
                diagonal: 2
            }
        );
    }

    #[test]
    fn test_trilu_config_with_negative_diagonal() {
        // Test with diagonal=-3 input (offset 3 below main diagonal)
        let node = create_test_node(None, Some(-3)).build_with_graph_data(16);

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: true,
                diagonal: -3
            }
        );
    }

    #[test]
    fn test_trilu_config_both_params() {
        // Test with both upper attribute and diagonal input
        let node = create_test_node(Some(0), Some(1)).build_with_graph_data(16);

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: false,
                diagonal: 1
            }
        );
    }

    #[test]
    fn test_trilu_config_non_binary_upper() {
        // Test with non-binary values for the upper attribute
        // Any non-zero value should be treated as true
        let node = create_test_node(Some(42), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: true,
                diagonal: 0
            }
        );
    }

    #[test]
    fn test_trilu_config_negative_non_binary_upper() {
        // Test with negative values for the upper attribute
        // Any non-zero value should be treated as true
        let node = create_test_node(Some(-5), None).build();

        let mut node = node;
        let processor = TriluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TriluConfig>();

        assert_eq!(
            *config,
            TriluConfig {
                upper: true,
                diagonal: 0
            }
        );
    }
}
