//! # Trilu
//!
//! Returns the upper or lower triangular part of a 2-D matrix or batches of 2-D matrices.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Trilu.html>
//!
//! ## Attributes
//! - `upper` (int, default=1): Boolean indicating whether to return upper (1) or lower (0) triangular part.
//!   Any non-zero value is treated as true.
//!
//! ## Inputs
//! - `input` (T): Input tensor of rank 2 or higher (at least 2D for the last two dimensions).
//! - `k` (tensor(int64), optional): A 0-D tensor containing the diagonal offset:
//!   - 0 = main diagonal (default)
//!   - Positive k = diagonals above main diagonal
//!   - Negative k = diagonals below main diagonal
//!
//! **FIXME**: The implementation does not validate that the input tensor has rank >= 2, which is
//! required by the ONNX spec. This should be validated in infer_types.
//!
//! ## Outputs
//! - `output` (T): Output tensor with the same shape as input, containing the triangular part.
//!   Elements outside the triangle are set to zero.
//!
//! ## Behavior
//! - If `upper=1` (true):
//!   - Positive k: Retains upper triangle excluding main diagonal and (k-1) diagonals above it
//!   - Negative k: Retains upper triangle including main diagonal and |k| diagonals below it
//! - If `upper=0` (false):
//!   - Positive k: Retains lower triangle including main diagonal and k diagonals above it
//!   - Negative k: Retains lower triangle excluding main diagonal and (|k|-1) diagonals below it
//!
//! ## Opset Versions
//! - **Opset 14**: Initial version introducing triangular matrix extraction with optional diagonal offset.

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::{ArgType, Node, NodeConfig, TensorDataExt};

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
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 14,
            max_opset: None,
            inputs: InputSpec::Range(1, 2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift diagonal input (input[1]) if present
        // FIXME: This should check if the input is constant before attempting to lift,
        // similar to other processors. Currently it lifts unconditionally if present.
        // Should use: if node.inputs[1].is_constant() { node.inputs[1].to_static()?; }
        if node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Missing validation that input tensor is at least rank 2.
        // ONNX spec requires last two dimensions to form a matrix, so rank must be >= 2.
        let input_rank = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.rank,
            _ => 0,
        };

        if input_rank < 2 {
            return Err(ProcessError::Custom(format!(
                "Trilu: input must have rank >= 2, got rank {}",
                input_rank
            )));
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
        let mut upper = true;
        let mut diagonal = 0;
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "upper" {
                upper = value.clone().into_i64() != 0
            }
        }
        // The second input of the Trilu node is the diagonal value, coming from a constant node
        // FIXME: The spec states that `k` should be a 0-D tensor (scalar tensor), but the
        // implementation assumes Data::Int64 (scalar value). This should handle the proper
        // tensor extraction with shape validation to ensure it's 0-D.
        // Should validate: k tensor has shape [] or [1] and contains single int64 value.
        if let Some(diagonal_arg) = node.inputs.get(1) {
            if let Some(tensor_data) = diagonal_arg.value() {
                // Extract scalar value, converting from any numeric type to i64
                diagonal = match tensor_data.scalar_i64() {
                    Ok(val) => val,
                    Err(e) => {
                        log::warn!(
                            "Trilu node {}: Failed to extract diagonal value: {:?}",
                            node.name,
                            e
                        );
                        0
                    }
                };
            } else {
                log::warn!(
                    "Trilu node {}: diagonal input has no value (not constant)",
                    node.name
                );
            }
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

    #[test]
    #[ignore] // Manual test
    fn test_parse_actual_trilu_onnx() {
        use crate::pipeline::parse_onnx;
        use std::path::Path;

        let path = Path::new("burn/crates/burn-import/onnx-tests/tests/trilu/trilu_upper.onnx");
        if !path.exists() {
            println!("Skipping test - file not found: {}", path.display());
            return;
        }

        let graph = parse_onnx(path).expect("Failed to parse ONNX file");

        // Print all nodes
        println!("\n=== All nodes ===");
        for (i, node) in graph.nodes.iter().enumerate() {
            println!("Node {}: type={:?}, name={}", i, node.node_type, node.name);
            for (j, input) in node.inputs.iter().enumerate() {
                println!(
                    "  Input {}: name='{}', value_source={:?}",
                    j, input.name, input.value_source
                );
            }
        }

        // Find the Trilu node
        let trilu_node = graph
            .nodes
            .iter()
            .find(|n| matches!(n.node_type, NodeType::Trilu));
        assert!(trilu_node.is_some(), "Trilu node not found");

        let trilu_node = trilu_node.unwrap();
        println!("\n=== Trilu node ===");
        println!("Trilu node: {}", trilu_node.name);

        // Check the config
        let config = trilu_node.config::<TriluConfig>();
        println!(
            "Config: upper={}, diagonal={}",
            config.upper, config.diagonal
        );

        // Should have diagonal=1 according to the ONNX file
        assert_eq!(config.diagonal, 1, "Expected diagonal to be 1");
        assert_eq!(config.upper, true, "Expected upper to be true");
    }

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

    // TODO: Missing test for rank-1 (1D) input - should be rejected as rank must be >= 2.

    // TODO: Missing test for non-square matrices - e.g., shape [3, 5] should work.
    // Trilu works on rectangular matrices, not just square ones.

    // TODO: Missing test for batched inputs - e.g., shape [2, 3, 4, 5].
    // ONNX spec supports batched triangular operations on last two dimensions.

    // TODO: Missing test for very large diagonal offset.
    // E.g., diagonal=100 for 5x5 matrix - should result in all zeros (upper) or all values (lower).

    // TODO: Missing test for diagonal offset equal to matrix dimensions.
    // Edge cases where k = H or k = -W.

    // TODO: Missing test for k input type validation.
    // k must be int64 tensor per spec, should reject other types.

    // TODO: Missing test for k input shape validation.
    // k should be 0-D tensor (scalar), test with 1D tensor [k] to verify handling.
}
