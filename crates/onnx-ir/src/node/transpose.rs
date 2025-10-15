use crate::ir::{ArgType, Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::util::same_as_input;
use std::any::Any;

/// Configuration for Transpose operations
#[derive(Debug, Clone)]
pub struct TransposeConfig {
    /// Permutation of dimensions
    pub perm: Vec<i64>,
}

impl NodeConfig for TransposeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct TransposeProcessor;

impl NodeProcessor for TransposeProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::util::validate_opset(opset, 1)?;

        // Validate input count
        crate::util::validate_input_count(node, 1)?;

        // Validate output count
        crate::util::validate_output_count(node, 1)?;

        // Get reference to config for type inference
        let _config = node.config::<TransposeConfig>();

        // Infer output type
        same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Default: reverse the dimensions
        let mut perm = (0..tensor.rank as i64).rev().collect::<Vec<i64>>();

        if let Some(axes) = node.attrs.get("perm") {
            perm = axes.clone().into_i64s();
        }

        let config = TransposeConfig { perm };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(perm: Option<Vec<i64>>, rank: usize) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Transpose, "test_transpose")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("transposed", rank, None);

        if let Some(perm_val) = perm {
            builder = builder.attr_ints("perm", perm_val);
        }

        builder.build()
    }

    #[test]
    fn test_transpose_config_default() {
        let node = create_test_node(None, 3);
        let mut node = node;
        let processor = TransposeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TransposeConfig>();
        assert_eq!(config.perm, vec![2, 1, 0]); // Default is to reverse the dimensions
    }

    #[test]
    fn test_transpose_config_with_perm() {
        let node = create_test_node(Some(vec![0, 2, 1]), 3);
        let mut node = node;
        let processor = TransposeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TransposeConfig>();
        assert_eq!(config.perm, vec![0, 2, 1]);
    }

    #[test]
    fn test_transpose_config_multiple_inputs() {
        let mut node = create_test_node(None, 3);
        // Add an extra input to cause the expected error
        node.inputs.push(crate::ir::Argument {
            name: "extra".to_string(),
            ty: crate::ir::ArgType::Tensor(crate::ir::TensorType {
                elem_type: crate::ir::ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value_store: None,
        });
        let mut node = node;
        let processor = TransposeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }
}
