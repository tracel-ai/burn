use crate::ir::{ArgType, Node, NodeConfig};
use crate::processor::NodeProcessor;
use crate::util::{same_as_input, validate_opset};
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
    fn process_config(&self, node: &mut Node, opset: usize) {
        // Transpose implementation supports opset 1+
        validate_opset(&node.node_type, opset, 1);

        // ALL logic from transpose_config inlined here
        if node.inputs.len() != 1 {
            panic!(
                "Transpose: multiple inputs are not supported (got {:?})",
                node.inputs.len()
            );
        }

        // Extract the shape of the input tensor
        let tensor = match node.inputs.first().unwrap().clone().ty {
            ArgType::Tensor(tensor) => tensor,
            _ => panic!("Only tensor input is valid"),
        };

        // Default: reverse the dimensions
        let mut perm = (0..tensor.rank as i64).rev().collect::<Vec<i64>>();

        if let Some(axes) = node.attrs.get("perm") {
            perm = axes.clone().into_i64s();
        }

        let config = TransposeConfig { perm };
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        same_as_input(node);
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
        processor.process_config(&mut node, 16);
        let config = node.config::<TransposeConfig>();
        assert_eq!(config.perm, vec![2, 1, 0]); // Default is to reverse the dimensions
    }

    #[test]
    fn test_transpose_config_with_perm() {
        let node = create_test_node(Some(vec![0, 2, 1]), 3);
        let mut node = node;
        let processor = TransposeProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<TransposeConfig>();
        assert_eq!(config.perm, vec![0, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "Transpose: multiple inputs are not supported")]
    fn test_transpose_config_multiple_inputs() {
        let mut node = create_test_node(None, 3);
        // Add an extra input to cause the expected panic
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
        processor.process_config(&mut node, 16);
    }
}
