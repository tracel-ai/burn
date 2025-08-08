use crate::{Data, Node, TensorData};

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

/// Creates a TriluConfig from the node attributes and inputs.
pub fn trilu_config(node: &Node) -> TriluConfig {
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
        }) = &diagonal_arg.value
    {
        diagonal = *diagonal_val;
    }
    TriluConfig::new(upper, diagonal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    /// Helper function to create test nodes for Trilu tests
    fn create_test_node(upper_attr: Option<i64>, diagonal_input: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Trilu, "test_trilu")
            .input_tensor_f32("X", 2, None) // Typically a matrix
            .output_tensor_f32("Y", 2, None);

        // Add diagonal input if provided
        if let Some(diag) = diagonal_input {
            builder = builder.input_scalar_tensor_i64("k", diag);
        }

        // Add upper attribute if provided
        if let Some(upper) = upper_attr {
            builder = builder.attr_int("upper", upper);
        }

        builder.build()
    }

    #[test]
    fn test_trilu_config_default() {
        // Test with no attributes or inputs - should use defaults (upper=true, diagonal=0)
        let node = create_test_node(None, None);

        let config = trilu_config(&node);

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: 0
            }
        );
    }

    #[test]
    fn test_trilu_config_upper_true() {
        // Test with upper=1 attribute
        let node = create_test_node(Some(1), None);

        let config = trilu_config(&node);

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: 0
            }
        );
    }

    #[test]
    fn test_trilu_config_upper_false() {
        // Test with upper=0 attribute (lower triangular)
        let node = create_test_node(Some(0), None);

        let config = trilu_config(&node);

        assert_eq!(
            config,
            TriluConfig {
                upper: false,
                diagonal: 0
            }
        );
    }

    #[test]
    fn test_trilu_config_with_diagonal() {
        // Test with diagonal=2 input (offset 2 above main diagonal)
        let node = create_test_node(None, Some(2));

        let config = trilu_config(&node);

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: 2
            }
        );
    }

    #[test]
    fn test_trilu_config_with_negative_diagonal() {
        // Test with diagonal=-3 input (offset 3 below main diagonal)
        let node = create_test_node(None, Some(-3));

        let config = trilu_config(&node);

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: -3
            }
        );
    }

    #[test]
    fn test_trilu_config_both_params() {
        // Test with both upper attribute and diagonal input
        let node = create_test_node(Some(0), Some(1));

        let config = trilu_config(&node);

        assert_eq!(
            config,
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
        let node = create_test_node(Some(42), None);

        let config = trilu_config(&node);

        assert_eq!(
            config,
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
        let node = create_test_node(Some(-5), None);

        let config = trilu_config(&node);

        assert_eq!(
            config,
            TriluConfig {
                upper: true,
                diagonal: 0
            }
        );
    }
}
