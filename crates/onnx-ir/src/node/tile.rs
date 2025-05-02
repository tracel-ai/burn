use crate::{Node, TensorData};

/// Configuration for the Tile operation.
#[derive(Debug, Clone, PartialEq)]
pub struct TileConfig {
    /// The number of times to repeat each dimension.
    pub repeats: Vec<usize>,
}

impl TileConfig {
    pub fn new(repeats: Vec<usize>) -> Self {
        TileConfig { repeats }
    }
}

/// Creates a TileConfig from the node attributes and inputs.
pub fn tile_config(node: &Node) -> TileConfig {
    let repeat = node
        .inputs
        .get(1)
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
    TileConfig::new(repeat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, Data, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    /// Helper function to create test nodes with different repeat values
    fn create_test_node(repeats: Option<Vec<i64>>, input_rank: usize) -> Node {
        let mut inputs = vec![
            // First input: the tensor to tile
            Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
        ];

        // Add repeats input if provided
        if let Some(reps) = repeats {
            inputs.push(Argument {
                name: "repeats".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: Some(vec![reps.len()]),
                }),
                value: Some(TensorData {
                    shape: vec![reps.len()],
                    data: Data::Int64s(reps),
                }),
                passed: true,
            });
        }

        Node {
            node_type: NodeType::Tile,
            name: "test_tile".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank, // Same rank as input initially
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn test_tile_config_with_repeats() {
        // Test with normal repeats values
        let repeats = vec![2, 3, 4];
        let node = create_test_node(Some(repeats.clone()), 3);

        let config = tile_config(&node);

        // Should extract repeats correctly
        assert_eq!(config.repeats, vec![2, 3, 4]);
    }

    #[test]
    fn test_tile_config_with_single_repeat() {
        // Test with single repeat value
        let repeats = vec![5];
        let node = create_test_node(Some(repeats.clone()), 1);

        let config = tile_config(&node);

        assert_eq!(config.repeats, vec![5]);
    }

    #[test]
    fn test_tile_config_with_zero_repeats() {
        // Test with repeats including zeros
        let repeats = vec![0, 1, 0];
        let node = create_test_node(Some(repeats.clone()), 3);

        let config = tile_config(&node);

        assert_eq!(config.repeats, vec![0, 1, 0]);
    }

    #[test]
    fn test_tile_config_with_large_repeats() {
        // Test with large repeats values
        let repeats = vec![100, 200];
        let node = create_test_node(Some(repeats.clone()), 2);

        let config = tile_config(&node);

        assert_eq!(config.repeats, vec![100, 200]);
    }

    #[test]
    fn test_tile_config_without_repeats_input() {
        // Test when repeats input is missing
        let node = create_test_node(None, 3);

        let config = tile_config(&node);

        // Should return empty repeats
        assert_eq!(config.repeats, vec![]);
    }

    #[test]
    fn test_tile_config_with_negative_repeats() {
        // Test with negative repeats values (will be converted to usize)
        let repeats = vec![-1, 2, -3];
        let node = create_test_node(Some(repeats), 3);

        let config = tile_config(&node);

        // Negative values get converted to very large positive values due to usize conversion
        // This is expected behavior for this function (though may cause issues elsewhere)
        assert!(config.repeats[0] > 0);
        assert_eq!(config.repeats[1], 2);
        assert!(config.repeats[2] > 0);
    }

    #[test]
    fn test_tile_config_with_empty_repeats() {
        // Test with empty repeats array
        let repeats = vec![];
        let node = create_test_node(Some(repeats), 3);

        let config = tile_config(&node);

        assert_eq!(config.repeats, vec![]);
    }

    #[test]
    fn test_tile_config_with_missing_value() {
        // Test with repeats input that has no value
        let mut node = create_test_node(None, 3);

        // Add repeats input with no value
        node.inputs.push(Argument {
            name: "repeats".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: Some(vec![3]),
            }),
            value: None, // No value provided
            passed: true,
        });

        let config = tile_config(&node);

        // Should return empty repeats
        assert_eq!(config.repeats, vec![]);
    }
}
