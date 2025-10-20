//! # Tile
//!
//! Constructs a tensor by tiling the input tensor along specified axes.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Tile.html>
//!
//! ## Attributes
//! None
//!
//! ## Inputs
//! - `input` (T): Input tensor of any shape to be tiled
//! - `repeats` (tensor(int64)): 1D tensor specifying the number of repetitions for each axis.
//!   Must have the same length as the input's rank. Each element indicates how many times
//!   to repeat the input along that dimension.
//!
//! ## Outputs
//! - `output` (T): Output tensor with the same type as input. For each dimension i,
//!   `output_shape[i] = input_shape[i] * repeats[i]`
//!
//! ## Opset Versions
//! - **Opset 1**: Used 'repeats' as an attribute (not supported in this implementation).
//! - **Opset 6**: Changed repeats from attribute to input, enabling dynamic tiling.
//! - **Opset 13**: Added support for bfloat16 and expanded type constraints.
//!
//! **Implementation Note**: This implementation requires opset 6+ (repeats as input).
//!
//! **FIXME**: The implementation does not validate that the repeats tensor has the same length as
//! the input rank, which is required by the ONNX spec. This should be validated in extract_config
//! or infer_types.
//!
//! ## Example
//! Given input = [[1, 2], [3, 4]] with shape (2, 2) and repeats = [1, 2]:
//! Output = [[1, 2, 1, 2], [3, 4, 3, 4]] with shape (2, 4)

use crate::ir::RuntimeInputRef;
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::{Node, NodeConfig};
use std::any::Any;

/// Represents either a static value or a runtime argument for tile repeats.
#[derive(Debug, Clone)]
pub enum TileInput {
    /// Static repeats known at compile time.
    Static(Vec<usize>),
    /// Runtime repeats determined during execution.
    Runtime(RuntimeInputRef),
}

/// Configuration for the Tile operation.
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// The number of times to repeat each dimension.
    pub repeats: TileInput,
}

impl NodeConfig for TileConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct TileProcessor;

impl NodeProcessor for TileProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift repeats input (input[1]) if present
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
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
        // Validate opset
        crate::processor::validate_opset(opset, 6)?;

        // Validate input count (at least data input)
        crate::processor::validate_min_inputs(node, 1)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // Infer output type - same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract repeats config
        fn get_repeats(node: &Node) -> TileInput {
            if let Some(input) = node.inputs.get(1) {
                match input.value() {
                    None => {
                        // Runtime input - store reference instead of cloning the argument
                        TileInput::Runtime(RuntimeInputRef::new(input.name.clone(), 1))
                    }
                    Some(tensor_data) => {
                        let repeats = tensor_data
                            .data
                            .into_i64s()
                            .iter()
                            .map(|&x| x as usize)
                            .collect();
                        TileInput::Static(repeats)
                    }
                }
            } else {
                // No repeats input provided - default to empty
                TileInput::Static(vec![])
            }
        }

        let repeats = get_repeats(node);
        let config = TileConfig { repeats };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    /// Helper function to create test nodes with different repeat values
    fn create_test_node(repeats: Option<Vec<i64>>, input_rank: usize) -> NodeBuilder {
        let mut builder = NodeBuilder::new(NodeType::Tile, "test_tile")
            .input_tensor_f32("input", input_rank, None)
            .output_tensor_f32("output", input_rank, None); // Same rank as input initially

        // Add repeats input if provided
        if let Some(reps) = repeats {
            builder = builder.input_tensor_i64_data("repeats", reps.clone(), vec![reps.len()]);
        }

        builder
    }

    #[test]
    fn test_tile_config_with_repeats() {
        // Test with normal repeats values
        let repeats = vec![2, 3, 4];
        let node = create_test_node(Some(repeats.clone()), 3).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        // Should extract repeats correctly
        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![2, 3, 4]));
    }

    #[test]
    fn test_tile_config_with_single_repeat() {
        // Test with single repeat value
        let repeats = vec![5];
        let node = create_test_node(Some(repeats.clone()), 1).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![5]));
    }

    #[test]
    fn test_tile_config_with_zero_repeats() {
        // Test with repeats including zeros
        let repeats = vec![0, 1, 0];
        let node = create_test_node(Some(repeats.clone()), 3).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![0, 1, 0]));
    }

    #[test]
    fn test_tile_config_with_large_repeats() {
        // Test with large repeats values
        let repeats = vec![100, 200];
        let node = create_test_node(Some(repeats.clone()), 2).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        assert!(matches!(&config.repeats, TileInput::Static(r) if r == &vec![100, 200]));
    }

    #[test]
    fn test_tile_config_without_repeats_input() {
        // Test when repeats input is missing
        let node = create_test_node(None, 3).build();

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        // Should return empty repeats
        assert!(matches!(&config.repeats, TileInput::Static(r) if r.is_empty()));
    }

    #[test]
    fn test_tile_config_with_negative_repeats() {
        // Test with negative repeats values (will be converted to usize)
        let repeats = vec![-1, 2, -3];
        let node = create_test_node(Some(repeats), 3).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        // Negative values get converted to very large positive values due to usize conversion
        // This is expected behavior for this function (though may cause issues elsewhere)
        if let TileInput::Static(r) = &config.repeats {
            assert!(r[0] > 0);
            assert_eq!(r[1], 2);
            assert!(r[2] > 0);
        } else {
            panic!("Expected Static repeats");
        }
    }

    #[test]
    fn test_tile_config_with_empty_repeats() {
        // Test with empty repeats array
        let repeats = vec![];
        let node = create_test_node(Some(repeats), 3).build_with_graph_data(16);

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        assert!(matches!(&config.repeats, TileInput::Static(r) if r.is_empty()));
    }

    #[test]
    fn test_tile_config_with_runtime_repeats() {
        // Test with repeats input that has no static value (runtime)
        let mut node = create_test_node(None, 3).build();

        // Add repeats input with no value
        node.inputs.push(
            NodeBuilder::new(NodeType::Identity, "temp")
                .input_tensor_i64("repeats", 1, Some(vec![3]))
                .build()
                .inputs
                .pop()
                .unwrap(),
        );

        let mut node = node;
        let processor = TileProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TileConfig>();

        // Should return Runtime repeats
        assert!(matches!(&config.repeats, TileInput::Runtime(arg) if arg.name == "repeats"));
    }
}
