use crate::Argument;
use crate::ir::{ArgType, Data, Node, TensorType};

/// Configuration for the Gather operation.
#[derive(Debug, Clone)]
pub struct GatherConfig {
    pub indices: GatherInput,
    pub axis: usize,
}

/// Represents either a static value or a runtime argument for gather indices.
#[derive(Debug, Clone)]
pub enum GatherInput {
    /// Static value known at compile time.
    Static(Vec<i64>),
    /// Runtime argument determined during execution.
    Runtime(Argument),
}

/// Update output rank for Gather based on input and indices ranks.
pub fn gather_update_outputs(node: &mut Node) {
    log::debug!("Gather rank inference for node {}", node.name);

    if node.inputs.len() != 2 {
        panic!("Gather requires two inputs: data and indices");
    }

    let indices_rank = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0,
        ArgType::Shape(_) => 1, // Shape indices become 1D tensors at runtime
    };
    log::debug!("Gather indices rank for {}: {}", node.name, indices_rank);

    match &node.inputs[0].ty {
        ArgType::Tensor(input_tensor) => {
            log::debug!(
                "Gather input tensor rank for {}: {}",
                node.name,
                input_tensor.rank
            );
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_rank + input_tensor.rank - 1;
            log::debug!("Gather output rank for {}: {}", node.name, output_rank);

            if output_rank == 0 {
                // Output is scalar when gathering a single element
                node.outputs[0].ty = ArgType::Scalar(input_tensor.elem_type.clone());
                log::debug!("Gather result for {} is scalar", node.name);
            } else {
                // Output is tensor
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: input_tensor.elem_type.clone(),
                    rank: output_rank,
                    static_shape: None,
                });
                log::debug!(
                    "Gather result for {} is tensor with rank {}",
                    node.name,
                    output_rank
                );
            }
        }
        ArgType::Shape(_shape_rank) => {
            log::debug!("Gather input is shape for {}", node.name);
            // When gathering from a shape:
            // - If indices are scalar (rank 0), output is a scalar (single dimension value)
            // - Otherwise, output is a shape with same rank as indices
            if indices_rank == 0 {
                node.outputs[0].ty = ArgType::Scalar(crate::ir::ElementType::Int64);
                log::debug!("Gather result for {} is scalar (from shape)", node.name);
            } else {
                node.outputs[0].ty = ArgType::Shape(indices_rank);
                log::debug!(
                    "Gather result for {} is shape with rank {} (from shape)",
                    node.name,
                    indices_rank
                );
            }
        }
        ty => panic!("Only tensor/shape input is valid, got {ty:?}"),
    }
}

/// Create a GatherConfig from the attributes of the node
pub fn gather_config(curr: &Node) -> GatherConfig {
    // Default: 0 per ONNX spec
    let mut dim: i64 = 0;

    // check if the node has only one input
    if curr.inputs.len() != 2 {
        panic!("Gather: index tensor must be present");
    }

    // extract the shape of the input tensor
    let input_dim = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor.rank as i64,
        ArgType::Shape(shape_rank) => shape_rank as i64, // Shape dimension
        other => panic!("Only tensor or shape input is valid, got {other:?}"),
    };

    // extract the attributes
    for (key, value) in curr.attrs.iter() {
        if key.as_str() == "axis" {
            dim = value.clone().into_i64()
        }
    }

    // if dim is negative, it is counted from the end
    if dim < 0 {
        dim += input_dim;
    }

    // Get indices input - similar to how slice handles its inputs
    let indices_input = &curr.inputs[1];
    log::debug!(
        "Gather indices input for {}: {:?}",
        curr.name,
        indices_input
    );

    let indices = if let Some(value) = &indices_input.value {
        // Static indices
        log::debug!("Gather {} has static indices value: {:?}", curr.name, value);
        match &value.data {
            Data::Int64s(vals) => {
                log::debug!("Gather {} static indices: {:?}", curr.name, vals);
                GatherInput::Static(vals.clone())
            }
            Data::Int32s(vals) => {
                let int64_vals = vals.iter().map(|&v| v as i64).collect::<Vec<_>>();
                log::debug!(
                    "Gather {} static indices (from int32): {:?}",
                    curr.name,
                    int64_vals
                );
                GatherInput::Static(int64_vals)
            }
            other => panic!("Gather indices must be int32 or int64, got {other:?}"),
        }
    } else {
        // Runtime indices
        log::debug!("Gather {} has runtime indices", curr.name);
        GatherInput::Runtime(indices_input.clone())
    };

    GatherConfig {
        indices,
        axis: dim as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize, is_shape: bool) -> Node {
        // Start building the node with the appropriate input type
        let mut builder = NodeBuilder::new(NodeType::Gather, "test_gather").attr_int("axis", axis);

        if is_shape {
            builder = builder.add_input("data", ArgType::Shape(input_rank));
        } else {
            builder = builder.input_tensor_f32("data", input_rank, None);
        }

        // Add indices and output
        builder = builder
            .input_tensor_i64("indices", 1, None)
            .output_tensor_f32("output", input_rank, None);

        builder.build()
    }

    #[test]
    fn test_gather_config_basic() {
        let node = create_test_node(0, 3, false);
        let config = gather_config(&node);
        assert_eq!(config.axis, 0);
    }

    #[test]
    fn test_gather_config_negative_axis() {
        let node = create_test_node(-2, 3, false);
        let config = gather_config(&node);
        assert_eq!(config.axis, 1); // -2 + 3 = 1
    }

    #[test]
    fn test_gather_config_shape_input() {
        let node = create_test_node(0, 4, true); // Shape of a 4D tensor
        let config = gather_config(&node);
        assert_eq!(config.axis, 0);
    }

    #[test]
    #[should_panic(expected = "Gather: index tensor must be present")]
    fn test_gather_config_missing_index() {
        let mut node = create_test_node(0, 3, false);
        node.inputs.pop(); // Remove the indices input
        let _ = gather_config(&node);
    }

    fn create_runtime_gather_node(axis: i64, input_rank: usize) -> Node {
        let builder = NodeBuilder::new(NodeType::Gather, "test_runtime_gather")
            .attr_int("axis", axis)
            .input_tensor_f32("data", input_rank, None)
            .input_tensor_i64("indices", 1, None) // No static value - runtime input
            .output_tensor_f32("output", input_rank, None);

        builder.build()
    }

    #[test]
    fn test_gather_config_runtime_indices() {
        let node = create_runtime_gather_node(0, 3);
        let config = gather_config(&node);
        assert_eq!(config.axis, 0);

        // Check that indices is runtime
        match config.indices {
            GatherInput::Runtime(arg) => {
                assert_eq!(arg.name, "indices");
            }
            _ => panic!("Expected runtime indices"),
        }
    }

    #[test]
    fn test_gather_config_static_indices() {
        let builder = NodeBuilder::new(NodeType::Gather, "test_static_gather")
            .attr_int("axis", 1)
            .input_tensor_f32("data", 3, None)
            .input_tensor_i64_data("indices", vec![0, 2, 1], vec![3])
            .output_tensor_f32("output", 3, None);

        let node = builder.build();
        let config = gather_config(&node);
        assert_eq!(config.axis, 1);

        // Check that indices is static
        match config.indices {
            GatherInput::Static(vals) => {
                assert_eq!(vals, vec![0, 2, 1]);
            }
            _ => panic!("Expected static indices"),
        }
    }

    #[test]
    fn test_gather_update_outputs_scalar_result() {
        // Test gather with scalar indices on 1D tensor -> scalar output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_scalar_gather")
            .attr_int("axis", 0)
            .input_tensor_f32("data", 1, None)
            .add_input("indices", ArgType::Scalar(crate::ir::ElementType::Int64))
            .output_tensor_f32("output", 1, None)
            .build();

        gather_update_outputs(&mut node);

        // Should output scalar, not tensor
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, crate::ir::ElementType::Float32);
            }
            other => panic!("Expected scalar output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_tensor_result() {
        // Test gather with 1D indices on 2D tensor -> 2D tensor output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_tensor_gather")
            .attr_int("axis", 0)
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("indices", 1, None)
            .output_tensor_f32("output", 2, None)
            .build();

        gather_update_outputs(&mut node);

        // Should output tensor with rank 2 (1 + 2 - 1)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.elem_type, crate::ir::ElementType::Float32);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_shape_indices() {
        // Test gather with Shape indices - this was the bug that caused the original issue
        // Gathering from a shape tensor using shape indices should work correctly
        let mut node = NodeBuilder::new(NodeType::Gather, "test_gather_shape_indices")
            .attr_int("axis", 0)
            .input_shape("data", 3) // Shape input (represents shape of a 3D tensor)
            .add_input("indices", ArgType::Shape(1)) // Shape(1) indices - this was causing the panic
            .output_shape("output", 1) // Output should be Shape(1)
            .build();

        // This should not panic - it was panicking before the fix
        gather_update_outputs(&mut node);

        // Should output Shape(1) since we're gathering from Shape(3) with Shape(1) indices
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1);
            }
            other => panic!("Expected Shape output, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_update_outputs_shape_scalar_indices() {
        // Test gather with scalar indices on shape input -> scalar output
        let mut node = NodeBuilder::new(NodeType::Gather, "test_gather_shape_scalar")
            .attr_int("axis", 0)
            .input_shape("data", 2) // Shape input (represents shape of a 2D tensor)
            .add_input("indices", ArgType::Scalar(crate::ir::ElementType::Int64)) // Scalar indices
            .output_tensor_i64("output", 0, None) // Will be updated by gather_update_outputs
            .build();

        gather_update_outputs(&mut node);

        // Should output scalar when gathering from shape with scalar indices
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected scalar output, got {:?}", other),
        }
    }
}
