use crate::ir::{ArgType, Data, Node, TensorData};

/// Creates a configuration for tensor slicing based on the ONNX Slice operator.
/// Returns a vector of optional ranges representing start and end indices for each dimension.
///
/// Note: we leave the negative indices as is, but we need to handle them properly when slicing
/// during the actual slicing operation using the dynamic shape information.
pub fn slice_config(node: &Node) -> Vec<Option<(i64, i64)>> {
    /// Extracts int64 values from a node's input at the specified index.
    /// Returns an empty vector if the input is not provided.
    fn get_input_values(node: &Node, index: usize) -> Vec<i64> {
        if node.inputs.get(index).is_none() {
            return Vec::new();
        }

        match &node.inputs[index].value {
            Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) => shape.clone(),

            _ => panic!("Tensor data type must be int64"),
        }
    }

    let mut starts = get_input_values(node, 1);
    let mut ends = get_input_values(node, 2);
    let mut axes = get_input_values(node, 3);
    let mut steps = get_input_values(node, 4);

    // Reference: https://burn.dev/docs/burn/prelude/struct.Tensor.html#method.slice
    // TODO: Default missing axes ranges to the full range of the corresponding axis
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "starts" => starts = value.clone().into_i64s(),
            "ends" => ends = value.clone().into_i64s(),
            "axes" => axes = value.clone().into_i64s(),
            "steps" => steps = value.clone().into_i64s(),
            _ => {}
        }
    }

    if !steps.is_empty() && steps.iter().any(|&x| x != 1) {
        panic!("Slice: steps other than 1 are not supported");
    }

    // Extract the rank of the input tensor
    let input_rank = match node.inputs.first().unwrap().clone().ty {
        crate::ir::ArgType::Tensor(tensor) => tensor.rank,
        crate::ir::ArgType::Shape(_) => 1,
        _ => panic!("Only tensor input is valid"),
    };

    // Default to all axes if not specified
    if axes.is_empty() {
        axes = (0..starts.len() as i64).collect();
    }

    // Validate input dimensions
    if starts.len() != ends.len() || starts.len() != axes.len() {
        panic!("Slice: starts, ends, and axes must have the same length");
    }

    // Convert negative axes indices to positive (counting from the end)
    for axis in &mut axes {
        if *axis < 0 {
            *axis += input_rank as i64;
        }
    }

    // Create ranges vector with None for dimensions not being sliced
    let mut ranges: Vec<Option<(i64, i64)>> = vec![None; input_rank];
    for i in 0..axes.len() {
        let axis = axes[i] as usize;
        ranges[axis] = Some((starts[i], ends[i]));
    }

    ranges
}

/// Update output type for Slice operation.
/// If the input is a Tensor, the output type remains the same.
/// If the input is a Shape, the output becomes a rank-1 Int64 Tensor representing the sliced dimension.
pub fn slice_update_output_rank(node: &mut Node) {
    log::debug!("Slice rank inference for node {}", node.name);

    match &node.inputs[0].ty {
        ArgType::Tensor(_) => {
            // Slicing a tensor preserves its type and rank during rank inference.
            // Shape inference pass will handle the actual shape changes.
            log::debug!("Slice input for {} is Tensor, preserving type", node.name);
            node.outputs[0].ty = node.inputs[0].ty.clone();
        }
        ArgType::Shape(_) => {
            // Slicing a Shape extracts a sub-part, resulting in a rank-1 Tensor.
            log::debug!("Slice input for {} is Shape", node.name);
            let config = slice_config(node);
            assert_eq!(
                config.len(),
                1,
                "Slice on Shape input requires exactly one dimension slice config for node {}",
                node.name
            );

            let (start, end) = config[0].unwrap_or_else(|| {
                panic!(
                    "Slice config for Shape input must contain start and end indices for node {}",
                    node.name
                )
            });

            let output_len = end as usize - start as usize;

            node.outputs[0].ty = ArgType::Shape(output_len);
        }
        // Handle unsupported input types
        unsupported_type => {
            panic!(
                "Slice: Only Tensor and Shape inputs are supported for node {}, got {:?}",
                node.name, unsupported_type
            )
        }
    }

    log::debug!(
        "Slice output type determined for {}: {:?}",
        node.name,
        node.outputs[0].ty
    );
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorType};

    use super::*;

    fn create_test_node(
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Option<Vec<i64>>,
        use_attrs: bool,
    ) -> Node {
        let mut inputs = vec![Argument {
            name: "data".to_string(),
            ty: crate::ir::ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        if !use_attrs {
            // Add inputs as tensors
            inputs.push(Argument {
                name: "starts".to_string(),
                ty: crate::ir::ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: Some(vec![starts.len()]),
                }),
                value: Some(TensorData {
                    data: Data::Int64s(starts.clone()),
                    shape: vec![starts.len()],
                }),
                passed: true,
            });

            inputs.push(Argument {
                name: "ends".to_string(),
                ty: crate::ir::ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: Some(vec![ends.len()]),
                }),
                value: Some(TensorData {
                    data: Data::Int64s(ends.clone()),
                    shape: vec![ends.len()],
                }),
                passed: true,
            });

            if let Some(axes_vec) = &axes {
                inputs.push(Argument {
                    name: "axes".to_string(),
                    ty: crate::ir::ArgType::Tensor(TensorType {
                        elem_type: ElementType::Int64,
                        rank: 1,
                        static_shape: Some(vec![axes_vec.len()]),
                    }),
                    value: Some(TensorData {
                        data: Data::Int64s(axes_vec.clone()),
                        shape: vec![axes_vec.len()],
                    }),
                    passed: true,
                });
            }
        }

        let mut attrs = HashMap::new();
        if use_attrs {
            attrs.insert("starts".to_string(), AttributeValue::Int64s(starts));
            attrs.insert("ends".to_string(), AttributeValue::Int64s(ends));
            if let Some(axes_vec) = axes {
                attrs.insert("axes".to_string(), AttributeValue::Int64s(axes_vec));
            }
        }

        Node {
            node_type: NodeType::Slice,
            name: "test_slice".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    fn create_shape_input_node(start: i64, end: i64) -> Node {
        let mut node = Node {
            node_type: NodeType::Slice,
            name: "test_slice_shape".to_string(),
            inputs: vec![Argument {
                name: "data".to_string(),
                ty: ArgType::Shape(5), // 1-dimensional shape (important: matches what the tests expect)
                value: None,
                passed: true,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                value: None,
                passed: true,
            }],
            attrs: HashMap::new(),
        };

        // Add starts and ends as tensors
        node.inputs.push(Argument {
            name: "starts".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: Some(vec![1]),
            }),
            value: Some(TensorData {
                data: Data::Int64s(vec![start]),
                shape: vec![1],
            }),
            passed: true,
        });

        node.inputs.push(Argument {
            name: "ends".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: Some(vec![1]),
            }),
            value: Some(TensorData {
                data: Data::Int64s(vec![end]),
                shape: vec![1],
            }),
            passed: true,
        });

        // Add axes tensor to specify dimension 0
        node.inputs.push(Argument {
            name: "axes".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: Some(vec![1]),
            }),
            value: Some(TensorData {
                data: Data::Int64s(vec![0]),
                shape: vec![1],
            }),
            passed: true,
        });

        node
    }

    #[test]
    fn test_slice_config_basic() {
        // Create a node with inputs for basic slicing
        let node = create_test_node(vec![1, 0], vec![3, 2], Some(vec![0, 2]), false);

        let result = slice_config(&node);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some((1, 3)));
        assert_eq!(result[1], None);
        assert_eq!(result[2], Some((0, 2)));
    }

    #[test]
    fn test_slice_config_with_attrs() {
        // Create a node with attributes instead of inputs
        let node = create_test_node(vec![1, 0], vec![3, 2], Some(vec![0, 2]), true);

        let result = slice_config(&node);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some((1, 3)));
        assert_eq!(result[1], None);
        assert_eq!(result[2], Some((0, 2)));
    }

    #[test]
    fn test_slice_config_negative_axes() {
        // Test with negative axes values
        let node = create_test_node(vec![1], vec![3], Some(vec![-3]), false);

        let result = slice_config(&node);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some((1, 3))); // -3 -> 0 (first dimension)
        assert_eq!(result[1], None);
        assert_eq!(result[2], None);
    }

    #[test]
    fn test_slice_config_default_axes() {
        // Test the default axes behavior (when axes input is not provided)
        let node = create_test_node(vec![1, 2], vec![3, 4], None, false);

        let result = slice_config(&node);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Some((1, 3)));
        assert_eq!(result[1], Some((2, 4)));
        assert_eq!(result[2], None);
    }

    #[test]
    fn test_slice_update_output_rank_tensor_input() {
        // Test when input is a Tensor - output should preserve the same type
        let mut node = create_test_node(vec![1, 2], vec![3, 4], None, false);

        // Before calling, input is Tensor and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Tensor(_)));
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(_)));

        slice_update_output_rank(&mut node);

        // After calling, output should be the same type as input
        assert!(
            matches!(&node.outputs[0].ty, ArgType::Tensor(tensor_type) if tensor_type.elem_type == ElementType::Float32 && tensor_type.rank == 3)
        );
    }

    #[test]
    fn test_slice_update_output_rank_shape_input() {
        // Test when input is a Shape - output should be a rank-1 Int64 Tensor
        let mut node = create_shape_input_node(1, 3);

        // Before calling, input is Shape and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Shape(5)));
        // Default output type is Tensor with rank 0
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(ref t) if t.rank == 0));

        slice_update_output_rank(&mut node);

        // After calling, output should be ArgType::Shape with the calculated length
        // start = 1, end = 3 => output_len = 3 - 1 = 2
        assert!(matches!(&node.outputs[0].ty, ArgType::Shape(2)));
    }
}
