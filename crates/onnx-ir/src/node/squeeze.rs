use crate::ir::{ArgType, Data, Node, TensorType};

pub fn squeeze_config(curr: &Node) -> Vec<i64> {
    let axes = curr
        .attrs
        .iter()
        .filter_map(|(key, value)| {
            if key == "axes" {
                Some(value.clone().into_i64s())
            } else {
                None
            }
        })
        .next()
        .unwrap_or_else(Vec::new);

    // Validate input type (Tensor, Shape, and Scalar are valid)
    match &curr.inputs.first().unwrap().ty {
        ArgType::Tensor(_) | ArgType::Shape(_) | ArgType::Scalar(_) => {}
    };

    axes
}

/// Update output rank for Squeeze based on axes.
pub fn squeeze_update_output(node: &mut Node) {
    log::debug!("Squeeze rank inference for node {}", node.name);

    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(axes) => Some(axes.clone()),
                _ => panic!("Squeeze: invalid input types"),
            },
            None => None,
        }
    } else {
        node.attrs.get("axes").cloned().map(|v| v.into_i64s())
    };

    // If axes is None, ONNX spec squeezes all dimensions of size 1
    // For scalar inputs or Shape inputs, this is handled differently
    let axes = axes.unwrap_or_else(Vec::new);
    log::debug!("Squeeze axes for {}: {:?}", node.name, axes);

    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            log::debug!("Squeeze input rank for {}: {}", node.name, tensor.rank);
            let output_rank = if axes.is_empty() {
                // When axes is empty, ONNX spec squeezes all dimensions of size 1
                // Without static shape info, we can't know which dims are size 1
                // The output type will be corrected later if ONNX provides it
                // TODO: Infer rank from output tensor shape based on static shape inference
                if let Some(ref static_shape) = tensor.static_shape {
                    // Count the number of dimensions not equal to 1
                    static_shape.iter().filter(|&&dim| dim != 1).count()
                } else {
                    panic!(
                        "Squeeze: Cannot infer output rank when axes is empty and input tensor static shape is unknown. Please provide static shape information for accurate inference."
                    );
                }
            } else {
                tensor.rank - axes.len()
            };
            log::debug!("Squeeze output rank for {}: {}", node.name, output_rank);

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: tensor.elem_type.clone(),
                rank: output_rank,
                static_shape: None,
            });
        }
        ArgType::Shape(shape_rank) => {
            log::debug!("Squeeze input is Shape({}) for {}", shape_rank, node.name);

            // Shape is always a 1D array. We can only squeeze axis 0.
            // - If Shape has 1 element (Shape(1)), squeezing axis 0 produces a scalar
            // - If Shape has >1 elements (Shape(n) where n>1), squeezing axis 0 is a no-op
            //   because the dimension has size > 1

            if !axes.is_empty() && (axes.len() != 1 || axes[0] != 0) {
                panic!("Squeeze on Shape input only supports squeezing axis 0, got axes: {axes:?}");
            }

            if *shape_rank == 1 {
                // Shape(1) squeezed on axis 0 produces a scalar
                node.outputs[0].ty = ArgType::Scalar(crate::ir::ElementType::Int64);
                log::debug!("Squeeze Shape(1) to Scalar for {}", node.name);
            } else {
                // Shape(n) where n > 1 remains unchanged
                node.outputs[0].ty = ArgType::Shape(*shape_rank);
                log::debug!("Squeeze Shape({}) unchanged for {}", shape_rank, node.name);
            }
        }
        ArgType::Scalar(scalar_type) => {
            // Scalar squeeze is a no-op
            node.outputs[0].ty = ArgType::Scalar(scalar_type.clone());
            log::debug!("Squeeze Scalar unchanged for {}", node.name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axes: Option<Vec<i64>>, rank: usize) -> Node {
        let output_rank = rank - (axes.as_ref().map_or(0, |a| a.len()));

        let mut builder = NodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("squeezed", output_rank, None);

        if let Some(axes_val) = axes {
            builder = builder.attr_ints("axes", axes_val);
        }

        builder.build()
    }

    #[test]
    fn test_squeeze_config_with_axes() {
        let node = create_test_node(Some(vec![0, 2]), 4);
        let axes = squeeze_config(&node);
        assert_eq!(axes, vec![0, 2]);
    }

    #[test]
    fn test_squeeze_config_no_axes() {
        let node = create_test_node(None, 4);
        let axes = squeeze_config(&node);
        assert!(axes.is_empty());
    }
}
