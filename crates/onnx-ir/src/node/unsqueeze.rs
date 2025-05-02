use crate::{
    Argument, TensorData,
    ir::{ArgType, Data, Node, TensorType},
};

/// Update output rank for Unsqueeze based on axes.
/// Update the output tensor dimension based on the "axes" attribute or the second input
pub fn unsqueeze_update_output(node: &mut Node) {
    log::debug!("Unsqueeze rank inference for node {}", node.name);

    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(a) => Some(a.clone()),
                _ => panic!("Unsqueeze: invalid input types"),
            },
            None => None,
        }
    } else {
        let axes = node.attrs.get("axes").cloned().map(|v| {
            let axes = v.into_i64s();
            log::debug!(
                "Unsqueeze axes from attribute for {}: {:?}",
                node.name,
                axes
            );
            axes
        });
        axes
    };

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => {
            0 // treat scalar as 0-dim tensor
        }
        _ => panic!("Unsqueeze: invalid input type"),
    };

    let output_elem = match &node.outputs[0].ty {
        ArgType::Tensor(_) => node.inputs[0].ty.elem_type().clone(),
        ArgType::Scalar(elem_type) => elem_type.clone(),
        _ => panic!("Unsqueeze: invalid output type"),
    };

    let output_rank = if let Some(axes) = axes {
        input_rank + axes.len()
    } else if let ArgType::Tensor(tensor) = &node.inputs[1].ty {
        if let Some(static_shape) = &tensor.static_shape {
            input_rank + *static_shape.first().expect("Empty shape")
        } else {
            panic!("Unsqueeze: should have static shape")
        }
    } else {
        panic!("Unsqueeze: missing axes information")
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape: None, // shape is tracked and calculated at runtime
        elem_type: output_elem,
    });

    log::debug!("Unsqueeze output rank for {}: {}", node.name, output_rank);
}

/// Axes specification for the Unsqueeze operation.
#[derive(Debug, Clone)]
pub enum UnsqueezeConfig {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes that will be determined during execution.
    Runtime(Argument),
}

/// Creates UnsqueezeAxes configuration from the node attributes.
///
/// Note: This function should only execute if the second input is a constant.
/// If it wasn't and the output shape was known, unsqueeze has been remapped to reshape.
pub fn unsqueeze_config(node: &Node) -> UnsqueezeConfig {
    // Check if axes attribute exists
    for (key, value) in node.attrs.iter() {
        if key.as_str() == "axes" {
            return UnsqueezeConfig::Static(value.clone().into_i64s());
        }
    }

    assert!(
        !node.inputs.is_empty(),
        "Unsqueeze: axes tensor must be present"
    );

    let input_value = &node.inputs[1];

    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.rank, 1, "Unsqueeze: axes tensor must be 1D");
            if let Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) = input_value.value.as_ref()
            {
                UnsqueezeConfig::Static(shape.clone())
            } else {
                UnsqueezeConfig::Runtime(node.inputs[1].clone())
            }
        }
        _ => panic!("Arg for unsqueeze must be tensor or scalar"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorData};
    use std::collections::HashMap;

    // Implement custom equality for UnsqueezeConfig to make testing easier
    impl PartialEq<UnsqueezeConfig> for UnsqueezeConfig {
        fn eq(&self, other: &UnsqueezeConfig) -> bool {
            match (self, other) {
                (UnsqueezeConfig::Static(a), UnsqueezeConfig::Static(b)) => a == b,
                (UnsqueezeConfig::Runtime(a), UnsqueezeConfig::Runtime(b)) => a.name == b.name,
                _ => false,
            }
        }
    }

    fn create_test_node_with_attr(input_rank: usize, axes: Vec<i64>) -> Node {
        let inputs = vec![Argument {
            name: "X".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: input_rank,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let outputs = vec![Argument {
            name: "Y".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), AttributeValue::Int64s(axes.clone()));

        Node {
            node_type: NodeType::Unsqueeze,
            name: "test_unsqueeze".to_string(),
            inputs,
            outputs,
            attrs,
        }
    }

    fn create_test_node_with_input(input_rank: usize, axes: Vec<i64>, with_value: bool) -> Node {
        let axes_len = axes.len();
        let inputs = vec![
            Argument {
                name: "X".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "axes".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: Some(vec![axes_len]),
                }),
                value: if with_value {
                    Some(TensorData {
                        data: Data::Int64s(axes.clone()),
                        shape: vec![axes_len],
                    })
                } else {
                    None
                },
                passed: true,
            },
        ];

        let outputs = vec![Argument {
            name: "Y".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        Node {
            node_type: NodeType::Unsqueeze,
            name: "test_unsqueeze".to_string(),
            inputs,
            outputs,
            attrs: HashMap::new(),
        }
    }

    // Tests for unsqueeze_update_output function

    #[test]
    fn test_unsqueeze_with_attr() {
        let mut node = create_test_node_with_attr(2, vec![0, 3]);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4); // 2 + 2 = 4
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_with_input() {
        let mut node = create_test_node_with_input(3, vec![1, 2, 4], true);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 6); // 3 + 3 = 6
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar() {
        let mut node = create_test_node_with_attr(0, vec![0]);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1); // 0 + 1 = 1
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Unsqueeze: invalid input type")]
    fn test_unsqueeze_invalid_input() {
        let mut node = create_test_node_with_attr(2, vec![0]);
        node.inputs[0].ty = ArgType::Shape(1);
        unsqueeze_update_output(&mut node);
    }

    // Tests for unsqueeze_config function

    #[test]
    fn test_unsqueeze_config_with_attr() {
        // Test with axes provided as attribute
        let axes = vec![0, 2, 4];
        let node = create_test_node_with_attr(3, axes.clone());

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_with_static_input() {
        // Test with axes provided as input tensor with static value
        let axes = vec![1, 3];
        let node = create_test_node_with_input(2, axes.clone(), true);

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_with_runtime_input() {
        // Test with axes provided as input tensor but without static value
        let axes = vec![0, 2];
        let node = create_test_node_with_input(2, axes.clone(), false);

        let config = unsqueeze_config(&node);

        // Should return a Runtime config since the axes are only known at runtime
        match config {
            UnsqueezeConfig::Static(_) => panic!("Expected Runtime config"),
            UnsqueezeConfig::Runtime(arg) => {
                assert_eq!(arg.name, "axes");
            }
        }
    }

    #[test]
    fn test_unsqueeze_config_negative_axes() {
        // Test with negative axes (should be handled by the caller)
        let axes = vec![-1, -3];
        let node = create_test_node_with_attr(3, axes.clone());

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    fn test_unsqueeze_config_empty_axes() {
        // Test with empty axes array (edge case)
        let axes = vec![];
        let node = create_test_node_with_attr(2, axes.clone());

        let config = unsqueeze_config(&node);

        assert_eq!(config, UnsqueezeConfig::Static(axes));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_unsqueeze_config_missing_axes() {
        // Test with neither axes attribute nor input
        let mut node = create_test_node_with_attr(2, vec![0]);
        node.attrs.clear(); // Remove the axes attribute
        node.inputs = vec![node.inputs[0].clone()]; // Remove the axes input

        let _ = unsqueeze_config(&node);
    }

    #[test]
    #[should_panic(expected = "Unsqueeze: axes tensor must be 1D")]
    fn test_unsqueeze_config_invalid_axes_rank() {
        // Test with axes tensor that is not 1D
        let mut node = create_test_node_with_input(2, vec![0, 1], true);
        if let ArgType::Tensor(ref mut tensor) = node.inputs[1].ty {
            tensor.rank = 2; // Invalid rank for axes
        }

        let _ = unsqueeze_config(&node);
    }

    #[test]
    #[should_panic(expected = "Arg for unsqueeze must be tensor or scalar")]
    fn test_unsqueeze_config_invalid_axes_type() {
        // Test with axes input that is not a tensor
        let mut node = create_test_node_with_input(2, vec![0], false);
        node.inputs[1].ty = ArgType::Shape(1); // Invalid type for axes

        let _ = unsqueeze_config(&node);
    }
}
