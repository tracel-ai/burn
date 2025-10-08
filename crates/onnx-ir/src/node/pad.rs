use crate::processor::{NodeProcessor, ProcessorContext};
use crate::util::same_as_input;

use crate::ir::{ArgType, AttributeValue, Data, Node, TensorData};

/// Configuration for the Pad operation.
#[derive(Debug, Clone, PartialEq)]
pub struct PadConfig {
    /// The paddings to be applied to each dimension.
    pub pads: Vec<usize>,
    /// The constant value to fill the padded areas with.
    pub constant_value: f32,
}

impl PadConfig {
    pub fn new(pads: Vec<usize>, constant_value: f32) -> Self {
        PadConfig {
            pads,
            constant_value,
        }
    }
}

/// Creates a PadConfig from the node attributes and inputs.
pub fn pad_config(node: &Node) -> PadConfig {
    fn get_pads_input(node: &Node) -> Vec<i64> {
        if node.inputs.len() <= 1 {
            return Vec::new();
        }

        match &node.inputs[1].value {
            Some(TensorData { data, .. }) => data.clone().into_i64s(),
            _ => Vec::new(),
        }
    }
    fn get_pads(node: &Node) -> Vec<usize> {
        if node.inputs.is_empty() {
            panic!("Pad: must provide data as input")
        }
        if node.inputs.len() >= 4 {
            panic!("Pad: axes input is not supported")
        }

        let input_dim = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => panic!("Pad: Only tensor input is valid"),
        };

        // TODO: Handle more possible attributes
        let mut pads: Vec<usize> = get_pads_input(node)
            .into_iter()
            .map(|x| x as usize)
            .collect();

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "pads" => {
                    pads = value
                        .clone()
                        .into_i64s()
                        .iter()
                        .map(|&x| {
                            if x < 0 {
                                panic!("Pad: Negative pad is not supported");
                            }
                            x as usize
                        })
                        .collect()
                }
                "mode" => {
                    let mode = value.clone().into_string();
                    if mode != "constant" {
                        panic!("only constant mode is supported, given mode is {mode}");
                    }
                }

                _ => {}
            }
        }

        if pads.is_empty() {
            panic!("Pad: pads should be given as attribute or as input");
        }

        if pads.len() != input_dim * 2 {
            panic!("Pad: pads should be a 1D tensor of shape [2 * num_axes]");
        }
        // TODO: Burn's pad should support 1D tensor
        if input_dim < 2 {
            panic!("Pad: input tensor should be rank 2 or higher");
        }

        let left_index = input_dim - 1;
        let top_index = input_dim - 2;
        let right_index = pads.len() - 1;
        let bottom_index = pads.len() - 2;
        let index_list = [left_index, top_index, right_index, bottom_index];

        for (index, &item) in pads.iter().enumerate() {
            if !index_list.contains(&index) && item != 0 {
                panic!(
                    "Pad: padding will only be applied to the last two dimensions but found non zero padding for other dimensions"
                );
            }
        }

        let left = pads[left_index];
        let top = pads[top_index];
        let right = pads[right_index];
        let bottom = pads[bottom_index];
        vec![left, right, top, bottom]
    }
    fn get_constant_value(node: &Node) -> f32 {
        // TODO: Support int, boolean
        let mut constant_value = node.inputs
                .get(2)
                .and_then(|input| match &input.value.as_ref().expect("Value input must be present").data {
                    Data::Float16s(constant_value) => {
                        constant_value.first().map(|&f| f32::from(f))
                    }
                    Data::Float32s(constant_value) => {
                        constant_value.first().copied()
                    }
                    Data::Float64s(constant_value) => {
                        constant_value.first().map(|&f| f as f32)
                    }
                    Data::Float16(constant_value) => Some(f32::from(*constant_value)),
                    Data::Float32(constant_value) => Some(*constant_value),
                    Data::Float64(constant_value) => Some(*constant_value as f32),
                     _ => panic!("Pad: only float values are currently supported for constant value, submit an issue on github"),
                })
                .unwrap_or(0.0);

        if node.attrs.contains_key("value") {
            constant_value = node.attrs.get("value").map(|value| match value {
                AttributeValue::Float32(value) => *value,
                _ => panic!("Pad: only float32 values are currently supported for constant value as attribute, submit an issue on github"),
            }).expect("constant_value should have had a value now");
        }
        constant_value
    }

    let pads = get_pads(node);
    let constant_value = get_constant_value(node);

    PadConfig::new(pads, constant_value)
}

pub struct PadProcessor;

impl NodeProcessor for PadProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (2, None)
    }

    fn process(&self, node: &mut Node, _context: &ProcessorContext) {
        same_as_input(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, Data, ElementType, NodeType, TensorData, TensorType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        pad_attrs: Option<Vec<i64>>,
        pad_inputs: Option<Vec<i64>>,
        constant_value_attr: Option<f32>,
        constant_value_input: Option<f32>,
        mode: Option<&str>,
        rank: usize,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("output", rank, None);

        // Add pad inputs if provided
        if let Some(pads) = pad_inputs.clone() {
            builder = builder.input_tensor_i64_data("pads", pads, vec![]);
        }

        // Add constant value input if provided
        if let Some(value) = constant_value_input {
            builder = builder.input_scalar_tensor_f32("constant_value", Some(value));
        }

        // Add attributes if provided
        if let Some(pads) = pad_attrs {
            builder = builder.attr_ints("pads", pads);
        }

        if let Some(value) = constant_value_attr {
            builder = builder.attr_float("value", value);
        }

        if let Some(mode_val) = mode {
            builder = builder.attr_string("mode", mode_val);
        }

        builder.build()
    }

    #[test]
    fn test_pad_config_with_attrs() {
        // Test for 2D tensor (rank 2)
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(
            Some(pads.clone()),
            None,
            Some(0.0),
            None,
            Some("constant"),
            2,
        );
        let config = pad_config(&node);
        assert_eq!(
            config,
            PadConfig {
                pads: vec![0, 1, 0, 1],
                constant_value: 0.0
            }
        );
    }

    #[test]
    fn test_pad_config_with_inputs() {
        // For a 2D tensor, pads should have 4 values (2*rank)
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(None, Some(pads.clone()), None, Some(1.0), None, 2);
        let config = pad_config(&node);
        assert_eq!(
            config,
            PadConfig {
                pads: vec![0, 1, 0, 1],
                constant_value: 1.0
            }
        );
    }

    #[test]
    fn test_pad_config_with_3d_tensor() {
        // For a 3D tensor, pads should have 6 values (2*rank)
        let pads = vec![0, 0, 0, 0, 1, 1];
        let node = create_test_node(
            Some(pads.clone()),
            None,
            Some(0.5),
            None,
            Some("constant"),
            3,
        );
        let config = pad_config(&node);
        assert_eq!(
            config,
            PadConfig {
                pads: vec![0, 1, 0, 1],
                constant_value: 0.5
            }
        );
    }

    #[test]
    fn test_pad_config_attrs_override_inputs() {
        // Attributes should override inputs
        let attr_pads = vec![0, 0, 2, 2];
        let input_pads = vec![0, 0, 1, 1];
        let node = create_test_node(
            Some(attr_pads.clone()),
            Some(input_pads),
            Some(0.0),
            Some(1.0),
            Some("constant"),
            2,
        );
        let config = pad_config(&node);
        assert_eq!(
            config,
            PadConfig {
                pads: vec![0, 2, 0, 2],
                constant_value: 0.0
            }
        );
    }

    #[test]
    #[should_panic(expected = "Pad: must provide data as input")]
    fn test_pad_config_no_inputs() {
        let mut node = create_test_node(None, None, None, None, None, 2);
        node.inputs = vec![];
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "Pad: Only tensor input is valid")]
    fn test_pad_config_invalid_input_type() {
        let mut node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, None, 2);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "Pad: axes input is not supported")]
    fn test_pad_config_with_axes_input() {
        // Create node with 4 inputs (including axes)
        let mut node = create_test_node(None, Some(vec![0, 0, 1, 1]), None, Some(0.0), None, 2);
        node.inputs.push(Argument {
            name: "axes".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: None,
            }),
            value: Some(TensorData {
                data: Data::Int64s(vec![0, 1]),
                shape: vec![],
            }),
            passed: true,
        });
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "Pad: Negative pad is not supported")]
    fn test_pad_config_negative_pad() {
        let node = create_test_node(Some(vec![0, 0, -1, 1]), None, None, None, None, 2);
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "only constant mode is supported")]
    fn test_pad_config_unsupported_mode() {
        let node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, Some("reflect"), 2);
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "Pad: pads should be given as attribute or as input")]
    fn test_pad_config_no_pads() {
        let node = create_test_node(None, None, None, None, None, 2);
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "Pad: pads should be a 1D tensor of shape [2 * num_axes]")]
    fn test_pad_config_invalid_pads_length() {
        let node = create_test_node(Some(vec![0, 0, 1]), None, None, None, None, 2);
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "Pad: input tensor should be rank 2 or higher")]
    fn test_pad_config_invalid_tensor_rank() {
        let node = create_test_node(Some(vec![0, 1]), None, None, None, None, 1);
        let _ = pad_config(&node);
    }

    #[test]
    #[should_panic(expected = "Pad: padding will only be applied to the last two dimensions")]
    fn test_pad_config_non_zero_padding_on_other_dimensions() {
        // For a 3D tensor, we try to set non-zero padding on first dimension
        let node = create_test_node(Some(vec![1, 0, 0, 0, 1, 1]), None, None, None, None, 3);
        let _ = pad_config(&node);
    }
}
