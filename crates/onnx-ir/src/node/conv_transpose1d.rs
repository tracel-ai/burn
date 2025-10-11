use crate::ir::Node;
use crate::processor::{NodeProcessor, ProcessorContext};
use crate::util::same_as_input;

/// Configuration for ConvTranspose1d operations extracted from ONNX nodes
#[derive(Debug, Clone)]
pub struct ConvTranspose1dConfig {
    /// Input channels
    pub channels_in: usize,
    /// Output channels
    pub channels_out: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Dilation
    pub dilation: usize,
    /// Number of groups
    pub groups: usize,
    /// Whether bias is used
    pub bias: bool,
    /// Padding size
    pub padding: usize,
    /// Output padding size
    pub padding_out: usize,
}

impl ConvTranspose1dConfig {
    /// Create a new ConvTranspose1dConfig
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        padding_out: usize,
    ) -> Self {
        Self {
            channels_in,
            channels_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_out,
        }
    }
}

/// Create a ConvTranspose1dConfig from the attributes of the node
pub fn conv_transpose1d_config(
    curr: &Node,
    graph_data: &mut crate::from_onnx::GraphData,
) -> ConvTranspose1dConfig {
    let mut kernel_shape = Vec::new();
    let mut stride = vec![1]; // Default stride to 1
    let mut pads = vec![0, 0]; // Default padding to 0
    let mut dilations = vec![1]; // Default dilation to 1
    let mut group: usize = 1; // Default group to 1
    let mut output_padding = vec![0]; // Default output padding to 0

    // Extract attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => stride = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64() as usize,
            "output_padding" => output_padding = value.clone().into_i64s(),
            "auto_pad" => {
                let auto_pad = value.clone().into_string();
                if auto_pad != "NOTSET" {
                    panic!("Unsupported 'auto_pad' value: {auto_pad}");
                }
            }
            _ => panic!("Unexpected attribute for ConvTranspose1d: {key}"),
        }
    }

    // Check the pads are symmetric
    if pads.len() != 2 || pads[0] != pads[1] {
        panic!("Asymmetric padding is not supported for ConvTranspose1d: {pads:?}");
    }

    let weight_shape = curr.inputs[1]
        .into_value()
        .expect("ConvTranspose1d: weight tensor must be present")
        .shape
        .clone();

    // Check if bias is present (third input)
    let bias = curr.inputs.len() == 3;

    // Extract channels from the weight tensor shape [out_channels, in_channels]
    let channels_in = weight_shape[1] * group;
    let channels_out = weight_shape[0];

    let kernel_size = if kernel_shape.is_empty() {
        // https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
        // Spec says if kernel shape not present in attributes it should be inferred from the weight tensor
        if weight_shape.len() != 3 {
            panic!(
                "expected to infer kernel shape from a weight tensor of rank 3 but got shape {weight_shape:?}"
            );
        }

        weight_shape[2]
    } else {
        // Was set explicitly via attributes- use that
        kernel_shape[0] as _
    };

    ConvTranspose1dConfig {
        channels_in,
        channels_out,
        kernel_size,
        stride: stride[0] as usize,
        padding: pads[0] as usize,
        dilation: dilations[0] as usize,
        padding_out: output_padding[0] as usize,
        groups: group,
        bias,
    }
}

pub struct Convtranspose1dProcessor;

impl NodeProcessor for Convtranspose1dProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (1, None)
    }

    fn process(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        same_as_input(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    #[allow(clippy::too_many_arguments)]
    fn create_test_node(
        kernel_shape: Vec<i64>,
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        output_padding: Vec<i64>,
        has_bias: bool,
        auto_pad: Option<&str>,
    ) -> NodeBuilder {
        // Create weight tensor data
        let weight_data = vec![0.1; 16];

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = NodeBuilder::new(NodeType::ConvTranspose1d, "test_conv_transpose1d")
            .input_tensor_f32("data", 3, None)
            .input_tensor_f32_data(
                "weight",
                weight_data,
                vec![2, 2, 4], // [out_channels, in_channels, kernel_size]
            )
            .output_tensor_f32("output", 3, None);

        // Add bias if needed
        if has_bias {
            builder = builder.input_tensor_f32_data("bias", vec![0.1, 0.2], vec![2]);
        }

        // Add attributes
        builder = builder
            .attr_ints("strides", stride)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group)
            .attr_ints("output_padding", output_padding);

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        builder
    }

    #[test]
    fn test_conv_transpose1d_config_basic() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            None,
        )
        .build_with_graph_data(&mut graph_data);
        let config = conv_transpose1d_config(&node, &mut graph_data);

        assert_eq!(config.channels_in, 2);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.padding, 0);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.padding_out, 0);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
    }

    #[test]
    fn test_conv_transpose1d_config_with_params() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(
            vec![4],
            vec![2],
            vec![1, 1],
            vec![2],
            2,
            vec![1],
            true,
            None,
        )
        .build_with_graph_data(&mut graph_data);
        let config = conv_transpose1d_config(&node, &mut graph_data);

        assert_eq!(config.channels_in, 4); // weight_shape[1] * group = 2 * 2
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.padding, 1);
        assert_eq!(config.dilation, 2);
        assert_eq!(config.padding_out, 1);
        assert_eq!(config.groups, 2);
        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_conv_transpose1d_config_asymmetric_padding() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![1, 2],
            vec![1],
            1,
            vec![0],
            false,
            None,
        )
        .build_with_graph_data(&mut graph_data);
        let _ = conv_transpose1d_config(&node, &mut graph_data);
    }

    #[test]
    fn test_conv_transpose1d_config_autopad_not_set() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(&mut graph_data);
        let config = conv_transpose1d_config(&node, &mut graph_data);

        assert_eq!(config.channels_in, 2);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.padding, 0);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.padding_out, 0);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
    }

    #[test]
    #[should_panic = "Unsupported 'auto_pad' value"]
    fn test_conv_transpose1d_config_autopad_not_supported() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(&mut graph_data);
        let _config = conv_transpose1d_config(&node, &mut graph_data);
    }

    #[test]
    fn test_conv_transpose1d_config_kernel_shape_not_set() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(
            vec![],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            None,
        )
        .build_with_graph_data(&mut graph_data);
        let config = conv_transpose1d_config(&node, &mut graph_data);

        assert_eq!(config.kernel_size, 4); // Inferred via weight tensor shape
    }
}
