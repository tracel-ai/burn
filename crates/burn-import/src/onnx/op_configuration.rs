use burn::nn::{
    conv::{
        Conv1dConfig, Conv2dConfig, Conv3dConfig, ConvTranspose1dConfig, ConvTranspose2dConfig,
        ConvTranspose3dConfig,
    },
    pool::{AvgPool1dConfig, AvgPool2dConfig, MaxPool1dConfig, MaxPool2dConfig},
    BatchNormConfig, DropoutConfig, LayerNormConfig, LinearConfig, PaddingConfig1d,
    PaddingConfig2d, PaddingConfig3d,
};

use crate::burn::node::{
    expand::ExpandShape, pad::PadConfig, split::SplitConfig, tile::TileConfig, top_k::TopKConfig,
    trilu::TriluConfig,
};
use onnx_ir::ir::{ArgType, AttributeValue, Data, ElementType, Node};

/// Create a Conv1dConfig from the attributes of the node
pub fn conv1d_config(curr: &Node) -> Conv1dConfig {
    let mut kernel_shape = Vec::new(); // TODO default inferred from weight tensor per spec
    let mut strides = vec![1];
    let mut pads = vec![0, 0];
    let mut dilations = vec![1];
    let mut group: usize = 1;

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("Conv1d: weight tensor must be present");
    };

    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64() as usize,
            _ => {}
        }
    }

    if kernel_shape.is_empty() {
        kernel_shape = onnx_ir::util::infer_conv_kernel_shape(&curr.inputs[1].ty);
    }

    // the channels are inverted in the weight tensor
    let shape = weight.shape.clone().unwrap();
    let channels_in = shape[1] * group;
    let channels_out = shape[0];

    let padding = padding_config_1d(&pads);

    Conv1dConfig::new(channels_in, channels_out, kernel_shape[0] as usize)
        .with_stride(strides[0] as usize)
        .with_dilation(dilations[0] as usize)
        .with_groups(group)
        .with_bias(bias)
        .with_padding(padding)
}

/// Create a Conv2dConfig from the attributes of the node
pub fn conv2d_config(curr: &Node) -> Conv2dConfig {
    let mut kernel_shape = Vec::new(); // TODO default inferred from weight tensor per spec
    let mut strides = vec![1, 1];
    let mut pads = vec![0, 0, 0, 0];
    let mut dilations = vec![1, 1];
    let mut group: usize = 1;

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("Conv2d: weight tensor must be present");
    };
    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64() as usize,
            _ => {}
        }
    }

    if kernel_shape.is_empty() {
        kernel_shape = onnx_ir::util::infer_conv_kernel_shape(&curr.inputs[1].ty);
    }

    // the channels are inverted in the weight tensor
    let shape = weight.shape.clone().unwrap();
    let channels: [usize; 2] = [shape[1] * group, shape[0]];

    let padding = padding_config_2d(&pads);

    Conv2dConfig::new(
        channels,
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
    )
    .with_stride([strides[0] as usize, strides[1] as usize])
    .with_dilation([dilations[0] as usize, dilations[1] as usize])
    .with_groups(group)
    .with_bias(bias)
    .with_padding(padding)
}

/// Create a Conv3dConfig from the attributes of the node
pub fn conv3d_config(curr: &Node) -> Conv3dConfig {
    let mut kernel_shape = Vec::new(); // TODO default inferred from weight tensor per spec
    let mut strides = vec![1, 1, 1];
    let mut pads = vec![0, 0, 0, 0, 0, 0];
    let mut dilations = vec![1, 1, 1];
    let mut group: usize = 1;

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("Conv3d: weight tensor must be present");
    };
    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64() as usize,
            _ => {}
        }
    }

    if kernel_shape.is_empty() {
        kernel_shape = onnx_ir::util::infer_conv_kernel_shape(&curr.inputs[1].ty);
    }

    // the channels are inverted in the weight tensor
    let shape = weight.shape.clone().unwrap();
    let channels: [usize; 2] = [shape[1] * group, shape[0]];

    let padding = padding_config_3d(&pads);

    Conv3dConfig::new(
        channels,
        [
            kernel_shape[0] as usize,
            kernel_shape[1] as usize,
            kernel_shape[2] as usize,
        ],
    )
    .with_stride([
        strides[0] as usize,
        strides[1] as usize,
        strides[2] as usize,
    ])
    .with_dilation([
        dilations[0] as usize,
        dilations[1] as usize,
        dilations[2] as usize,
    ])
    .with_groups(group)
    .with_bias(bias)
    .with_padding(padding)
}

/// Create a MaxPool2dConfig from the attributes of the node
pub fn max_pool1d_config(curr: &Node) -> MaxPool1dConfig {
    let mut kernel_shape = Vec::new();
    let mut stride = vec![1];
    let mut pads = vec![0, 0];
    let mut dilation = vec![1];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => stride = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilation = value.clone().into_i64s(),
            _ => {}
        }
    }
    assert_eq!(kernel_shape.len(), 1);
    assert_eq!(dilation.len(), 1);
    assert_eq!(stride.len(), 1);
    let padding = padding_config_1d(&pads);

    MaxPool1dConfig::new(kernel_shape[0] as usize)
        .with_stride(stride[0] as usize)
        .with_padding(padding)
        .with_dilation(dilation[0] as usize)
}

/// Create a MaxPool2dConfig from the attributes of the node
pub fn max_pool2d_config(curr: &Node) -> MaxPool2dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = vec![1, 1];
    let mut pads = vec![0, 0, 0, 0];
    let mut dilations = vec![1, 1];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            _ => {}
        }
    }

    let padding = padding_config_2d(&pads);

    MaxPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
        .with_dilation([dilations[0] as usize, dilations[1] as usize])
}

pub fn conv_transpose1d_config(curr: &Node) -> ConvTranspose1dConfig {
    let mut attrs = curr.attrs.clone();

    // Extract kernel_shape, default to an empty vector if not present
    let kernel_shape = attrs
        .remove("kernel_shape")
        .map(AttributeValue::into_i64s)
        .unwrap_or_default();

    // Extract strides, default to 1 if not present
    let stride = attrs
        .remove("strides")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1]);

    // Extract padding, default to 0 if not present
    let pads = attrs
        .remove("pads")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0]);

    // Extract dilations, default to 1 if not present
    let dilations = attrs
        .remove("dilations")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1]);

    // Extract group attribute, default to 1
    let group = attrs
        .remove("group")
        .map(AttributeValue::into_i64)
        .unwrap_or(1) as usize;

    // Extract output_padding, default to 0 if not present
    let output_padding = attrs
        .remove("output_padding")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0]);

    // Ensure no unused attributes remain
    if !attrs.is_empty() {
        panic!("Not all attributes are used: {attrs:?}");
    }
    // Check the pads are symmetric.
    if pads.len() != 2 || pads[0] != pads[1] {
        panic!(
            "Asymmetric padding is not supported for ConvTranspose1d: {:?}",
            pads
        );
    }
    // Extract weight tensor, verify it's present
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("ConvTranspose1d: weight tensor must be present");
    };

    // Check if bias is present (third input)
    let bias = curr.inputs.len() == 3;

    // Extract channels from the weight tensor shape [out_channels, in_channels]
    let shape = weight.shape.clone().unwrap();
    let channels: [usize; 2] = [shape[1] * group, shape[0]];

    // Create the ConvTranspose1d configuration
    ConvTranspose1dConfig::new(channels, kernel_shape[0] as usize)
        .with_stride(stride[0] as usize)
        .with_padding(pads[0] as usize)
        .with_dilation(dilations[0] as usize)
        .with_padding_out(output_padding[0] as usize)
        .with_groups(group)
        .with_bias(bias)
}

pub fn conv_transpose2d_config(curr: &Node) -> ConvTranspose2dConfig {
    let mut attrs = curr.attrs.clone();
    let kernel_shape = attrs
        .remove("kernel_shape")
        .map(AttributeValue::into_i64s)
        .unwrap_or_default();
    let stride = attrs
        .remove("strides")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1, 1]);
    let pads = attrs
        .remove("pads")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0, 0, 0]);
    let dilations = attrs
        .remove("dilations")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1, 1]);
    let group = attrs
        .remove("group")
        .map(AttributeValue::into_i64)
        .unwrap_or(1) as usize;
    let output_padding = attrs
        .remove("output_padding")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0]);

    // Trick with remove + empty check is simplest way to not forget some attribute for runtime:
    if !attrs.is_empty() {
        panic!("Not all attributes are used: {attrs:?}");
    }
    // Check the pads are symmetric.
    let [left, top, right, bottom] = [pads[0], pads[1], pads[2], pads[3]];
    if left < 0 || top < 0 || right < 0 || bottom < 0 {
        panic!("Negative pad values are not supported");
    } else if (left != right) || (top != bottom) {
        panic!("Asymmetric padding is not supported");
    }
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("ConvTranspose2d: weight tensor must be present");
    };

    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    // the channels are inverted in the weight tensor
    let shape = weight.shape.clone().unwrap();
    let channels: [usize; 2] = [shape[1] * group, shape[0]];

    ConvTranspose2dConfig::new(
        channels,
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
    )
    .with_stride([stride[0] as usize, stride[1] as usize])
    .with_padding([pads[0] as usize, pads[1] as usize])
    .with_dilation([dilations[0] as usize, dilations[1] as usize])
    .with_padding_out([output_padding[0] as usize, output_padding[1] as usize])
    .with_groups(group)
    .with_bias(bias)
}

pub fn conv_transpose3d_config(curr: &Node) -> ConvTranspose3dConfig {
    let mut attrs = curr.attrs.clone();
    let kernel_shape = attrs
        .remove("kernel_shape")
        .map(AttributeValue::into_i64s)
        .unwrap_or_default();
    let stride = attrs
        .remove("strides")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1, 1, 1]);
    let pads = attrs
        .remove("pads")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0, 0, 0, 0, 0]);
    let dilations = attrs
        .remove("dilations")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1, 1, 1]);
    let group = attrs
        .remove("group")
        .map(AttributeValue::into_i64)
        .unwrap_or(1) as usize;
    let output_padding = attrs
        .remove("output_padding")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0, 0]);

    // Trick with remove + empty check is simplest way to not forget some attribute for runtime:
    if !attrs.is_empty() {
        panic!("Not all attributes are used: {attrs:?}");
    }
    // Check the pads are symmetric.
    let [left, top, front, right, bottom, back] =
        [pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]];

    if left < 0 || top < 0 || front < 0 || right < 0 || bottom < 0 || back < 0 {
        panic!("Negative pad values are not supported");
    } else if (left != right) || (top != bottom) || (front != back) {
        panic!("Asymmetric padding is not supported");
    }
    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("ConvTranspose3d: weight tensor must be present");
    };

    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    // the channels are inverted in the weight tensor
    let shape = weight.shape.clone().unwrap();
    let channels: [usize; 2] = [shape[1] * group, shape[0]];

    ConvTranspose3dConfig::new(
        channels,
        [
            kernel_shape[0] as usize,
            kernel_shape[1] as usize,
            kernel_shape[2] as usize,
        ],
    )
    .with_stride([stride[0] as usize, stride[1] as usize, stride[2] as usize])
    .with_padding([pads[0] as usize, pads[1] as usize, pads[2] as usize])
    .with_dilation([
        dilations[0] as usize,
        dilations[1] as usize,
        dilations[2] as usize,
    ])
    .with_padding_out([
        output_padding[0] as usize,
        output_padding[1] as usize,
        output_padding[2] as usize,
    ])
    .with_groups(group)
    .with_bias(bias)
}

pub fn avg_pool1d_config(curr: &Node) -> AvgPool1dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = vec![1];
    let mut pads = vec![0, 0];
    let mut count_include_pad: i64 = 0;
    let mut ceil_mode: i64 = 0;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "count_include_pad" => count_include_pad = value.clone().into_i64(),
            "ceil_mode" => ceil_mode = value.clone().into_i64(),
            _ => {}
        }
    }
    assert_eq!(kernel_shape.len(), 1);
    assert_eq!(strides.len(), 1);

    if ceil_mode == 1 {
        panic!("ceil_mode is not supported");
    }

    let padding = padding_config_1d(&pads);

    AvgPool1dConfig::new(kernel_shape[0] as usize)
        .with_stride(strides[0] as usize)
        .with_padding(padding)
        .with_count_include_pad(count_include_pad == 1)
}
/// Create a AvgPool2dConfig from the attributes of the node
pub fn avg_pool2d_config(curr: &Node) -> AvgPool2dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = vec![1, 1];
    let mut pads = vec![0, 0, 0, 0];
    let mut count_include_pad: i64 = 0;
    let mut ceil_mode: i64 = 0;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "count_include_pad" => count_include_pad = value.clone().into_i64(),
            "ceil_mode" => ceil_mode = value.clone().into_i64(),
            _ => {}
        }
    }

    if ceil_mode == 1 {
        panic!("ceil_mode is not supported");
    }

    let padding = padding_config_2d(&pads);

    AvgPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
        .with_count_include_pad(count_include_pad == 1)
}

pub fn expand_config(node: &Node) -> ExpandShape {
    let input_value = &node.inputs[1].value;
    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.dim, 1, "Expand: shape tensor must be 1D");
            assert!(
                tensor.shape.is_some(),
                "Expand: shape tensor shape must be known!"
            );
            assert!(
                matches!(tensor.elem_type, ElementType::Int64),
                "Expand: shape tensor must have element type int64"
            );
        }
        ArgType::Shape(_) => {
            // Shapes are always 1-D int64 data, so nothing to assert here
        }
        _ => panic!("Only tensor input is valid for shape"),
    }

    match input_value.as_ref() {
        Some(Data::Int64s(shape)) => ExpandShape::Static(shape.clone()),
        None => {
            // we were unable to statically determine the input value, so we'll need to fetch it at runtime
            ExpandShape::Runtime(crate::burn::Type::from(&node.inputs[1]))
        }
        _ => panic!("Shape data type must be int64, is {:?}", input_value),
    }
}

/// Create a FlattenConfig from the attributes of the node
pub fn flatten_config(curr: &Node) -> (usize, usize) {
    // the begin dimension is the first dimension (Default: 1 per ONNX spec)
    let mut start_dim: i64 = 1;

    // check if the node has only one input
    if curr.inputs.len() != 1 {
        panic!(
            "Flatten: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // check if the input tensor has at least 2 dimensions
    if tensor.dim < 2 {
        panic!(
            "Flatten: input tensor must have at least 2 dimensions (got {:?})",
            tensor.dim
        );
    }

    // the end dimension is the last dimension
    let end_dim = tensor.dim - 1;

    // extract the attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "axis" => start_dim = value.clone().into_i64(),
            _ => {}
        }
    }

    // if beg_dim is negative, it is counted from the end
    if start_dim < 0 {
        start_dim += tensor.dim as i64;
    }

    (start_dim as usize, end_dim)
}

/// Create a GatherConfig from the attributes of the node
pub fn gather_config(curr: &Node) -> usize {
    // Default: 0 per ONNX spec
    let mut dim: i64 = 0;

    // check if the node has only one input
    if curr.inputs.len() != 2 {
        panic!("Gather: index tensor must be present");
    }

    // extract the shape of the input tensor
    let input_dim = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor.dim as i64,
        ArgType::Shape(_shape) => 1, //Shape is always 1-D
        other => panic!("Only tensor or shape input is valid, got {:?}", other),
    };

    // extract the attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "axis" => dim = value.clone().into_i64(),
            _ => {}
        }
    }

    // if dim is negative, it is counted from the end
    if dim < 0 {
        dim += input_dim;
    }

    dim as usize
}

/// Create a LinearConfig from the attributes of the node
pub fn linear_config(node: &Node) -> LinearConfig {
    if node.inputs.len() < 2 {
        panic!("Linear: missing weight tensor");
    }

    // extract the shape of the weight tensor
    let weight = if let ArgType::Tensor(ref weight) = node.inputs[1].ty {
        weight
    } else {
        panic!("Linear: weight tensor must be present");
    };

    // check if the weight tensor has at least 2 dimensions
    if weight.dim < 2 {
        panic!(
            "Linear: weight tensor must have at least 2 dimensions (got {:?})",
            weight.dim
        );
    }

    let shape = weight.shape.clone().unwrap();
    let (in_size, out_size) = (shape[0], shape[1]);

    // check if the bias is present
    let bias = node.inputs.len() == 3 && node.inputs[2].value.is_some();

    LinearConfig::new(in_size, out_size).with_bias(bias)
}

/// Create a DropoutConfig from an attribute and state of the node
pub fn dropout_config(node: &Node) -> DropoutConfig {
    // Opset 7 and older store probability as an attribute
    if node.attrs.contains_key("ratio") {
        let prob = node.attrs.get("ratio").unwrap().clone().into_f32();
        return DropoutConfig::new(prob as f64);
    }

    if node.inputs.len() < 2 {
        panic!("Dropout configuration must have at least 2 inputs");
    }

    let ratio = node.inputs[1]
        .value
        .clone()
        .expect("Dropout ratio must be passed in the second input")
        .into_scalar();

    let prob = match ratio {
        Data::Float16(ratio) => f64::from(f32::from(ratio)),
        Data::Float32(ratio) => ratio as f64,
        Data::Float64(ratio) => ratio,
        _ => panic!("Dropout ratio must be a float"),
    };

    DropoutConfig::new(prob)
}

/// Create log_softmax config from the attributes of the node
pub fn log_softmax_config(node: &Node) -> usize {
    // the axis is the last dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = -1;

    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "LogSoftmax: multiple inputs are not supported (got {:?})",
            node.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            _ => {}
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.dim as i64;
    }

    axis as usize
}

/// Create softmax config from the attributes of the node
pub fn softmax_config(node: &Node) -> usize {
    // the axis is the last dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = -1;

    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "Softmax: multiple inputs are not supported (got {:?})",
            node.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            _ => {}
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.dim as i64;
    }

    axis as usize
}

/// Create argmax config from the attributes of the node
pub fn argmax_config(node: &Node) -> usize {
    let mut axis: i64 = 0;

    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "Argmax: multiple inputs are not supported (got {:?})",
            node.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "select_last_index" => {
                // not all params are supported in burn
                if value.clone().into_i64() != 0 {
                    log::warn!(
                        "only select_last_index=0 is supported for argmax in burn. Ignoring supplied value (got {:?})",
                        value
                    );
                }
            }
            "keepdims" => {
                // not all params are supported in burn
                if value.clone().into_i64() != 1 {
                    panic!(
                        "Only keepdims=1 is supported for argmax in burn (got {:?})",
                        value
                    );
                }
            }
            _ => {}
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.dim as i64;
    }

    axis as usize
}

/// Create concat config from the attributes of the node
pub fn concat_config(node: &Node) -> usize {
    // the axis is the last dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = 1;

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            _ => {}
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.dim as i64;
    }

    axis as usize
}

/// Create a BatchNormConfig from the attributes of the node
pub fn batch_norm_config(node: &Node) -> BatchNormConfig {
    // extract the shape of the weight tensor
    let tensor_type = if let ArgType::Tensor(ref tensor_type) = node.inputs[1].ty {
        tensor_type
    } else {
        panic!("BatchNorm: weight tensor must be present");
    };

    let num_features: usize = tensor_type.shape.clone().unwrap()[0];

    let mut epsilon = 0f32;
    let mut momentum = 0f32;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "momentum" => momentum = value.clone().into_f32(),
            "epsilon" => epsilon = value.clone().into_f32(),
            _ => {}
        }
    }

    BatchNormConfig::new(num_features)
        .with_epsilon(epsilon as f64)
        .with_momentum(momentum as f64)
}

/// Create a LayerNormConfig from the attributes of the node
pub fn layer_norm_config(node: &Node) -> (LayerNormConfig, bool) {
    // Extract the shape of the weight tensor
    let tensor_type = if let ArgType::Tensor(ref tensor_type) = node.inputs[1].ty {
        tensor_type
    } else {
        panic!("LayerNorm: weight tensor must be present");
    };

    let num_features: usize = tensor_type.shape.clone().unwrap()[0];

    // When `stash_type` is `1` (default), perform operations in 32-bit float and
    // cast the results back to original dtype
    let mut stash_type = 1;
    let mut axis = -1;
    let mut epsilon = 1e-5;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "epsilon" => epsilon = value.clone().into_f32(),
            "stash_type" => stash_type = value.clone().into_i64(),
            _ => {}
        }
    }

    if axis != -1 && axis != tensor_type.dim as i64 - 1 {
        panic!("LayerNorm: normalization is only supported on the last axis right now")
    }

    (
        LayerNormConfig::new(num_features).with_epsilon(epsilon as f64),
        stash_type == 1,
    )
}

/// Create a TileConfig from the attributes of the node
pub fn tile_config(node: &Node) -> TileConfig {
    let repeat = node
        .inputs
        .get(1)
        .map(|input| {
            if let Some(data) = &input.value {
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

/// Create a TopKConfig from the attributes of the node.
pub fn top_k_config(node: &Node) -> TopKConfig {
    // extract the shape of the input data tensor
    let data_tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    let k = match node.inputs.get(1) {
        Some(k_tensor) => k_tensor
            .clone()
            .value
            .expect("TopK: only constant 'k' tensor is currently supported")
            .into_i64s()[0],
        _ => node
            .attrs
            .get("k")
            .expect("TopK: number of top elements 'k' is missing")
            .clone()
            .into_i64(),
    };

    let mut axis = match node.attrs.get("axis") {
        Some(axis) => axis.clone().into_i64(),
        None => -1,
    };

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += data_tensor.dim as i64;
    }

    if let Some(largest) = node.attrs.get("largest") {
        if largest.clone().into_i64() != 1 {
            unimplemented!("TopK: only largest elements is supported")
        }
    };

    if let Some(sorted) = node.attrs.get("sorted") {
        if sorted.clone().into_i64() != 1 {
            unimplemented!("TopK: only sorted elements is supported")
        }
    };

    TopKConfig::new(axis as usize, k as usize)
}

/// Create a TriluConfig from the attributes of the node
pub fn trilu_config(node: &Node) -> TriluConfig {
    let mut upper = true;
    let mut diagonal = 0;
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "upper" => upper = value.clone().into_i64() != 0,
            _ => {}
        }
    }
    // The second input of the Trilu node is the diagonal value, coming from a constant node
    if let Some(diagonal_arg) = node.inputs.get(1) {
        if let Some(Data::Int64(diagonal_val)) = &diagonal_arg.value {
            diagonal = *diagonal_val;
        }
    }
    TriluConfig::new(upper, diagonal)
}

/// Create a PadConfig from the attributes of the node
pub fn pad_config(node: &Node) -> PadConfig {
    fn get_pads_input(node: &Node) -> Vec<i64> {
        // If the input is not provided, return an empty vector
        if node.inputs.get(1).is_none() {
            return Vec::new();
        }

        match &node.inputs[1].value {
            Some(Data::Int64s(shape)) => shape.clone(),
            _ => panic!("Tensor data type must be int64"),
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
            ArgType::Tensor(tensor) => tensor.dim,
            _ => panic!("Pad: Only tensor input is valid"),
        };

        //TODO : handle more possible attributes
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
                        panic!("only constant mode is supported, given mode is {}", mode);
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
                panic!("Pad: padding will only be applied to the last two dimensions but found non zero padding for other dimensions");
            }
        }

        let left = pads[left_index];
        let top = pads[top_index];
        let right = pads[right_index];
        let bottom = pads[bottom_index];
        vec![left, right, top, bottom]
    }
    fn get_constant_value(node: &Node) -> f32 {
        // TODO: support int, boolean
        let mut constant_value = node.inputs
                .get(2)
                .and_then(|input| match &input.value {
                    Some(Data::Float16s(constant_value)) => {
                        constant_value.first().map(|&f| f32::from(f))
                    }
                    Some(Data::Float32s(constant_value)) => {
                        constant_value.first().copied()
                    }
                    Some(Data::Float64s(constant_value)) => {
                        constant_value.first().map(|&f| f as f32)
                    }
                    Some(Data::Float16(constant_value)) => Some(f32::from(*constant_value)),
                    Some(Data::Float32(constant_value)) => Some(*constant_value),
                    Some(Data::Float64(constant_value)) => Some(*constant_value as f32),
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

/// Calculate the padding configuration for a 1D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values
///
/// # Panics
///
/// * If the padding is negative
/// * If the padding is not symmetric
///
/// # Returns
///
/// * The padding configuration
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
fn padding_config_1d(pads: &[i64]) -> PaddingConfig1d {
    let [left, right] = [pads[0], pads[1]];

    if left < 0 || right < 0 {
        panic!("Negative pad values are not supported");
    } else if left != right {
        panic!("Asymmetric padding is not supported");
    } else if left == 0 && right == 0 {
        // i.e. [0, 0]
        PaddingConfig1d::Valid
    } else if left == right {
        // i.e. [2, 2]
        PaddingConfig1d::Explicit(left as usize)
    } else {
        // Unaccounted for padding configuration
        panic!("Padding configuration ({:?}) not supported", pads);
    }
}

/// Calculate the padding configuration for a 2D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values
///
/// # Panics
///
/// * If the padding is negative
/// * If the padding is not symmetric
///
/// # Returns
///
/// * The padding configuration
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
fn padding_config_2d(pads: &[i64]) -> PaddingConfig2d {
    let [left, top, right, bottom] = [pads[0], pads[1], pads[2], pads[3]];

    if left < 0 || top < 0 || right < 0 || bottom < 0 {
        panic!("Negative pad values are not supported");
    } else if (left != right) || (top != bottom) {
        panic!("Asymmetric padding is not supported");
    } else if left == 0 && top == 0 && right == 0 && bottom == 0 {
        // i.e [0, 0, 0, 0]
        PaddingConfig2d::Valid
    } else if left == right && top == bottom {
        // i.e [2, 3, 2, 3]
        PaddingConfig2d::Explicit(left as usize, top as usize)
    } else {
        // Unaccounted for padding configuration
        panic!("Padding configuration ({:?}) not supported", pads);
    }
}

/// Calculate the padding configuration for a 3D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values
///
/// # Panics
///
/// * If the padding is negative
/// * If the padding is not symmetric
///
/// # Returns
///
/// * The padding configuration
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
fn padding_config_3d(pads: &[i64]) -> PaddingConfig3d {
    let [left, top, front, right, bottom, back] =
        [pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]];

    if left < 0 || top < 0 || front < 0 || right < 0 || bottom < 0 || back < 0 {
        panic!("Negative pad values are not supported");
    } else if (left != right) || (top != bottom) || (front != back) {
        panic!("Asymmetric padding is not supported");
    } else if left == 0 && top == 0 && front == 0 && right == 0 && bottom == 0 && back == 0 {
        // i.e [0, 0, 0, 0]
        PaddingConfig3d::Valid
    } else if left == right && top == bottom && front == back {
        // i.e [2, 3, 2, 3]
        PaddingConfig3d::Explicit(left as usize, top as usize, front as usize)
    } else {
        // Unaccounted for padding configuration
        panic!("Padding configuration ({:?}) not supported", pads);
    }
}

// Create a LeakyReluConfig from the alpha attribute of the node
pub fn leaky_relu_config(node: &Node) -> f64 {
    let mut alpha = 0.01;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "alpha" => alpha = value.clone().into_f32() as f64,
            _ => {}
        }
    }

    alpha
}

// Create a HardSigmoidConfig from the alpha and beta attributes of the node
pub fn hard_sigmoid_config(node: &Node) -> (f64, f64) {
    let mut alpha = 0.2;
    let mut beta = 0.5;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "alpha" => alpha = value.clone().into_f32() as f64,
            "beta" => beta = value.clone().into_f32() as f64,
            _ => {}
        }
    }

    (alpha, beta)
}

pub fn reshape_config(node: &Node) -> Vec<i64> {
    let mut allowzero = 0;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "allowzero" => allowzero = value.clone().into_i64(),
            _ => {}
        }
    }

    // Burn does not support zero size shape (0 means false in ONNX)
    // (see https://onnx.ai/onnx/operators/onnx__Reshape.html#attributes)
    if allowzero != 0 {
        panic!("Zero shape size is not supported");
    }

    // TODO: check "shape" attribute
    if node.inputs.len() != 2 || node.inputs[1].value.is_none() {
        panic!("Reshape: shape tensor must be present for {:?}", node);
    }

    let input_value = &node.inputs[1].value;
    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.dim, 1, "Reshape: shape tensor must be 1D");

            if let Some(Data::Int64s(shape)) = input_value.as_ref() {
                shape.clone()
            } else {
                panic!("Tensor data type must be int64")
            }
        }
        _ => panic!("Only tensor input is valid for shape"),
    }
}

pub fn resize_config(node: &Node) -> (String, Vec<f32>, Vec<usize>) {
    let mut mode: String = "".to_string();

    let mut scales: Vec<f32>;
    let mut sizes: Vec<usize>;

    let input = if let ArgType::Tensor(tensor) = &node
        .inputs
        .first()
        .expect("Resize: Input tensor must be present")
        .ty
    {
        tensor
    } else {
        panic!("Resize: input must be a tensor")
    };

    // Note: we are ignoring some attributes because results are approximately the same
    // and we are not supporting all the attributes of the Resize operator.
    // However, some attributes are important to be checked and we are checking
    // against the default values of the attributes.
    // TODO revisit this when we have more Resize operators in the model
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "antialias" => assert_eq!(
                value.clone().into_i32(),
                0,
                "Resize: antialias other than 0 is not supported"
            ),
            "axes" => panic!("Resize: custom axes attribute is not supported"),
            "coordinate_transformation_mode" => {
                log::warn!("Resize: coordinate_transformation_mode is ignored")
            }

            "cubic_coeff_a" => log::warn!("Resize: cubic_coeff_a is ignored"),
            "exclude_outside" => assert_eq!(
                value.clone().into_i32(),
                0,
                "Resize: exclude_outside other than 0 is not supported"
            ),
            "extrapolation_value" => assert_eq!(
                value.clone().into_f32(),
                0.0,
                "Resize: extrapolation_value other than 0.0 is not supported"
            ),
            "keep_aspect_ratio_policy" => {
                assert_eq!(
                    value.clone().into_string().to_lowercase(),
                    "stretch",
                    "Resize: keep_aspect_ratio_policy other than 'stretch' is not supported"
                )
            }
            "mode" => mode = value.clone().into_string().to_lowercase(),
            "nearest_mode" => log::warn!("Resize: nearest_mode is ignored"),

            _ => {}
        }
    }

    let roi: Vec<f32> = node
        .inputs
        .get(1)
        .map(|input| {
            if let Some(data) = &input.value {
                data.clone().into_f32s()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();

    scales = node
        .inputs
        .get(2)
        .map(|input| {
            if let Some(data) = &input.value {
                data.clone().into_f32s()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();

    sizes = node
        .inputs
        .get(3)
        .map(|input| {
            if let Some(data) = &input.value {
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

    if mode.is_empty() {
        panic!("Resize: mode attribute is required")
    }

    if !roi.is_empty() {
        panic!("Resize: roi input is not supported")
    }

    if scales.is_empty() && sizes.is_empty() {
        panic!("Resize: either scales or sizes input is required")
    }

    if !scales.is_empty() {
        assert!(scales.len() == input.dim);
        // ignore the fist two items from scales
        // because they are the batch and channel dimensions
        scales = scales.iter().skip(2).cloned().collect();
    }

    if !sizes.is_empty() {
        assert!(sizes.len() == input.dim);
        // ignore the fist two items from sizes
        // because they are the batch and channel dimensions
        sizes = sizes.iter().skip(2).cloned().collect();
    }

    (mode, scales, sizes)
}

//Note this function should only execute if the second input is a constant
//if it wasn't and the output shape was known, unsqueeze has been remapped to reshape
pub fn unsqueeze_config(node: &Node) -> Vec<i64> {
    // Check if axes attribute exists
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => return value.clone().into_i64s(),
            _ => {}
        }
    }

    assert!(
        !node.inputs.is_empty(),
        "Unsqueeze: axes tensor must be present"
    );

    let input_value = &node.inputs[1];

    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.dim, 1, "Unsqueeze: axes tensor must be 1D");
            if let Some(Data::Int64s(shape)) = input_value.value.as_ref() {
                shape.clone()
            } else {
                panic!("Tensor data type must be int64")
            }
        }
        _ => panic!("Arg for unsqueeze must be tensor or scalar"),
    }
}

pub fn clip_config(node: &Node) -> (Option<f64>, Option<f64>) {
    let mut min_result: Option<f64> = None;
    let mut max_result: Option<f64> = None;

    // For Clip Opset 6+ , the min and max values are attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "min" => {
                let min = value.clone().into_f32() as f64;
                min_result = Some(min);
            }
            "max" => {
                let max = value.clone().into_f32();
                max_result = Some(max as f64);
            }
            _ => {}
        }
    }

    // For Clip Opset 11+ , the min and max values are inputs
    // Get the min and max values from the input values
    if min_result.is_none() && max_result.is_none() {
        let min = &node.inputs[1].value;
        let max = &node.inputs[2].value;

        if min_result.is_none() && min.is_some() {
            let min = min.clone().unwrap().into_scalar();
            min_result = match min {
                Data::Float16(min) => Some(f32::from(min) as f64),
                Data::Float32(min) => Some(min as f64),
                Data::Float64(min) => Some(min),
                _ => panic!("Clip: only float min is supported"),
            };
        }

        if max_result.is_none() && max.is_some() {
            let max = max.clone().unwrap().into_scalar();
            max_result = match max {
                Data::Float16(max) => Some(f32::from(max) as f64),
                Data::Float32(max) => Some(max as f64),
                Data::Float64(max) => Some(max),
                _ => panic!("Clip: only float max is supported"),
            };
        }
    }

    if min_result.is_none() && max_result.is_none() {
        panic!("Clip: min and max values must be either attributes or inputs");
    }

    (min_result, max_result)
}

pub fn reduce_max_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => axes = value.clone().into_i64s(),
            "keepdims" => keepdims = value.clone().into_i64(),
            _ => {}
        }
    }

    if axes.len() > 1 {
        panic!("ReduceMax: reducing on multiple dimensions is not supported")
    }

    if axes.is_empty() && keepdims == 1 {
        panic!("ReduceMax: axes must be provided with keepdims")
    }

    if !axes.is_empty() && keepdims == 0 {
        // Not supported in Burn
        panic!("ReduceMax: the reduce operation must preserve the reduced dimension")
    }

    if axes.is_empty() {
        None
    } else {
        let mut dim = axes[0];

        if dim < 0 {
            // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
            dim += tensor.dim as i64;
        }
        Some(dim as usize)
    }
}

pub fn reduce_min_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => axes = value.clone().into_i64s(),
            "keepdims" => keepdims = value.clone().into_i64(),
            _ => {}
        }
    }

    if axes.len() > 1 {
        panic!("ReduceMin: reducing on multiple dimensions is not supported")
    }

    if axes.is_empty() && keepdims == 1 {
        panic!("ReduceMin: axes must be provided with keepdims")
    }

    if !axes.is_empty() && keepdims == 0 {
        panic!("ReduceMin: the reduce operation must preserve the reduced dimension")
    }

    if axes.is_empty() {
        None
    } else {
        let mut dim = axes[0];

        if dim < 0 {
            dim += tensor.dim as i64;
        }
        Some(dim as usize)
    }
}

pub fn reduce_mean_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => axes = value.clone().into_i64s(),
            "keepdims" => keepdims = value.clone().into_i64(),
            _ => {}
        }
    }

    if axes.len() > 1 {
        panic!("ReduceMean: reducing on multiple dimensions is not supported")
    }

    if axes.is_empty() && keepdims == 1 {
        panic!("ReduceMean: axes must be provided with keepdims")
    }

    if !axes.is_empty() && keepdims == 0 {
        // Not supported in Burn
        panic!("ReduceMean: the reduce operation must preserve the reduced dimension")
    }

    if axes.is_empty() {
        None
    } else {
        let mut dim = axes[0];

        if dim < 0 {
            // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
            dim += tensor.dim as i64;
        }
        Some(dim as usize)
    }
}

pub fn reduce_prod_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => axes = value.clone().into_i64s(),
            "keepdims" => keepdims = value.clone().into_i64(),
            // TODO: handle noop_with_empty_axes (opset 18)
            _ => {}
        }
    }

    if axes.len() > 1 {
        panic!("ReduceProd: reducing on multiple dimensions is not supported")
    }

    if axes.is_empty() && keepdims == 1 {
        panic!("ReduceProd: axes must be provided with keepdims")
    }

    if !axes.is_empty() && keepdims == 0 {
        // Not supported in Burn
        panic!("ReduceProd: the reduce operation must preserve the reduced dimension")
    }

    if axes.is_empty() {
        None
    } else {
        let mut dim = axes[0];

        if dim < 0 {
            // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
            dim += tensor.dim as i64;
        }
        Some(dim as usize)
    }
}

pub fn reduce_sum_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "keepdims" => keepdims = value.clone().into_i64(),
            "axes" => axes = value.clone().into_i64s(),
            // TODO: handle noop_with_empty_axes
            _ => {}
        }
    }

    // TODO: Handle case where axes are passed in. Will require its own ReduceSumNode instead of a UnaryNode.
    if let Some(value) = node
        .inputs
        .get(1)
        .and_then(|argument| argument.value.as_ref())
    {
        axes = value.clone().into_i64s();
    }

    if axes.len() > 1 {
        panic!("ReduceMean: reducing on multiple dimensions is not supported")
    }

    if axes.is_empty() && keepdims == 1 {
        panic!("ReduceMean: axes must be provided with keepdims")
    }

    if !axes.is_empty() && keepdims == 0 {
        // Not supported in Burn
        panic!("ReduceMean: the reduce operation must preserve the reduced dimension")
    }

    if axes.is_empty() {
        None
    } else {
        let mut dim = axes[0];

        if dim < 0 {
            // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
            dim += tensor.dim as i64;
        }
        Some(dim as usize)
    }
}

pub fn shape_config(curr: &Node) -> (usize, usize) {
    if curr.inputs.len() != 1 {
        panic!(
            "Shape: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Default: all axes up to the last one (included)
    let mut start_dim: i64 = 0;
    let mut end_dim: i64 = tensor.dim as i64;

    // Extract the attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "start" => start_dim = value.clone().into_i64(),
            "end" => end_dim = value.clone().into_i64(),
            _ => {}
        }
    }

    // If dim is negative, it is counted from the end
    if start_dim < 0 {
        start_dim += tensor.dim as i64;
    }
    if end_dim < 0 {
        end_dim += tensor.dim as i64;
    }

    (start_dim as usize, end_dim as usize)
}

pub fn slice_config(node: &Node) -> Vec<Option<(i64, i64)>> {
    fn get_input_values(node: &Node, index: usize) -> Vec<i64> {
        // If the input is not provided, return an empty vector
        if node.inputs.get(index).is_none() {
            return Vec::new();
        }

        match &node.inputs[index].value {
            Some(Data::Int64s(shape)) => shape.clone(),

            _ => panic!("Tensor data type must be int64"),
        }
    }

    let mut starts = get_input_values(node, 1);
    let mut ends = get_input_values(node, 2);
    let mut axes = get_input_values(node, 3);
    let mut steps = get_input_values(node, 4);

    // https://burn.dev/docs/burn/prelude/struct.Tensor.html#method.slice
    // TODO default missing axes ranges to the full range of the corresponding axis
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

    // Extract the shape of the input tensor
    let input_dim = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor.dim,
        _ => panic!("Only tensor input is valid"),
    };

    // If axes is not provided, it defaults to all axes
    if axes.is_empty() {
        axes = (0..starts.len() as i64).collect();
    }

    // assert len(starts) == len(ends) == len(axes)
    if starts.len() != ends.len() || starts.len() != axes.len() {
        panic!("Slice: starts, ends, and axes must have the same length");
    }

    // If dim is negative, it is counted from the end
    // Negative value means counting dimensions from the back.
    for axis in &mut axes {
        if *axis < 0 {
            *axis += input_dim as i64;
        }
    }

    // convert starts, ends, and axes to ranges. Use None for missing axes ranges
    let mut ranges: Vec<Option<(i64, i64)>> = vec![None; input_dim];
    for i in 0..axes.len() {
        let axis = axes[i] as usize;
        ranges[axis] = Some((starts[i], ends[i]));
    }

    ranges
}

pub fn transpose_config(curr: &Node) -> Vec<i64> {
    if curr.inputs.len() != 1 {
        panic!(
            "Transpose: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Default: reverse the dimensions
    let mut perm = (0..tensor.dim as i64).rev().collect::<Vec<i64>>();

    if let Some(axes) = curr.attrs.get("perm") {
        perm = axes.clone().into_i64s();
    }

    perm
}

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

    match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    axes
}

pub fn split_config(node: &Node) -> SplitConfig {
    // Axis to split along (default is 0 per ONNX spec)
    let mut axis: i64 = 0;
    let mut split_size: Option<usize> = None;
    let mut split_sizes: Option<Vec<usize>> = None;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "num_outputs" => {
                let num_outputs = value.clone().into_i64() as usize;

                if num_outputs == 0 {
                    panic!("Split error: 'num_outputs' must be greater than zero.");
                }

                let dim_size = tensor.shape.clone().unwrap()[axis as usize];
                let calculated_split_size =
                    dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

                if calculated_split_size == 0 {
                    panic!(
                        "Split error: Computed split size is zero. Ensure 'num_outputs' is valid."
                    );
                }

                split_size = Some(calculated_split_size);
            }
            _ => {}
        }
    }

    if axis < 0 {
        axis += tensor.dim as i64;
    }

    if node.inputs.len() > 1 {
        let split_input_arg = &node.inputs[1];
        if let Some(Data::Int64s(sizes)) = &split_input_arg.value {
            let sizes: Vec<usize> = sizes.iter().map(|&x| x as usize).collect();
            split_sizes = Some(sizes);
        }
    }

    // Only one of 'split_sizes' or 'num_outputs' is provided
    if split_sizes.is_some() && split_size.is_some() {
        panic!("Split: Either 'split' input or 'num_outputs' attribute should be specified, but not both.");
    }

    // Infer split_size if neither split_sizes nor split_size is provided
    if split_sizes.is_none() && split_size.is_none() {
        let num_outputs = node.outputs.len();
        let dim_size = tensor.shape.unwrap()[axis as usize];

        let calculated_split_size =
            dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

        if calculated_split_size == 0 {
            panic!("Split error: Computed split size is zero. Ensure 'num_outputs' is valid.");
        }

        split_size = Some(calculated_split_size);
    }

    SplitConfig {
        axis: axis as usize,
        split_size,
        split_sizes,
    }
}

pub fn one_hot_config(curr: &Node) -> (usize, [f32; 2], i64) {
    let depth = curr.inputs[1]
        .value
        .clone()
        .expect("OneHot: Only constant depth is currently supported")
        .into_i64();

    let values = curr.inputs[2]
        .value
        .clone()
        .expect("OneHot: Only constant on/off values is currently supported")
        .into_f32s();
    let axis = curr
        .attrs
        .get("axis")
        .map(|val| val.clone().into_i64())
        .unwrap_or(-1);
    (depth as usize, values.try_into().unwrap(), axis)
}
