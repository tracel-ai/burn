use burn::nn::{
    conv::{Conv1dConfig, Conv2dConfig, ConvTranspose2dConfig},
    pool::{AvgPool1dConfig, AvgPool2dConfig, MaxPool1dConfig, MaxPool2dConfig},
    BatchNormConfig, DropoutConfig, LayerNormConfig, LinearConfig, PaddingConfig1d,
    PaddingConfig2d,
};

use super::ir::{ArgType, AttributeValue, Data, Node};
use crate::burn::node::resize::ResizeMode;

/// Create a Conv1dConfig from the attributes of the node
pub fn conv1d_config(curr: &Node) -> Conv1dConfig {
    let mut kernel_shape = Vec::new(); // TODO default inferred from weight tensor per spec
    let mut strides = vec![1];
    let mut pads = vec![0, 0];
    let mut dilations = vec![1];
    let mut group: i64 = 1;

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("Conv1d: weight tensor must be present");
    };

    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    // the channels are inverted in the weight tensor
    let shape = weight.shape.clone().unwrap();
    let channels_in = shape[1];
    let channels_out = shape[0];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64(),
            _ => {}
        }
    }

    let padding = padding_config_1d(&pads);

    Conv1dConfig::new(channels_in, channels_out, kernel_shape[0] as usize)
        .with_stride(strides[0] as usize)
        .with_dilation(dilations[0] as usize)
        .with_groups(group as usize)
        .with_bias(bias)
        .with_padding(padding)
}

/// Create a Conv2dConfig from the attributes of the node
pub fn conv2d_config(curr: &Node) -> Conv2dConfig {
    let mut kernel_shape = Vec::new(); // TODO default inferred from weight tensor per spec
    let mut strides = vec![1, 1];
    let mut pads = vec![0, 0, 0, 0];
    let mut dilations = vec![1, 1];
    let mut group: i64 = 1;

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let weight = if let ArgType::Tensor(ref weight) = curr.inputs[1].ty {
        weight
    } else {
        panic!("Conv1d: weight tensor must be present");
    };
    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    // the channels are inverted in the weight tensor
    let shape = weight.shape.clone().unwrap();
    let channels: [usize; 2] = [shape[1], shape[0]];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64(),
            _ => {}
        }
    }

    let padding = padding_config(&pads);

    Conv2dConfig::new(
        channels,
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
    )
    .with_stride([strides[0] as usize, strides[1] as usize])
    .with_dilation([dilations[0] as usize, dilations[1] as usize])
    .with_groups(group as usize)
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

    let padding = padding_config(&pads);

    MaxPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
        .with_dilation([dilations[0] as usize, dilations[1] as usize])
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
        .unwrap_or_else(|| vec![0, 0]);
    let dilations = attrs
        .remove("dilations")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1, 1]);
    let group = attrs
        .remove("group")
        .map(AttributeValue::into_i64)
        .unwrap_or(1);

    // Trick with remove + empty check is simplest way to not forget some attribute for runtime:
    if !attrs.is_empty() {
        panic!("Not all attributes are used: {attrs:?}");
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
    let channels: [usize; 2] = [shape[1], shape[0]];

    ConvTranspose2dConfig::new(
        channels,
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
    )
    .with_stride([stride[0] as usize, stride[1] as usize])
    .with_padding([pads[0] as usize, pads[1] as usize])
    .with_dilation([dilations[0] as usize, dilations[1] as usize])
    .with_groups(group as usize)
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

    let padding = padding_config(&pads);

    AvgPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
        .with_count_include_pad(count_include_pad == 1)
}

pub fn expand_config(node: &Node) -> Vec<i64> {
    let input_value = &node.inputs[1].value;
    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.dim, 1, "Expand: shape tensor must be 1D");
            if let Some(Data::Int64s(shape)) = input_value.as_ref() {
                shape.clone()
            } else {
                panic!("Tensor data type must be int64")
            }
        }
        _ => panic!("Only tensor input is valid for shape"),
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
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
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
        dim += tensor.dim as i64;
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
fn padding_config(pads: &[i64]) -> PaddingConfig2d {
    let [left, top, right, bottom] = [pads[0], pads[1], pads[2], pads[3]];

    if left < 0 || top < 0 || right < 0 || bottom < 0 {
        panic!("Negative pad values are not supported");
    } else if (left != right) || (top != bottom) {
        panic!("Asymmetric padding is not supported");
    } else if left == top && top == right && right == bottom && bottom == 0 {
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

pub fn resize_config(node: &Node) -> ResizeMode {
    let mut mode: String = "".to_string();
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "coordinate_transformation_mode" => {}
            "cubic_coeff_a" => {}
            "mode" => mode = value.clone().into_string(),
            "nearest_mode" => {}
            _ => {}
        }
    }

    let mode = match mode.as_str() {
        "nearest" => ResizeMode::Nearest,
        "linear" => ResizeMode::Linear,
        "cubic" => ResizeMode::Cubic,
        _ => panic!("Resize: invalid mode string, must be 'nearest', 'linear', or 'cubic'"),
    };

    mode
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
    } else if left == right && right == 0 {
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

pub fn slice_config(node: &Node) -> (Vec<usize>, Vec<usize>) {
    let start_value = &node.inputs[1].value;
    let end_value = &node.inputs[2].value;

    let starts = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.dim, 1, "Slice: ends tensor must be 1D");
            if let Some(Data::Int64s(shape)) = start_value.as_ref() {
                shape
                    .iter()
                    .map(|x| {
                        assert!(*x >= 0, "Slice: start must be positive");
                        *x as usize
                    })
                    .collect()
            } else {
                panic!("Tensor data type must be int64")
            }
        }
        _ => panic!("Only tensor input is valid for shape"),
    };

    let ends = match &node.inputs[2].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.dim, 1, "Slice: ends tensor must be 1D");
            if let Some(Data::Int64s(shape)) = end_value.as_ref() {
                shape
                    .iter()
                    .map(|x| {
                        assert!(*x >= 0, "Slice: end must be positive");
                        *x as usize
                    })
                    .collect()
            } else {
                panic!("Tensor data type must be int64")
            }
        }
        _ => panic!("Only tensor input is valid for shape"),
    };

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => {
                let mut i = 0;
                value.clone().into_i64s().iter().for_each(|x| {
                    assert_eq!(*x, i, "Slice: axes must be consecutive");
                    i += 1;
                })
            }
            "steps" => value.clone().into_i64s().into_iter().for_each(|x| {
                if x != 1 {
                    panic!("Slice: steps other than 1 are not supported");
                }
            }),
            _ => {}
        }
    }

    (starts, ends)
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
