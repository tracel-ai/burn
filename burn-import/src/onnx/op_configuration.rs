use burn::nn::{
    conv::Conv1dConfig,
    conv::Conv2dConfig,
    pool::{AvgPool2dConfig, MaxPool2dConfig},
    BatchNormConfig, DropoutConfig, LinearConfig, PaddingConfig1d, PaddingConfig2d,
};

use crate::onnx::ir::TensorData;

use super::ir::{ArgType, AttributeValue, Node, StateType};

#[inline(always)]
pub fn attr_value_vec_i64(value: &AttributeValue, target: &mut Vec<i64>) {
    if let AttributeValue::Int64s(val) = value {
        *target = val.clone();
    }
}

#[inline(always)]
pub fn attr_value_i64(value: &AttributeValue, target: &mut i64) {
    if let AttributeValue::Int64(val) = value {
        *target = *val;
    }
}

#[inline(always)]
pub fn attr_value_f32(value: &AttributeValue, target: &mut f32) {
    if let AttributeValue::Float32(val) = value {
        *target = *val;
    }
}

/// Create a Conv1dConfig from the attributes of the node
pub fn conv1d_config(curr: &Node) -> Conv1dConfig {
    let mut kernel_shape = 1;
    let mut strides = 1;
    let mut pads = vec![0];
    let mut dilations = 1;
    let mut group: i64 = 1;

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let StateType::Tensor(tensor) = curr.states.get(0).unwrap().clone().ty;

    // check if the bias is present
    let bias = curr.states.len() == 2;

    // the channels are inverted in the weight tensor
    let shape = tensor.shape.unwrap();
    let channels_in = shape[1];
    let channels_out = shape[0];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => attr_value_i64(value, &mut kernel_shape),
            "strides" => attr_value_i64(value, &mut strides),
            "pads" => attr_value_vec_i64(value, &mut pads),
            "dilations" => attr_value_i64(value, &mut dilations),
            "group" => attr_value_i64(value, &mut group),
            _ => {}
        }
    }

    let padding = padding_config_1d(&pads);

    Conv1dConfig::new(channels_in, channels_out, kernel_shape as usize)
        .with_stride(strides as usize)
        .with_dilation(dilations as usize)
        .with_groups(group as usize)
        .with_bias(bias)
        .with_padding(padding)
}

/// Create a Conv2dConfig from the attributes of the node
pub fn conv2d_config(curr: &Node) -> Conv2dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = vec![1, 1];
    let mut pads = Vec::new();
    let mut dilations = vec![1, 1];
    let mut group: i64 = 1;

    // extract the channels from the weight tensor's shape [out_channels, in_channels, ...]
    let StateType::Tensor(tensor) = curr.states.get(0).unwrap().clone().ty;

    // check if the bias is present
    let bias = curr.states.len() == 2;

    // the channels are inverted in the weight tensor
    let shape = tensor.shape.unwrap();
    let channels: [usize; 2] = [shape[1], shape[0]];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => attr_value_vec_i64(value, &mut kernel_shape),
            "strides" => attr_value_vec_i64(value, &mut strides),
            "pads" => attr_value_vec_i64(value, &mut pads),
            "dilations" => attr_value_vec_i64(value, &mut dilations),
            "group" => attr_value_i64(value, &mut group),
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
pub fn max_pool2d_config(curr: &Node) -> MaxPool2dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = Vec::new();
    let mut pads = Vec::new();
    let mut dilations = Vec::new();

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => attr_value_vec_i64(value, &mut kernel_shape),
            "strides" => attr_value_vec_i64(value, &mut strides),
            "pads" => attr_value_vec_i64(value, &mut pads),
            "dilations" => attr_value_vec_i64(value, &mut dilations),
            _ => {}
        }
    }

    if !dilations.is_empty() && (dilations[0] != 1 || dilations[1] != 1) {
        todo!("MaxPool2d: dilations are not supported. See https://github.com/burn-rs/burn/issues/622");
    }

    let padding = padding_config(&pads);

    MaxPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
}

/// Create a AvgPool2dConfig from the attributes of the node
pub fn avg_pool2d_config(curr: &Node) -> AvgPool2dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = Vec::new();
    let mut pads = Vec::new();
    let mut count_include_pad: i64 = 0;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => attr_value_vec_i64(value, &mut kernel_shape),
            "strides" => attr_value_vec_i64(value, &mut strides),
            "pads" => attr_value_vec_i64(value, &mut pads),
            "count_include_pad" => attr_value_i64(value, &mut count_include_pad),
            _ => {}
        }
    }

    let padding = padding_config(&pads);

    if count_include_pad == 1 && padding != PaddingConfig2d::Valid {
        todo!("AvgPool2d: count_include_pad is not supported. See https://github.com/burn-rs/burn/issues/636");
    }

    AvgPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
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
    let tensor = match curr.inputs.get(0).unwrap().clone().ty {
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
            "axis" => attr_value_i64(value, &mut start_dim),
            _ => {}
        }
    }

    // if beg_dim is negative, it is counted from the end
    if start_dim < 0 {
        start_dim += tensor.dim as i64;
    }

    (start_dim as usize, end_dim)
}

/// Create a LinearConfig from the attributes of the node
pub fn linear_config(node: &Node) -> LinearConfig {
    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "Linear: multiple inputs are not supported (got {:?})",
            node.inputs.len()
        );
    }

    if node.states.is_empty() {
        panic!("Linear: no state found");
    }

    // extract the shape of the weight tensor
    let StateType::Tensor(tensor) = node.states.get(0).unwrap().clone().ty;

    // check if the weight tensor has at least 2 dimensions
    if tensor.dim < 2 {
        panic!(
            "Linear: weight tensor must have at least 2 dimensions (got {:?})",
            tensor.dim
        );
    }
    let shape = tensor.shape.unwrap();
    let (in_size, out_size) = (shape[0], shape[1]);

    // check if the bias is present
    let bias = node.states.len() == 2;

    LinearConfig::new(in_size, out_size).with_bias(bias)
}

/// Create a DropoutConfig from an attribute and state of the node
pub fn dropout_config(node: &Node) -> DropoutConfig {
    // Opset 7 and older store probability as an attribute
    if node.attrs.contains_key("ratio") {
        let mut prob: f32 = 0.0;
        attr_value_f32(node.attrs.get("ratio").unwrap(), &mut prob);

        return DropoutConfig::new(prob as f64);
    }

    if node.states.is_empty() {
        panic!("Dropout: no state found needed for configuration");
    }

    // extract the tensor from the state
    let StateType::Tensor(tensor) = node.states.get(0).unwrap().clone().ty;

    // Zero dim tensor is treated as a scalar
    assert_eq!(tensor.dim, 0);

    let prob = match tensor.data.unwrap() {
        TensorData::Float32(prob) => *prob.first().unwrap() as f64,
        TensorData::Float64(prob) => *prob.first().unwrap(),
        _ => panic!("Dropout: only float probability is supported"),
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
    let tensor = match node.inputs.get(0).unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => attr_value_i64(value, &mut axis),
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
    let tensor = match node.inputs.get(0).unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => attr_value_i64(value, &mut axis),
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
    let tensor = match node.inputs.get(0).unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => attr_value_i64(value, &mut axis),
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
    let StateType::Tensor(tensor) = node.states.get(0).unwrap().clone().ty;

    let num_features: usize = tensor.shape.unwrap()[0];

    let mut epsilon = 0f32;
    let mut momentum = 0f32;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "momentum" => attr_value_f32(value, &mut momentum),
            "epsilon" => attr_value_f32(value, &mut epsilon),
            _ => {}
        }
    }

    BatchNormConfig::new(num_features)
        .with_epsilon(epsilon as f64)
        .with_momentum(momentum as f64)
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

pub fn reshape_config(node: &Node) -> Vec<i64> {
    let mut allowzero = 0;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "allowzero" => attr_value_i64(value, &mut allowzero),
            _ => {}
        }
    }

    // Burn does not support zero size shape (0 means false in ONNX)
    // (see https://onnx.ai/onnx/operators/onnx__Reshape.html#attributes)
    if allowzero != 0 {
        panic!("Zero shape size is not supported");
    }

    let shape = match node.states.first() {
        Some(state) => match &state.ty {
            StateType::Tensor(tensor) => match tensor.data.as_ref() {
                Some(TensorData::Int64(data)) => data.clone(),
                _ => panic!("Reshape: invalid state data for shape"),
            },
        },
        None => panic!("Reshape: missing state required for shape"),
    };

    shape
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
