use burn::nn::{
    conv::Conv2dConfig, pool::MaxPool2dConfig, BatchNormConfig, DropoutConfig, LinearConfig,
    PaddingConfig2d,
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

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => attr_value_vec_i64(value, &mut kernel_shape),
            "strides" => attr_value_vec_i64(value, &mut strides),
            "pads" => attr_value_vec_i64(value, &mut pads),
            _ => {}
        }
    }

    let padding = padding_config(&pads);

    MaxPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
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

/// Create a DropoutConfig from the attributes of the node
pub fn dropout_config(node: &Node) -> DropoutConfig {
    // the dropout probability comes as input, which is copied to state.

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
