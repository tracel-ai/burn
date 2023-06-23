use burn::nn::{
    conv::{Conv2dConfig, Conv2dPaddingConfig},
    BatchNormConfig, LinearConfig,
};

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
    let mut strides = Vec::new();
    let mut pads = Vec::new();
    let mut dilations = Vec::new();
    let mut group: i64 = 0;

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

    let padding = if pads.iter().all(|&x| x == 0) {
        Conv2dPaddingConfig::Valid
    } else {
        todo!("Conv2d: padding({pads:?}) is not fully supported");
    };

    if strides.iter().all(|&x| x != 1) {
        todo!("Conv2d: strides({strides:?}) are not fully supported");
    };

    if dilations.iter().all(|&x| x != 1) {
        todo!("Conv2d: dilations({dilations:?}) are not fully supported");
    };

    if group != 1 {
        todo!("Conv2d: group ({group}) is not fully supported");
    };

    Conv2dConfig::new(
        channels,
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
    )
    .with_bias(bias)
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
