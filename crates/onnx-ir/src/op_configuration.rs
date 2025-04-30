// TODO Move op_configuration.rs from burn-import to onnx-ir #3091
// See https://github.com/tracel-ai/burn/issues/3091

use crate::ir::{ArgType, Data, Node, TensorData};

// pub fn expand_config(node: &Node) -> ExpandShape {
//     match &node.inputs[1].ty {
//         ArgType::Tensor(tensor) => {
//             assert_eq!(tensor.rank, 1, "Expand: shape tensor must be 1D");
//             assert!(
//                 matches!(tensor.elem_type, ElementType::Int64),
//                 "Expand: shape tensor must have element type int64"
//             );
//         }
//         ArgType::Shape(_) => {
//             // Shapes are always 1-D int64 data, so nothing to assert here
//         }
//         _ => panic!("Only tensor input is valid for shape"),
//     }

//     match &node.inputs[1].value {
//         Some(TensorData {
//             data: Data::Int64s(shape),
//             ..
//         }) => ExpandShape::Static(shape.clone()),
//         None => {
//             // we were unable to statically determine the input value, so we'll need to fetch it at runtime
//             ExpandShape::Runtime(crate::burn::Type::from(&node.inputs[1]))
//         }
//         _ => panic!(
//             "Shape data type must be int64, is {:?}",
//             &node.inputs[1].value
//         ),
//     }
// }

// /// Create a TileConfig from the attributes of the node
// pub fn tile_config(node: &Node) -> TileConfig {
//     let repeat = node
//         .inputs
//         .get(1)
//         .map(|input| {
//             if let Some(TensorData { data, .. }) = &input.value {
//                 data.clone()
//                     .into_i64s()
//                     .iter()
//                     .map(|&x| x as usize)
//                     .collect()
//             } else {
//                 vec![]
//             }
//         })
//         .unwrap_or_default();
//     TileConfig::new(repeat)
// }

// /// Create a TopKConfig from the attributes of the node.
// pub fn top_k_config(node: &Node) -> TopKConfig {
//     // extract the shape of the input data tensor
//     let data_tensor = match node.inputs.first().unwrap().clone().ty {
//         ArgType::Tensor(tensor) => tensor,
//         _ => panic!("Only tensor input is valid"),
//     };

//     let k = match node.inputs.get(1) {
//         Some(k_tensor) => k_tensor
//             .clone()
//             .value
//             .expect("TopK: only constant 'k' tensor is currently supported")
//             .data
//             .into_i64s()[0],
//         _ => node
//             .attrs
//             .get("k")
//             .expect("TopK: number of top elements 'k' is missing")
//             .clone()
//             .into_i64(),
//     };

//     let mut axis = match node.attrs.get("axis") {
//         Some(axis) => axis.clone().into_i64(),
//         None => -1,
//     };

//     // if axis is negative, it is counted from the end
//     if axis < 0 {
//         axis += data_tensor.rank as i64;
//     }

//     if let Some(largest) = node.attrs.get("largest") {
//         if largest.clone().into_i64() != 1 {
//             unimplemented!("TopK: only largest elements is supported")
//         }
//     };

//     if let Some(sorted) = node.attrs.get("sorted") {
//         if sorted.clone().into_i64() != 1 {
//             unimplemented!("TopK: only sorted elements is supported")
//         }
//     };

//     TopKConfig::new(axis as usize, k as usize)
// }

// /// Create a TriluConfig from the attributes of the node
// pub fn trilu_config(node: &Node) -> TriluConfig {
//     let mut upper = true;
//     let mut diagonal = 0;
//     for (key, value) in node.attrs.iter() {
//         match key.as_str() {
//             "upper" => upper = value.clone().into_i64() != 0,
//             _ => {}
//         }
//     }
//     // The second input of the Trilu node is the diagonal value, coming from a constant node
//     if let Some(diagonal_arg) = node.inputs.get(1) {
//         if let Some(TensorData {
//             data: Data::Int64(diagonal_val),
//             ..
//         }) = &diagonal_arg.value
//         {
//             diagonal = *diagonal_val;
//         }
//     }
//     TriluConfig::new(upper, diagonal)
// }

// /// Create a PadConfig from the attributes of the node
// pub fn pad_config(node: &Node) -> PadConfig {
//     fn get_pads_input(node: &Node) -> Vec<i64> {
//         match &node.inputs[1].value {
//             Some(TensorData { data, .. }) => data.clone().into_i64s(),
//             _ => Vec::new(),
//         }
//     }
//     fn get_pads(node: &Node) -> Vec<usize> {
//         if node.inputs.is_empty() {
//             panic!("Pad: must provide data as input")
//         }
//         if node.inputs.len() >= 4 {
//             panic!("Pad: axes input is not supported")
//         }

//         let input_dim = match &node.inputs.first().unwrap().ty {
//             ArgType::Tensor(tensor) => tensor.rank,
//             _ => panic!("Pad: Only tensor input is valid"),
//         };

//         //TODO : handle more possible attributes
//         let mut pads: Vec<usize> = get_pads_input(node)
//             .into_iter()
//             .map(|x| x as usize)
//             .collect();

//         for (key, value) in node.attrs.iter() {
//             match key.as_str() {
//                 "pads" => {
//                     pads = value
//                         .clone()
//                         .into_i64s()
//                         .iter()
//                         .map(|&x| {
//                             if x < 0 {
//                                 panic!("Pad: Negative pad is not supported");
//                             }
//                             x as usize
//                         })
//                         .collect()
//                 }
//                 "mode" => {
//                     let mode = value.clone().into_string();
//                     if mode != "constant" {
//                         panic!("only constant mode is supported, given mode is {}", mode);
//                     }
//                 }

//                 _ => {}
//             }
//         }

//         if pads.is_empty() {
//             panic!("Pad: pads should be given as attribute or as input");
//         }

//         if pads.len() != input_dim * 2 {
//             panic!("Pad: pads should be a 1D tensor of shape [2 * num_axes]");
//         }
//         // TODO: Burn's pad should support 1D tensor
//         if input_dim < 2 {
//             panic!("Pad: input tensor should be rank 2 or higher");
//         }

//         let left_index = input_dim - 1;
//         let top_index = input_dim - 2;
//         let right_index = pads.len() - 1;
//         let bottom_index = pads.len() - 2;
//         let index_list = [left_index, top_index, right_index, bottom_index];

//         for (index, &item) in pads.iter().enumerate() {
//             if !index_list.contains(&index) && item != 0 {
//                 panic!(
//                     "Pad: padding will only be applied to the last two dimensions but found non zero padding for other dimensions"
//                 );
//             }
//         }

//         let left = pads[left_index];
//         let top = pads[top_index];
//         let right = pads[right_index];
//         let bottom = pads[bottom_index];
//         vec![left, right, top, bottom]
//     }
//     fn get_constant_value(node: &Node) -> f32 {
//         // TODO: support int, boolean
//         let mut constant_value = node.inputs
//                 .get(2)
//                 .and_then(|input| match &input.value.as_ref().expect("Value input must be present").data {
//                     Data::Float16s(constant_value) => {
//                         constant_value.first().map(|&f| f32::from(f))
//                     }
//                     Data::Float32s(constant_value) => {
//                         constant_value.first().copied()
//                     }
//                     Data::Float64s(constant_value) => {
//                         constant_value.first().map(|&f| f as f32)
//                     }
//                     Data::Float16(constant_value) => Some(f32::from(*constant_value)),
//                     Data::Float32(constant_value) => Some(*constant_value),
//                     Data::Float64(constant_value) => Some(*constant_value as f32),
//                      _ => panic!("Pad: only float values are currently supported for constant value, submit an issue on github"),
//                 })
//                 .unwrap_or(0.0);

//         if node.attrs.contains_key("value") {
//             constant_value = node.attrs.get("value").map(|value| match value {
//                 AttributeValue::Float32(value) => *value,
//                 _ => panic!("Pad: only float32 values are currently supported for constant value as attribute, submit an issue on github"),
//             }).expect("constant_value should have had a value now");
//         }
//         constant_value
//     }

//     let pads = get_pads(node);
//     let constant_value = get_constant_value(node);

//     PadConfig::new(pads, constant_value)
// }

// Create a LeakyReluConfig from the alpha attribute of the node
pub fn leaky_relu_config(node: &Node) -> f64 {
    let mut alpha = 0.01;

    for (key, value) in node.attrs.iter() {
        if key.as_str() == "alpha" {
            alpha = value.clone().into_f32() as f64
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
        if key.as_str() == "allowzero" {
            allowzero = value.clone().into_i64()
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

    match &node.inputs[1].value {
        Some(TensorData { data, shape, .. }) => {
            assert_eq!(shape.len(), 1, "Reshape: shape tensor must be 1D");
            data.clone().into_i64s()
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
            if let Some(TensorData { data, .. }) = &input.value {
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
            if let Some(TensorData { data, .. }) = &input.value {
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
            if let Some(TensorData { data, .. }) = &input.value {
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
        assert!(scales.len() == input.rank);
        // ignore the fist two items from scales
        // because they are the batch and channel dimensions
        scales = scales.iter().skip(2).cloned().collect();
    }

    if !sizes.is_empty() {
        assert!(sizes.len() == input.rank);
        // ignore the fist two items from sizes
        // because they are the batch and channel dimensions
        sizes = sizes.iter().skip(2).cloned().collect();
    }

    (mode, scales, sizes)
}

// //Note this function should only execute if the second input is a constant
// //if it wasn't and the output shape was known, unsqueeze has been remapped to reshape
// pub fn unsqueeze_config(node: &Node) -> UnsqueezeAxes {
//     // Check if axes attribute exists
//     for (key, value) in node.attrs.iter() {
//         match key.as_str() {
//             "axes" => return UnsqueezeAxes::Static(value.clone().into_i64s()),
//             _ => {}
//         }
//     }

//     assert!(
//         !node.inputs.is_empty(),
//         "Unsqueeze: axes tensor must be present"
//     );

//     let input_value = &node.inputs[1];

//     match &node.inputs[1].ty {
//         ArgType::Tensor(tensor) => {
//             assert_eq!(tensor.rank, 1, "Unsqueeze: axes tensor must be 1D");
//             if let Some(TensorData {
//                 data: Data::Int64s(shape),
//                 ..
//             }) = input_value.value.as_ref()
//             {
//                 UnsqueezeAxes::Static(shape.clone())
//             } else {
//                 UnsqueezeAxes::Runtime(crate::burn::Type::from(&node.inputs[1]))
//             }
//         }
//         _ => panic!("Arg for unsqueeze must be tensor or scalar"),
//     }
// }

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
            let min = min.clone().unwrap().data.into_scalar();
            min_result = match min {
                Data::Float16(min) => Some(f32::from(min) as f64),
                Data::Float32(min) => Some(min as f64),
                Data::Float64(min) => Some(min),
                _ => panic!("Clip: only float min is supported"),
            };
        }

        if max_result.is_none() && max.is_some() {
            let max = max.clone().unwrap().data.into_scalar();
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
            dim += tensor.rank as i64;
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
            dim += tensor.rank as i64;
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
            dim += tensor.rank as i64;
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
            dim += tensor.rank as i64;
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
        axes = value.clone().data.into_i64s();
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
            dim += tensor.rank as i64;
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
    let mut end_dim: i64 = tensor.rank as i64;

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
        start_dim += tensor.rank as i64;
    }
    if end_dim < 0 {
        end_dim += tensor.rank as i64;
    }

    (start_dim as usize, end_dim as usize)
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
    let mut perm = (0..tensor.rank as i64).rev().collect::<Vec<i64>>();

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
// pub fn split_config(node: &Node) -> SplitConfig {
//     // Initialize the axis to split along (default is 0 as per ONNX specification)
//     let mut axis: i64 = 0;
//     // Holds the uniform split size if calculated or provided
//     let mut split_size: Option<usize> = None;
//     // Holds the custom split sizes if provided as input
//     let mut split_sizes: Option<Vec<usize>> = None;

//     // Extract the input tensor type to determine rank and shape
//     let tensor = match node.inputs.first().unwrap().ty {
//         ArgType::Tensor(ref tensor) => tensor,
//         _ => panic!("Split: Input must be a valid tensor"),
//     };

//     // Optionally store the number of outputs if provided as an attribute
//     let mut num_outputs: Option<usize> = None;

//     // Iterate through node attributes to extract relevant values
//     for (key, value) in node.attrs.iter() {
//         match key.as_str() {
//             "axis" => axis = value.clone().into_i64(),
//             "num_outputs" => num_outputs = Some(value.clone().into_i64() as usize),
//             _ => {}
//         }
//     }

//     // Handle the case when num_outputs is provided to calculate uniform split size
//     if let Some(num_outputs) = num_outputs {
//         if num_outputs == 0 {
//             panic!("Split: 'num_outputs' must be a positive value greater than zero");
//         }

//         let dim_size = tensor
//             .static_shape
//             .as_ref()
//             .expect("Split: Static shape must be known to calculate split size")[axis as usize];

//         // Calculate the split size considering any remainder for non-evenly divisible dimensions
//         let calculated_split_size =
//             dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

//         if calculated_split_size == 0 {
//             panic!(
//                 "Split: Calculated split size is zero. Please ensure 'num_outputs' is valid for the dimension size"
//             );
//         }

//         // Assign the calculated split size
//         split_size = Some(calculated_split_size);
//     }

//     // Adjust axis if negative to count from the end as per ONNX spec
//     if axis < 0 {
//         axis += tensor.rank as i64;
//     }

//     // Check for custom split sizes provided as a second input
//     if node.inputs.len() > 1 && node.inputs[1].value.is_some() {
//         let sizes = node.inputs[1]
//             .value
//             .as_ref()
//             .unwrap()
//             .data
//             .clone()
//             .into_usizes();

//         if !sizes.is_empty() {
//             split_sizes = Some(sizes);
//         }
//     }

//     // Ensure that only one of 'split_sizes' or 'num_outputs' is specified
//     if split_sizes.is_some() && split_size.is_some() {
//         panic!(
//             "Split: Cannot specify both 'split' input and 'num_outputs' attribute simultaneously"
//         );
//     }

//     // Infer split_size if neither custom split_sizes nor split_size is provided
//     if split_sizes.is_none() && split_size.is_none() {
//         let num_outputs = node.outputs.len();
//         let dim_size = tensor
//             .static_shape
//             .as_ref()
//             .expect("Split: Static shape must be known to infer split size")[axis as usize];

//         // Calculate inferred split size based on number of outputs
//         let calculated_split_size =
//             dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

//         if calculated_split_size == 0 {
//             panic!(
//                 "Split: Inferred split size is zero. Please ensure the number of outputs is valid for the dimension size"
//             );
//         }

//         split_size = Some(calculated_split_size);
//     }

//     // Return the configuration for splitting operation
//     SplitConfig {
//         axis: axis as usize,
//         split_size,
//         split_sizes,
//     }
// }

pub fn one_hot_config(curr: &Node) -> (usize, [f32; 2], i64) {
    let depth = curr.inputs[1]
        .value
        .clone()
        .expect("OneHot: Only constant depth is currently supported")
        .data
        .into_i64();

    let values = curr.inputs[2]
        .value
        .clone()
        .expect("OneHot: Only constant on/off values is currently supported")
        .data
        .into_f32s();

    let axis = curr
        .attrs
        .get("axis")
        .map(|val| val.clone().into_i64())
        .unwrap_or(-1);

    (depth as usize, values.try_into().unwrap(), axis)
}

pub fn gemm_config(curr: &Node) -> (f32, f32, i64, i64) {
    let alpha = curr
        .attrs
        .get("alpha")
        .map(|val| val.clone().into_f32())
        .unwrap_or(1.0);
    let beta = curr
        .attrs
        .get("beta")
        .map(|val| val.clone().into_f32())
        .unwrap_or(1.0);
    let trans_a = curr
        .attrs
        .get("transA")
        .map(|val| val.clone().into_i64())
        .unwrap_or(0);
    let trans_b = curr
        .attrs
        .get("transB")
        .map(|val| val.clone().into_i64())
        .unwrap_or(0);

    (alpha, beta, trans_a, trans_b)
}
