//! This file contains the configuration structs and functions for various ONNX operations.

// use crate::burn::Type;
use onnx_ir::ir::{ArgType, Argument, AttributeValue, Data, ElementType, Node, TensorData};

use crate::burn::Type;

/// Configuration for the Pad operation.
#[derive(Debug, Clone, new)]
pub struct PadConfig {
    /// The paddings to be applied to each dimension.
    pub pads: Vec<usize>,
    /// The constant value to fill the padded areas with.
    pub constant_value: f32,
}

/// Shape information for the Expand operation.
#[derive(Debug, Clone)]
pub enum ExpandShape {
    /// Static shape information known at compile time.
    Static(Vec<i64>),
    /// Runtime shape that will be determined during execution.
    Runtime(Argument),
}

/// Creates an ExpandShape configuration from the given Node.
///
/// Extracts shape information from the node's second input to determine
/// whether to use static or runtime shape expansion.
pub fn expand_config(node: &Node) -> ExpandShape {
    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.rank, 1, "Expand: shape tensor must be 1D");
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

    match &node.inputs[1].value {
        Some(TensorData {
            data: Data::Int64s(shape),
            ..
        }) => ExpandShape::Static(shape.clone()),
        None => {
            // we were unable to statically determine the input value, so we'll need to fetch it at runtime
            ExpandShape::Runtime(node.inputs[1].clone())
        }
        _ => panic!(
            "Shape data type must be int64, is {:?}",
            &node.inputs[1].value
        ),
    }
}

/// Configuration for the Split operation.
#[derive(Clone, Debug, new)]
pub struct SplitConfig {
    /// The axis along which to split the input tensor.
    pub axis: usize,
    /// The uniform size of each split when splitting evenly.
    pub split_size: Option<usize>,
    /// Custom sizes for each split when splitting unevenly.
    pub split_sizes: Option<Vec<usize>>,
}

/// Configuration for the Tile operation.
#[derive(Debug, Clone, new)]
pub struct TileConfig {
    /// The number of times to repeat each dimension.
    pub repeats: Vec<usize>,
}

/// Axes specification for the Unsqueeze operation.
#[derive(Debug, Clone, new)]
pub enum UnsqueezeAxes {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes that will be determined during execution.
    Runtime(Type),
}

/// Creates a TileConfig from the node attributes and inputs.
pub fn tile_config(node: &Node) -> TileConfig {
    let repeat = node
        .inputs
        .get(1)
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
    TileConfig::new(repeat)
}

/// Configuration for the TopK operation.
#[derive(Debug, Clone, new)]
pub struct TopKConfig {
    /// The axis along which to perform the top-k selection.
    pub axis: usize,
    /// The number of top elements to select.
    pub k: usize,
}

/// Creates a TopKConfig from the node attributes and inputs.
pub fn top_k_config(node: &Node) -> TopKConfig {
    // Extract the shape of the input data tensor
    let data_tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    let k = match node.inputs.get(1) {
        Some(k_tensor) => k_tensor
            .clone()
            .value
            .expect("TopK: only constant 'k' tensor is currently supported")
            .data
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

    // If axis is negative, it is counted from the end
    if axis < 0 {
        axis += data_tensor.rank as i64;
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

/// Configuration for the Trilu operation.
#[derive(Debug, Clone, new)]
pub struct TriluConfig {
    /// Whether to return the upper triangular matrix.
    pub upper: bool,
    /// The diagonal offset.
    pub diagonal: i64,
}

/// Creates a TriluConfig from the node attributes and inputs.
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
        if let Some(TensorData {
            data: Data::Int64(diagonal_val),
            ..
        }) = &diagonal_arg.value
        {
            diagonal = *diagonal_val;
        }
    }
    TriluConfig::new(upper, diagonal)
}

/// Creates a PadConfig from the node attributes and inputs.
pub fn pad_config(node: &Node) -> PadConfig {
    fn get_pads_input(node: &Node) -> Vec<i64> {
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

/// Creates UnsqueezeAxes configuration from the node attributes.
///
/// Note: This function should only execute if the second input is a constant.
/// If it wasn't and the output shape was known, unsqueeze has been remapped to reshape.
pub fn unsqueeze_config(node: &Node) -> UnsqueezeAxes {
    // Check if axes attribute exists
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => return UnsqueezeAxes::Static(value.clone().into_i64s()),
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
            assert_eq!(tensor.rank, 1, "Unsqueeze: axes tensor must be 1D");
            if let Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) = input_value.value.as_ref()
            {
                UnsqueezeAxes::Static(shape.clone())
            } else {
                UnsqueezeAxes::Runtime(crate::burn::Type::from(&node.inputs[1]))
            }
        }
        _ => panic!("Arg for unsqueeze must be tensor or scalar"),
    }
}

/// Creates a SplitConfig from the node attributes and inputs.
pub fn split_config(node: &Node) -> SplitConfig {
    // Initialize the axis to split along (default is 0 as per ONNX specification)
    let mut axis: i64 = 0;
    // Holds the uniform split size if calculated or provided
    let mut split_size: Option<usize> = None;
    // Holds the custom split sizes if provided as input
    let mut split_sizes: Option<Vec<usize>> = None;

    // Extract the input tensor type to determine rank and shape
    let tensor = match node.inputs.first().unwrap().ty {
        ArgType::Tensor(ref tensor) => tensor,
        _ => panic!("Split: Input must be a valid tensor"),
    };

    // Optionally store the number of outputs if provided as an attribute
    let mut num_outputs: Option<usize> = None;

    // Iterate through node attributes to extract relevant values
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "num_outputs" => num_outputs = Some(value.clone().into_i64() as usize),
            _ => {}
        }
    }

    // Handle the case when num_outputs is provided to calculate uniform split size
    if let Some(num_outputs) = num_outputs {
        if num_outputs == 0 {
            panic!("Split: 'num_outputs' must be a positive value greater than zero");
        }

        let dim_size = tensor
            .static_shape
            .as_ref()
            .expect("Split: Static shape must be known to calculate split size")[axis as usize];

        // Calculate the split size considering any remainder for non-evenly divisible dimensions
        let calculated_split_size =
            dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

        if calculated_split_size == 0 {
            panic!(
                "Split: Calculated split size is zero. Please ensure 'num_outputs' is valid for the dimension size"
            );
        }

        // Assign the calculated split size
        split_size = Some(calculated_split_size);
    }

    // Adjust axis if negative to count from the end as per ONNX spec
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    // Check for custom split sizes provided as a second input
    if node.inputs.len() > 1 && node.inputs[1].value.is_some() {
        let sizes = node.inputs[1]
            .value
            .as_ref()
            .unwrap()
            .data
            .clone()
            .into_usizes();

        if !sizes.is_empty() {
            split_sizes = Some(sizes);
        }
    }

    // Ensure that only one of 'split_sizes' or 'num_outputs' is specified
    if split_sizes.is_some() && split_size.is_some() {
        panic!(
            "Split: Cannot specify both 'split' input and 'num_outputs' attribute simultaneously"
        );
    }

    // Infer split_size if neither custom split_sizes nor split_size is provided
    if split_sizes.is_none() && split_size.is_none() {
        let num_outputs = node.outputs.len();
        let dim_size = tensor
            .static_shape
            .as_ref()
            .expect("Split: Static shape must be known to infer split size")[axis as usize];

        // Calculate inferred split size based on number of outputs
        let calculated_split_size =
            dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

        if calculated_split_size == 0 {
            panic!(
                "Split: Inferred split size is zero. Please ensure the number of outputs is valid for the dimension size"
            );
        }

        split_size = Some(calculated_split_size);
    }

    // Return the configuration for splitting operation
    SplitConfig {
        axis: axis as usize,
        split_size,
        split_sizes,
    }
}
