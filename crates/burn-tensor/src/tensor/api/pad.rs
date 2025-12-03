use alloc::vec::Vec;
use core::ops::Range;

use crate::{Element, ElementConversion, Tensor, backend::Backend, ops::PadMode};

use super::Numeric;

/// Helper to build a range array for slice_assign, selecting a portion of one dimension.
fn build_slice_ranges<const D: usize>(
    dims: [usize; D],
    target_dim: usize,
    start: usize,
    len: usize,
) -> [Range<usize>; D] {
    dims.iter()
        .enumerate()
        .map(|(i, &size)| {
            if i == target_dim {
                start..start + len
            } else {
                0..size
            }
        })
        .collect::<Vec<Range<usize>>>()
        .try_into()
        .unwrap()
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    /// Pads the tensor on the last two dimensions using the specified padding mode.
    ///
    /// **Note**: Currently, padding is only supported on the last two dimensions of a tensor
    /// (typically height and width for image data in NCHW format).
    ///
    /// # Arguments
    ///
    /// * `padding` - A tuple `(left, right, top, bottom)` specifying padding for the last two dimensions.
    /// * `mode` - The padding mode: `Constant(value)`, `Reflect`, or `Edge`.
    ///
    /// # Returns
    ///
    /// A new tensor with the specified padding applied.
    ///
    /// # Panics
    ///
    /// - `Reflect` mode panics if padding exceeds `dimension_size - 1`.
    /// - `Edge` mode panics if padding is applied to a zero-sized dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    /// use burn_tensor::ops::PadMode;
    ///
    /// fn example<B: Backend<FloatElem: From<f32>>>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///
    ///    // Constant padding with value 0.0
    ///    let padded = tensor.clone().pad((1, 1, 1, 1), PadMode::Constant(0.0));
    ///    // [
    ///    //   [0.0, 0.0, 0.0, 0.0, 0.0],
    ///    //   [0.0, 12.0, -2.0, 3.0, 0.0],
    ///    //   [0.0, 5.0, 3.0, 6.0, 0.0],
    ///    //   [0.0, 0.0, 0.0, 0.0, 0.0]
    ///    // ]
    ///
    ///    // Reflect padding
    ///    let padded = tensor.clone().pad((1, 1, 0, 0), PadMode::Reflect);
    ///    // [[−2.0, 12.0, −2.0, 3.0, −2.0], [3.0, 5.0, 3.0, 6.0, 3.0]]
    ///
    ///    // Edge padding
    ///    let padded = tensor.pad((1, 1, 0, 0), PadMode::Edge);
    ///    // [[12.0, 12.0, −2.0, 3.0, 3.0], [5.0, 5.0, 3.0, 6.0, 6.0]]
    /// }
    /// ```
    pub fn pad(self, padding: (usize, usize, usize, usize), mode: PadMode) -> Self {
        match mode {
            PadMode::Constant(value) => pad_constant(self, padding, value),
            PadMode::Reflect => pad_reflect(self, padding),
            PadMode::Edge => pad_edge(self, padding),
        }
    }
}

/// Pad with a constant value.
pub fn pad_constant<B, const D: usize, K, E>(
    tensor: Tensor<B, D, K>,
    padding: (usize, usize, usize, usize),
    value: E,
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
    E: ElementConversion,
{
    let (left, right, top, bottom) = padding;

    let mut padded_dims: [usize; D] = tensor.dims();

    // Update the last two dimensions with padding
    padded_dims[D - 2] += top + bottom;
    padded_dims[D - 1] += left + right;

    // Create the ranges for the padded tensor
    let ranges: [core::ops::Range<usize>; D] = padded_dims
        .iter()
        .enumerate()
        .map(|(i, &dim)| {
            if i == D - 2 {
                top..dim - bottom
            } else if i == D - 1 {
                left..dim - right
            } else {
                0..dim
            }
        })
        .collect::<Vec<core::ops::Range<usize>>>()
        .try_into()
        .unwrap();

    // Create the padded tensor
    let padded_tensor = Tensor::full(padded_dims, value, &tensor.device());

    // Assign the original tensor data to the appropriate slice of the padded tensor
    padded_tensor.slice_assign(ranges, tensor)
}

/// Pad using reflection at the boundaries (excluding edge values).
///
/// For ONNX "reflect" mode: mirrors from index 1, not index 0.
/// Example: `[1, 2, 3, 4]` with left padding 2 becomes `[3, 2, 1, 2, 3, 4]`
pub fn pad_reflect<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    padding: (usize, usize, usize, usize),
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    let (left, right, top, bottom) = padding;
    let dims = tensor.dims();

    // Validate padding doesn't exceed tensor dimensions
    // For reflect mode, padding must be less than the corresponding dimension
    assert!(
        top < dims[D - 2] && bottom < dims[D - 2],
        "Reflect padding on height ({}, {}) must be less than height dimension ({})",
        top,
        bottom,
        dims[D - 2]
    );
    assert!(
        left < dims[D - 1] && right < dims[D - 1],
        "Reflect padding on width ({}, {}) must be less than width dimension ({})",
        left,
        right,
        dims[D - 1]
    );

    let mut result = tensor;

    // Pad height dimension (D - 2): top and bottom
    if top > 0 || bottom > 0 {
        result = pad_reflect_dim(result, D - 2, top, bottom);
    }

    // Pad width dimension (D - 1): left and right
    if left > 0 || right > 0 {
        result = pad_reflect_dim(result, D - 1, left, right);
    }

    result
}

/// Helper to pad a single dimension using reflection.
fn pad_reflect_dim<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    dim: usize,
    pad_before: usize,
    pad_after: usize,
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    let dims = tensor.dims();
    let dim_size = dims[dim];

    // Calculate output dimensions
    let mut output_dims = dims;
    output_dims[dim] += pad_before + pad_after;

    // Create output tensor and place original in the center
    let output = Tensor::zeros(output_dims, &tensor.device());
    let original_range = build_slice_ranges(output_dims, dim, pad_before, dim_size);
    let mut output = output.slice_assign(original_range, tensor.clone());

    // Assign reflected "before" padding (e.g., top or left)
    // Reflect excludes the edge, so we take indices [1..pad_before+1] and flip
    if pad_before > 0 {
        let before_slice = tensor.clone().narrow(dim, 1, pad_before);
        let before_flipped = before_slice.flip([dim as isize]);
        let before_range = build_slice_ranges(output_dims, dim, 0, pad_before);
        output = output.slice_assign(before_range, before_flipped);
    }

    // Assign reflected "after" padding (e.g., bottom or right)
    // Take indices [dim_size - pad_after - 1..dim_size - 1] and flip
    if pad_after > 0 {
        let start = dim_size - pad_after - 1;
        let after_slice = tensor.narrow(dim, start, pad_after);
        let after_flipped = after_slice.flip([dim as isize]);
        let after_range = build_slice_ranges(output_dims, dim, pad_before + dim_size, pad_after);
        output = output.slice_assign(after_range, after_flipped);
    }

    output
}

/// Pad by replicating edge values.
///
/// Example: `[1, 2, 3, 4]` with left padding 2 becomes `[1, 1, 1, 2, 3, 4]`
pub fn pad_edge<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    padding: (usize, usize, usize, usize),
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    let (left, right, top, bottom) = padding;
    let dims = tensor.dims();

    // Validate dimensions are non-zero when padding is requested
    if top > 0 || bottom > 0 {
        assert!(
            dims[D - 2] > 0,
            "Cannot apply edge padding to zero-sized height dimension"
        );
    }
    if left > 0 || right > 0 {
        assert!(
            dims[D - 1] > 0,
            "Cannot apply edge padding to zero-sized width dimension"
        );
    }

    let mut result = tensor;

    // Pad height dimension (D - 2): top and bottom
    if top > 0 || bottom > 0 {
        result = pad_edge_dim(result, D - 2, top, bottom);
    }

    // Pad width dimension (D - 1): left and right
    if left > 0 || right > 0 {
        result = pad_edge_dim(result, D - 1, left, right);
    }

    result
}

/// Helper to pad a single dimension by replicating edge values.
fn pad_edge_dim<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    dim: usize,
    pad_before: usize,
    pad_after: usize,
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    let dims = tensor.dims();
    let dim_size = dims[dim];

    // Calculate output dimensions
    let mut output_dims = dims;
    output_dims[dim] += pad_before + pad_after;

    // Create output tensor and place original in the center
    let output = Tensor::zeros(output_dims, &tensor.device());
    let original_range = build_slice_ranges(output_dims, dim, pad_before, dim_size);
    let mut output = output.slice_assign(original_range, tensor.clone());

    // Assign "before" padding by repeating the first element
    if pad_before > 0 {
        let first_slice = tensor.clone().narrow(dim, 0, 1);
        let before_pad = first_slice.repeat_dim(dim, pad_before);
        let before_range = build_slice_ranges(output_dims, dim, 0, pad_before);
        output = output.slice_assign(before_range, before_pad);
    }

    // Assign "after" padding by repeating the last element
    if pad_after > 0 {
        let last_slice = tensor.narrow(dim, dim_size - 1, 1);
        let after_pad = last_slice.repeat_dim(dim, pad_after);
        let after_range = build_slice_ranges(output_dims, dim, pad_before + dim_size, pad_after);
        output = output.slice_assign(after_range, after_pad);
    }

    output
}
