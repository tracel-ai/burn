use alloc::vec::Vec;
use core::ops::Range;

use crate::{Element, ElementConversion, Tensor, backend::Backend, ops::PadMode};

use super::Numeric;

/// Trait for types that can be used as padding specifications.
///
/// Padding is specified as `(before, after)` pairs per dimension.
/// If fewer pairs than dimensions are provided, they apply to the **last** N dimensions
/// (earlier dimensions are left unpadded).
pub trait IntoPadding {
    /// Converts into a list of `(before, after)` padding pairs.
    fn into_padding(self) -> Vec<(usize, usize)>;
}

impl IntoPadding for Vec<(usize, usize)> {
    fn into_padding(self) -> Vec<(usize, usize)> {
        self
    }
}

impl IntoPadding for &[(usize, usize)] {
    fn into_padding(self) -> Vec<(usize, usize)> {
        self.to_vec()
    }
}

impl<const N: usize> IntoPadding for [(usize, usize); N] {
    fn into_padding(self) -> Vec<(usize, usize)> {
        self.to_vec()
    }
}

/// Backward-compatible: `(left, right, top, bottom)` maps to last 2 dimensions.
///
/// Equivalent to `&[(top, bottom), (left, right)]`.
impl IntoPadding for (usize, usize, usize, usize) {
    fn into_padding(self) -> Vec<(usize, usize)> {
        let (left, right, top, bottom) = self;
        alloc::vec![(top, bottom), (left, right)]
    }
}

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

/// Expand padding pairs to length D, left-padding with (0, 0) for unspecified leading dimensions.
fn expand_padding<const D: usize>(pairs: Vec<(usize, usize)>) -> [(usize, usize); D] {
    assert!(
        pairs.len() <= D,
        "Padding has {} pairs but tensor only has {} dimensions",
        pairs.len(),
        D
    );
    let mut result = [(0usize, 0usize); D];
    let offset = D - pairs.len();
    for (i, pair) in pairs.into_iter().enumerate() {
        result[offset + i] = pair;
    }
    result
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    /// Pads the tensor using the specified padding mode.
    ///
    /// Padding is specified as `(before, after)` pairs. If fewer pairs than tensor dimensions
    /// are provided, they apply to the **last** N dimensions (unspecified leading dimensions
    /// are left unpadded).
    ///
    /// For backward compatibility, a `(left, right, top, bottom)` tuple is also accepted,
    /// which pads the last two dimensions.
    ///
    /// # Arguments
    ///
    /// * `padding` - Padding specification. Accepts:
    ///   - `&[(before, after)]` slice of pairs per dimension
    ///   - `[(before, after); N]` fixed-size array of pairs
    ///   - `Vec<(before, after)>` vector of pairs
    ///   - `(left, right, top, bottom)` tuple for last-2-dim backward compatibility
    /// * `mode` - The padding mode: `Constant(value)`, `Reflect`, or `Edge`.
    ///
    /// # Returns
    ///
    /// A new tensor with the specified padding applied.
    ///
    /// # Panics
    ///
    /// - Panics if more padding pairs are provided than tensor dimensions.
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
    ///    // Constant padding with value 0.0 (backward-compatible tuple)
    ///    let padded = tensor.clone().pad((1, 1, 1, 1), PadMode::Constant(0.0));
    ///
    ///    // Pad arbitrary dimensions with slice of (before, after) pairs
    ///    let padded = tensor.clone().pad([(1, 1), (2, 2)], PadMode::Constant(0.0));
    ///
    ///    // Pad only the last dimension
    ///    let padded = tensor.pad([(1, 1)], PadMode::Reflect);
    /// }
    /// ```
    pub fn pad(self, padding: impl IntoPadding, mode: impl Into<PadMode>) -> Self {
        let pairs = expand_padding::<D>(padding.into_padding());
        match mode.into() {
            PadMode::Constant(value) => pad_constant(self, &pairs, value),
            PadMode::Reflect => pad_reflect(self, &pairs),
            PadMode::Edge => pad_edge(self, &pairs),
        }
    }
}

/// Pad with a constant value.
fn pad_constant<B, const D: usize, K, E>(
    tensor: Tensor<B, D, K>,
    padding: &[(usize, usize); D],
    value: E,
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
    E: ElementConversion,
{
    let mut padded_dims: [usize; D] = tensor.dims();

    for (i, &(before, after)) in padding.iter().enumerate() {
        padded_dims[i] += before + after;
    }

    let ranges: [Range<usize>; D] = padded_dims
        .iter()
        .enumerate()
        .map(|(i, &dim)| {
            let (before, after) = padding[i];
            before..dim - after
        })
        .collect::<Vec<Range<usize>>>()
        .try_into()
        .unwrap();

    let padded_tensor = Tensor::full(padded_dims, value, &tensor.device());

    padded_tensor.slice_assign(ranges, tensor)
}

/// Pad using reflection at the boundaries (excluding edge values).
///
/// For ONNX "reflect" mode: mirrors from index 1, not index 0.
/// Example: `[1, 2, 3, 4]` with left padding 2 becomes `[3, 2, 1, 2, 3, 4]`
fn pad_reflect<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    padding: &[(usize, usize); D],
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    let dims = tensor.dims();

    for (i, &(before, after)) in padding.iter().enumerate() {
        if before > 0 || after > 0 {
            assert!(
                before < dims[i] && after < dims[i],
                "Reflect padding ({}, {}) must be less than dimension {} size ({})",
                before,
                after,
                i,
                dims[i]
            );
        }
    }

    let mut result = tensor;

    for (i, &(before, after)) in padding.iter().enumerate() {
        if before > 0 || after > 0 {
            result = pad_reflect_dim(result, i, before, after);
        }
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
fn pad_edge<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    padding: &[(usize, usize); D],
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    let dims = tensor.dims();

    for (i, &(before, after)) in padding.iter().enumerate() {
        if before > 0 || after > 0 {
            assert!(
                dims[i] > 0,
                "Cannot apply edge padding to zero-sized dimension {}",
                i
            );
        }
    }

    let mut result = tensor;

    for (i, &(before, after)) in padding.iter().enumerate() {
        if before > 0 || after > 0 {
            result = pad_edge_dim(result, i, before, after);
        }
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
