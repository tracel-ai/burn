use alloc::vec::Vec;

use crate::{Tensor, kind::Numeric, ops::PadMode};

/// Trait for types that can be used as padding specifications.
///
/// Padding is specified as `(before, after)` pairs per dimension, returned as a
/// fixed-size array `[(usize, usize); D]`. If fewer pairs than dimensions are provided,
/// they apply to the **last** N dimensions (earlier dimensions are left unpadded).
pub trait IntoPadding<const D: usize> {
    /// Converts into a fixed-size array of `(before, after)` padding pairs.
    fn into_padding(self) -> [(usize, usize); D];
}

impl<const D: usize, const N: usize> IntoPadding<D> for [(usize, usize); N] {
    fn into_padding(self) -> [(usize, usize); D] {
        assert!(
            N <= D,
            "Padding has {} pairs but tensor only has {} dimensions",
            N,
            D
        );
        let mut result = [(0usize, 0usize); D];
        let offset = D - N;
        for (i, pair) in self.into_iter().enumerate() {
            result[offset + i] = pair;
        }
        result
    }
}

/// Backward-compatible: `(left, right, top, bottom)` maps to last 2 dimensions.
///
/// Equivalent to `[(top, bottom), (left, right)]`.
impl<const D: usize> IntoPadding<D> for (usize, usize, usize, usize) {
    fn into_padding(self) -> [(usize, usize); D] {
        let (left, right, top, bottom) = self;
        let mut result = [(0usize, 0usize); D];
        result[D - 2] = (top, bottom);
        result[D - 1] = (left, right);
        result
    }
}

impl<const D: usize> IntoPadding<D> for &[(usize, usize)] {
    fn into_padding(self) -> [(usize, usize); D] {
        assert!(
            self.len() <= D,
            "Padding has {} pairs but tensor only has {} dimensions",
            self.len(),
            D
        );
        let mut result = [(0usize, 0usize); D];
        let offset = D - self.len();
        for (i, &pair) in self.iter().enumerate() {
            result[offset + i] = pair;
        }
        result
    }
}

impl<const D: usize> IntoPadding<D> for Vec<(usize, usize)> {
    fn into_padding(self) -> [(usize, usize); D] {
        assert!(
            self.len() <= D,
            "Padding has {} pairs but tensor only has {} dimensions",
            self.len(),
            D
        );
        let mut result = [(0usize, 0usize); D];
        let offset = D - self.len();
        for (i, pair) in self.into_iter().enumerate() {
            result[offset + i] = pair;
        }
        result
    }
}

impl<const D: usize, K> Tensor<D, K>
where
    K: Numeric,
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
    ///   - `[(before, after); N]` fixed-size array of pairs (N <= D)
    ///   - `&[(before, after)]` slice of pairs per dimension
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
    /// use burn_tensor::{Tensor, Shape};
    /// use burn_tensor::ops::PadMode;
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    ///
    /// // Constant padding with value 0.0 (backward-compatible tuple)
    /// let padded = tensor.clone().pad((1, 1, 1, 1), PadMode::Constant(0.0));
    ///
    /// // Pad arbitrary dimensions with slice of (before, after) pairs
    /// let padded = tensor.clone().pad([(1, 1), (2, 2)], PadMode::Constant(0.0));
    ///
    /// // Pad only the last dimension
    /// let padded = tensor.pad([(1, 1)], PadMode::Reflect);
    /// ```
    pub fn pad(self, padding: impl IntoPadding<D>, mode: impl Into<PadMode>) -> Self {
        let pairs = padding.into_padding();
        Tensor::new(K::pad(self.primitive, &pairs, mode.into()))
    }
}
