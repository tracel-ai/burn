use crate::{backend::Backend, Int, Tensor};
use core::ops::Range;

impl<B> Tensor<B, 1, Int>
where
    B: Backend,
{
    /// Returns a new integer tensor on the default device which values are generated from the given range.
    pub fn arange(range: Range<usize>) -> Self {
        Tensor::new(B::arange(range, &B::Device::default()))
    }
    /// Returns a new integer tensor on the specified device which values are generated from the given range.
    pub fn arange_device(range: Range<usize>, device: &B::Device) -> Self {
        Tensor::new(B::arange(range, device))
    }
}
