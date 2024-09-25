use alloc::format;

use crate::backend::Backend;
use crate::{Distribution, NamedDims, Shape, Tensor};

/// A tensor with named dimensions.
#[derive(Debug, Clone)]
pub struct NamedTensor<B: Backend, D: NamedDims<B>> {
    pub(crate) tensor: D::Tensor,
}

impl<B: Backend, ND: NamedDims<B, Tensor = Tensor<B, D>>, const D: usize> From<NamedTensor<B, ND>>
    for Tensor<B, D>
{
    fn from(nt: NamedTensor<B, ND>) -> Self {
        nt.tensor
    }
}

impl<B: Backend, ND: NamedDims<B, Tensor = Tensor<B, D>>, const D: usize> From<Tensor<B, D>>
    for NamedTensor<B, ND>
{
    fn from(tensor: Tensor<B, D>) -> Self {
        Self::from_tensor(tensor)
    }
}

impl<B: Backend, const D: usize, ND: NamedDims<B>> core::fmt::Display for NamedTensor<B, ND>
where
    ND: NamedDims<B, Tensor = Tensor<B, D>>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&format!(
            "NamedTensor[shape={:?}, dims={}]",
            self.shape().dims,
            ND::to_string(),
        ))
    }
}

impl<B: Backend, const D: usize, ND> NamedTensor<B, ND>
where
    ND: NamedDims<B, Tensor = Tensor<B, D>>,
{
    /// Create a named tensor from a tensor.
    pub fn from_tensor(tensor: Tensor<B, D>) -> Self {
        Self { tensor }
    }

    /// Create a random named tensor of the given shape where each element is sampled from
    /// the given distribution.
    pub fn random<S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        device: &B::Device,
    ) -> Self {
        Self::from_tensor(Tensor::random(shape, distribution, device))
    }

    /// Returns the shape of the current tensor.
    pub fn shape(&self) -> Shape {
        self.tensor.shape()
    }

    /// Applies element wise multiplication operation.
    ///
    /// `y = x2 * x1`
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        Self::from_tensor(self.tensor.mul(rhs.tensor))
    }

    /// Reshape the tensor to have the given shape.
    ///
    /// # Panics
    ///
    /// If the tensor can not be reshape to the given shape.
    pub fn reshape<const D2: usize, S, ND2>(self, shape: S, _: ND2) -> NamedTensor<B, ND2>
    where
        S: Into<Shape>,
        ND2: NamedDims<B, Tensor = Tensor<B, D2>>,
    {
        NamedTensor::from_tensor(self.tensor.reshape(shape.into()))
    }
}
