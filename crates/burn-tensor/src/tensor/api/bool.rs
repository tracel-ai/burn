use crate::{backend::Backend, Bool, Data, Int, Tensor};
use alloc::vec::Vec;

#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
use crate::{argwhere, tensor::Shape};

impl<B, const D: usize> Tensor<B, D, Bool>
where
    B: Backend,
{
    /// Create a boolean tensor from data on the given device.
    pub fn from_bool(data: Data<bool, D>, device: &B::Device) -> Self {
        Self::new(B::bool_from_data(data, device))
    }

    /// Convert the bool tensor into an int tensor.
    pub fn int(self) -> Tensor<B, D, Int> {
        Tensor::new(B::bool_into_int(self.primitive))
    }

    /// Convert the bool tensor into an float tensor.
    pub fn float(self) -> Tensor<B, D> {
        Tensor::new(B::bool_into_float(self.primitive))
    }

    /// Inverses boolean values.
    pub fn bool_not(self) -> Self {
        Tensor::new(B::bool_not(self.primitive))
    }

    /// Compute the indices of the elements that are non-zero.
    ///
    /// # Returns
    ///
    /// A vector of tensors, one for each dimension of the given tensor, containing the indices of
    /// the non-zero elements in that dimension.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn nonzero(self) -> Vec<Tensor<B, 1, Int>> {
        B::bool_nonzero(self.primitive)
            .into_iter()
            .map(Tensor::new)
            .collect()
    }

    /// Compute the indices of the elements that are non-zero.
    ///
    /// # Returns
    ///
    /// A vector of tensors, one for each dimension of the given tensor, containing the indices of
    /// the non-zero elements in that dimension.
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn nonzero(self) -> Vec<Tensor<B, 1, Int>> {
        let indices = self.argwhere().await.primitive;
        let dims = B::int_shape(&indices).dims;
        B::int_chunk(indices, dims[1], 1)
            .into_iter()
            .map(|t| B::int_reshape(t, Shape::new([dims[0]])))
            .map(Tensor::new)
            .collect()
    }

    /// Compute the indices of the elements that are non-zero, grouped by element.
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of all non-zero elements of the given tensor. Each row in the
    /// result contains the indices of a non-zero element.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn argwhere(self) -> Tensor<B, 2, Int> {
        Tensor::new(B::bool_argwhere(self.primitive))
    }

    /// Compute the indices of the elements that are non-zero, grouped by element.
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of all non-zero elements of the given tensor. Each row in the
    /// result contains the indices of a non-zero element.
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn argwhere(self) -> Tensor<B, 2, Int> {
        Tensor::new(argwhere::<B, D>(self.primitive).await)
    }
}
