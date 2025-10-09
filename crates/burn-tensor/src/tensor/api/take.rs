use alloc::vec::Vec;

use crate::{
    BasicOps, Int, Tensor,
    backend::Backend,
    check,
    check::TensorCheck,
    indexing::{AsIndex, canonicalize_dim},
};

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    /// Takes elements from the tensor along the given dimension using indices of any dimensionality.
    ///
    /// This behaves like numpy's take function. When indices is multi-dimensional,
    /// the output shape will be: input.shape\[:dim\] + indices.shape + input.shape\[dim+1:\]
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to select elements. Supports negative indexing.
    /// * `indices` - The indices of elements to select. Can be any dimensionality.
    ///   Must be valid indices in the range [0, dim_size).
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///
    ///   // Example with 1D indices
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    ///   let indices = Tensor::<B, 1, Int>::from_data([2, 0, 1], &device);
    ///   let result: Tensor<B, 2> = tensor.clone().take::<1, 2>(-1, indices);  // -1 refers to last dimension
    ///   println!("{result}");
    ///   // [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]
    ///
    ///   // Example with 2D indices - output will have +1 dimension (2D -> 3D)
    ///   let indices_2d = Tensor::<B, 2, Int>::from_data([[0, 2], [1, 0]], &device);
    ///   let result: Tensor<B, 3> = tensor.take::<2, 3>(1, indices_2d);
    ///   println!("{result}");
    ///   // [[[1.0, 3.0], [2.0, 1.0]], [[4.0, 6.0], [5.0, 4.0]]]
    /// }
    /// ```
    pub fn take<const DI: usize, const DO: usize>(
        self,
        dim: impl AsIndex,
        indices: Tensor<B, DI, Int>,
    ) -> Tensor<B, DO, K> {
        let dim = canonicalize_dim(dim, D, false);
        check!(TensorCheck::take::<D, DI, DO>(dim));

        // Store the indices shape for reshaping later
        let indices_shape = indices.shape();
        let indices_dims = indices_shape.clone();

        // Flatten indices to 1D for processing
        let indices_flat = indices.reshape([indices_shape.num_elements()]);

        // Perform the selection with the flattened indices
        let selected = self.select(dim, indices_flat);

        // Build the output shape
        // Output shape = input.shape[:dim] + indices.shape + input.shape[dim+1:]
        let selected_shape = selected.shape();
        let mut new_shape = Vec::with_capacity(DO);

        // Add dimensions before the selected dimension
        for i in 0..dim {
            new_shape.push(selected_shape[i]);
        }

        // Add all indices dimensions
        for idx_dim in indices_dims {
            new_shape.push(idx_dim);
        }

        // Add dimensions after the selected dimension
        for i in (dim + 1)..D {
            new_shape.push(selected_shape[i]);
        }

        // Verify we have the correct number of dimensions
        assert_eq!(
            new_shape.len(),
            DO,
            "Internal error: shape calculation resulted in {} dims but expected {}",
            new_shape.len(),
            DO
        );

        // Convert to fixed-size array for reshape
        let mut shape_array = [0; DO];
        for (i, &s) in new_shape.iter().enumerate() {
            shape_array[i] = s;
        }

        selected.reshape(shape_array)
    }
}
