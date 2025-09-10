use alloc::vec::Vec;

use crate::{BasicOps, Int, Tensor, backend::Backend, check, check::TensorCheck};

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    /// Takes elements from the tensor along the given dimension using indices of any dimensionality.
    ///
    /// This behaves like numpy's take function. When indices is multi-dimensional,
    /// the output shape will be: input.shape[:dim] + indices.shape + input.shape[dim+1:]
    ///
    /// Negative indices are supported and count from the end of the dimension.
    /// For example, -1 refers to the last element, -2 to the second-to-last, etc.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to select elements.
    /// * `indices` - The indices of elements to select. Can be any dimensionality.
    ///   Negative values count from the end.
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
    ///   // Example with positive and negative indices (1D indices -> same dims)
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    ///   let indices = Tensor::<B, 1, Int>::from_data([2, -3, -2], &device);  // [2, 0, 1]
    ///   let result: Tensor<B, 2> = tensor.clone().take::<1, 2>(1, indices);
    ///   println!("{result}");
    ///   // [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]
    ///
    ///   // Example with 2D indices - output will have +1 dimension (2D -> 3D)
    ///   let indices_2d = Tensor::<B, 2, Int>::from_data([[0, -1], [1, 0]], &device);  // [[0, 2], [1, 0]]
    ///   let result: Tensor<B, 3> = tensor.take::<2, 3>(1, indices_2d);
    ///   println!("{result}");
    ///   // [[[1.0, 3.0], [2.0, 1.0]], [[4.0, 6.0], [5.0, 4.0]]]
    /// }
    /// ```
    pub fn take<const DI: usize, const DO: usize>(
        self,
        dim: usize,
        indices: Tensor<B, DI, Int>,
    ) -> Tensor<B, DO, K> {
        check!(TensorCheck::take::<D, DI, DO>(dim));

        // Get the size of the dimension we're selecting from
        let input_shape = self.shape();
        let dim_size = input_shape.dims[dim] as i64;

        // Store the indices shape for reshaping later
        let indices_shape = indices.shape();
        let indices_dims = indices_shape.dims.clone();

        // Flatten indices to 1D for processing
        let indices_flat = if DI > 1 {
            indices.flatten(0, DI - 1)
        } else {
            indices.reshape([indices_shape.num_elements()])
        };

        // Normalize negative indices: if index < 0, add dim_size to it
        // For example, with dim_size=3: -1 becomes 2, -2 becomes 1, -3 becomes 0
        // This is done with: normalized = (indices % dim_size + dim_size) % dim_size
        // Which handles both positive and negative indices correctly
        let dim_size_scalar = dim_size as i32;
        let normalized_indices = indices_flat
            .clone()
            .remainder_scalar(dim_size_scalar)
            .add_scalar(dim_size_scalar)
            .remainder_scalar(dim_size_scalar);

        // Perform the selection with normalized indices
        let selected = self.select(dim, normalized_indices);

        // Build the output shape
        // Output shape = input.shape[:dim] + indices.shape + input.shape[dim+1:]
        let selected_shape = selected.shape();
        let mut new_shape = Vec::with_capacity(DO);

        // Add dimensions before the selected dimension
        for i in 0..dim {
            new_shape.push(selected_shape.dims[i]);
        }

        // Add all indices dimensions
        for idx_dim in indices_dims {
            new_shape.push(idx_dim);
        }

        // Add dimensions after the selected dimension
        for i in (dim + 1)..D {
            new_shape.push(selected_shape.dims[i]);
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
