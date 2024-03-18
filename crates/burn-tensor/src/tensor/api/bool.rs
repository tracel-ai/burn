use crate::{backend::Backend, Bool, Data, Int, Shape, Tensor};
use alloc::vec::Vec;

#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
use crate::argwhere;

/// The part of the tensor to keep when creating a triangular mask.
enum TriPart {
    /// Upper triangular part.
    Upper,

    /// Lower triangular part.
    Lower,

    /// Diagonal part.
    Diagonal,
}

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

    /// Compute the indices of the elements that are true.
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

    /// Compute the indices of the elements that are true, grouped by element.
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of all non-zero elements of the given tensor. Each row in the
    /// result contains the indices of a non-zero element.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn argwhere(self) -> Tensor<B, 2, Int> {
        Tensor::new(B::bool_argwhere(self.primitive))
    }

    /// Compute the indices of the elements that are true, grouped by element.
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of all non-zero elements of the given tensor. Each row in the
    /// result contains the indices of a non-zero element.
    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    pub async fn argwhere(self) -> Tensor<B, 2, Int> {
        Tensor::new(argwhere::<B, D>(self.primitive).await)
    }

    /// Creates a mask for the upper, lower triangle, or diagonal of a matrix, which can be used to
    /// fill the specified area with a value.
    fn tri_mask<S: Into<Shape<D>>>(
        shape: S,
        tri_part: TriPart,
        offset: i64,
        device: &B::Device,
    ) -> Self {
        let shape = shape.into();
        let height = shape.dims[D - 2];
        let width = shape.dims[D - 1];

        // Generate row and column index tensors.
        let row_indices: Tensor<B, 1, Int> = Tensor::arange(0..height as i64, device);
        let col_indices: Tensor<B, 1, Int> = Tensor::arange(0..width as i64, device);

        // Prepare shapes for broadcasting.
        let mut row_shape = [1; D];
        row_shape[D - 2] = height;
        let mut col_shape = [1; D];
        col_shape[D - 1] = width;

        // Reshape for broadcasting.
        let row_broadcast = row_indices.reshape(Shape::new(row_shape));
        let col_broadcast = col_indices.reshape(Shape::new(col_shape));

        // Broadcasting trick to create a matrix that facilitates comparison for mask generation.
        let matrix = row_broadcast.clone() - (col_broadcast.clone() - offset);

        // Select the appropriate comparison function based on `tri_part`.
        let compare = match tri_part {
            TriPart::Upper => Tensor::greater_elem,
            TriPart::Lower => Tensor::lower_elem,
            TriPart::Diagonal => Tensor::not_equal_elem,
        };

        // Generate and return the mask by applying the comparison to the matrix.
        compare(matrix, 0).unsqueeze()
    }

    /// Creates a mask for the upper triangle of a matrix, which can be used to fill the specified
    /// area with a value.
    ///
    /// This function generates a boolean tensor representing the mask of the upper triangle of a matrix.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the matrix.
    /// * `offset`: The offset from the diagonal, where 0 means the diagonal, and positive values shift
    ///  towards the upper triangle.
    /// * `device`: The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// Returns a boolean tensor where `true` indicates the elements of the matrix that are part of the
    /// upper triangle taking into account the specified `offset`.
    pub fn triu_mask<S: Into<Shape<D>>>(shape: S, offset: i64, device: &B::Device) -> Self {
        Self::tri_mask(shape, TriPart::Upper, offset, device)
    }

    /// Creates a mask for the lower triangle of a matrix, which can be used to fill the specified
    /// area with a value.
    ///
    /// This function generates a boolean tensor representing the mask of the lower triangle of a matrix.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the matrix.
    /// * `offset`: The offset from the diagonal, where 0 means the diagonal, and negative values shift
    /// towards the lower triangle.
    /// * `device`: The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// Returns a boolean tensor where `true` indicates the elements of the matrix that are part of the
    /// lower triangle taking into account the specified `offset`.
    pub fn tril_mask<S: Into<Shape<D>>>(shape: S, offset: i64, device: &B::Device) -> Self {
        Self::tri_mask(shape, TriPart::Lower, offset, device)
    }

    /// Creates a mask for the diagonal of a matrix, which can be used to fill the specified
    /// area with a value.
    ///
    /// This function generates a boolean tensor representing the mask of the diagonal of a matrix.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the matrix.
    /// * `device`: The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// Returns a boolean tensor where `true` indicates the elements of the matrix that are part of the
    /// diagonal.
    pub fn diag_mask<S: Into<Shape<D>>>(shape: S, offset: i64, device: &B::Device) -> Self {
        Self::tri_mask(shape, TriPart::Diagonal, offset, device)
    }
}
