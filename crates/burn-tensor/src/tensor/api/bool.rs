use crate::{Bool, Cast, Device, Int, Shape, Tensor, TensorData, TensorPrimitive};
use alloc::{vec, vec::Vec};
use burn_backend::ops::BoolTensorOps;
use burn_dispatch::Dispatch;

use crate::try_read_sync;

/// The part of the tensor to keep when creating a triangular mask.
enum TriPart {
    /// Upper triangular part.
    Upper,

    /// Lower triangular part.
    Lower,

    /// Diagonal part.
    Diagonal,
}

impl<const D: usize> Tensor<D, Bool> {
    /// Create a boolean tensor from data on the given device.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor data.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// A boolean tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<2, Bool>::from_bool([[true, false], [false, true]].into(), &device);
    ///     println!("{tensor}");
    /// }
    /// ```
    pub fn from_bool(data: TensorData, device: &Device) -> Self {
        Self::from_data(data, device)
    }

    /// Convert the bool tensor into an int tensor.
    ///
    /// # Returns
    ///
    /// An integer tensor where `true` is converted to `1` and `false` to `0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let bool_tensor = Tensor::<1, Bool>::from_bool([true, false, true].into(), &device);
    ///     let int_tensor = bool_tensor.int();
    ///     println!("{int_tensor}"); // [1, 0, 1]
    /// }
    /// ```
    pub fn int(self) -> Tensor<D, Int> {
        let out_dtype = self.device().settings().int_dtype;
        Tensor::new(Dispatch::bool_into_int(self.primitive, out_dtype))
    }

    /// Convert the bool tensor into a float tensor.
    ///
    /// # Returns
    ///
    /// A float tensor where `true` is converted to `1.0` and `false` to `0.0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let bool_tensor = Tensor::<1, Bool>::from_bool([true, false, true].into(), &device);
    ///     let float_tensor = bool_tensor.float();
    ///     println!("{float_tensor}"); // [1.0, 0.0, 1.0]
    /// }
    /// ```
    pub fn float(self) -> Tensor<D> {
        let out_dtype = self.device().settings().float_dtype;
        Tensor::new(TensorPrimitive::Float(Dispatch::bool_into_float(
            self.primitive,
            out_dtype,
        )))
    }

    /// Converts a bool tensor to the specified data type.
    ///
    /// Supports casting to [`IntDType`](crate::IntDType) (producing an int tensor)
    /// or [`FloatDType`](crate::FloatDType) (producing a float tensor).
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool, IntDType, FloatDType};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let bool_tensor = Tensor::<1, Bool>::from_bool([true, false, true].into(), &device);
    ///
    ///     // Cast to int
    ///     let int_tensor = bool_tensor.clone().cast(IntDType::I64);
    ///
    ///     // Cast to float
    ///     let float_tensor = bool_tensor.cast(FloatDType::F32);
    /// }
    /// ```
    #[must_use]
    pub fn cast<T: Cast<Bool>>(self, dtype: T) -> Tensor<D, T::OutputKind> {
        Tensor::new(T::cast(self.primitive, dtype))
    }

    /// Inverses boolean values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<2, Bool>::from_bool([[true, false], [false, true]].into(), &device);
    ///     let inverted = tensor.bool_not();
    ///     println!("{inverted}"); // [[false, true], [true, false]]
    /// }
    /// ```
    pub fn bool_not(self) -> Self {
        Tensor::new(Dispatch::bool_not(self.primitive))
    }

    /// Performs logical and (`&&`) on two boolean tensors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor for the AND operation.
    ///
    /// # Returns
    ///
    /// A boolean tensor where each element is the result of `self[i] && rhs[i]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let a = Tensor::<2, Bool>::from_bool([[true, true], [false, false]].into(), &device);
    ///     let b = Tensor::<2, Bool>::from_bool([[true, false], [true, false]].into(), &device);
    ///     let result = a.bool_and(b);
    ///     println!("{result}"); // [[true, false], [false, false]]
    /// }
    /// ```
    pub fn bool_and(self, rhs: Tensor<D, Bool>) -> Tensor<D, Bool> {
        Tensor::new(Dispatch::bool_and(self.primitive, rhs.primitive))
    }

    /// Performs logical or (`||`) on two boolean tensors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor for the OR operation.
    ///
    /// # Returns
    ///
    /// A boolean tensor where each element is the result of `self[i] || rhs[i]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let a = Tensor::<2, Bool>::from_bool([[true, true], [false, false]].into(), &device);
    ///     let b = Tensor::<2, Bool>::from_bool([[true, false], [true, false]].into(), &device);
    ///     let result = a.bool_or(b);
    ///     println!("{result}"); // [[true, true], [true, false]]
    /// }
    /// ```
    pub fn bool_or(self, rhs: Tensor<D, Bool>) -> Tensor<D, Bool> {
        Tensor::new(Dispatch::bool_or(self.primitive, rhs.primitive))
    }

    /// Performs logical xor (`^`) on two boolean tensors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor for the XOR operation.
    ///
    /// # Returns
    ///
    /// A boolean tensor where each element is the result of `self[i] ^ rhs[i]`.
    /// Returns `true` when exactly one of the operands is `true`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let a = Tensor::<2, Bool>::from_bool([[true, true], [false, false]].into(), &device);
    ///     let b = Tensor::<2, Bool>::from_bool([[true, false], [true, false]].into(), &device);
    ///     let result = a.bool_xor(b);
    ///     println!("{result}"); // [[false, true], [true, false]]
    /// }
    /// ```
    pub fn bool_xor(self, rhs: Tensor<D, Bool>) -> Tensor<D, Bool> {
        Tensor::new(Dispatch::bool_xor(self.primitive, rhs.primitive))
    }

    /// Compute the indices of `true` elements in the tensor (i.e., non-zero for boolean tensors).
    ///
    /// # Returns
    ///
    /// A vector of tensors, one for each dimension of the given tensor, containing the indices of
    /// the non-zero elements in that dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<2, Bool>::from_bool(
    ///         [[true, false, true], [false, true, false], [false, true, false]].into(),
    ///         &device,
    ///     );
    ///     let indices = tensor.nonzero();
    ///     println!("{}", indices[0]); // [0, 0, 1, 2]
    ///     println!("{}", indices[1]); // [0, 2, 1, 1]
    /// }
    /// ```
    pub fn nonzero(self) -> Vec<Tensor<1, Int>> {
        try_read_sync(self.nonzero_async())
            .expect("Failed to read tensor data synchronously. Try using nonzero_async instead.")
    }

    /// Compute the indices of `true` elements in the tensor (i.e., non-zero for boolean tensors).
    ///
    /// # Returns
    ///
    /// A vector of tensors, one for each dimension of the given tensor, containing the indices of
    /// the non-zero elements in that dimension.
    pub async fn nonzero_async(self) -> Vec<Tensor<1, Int>> {
        let indices = self.argwhere_async().await;

        if indices.shape().num_elements() == 0 {
            // Return empty vec when all elements are zero
            return vec![];
        }

        let dims = indices.shape();
        indices
            .chunk(dims[1], 1)
            .into_iter()
            .map(|t| t.reshape(Shape::new([dims[0]])))
            .collect()
    }

    /// Compute the indices of the elements that are true, grouped by element.
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of all non-zero elements of the given tensor. Each row in the
    /// result contains the indices of a non-zero element.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<2, Bool>::from_bool(
    ///         [[true, false, true], [false, true, false], [false, true, false]].into(),
    ///         &device,
    ///     );
    ///     let indices = tensor.argwhere();
    ///     println!("{indices}"); // [[0, 0], [0, 2], [1, 1], [2, 1]]
    /// }
    /// ```
    pub fn argwhere(self) -> Tensor<2, Int> {
        try_read_sync(self.argwhere_async())
            .expect("Failed to read tensor data synchronously. Try using argwhere_async instead.")
    }

    /// Compute the indices of the elements that are true, grouped by element.
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of all non-zero elements of the given tensor. Each row in the
    /// result contains the indices of a non-zero element.
    pub async fn argwhere_async(self) -> Tensor<2, Int> {
        let out_dtype = self.device().settings().int_dtype;
        Tensor::new(Dispatch::bool_argwhere(self.primitive, out_dtype).await)
    }

    /// Creates a mask for the upper, lower triangle, or diagonal of a matrix, which can be used to
    /// fill the specified area with a value.
    fn tri_mask<S: Into<Shape>>(shape: S, tri_part: TriPart, offset: i64, device: &Device) -> Self {
        let shape: Shape = shape.into();
        let height = shape[D - 2];
        let width = shape[D - 1];

        // Generate row and column index tensors.
        let row_indices: Tensor<1, Int> = Tensor::arange(0..height as i64, device);
        let col_indices: Tensor<1, Int> = Tensor::arange(0..width as i64, device);

        // Prepare shapes for broadcasting.
        let mut row_shape = [1; D];
        row_shape[D - 2] = height;
        let mut col_shape = [1; D];
        col_shape[D - 1] = width;

        // Reshape for broadcasting.
        let row_broadcast: Tensor<D, Int> = row_indices.reshape(Shape::new(row_shape));
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
    ///   towards the upper triangle.
    /// * `device`: The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// Returns a boolean tensor where `false` indicates the elements of the matrix that are part of the
    /// upper triangle taking into account the specified `offset`. All other elements are `true`.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///   let mask = Tensor::<2, Bool>::triu_mask([3, 3], 0, &Default::default());
    ///   println!("{mask}");
    ///   // [[false, false, false],
    ///   //  [true, false, false],
    ///   //  [true, true, false]]
    /// }
    /// ```
    pub fn triu_mask<S: Into<Shape>>(shape: S, offset: i64, device: &Device) -> Self {
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
    ///   towards the lower triangle.
    /// * `device`: The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// Returns a boolean tensor where `false` indicates the elements of the matrix that are part of the
    /// lower triangle taking into account the specified `offset`. All other elements are `true`.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///   let mask = Tensor::<2, Bool>::tril_mask([3, 3], 0, &Default::default());
    ///   println!("{mask}");
    ///   // [[false, true, true],
    ///   //  [false, false, true],
    ///   //  [false, false, false]]
    /// }
    /// ```
    pub fn tril_mask<S: Into<Shape>>(shape: S, offset: i64, device: &Device) -> Self {
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
    /// * `offset`: The offset from the diagonal, where 0 means the diagonal, and positive values shift
    ///   towards the upper triangle.
    /// * `device`: The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// Returns a boolean tensor where `false` indicates the elements of the matrix that are part of the
    /// diagonal. All other elements are `true`.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example() {
    ///   let mask = Tensor::<2, Bool>::diag_mask([3, 3], 0, &Default::default());
    ///   println!("{mask}");
    ///   // [[false, true, true],
    ///   //  [true, false, true],
    ///   //  [true, true, false]]
    /// }
    /// ```
    pub fn diag_mask<S: Into<Shape>>(shape: S, offset: i64, device: &Device) -> Self {
        Self::tri_mask(shape, TriPart::Diagonal, offset, device)
    }
}
