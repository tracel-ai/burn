use burn_backend::{Scalar, get_device_settings};

use crate::{
    Cast, Float, Int, Shape, Tensor, TensorCreationOptions, TensorData, TensorPrimitive,
    backend::Backend, cartesian_grid,
};

use core::ops::Range;

impl<B> Tensor<B, 1, Int>
where
    B: Backend,
{
    /// Returns a new 1D integer tensor with values from `range.start` (inclusive)
    /// to `range.end` (exclusive), i.e. `[start, start+1, ..., end)`.
    ///
    /// # Arguments
    ///
    /// * `range` - The half-open range of values to generate.
    /// * `options` - Accepts either `&device` (dtype resolved from backend policy) or
    ///   `(&device, DType)` (dtype pinned explicitly). See [`TensorCreationOptions`].
    ///
    /// # Dtype resolution
    ///
    /// When `options` is just `&device`, the dtype is the backend's default `IntElem`
    /// per the device's [default settings](crate::set_default_dtypes). That default
    /// is backend-specific - for example, `burn-flex` defaults to I32. When
    /// `options` is `(&device, DType)`, the given dtype is used verbatim.
    ///
    /// # Index tensors
    ///
    /// `arange` is the most common way to build an index tensor. When the result
    /// will be passed to [`Tensor::select`](crate::Tensor::select),
    /// [`gather`](crate::Tensor::gather), [`scatter`](crate::Tensor::scatter),
    /// [`select_assign`](crate::Tensor::select_assign), or
    /// [`take`](crate::Tensor::take), prefer pinning the dtype to `DType::I64`:
    ///
    /// ```ignore
    /// use burn_tensor::{DType, Int, Tensor};
    /// # fn example<B: burn_tensor::backend::Backend>(device: B::Device) {
    /// let idx = Tensor::<B, 1, Int>::arange(0..k as i64, (&device, DType::I64));
    /// # let _: Tensor<B, 1, Int> = idx;
    /// # }
    /// ```
    ///
    /// I64 is the cross-backend portable choice for indices: it matches the PyTorch
    /// and ONNX conventions, is required by backends that wrap libtorch (e.g.
    /// `burn-tch`), and avoids the ~2.1B range ceiling of I32. Relying on the
    /// backend-default `IntElem` is risky because that default varies per backend.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{DType, Int, Tensor};
    /// use burn_tensor::backend::Backend;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     // Dtype from backend default:
    ///     let a = Tensor::<B, 1, Int>::arange(0..5, &device);
    ///     // Dtype pinned to I64 for portability (recommended for indices):
    ///     let b = Tensor::<B, 1, Int>::arange(0..5, (&device, DType::I64));
    /// }
    /// ```
    pub fn arange(range: Range<i64>, options: impl Into<TensorCreationOptions<B>>) -> Self {
        let opt = options.into();
        let dtype = opt.resolve_dtype::<Int>();
        Tensor::new(B::int_arange(range, &opt.device, dtype.into()))
    }

    /// Returns a new 1D integer tensor with values `start`, `start + step`,
    /// `start + 2 * step`, ..., up to but not including `range.end`.
    ///
    /// # Arguments
    ///
    /// * `range` - The half-open range of values to generate.
    /// * `step` - The step between each value.
    /// * `options` - Accepts either `&device` or `(&device, DType)`. See
    ///   [`arange`](Self::arange) for full dtype resolution semantics and the
    ///   recommended I64 pinning for index tensors.
    pub fn arange_step(
        range: Range<i64>,
        step: usize,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Self {
        let opt = options.into();
        let dtype = opt.resolve_dtype::<Int>();
        Tensor::new(B::int_arange_step(range, step, &opt.device, dtype.into()))
    }
}

impl<const D: usize, B> Tensor<B, D, Int>
where
    B: Backend,
{
    /// Create a tensor from integers (i32), placing it on a given device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let _x: Tensor<B, 1, Int> = Tensor::from_ints([1, 2], &device);
    ///     let _y: Tensor<B, 2, Int> = Tensor::from_ints([[1, 2], [3, 4]], &device);
    /// }
    /// ```
    pub fn from_ints<A: Into<TensorData>>(ints: A, device: &B::Device) -> Self {
        Self::from_data(ints.into().convert::<i32>(), device)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// cast to Float.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let int_tensor = Tensor::<B, 1, Int>::arange(0..5, &device);
    ///     let float_tensor = int_tensor.float();
    /// }
    /// ```
    pub fn float(self) -> Tensor<B, D, Float> {
        let out_dtype = get_device_settings::<B>(&self.device()).float_dtype;
        Tensor::new(TensorPrimitive::Float(B::int_into_float(
            self.primitive,
            out_dtype,
        )))
    }

    /// Generates a cartesian grid for the given tensor shape on the specified device.
    /// The generated tensor is of dimension `D2 = D + 1`, where each element at dimension D contains the cartesian grid coordinates for that element.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape specifying the dimensions of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Panics
    ///
    /// Panics if `D2` is not equal to `D+1`.
    ///
    /// # Examples
    ///
    /// ```rust
    ///    use burn_tensor::Int;
    ///    use burn_tensor::{backend::Backend, Shape, Tensor};
    ///    fn example<B: Backend>() {
    ///        let device = Default::default();
    ///        let result: Tensor<B, 3, _> = Tensor::<B, 2, Int>::cartesian_grid([2, 3], &device);
    ///        println!("{}", result);
    ///    }
    /// ```
    pub fn cartesian_grid<S: Into<Shape>, const D2: usize>(
        shape: S,
        device: &B::Device,
    ) -> Tensor<B, D2, Int> {
        cartesian_grid::<B, S, D, D2>(shape, device)
    }

    /// Applies the bitwise logical and operation with each bit representing the integer.
    pub fn bitwise_and(self, other: Self) -> Self {
        Self::new(B::bitwise_and(self.primitive, other.primitive))
    }

    /// Applies the bitwise logical or operation with another tensor.
    pub fn bitwise_or(self, other: Self) -> Self {
        Self::new(B::bitwise_or(self.primitive, other.primitive))
    }

    /// Applies the bitwise logical xor operation with another tensor.
    pub fn bitwise_xor(self, other: Self) -> Self {
        Self::new(B::bitwise_xor(self.primitive, other.primitive))
    }

    /// Applies the bitwise logical not operation.
    pub fn bitwise_not(self) -> Self {
        Self::new(B::bitwise_not(self.primitive))
    }

    /// Applies the bitwise logical and operation with each bit in the scalar and the integers in the tensor.
    pub fn bitwise_and_scalar(self, other: B::IntElem) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(B::bitwise_and_scalar(self.primitive, other))
    }

    /// Applies the bitwise logical or operation with each bit in the scalar and the integers in the tensor.
    pub fn bitwise_or_scalar(self, other: B::IntElem) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(B::bitwise_or_scalar(self.primitive, other))
    }

    /// Applies bitwise logical xor operation with each bit in the scalar and the integers in the tensor.
    pub fn bitwise_xor_scalar(self, other: B::IntElem) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(B::bitwise_xor_scalar(self.primitive, other))
    }

    /// Applies the bitwise left shift operation with the integers in the tensor.
    pub fn bitwise_left_shift(self, other: Self) -> Self {
        Self::new(B::bitwise_left_shift(self.primitive, other.primitive))
    }

    /// Applies the bitwise right shift operation with the integers in the tensor.
    pub fn bitwise_right_shift(self, other: Self) -> Self {
        Self::new(B::bitwise_right_shift(self.primitive, other.primitive))
    }

    /// Applies the bitwise left shift operation with the scalar.
    pub fn bitwise_left_shift_scalar(self, other: B::IntElem) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(B::bitwise_left_shift_scalar(self.primitive, other))
    }

    /// Applies the bitwise right shift operation with the scalar.
    pub fn bitwise_right_shift_scalar(self, other: B::IntElem) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(B::bitwise_right_shift_scalar(self.primitive, other))
    }

    /// Converts a tensor to the specified data type.
    ///
    /// Supports both within-kind casting (e.g., `IntDType::I64`) and cross-kind casting
    /// (e.g., `FloatDType::F32` to produce a float tensor).
    ///
    /// This is a no-op when casting to the current dtype within the same kind.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int, IntDType, FloatDType};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let int_tensor = Tensor::<B, 1, Int>::arange(0..5, &device);
    ///
    ///     // Within-kind cast (int to int)
    ///     let i64_tensor = int_tensor.clone().cast(IntDType::I64);
    ///
    ///     // Cross-kind cast (int to float)
    ///     let float_tensor = int_tensor.cast(FloatDType::F32);
    /// }
    /// ```
    #[must_use]
    pub fn cast<T: Cast<B, Int>>(self, dtype: T) -> Tensor<B, D, T::OutputKind> {
        Tensor::new(T::cast(self.primitive, dtype))
    }
}
