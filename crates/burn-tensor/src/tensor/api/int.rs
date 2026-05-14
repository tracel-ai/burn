use burn_backend::{ElementConversion, Scalar, ops::IntTensorOps};
use burn_dispatch::Dispatch;

use crate::{
    Cast, Device, Float, Int, Shape, Tensor, TensorCreationOptions, TensorData, TensorPrimitive,
    cartesian_grid,
};

use core::ops::Range;

impl Tensor<1, Int> {
    /// Returns a new integer tensor on the specified device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    /// * `device` - The device to create the tensor on.
    pub fn arange(range: Range<i64>, options: impl Into<TensorCreationOptions>) -> Self {
        let opt = options.into();
        let dtype = opt.resolve_dtype::<Int>();
        Tensor::new(Dispatch::int_arange(
            range,
            &opt.device.dispatch,
            dtype.into(),
        ))
    }

    /// Returns a new integer tensor on the specified device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    /// * `step` - The step between each value.
    pub fn arange_step(
        range: Range<i64>,
        step: usize,
        options: impl Into<TensorCreationOptions>,
    ) -> Self {
        let opt = options.into();
        let dtype = opt.resolve_dtype::<Int>();
        Tensor::new(Dispatch::int_arange_step(
            range,
            step,
            &opt.device.dispatch,
            dtype.into(),
        ))
    }
}

impl<const D: usize> Tensor<D, Int> {
    /// Create a tensor from integers (i32), placing it on a given device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Int};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let _x: Tensor<1, Int> = Tensor::from_ints([1, 2], &device);
    ///     let _y: Tensor<2, Int> = Tensor::from_ints([[1, 2], [3, 4]], &device);
    /// }
    /// ```
    pub fn from_ints<A: Into<TensorData>>(ints: A, device: &Device) -> Self {
        Self::from_data(ints.into().convert::<i32>(), device)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// cast to Float.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let int_tensor = Tensor::<1, Int>::arange(0..5, &device);
    ///     let float_tensor = int_tensor.float();
    /// }
    /// ```
    pub fn float(self) -> Tensor<D, Float> {
        let out_dtype = self.device().settings().float_dtype;
        Tensor::new(TensorPrimitive::Float(Dispatch::int_into_float(
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
    ///    use burn_tensor::{Shape, Tensor};
    ///    fn example() {
    ///        let device = Default::default();
    ///        let result: Tensor<3, _> = Tensor::<2, Int>::cartesian_grid([2, 3], &device);
    ///        println!("{}", result);
    ///    }
    /// ```
    pub fn cartesian_grid<S: Into<Shape>, const D2: usize>(
        shape: S,
        device: &Device,
    ) -> Tensor<D2, Int> {
        cartesian_grid::<S, D, D2>(shape, device)
    }

    /// Applies the bitwise logical and operation with each bit representing the integer.
    pub fn bitwise_and(self, other: Self) -> Self {
        Self::new(Dispatch::bitwise_and(self.primitive, other.primitive))
    }

    /// Applies the bitwise logical or operation with another tensor.
    pub fn bitwise_or(self, other: Self) -> Self {
        Self::new(Dispatch::bitwise_or(self.primitive, other.primitive))
    }

    /// Applies the bitwise logical xor operation with another tensor.
    pub fn bitwise_xor(self, other: Self) -> Self {
        Self::new(Dispatch::bitwise_xor(self.primitive, other.primitive))
    }

    /// Applies the bitwise logical not operation.
    pub fn bitwise_not(self) -> Self {
        Self::new(Dispatch::bitwise_not(self.primitive))
    }

    /// Applies the bitwise logical and operation with each bit in the scalar and the integers in the tensor.
    pub fn bitwise_and_scalar(self, other: impl ElementConversion) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(Dispatch::bitwise_and_scalar(self.primitive, other))
    }

    /// Applies the bitwise logical or operation with each bit in the scalar and the integers in the tensor.
    pub fn bitwise_or_scalar(self, other: impl ElementConversion) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(Dispatch::bitwise_or_scalar(self.primitive, other))
    }

    /// Applies bitwise logical xor operation with each bit in the scalar and the integers in the tensor.
    pub fn bitwise_xor_scalar(self, other: impl ElementConversion) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(Dispatch::bitwise_xor_scalar(self.primitive, other))
    }

    /// Applies the bitwise left shift operation with the integers in the tensor.
    pub fn bitwise_left_shift(self, other: Self) -> Self {
        Self::new(Dispatch::bitwise_left_shift(
            self.primitive,
            other.primitive,
        ))
    }

    /// Applies the bitwise right shift operation with the integers in the tensor.
    pub fn bitwise_right_shift(self, other: Self) -> Self {
        Self::new(Dispatch::bitwise_right_shift(
            self.primitive,
            other.primitive,
        ))
    }

    /// Applies the bitwise left shift operation with the scalar.
    pub fn bitwise_left_shift_scalar(self, other: impl ElementConversion) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(Dispatch::bitwise_left_shift_scalar(self.primitive, other))
    }

    /// Applies the bitwise right shift operation with the scalar.
    pub fn bitwise_right_shift_scalar(self, other: impl ElementConversion) -> Self {
        let other = Scalar::new(other, &self.dtype());
        Self::new(Dispatch::bitwise_right_shift_scalar(self.primitive, other))
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
    /// use burn_tensor::{Tensor, Int, IntDType, FloatDType};
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let int_tensor = Tensor::<1, Int>::arange(0..5, &device);
    ///
    ///     // Within-kind cast (int to int)
    ///     let i64_tensor = int_tensor.clone().cast(IntDType::I64);
    ///
    ///     // Cross-kind cast (int to float)
    ///     let float_tensor = int_tensor.cast(FloatDType::F32);
    /// }
    /// ```
    #[must_use]
    pub fn cast<T: Cast<Int>>(self, dtype: T) -> Tensor<D, T::OutputKind> {
        Tensor::new(T::cast(self.primitive, dtype))
    }
}
