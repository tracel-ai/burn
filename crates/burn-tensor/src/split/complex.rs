use burn_backend::{Layout, SplitLayout, ops::ComplexTensorOps, try_read_sync};
use burn_dispatch::DispatchTensor;
use burn_std::{
    AsIndex, ComplexScalar, Distribution, Element, ElementConversion, ExecutionError, Scalar,
    Shape, TensorData,
};

use crate::{
    Bool, BroadcastArgs, Complex, Device, Float, Int, ReshapeArgs, Tensor, TensorCreationOptions,
    atan2_impl, bool_and_impl, bool_or_impl,
    check::TensorCheck,
    ops::{BasicOps, BridgeTensor, FloatMathOps, Numeric},
    split::base::{SplitBackend, SplitPrimitive, SplitTensor},
};

#[derive(Clone, Debug)]
pub struct SplitComplexLayout;

mod backend;

impl Layout for SplitComplexLayout {}
impl SplitLayout for SplitComplexLayout {}

impl<const D: usize> SplitTensor<D, Complex> {
    /// Creates a complex tensor from separate real and imaginary tensor components.
    ///
    /// # Panics
    ///
    /// Panics if the two components do not have the same shape or dtype.
    pub fn new(real: BridgeTensor, imag: BridgeTensor) -> Self {
        assert_eq!(
            real.shape(),
            imag.shape(),
            "Real and imaginary parts must have the same shape"
        );
        assert_eq!(
            real.dtype(),
            imag.dtype(),
            "Real and imaginary parts must have the same dtype"
        );
        Self {
            _kind: core::marker::PhantomData,
            components: [real, imag],
        }
    }
    #[allow(unused)]
    /// Converts this split complex tensor into its backend primitive representation.
    pub(crate) fn into_primitive(self) -> SplitPrimitive<DispatchTensor, 2> {
        let [real, imag] = self.components;
        SplitPrimitive([real.into(), imag.into()])
    }

    /// Returns the public complex dtype of this tensor.
    pub fn dtype(&self) -> burn_std::DType {
        burn_std::complex_utils::real_to_complex_dtype(self.inner_dtype())
    }

    /// Returns the underlying real dtype used by the split tensor components.
    pub fn inner_dtype(&self) -> burn_std::DType {
        self.components[0].dtype()
    }

    /// Creates a complex tensor from separate real and imaginary tensor data.
    ///
    /// # Panics
    ///
    /// Panics if the real and imaginary data do not have the same shape or dtype.
    pub fn from_parts_data(real: TensorData, imag: TensorData, device: &Device) -> Self {
        let dtype = real.dtype;
        let shape = &real.shape;
        assert_eq!(
            shape, &imag.shape,
            "Real and imaginary parts must have the same shape"
        );
        assert_eq!(
            dtype, imag.dtype,
            "Real and imaginary parts must have the same dtype"
        );
        let real_tensor = Float::from_data(real, device, dtype);
        let imag_tensor = Float::from_data(imag, device, dtype);

        Self::new(real_tensor, imag_tensor)
    }

    /// Creates a complex tensor from real-valued data, filling the imaginary part with zeros.
    pub fn from_real_data(data: TensorData, device: &Device) -> Self {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        Self {
            _kind: core::marker::PhantomData,
            components: [
                Float::from_data(data, device, dtype),
                Float::zeros(
                    shape,
                    device,
                    burn_std::complex_utils::real_to_complex_dtype(dtype),
                ),
            ],
        }
    }

    /// Creates a complex tensor from imaginary-valued data, filling the real part with zeros.
    pub fn from_imag_data(data: TensorData, device: &Device) -> Self {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        Self {
            _kind: core::marker::PhantomData,
            components: [
                Float::zeros(shape, device, dtype),
                Float::from_data(data, device, dtype),
            ],
        }
    }

    /// Creates a complex tensor from a tuple of real and imaginary tensor data.
    ///
    /// # Panics
    ///
    /// Panics if the real and imaginary data do not have the same shape or dtype.
    pub fn from_split_data(data: (TensorData, TensorData), device: &Device) -> Self {
        let (real, imag) = data;
        let shape = &real.shape;
        let dtype = real.dtype;
        assert_eq!(
            shape, &imag.shape,
            "Real and imaginary parts must have the same shape"
        );
        assert_eq!(
            dtype, imag.dtype,
            "Real and imaginary parts must have the same dtype"
        );

        Self {
            _kind: core::marker::PhantomData,
            components: [
                Float::from_data(real, device, dtype),
                Float::from_data(imag, device, dtype),
            ],
        }
    }

    /// Returns the real component as a float tensor, consuming the complex tensor.
    pub fn real(self) -> Tensor<D, Float> {
        let [real, _imag] = self.components;
        Tensor::new(real)
    }

    /// Returns the imaginary component as a float tensor, consuming the complex tensor.
    pub fn imag(self) -> Tensor<D, Float> {
        let [_real, imag] = self.components;
        Tensor::new(imag)
    }

    /// Returns a shared reference to the real component tensor.
    pub fn real_ref(&self) -> &BridgeTensor {
        &self.components[0]
    }

    /// Returns a shared reference to the imaginary component tensor.
    pub fn imag_ref(&self) -> &BridgeTensor {
        &self.components[1]
    }

    /// Splits this complex tensor into its real and imaginary float tensor parts.
    pub fn into_parts(self) -> (Tensor<D, Float>, Tensor<D, Float>) {
        let [real, imag] = self.components;
        (Tensor::new(real), Tensor::new(imag))
    }

    /// Converts the data of the current complex tensor asynchronously.
    ///
    /// Returns the data as interleaved real and imaginary values.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub async fn into_interleaved_data(self) -> Result<TensorData, ExecutionError> {
        let [real, imag] = self.components;
        Ok(burn_std::complex_utils::interleaved_data_from_parts_data(
            Float::into_data_async(real).await?,
            Float::into_data_async(imag).await?,
        ))
    }
}

// ComplexOps
impl<const D: usize> SplitTensor<D, Complex> {
    /// Returns the complex conjugate of each element.
    ///
    /// For a complex number `a + bi`, the conjugate is `a - bi`.
    pub fn conj(self) -> Self {
        // conj(a + bi) = a - bi
        let [real, imag] = self.components;
        SplitTensor::new(real, Float::neg(imag))
    }

    /// Applies element wise power operation with a complex Tensor exponent.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to apply the power operation with.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Complex, ComplexScalar, Shape, Int};
    ///
    /// fn example() {
    ///    let device = Default::default();
    ///    let tensor1 = Tensor::<2, Complex>::from_complex([[ComplexScalar::new(1.0, -2.0), ComplexScalar::new(3.0, 4.0), ComplexScalar::new(0.0, -1.0)], [ComplexScalar::new(1.0, -2.0), ComplexScalar::new(0.0, -1.0), ComplexScalar::new(2.0, 2.0)]], &device);
    ///    let tensor2 = Tensor::<2, Complex>::from_complex([[ComplexScalar::new(5.0, -1.0), ComplexScalar::new(2.0, 3.0), ComplexScalar::new(1.0, -2.0)], [ComplexScalar::new(1.0, -3.0), ComplexScalar::new(1.0, -3.0), ComplexScalar::new(6.0, 2.0)]], &device);
    ///    let tensor = tensor1.powc(tensor2);
    ///    println!("{tensor}");
    ///    // [[ 1.84452120e+01-1.05764765e+00i,  1.42600948e+00+6.02434630e-01i,
    ///    // 2.64608933e-18-4.32139183e-02i],
    ///    //  [-7.49735280e-02+2.99204278e-02i,  5.50067930e-19-8.98329102e-03i,
    ///    //  9.29602961e+01+5.18329310e+01i]]
    /// }
    /// ```
    pub fn powc(self, exponent: Self) -> Self {
        SplitBackend::complex_powc(self.into(), exponent.into()).into()
    }
    /// Create a Complex Tensor from a float tensor representing the real part, filling the imaginary part with zeros.
    pub fn from_real(tensor: Tensor<D, Float>) -> Self {
        let shape = tensor.shape();
        let dtype = tensor.dtype();
        let device = tensor.device();
        Self::new(tensor.primitive, Float::zeros(shape, &device, dtype))
    }

    /// Returns the argument (phase angle) of each element, in radians.
    ///
    /// For a complex number `a + bi`, the phase is `atan2(b, a)`, ranging from `-π` to `π`.
    pub fn phase(self) -> Tensor<D, Float> {
        // arg(a + bi) = atan2(b, a)
        let [real, imag] = self.components;
        Tensor::new(atan2_impl(imag, real))
    }

    /// Returns the magnitude (absolute value, modulus) of each element.
    ///
    /// For a complex number `a + bi`, the magnitude is `sqrt(a² + b²)`.
    pub fn magnitude(self) -> Tensor<D, Float> {
        //could use a hypot function for float kinds
        let [real, imag] = self.components;
        Tensor::new(Float::sqrt(Float::add(
            Float::mul(real.clone(), real),
            Float::mul(imag.clone(), imag),
        )))
    }

    /// Applies element-wise complex exponential.
    ///
    /// For a complex number `a + bi`, computes `exp(a) * (cos(b) + i·sin(b))`.
    pub fn exp(self) -> Self {
        // formula: e^(a + bi) = e^a * (cos(b) + i*sin(b)) = from_polar(e^a, b)
        //TODO: add the checks for corner cases +∞, -∞, and NaN
        //https://github.com/skewballfox/burn/blob/67d84b677b3d718cb25fbdc2535dbf04706b0863/crates/burn-complex/src/base/element.rs#L322-L323
        let [real, imag] = self.components;
        let exp_real = Float::exp(real);
        let cos_imag = Float::cos(imag.clone());
        let sin_imag = Float::sin(imag);
        SplitTensor::new(
            Float::mul(exp_real.clone(), cos_imag),
            Float::mul(exp_real, sin_imag),
        )
    }

    /// Applies element-wise complex sine.
    pub fn sin(self) -> Self {
        // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let [real, imag] = self.components;
        SplitTensor::new(
            Float::mul(Float::sin(real.clone()), Float::cosh(imag.clone())),
            Float::mul(Float::cos(real), Float::sinh(imag)),
        )
    }

    /// Create a complex tensor from separate real and imaginary data.
    ///
    /// # Arguments
    ///
    /// * `real` - The real part data.
    /// * `imag` - The imaginary part data.
    pub fn from_parts<T: Into<TensorData>>(real: T, imag: T, device: &Device) -> Self {
        let (real, imag) = (real.into(), imag.into());
        let dtype = real.dtype;
        assert_eq!(
            dtype, imag.dtype,
            "from_parts: real and imaginary data must have the same dtype, got {:?} and {:?}",
            dtype, imag.dtype
        );
        let real = Float::from_data(real, device, dtype);
        let imag = Float::from_data(imag, device, dtype);
        Self::new(real, imag)
    }

    /// Create a complex tensor from interleaved (real, imaginary) data.
    ///
    /// The input data should contain alternating real and imaginary values.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved complex data.
    /// * `device` - The device to create the tensor on.
    pub fn from_interleaved_data(data: TensorData, device: &Device) -> Self {
        Self::from_split_data(
            burn_std::complex_utils::split_from_interleaved_data(data),
            device,
        )
    }

    /// Create a complex tensor from polar form.
    ///
    /// Constructs a complex tensor where each element `z = r · exp(i · θ)`,
    /// given magnitude `r` and phase angle `θ`.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - The magnitude (modulus) of each element.
    /// * `phase` - The phase angle of each element, in radians.
    pub fn from_polar(magnitude: Tensor<D, Float>, phase: Tensor<D, Float>) -> Self {
        Self::new(
            (magnitude.clone() * phase.clone().cos().clone()).primitive,
            (magnitude * phase.sin()).primitive,
        )
    }

    /// Converts the data of the current complex tensor asynchronously.
    ///
    /// Returns the data as interleaved real and imaginary values.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub async fn into_data_async(self) -> Result<TensorData, ExecutionError> {
        self.into_interleaved_data().await
    }

    /// Converts the data of the current tensor and returns any error that might have occurred since the
    /// last time the device was synchronized.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn try_into_data(self) -> Result<TensorData, ExecutionError> {
        try_read_sync(self.into_data_async()).expect(
            "Failed to read tensor data synchronously.
        This can happen on platforms that don't support blocking futures like WASM.
        If possible, try using into_data_async instead.",
        )
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn into_data(self) -> TensorData {
        self.try_into_data().expect(
            "Error while reading data: use `try_into_data` instead to catch the error at runtime",
        )
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn to_data(&self) -> TensorData {
        self.clone().into_data()
    }

    /// Create a split complex tensor from the given interleaved complex data on the given device.
    ///
    /// # Arguments
    ///
    /// * `data` - The interleaved complex data (alternating real and imaginary values).
    /// * `options` - Options to control creation, including device and dtype.
    pub fn from_data<T>(data: T, options: impl Into<TensorCreationOptions>) -> Self
    where
        T: Into<TensorData>,
    {
        let data = data.into();
        let opt = options.into();
        Self::from_split_data(
            burn_std::complex_utils::split_from_interleaved_data(data),
            &opt.device,
        )
    }
}

// Basic Operations that should be generic but I haven't yet figured out how to make them work in safe rust
// Pretty much every operation that either creates a new tensor without a self argument, or potentially changes the rank
impl<const D: usize> SplitTensor<D, Complex> {
    /// Create an empty complex tensor of the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn empty<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions>) -> Self {
        let opt = options.into();
        let shape = shape.into();
        //check!(TensorCheck::creation_ops::<D>("Empty", &shape));
        Self::zeros(shape, &opt.device)
    }

    /// Create a tensor of the given shape where each element is one.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<2>::ones(Shape::new([2, 3]), &device);
    ///   println!("{tensor}");
    ///   // [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    /// }
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions>) -> Self {
        let opt = options.into();
        let shape = shape.into();
        let dtype = opt.resolve_dtype::<Float>();
        crate::check!(TensorCheck::creation_ops::<D>("Ones", &shape));
        Self::new(
            Float::ones(shape.clone(), &opt.device, dtype),
            Float::zeros(shape, &opt.device, dtype),
        )
    }

    /// Create a tensor of the given shape where each element is zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<2>::zeros(Shape::new([2, 3]), &device);
    ///    println!("{tensor}");
    ///    // [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    /// }
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions>) -> Self {
        let opt = options.into();
        let shape = shape.into();
        let dtype = opt.resolve_dtype::<Float>();
        crate::check!(TensorCheck::creation_ops::<D>("Zeros", &shape));
        Self::new(
            Float::zeros(shape.clone(), &opt.device, dtype),
            Float::zeros(shape, &opt.device, dtype),
        )
    }

    /// Applies element-wise equal comparison with a scalar and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare each complex element against.
    pub fn equal_elem<E: Element>(self, other: E) -> Tensor<D, Bool> {
        let rhs = other.elem::<ComplexScalar<f64>>();
        let [lhs_real, lhs_imag] = self.components;
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();
        let real_cmp = Float::equal_elem(lhs_real, burn_std::Scalar::Float(rhs_real));
        let imag_cmp = Float::equal_elem(lhs_imag, burn_std::Scalar::Float(rhs_imag));
        Tensor::new(bool_and_impl(real_cmp, imag_cmp))
    }

    /// Applies element-wise non-equality comparison with a scalar and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare each complex element against.
    pub fn not_equal_elem<E: Element>(self, other: E) -> Tensor<D, Bool> {
        let rhs = other.elem::<ComplexScalar<f64>>();
        let [lhs_real, lhs_imag] = self.components;
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();
        let real_cmp = Float::not_equal_elem(lhs_real, burn_std::Scalar::Float(rhs_real));
        let imag_cmp = Float::not_equal_elem(lhs_imag, burn_std::Scalar::Float(rhs_imag));
        Tensor::new(bool_or_impl(real_cmp, imag_cmp))
    }

    /// Concatenates all tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// - If `dim` is higher than the rank.
    /// - If `tensors` is an empty vector.
    /// - If all tensors don't have the same shape (the dimension `dim` is ignored).
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::Tensor;
    ///
    /// fn example() {
    ///     let device = Default::default();
    ///     let t1 = Tensor::<2>::from_data([[3.0, 4.9, 2.0, 1.0], [2.0, 1.9, 3.0, 1.0]], &device);
    ///     let t2 = Tensor::<2>::from_data([[4.0, 5.9, 8.0], [1.4, 5.8, 6.0]], &device);
    ///
    ///     // Concatenate the two tensors with shapes [2, 4] and [2, 3] along the dimension 1.
    ///     // [[3.0, 4.9, 2.0, 1.0, 4.0, 5.9, 8.0], [2.0, 1.9, 3.0, 1.0, 1.4, 5.8, 6.0]]
    ///     // The resulting tensor will have shape [2, 7].
    ///     let concat = Tensor::cat(vec![t1, t2], 1);
    ///     println!("{concat}");
    /// }
    /// ```
    pub fn cat(tensors: alloc::vec::Vec<Self>, dim: usize) -> Self {
        //crate::check!(TensorCheck::cat(tensors.as_slice(), dim));

        // Filter out tensors with size 0 along the concatenation dimension.
        // Empty tensors don't contribute to the output and would cause issues
        // in backend implementations (e.g., division by zero in slice_assign).
        // Safety: TensorCheck::cat ensures tensors is non-empty
        let first_tensor = tensors.first().unwrap();
        let device = first_tensor.device();
        let mut shape = first_tensor.shape();

        let (non_empty_reals, non_empty_imags): (alloc::vec::Vec<_>, alloc::vec::Vec<_>) = tensors
            .into_iter()
            .filter(|t| t.shape()[dim] > 0)
            .map(|t| {
                let [real, imag] = t.components;
                (real, imag)
            })
            .unzip();

        // If all tensors were empty, return an empty tensor with size 0 on concat dim
        if non_empty_reals.is_empty() {
            shape[dim] = 0;
            return Self::zeros(shape, &device);
        }

        Self::new(
            Float::cat(non_empty_reals, dim),
            Float::cat(non_empty_imags, dim),
        )
    }

    /// Create a random complex tensor of the given shape where each element is sampled from
    /// the given distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `distribution` - The distribution to sample from.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn random<S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        options: impl Into<TensorCreationOptions>,
    ) -> Self {
        // Use the given dtype when provided, otherwise default device dtype
        let opt = options.into();
        let shape = shape.into();
        let dtype = opt.resolve_dtype::<Float>();
        Self::new(
            Float::random(shape.clone(), distribution, &opt.device, dtype),
            Float::random(shape.clone(), distribution, &opt.device, dtype),
        )
    }

    /// Returns a view of the complex tensor with an additional dimension of size `size`
    /// obtained by slicing the tensor along `dim` with step `step`.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to unfold.
    /// * `size` - The size of each unfolded window.
    /// * `step` - The step between each window.
    ///
    /// # Returns
    ///
    /// A tensor with shape `[..., windows, ..., size]` where the extra `size` dimension
    /// is appended at the end.
    pub fn unfold<const D2: usize, I: AsIndex>(
        self,
        dim: I,
        size: usize,
        step: usize,
    ) -> SplitTensor<D2, Complex> {
        let dim = dim.expect_dim_index(D);
        let [real, imag] = self.components;
        SplitTensor::new(
            Float::unfold(real, dim, size, step),
            Float::unfold(imag, dim, size, step),
        )
    }

    /// Broadcast the complex tensor to the given shape.
    ///
    /// Only singleton dimensions can be expanded to a larger size. Other dimensions must have
    /// the same size (which can be inferred with `-1`).
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape to broadcast the tensor to. Can contain -1 for dimensions that
    ///   should be inferred.
    ///
    /// # Panics
    ///
    /// If the tensor cannot be broadcasted to the given shape.
    pub fn expand<const D2: usize, S: BroadcastArgs<D, D2>>(
        self,
        shape: S,
    ) -> SplitTensor<D2, Complex> {
        let shape = shape.into_shape(&self.shape());
        crate::check!(TensorCheck::expand::<D, D2>(
            "expand",
            &self.shape(),
            &shape,
        ));
        let [real, imag] = self.components;
        SplitTensor::<D2, Complex>::new(
            Float::expand(real, shape.clone()),
            Float::expand(imag, shape),
        )
    }

    /// Reshape the complex tensor to have the given shape.
    ///
    /// The tensor has the same data and number of elements as the input.
    ///
    /// A `-1` in the shape is used to infer the remaining dimensions, e.g.: `[2, -1]`
    /// will reshape the tensor with [2, 3, 4] dimensions to [2, 12].
    ///
    /// # Arguments
    /// - `shape`: The new shape of the tensor.
    pub fn reshape<const D2: usize, S: ReshapeArgs<D2>>(
        self,
        shape: S,
    ) -> SplitTensor<D2, Complex> {
        // Convert reshape args to shape
        let shape = shape.into_shape::<D2>(self.shape());
        let [real, imag] = self.components;
        SplitTensor::<D2, Complex>::new(
            Float::reshape(real, shape.clone()),
            Float::reshape(imag, shape),
        )
    }

    /// Update the complex tensor with the scalar value where the mask is true.
    ///
    /// This is similar to [`mask_where`](Self::mask_where), however the value is a scalar
    /// instead of a tensor.
    ///
    /// # Arguments
    ///
    /// * `mask` - A boolean tensor with the same shape as the input tensor.
    /// * `value` - The scalar value to assign where the mask is true.
    pub fn mask_fill<E: ElementConversion>(self, mask: Tensor<D, Bool>, value: E) -> Self {
        let value_complex = value.elem::<ComplexScalar<f64>>();
        let mask = mask.primitive;
        let value_real = burn_std::Scalar::Float(value_complex.real);
        let value_imag = burn_std::Scalar::Float(value_complex.imag);
        let [real, imag] = self.components;
        SplitTensor::new(
            Float::mask_fill(real, mask.clone(), value_real),
            Float::mask_fill(imag, mask, value_imag),
        )
    }
}
// Numeric operations that won't be the same universally
impl<const D: usize> SplitTensor<D, Complex> {
    /// Helper function for computing the squared norm of a complex tensor, which is the sum of squares of the real and imaginary parts.
    pub fn squared_norm(self) -> Tensor<D, Float> {
        let [real, imag] = self.components;
        let real_sq = Float::mul(real.clone(), real);
        let imag_sq = Float::mul(imag.clone(), imag);
        Tensor::new(Float::add(real_sq, imag_sq))
    }
    /// Applies element-wise division operation.
    ///
    /// `y = x2 / x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to divide by.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, rhs: Self) -> Self {
        // (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
        //   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
        let norm_sqr = rhs.clone().squared_norm().primitive;
        let [real, imag] = self.components;
        let [rhs_real, rhs_imag] = rhs.components;
        SplitTensor::new(
            Float::div(
                Float::add(
                    Float::mul(real.clone(), rhs_real.clone()),
                    Float::mul(imag.clone(), rhs_imag.clone()),
                ),
                norm_sqr.clone(),
            ),
            Float::div(
                Float::sub(Float::mul(imag, rhs_real), Float::mul(real, rhs_imag)),
                norm_sqr,
            ),
        )
    }
    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    pub fn matmul(self, rhs: Self) -> Self {
        // (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        let [real, imag] = self.components;
        let [rhs_real, rhs_imag] = rhs.components;
        SplitTensor::new(
            Float::sub(
                Float::matmul(real.clone(), rhs_real.clone()),
                Float::matmul(imag.clone(), rhs_imag.clone()),
            ),
            Float::add(Float::matmul(real, rhs_imag), Float::matmul(imag, rhs_real)),
        )
    }

    /// Applies element-wise division operation with a scalar.
    ///
    /// `y = x / s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to divide by, element-wise.
    pub fn div_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<ComplexScalar<f64>>();
        let scalar_tensor = Self::full(
            shape,
            scalar_complex,
            TensorCreationOptions::new(device).with_dtype(self.dtype()),
        );
        self.div(scalar_tensor)
    }

    /// Applies element-wise the remainder operation.
    ///
    /// `y = x2 % x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to compute the remainder with.
    pub fn remainder(self, rhs: Self) -> Self {
        // Componentwise remainder (matching Complex<E> Rem impl)
        let [real, imag] = self.components;
        let [rhs_real, rhs_imag] = rhs.components;
        SplitTensor::new(
            Float::remainder(real, rhs_real),
            Float::remainder(imag, rhs_imag),
        )
    }

    /// Applies element-wise the remainder operation with a scalar.
    ///
    /// `y = x % s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to compute the remainder with, element-wise.
    pub fn remainder_scalar(self, rhs: burn_std::Scalar) -> Self {
        let rhs = rhs.elem::<ComplexScalar<f64>>();
        let [real, imag] = self.components;
        SplitTensor::new(
            Float::remainder_scalar(real, burn_std::Scalar::Float(rhs.real)),
            Float::remainder_scalar(imag, burn_std::Scalar::Float(rhs.imag)),
        )
    }

    /// Applies element-wise multiplication operation.
    ///
    /// `y = x2 * x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to multiply.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        // (a + i b) * (c + i d) == (a*c - b*d) + i (a*d + b*c)
        let [real, imag] = self.components;
        let [rhs_real, rhs_imag] = rhs.components;
        SplitTensor::new(
            Float::sub(
                Float::mul(real.clone(), rhs_real.clone()),
                Float::mul(imag.clone(), rhs_imag.clone()),
            ),
            Float::add(Float::mul(real, rhs_imag), Float::mul(rhs_real, imag)),
        )
    }

    /// Applies element-wise multiplication operation with a scalar.
    ///
    /// `y = x * s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to multiply, element-wise.
    pub fn mul_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<ComplexScalar<f64>>();
        let scalar_tensor = Self::full(
            shape,
            scalar_complex,
            TensorCreationOptions::new(device).with_dtype(self.dtype()),
        );
        self.mul(scalar_tensor)
    }

    /// Applies element-wise power operation with an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The integer tensor to apply the power operation with.
    pub fn powi(self, other: Tensor<D, Int>) -> Self {
        self.powf(other.float())
    }

    /// Applies element-wise power operation with a floating-point tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The floating-point tensor to apply the power operation with.
    pub fn powf(self, other: Tensor<D, Float>) -> Self {
        // z^w = exp(w * ln(z)) where w is a real tensor
        let [mut log_z_real, mut log_z_imag] = self.log().components;

        log_z_real = Float::mul(other.primitive.clone(), log_z_real);
        log_z_imag = Float::mul(other.primitive, log_z_imag);

        Self {
            _kind: core::marker::PhantomData,
            components: [log_z_real, log_z_imag],
        }
        .exp()
    }

    /// Applies element-wise power operation with an integer scalar.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        self.powf_scalar(other)
    }

    /// Applies element-wise power operation with a floating-point scalar.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    fn powf_scalar<E: ElementConversion>(self, other: E) -> Self {
        let other = Scalar::new(
            other.elem::<f64>(),
            &burn_std::complex_utils::complex_to_real_dtype(self.dtype()),
        );
        let log_z = self.log();
        let [real, imag] = log_z.components;
        let w_log_z = SplitTensor::new(
            Float::mul_scalar(real, other),
            Float::mul_scalar(imag, other),
        );
        w_log_z.exp()
    }

    /// Applies element-wise complex cosine.
    pub fn cos(self) -> Self {
        // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let [real, imag] = self.components;
        SplitTensor::new(
            Float::mul(Float::cos(real.clone()), Float::cosh(imag.clone())),
            Float::neg(Float::mul(Float::sin(real), Float::sinh(imag))),
        )
    }

    /// Applies element-wise complex tangent.
    pub fn tan(self) -> Self {
        // tan(z) = sin(z) / cos(z)
        // Compute sin(a), cos(a), sinh(b), cosh(b) once and share between numerator/denominator.
        let [real, imag] = self.components;
        let sin_a = Float::sin(real.clone());
        let cos_a = Float::cos(real);
        let sinh_b = Float::sinh(imag.clone());
        let cosh_b = Float::cosh(imag);
        let sin_z = SplitTensor::new(
            Float::mul(sin_a.clone(), cosh_b.clone()),
            Float::mul(cos_a.clone(), sinh_b.clone()),
        );
        let cos_z = SplitTensor::new(
            Float::mul(cos_a, cosh_b),
            Float::neg(Float::mul(sin_a, sinh_b)),
        );
        sin_z.div(cos_z)
    }

    /// Applies element-wise complex arccosine.
    pub fn acos(self) -> Self {
        // acos(z) = -i * ln(z + i * sqrt(1 - z²))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype();
        let ones = SplitTensor::new(
            Float::ones(shape.clone(), &device, fdtype),
            Float::zeros(shape, &device, fdtype),
        );
        // 1 - z²
        let z_sq = self.clone().mul(self.clone());
        let one_minus_z_sq = ones.sub(z_sq);
        // i * sqrt(1 - z²): multiply by i via (-imag, real)
        let sqrt_term = one_minus_z_sq.sqrt();
        let [sqrt_real, sqrt_imag] = sqrt_term.components;
        let i_sqrt = SplitTensor::new(Float::neg(sqrt_imag), sqrt_real);
        // z + i*sqrt(1 - z²)
        // -i * ln(inner): multiply by -i via (imag, -real)
        let [log_inner_real, log_inner_imag] = self.add(i_sqrt).log().components;
        SplitTensor::new(log_inner_imag, Float::neg(log_inner_real))
    }

    /// Computes the cumulative product of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative product.
    pub fn cumprod(self, dim: usize) -> Self {
        // cumprod(z, dim) = exp(cumsum(log(z), dim))
        self.log().cumsum(dim).exp()
    }

    /// Aggregate all elements in the complex tensor with the product operation.
    pub fn prod(self) -> Self {
        // prod(z) = exp(sum(log(z)))
        self.log().sum().exp()
    }

    /// Applies element-wise complex hyperbolic arccosine.
    pub fn acosh(self) -> Self {
        // acosh(z) = ln(z + sqrt(z² - 1))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype();
        let ones = SplitTensor::new(
            Float::ones(shape.clone(), &device, fdtype),
            Float::zeros(shape, &device, fdtype),
        );
        // z² - 1
        let z_sq = self.clone().mul(self.clone());
        let z_sq_minus_one = z_sq.sub(ones);
        // z + sqrt(z² - 1)
        let sqrt_term = z_sq_minus_one.sqrt();
        let inner = self.add(sqrt_term);
        inner.log()
    }

    /// Applies element-wise complex arcsine.
    pub fn asin(self) -> Self {
        // asin(z) = -i * ln(i*z + sqrt(1 - z²))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype();
        let ones = SplitTensor::new(
            Float::ones(shape.clone(), &device, fdtype),
            Float::zeros(shape, &device, fdtype),
        );
        // z² and i*z — clone before partial-moving self
        let z_sq = self.clone().mul(self.clone());
        // i*z = (-imag, real)
        let [real, imag] = self.components;
        let i_z = SplitTensor::new(Float::neg(imag), real);
        // 1 - z²
        let one_minus_z_sq = ones.sub(z_sq);
        // i*z + sqrt(1 - z²)
        let sqrt_term = one_minus_z_sq.sqrt();
        let inner = i_z.add(sqrt_term);
        // -i * ln(inner): (imag, -real)
        let [log_inner_real, log_inner_imag] = inner.log().components;
        SplitTensor::new(log_inner_imag, Float::neg(log_inner_real))
    }

    /// Applies element-wise complex hyperbolic arcsine.
    pub fn asinh(self) -> Self {
        // asinh(z) = ln(z + sqrt(z² + 1))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype();
        let ones = SplitTensor::new(
            Float::ones(shape.clone(), &device, fdtype),
            Float::zeros(shape, &device, fdtype),
        );
        // z² + 1
        let z_sq = self.clone().mul(self.clone());
        let z_sq_plus_one = z_sq.add(ones);
        // z + sqrt(z² + 1)
        let sqrt_term = z_sq_plus_one.sqrt();
        let inner = self.add(sqrt_term);
        inner.log()
    }

    /// Applies element-wise complex arctangent.
    pub fn atan(self) -> Self {
        SplitBackend::complex_atan(self.into()).into()
    }

    /// Applies element-wise complex hyperbolic arctangent.
    pub fn atanh(self) -> Self {
        // atanh(z) = (1/2) * ln((1 + z) / (1 - z))
        SplitBackend::complex_atanh(self.into()).into()
    }

    /// Applies element-wise complex natural logarithm.
    ///
    /// For a complex number `z = r · exp(i · θ)`, computes `ln(r) + i · θ`.
    pub fn log(self) -> Self {
        // formula: ln(z) = ln|z| + i*arg(z)
        // where |z| = sqrt(real^2 + imag^2) and arg(z) = atan2(imag, real)

        // Compute norm: sqrt(real^2 + imag^2)
        let norm = self.clone().squared_norm().sqrt();
        let [real, imag] = self.components;
        // Compute arg: atan2(imag, real)
        let arg = atan2_impl(imag, real);

        SplitTensor::<D, Complex>::new(norm.log().primitive, arg)
    }
    /// Applies element-wise complex square root.
    pub fn sqrt(self) -> Self {
        // sqrt(z) = from_polar(sqrt(|z|), arg(z) / 2)
        let abs = self.clone().magnitude();
        let [real, imag] = self.components;
        let sqrt_abs = abs.sqrt();
        let arg = atan2_impl(imag, real);
        let half_arg = Float::div_scalar(arg, burn_std::Scalar::Float(2.0));
        Self::from_polar(
            Tensor::<D, Float>::new(sqrt_abs.primitive),
            Tensor::new(half_arg),
        )
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the product operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn prod_dim(self, dim: usize) -> Self {
        // prod_dim(z, dim) = exp(sum_dim(log(z), dim))
        self.log().sum_dim(dim).exp()
    }

    /// Returns the signs of the elements of the complex tensor.
    ///
    /// For a non-zero element `z`, returns `z / |z|`. For zero, returns zero.
    pub fn sign(self) -> Self {
        // sign(z) = z / |z| = from_polar(1, arg(z))
        let abs = self.clone().magnitude().primitive;
        let [real, imag] = self.components;
        SplitTensor::new(Float::div(real, abs.clone()), Float::div(imag, abs))
    }
}
// #[derive(Debug, Clone)]
// /// A newtype that wraps a real backend B and exposes a split-layout complex backend.
// pub struct SplitBackend;

// impl BackendTypes for SplitBackend<B> {
//     type Device = B::Device;

//     type FloatTensorPrimitive = B::FloatTensorPrimitive;

//     type IntTensorPrimitive = B::IntTensorPrimitive;

//     type BoolTensorPrimitive = B::BoolTensorPrimitive;

//     type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;

//     type ComplexTensorPrimitive = SplitPrimitive<B::FloatTensorPrimitive, 2>;

// }
