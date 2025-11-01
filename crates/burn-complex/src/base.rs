pub mod element;
/*
The base implementation for complex tensors, contains everything that would be in burn-tensor.
May get split into separate files at some point, but for now it's easier to keep all the base
definitions in one spot.
*/
use burn_tensor::{
    BasicOps, Bytes, DType, Device, Distribution, Element, ElementConversion, FloatDType, Int,
    Numeric, Shape, Slice, Tensor, TensorData, TensorKind, TensorMetadata, Transaction,
    backend::Backend, ops::FloatTensor,
};

use core::ops::Range;

/// The layout of the complex tensor. Used to define shared behavior only meant
/// to be used for a specific layout (such as butterfly operations).
pub trait ComplexLayout {}

/// Complex element type used by backend.
pub type ComplexElem<B> = <B as ComplexTensorBackend>::ComplexElem;

/// Complex tensor primitive type used by the backend.
pub type ComplexTensor<B> = <B as ComplexTensorBackend>::ComplexTensorPrimitive;

pub trait ComplexTensorBackend: Backend + ComplexTensorOps<Self> {
    /// The inner backend type.
    type InnerBackend: Backend<Device = Self::Device, FloatElem = Self::FloatElem>;

    /// Tensor primitive to be used for all complex operations.
    type ComplexTensorPrimitive: TensorMetadata + 'static;

    /// Complex element type.
    type ComplexElem: Element;

    /// The underlaying layout for the complex elements
    type Layout: ComplexLayout;

    /// Returns the real part of a complex tensor.
    fn real(tensor: ComplexTensor<Self>) -> FloatTensor<Self::InnerBackend>;
    /// Returns the imaginary part of a complex tensor.
    fn imag(tensor: ComplexTensor<Self>) -> FloatTensor<Self::InnerBackend>;

    fn to_complex(tensor: FloatTensor<Self::InnerBackend>) -> ComplexTensor<Self>;
    // can reuse float random
}
//Note: changing to adopt terminology used in fftw doc

/// Indicates that the underlying implementation has separate real and imaginary tensors.
pub struct SplitLayout;

/// Indicates that the underlying implementation uses a complex primitive type [float,float] like that found in the
/// num_complex trait.
pub struct InterleavedLayout;

pub struct InterleavedTensorData {
    /// The values of the tensor (as bytes).
    pub bytes: Bytes,

    /// The shape of the tensor.
    pub shape: Vec<usize>,

    /// The data type of the tensor.
    pub dtype: DType,
}

pub struct SplitTensorData {
    /// The real values of the tensor (as bytes).
    pub real_bytes: Bytes,

    /// The imaginary values of the tensor (as bytes).
    pub imag_bytes: Bytes,

    /// The shape of the tensor.
    pub shape: Vec<usize>,

    /// The data type of the tensor.
    pub dtype: DType,
}
// /// Indicates that the underlying implementation uses an interleaved layout for complex numbers.
// pub struct InterleavedLayout;

impl ComplexLayout for SplitLayout {}

impl ComplexLayout for InterleavedLayout {}
// impl ComplexLayout for InterleavedLayout {}

/// Operations on complex tensors.
pub trait ComplexTensorOps<B: ComplexTensorBackend> {
    type Layout: ComplexLayout;
    /// Creates a new complex tensor from the data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given data.
    fn complex_from_data(data: TensorData, device: &Device<B>) -> ComplexTensor<B>;

    /// Creates a new complex tensor with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `distribution` - The distribution to sample from.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and random values.
    fn complex_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<B>,
    ) -> ComplexTensor<B>;

    /// Creates a new complex tensor with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and zeros.
    fn complex_zeros(shape: Shape, device: &Device<B>) -> ComplexTensor<B> {
        Self::complex_from_data(TensorData::zeros::<ComplexElem<B>, _>(shape), device)
    }

    /// Creates a new complex tensor with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and ones.
    fn complex_ones(shape: Shape, device: &Device<B>) -> ComplexTensor<B> {
        Self::complex_from_data(TensorData::ones::<ComplexElem<B>, _>(shape), device)
    }

    /// Creates a new complex tensor with the given shape and a single value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `value` - The value to fill the tensor with.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and value.
    fn complex_full(
        shape: Shape,
        fill_value: ComplexElem<B>,
        device: &Device<B>,
    ) -> ComplexTensor<B> {
        Self::complex_from_data(TensorData::full(shape, fill_value), device)
    }

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn complex_shape(tensor: &ComplexTensor<B>) -> Shape;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn complex_to_data(tensor: &ComplexTensor<B>) -> TensorData;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn complex_device(tensor: &ComplexTensor<B>) -> Device<B>;

    /// Moves the tensor to the given device.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    ///
    /// The tensor on the given device.
    fn complex_to_device(tensor: ComplexTensor<B>, device: &Device<B>) -> ComplexTensor<B>;

    /// Converts the tensor to a different element type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the new element type.
    fn complex_into_data(tensor: ComplexTensor<B>) -> TensorData;

    /// Reshapes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `shape` - The new shape.
    ///
    /// # Returns
    ///
    /// The tensor with the new shape.
    fn complex_reshape(tensor: ComplexTensor<B>, shape: Shape) -> ComplexTensor<B>;

    /// Transposes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn complex_transpose(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Adds two tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of adding the two tensors together.
    fn complex_add(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Subtracts the second tensor from the first tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of subtracting the second tensor from the first tensor.
    fn complex_sub(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Multiplies two complex tensors together using complex multiplication.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of multiplying the two tensors together.
    fn complex_mul(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Divides the first tensor by the second tensor using complex division.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of dividing the first tensor by the second tensor.
    fn complex_div(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Negates the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The negated tensor.
    fn complex_neg(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Returns the complex conjugate of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The complex conjugate of the tensor.
    fn complex_conj(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Returns the real part of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the real parts.
    fn complex_real(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Returns the imaginary part of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the imaginary parts.
    fn complex_imag(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Returns the magnitude (absolute value) of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the magnitudes.
    fn complex_abs(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Returns the phase (argument) of the complex tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing the phases in radians.
    fn complex_arg(tensor: ComplexTensor<B>) -> B::FloatTensorPrimitive;

    /// Creates a complex tensor from real and imaginary parts.
    ///
    /// # Arguments
    ///
    /// * `real` - The real part tensor.
    /// * `imag` - The imaginary part tensor.
    ///
    /// # Returns
    ///
    /// A complex tensor constructed from the real and imaginary parts.
    fn complex_from_parts(
        real: B::FloatTensorPrimitive,
        imag: B::FloatTensorPrimitive,
    ) -> ComplexTensor<B>;

    /// Creates a complex tensor from magnitude and phase.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - The magnitude tensor.
    /// * `phase` - The phase tensor (in radians).
    ///
    /// # Returns
    ///
    /// A complex tensor constructed from polar coordinates.
    fn complex_from_polar(
        magnitude: B::FloatTensorPrimitive,
        phase: B::FloatTensorPrimitive,
    ) -> ComplexTensor<B>;

    /// Complex exponential function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The exponential of the tensor.
    fn complex_exp(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex natural logarithm.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the tensor.
    fn complex_log(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex power function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The base tensor.
    /// * `exponent` - The exponent tensor.
    ///
    /// # Returns
    ///
    /// The result of raising the base to the exponent.
    fn complex_powc(lhs: ComplexTensor<B>, rhs: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex square root.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The square root of the tensor.
    fn complex_sqrt(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex sine function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The sine of the tensor.
    fn complex_sin(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex cosine function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The cosine of the tensor.
    fn complex_cos(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex tangent function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The tangent of the tensor.
    fn complex_tan(tensor: ComplexTensor<B>) -> ComplexTensor<B>;

    /// Complex select function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to select.
    /// * `indices` - The indices to select.
    ///
    /// # Returns
    ///
    /// The selected tensor.
    fn select(tensor: ComplexTensor<B>, dim: usize, indices: Tensor<B, 1, Int>)
    -> ComplexTensor<B>;

    /// Complex select assign function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to select.
    /// * `indices` - The indices to select.
    /// * `values` - The values to assign.
    ///
    /// # Returns
    ///
    /// The assigned tensor.
    fn select_assign(
        tensor: ComplexTensor<B>,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: ComplexTensor<B>,
    ) -> ComplexTensor<B>;

    /// Complex slice function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to slice.
    /// * `start` - The start index.
    /// * `end` - The end index.
    ///
    /// # Returns
    ///
    /// The sliced tensor.
    fn slice(tensor: ComplexTensor<B>, dim: usize, start: usize, end: usize) -> ComplexTensor<B>;

    /// Assign the selected elements corresponding for the given ranges to the given value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `ranges` - The ranges to select.
    /// * `value` - The value to assign.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given value.
    fn complex_slice_assign(
        tensor: ComplexTensor<B>,
        ranges: &[Range<usize>],
        value: ComplexTensor<B>,
    ) -> ComplexTensor<B>;
}

/// A type-level representation of the kind of a complex tensor.
#[derive(Clone, Debug)]
pub struct Complex;

#[allow(unused_variables)]
impl<B: ComplexTensorBackend> BasicOps<B> for Complex {
    type Elem = B::ComplexElem;

    fn empty(shape: Shape, device: &B::Device, dtype: DType) -> Self::Primitive {
        // should I check then pass the dtype?
        B::complex_zeros(shape, device)
    }

    fn register_transaction(tr: &mut Transaction<B>, tensor: Self::Primitive) {
        // Complex tensors don't support transactions yet
        // TODO: Implement complex tensor transaction support
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::complex_reshape(tensor, shape)
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        B::complex_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        // TODO: Implement complex_swap_dims in ComplexTensorOps
        todo!("complex_swap_dims not yet implemented")
    }

    fn slice(tensor: Self::Primitive, ranges: &[Slice]) -> Self::Primitive {
        //TensorPrimitive::Complex(B::complex_slice(tensor, ranges))
        todo!()
    }

    fn slice_assign(
        tensor: Self::Primitive,
        ranges: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        //TensorPrimitive::Complex(B::complex_slice_assign(tensor, ranges, value))
        todo!()
    }

    fn device(tensor: &Self::Primitive) -> Device<B> {
        B::complex_device(tensor)
    }

    fn to_device(tensor: Self::Primitive, device: &Device<B>) -> Self::Primitive {
        B::complex_to_device(tensor, device)
    }

    async fn into_data_async(tensor: Self::Primitive) -> TensorData {
        B::complex_into_data(tensor)
    }

    fn from_data(data: TensorData, device: &B::Device) -> Self::Primitive {
        B::complex_from_data(data.convert::<B::ComplexElem>(), device)
    }

    fn from_data_dtype(data: TensorData, device: &B::Device, dtype: DType) -> Self::Primitive {
        if !dtype.is_complex() {
            panic!("Expected complex dtype, got {dtype:?}")
        }
        B::complex_from_data(data.convert_dtype(dtype), device)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        // TODO: Implement complex_repeat_dim in ComplexTensorOps
        todo!("complex_repeat_dim not yet implemented")
    }

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        // TODO: Implement complex_equal in ComplexTensorOps
        todo!("complex_equal not yet implemented")
    }

    fn not_equal(
        lhs: Self::Primitive,
        rhs: Self::Primitive,
    ) -> <B as Backend>::BoolTensorPrimitive {
        // TODO: Implement complex_not_equal in ComplexTensorOps
        todo!("complex_not_equal not yet implemented")
    }

    fn cat(tensors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        // TODO: Implement complex_cat in ComplexTensorOps
        todo!("complex_cat not yet implemented")
    }

    fn any(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        // For complex numbers, "any" typically means any non-zero element
        // TODO: Implement complex_any in ComplexTensorOps
        todo!("complex_any not yet implemented")
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        // TODO: Implement complex_any_dim in ComplexTensorOps
        todo!("complex_any_dim not yet implemented")
    }

    fn all(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        // For complex numbers, "all" typically means all non-zero elements
        // TODO: Implement complex_all in ComplexTensorOps
        todo!("complex_all not yet implemented")
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        // TODO: Implement complex_all_dim in ComplexTensorOps
        todo!("complex_all_dim not yet implemented")
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        // TODO: Implement complex_permute in ComplexTensorOps
        todo!("complex_permute not yet implemented")
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        // TODO: Implement complex_expand in ComplexTensorOps
        todo!("complex_expand not yet implemented")
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        // TODO: Implement complex_flip in ComplexTensorOps
        todo!("complex_flip not yet implemented")
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: Tensor<B, 1, Int>) -> Self::Primitive {
        //TensorPrimitive::Complex(B::complex_select(tensor, dim, indices))
        todo!()
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self::Primitive,
    ) -> Self::Primitive {
        //TensorPrimitive::Complex(B::complex_select_assign(tensor, dim, indices, values))
        todo!()
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        //B::float_unfold(tensor.tensor(), dim, size, step)
        todo!()
    }

    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &<B as Backend>::Device,
        dtype: DType,
    ) -> Self::Primitive {
        todo!()
    }
}

#[allow(unused_variables)]
impl<B: ComplexTensorBackend> Numeric<B> for Complex
where
    B::ComplexElem: Element,
{
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_add(lhs, rhs)
    }

    fn add_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        // TODO: Implement complex_add_scalar in ComplexTensorOps
        // For now, create a tensor with the scalar value and use add
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_add(lhs, scalar_tensor)
    }

    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_sub(lhs, rhs)
    }

    fn sub_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        // TODO: Implement complex_sub_scalar in ComplexTensorOps
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_sub(lhs, scalar_tensor)
    }

    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_mul(lhs, rhs)
    }

    fn mul_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        // TODO: Implement complex_mul_scalar in ComplexTensorOps
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_mul(lhs, scalar_tensor)
    }

    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_div(lhs, rhs)
    }

    fn div_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        // TODO: Implement complex_div_scalar in ComplexTensorOps
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_div(lhs, scalar_tensor)
    }

    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        // Complex remainder is not mathematically well-defined
        todo!("Complex remainder operation is not supported")
    }

    fn remainder_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        // Complex remainder is not mathematically well-defined
        todo!("Complex remainder operation is not supported")
    }

    // fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive {
    //     B::complex_zeros(shape, device)
    // }

    // fn ones(shape: Shape, device: &B::Device) -> Self::Primitive {
    //     B::complex_ones(shape, device)
    // }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_sum in ComplexTensorOps
        todo!("complex_sum not yet implemented")
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        // TODO: Implement complex_sum_dim in ComplexTensorOps
        todo!("complex_sum_dim not yet implemented")
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_prod in ComplexTensorOps
        todo!("complex_prod not yet implemented")
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        // TODO: Implement complex_prod_dim in ComplexTensorOps
        todo!("complex_prod_dim not yet implemented")
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_mean in ComplexTensorOps
        todo!("complex_mean not yet implemented")
    }

    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        // TODO: Implement complex_mean_dim in ComplexTensorOps
        todo!("complex_mean_dim not yet implemented")
    }

    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        B::complex_neg(tensor)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_equal_elem in ComplexTensorOps
        todo!("complex_equal_elem not yet implemented")
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_not_equal_elem in ComplexTensorOps
        todo!("complex_not_equal_elem not yet implemented")
    }

    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        // Greater comparison for complex numbers is typically based on magnitude
        // TODO: Implement complex_greater in ComplexTensorOps
        todo!("complex_greater not yet implemented")
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_greater_elem in ComplexTensorOps
        todo!("complex_greater_elem not yet implemented")
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_greater_equal in ComplexTensorOps
        todo!("complex_greater_equal not yet implemented")
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_greater_equal_elem in ComplexTensorOps
        todo!("complex_greater_equal_elem not yet implemented")
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_lower in ComplexTensorOps
        todo!("complex_lower not yet implemented")
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_lower_elem in ComplexTensorOps
        todo!("complex_lower_elem not yet implemented")
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_lower_equal in ComplexTensorOps
        todo!("complex_lower_equal not yet implemented")
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // TODO: Implement complex_lower_equal_elem in ComplexTensorOps
        todo!("complex_lower_equal_elem not yet implemented")
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        // TODO: Implement complex_mask_where in ComplexTensorOps
        todo!("complex_mask_where not yet implemented")
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Self::Elem,
    ) -> Self::Primitive {
        // TODO: Implement complex_mask_fill in ComplexTensorOps
        todo!("complex_mask_fill not yet implemented")
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive {
        // TODO: Implement complex_gather in ComplexTensorOps
        todo!("complex_gather not yet implemented")
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        value: Self::Primitive,
    ) -> Self::Primitive {
        // TODO: Implement complex_scatter in ComplexTensorOps
        todo!("complex_scatter not yet implemented")
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_sign in ComplexTensorOps
        todo!("complex_sign not yet implemented")
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> B::IntTensorPrimitive {
        // TODO: Implement complex_argmax in ComplexTensorOps
        todo!("complex_argmax not yet implemented")
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> B::IntTensorPrimitive {
        // TODO: Implement complex_argmin in ComplexTensorOps
        todo!("complex_argmin not yet implemented")
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_max in ComplexTensorOps
        todo!("complex_max not yet implemented")
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        // TODO: Implement complex_max_dim in ComplexTensorOps
        todo!("complex_max_dim not yet implemented")
    }

    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        // TODO: Implement complex_max_dim_with_indices in ComplexTensorOps
        todo!("complex_max_dim_with_indices not yet implemented")
    }

    fn max_abs(tensor: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_max_abs in ComplexTensorOps
        todo!("complex_max_abs not yet implemented")
    }

    fn max_abs_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        // TODO: Implement complex_max_abs_dim in ComplexTensorOps
        todo!("complex_max_abs_dim not yet implemented")
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_min in ComplexTensorOps
        todo!("complex_min not yet implemented")
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        // TODO: Implement complex_min_dim in ComplexTensorOps
        todo!("complex_min_dim not yet implemented")
    }

    fn min_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        // TODO: Implement complex_min_dim_with_indices in ComplexTensorOps
        todo!("complex_min_dim_with_indices not yet implemented")
    }

    fn clamp(tensor: Self::Primitive, min: Self::Elem, max: Self::Elem) -> Self::Primitive {
        // TODO: Implement complex_clamp in ComplexTensorOps
        todo!("complex_clamp not yet implemented")
    }

    fn clamp_min(tensor: Self::Primitive, min: Self::Elem) -> Self::Primitive {
        // TODO: Implement complex_clamp_min in ComplexTensorOps
        todo!("complex_clamp_min not yet implemented")
    }

    fn clamp_max(tensor: Self::Primitive, max: Self::Elem) -> Self::Primitive {
        // TODO: Implement complex_clamp_max in ComplexTensorOps
        todo!("complex_clamp_max not yet implemented")
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        // For complex numbers, abs returns the magnitude as a complex number (real part = magnitude, imag = 0)
        let magnitude = B::complex_abs(tensor.clone());
        let zeros = B::float_zeros(
            B::complex_shape(&tensor),
            &B::complex_device(&tensor),
            match tensor.dtype() {
                DType::Complex32 => FloatDType::F32,
                DType::Complex64 => FloatDType::F64,
                _ => panic!("Unsupported complex dtype"),
            },
        );
        B::complex_from_parts(magnitude, zeros)
    }

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_powc(lhs, rhs)
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_powi in ComplexTensorOps
        todo!("complex_powi not yet implemented")
    }

    fn powf_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        let device = B::complex_device(&lhs);
        let shape = B::complex_shape(&lhs);
        let scalar_complex: B::ComplexElem = rhs.elem();
        let scalar_tensor = B::complex_full(shape, scalar_complex, &device);
        B::complex_powc(lhs, scalar_tensor)
    }

    fn powi_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive {
        // TODO: Implement complex_powi_scalar in ComplexTensorOps
        todo!("complex_powi_scalar not yet implemented")
    }

    fn random(shape: Shape, distribution: Distribution, device: &Device<B>) -> Self::Primitive {
        B::complex_random(shape, distribution, device)
    }

    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        // TODO: Implement complex_sort in ComplexTensorOps (based on magnitude)
        todo!("complex_sort not yet implemented")
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, B::IntTensorPrimitive) {
        // TODO: Implement complex_sort_with_indices in ComplexTensorOps
        todo!("complex_sort_with_indices not yet implemented")
    }

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> B::IntTensorPrimitive {
        // TODO: Implement complex_argsort in ComplexTensorOps
        todo!("complex_argsort not yet implemented")
    }

    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        // TODO: Implement complex_matmul in ComplexTensorOps
        todo!("complex_matmul not yet implemented")
    }

    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        todo!()
    }

    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        todo!()
    }

    fn cummin(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        todo!()
    }

    fn cummax(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        todo!()
    }

    fn zeros(shape: Shape, device: &<B as Backend>::Device, dtype: DType) -> Self::Primitive {
        todo!()
    }

    fn ones(shape: Shape, device: &<B as Backend>::Device, dtype: DType) -> Self::Primitive {
        todo!()
    }
}

// // Complex-specific methods for Tensor<B, D, Complex>
// impl<B, const D: usize> Tensor<B, D, Complex>
// where
//     B: ComplexTensorBackend,
// {
//     /// Returns the complex conjugate of the tensor.
//     ///
//     /// For a complex number z = a + bi, the conjugate is z* = a - bi.
//     pub fn conj(self) -> Self {
//         Self::new(B::complex_conj(self.primitive))
//     }

//     /// Returns the real part of the complex tensor as a float tensor.
//     pub fn real(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_real(self.primitive)))
//     }

//     /// Returns the imaginary part of the complex tensor as a float tensor.
//     pub fn imag(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_imag(self.primitive)))
//     }

//     /// Returns the magnitude (absolute value) of the complex tensor as a float tensor.
//     pub fn magnitude(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_abs(self.primitive)))
//     }

//     /// Returns the phase (argument) of the complex tensor as a float tensor.
//     pub fn phase(self) -> Tensor<B, D> {
//         Tensor::new(TensorPrimitive::Float(B::complex_arg(self.primitive)))
//     }

//     /// Creates a complex tensor from real and imaginary parts.
//     pub fn from_parts(real: Tensor<B, D>, imag: Tensor<B, D>) -> Self {
//         Self::new(B::complex_from_parts(
//             real.primitive.tensor(),
//             imag.primitive.tensor(),
//         ))
//     }

//     /// Creates a complex tensor from magnitude and phase (polar coordinates).
//     pub fn from_polar(magnitude: Tensor<B, D>, phase: Tensor<B, D>) -> Self {
//         Self::new(B::complex_from_polar(
//             magnitude.primitive.tensor(),
//             phase.primitive.tensor(),
//         ))
//     }

//     /// Complex exponential function.
//     pub fn exp(self) -> Self {
//         Self::new(B::complex_exp(self.primitive))
//     }

//     /// Complex natural logarithm.
//     pub fn log(self) -> Self {
//         Self::new(B::complex_log(self.primitive))
//     }

//     /// Complex power function.
//     pub fn powc(self, rhs: Self) -> Self {
//         Self::new(B::complex_powc(self.primitive, rhs.primitive))
//     }

//     /// Complex square root.
//     pub fn sqrt(self) -> Self {
//         Self::new(B::complex_sqrt(self.primitive))
//     }

//     /// Complex sine function.
//     pub fn sin(self) -> Self {
//         Self::new(B::complex_sin(self.primitive))
//     }

//     /// Complex cosine function.
//     pub fn cos(self) -> Self {
//         Self::new(B::complex_cos(self.primitive))
//     }

//     /// Complex tangent function.
//     pub fn tan(self) -> Self {
//         Self::new(B::complex_tan(self.primitive))
//     }
// }

impl<B: ComplexTensorBackend> TensorKind<B> for Complex {
    type Primitive = B::ComplexTensorPrimitive;
    fn name() -> &'static str {
        "Complex"
    }
}
