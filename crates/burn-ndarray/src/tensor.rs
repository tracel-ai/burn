use core::mem;

use burn_tensor::{
    DType, Element, Shape, TensorData, TensorMetadata,
    quantization::{
        QParams, QTensorPrimitive, QuantInputType, QuantLevel, QuantMode, QuantScheme,
        QuantizationStrategy, SymmetricQuantization,
    },
};

use alloc::vec::Vec;
use ndarray::{ArcArray, ArrayD, IxDyn};

use crate::element::QuantElement;

/// Tensor primitive used by the [ndarray backend](crate::NdArray).
#[derive(new, Debug, Clone)]
pub struct NdArrayTensor<E> {
    /// Dynamic array that contains the data of type E.
    pub array: ArcArray<E, IxDyn>,
}

impl<E: Element> TensorMetadata for NdArrayTensor<E> {
    fn dtype(&self) -> DType {
        E::dtype()
    }

    fn shape(&self) -> Shape {
        Shape::from(self.array.shape().to_vec())
    }
}

/// Float tensor primitive.
#[derive(Debug, Clone)]
pub enum NdArrayTensorFloat {
    /// 32-bit float.
    F32(NdArrayTensor<f32>),
    /// 64-bit float.
    F64(NdArrayTensor<f64>),
}

impl From<NdArrayTensor<f32>> for NdArrayTensorFloat {
    fn from(value: NdArrayTensor<f32>) -> Self {
        NdArrayTensorFloat::F32(value)
    }
}

impl From<NdArrayTensor<f64>> for NdArrayTensorFloat {
    fn from(value: NdArrayTensor<f64>) -> Self {
        NdArrayTensorFloat::F64(value)
    }
}

impl TensorMetadata for NdArrayTensorFloat {
    fn dtype(&self) -> DType {
        match self {
            NdArrayTensorFloat::F32(tensor) => tensor.dtype(),
            NdArrayTensorFloat::F64(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            NdArrayTensorFloat::F32(tensor) => tensor.shape(),
            NdArrayTensorFloat::F64(tensor) => tensor.shape(),
        }
    }
}

/// Macro to create a new [float tensor](NdArrayTensorFloat) based on the element type.
#[macro_export]
macro_rules! new_tensor_float {
    // Op executed with default dtype
    ($tensor:expr) => {{
        match E::dtype() {
            burn_tensor::DType::F64 => $crate::NdArrayTensorFloat::F64($tensor),
            burn_tensor::DType::F32 => $crate::NdArrayTensorFloat::F32($tensor),
            // FloatNdArrayElement only implemented for f64 and f32
            _ => unimplemented!("Unsupported dtype"),
        }
    }};
}

/// Macro to execute an operation a given element type.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_float_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                $crate::NdArrayTensorFloat::F64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                $crate::NdArrayTensorFloat::F32($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                type $element = f64;
                $crate::NdArrayTensorFloat::F64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                type $element = f32;
                $crate::NdArrayTensorFloat::F32($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: type automatically inferred by the compiler but return type is not a float tensor
    (($lhs:expr, $rhs:expr) => $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                $op(lhs, rhs)
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => $crate::NdArrayTensorFloat::F64($op(tensor)),
            $crate::NdArrayTensorFloat::F32(tensor) => $crate::NdArrayTensorFloat::F32($op(tensor)),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => {
                type $element = f64;
                $crate::NdArrayTensorFloat::F64($op(tensor))
            }
            $crate::NdArrayTensorFloat::F32(tensor) => {
                type $element = f32;
                $crate::NdArrayTensorFloat::F32($op(tensor))
            }
        }
    }};

    // Unary op: type automatically inferred by the compiler but return type is not a float tensor
    ($tensor:expr => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => $op(tensor),
            $crate::NdArrayTensorFloat::F32(tensor) => $op(tensor),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation and return type is not a float tensor
    ($tensor:expr, $element:ident => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => {
                type $element = f64;
                $op(tensor)
            }
            $crate::NdArrayTensorFloat::F32(tensor) => {
                type $element = f32;
                $op(tensor)
            }
        }
    }};
}

/// Int tensor primitive.
#[derive(Debug, Clone)]
pub enum NdArrayTensorInt {
    I64(NdArrayTensor<i64>),
    I32(NdArrayTensor<i32>),
    I16(NdArrayTensor<i16>),
    I8(NdArrayTensor<i8>),
    U64(NdArrayTensor<u64>),
    U32(NdArrayTensor<u32>),
    U16(NdArrayTensor<u16>),
    U8(NdArrayTensor<u8>),
}

impl From<NdArrayTensor<i64>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i64>) -> Self {
        NdArrayTensorInt::I64(value)
    }
}
impl From<NdArrayTensor<i32>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i32>) -> Self {
        NdArrayTensorInt::I32(value)
    }
}

impl From<NdArrayTensor<i16>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i16>) -> Self {
        NdArrayTensorInt::I16(value)
    }
}

impl From<NdArrayTensor<i8>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i8>) -> Self {
        NdArrayTensorInt::I8(value)
    }
}
impl From<NdArrayTensor<u64>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u64>) -> Self {
        NdArrayTensorInt::U64(value)
    }
}

impl From<NdArrayTensor<u32>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u32>) -> Self {
        NdArrayTensorInt::U32(value)
    }
}

impl From<NdArrayTensor<u16>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u16>) -> Self {
        NdArrayTensorInt::U16(value)
    }
}

impl From<NdArrayTensor<u8>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u8>) -> Self {
        NdArrayTensorInt::U8(value)
    }
}

impl TensorMetadata for NdArrayTensorInt {
    fn dtype(&self) -> DType {
        match self {
            NdArrayTensorInt::I64(tensor) => tensor.dtype(),
            NdArrayTensorInt::I32(tensor) => tensor.dtype(),
            NdArrayTensorInt::I16(tensor) => tensor.dtype(),
            NdArrayTensorInt::I8(tensor) => tensor.dtype(),
            NdArrayTensorInt::U64(tensor) => tensor.dtype(),
            NdArrayTensorInt::U32(tensor) => tensor.dtype(),
            NdArrayTensorInt::U16(tensor) => tensor.dtype(),
            NdArrayTensorInt::U8(tensor) => tensor.dtype(),
        }

    }

    fn shape(&self) -> Shape {
        match self {
            NdArrayTensorInt::I64(tensor) => tensor.shape(),
            NdArrayTensorInt::I32(tensor) => tensor.shape(),
            NdArrayTensorInt::I16(tensor) => tensor.shape(),
            NdArrayTensorInt::I8(tensor) => tensor.shape(),
            NdArrayTensorInt::U64(tensor) => tensor.shape(),
            NdArrayTensorInt::U32(tensor) => tensor.shape(),
            NdArrayTensorInt::U16(tensor) => tensor.shape(),
            NdArrayTensorInt::U8(tensor) => tensor.shape(),
        }
    }
}

/// Macro to create a new [int tensor](NdArrayIntTensor) based on the element type.
#[macro_export]
macro_rules! new_tensor_int {
    // Op executed with default dtype
    ($tensor:expr) => {{
        match E::dtype() {
            burn_tensor::DType::I64 => $crate::NdArrayTensorFloat::I64($tensor),
            burn_tensor::DType::I32 => $crate::NdArrayTensorFloat::I32($tensor),
            burn_tensor::DType::I16 => $crate::NdArrayTensorFloat::I16($tensor),
            burn_tensor::DType::I8 => $crate::NdArrayTensorFloat::I8($tensor),
            burn_tensor::DType::U64 => $crate::NdArrayTensorFloat::U64($tensor),
            burn_tensor::DType::U32 => $crate::NdArrayTensorFloat::U32($tensor),
            burn_tensor::DType::U16 => $crate::NdArrayTensorFloat::U16($tensor),
            burn_tensor::DType::U8 => $crate::NdArrayTensorFloat::U8($tensor),
            _ => unimplemented!("Unsupported dtype"),
        }
    }};
}


/// Macro to execute an operation a given element type.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_int_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorInt::I64(lhs), $crate::NdArrayTensorInt::I64(rhs)) => {
                $crate::NdArrayTensorInt::I64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I32(lhs), $crate::NdArrayTensorInt::I32(rhs)) => {
                $crate::NdArrayTensorInt::I32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I16(lhs), $crate::NdArrayTensorInt::I16(rhs)) => {
                $crate::NdArrayTensorInt::I16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I8(lhs), $crate::NdArrayTensorInt::I8(rhs)) => {
                $crate::NdArrayTensorInt::I8($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U64(lhs), $crate::NdArrayTensorInt::U64(rhs)) => {
                $crate::NdArrayTensorInt::U64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U32(lhs), $crate::NdArrayTensorInt::U32(rhs)) => {
                $crate::NdArrayTensorInt::U32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U16(lhs), $crate::NdArrayTensorInt::U16(rhs)) => {
                $crate::NdArrayTensorInt::U16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U8(lhs), $crate::NdArrayTensorInt::U8(rhs)) => {
                $crate::NdArrayTensorInt::U8($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorInt::I64(lhs), $crate::NdArrayTensorInt::I64(rhs)) => {
                type $element = i64;
                $crate::NdArrayTensorInt::I64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I32(lhs), $crate::NdArrayTensorInt::I32(rhs)) => {
                type $element = i32;
                $crate::NdArrayTensorInt::I32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I16(lhs), $crate::NdArrayTensorInt::I16(rhs)) => {
                type $element = i16;
                $crate::NdArrayTensorInt::I16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I8(lhs), $crate::NdArrayTensorInt::I8(rhs)) => {
                type $element = i8;
                $crate::NdArrayTensorInt::I8($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U64(lhs), $crate::NdArrayTensorInt::U64(rhs)) => {
                type $element = u64;
                $crate::NdArrayTensorInt::U64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U32(lhs), $crate::NdArrayTensorInt::U32(rhs)) => {
                type $element = u32;
                  $crate::NdArrayTensorInt::U32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U16(lhs), $crate::NdArrayTensorInt::U16(rhs)) => {
                type $element = u16;
                $crate::NdArrayTensorInt::U16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U8(lhs), $crate::NdArrayTensorInt::U8(rhs)) => {
                type $element = u8;
                $crate::NdArrayTensorInt::U8($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: type automatically inferred by the compiler but return type is not a float tensor
    (($lhs:expr, $rhs:expr) => $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorInt::I64(lhs), $crate::NdArrayTensorInt::I64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::I32(lhs), $crate::NdArrayTensorInt::I32(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::I16(lhs), $crate::NdArrayTensorInt::I16(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::I8(lhs), $crate::NdArrayTensorInt::I8(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U64(lhs), $crate::NdArrayTensorInt::U64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U32(lhs), $crate::NdArrayTensorInt::U32(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U16(lhs), $crate::NdArrayTensorInt::U16(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U8(lhs), $crate::NdArrayTensorInt::U8(rhs)) {
                $op(lhs, rhs)
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        match $tensor {
            $crate::NdArrayIntTensor::I64(tensor) => $crate::NdArrayIntTensor::I64($op(tensor)),
            $crate::NdArrayIntTensor::I32(tensor) => $crate::NdArrayIntTensor::I32($op(tensor)),
            $crate::NdArrayIntTensor::I16(tensor) => $crate::NdArrayIntTensor::I16($op(tensor)),
            $crate::NdArrayIntTensor::I8(tensor) => $crate::NdArrayIntTensor::I8($op(tensor)),
            $crate::NdArrayIntTensor::U64(tensor) => $crate::NdArrayIntTensor::U64($op(tensor)),
            $crate::NdArrayIntTensor::U32(tensor) => $crate::NdArrayIntTensor::U32($op(tensor)),
            $crate::NdArrayIntTensor::U16(tensor) => $crate::NdArrayIntTensor::U16($op(tensor)),
            $crate::NdArrayIntTensor::U8(tensor) => $crate::NdArrayIntTensor::U8($op(tensor)),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::I64(tensor) => {
                type $element = i64;
                $crate::NdArrayTensorFloat::I64($op(tensor))
            }
            $crate::NdArrayTensorFloat::I32(tensor) => {
                type $element = i32;
                $crate::NdArrayTensorFloat::I32($op(tensor))
            }
            $crate::NdArrayTensorFloat::I16(tensor) => {
                type $element = i16;
                $crate::NdArrayTensorFloat::I16($op(tensor))
            }
            $crate::NdArrayTensorFloat::I8(tensor) => {
                type $element = i8;
                $crate::NdArrayTensorFloat::I8($op(tensor))
            }
            $crate::NdArrayTensorFloat::U64(tensor) => {
                type $element = u64;
                $crate::NdArrayTensorFloat::U64($op(tensor))
            }
            $crate::NdArrayTensorFloat::U32(tensor) => {
                type $element = u32;
                $crate::NdArrayTensorFloat::U32($op(tensor))
            }
            $crate::NdArrayTensorFloat::U16(tensor) => {
                type $element = u16;
                $crate::NdArrayTensorFloat::U16($op(tensor))
            }
            $crate::NdArrayTensorFloat::U8(tensor) => {
                type $element = u8;
                $crate::NdArrayTensorFloat::U8($op(tensor))
            }
        }
    }};

    // Unary op: type automatically inferred by the compiler but return type is not a float tensor
    ($tensor:expr => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::I32(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::I16(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::I8(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U64(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U32(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U16(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U8(tensor) => $op(tensor),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation and return type is not a float tensor
    ($tensor:expr, $element:ident => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => {
                type $element = i64;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::I32(tensor) => {
                type $element = i32;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::I16(tensor) => {
                type $element = i16;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::I8(tensor) => {
                type $element = i8;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U64(tensor) => {
                type $element = u64;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U32(tensor) => {
                type $element = u32;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U16(tensor) => {
                type $element = u16;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U8(tensor) => {
                type $element = u8;
                $op(tensor)
            }    
        }

    }};
}

mod utils {
    use burn_common::tensor::is_contiguous;

    use super::*;

    impl<E> NdArrayTensor<E>
    where
        E: Element,
    {
        pub(crate) fn into_data(self) -> TensorData {
            let shape = self.shape();

            let vec = if self.is_contiguous() {
                match self.array.try_into_owned_nocopy() {
                    Ok(owned) => {
                        let (mut vec, offset) = owned.into_raw_vec_and_offset();
                        if let Some(offset) = offset {
                            vec.drain(..offset);
                        }
                        if vec.len() > shape.num_elements() {
                            vec.drain(shape.num_elements()..vec.len());
                        }
                        vec
                    }
                    Err(array) => array.into_iter().collect(),
                }
            } else {
                self.array.into_iter().collect()
            };

            TensorData::new(vec, shape)
        }

        pub(crate) fn is_contiguous(&self) -> bool {
            let shape = self.array.shape();
            let mut strides = Vec::with_capacity(self.array.strides().len());

            for &stride in self.array.strides() {
                if stride <= 0 {
                    return false;
                }
                strides.push(stride as usize);
            }
            is_contiguous(shape, &strides)
        }
    }
}

/// Converts a slice of usize to a typed dimension.
#[macro_export(local_inner_macros)]
macro_rules! to_typed_dims {
    (
        $n:expr,
        $dims:expr,
        justdim
    ) => {{
        let mut dims = [0; $n];
        for i in 0..$n {
            dims[i] = $dims[i];
        }
        let dim: Dim<[usize; $n]> = Dim(dims);
        dim
    }};
}

/// Reshapes an array into a tensor.
#[macro_export(local_inner_macros)]
macro_rules! reshape {
    (
        ty $ty:ty,
        n $n:expr,
        shape $shape:expr,
        array $array:expr
    ) => {{
        let dim = $crate::to_typed_dims!($n, $shape.dims, justdim);
        let array: ndarray::ArcArray<$ty, Dim<[usize; $n]>> = match $array.is_standard_layout() {
            true => {
                match $array.to_shape(dim) {
                    Ok(val) => val.into_shared(),
                    Err(err) => {
                        core::panic!("Shape should be compatible shape={dim:?}: {err:?}");
                    }
                }
            },
            false => $array.to_shape(dim).unwrap().as_standard_layout().into_shared(),
        };
        let array = array.into_dyn();

        NdArrayTensor::new(array)
    }};
    (
        ty $ty:ty,
        shape $shape:expr,
        array $array:expr,
        d $D:expr
    ) => {{
        match $D {
            1 => reshape!(ty $ty, n 1, shape $shape, array $array),
            2 => reshape!(ty $ty, n 2, shape $shape, array $array),
            3 => reshape!(ty $ty, n 3, shape $shape, array $array),
            4 => reshape!(ty $ty, n 4, shape $shape, array $array),
            5 => reshape!(ty $ty, n 5, shape $shape, array $array),
            6 => reshape!(ty $ty, n 6, shape $shape, array $array),
            _ => core::panic!("NdArray supports arrays up to 6 dimensions, received: {}", $D),
        }
    }};
}

impl<E> NdArrayTensor<E>
where
    E: Element,
{
    /// Create a new [ndarray tensor](NdArrayTensor) from [data](TensorData).
    pub fn from_data(mut data: TensorData) -> NdArrayTensor<E> {
        let shape = mem::take(&mut data.shape);

        let array = match data.into_vec::<E>() {
            // Safety: TensorData checks shape validity on creation, so we don't need to repeat that check here
            Ok(vec) => unsafe { ArrayD::from_shape_vec_unchecked(shape, vec) }.into_shared(),
            Err(err) => panic!("Data should have the same element type as the tensor {err:?}"),
        };

        NdArrayTensor::new(array)
    }
}

/// A quantized tensor for the ndarray backend.
#[derive(Clone, Debug)]
pub struct NdArrayQTensor<Q: QuantElement> {
    /// The quantized tensor.
    pub qtensor: NdArrayTensor<Q>,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The quantization parameters.
    pub qparams: Vec<QParams<f32, Q>>,
}

impl<Q: QuantElement> NdArrayQTensor<Q> {
    /// Returns the quantization strategy, including quantization parameters, for the given tensor.
    pub fn strategy(&self) -> QuantizationStrategy {
        match self.scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            } => QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                self.qparams[0].scale,
            )),
        }
    }
}

impl<Q: QuantElement> QTensorPrimitive for NdArrayQTensor<Q> {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }
}

impl<Q: QuantElement> TensorMetadata for NdArrayQTensor<Q> {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.qtensor.shape()
    }
}

#[cfg(test)]
mod tests {
    use crate::NdArray;

    use super::*;
    use burn_common::rand::get_seeded_rng;
    use burn_tensor::{
        Distribution,
        ops::{FloatTensorOps, QTensorOps},
        quantization::QuantizationParametersPrimitive,
    };

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([3]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([2, 3]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_3d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([2, 3, 4]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_4d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([2, 3, 4, 2]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::<f32>::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_qtensor_strategy() {
        type B = NdArray<f32, i64, i8>;
        let scale: f32 = 0.009_019_608;
        let device = Default::default();

        let tensor = B::float_from_data(TensorData::from([-1.8f32, -1.0, 0.0, 0.5]), &device);
        let scheme = QuantScheme::default();
        let qparams = QuantizationParametersPrimitive {
            scale: B::float_from_data(TensorData::from([scale]), &device),
            offset: None,
        };
        let qtensor: NdArrayQTensor<i8> = B::quantize(tensor, &scheme, qparams);

        assert_eq!(qtensor.scheme(), &scheme);
        assert_eq!(
            qtensor.strategy(),
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(scale))
        );
    }
}
