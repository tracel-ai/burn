use core::mem;

use burn_backend::{
    DType, Element, QTensorPrimitive, Shape, TensorData, TensorMetadata,
    quantization::{QParams, QuantLevel, QuantMode, QuantScheme, QuantValue},
};

use crate::NdArrayStorage;
use crate::ops::quantization::{QuantizationStrategy, SymmetricQuantization};
use alloc::vec::Vec;
use ndarray::{ArcArray, ArrayD, IxDyn};

/// Concrete storage type for ndarray (owned with COW semantics via Arc)
pub type SharedArray<E> = ArcArray<E, IxDyn>;

/// Tensor primitive used by the [ndarray backend](crate::NdArray).
///
/// Supports both owned and borrowed (zero-copy) data via [`NdArrayStorage`].
/// When data is borrowed from external sources (like burnpack files),
/// it remains zero-copy until a mutating operation is performed.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum NdArrayTensor {
    F64(NdArrayStorage<f64>),
    F32(NdArrayStorage<f32>),
    I64(NdArrayStorage<i64>),
    I32(NdArrayStorage<i32>),
    I16(NdArrayStorage<i16>),
    I8(NdArrayStorage<i8>),
    U64(NdArrayStorage<u64>),
    U32(NdArrayStorage<u32>),
    U16(NdArrayStorage<u16>),
    U8(NdArrayStorage<u8>),
    Bool(NdArrayStorage<bool>),
}

impl NdArrayTensor {
    /// Extract bool array, converting to owned if necessary.
    pub(crate) fn bool(self) -> SharedArray<bool> {
        match self {
            NdArrayTensor::Bool(storage) => storage.into_shared(),
            _ => unimplemented!("Expected bool tensor, got {:?}", self.dtype()),
        }
    }

    /// Returns true if this tensor uses borrowed (zero-copy) storage.
    #[inline]
    pub fn is_borrowed(&self) -> bool {
        macro_rules! check {
            ($($variant:ident),*) => {
                match self {
                    $(NdArrayTensor::$variant(s) => s.is_borrowed(),)*
                }
            };
        }
        check!(F64, F32, I64, I32, I16, I8, U64, U32, U16, U8, Bool)
    }
}

pub(crate) fn cast_to_dtype<E1: Element>(array: SharedArray<E1>, dtype: DType) -> NdArrayTensor
where
    NdArrayTensor: From<SharedArray<E1>>,
{
    fn cast<E1: Element, E2: Element>(array: SharedArray<E1>) -> SharedArray<E2> {
        array.mapv(|a| a.elem()).into_shared()
    }

    if E1::dtype() == dtype {
        return array.into();
    }

    match dtype {
        DType::F64 => cast::<E1, f64>(array).into(),
        DType::F32 => cast::<E1, f32>(array).into(),
        DType::Flex32 => cast::<E1, f32>(array).into(),
        DType::I64 => cast::<E1, i64>(array).into(),
        DType::I32 => cast::<E1, i32>(array).into(),
        DType::I16 => cast::<E1, i16>(array).into(),
        DType::I8 => cast::<E1, i8>(array).into(),
        DType::U64 => cast::<E1, u64>(array).into(),
        DType::U32 => cast::<E1, u32>(array).into(),
        DType::U16 => cast::<E1, u16>(array).into(),
        DType::U8 => cast::<E1, u8>(array).into(),
        DType::Bool => cast::<E1, bool>(array).into(),
        dtype => panic!("Unsupported dtype: {dtype:?}"),
    }
}

macro_rules! impl_from {
    ($($ty: ty => $dtype: ident),*) => {
        // From SharedArray (owned) -> NdArrayTensor
        $(impl From<SharedArray<$ty>> for NdArrayTensor {
           fn from(value: SharedArray<$ty>) -> NdArrayTensor {
                NdArrayTensor::$dtype(NdArrayStorage::from_owned(value))
           }
        })*

        // From NdArrayStorage -> NdArrayTensor
        $(impl From<NdArrayStorage<$ty>> for NdArrayTensor {
           fn from(value: NdArrayStorage<$ty>) -> NdArrayTensor {
                NdArrayTensor::$dtype(value)
           }
        })*
    };
}

impl_from!(
    f64 => F64, f32 => F32,
    i64 => I64, i32 => I32, i16 => I16, i8 => I8,
    u64 => U64, u32 => U32, u16 => U16, u8 => U8,
    bool => Bool
);

/// Macro to execute an operation on a given element type.
///
/// Extracts the storage from NdArrayTensor, converts to SharedArray, and passes to operation.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_dtype {
    (($lhs:expr, $rhs:expr),$element:ident,  $op:expr, [$($dtype: ident => $ty: ty),*]) => {{
        let lhs_dtype = burn_backend::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_backend::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            $(
                ($crate::NdArrayTensor::$dtype(lhs), $crate::NdArrayTensor::$dtype(rhs)) => {
                    #[allow(unused)]
                    type $element = $ty;
                    // Convert storage to SharedArray for compatibility with existing operations
                    $op(lhs.into_shared(), rhs.into_shared()).into()
                }
            )*
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        $crate::execute_with_dtype!(($lhs, $rhs), E, $op)
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
            F64 => f64, F32 => f32,
            I64 => i64, I32 => i32, I16 => i16, I8 => i8,
            U64 => u64, U32 => u32, U16 => u16, U8 => u8,
            Bool => bool
        ])
    }};

    ($tensor:expr, $element:ident, $op:expr, [$($dtype: ident => $ty: ty),*]) => {{
        match $tensor {
            $(
                $crate::NdArrayTensor::$dtype(storage) => {
                    #[allow(unused)]
                    type $element = $ty;
                    // Convert to SharedArray for compatibility with most operations
                    $op(storage.into_shared()).into()
                }
            )*
            #[allow(unreachable_patterns)]
            other => unimplemented!("unsupported dtype: {:?}", other.dtype())
        }
    }};
    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        $crate::execute_with_dtype!($tensor, E, $op)
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!($tensor, $element, $op, [
            F64 => f64, F32 => f32,
            I64 => i64, I32 => i32, I16 => i16, I8 => i8,
            U64 => u64, U32 => u32, U16 => u16, U8 => u8,
            Bool => bool
        ])
    }};
}

/// Macro to execute an operation a given element type.
/// Only handles float types.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_float_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        $crate::execute_with_float_dtype!(($lhs, $rhs), E, $op)
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
            F64 => f64, F32 => f32
        ])
    }};

    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        $crate::execute_with_float_dtype!($tensor, E, $op)
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!($tensor, $element, $op, [
            F64 => f64, F32 => f32
        ])
    }};
}

/// Macro to execute an operation a given element type.
/// Only handles int types.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_int_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        $crate::execute_with_int_dtype!(($lhs, $rhs), E, $op)
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
            I64 => i64, I32 => i32, I16 => i16, I8 => i8,
            U64 => u64, U32 => u32, U16 => u16, U8 => u8
        ])
    }};

    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        $crate::execute_with_int_dtype!($tensor, E, $op)
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!($tensor, $element, $op, [
            I64 => i64, I32 => i32, I16 => i16, I8 => i8,
            U64 => u64, U32 => u32, U16 => u16, U8 => u8
        ])
    }};
}

/// Macro to execute an operation a given element type.
/// Only handles numeric types
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_numeric_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        $crate::execute_with_numeric_dtype!(($lhs, $rhs), E, $op)
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
            F64 => f64, F32 => f32,
            I64 => i64, I32 => i32, I16 => i16, I8 => i8,
            U64 => u64, U32 => u32, U16 => u16, U8 => u8
        ])
    }};

    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        $crate::execute_with_numeric_dtype!($tensor, E, $op)
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        $crate::execute_with_dtype!($tensor, $element, $op, [
            F64 => f64, F32 => f32,
            I64 => i64, I32 => i32, I16 => i16, I8 => i8,
            U64 => u64, U32 => u32, U16 => u16, U8 => u8
        ])
    }};
}

/// Macro to execute a cat operation on a given set of element types.
///
/// Uses zero-copy views from storage for concatenation.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! cat_with_dtype {
    ($tensors: expr, $dim: expr, [$($dtype: ident),*]) => {
        match &$tensors[0] {
            $(NdArrayTensor::$dtype(_) => {
                let tensors = $tensors
                    .iter()
                    .map(|t| {
                        if let NdArrayTensor::$dtype(storage) = t {
                            // Use storage.view() for zero-copy access
                            storage.view()
                        } else {
                            panic!("Concatenate data type mismatch (expected {:?}, got {:?})", $tensors[0].dtype(), t.dtype())
                        }
                    })
                    .collect::<Vec<_>>();
                NdArrayOps::concatenate(&tensors, $dim).into()
            })*
            _ => panic!("Unsupported dtype: {:?}", $tensors[0].dtype())
        }
    };
}

impl TensorMetadata for NdArrayTensor {
    fn dtype(&self) -> DType {
        match self {
            NdArrayTensor::F64(_) => DType::F64,
            NdArrayTensor::F32(_) => DType::F32,
            NdArrayTensor::I64(_) => DType::I64,
            NdArrayTensor::I32(_) => DType::I32,
            NdArrayTensor::I16(_) => DType::I16,
            NdArrayTensor::I8(_) => DType::I8,
            NdArrayTensor::U64(_) => DType::U64,
            NdArrayTensor::U32(_) => DType::U32,
            NdArrayTensor::U16(_) => DType::U16,
            NdArrayTensor::U8(_) => DType::U8,
            NdArrayTensor::Bool(_) => DType::Bool,
        }
    }

    fn shape(&self) -> Shape {
        // Use storage's shape method (works for both borrowed and owned)
        macro_rules! get_shape {
            ($($variant:ident),*) => {
                match self {
                    $(NdArrayTensor::$variant(storage) => Shape::from(storage.shape().to_vec()),)*
                }
            };
        }
        get_shape!(F64, F32, I64, I32, I16, I8, U64, U32, U16, U8, Bool)
    }

    fn rank(&self) -> usize {
        self.shape().num_dims()
    }
}

pub(crate) trait ShapeOps {
    fn num_dims(self) -> usize;
    fn num_elements(self) -> usize;
    fn dims<const N: usize>(self) -> [usize; N];
    fn into_shape(self) -> Shape;
}

impl ShapeOps for &[usize] {
    fn num_dims(self) -> usize {
        self.len()
    }

    fn num_elements(self) -> usize {
        self.iter().product()
    }

    fn dims<const N: usize>(self) -> [usize; N] {
        self.try_into().unwrap()
    }

    fn into_shape(self) -> Shape {
        Shape {
            dims: self.to_vec(),
        }
    }
}

mod utils {
    use burn_std::tensor::is_contiguous;

    use super::*;

    impl NdArrayTensor {
        pub(crate) fn into_data(self) -> TensorData {
            let shape = self.shape();
            let contiguous = self.is_contiguous();

            fn inner<E: Element>(
                shape: Shape,
                is_contiguous: bool,
                array: ArcArray<E, IxDyn>,
            ) -> TensorData {
                let vec = if is_contiguous {
                    match array.try_into_owned_nocopy() {
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
                    array.into_iter().collect()
                };

                TensorData::new(vec, shape)
            }

            // Convert storage to owned array before extracting data
            execute_with_dtype!(self, |arr| inner(shape, contiguous, arr))
        }

        pub(crate) fn is_contiguous(&self) -> bool {
            // For borrowed data, we assume it's contiguous (it came from TensorData which is contiguous)
            // For owned data, we check the strides
            macro_rules! check_contiguous {
                ($($variant:ident),*) => {
                    match self {
                        $(NdArrayTensor::$variant(storage) => {
                            match storage {
                                NdArrayStorage::Borrowed { .. } => {
                                    // Borrowed storage requires contiguous row-major data
                                    // (see NdArrayStorage::from_borrowed documentation)
                                    true
                                }
                                NdArrayStorage::Owned(array) => {
                                    let shape = array.shape();
                                    let mut strides = Vec::with_capacity(array.strides().len());
                                    for &stride in array.strides() {
                                        if stride <= 0 {
                                            return false;
                                        }
                                        strides.push(stride as usize);
                                    }
                                    is_contiguous(shape, &strides)
                                }
                            }
                        })*
                    }
                };
            }
            check_contiguous!(F64, F32, I64, I32, I16, I8, U64, U32, U16, U8, Bool)
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
        let array = match $array.is_standard_layout() {
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
        array.into_dyn()
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

impl NdArrayTensor {
    /// Create a new [ndarray tensor](NdArrayTensor) from [data](TensorData).
    ///
    /// This method attempts zero-copy loading when possible. If the data has properly
    /// aligned bytes that can be borrowed, it creates a borrowed tensor. Otherwise,
    /// it falls back to copying the data.
    ///
    /// Zero-copy loading works when:
    /// - The data's bytes are properly aligned for the element type
    /// - The bytes can be borrowed (e.g., from mmap'd file or static data)
    pub fn from_data(data: TensorData) -> NdArrayTensor {
        // Try zero-copy path first, taking ownership to avoid cloning bytes
        match Self::try_from_data_zero_copy(data) {
            Ok(tensor) => tensor,
            Err(data) => Self::from_data_copy(data),
        }
    }

    /// Try to create a tensor with zero-copy (borrowed) storage.
    ///
    /// Takes ownership of TensorData and returns it back on failure.
    /// No cloning occurs - bytes are moved into storage or returned on failure.
    ///
    /// Returns `Err(data)` if zero-copy is not possible (e.g., misaligned data).
    fn try_from_data_zero_copy(data: TensorData) -> Result<NdArrayTensor, TensorData> {
        let TensorData {
            bytes,
            shape,
            dtype,
        } = data;

        macro_rules! try_borrow {
            ($ty:ty, $variant:ident, $bytes:expr, $shape:expr) => {
                match NdArrayStorage::<$ty>::from_borrowed($bytes, $shape) {
                    Ok(storage) => return Ok(NdArrayTensor::$variant(storage)),
                    Err((bytes, shape)) => (bytes, shape),
                }
            };
        }

        // Try to create borrowed storage; get bytes back on failure
        let (bytes, shape) = match dtype {
            DType::F64 => try_borrow!(f64, F64, bytes, shape),
            DType::F32 => try_borrow!(f32, F32, bytes, shape),
            DType::I64 => try_borrow!(i64, I64, bytes, shape),
            DType::I32 => try_borrow!(i32, I32, bytes, shape),
            DType::I16 => try_borrow!(i16, I16, bytes, shape),
            DType::I8 => try_borrow!(i8, I8, bytes, shape),
            DType::U64 => try_borrow!(u64, U64, bytes, shape),
            DType::U32 => try_borrow!(u32, U32, bytes, shape),
            DType::U16 => try_borrow!(u16, U16, bytes, shape),
            DType::U8 => try_borrow!(u8, U8, bytes, shape),
            DType::Bool => try_borrow!(bool, Bool, bytes, shape),
            _ => (bytes, shape), // QFloat not supported for zero-copy
        };

        Err(TensorData {
            bytes,
            shape,
            dtype,
        })
    }

    /// Create a tensor by copying data (fallback path).
    fn from_data_copy(mut data: TensorData) -> NdArrayTensor {
        let shape = mem::take(&mut data.shape);

        macro_rules! execute {
            ($data: expr, [$($dtype: ident => $ty: ty),*]) => {
                match $data.dtype {
                    $(DType::$dtype => {
                        match data.into_vec::<$ty>() {
                            // Safety: TensorData checks shape validity on creation
                            Ok(vec) => unsafe { ArrayD::from_shape_vec_unchecked(shape, vec) }.into_shared(),
                            Err(err) => panic!("Data should have the same element type as the tensor {err:?}"),
                        }.into()
                    },)*
                    other => unimplemented!("Unsupported dtype {other:?}"),
                }
            };
        }

        execute!(data, [
            F64 => f64, F32 => f32,
            I64 => i64, I32 => i32, I16 => i16, I8 => i8,
            U64 => u64, U32 => u32, U16 => u16, U8 => u8,
            Bool => bool
        ])
    }
}

/// A quantized tensor for the ndarray backend.
#[derive(Clone, Debug)]
pub struct NdArrayQTensor {
    /// The quantized tensor.
    pub qtensor: NdArrayTensor,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The quantization parameters.
    pub qparams: Vec<QParams<f32>>,
}

impl NdArrayQTensor {
    /// Returns the quantization strategy, including quantization parameters, for the given tensor.
    pub fn strategy(&self) -> QuantizationStrategy {
        match self.scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                value:
                    QuantValue::Q8F
                    | QuantValue::Q8S
                    | QuantValue::E4M3
                    | QuantValue::E5M2
                    | QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::E2M1
                    | QuantValue::Q2F
                    | QuantValue::Q2S,
                ..
            } => QuantizationStrategy::PerTensorSymmetric(SymmetricQuantization::init(
                self.qparams[0].scales,
                self.scheme.value,
            )),
            QuantScheme {
                level: QuantLevel::Block(block_size),
                mode: QuantMode::Symmetric,
                value:
                    QuantValue::Q8F
                    | QuantValue::Q8S
                    | QuantValue::E4M3
                    | QuantValue::E5M2
                    | QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::E2M1
                    | QuantValue::Q2F
                    | QuantValue::Q2S,
                ..
            } => QuantizationStrategy::PerBlockSymmetric(
                self.qparams
                    .iter()
                    .map(|q| SymmetricQuantization::init(q.scales, self.scheme.value))
                    .collect(),
                block_size,
            ),
        }
    }
}

impl QTensorPrimitive for NdArrayQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }

    fn default_scheme() -> QuantScheme {
        QuantScheme::default().with_store(burn_backend::quantization::QuantStore::Native)
    }
}

impl TensorMetadata for NdArrayQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.qtensor.shape()
    }

    fn rank(&self) -> usize {
        self.shape().num_dims()
    }
}

#[cfg(test)]
mod tests {
    use crate::NdArray;
    use alloc::vec;

    use super::*;
    use burn_backend::{
        Distribution,
        ops::{FloatTensorOps, QTensorOps},
        quantization::{QuantStore, QuantizationParametersPrimitive},
    };
    use burn_std::rand::get_seeded_rng;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([3]),
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

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
        let tensor = NdArrayTensor::from_data(data_expected.clone());

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
        let tensor = NdArrayTensor::from_data(data_expected.clone());

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
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_qtensor_strategy() {
        type B = NdArray<f32, i64, i8>;
        let scale: f32 = 0.009_019_608;
        let device = Default::default();

        let tensor = B::float_from_data(TensorData::from([-1.8f32, -1.0, 0.0, 0.5]), &device);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let qparams = QuantizationParametersPrimitive {
            scales: B::float_from_data(TensorData::from([scale]), &device),
        };
        let qtensor: NdArrayQTensor = B::quantize(tensor, &scheme, qparams);

        assert_eq!(qtensor.scheme(), &scheme);
        assert_eq!(
            qtensor.strategy(),
            QuantizationStrategy::PerTensorSymmetric(SymmetricQuantization::init(
                scale,
                QuantValue::Q8S
            ))
        );
    }

    // ==========================================================================
    // Zero-copy integration tests
    // These tests verify end-to-end zero-copy behavior through NdArrayTensor.
    // ==========================================================================

    #[test]
    fn zero_copy_creates_borrowed_storage() {
        // Verify that from_data creates borrowed storage when possible.
        // Note: For native allocations, Bytes::clone() copies data internally,
        // but the storage type (Borrowed) is preserved, which is important for
        // the is_unique() behavior that triggers copy-on-write.
        use burn_std::Bytes;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let tensor_data = TensorData::from_bytes(bytes, Shape::new([2, 2]), DType::F32);

        let tensor = NdArrayTensor::from_data(tensor_data);

        match &tensor {
            NdArrayTensor::F32(storage) => {
                assert!(
                    storage.is_borrowed(),
                    "ZERO-COPY REGRESSION: from_data should create borrowed storage \
                     for properly aligned TensorData with Bytes"
                );
                assert!(
                    !storage.is_unique(),
                    "ZERO-COPY REGRESSION: borrowed storage must report is_unique() == false"
                );
            }
            _ => panic!("Expected F32 tensor"),
        }
    }

    #[test]
    fn zero_copy_data_integrity() {
        // Verify data is correctly accessible through borrowed storage
        use burn_std::Bytes;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let tensor_data = TensorData::from_bytes(bytes, Shape::new([2, 2]), DType::F32);

        let tensor = NdArrayTensor::from_data(tensor_data);

        match &tensor {
            NdArrayTensor::F32(storage) => {
                let view = storage.view();
                assert_eq!(view[[0, 0]], 1.0);
                assert_eq!(view[[0, 1]], 2.0);
                assert_eq!(view[[1, 0]], 3.0);
                assert_eq!(view[[1, 1]], 4.0);
            }
            _ => panic!("Expected F32 tensor"),
        }
    }

    #[test]
    fn zero_copy_fallback_when_bytes_owned() {
        // When TensorData owns bytes exclusively, it may use the copy path
        // This is expected behavior - verify it still works correctly
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor = NdArrayTensor::from_data(data.clone());
        let result = tensor.into_data();

        assert_eq!(data, result, "Data should round-trip correctly");
    }
}
