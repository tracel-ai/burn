use core::mem;

use burn_tensor::{
    DType, Element, Shape, TensorData, TensorMetadata,
    quantization::{QParams, QTensorPrimitive, QuantLevel, QuantMode, QuantScheme, QuantValue},
};

use crate::ops::quantization::{QuantizationStrategy, SymmetricQuantization};
use alloc::vec::Vec;
use ndarray::{ArcArray, ArrayD, IxDyn};

/// Concrete storage type for ndarray
pub type SharedArray<E> = ArcArray<E, IxDyn>;

/// Tensor primitive used by the [ndarray backend](crate::NdArray).
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum NdArrayTensor {
    F64(SharedArray<f64>),
    F32(SharedArray<f32>),
    I64(SharedArray<i64>),
    I32(SharedArray<i32>),
    I16(SharedArray<i16>),
    I8(SharedArray<i8>),
    U64(SharedArray<u64>),
    U32(SharedArray<u32>),
    U16(SharedArray<u16>),
    U8(SharedArray<u8>),
    Bool(SharedArray<bool>),
}

impl NdArrayTensor {
    pub(crate) fn bool(self) -> SharedArray<bool> {
        match self {
            NdArrayTensor::Bool(arr) => arr,
            _ => unimplemented!("Expected bool tensor, got {:?}", self.dtype()),
        }
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
        $(impl From<SharedArray<$ty>> for NdArrayTensor {
           fn from(value: SharedArray<$ty>) -> NdArrayTensor {
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

/// Macro to execute an operation a given element type.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_dtype {
    (($lhs:expr, $rhs:expr),$element:ident,  $op:expr, [$($dtype: ident => $ty: ty),*]) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            $(
                ($crate::NdArrayTensor::$dtype(lhs), $crate::NdArrayTensor::$dtype(rhs)) => {
                    #[allow(unused)]
                    type $element = $ty;
                    $op(lhs, rhs).into()
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
                $crate::NdArrayTensor::$dtype(lhs) => {
                    #[allow(unused)]
                    type $element = $ty;
                    $op(lhs).into()
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

/// Macro to execute an cat operation on a given set of element types.
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
                        if let NdArrayTensor::$dtype(tensor) = t {
                            tensor.view()
                        } else {
                            panic!("Concatenate data type mismatch (expected f32, got f64)")
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
        execute_with_dtype!(self, E, |a: &ArcArray<E, IxDyn>| Shape::from(
            a.shape().to_vec()
        ))
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

            execute_with_dtype!(self, |arr| inner(shape, contiguous, arr))
        }

        pub(crate) fn is_contiguous(&self) -> bool {
            fn inner<E: Element>(array: &ArcArray<E, IxDyn>) -> bool {
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

            execute_with_dtype!(self, inner)
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
    pub fn from_data(mut data: TensorData) -> NdArrayTensor {
        let shape = mem::take(&mut data.shape);

        macro_rules! execute {
            ($data: expr, [$($dtype: ident => $ty: ty),*]) => {
                match $data.dtype {
                    $(DType::$dtype => {
                        match data.into_vec::<$ty>() {
                            // Safety: TensorData checks shape validity on creation, so we don't need to repeat that check here
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
        QuantScheme::default().with_store(burn_tensor::quantization::QuantStore::Native)
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

    use super::*;
    use burn_std::rand::get_seeded_rng;
    use burn_tensor::{
        Distribution,
        ops::{FloatTensorOps, QTensorOps},
        quantization::{QuantStore, QuantizationParametersPrimitive},
    };

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
}
