use crate::execute_with_float_dtype;
use crate::ops::NdArrayMathOps;

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};
use burn_backend::{ElementConversion, TensorData, TensorMetadata};
use burn_complex::base::element::{Complex, ToComplexElement};
use burn_complex::base::{ComplexTensor, ComplexTensorBackend, InterleavedLayout};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ComplexTensorBackend
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<Complex<E>>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type InnerBackend = NdArray<E, I, Q>;

    type ComplexTensorPrimitive = NdArrayTensor;
    type ComplexElem = Complex<E>;

    type Layout = InterleavedLayout;
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement>
    burn_complex::base::ComplexTensorOps<NdArray<E, I, Q>> for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<Complex<E>>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type Layout = burn_complex::base::InterleavedLayout;
    fn real(tensor: ComplexTensor<Self>) -> NdArrayTensor {
        match tensor {
            crate::NdArrayTensor::Complex32(storage) => {
                #[allow(unused)]
                type E = burn_complex::base::element::Complex<f32>;
                (|array: SharedArray<E>| array.mapv(|a: E| a.imag).into_shared())(
                    storage.into_shared(),
                )
                .into()
            }
            crate::NdArrayTensor::Complex64(storage) => {
                #[allow(unused)]
                type E = burn_complex::base::element::Complex<f64>;
                (|array: SharedArray<E>| array.mapv(|a: E| a.imag).into_shared())(
                    storage.into_shared(),
                )
                .into()
            }
            #[allow(unreachable_patterns)]
            other => unimplemented!("unsupported dtype: {:?}", other.dtype()),
        }
    }

    fn imag(tensor: NdArrayTensor) -> NdArrayTensor {
        match tensor {
            crate::NdArrayTensor::Complex32(storage) => {
                #[allow(unused)]
                type E = burn_complex::base::element::Complex<f32>;
                (|array: SharedArray<E>| array.mapv(|a: E| a.imag).into_shared())(
                    storage.into_shared(),
                )
                .into()
            }
            crate::NdArrayTensor::Complex64(storage) => {
                #[allow(unused)]
                type E = burn_complex::base::element::Complex<f64>;
                (|array: SharedArray<E>| array.mapv(|a: E| a.imag).into_shared())(
                    storage.into_shared(),
                )
                .into()
            }
            #[allow(unreachable_patterns)]
            other => unimplemented!("unsupported dtype: {:?}", other.dtype()),
        }
    }
    //NOTE: May want to change complex types from ComplexE to Complex<E> in the future to match the element type (and allow quantized complex tensors)
    fn to_complex(tensor: NdArrayTensor) -> NdArrayTensor {
        {
            {
                match tensor {
                    crate::NdArrayTensor::F64(storage) => {
                        #[allow(unused)]
                        type E = f64;
                        (|array: SharedArray<E>| array.mapv(|a: E| a.to_complex64()).into_shared())(
                            storage.into_shared(),
                        )
                        .into()
                    }
                    crate::NdArrayTensor::F32(storage) => {
                        #[allow(unused)]
                        type E = f32;
                        (|array: SharedArray<E>| {
                            array.mapv_into_any(|a: E| a.to_complex32()).into_shared()
                        })(storage.into_shared())
                        .into()
                    }
                    #[allow(unreachable_patterns)]
                    other => unimplemented!("unsupported dtype: {:?}", other.dtype()),
                }
            }
        }
    }

    async fn complex_into_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(tensor.into_data())
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<NdArray<E, I, Q>>,
        rhs: <NdArray<E, I, Q> as ComplexTensorBackend>::ComplexElem,
    ) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::equal_elem(array, rhs.elem())
        })
    }
}

// TODO: actually fix this
/// Macro to execute an operation for complex dtypes (currently Complex32).
#[macro_export]
macro_rules! execute_with_complex_dtype {
        // Binary op: type automatically inferred by the compiler
        (($lhs:expr, $rhs:expr), $op:expr) => {{
            $crate::execute_with_complex_dtype!(($lhs, $rhs), E, $op)
        }};
        // Binary op
        (($lhs:expr, $rhs:expr),$element:ident, $op:expr) => {{
            $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
                Complex32 => burn_complex::base::element::Complex<f32>,
                Complex64 => burn_complex::base::element::Complex<f64>
            ])
        }};
        // Unary op: type automatically inferred by the compiler
        ($tensor:expr, $op:expr) => {{
            $crate::execute_with_complex_dtype!($tensor, E, $op)
        }};

        // Unary op
        ($tensor:expr, $element:ident, $op:expr) => {{
            $crate::execute_with_dtype!($tensor, $element, $op, [
                Complex32 => burn_complex::base::element::Complex<f32>,
                Complex64 => burn_complex::base::element::Complex<f64>
            ])
        }};
    }

// /// Macro to execute an operation a given element type.
// /// Only handles float types.
// ///
// /// # Panics
// /// Since there is no automatic type cast at this time, binary operations for different
// /// floating point precision data types will panic with a data type mismatch.
// #[macro_export]
// macro_rules! execute_with_float_dtype {
//     // Binary op: type automatically inferred by the compiler
//     (($lhs:expr, $rhs:expr), $op:expr) => {{
//         $crate::execute_with_float_dtype!(($lhs, $rhs), E, $op)
//     }};

//     // Binary op: generic type cannot be inferred for an operation
//     (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
//         $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
//             F64 => f64, F32 => f32
//         ])
//     }};

//     // Unary op: type automatically inferred by the compiler
//     ($tensor:expr, $op:expr) => {{
//         $crate::execute_with_float_dtype!($tensor, E, $op)
//     }};

//     // Unary op: generic type cannot be inferred for an operation
//     ($tensor:expr, $element:ident, $op:expr) => {{
//         $crate::execute_with_dtype!($tensor, $element, $op, [
//             F64 => f64, F32 => f32
//         ])
//     }};
// }

// #[macro_export]
// macro_rules! execute_with_dtype {
//     // 1. Binary op with explicit list
//     (($lhs:expr, $rhs:expr), $element:ident, $op:expr, [$( $(#[$meta:meta])* $dtype:ident => $ty:ty ),*]) => {{
//         let lhs_dtype = burn_backend::TensorMetadata::dtype(&$lhs);
//         let rhs_dtype = burn_backend::TensorMetadata::dtype(&$rhs);
//         match ($lhs, $rhs) {
//             $(
//                 $(#[$meta])*
//                 ($crate::NdArrayTensor::$dtype(lhs), $crate::NdArrayTensor::$dtype(rhs)) => {
//                     #[allow(unused)]
//                     type $element = $ty;
//                     // Convert storage to SharedArray for compatibility with existing operations
//                     $op(lhs.into_shared(), rhs.into_shared()).into()
//                 }
//             )*
//             _ => panic!(
//                 "Data type mismatch (lhs: {:?}, rhs: {:?})",
//                 lhs_dtype, rhs_dtype
//             ),
//         }
//     }};

//     // 2. Binary op: type automatically inferred
//     (($lhs:expr, $rhs:expr), $op:expr) => {{
//         $crate::execute_with_dtype!(($lhs, $rhs), E, $op)
//     }};

//     // 3. Binary op: default list (updated to include complex)
//     (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
//         $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
//             F64 => f64, F32 => f32,
//             I64 => i64, I32 => i32, I16 => i16, I8 => i8,
//             U64 => u64, U32 => u32, U16 => u16, U8 => u8,
//             Bool => bool,
//             #[cfg(feature = "complex")]
//             Complex32 => burn_complex::base::element::Complex<f32>,
//             #[cfg(feature = "complex")]
//             Complex64 => burn_complex::base::element::Complex<f64>
//         ])
//     }};

//     // 4. Unary op with explicit list
//     ($tensor:expr, $element:ident, $op:expr, [$( $(#[$meta:meta])* $dtype:ident => $ty:ty ),*]) => {{
//         match $tensor {
//             $(
//                 $(#[$meta])*
//                 $crate::NdArrayTensor::$dtype(storage) => {
//                     #[allow(unused)]
//                     type $element = $ty;
//                     // Convert to SharedArray for compatibility with most operations
//                     $op(storage.into_shared()).into()
//                 }
//             )*
//             #[allow(unreachable_patterns)]
//             other => unimplemented!("unsupported dtype: {:?}", other.dtype())
//         }
//     }};

//     // 5. Unary op: type automatically inferred
//     ($tensor:expr, $op:expr) => {{
//         $crate::execute_with_dtype!($tensor, E, $op)
//     }};

//     // 6. Unary op: default list (updated to include complex)
//     ($tensor:expr, $element:ident, $op:expr) => {{
//         $crate::execute_with_dtype!($tensor, $element, $op, [
//             F64 => f64, F32 => f32,
//             I64 => i64, I32 => i32, I16 => i16, I8 => i8,
//             U64 => u64, U32 => u32, U16 => u16, U8 => u8,
//             Bool => bool,
//             #[cfg(feature = "complex")]
//             Complex32 => burn_complex::base::element::Complex<f32>,
//             #[cfg(feature = "complex")]
//             Complex64 => burn_complex::base::element::Complex<f64>
//         ])
//     }};
// }
