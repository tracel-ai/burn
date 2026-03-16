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
