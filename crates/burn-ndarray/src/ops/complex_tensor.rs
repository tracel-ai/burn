use crate::execute_with_float_dtype;
use crate::ops::NdArrayMathOps;

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};

use burn_backend::{ElementConversion, TensorData, TensorMetadata, ops::FloatTensorOps};
use burn_complex::base::element::{Complex, ToComplexElement};
use burn_complex::base::{ComplexDevice, ComplexTensor, ComplexTensorBackend, InterleavedLayout};
use burn_complex::utils::{
    interleave_from_split_data, interleaved_data_from_imag_data, interleaved_data_from_real_data,
    interleaved_data_to_imag_data, interleaved_data_to_real_data, interleaved_data_to_split_data,
};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ComplexTensorBackend
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<Complex<E>>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type InnerBackend = NdArray<E, I, Q>;

    type ComplexScalar = Complex<E>;

    type Layout = InterleavedLayout<NdArrayTensor>;

    fn complex_from_real_data(
        data: TensorData,
        _device: &ComplexDevice<Self>,
    ) -> ComplexTensor<Self> {
        let interleaved_data = interleaved_data_from_real_data(data);

        NdArrayTensor::from_data(interleaved_data).into()
    }

    fn complex_from_imag_data(
        data: TensorData,
        _device: &ComplexDevice<Self>,
    ) -> ComplexTensor<Self> {
        let interleaved_data = interleaved_data_from_imag_data(data);

        NdArrayTensor::from_data(interleaved_data).into()
    }

    fn complex_from_interleaved_data(
        data: TensorData,
        _device: &<Self::InnerBackend as burn_backend::Backend>::Device,
    ) -> ComplexTensor<Self> {
        NdArrayTensor::from_data(data).into()
    }

    fn complex_from_split_data(
        real_data: TensorData,
        imag_data: TensorData,
        _device: &<Self::InnerBackend as burn_backend::Backend>::Device,
    ) -> ComplexTensor<Self> {
        let interleaved_data = interleave_from_split_data(real_data, imag_data);
        NdArrayTensor::from_data(interleaved_data).into()
    }
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement>
    burn_complex::base::ComplexTensorOps<NdArray<E, I, Q>> for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<Complex<E>>>,
    NdArrayTensor: From<SharedArray<I>>,
{
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
        rhs: <NdArray<E, I, Q> as ComplexTensorBackend>::ComplexScalar,
    ) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::equal_elem(array, rhs.elem())
        })
    }

    async fn complex_into_real_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
        _device: &ComplexDevice<Self>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_real_data(tensor.into_data()))
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
        _device: &ComplexDevice<Self>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_imag_data(tensor.into_data()))
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
        _device: &ComplexDevice<Self>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(tensor.into_data())
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
        _device: &ComplexDevice<Self>,
    ) -> Result<(TensorData, TensorData), burn_backend::ExecutionError> {
        Ok(interleaved_data_to_split_data(tensor.into_data()))
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
