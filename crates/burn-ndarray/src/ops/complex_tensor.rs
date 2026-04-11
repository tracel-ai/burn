use crate::ops::NdArrayMathOps;
use crate::{NdArrayDevice, execute_with_float_dtype};

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};

use num_traits::Zero;

use burn_backend::{ElementConversion, TensorData, TensorMetadata, ops::FloatTensorOps};
use burn_complex::base::element::{Complex, ToComplexElement};
use burn_complex::base::{
    ComplexDevice, ComplexTensor, ComplexTensorBackend, FloatTensor, InterleavedLayout,
};
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
        crate::execute_complex_to_real_op!(tensor, C, |array: SharedArray<C>| {
            array.mapv(|a| a.real).into_shared()
        })
    }

    fn imag(tensor: NdArrayTensor) -> NdArrayTensor {
        crate::execute_complex_to_real_op!(tensor, C, |array: SharedArray<C>| {
            array.mapv(|a| a.imag).into_shared()
        })
    }
    //NOTE: May want to change complex types from ComplexE to Complex<E> in the future to match the element type (and allow quantized complex tensors)
    fn to_complex(tensor: NdArrayTensor) -> NdArrayTensor {
        crate::execute_real_to_complex_op!(tensor, |array: SharedArray<E>| {
            array.mapv(|a: E| Complex::<E>::new(a, E::zero())).into_shared()
        })
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
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_real_data(tensor.into_data()))
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_imag_data(tensor.into_data()))
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(tensor.into_data())
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<NdArray<E, I, Q>>,
    ) -> Result<(TensorData, TensorData), burn_backend::ExecutionError> {
        Ok(interleaved_data_to_split_data(tensor.into_data()))
    }

    fn complex_device(
        _tensor: &ComplexTensor<NdArray<E, I, Q>>,
    ) -> ComplexDevice<NdArray<E, I, Q>> {
        NdArrayDevice::Cpu
    }

    fn complex_add(
        lhs: ComplexTensor<NdArray<E, I, Q>>,
        rhs: ComplexTensor<NdArray<E, I, Q>>,
    ) -> ComplexTensor<NdArray<E, I, Q>> {
        crate::execute_with_complex_dtype!((lhs, rhs), NdArrayMathOps::add)
    }

    fn complex_sub(
        lhs: ComplexTensor<NdArray<E, I, Q>>,
        rhs: ComplexTensor<NdArray<E, I, Q>>,
    ) -> ComplexTensor<NdArray<E, I, Q>> {
        crate::execute_with_complex_dtype!((lhs, rhs), NdArrayMathOps::sub)
    }

    fn complex_mul(
        lhs: ComplexTensor<NdArray<E, I, Q>>,
        rhs: ComplexTensor<NdArray<E, I, Q>>,
    ) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
    }

    fn complex_div(
        lhs: ComplexTensor<NdArray<E, I, Q>>,
        rhs: ComplexTensor<NdArray<E, I, Q>>,
    ) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
        //crate::execute_with_complex_dtype!((lhs, rhs), )
    }

    fn complex_abs(tensor: ComplexTensor<NdArray<E, I, Q>>) -> FloatTensor<NdArray<E, I, Q>> {
        todo!()
    }

    fn complex_from_parts(
        real: FloatTensor<NdArray<E, I, Q>>,
        imag: FloatTensor<NdArray<E, I, Q>>,
    ) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
    }

    fn complex_exp(tensor: ComplexTensor<NdArray<E, I, Q>>) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
    }

    fn complex_log(tensor: ComplexTensor<NdArray<E, I, Q>>) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
    }
    
    fn complex_squared_norm(tensor: ComplexTensor<NdArray<E, I, Q>>) -> FloatTensor<NdArray<E, I, Q>> {
        crate::execute_complex_to_real_op!(tensor, C, |array: SharedArray<C>| {
            array.mapv(|x| x.real*x.real+x.imag*x.imag).into_shared()
        })
    
    }
    
    fn complex_from_polar(magnitude: FloatTensor<NdArray<E, I, Q>>, phase: FloatTensor<NdArray<E, I, Q>>) -> ComplexTensor<NdArray<E, I, Q>> {
        todo!()
    }
}

/// Macro for ops that return the inner float component of a complex tensor
#[macro_export]
macro_rules! execute_complex_to_real_op {
    ($tensor:expr, $complex_elem:ident, $op:expr) => {{
        match $tensor {
            NdArrayTensor::Complex64(storage) => {
                type $complex_elem = Complex<f64>;
                let array = storage.into_shared();
                $op(array).into()
            }
            NdArrayTensor::Complex32(storage) => {
                type $complex_elem = Complex<f32>;
                let array = storage.into_shared();
                $op(array).into()
            }
            _ => unimplemented!("Not a complex tensor"),
        }
    }};
}

/// Macro for ops that return a complex value 
#[macro_export]
macro_rules! execute_real_to_complex_op {
    ($tensor:expr, $op:expr) => {
        match $tensor {
            NdArrayTensor::F64(storage) => {
                #[allow(unused)]
                type E = f64;
                let array = storage.into_shared();
                $op(array).into()
            }
            NdArrayTensor::F32(storage) => {
                #[allow(unused)]
                type E = f32;
                let array = storage.into_shared();
                $op(array).into()
            }
            _ => unimplemented!("Not a complex tensor"),
        }
    };
}

// TODO: actually fix this
/// Macro to execute an operation for complex dtypes
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

/// Macro to execute an operation that returns a given complex element type.
#[macro_export]
macro_rules! execute_with_complex_out_dtype {
    ($out_dtype:expr, $element:ident, $op:expr, [$($dtype: ident => $ty: ty),*]) => {{
        match $out_dtype {
            $(

                burn_std::DType::$dtype => {
                    #[allow(unused)]
                    type $element = $ty;
                    $op
                }
            )*
            #[allow(unreachable_patterns)]
            other => unimplemented!("unsupported complex dtype: {other:?}")
        }
    }};

    // Unary op: type automatically inferred
    ($out_dtype:expr, $op:expr) => {{
        $crate::execute_with_complex_out_dtype!($out_dtype, E, $op)
    }};

    // Unary op: default complex mapping
    ($out_dtype:expr, $element:ident, $op:expr) => {{
        $crate::execute_with_complex_out_dtype!($out_dtype, $element, $op, [
            Complex64 => Complex<f64>,
            Complex32 => Complex<f32>
        ])
    }};
}
