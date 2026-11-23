use burn_backend::{
    ExecutionError, FloatDType, IntDType, Shape, Slice, TensorData, TensorPrimitive,
    ops::QTensorOps,
    quantization::{QuantScheme, QuantizationParametersPrimitive},
    tensor::{FloatTensor, IntTensor, QuantizedTensor},
};
use burn_backend_extension::backend_dispatch;

use crate::{
    BackendTensor, Dispatch, DispatchAutodiffContext, DispatchDevice, DispatchTensor,
    DispatchTensorKind,
};

/// Wraps a concrete backend's `q_matmul` output.
macro_rules! wrap_q_matmul_concrete {
    ($Backend:ident, $output:expr, $autodiff:expr) => {{
        match $output {
            TensorPrimitive::QFloat(output) => TensorPrimitive::QFloat(DispatchTensor {
                kind: DispatchTensorKind::$Backend(BackendTensor::Quantized(output)),
                autodiff: $autodiff,
            }),
            TensorPrimitive::Float(output) => TensorPrimitive::Float(DispatchTensor {
                kind: DispatchTensorKind::$Backend(BackendTensor::Float(output)),
                autodiff: $autodiff,
            }),
        }
    }};
}

#[cfg(feature = "autodiff")]
/// Wraps an `Autodiff<Dispatch>` `q_matmul` output.
macro_rules! wrap_q_matmul_autodiff {
    ($output:expr, $strategy:expr) => {{
        match $output {
            TensorPrimitive::QFloat(output) => TensorPrimitive::QFloat(DispatchTensor {
                kind: output.kind,
                autodiff: DispatchAutodiffContext::Enabled($strategy),
            }),
            TensorPrimitive::Float(output) => TensorPrimitive::Float(DispatchTensor {
                kind: DispatchTensorKind::Autodiff(alloc::boxed::Box::new(output)),
                autodiff: DispatchAutodiffContext::Enabled($strategy),
            }),
        }
    }};
}

macro_rules! q_matmul_qq_arms {
    ($lhs:expr, $rhs:expr, $autodiff:expr; $([$Backend:ident, $cfg:meta]),*) => {{
        match ($lhs.kind, $rhs.kind) {
            $(
                #[cfg($cfg)]
                (DispatchTensorKind::$Backend(lhs), DispatchTensorKind::$Backend(rhs)) => {
                    type B = crate::backends::$Backend;
                    let output = B::q_matmul(
                        TensorPrimitive::QFloat(lhs.quantized()),
                        TensorPrimitive::QFloat(rhs.quantized()),
                    );
                    wrap_q_matmul_concrete!($Backend, output, $autodiff)
                }
            )*
            #[allow(unreachable_patterns)]
            _ => panic!("q_matmul inputs are on different backends"),
        }
    }};
}

macro_rules! q_matmul_fq_arms {
    ($lhs:expr, $rhs:expr; $([$Backend:ident, $cfg:meta]),*) => {{
        match ($lhs.kind, $rhs.kind) {
            $(
                #[cfg($cfg)]
                (DispatchTensorKind::$Backend(lhs), DispatchTensorKind::$Backend(rhs)) => {
                    type B = crate::backends::$Backend;
                    let output = B::q_matmul(
                        TensorPrimitive::Float(lhs.float()),
                        TensorPrimitive::QFloat(rhs.quantized()),
                    );
                    wrap_q_matmul_concrete!($Backend, output, DispatchAutodiffContext::Disabled)
                }
            )*
            #[allow(unreachable_patterns)]
            _ => panic!("q_matmul inputs are on different backends"),
        }
    }};
}

macro_rules! q_matmul_qf_arms {
    ($lhs:expr, $rhs:expr; $([$Backend:ident, $cfg:meta]),*) => {{
        match ($lhs.kind, $rhs.kind) {
            $(
                #[cfg($cfg)]
                (DispatchTensorKind::$Backend(lhs), DispatchTensorKind::$Backend(rhs)) => {
                    type B = crate::backends::$Backend;
                    let output = B::q_matmul(
                        TensorPrimitive::QFloat(lhs.quantized()),
                        TensorPrimitive::Float(rhs.float()),
                    );
                    wrap_q_matmul_concrete!($Backend, output, DispatchAutodiffContext::Disabled)
                }
            )*
            #[allow(unreachable_patterns)]
            _ => panic!("q_matmul inputs are on different backends"),
        }
    }};
}

#[cfg(feature = "autodiff")]
/// Panics when a float tensor's kind and autodiff context disagree.
fn validate_float_tensor(tensor: &DispatchTensor) {
    match (&tensor.kind, tensor.autodiff) {
        (DispatchTensorKind::Autodiff(_), DispatchAutodiffContext::Enabled(_)) => {}
        (DispatchTensorKind::Autodiff(_), DispatchAutodiffContext::Disabled) => {
            panic!("an autodiff float primitive must have an enabled autodiff context")
        }
        (_, DispatchAutodiffContext::Enabled(_)) => {
            panic!("an enabled float tensor must use an autodiff primitive")
        }
        _ => {}
    }
}

#[backend_dispatch]
impl QTensorOps<Self> for Dispatch {
    fn q_from_data(data: TensorData, device: &DispatchDevice) -> QuantizedTensor<Self> {
        B::q_from_data(data, device)
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        B::quantize(tensor, scheme, qparams)
    }

    fn quantize_dynamic(tensor: FloatTensor<Self>, scheme: &QuantScheme) -> QuantizedTensor<Self> {
        B::quantize_dynamic(tensor, scheme)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        B::dequantize(tensor, dtype)
    }

    #[backend_dispatch(skip)]
    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        device: &DispatchDevice,
    ) -> QuantizedTensor<Self> {
        to_device!(
            Quantized,
            quantized,
            tensor,
            device,
            q_to_device(),
            |inner, device| {
                let data =
                    burn_backend::read_sync(B1::q_into_data(inner)).expect("Should read data");
                B2::q_from_data(data, device)
            }
        )
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::q_into_data(tensor).await
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_expand(tensor, shape)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        B::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        B::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        B::q_flip(tensor, axes)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        B::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        B::q_slice(tensor, slices)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        B::q_gather(dim, tensor, indices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmax(tensor, dim, out_dtype)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmin(tensor, dim, out_dtype)
    }

    #[backend_dispatch(skip)]
    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        match (lhs, rhs) {
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                let autodiff = lhs.autodiff.merge(rhs.autodiff);
                // With no float input, the first tensor is the routing tensor.
                backend_list!(q_matmul_qq_arms, lhs, rhs, autodiff)
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                match lhs.autodiff.merge(rhs.autodiff) {
                    DispatchAutodiffContext::Disabled => {
                        backend_list!(q_matmul_fq_arms, lhs, rhs)
                    }
                    #[cfg(feature = "autodiff")]
                    DispatchAutodiffContext::Enabled(strategy) => {
                        validate_float_tensor(&lhs);
                        with_autodiff_backend!(strategy, |AD| {
                            let lhs = match lhs.kind {
                                DispatchTensorKind::Autodiff(tensor) => *tensor,
                                _ => <AD as burn_backend::AutodiffBackend>::from_inner(lhs),
                            };
                            let output = AD::q_matmul(
                                TensorPrimitive::Float(lhs),
                                TensorPrimitive::QFloat(rhs),
                            );
                            wrap_q_matmul_autodiff!(output, strategy)
                        })
                    }
                    #[cfg(not(feature = "autodiff"))]
                    DispatchAutodiffContext::Enabled(_) => {
                        panic!("autodiff context requires the `autodiff` feature")
                    }
                }
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                match lhs.autodiff.merge(rhs.autodiff) {
                    DispatchAutodiffContext::Disabled => {
                        backend_list!(q_matmul_qf_arms, lhs, rhs)
                    }
                    #[cfg(feature = "autodiff")]
                    DispatchAutodiffContext::Enabled(strategy) => {
                        validate_float_tensor(&rhs);
                        with_autodiff_backend!(strategy, |AD| {
                            let rhs = match rhs.kind {
                                DispatchTensorKind::Autodiff(tensor) => *tensor,
                                _ => <AD as burn_backend::AutodiffBackend>::from_inner(rhs),
                            };
                            let output = AD::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::Float(rhs),
                            );
                            wrap_q_matmul_autodiff!(output, strategy)
                        })
                    }
                    #[cfg(not(feature = "autodiff"))]
                    DispatchAutodiffContext::Enabled(_) => {
                        panic!("autodiff context requires the `autodiff` feature")
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}
