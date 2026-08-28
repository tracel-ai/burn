use burn_backend::{
    ExecutionError, FloatDType, Shape, Slice, TensorData, TensorPrimitive,
    ops::QTensorOps,
    quantization::{QuantScheme, QuantizationParametersPrimitive},
    tensor::{FloatTensor, IntTensor, QuantizedTensor},
};
use burn_backend_extension::backend_dispatch;

use crate::{
    BackendTensor, Dispatch, DispatchAutodiffContext, DispatchDevice, DispatchTensor,
    DispatchTensorKind,
};

macro_rules! wrap_q_matmul_concrete {
    ($Backend:ident, $output:expr, $autodiff:expr) => {{
        match $output {
            TensorPrimitive::QFloat(output) => TensorPrimitive::QFloat(DispatchTensor {
                kind: DispatchTensorKind::$Backend(BackendTensor::Quantized(output)),
                autodiff: $autodiff,
            }),
            TensorPrimitive::Float(output) => match $autodiff {
                DispatchAutodiffContext::Disabled => TensorPrimitive::Float(DispatchTensor {
                    kind: DispatchTensorKind::$Backend(BackendTensor::Float(output)),
                    autodiff: DispatchAutodiffContext::Disabled,
                }),
                #[cfg(feature = "autodiff")]
                DispatchAutodiffContext::Enabled(strategy) => {
                    with_autodiff_backend!($Backend, strategy, |AD| {
                        TensorPrimitive::Float(DispatchTensor {
                            kind: DispatchTensorKind::Autodiff(alloc::boxed::Box::new(
                                DispatchTensorKind::$Backend(BackendTensor::Autodiff(
                                    <AD as burn_backend::AutodiffBackend>::from_inner(output),
                                )),
                            )),
                            autodiff: DispatchAutodiffContext::Enabled(strategy),
                        })
                    })
                }
                #[cfg(not(feature = "autodiff"))]
                DispatchAutodiffContext::Enabled(_) => {
                    panic!("autodiff context requires the `autodiff` feature")
                }
            },
        }
    }};
}

#[cfg(feature = "autodiff")]
macro_rules! wrap_q_matmul_autodiff {
    ($Backend:ident, $output:expr, $strategy:expr) => {{
        match $output {
            TensorPrimitive::QFloat(output) => TensorPrimitive::QFloat(DispatchTensor {
                kind: DispatchTensorKind::$Backend(BackendTensor::Quantized(output)),
                autodiff: DispatchAutodiffContext::Enabled($strategy),
            }),
            TensorPrimitive::Float(output) => TensorPrimitive::Float(DispatchTensor {
                kind: DispatchTensorKind::Autodiff(alloc::boxed::Box::new(
                    DispatchTensorKind::$Backend(BackendTensor::Autodiff(output)),
                )),
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
    ($lhs:expr, $rhs:expr, $autodiff:expr; $([$Backend:ident, $cfg:meta]),*) => {{
        match ($lhs.kind, $rhs.kind) {
            $(
                #[cfg($cfg)]
                (DispatchTensorKind::$Backend(lhs), DispatchTensorKind::$Backend(rhs)) => {
                    match $autodiff {
                        DispatchAutodiffContext::Disabled => {
                            type B = crate::backends::$Backend;
                            let output = B::q_matmul(
                                TensorPrimitive::Float(lhs.float()),
                                TensorPrimitive::QFloat(rhs.quantized()),
                            );
                            wrap_q_matmul_concrete!($Backend, output, DispatchAutodiffContext::Disabled)
                        }
                        #[cfg(feature = "autodiff")]
                        DispatchAutodiffContext::Enabled(strategy) => {
                            with_autodiff_backend!($Backend, strategy, |B| {
                                let lhs = <B as burn_backend::AutodiffBackend>::from_inner(lhs.float());
                                let output = B::q_matmul(
                                    TensorPrimitive::Float(lhs),
                                    TensorPrimitive::QFloat(rhs.quantized()),
                                );
                                wrap_q_matmul_autodiff!($Backend, output, strategy)
                            })
                        }
                        #[cfg(not(feature = "autodiff"))]
                        DispatchAutodiffContext::Enabled(_) => {
                            panic!("autodiff context requires the `autodiff` feature")
                        }
                    }
                }
            )*
            #[cfg(feature = "autodiff")]
            (DispatchTensorKind::Autodiff(lhs), rhs) => match (*lhs, rhs) {
                $(
                    #[cfg($cfg)]
                    (DispatchTensorKind::$Backend(lhs), DispatchTensorKind::$Backend(rhs)) => {
                        let DispatchAutodiffContext::Enabled(strategy) = $autodiff else {
                            panic!("an autodiff float primitive must have an enabled autodiff context")
                        };
                        with_autodiff_backend!($Backend, strategy, |B| {
                            let output = B::q_matmul(
                                TensorPrimitive::Float(lhs.autodiff()),
                                TensorPrimitive::QFloat(rhs.quantized()),
                            );
                            wrap_q_matmul_autodiff!($Backend, output, strategy)
                        })
                    }
                )*
                #[allow(unreachable_patterns)]
                _ => panic!("q_matmul inputs are on different backends"),
            },
            #[allow(unreachable_patterns)]
            _ => panic!("q_matmul inputs are on different backends"),
        }
    }};
}

macro_rules! q_matmul_qf_arms {
    ($lhs:expr, $rhs:expr, $autodiff:expr; $([$Backend:ident, $cfg:meta]),*) => {{
        match ($lhs.kind, $rhs.kind) {
            $(
                #[cfg($cfg)]
                (DispatchTensorKind::$Backend(lhs), DispatchTensorKind::$Backend(rhs)) => {
                    match $autodiff {
                        DispatchAutodiffContext::Disabled => {
                            type B = crate::backends::$Backend;
                            let output = B::q_matmul(
                                TensorPrimitive::QFloat(lhs.quantized()),
                                TensorPrimitive::Float(rhs.float()),
                            );
                            wrap_q_matmul_concrete!($Backend, output, DispatchAutodiffContext::Disabled)
                        }
                        #[cfg(feature = "autodiff")]
                        DispatchAutodiffContext::Enabled(strategy) => {
                            with_autodiff_backend!($Backend, strategy, |B| {
                                let rhs = <B as burn_backend::AutodiffBackend>::from_inner(rhs.float());
                                let output = B::q_matmul(
                                    TensorPrimitive::QFloat(lhs.quantized()),
                                    TensorPrimitive::Float(rhs),
                                );
                                wrap_q_matmul_autodiff!($Backend, output, strategy)
                            })
                        }
                        #[cfg(not(feature = "autodiff"))]
                        DispatchAutodiffContext::Enabled(_) => {
                            panic!("autodiff context requires the `autodiff` feature")
                        }
                    }
                }
            )*
            #[cfg(feature = "autodiff")]
            (lhs, DispatchTensorKind::Autodiff(rhs)) => match (lhs, *rhs) {
                $(
                    #[cfg($cfg)]
                    (DispatchTensorKind::$Backend(lhs), DispatchTensorKind::$Backend(rhs)) => {
                        let DispatchAutodiffContext::Enabled(strategy) = $autodiff else {
                            panic!("an autodiff float primitive must have an enabled autodiff context")
                        };
                        with_autodiff_backend!($Backend, strategy, |B| {
                            let output = B::q_matmul(
                                TensorPrimitive::QFloat(lhs.quantized()),
                                TensorPrimitive::Float(rhs.autodiff()),
                            );
                            wrap_q_matmul_autodiff!($Backend, output, strategy)
                        })
                    }
                )*
                #[allow(unreachable_patterns)]
                _ => panic!("q_matmul inputs are on different backends"),
            },
            #[allow(unreachable_patterns)]
            _ => panic!("q_matmul inputs are on different backends"),
        }
    }};
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
            q_to_device,
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

    #[backend_dispatch(skip)]
    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        match (lhs, rhs) {
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                // With no float input, the first tensor is the routing tensor.
                let autodiff = lhs.autodiff;
                backend_list!(q_matmul_qq_arms, lhs, rhs, autodiff)
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                #[cfg(feature = "autodiff")]
                match (
                    matches!(&lhs.kind, DispatchTensorKind::Autodiff(_)),
                    lhs.autodiff,
                ) {
                    (true, DispatchAutodiffContext::Enabled(_))
                    | (false, DispatchAutodiffContext::Disabled) => {}
                    (true, DispatchAutodiffContext::Disabled) => {
                        panic!("an autodiff float primitive must have an enabled autodiff context")
                    }
                    (false, DispatchAutodiffContext::Enabled(_)) => {
                        panic!("an enabled float tensor must use an autodiff primitive")
                    }
                }
                // Float inputs take precedence for routing.
                let autodiff = lhs.autodiff;
                backend_list!(q_matmul_fq_arms, lhs, rhs, autodiff)
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                #[cfg(feature = "autodiff")]
                match (
                    matches!(&rhs.kind, DispatchTensorKind::Autodiff(_)),
                    rhs.autodiff,
                ) {
                    (true, DispatchAutodiffContext::Enabled(_))
                    | (false, DispatchAutodiffContext::Disabled) => {}
                    (true, DispatchAutodiffContext::Disabled) => {
                        panic!("an autodiff float primitive must have an enabled autodiff context")
                    }
                    (false, DispatchAutodiffContext::Enabled(_)) => {
                        panic!("an enabled float tensor must use an autodiff primitive")
                    }
                }
                // Float inputs take precedence for routing.
                let autodiff = rhs.autodiff;
                backend_list!(q_matmul_qf_arms, lhs, rhs, autodiff)
            }
            _ => unreachable!(),
        }
    }
}
