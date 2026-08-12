use burn_backend::{
    DeviceOps, ExecutionError, FloatDType, Shape, Slice, TensorData, TensorMetadata,
    TensorPrimitive,
    ops::QTensorOps,
    quantization::{QuantPropagation, QuantScheme, QuantizationParametersPrimitive},
    tensor::{FloatTensor, IntTensor, QuantizedTensor},
};
use burn_backend_extension::backend_dispatch;

use crate::{Dispatch, DispatchDevice};

#[backend_dispatch]
impl QTensorOps<Self> for Dispatch {
    #[backend_dispatch(skip)]
    fn q_from_data(data: TensorData, device: &DispatchDevice) -> QuantizedTensor<Self> {
        creation_op!(Quantized, device, |device| B::q_from_data(data, device))
    }

    #[backend_dispatch(skip)]
    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        // `binary_float` rather than `binary_op`: on an autodiff device the
        // tensor and its scales arrive autodiff-wrapped, and quantization
        // detaches them (the packed result carries no graph).
        binary_float!(
            (tensor, float),
            (qparams.scales, float),
            |tensor, scales| {
                B::quantize(tensor, scheme, QuantizationParametersPrimitive { scales })
            } => Quantized
        )
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

    #[backend_dispatch(skip)]
    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_op!(tensor, quantized, |tensor| B::q_into_data(tensor).await)
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
        // TODO: this would be much cleaner if we consolidated tensor primitive types
        match (lhs, rhs) {
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                let propagation = lhs.device().defaults().quantization.propagation;
                if matches!(propagation, QuantPropagation::Propagate) {
                    let out = binary_op!(
                        (lhs, quantized),
                        (rhs, quantized),
                        |lhs, rhs| {
                            if let TensorPrimitive::QFloat(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        } => Quantized
                    );
                    TensorPrimitive::QFloat(out)
                } else {
                    let out = binary_op!(
                        (lhs, quantized),
                        (rhs, quantized),
                        |lhs, rhs| {
                            if let TensorPrimitive::Float(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        } => Float
                    );
                    TensorPrimitive::Float(out)
                }
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                let propagation = rhs.device().defaults().quantization.propagation;
                // `binary_float` on the mixed cases: the float side may arrive
                // autodiff-wrapped, in which case the op runs on the autodiff
                // backend so gradients flow through the float operand.
                if matches!(propagation, QuantPropagation::Propagate) {
                    let out = binary_float!(
                        (lhs, float),
                        (rhs, quantized),
                        |lhs, rhs| {
                            if let TensorPrimitive::QFloat(out) = B::q_matmul(
                                TensorPrimitive::Float(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        } => Quantized
                    );
                    TensorPrimitive::QFloat(out)
                } else {
                    let out = binary_float!(
                        (lhs, float),
                        (rhs, quantized),
                        |lhs, rhs| {
                            if let TensorPrimitive::Float(out) = B::q_matmul(
                                TensorPrimitive::Float(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        } => Float
                    );
                    TensorPrimitive::Float(out)
                }
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                let propagation = lhs.device().defaults().quantization.propagation;
                if matches!(propagation, QuantPropagation::Propagate) {
                    let out = binary_float!(
                        (lhs, quantized),
                        (rhs, float),
                        |lhs, rhs| {
                            if let TensorPrimitive::QFloat(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::Float(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        } => Quantized
                    );
                    TensorPrimitive::QFloat(out)
                } else {
                    let out = binary_float!(
                        (lhs, quantized),
                        (rhs, float),
                        |lhs, rhs| {
                            if let TensorPrimitive::Float(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::Float(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        } => Float
                    );
                    TensorPrimitive::Float(out)
                }
            }
            _ => unreachable!(),
        }
    }
}
