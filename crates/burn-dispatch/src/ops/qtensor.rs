use burn_backend::{
    ExecutionError, QTensorPrimitive, TensorData, TensorPrimitive,
    ops::QTensorOps,
    quantization::QuantizationParametersPrimitive,
    tensor::{FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{QuantPropagation, Shape, Slice};

use crate::backends::*;
use crate::{DispatchDevice, Dispatch};

impl QTensorOps<Self> for Dispatch {
    fn q_from_data(data: TensorData, device: &DispatchDevice) -> QuantizedTensor<Self> {
        creation_op!(Quantized, device, |device| B::q_from_data(data, device))
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &burn_std::QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        binary_op!(
            (tensor, float),
            (qparams.scales, float),
            |tensor, scales| {
                B::quantize(tensor, scheme, QuantizationParametersPrimitive { scales })
            } => Quantized
        )
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, quantized, |tensor| B::dequantize(tensor) => Float)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> DispatchDevice {
        tensor.device()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &DispatchDevice) -> QuantizedTensor<Self> {
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
        unary_op!(tensor, quantized, |tensor| B::q_reshape(tensor, shape) => Quantized)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_op!(tensor, quantized, |tensor| B::q_into_data(tensor).await)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        unary_op!(tensor, quantized, |tensor| B::q_expand(tensor, shape) => Quantized)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        unary_op!(tensor, quantized, |tensor| B::q_swap_dims(tensor, dim1, dim2) => Quantized)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        unary_op!(tensor, quantized, |tensor| B::q_permute(tensor, axes) => Quantized)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        unary_op!(tensor, quantized, |tensor| B::q_flip(tensor, axes) => Quantized)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        binary_op!(
            (tensor, quantized),
            (indices, int),
            |tensor, indices| B::q_select(tensor, dim, indices) => Quantized
        )
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        unary_op!(tensor, quantized, |tensor| B::q_slice(tensor, slices) => Quantized)
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        // TODO: this would be much cleaner if we consolidated tensor primitive types
        match (lhs, rhs) {
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                if matches!(lhs.propagation(), QuantPropagation::Propagate) {
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
                if matches!(rhs.propagation(), QuantPropagation::Propagate) {
                    let out = binary_op!(
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
                    let out = binary_op!(
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
                if matches!(lhs.propagation(), QuantPropagation::Propagate) {
                    let out = binary_op!(
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
                    let out = binary_op!(
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
