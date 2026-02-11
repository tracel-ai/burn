use burn_backend::{
    ExecutionError, QTensorPrimitive, TensorData, TensorPrimitive,
    ops::QTensorOps,
    quantization::QuantizationParametersPrimitive,
    tensor::{FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{QuantPropagation, Shape, Slice};

use crate::{Device, Engine, EngineTensor};
use crate::{backends::*, unary_op};
use crate::{create_quantized, dispatch_async_quantized, multi_tensor_op, unary_quantized};

impl QTensorOps<Self> for Engine {
    fn q_from_data(data: TensorData, device: &Device) -> QuantizedTensor<Self> {
        create_quantized!(device, |device| B::q_from_data(data, device))
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &burn_std::QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        multi_tensor_op!(
            Quantized,
            float(tensor),
            float(qparams.scales),
            |tensor, scales| {
                B::quantize(tensor, scheme, QuantizationParametersPrimitive { scales })
            }
        )
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        unary_op!(Float, quantized, dequantize, tensor)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device {
        tensor.device()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device) -> QuantizedTensor<Self> {
        todo!() // TODO: backend bridge
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        unary_quantized!(q_reshape, tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        dispatch_async_quantized!(q_into_data, tensor)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        unary_quantized!(q_expand, tensor, shape)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        unary_quantized!(q_swap_dims, tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        unary_quantized!(q_permute, tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        unary_quantized!(q_flip, tensor, axes)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        multi_tensor_op!(
            Quantized,
            quantized, // tensor
            int,       // indices
            tensor,
            indices,
            |tensor, indices| B::q_select(tensor, dim, indices)
        )
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        unary_quantized!(q_slice, tensor, slices)
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        // TODO: this would be much cleaner if we consolidated tensor primitive types
        match (lhs, rhs) {
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::QFloat(rhs)) => {
                if matches!(lhs.propagation(), QuantPropagation::Propagate) {
                    let out = multi_tensor_op!(
                        Quantized,
                        quantized, // lhs
                        quantized, // rhs
                        lhs,
                        rhs,
                        |lhs, rhs| {
                            if let TensorPrimitive::QFloat(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        }
                    );
                    TensorPrimitive::QFloat(out)
                } else {
                    let out = multi_tensor_op!(
                        Float,
                        quantized, // lhs
                        quantized, // rhs
                        lhs,
                        rhs,
                        |lhs, rhs| {
                            if let TensorPrimitive::Float(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        }
                    );
                    TensorPrimitive::Float(out)
                }
            }
            (TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)) => {
                if matches!(rhs.propagation(), QuantPropagation::Propagate) {
                    let out = multi_tensor_op!(
                        Quantized,
                        float,     // lhs
                        quantized, // rhs
                        lhs,
                        rhs,
                        |lhs, rhs| {
                            if let TensorPrimitive::QFloat(out) = B::q_matmul(
                                TensorPrimitive::Float(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        }
                    );
                    TensorPrimitive::QFloat(out)
                } else {
                    let out = multi_tensor_op!(
                        Float,
                        float,     // lhs
                        quantized, // rhs
                        lhs,
                        rhs,
                        |lhs, rhs| {
                            if let TensorPrimitive::Float(out) = B::q_matmul(
                                TensorPrimitive::Float(lhs),
                                TensorPrimitive::QFloat(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        }
                    );
                    TensorPrimitive::Float(out)
                }
            }
            (TensorPrimitive::QFloat(lhs), TensorPrimitive::Float(rhs)) => {
                if matches!(lhs.propagation(), QuantPropagation::Propagate) {
                    let out = multi_tensor_op!(
                        Quantized,
                        quantized, // lhs
                        float,     // rhs
                        lhs,
                        rhs,
                        |lhs, rhs| {
                            if let TensorPrimitive::QFloat(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::Float(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        }
                    );
                    TensorPrimitive::QFloat(out)
                } else {
                    let out = multi_tensor_op!(
                        Float,
                        quantized, // lhs
                        float,     // rhs
                        lhs,
                        rhs,
                        |lhs, rhs| {
                            if let TensorPrimitive::Float(out) = B::q_matmul(
                                TensorPrimitive::QFloat(lhs),
                                TensorPrimitive::Float(rhs),
                            ) {
                                out
                            } else {
                                unreachable!()
                            }
                        }
                    );
                    TensorPrimitive::Float(out)
                }
            }
            _ => unreachable!(),
        }
    }
}
